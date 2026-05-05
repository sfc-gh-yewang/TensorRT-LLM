## §6. Blackwell B200 Kernel Infrastructure

*Subsection prefix `K.x` is scoped to §6.*

The five generations of NVIDIA tensor cores (Volta `mma.sync` → Turing `mma.sync` →
Ampere `mma.sync` → Hopper `wgmma` → Blackwell `tcgen05.mma`) collectively defined a
single trajectory: scale up the per-instruction tile and increasingly decouple the
issuing thread from the math array. Blackwell finishes that trajectory by reducing
the number of threads that participate in an MMA from 128 (the warp-group on Hopper)
to **a single thread**, by adding a dedicated 256 KB on-chip *Tensor Memory* that the
math array writes accumulators into asynchronously, and by fusing a pair of SMs into a
*Tensor Pair Cluster (TPC) "2-CTA UMMA"* that lets one MMA instruction span two SMs
sharing operand A, doubling effective per-issue throughput.

That hardware shift forced an equally aggressive software refactor. CUTLASS 3.7+,
Triton 3.5+, ThunderKittens, ParallelKittens, DeepGEMM, DeepEP, FlashInfer,
sgl-kernel, FLUX, COMET, Triton-distributed, PyTorch SymmetricMemory and Helion all
landed major releases in late 2025 / early 2026 specifically to wire together
`tcgen05.mma`, TMEM, 2-CTA UMMA, Cluster Launch Control (CLC), Programmatic Dependent
Launch (PDL / GDC), TMA, NVLink-5 NVSwitch in-network reduction (multimem), NVSHMEM
3.5+ tile-RMA, and CUDA 12.8/12.9/13.x VMM features. This section walks through that
stack from the bottom (SM100 ISA) up to user-visible DSLs (Helion), covering every
work cited as "kernel infrastructure" in the rest of the report.

---

## K.1 `tcgen05.mma`, TMEM, 2-CTA UMMA, and Grid Dependency Control — the SM100 hardware fundamentals

### K.1.1 What broke in the move from Hopper to Blackwell

Hopper's `wgmma.mma_async.sync.aligned.<shape>.f32.<typeA>.<typeB>` is a
*warp-group* instruction. All 128 threads of a warp-group must arrive at the
instruction together; A operand can come from registers or shared memory, B operand
must be in shared memory, and the accumulator (D) is held in the warp-group's
**registers** — typically 32 FP32 values per thread, taking 16 KB out of the 64 KB
register file. That register-resident accumulator is what makes the
warp-specialization pattern fragile: producer warps (TMA loaders) and consumer warps
(MMA + epilogue) must coexist on the SM, and the consumer warp-group permanently
sacrifices a quarter of its register file just to hold partial sums of a single
`128×256` BF16 tile.

Blackwell replaces this with `tcgen05.mma{.sp}.cta_group::[1|2].kind::<dtype>` — a
**single-thread** PTX instruction that issues asynchronous MMA work to the tensor
core array and writes the accumulator into a dedicated, software-managed on-chip
SRAM called **Tensor Memory (TMEM)**. The Microbenchmarking Blackwell paper
(arXiv:2512.02189) measures the consequences directly: a single
`tcgen05.mma m64n64k16` BF16 instruction has a single-instruction dependency-chain
latency of 11.0 cycles, compared to 32.0 cycles for `wgmma m64n64k16`,
and crucially **stays at 11.4 cycles** for `m256n256k16` while `wgmma` scales
linearly to 128.0 cycles for `m64n256k16`. That is consistent with a spatial-array
design where wider tiles consume more *throughput* (more MAC units busy each cycle)
but not more dependent latency. Across precisions, Blackwell achieves ≈96% of
theoretical TFLOPs peak: 1929.6 BF16 (vs 1515.2 on H200, 1.27×), 3850.6 FP8
(vs 3026.9, 1.27×), 5134.4 FP6 (new), 7700.2 FP4 (new), 3928.5 INT8.

### K.1.2 TMEM — a 256 KB software-managed accumulator SRAM

TMEM is a per-SM 2-D array of 512 columns × 128 lanes of 32-bit cells (256 KB),
addressed by a `(lane, column)` pair. It is not register-mapped, not L1/SMEM, and
**no legacy data-movement instruction can touch it**: not `ldmatrix`, not
`cp.async`, not `cp.async.bulk.tensor`. A new family — `tcgen05.alloc`,
`tcgen05.dealloc`, `tcgen05.cp`, `tcgen05.ld`, `tcgen05.st`, `tcgen05.shift` —
must be used. `tcgen05.alloc` allocates in **column granules of 32 columns**
(= 32 cols × 128 lanes × 4 B = 16 KB), so TMEM holds at most 16 simultaneous granules.
A single 2SM `KernelTmaWarpSpecialized2SmSm100` GEMM tile of `256×256x128` BF16 with a
4-stage TMEM accumulator pipeline routinely consumes 4–6 granules per SM; the
collective builder's `StageCountAutoCarveout` automation in CUTLASS sizes the
mainloop SMEM/TMEM stages by subtracting the epilogue's `sizeof(SharedStorage)` from
the architectural caps.

The microbenchmark paper measures TMEM bandwidth at ~16 TB/s read inside an SM,
operating most efficiently at 64×64-element tile granularity (4 KB per FP8 tile,
matching the 1024-bit memory interface). Tiles smaller than 32×32 underutilize the
wide port; tiles larger than 128×128 trigger multi-phase transfers. The Blackwell
async data flow becomes:

```
GMEM ──TMA (cp.async.bulk.tensor)──▶ SMEM
SMEM ──tcgen05.cp──▶ TMEM  (optional staging, async)
SMEM/TMEM ──tcgen05.mma──▶ TMEM (accumulator, async)
TMEM ──tcgen05.ld──▶ Registers ──TMA store──▶ GMEM
```

For a chained `D = (A·B)·C` (e.g., FlashAttention-3 style `softmax(QK^T)·V`),
keeping the QK^T result in TMEM eliminates a global-memory round-trip of the
intermediate; the paper estimates this avoids ≈12 TB/s of off-chip traffic on a
fully-utilized SM. This is exactly the access pattern that TK / PK and ThunderKittens'
`st_attn_h100.cu` Blackwell port exploit.

### K.1.3 2-CTA UMMA — the cluster `<2,1,1>` collective MMA

Two SMs that sit on the same TPC (Texture Processing Cluster) share an
intra-TPC interconnect that lets a `cta_group::2` `tcgen05.mma` instruction
execute jointly across two cooperating CTAs. The two CTAs each provide their
own A operand (or share via the dedicated bus) and split the N dimension of the
output tile. From the user's point of view this is encoded by:

1. A **cluster shape** `Shape<_2,_1,_1>` (or `<_2,_4,_1>`, `<_4,_4,_1>` — the
   first mode must be a multiple of 2 for 2SM kernels).
2. A **dispatch policy** `cutlass::gemm::KernelTmaWarpSpecialized2SmSm100`
   (or a precision-specific variant: `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100`,
   `KernelTmaWarpSpecialized2SmNvf4Sm100`, `KernelTmaWarpSpecialized2SmMxf4Sm100`).
3. An **MMA tile shape** that matches a row in CUTLASS Tables 4 / 5 / 11 / 12 / 13
   of `media/docs/cpp/blackwell_functionality.md` — for 2SM legacy types these are
   `128×64×4*MMA_K`, `128×128×4*MMA_K`, …, `256×256×4*MMA_K`; for 2SM `mxf4nvf4`
   they are `256×128×256`, `256×192×256`, `256×256×256`; and so on.

The half-tile each CTA actually computes — the **PerSmTileShape_MNK** — is the
MMA tile shape with M halved, e.g., `256×256×128` 2SM ⇒ `128×256×128` per SM.
This is what the epilogue collective sees, and is what governs SMEM/TMEM
budget per SM.

Because the two CTAs that form the 2-CTA UMMA pair must sit on the same TPC,
the hardware exposes them as an even-aligned pair within the cluster — i.e.,
a cluster of shape `<4,4,1>` decomposes into 8 TPC pairs (`{(0,j),(1,j)}` for
j∈[0..3], plus `{(2,j),(3,j)}`). DistGEMM ex.82 uses cluster `<2,1,1>` (one
TPC pair per cluster, 64 clusters total at M=N=K=16384), which keeps the
pairing trivial and lets the `MmaTileShape_MNK = <_256,_256,_128>` tile fit in
the 256 KB TMEM budget.

### K.1.4 Cluster Launch Control (CLC) — dynamic persistent scheduling

The Hopper-era persistent kernel pattern (a static tile scheduler running inside
the consumer warps) has a load-balance flaw: if some SMs are unavailable
(e.g., occupied by a prior kernel or a co-resident communication kernel), the
static schedule becomes uneven. Blackwell adds a hardware feature called
**Cluster Launch Control**: the application launches a non-persistent grid
with as many CTAs as there are output tiles, and a `clusterlaunchcontrol.try_cancel`
PTX instruction lets a *running* cluster atomically claim the next un-launched
ClcID. The semantics are:

- A ClcID will be launched physically when SMs free up; *or*
- An existing worker may steal it via `try_cancel`;
- Every ClcID is processed exactly once;
- CLC works at cluster granularity — a 2×2 cluster pulls 4 ClcIDs at a time.

CUTLASS's `PersistentTileSchedulerSm100` (in `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp`)
together with the `PipelineCLCFetchAsync` pipeline class implement this. Inside
the `Sm100GemmTmaWarpSpecialized` kernel, **warp 1 is the dedicated scheduler
warp** that issues the `try_cancel` instruction and broadcasts the 16-byte
response to a pipelined SMEM ring buffer. The warp assignment is:

| Warp Role        | Warp(s)     |
|------------------|-------------|
| MMA              | 0           |
| Scheduler (CLC)  | 1           |
| Mainloop Load    | 2           |
| Epilogue Load    | 3           |
| Epilogue         | 4, 5, 6, 7  |

The CLC pipeline depth is 3, providing 3 in-flight queries. The first ClcID is
just `blockIdx`, statically known. CLC is what makes 2-CTA UMMA persistent
*and* load-balanced with co-resident communication kernels — exactly the
configuration distributed kernels (DistGEMM ex.82, FLUX, Triton-distributed) need.

### K.1.5 Quantitative TMEM properties — what the microbenchmark paper measured

To make the TMEM design choices concrete, the Microbenchmarking Blackwell paper
quantifies five things that every downstream kernel implicitly assumes:

| Property | Measured value | Implication |
|---------|---------------|-------------|
| TMEM capacity                | 256 KB / SM (512 cols × 128 lanes × 4 B) | Holds 16 column-granules of 16 KB each |
| Optimal tile granularity     | 64×64 elements (4 KB FP8)                | Matches 1024-bit memory port width |
| Max tile before multi-phase  | 128×128                                  | Set by per-bank buffer limit |
| Read bandwidth               | ≈16 TB/s within an SM                    | Enables QK^T → softmax → V chained MMA |
| Off-chip traffic avoided     | ≈12 TB/s per saturated SM                | Justifies the chained-MMA-in-TMEM pattern |
| `tcgen05.mma m64n64k16` SI-LAT | 11.0 cycles (BF16, accum-carried)      | 2.9× lower than `wgmma m64n64k16` (32 cycles) |
| `tcgen05.mma m256n256k16` SI-LAT | 11.4 cycles                            | Constant in tile size (vs Hopper's linear scaling) |

The constant single-instruction latency across tile sizes is the strongest
evidence for a *spatial* tensor-core array on B200: bigger tiles consume more
of the spatial array each cycle (throughput) but don't add pipeline stages
(latency). On Hopper, the wgmma latency scaled linearly with the N dimension
because the warp-group pipeline executed multiple atoms serially. On Blackwell,
the math array is wide enough that even `m256n256k16` finishes in one
back-to-back atom — modulo scheduling cost.

End-to-end precision throughput, all measured at `m64n8k16` (smallest atom)
for the latency-of-a-single-instruction characterization, with the throughput
column extrapolated from `m256n256` saturating runs:

| Input  | Accum | Latency (cyc) | Throughput (TFLOPS or TOPS) | % of Theoretical Peak |
|--------|-------|---------------|------------------------------|------------------------|
| FP16   | FP16  | 11.2          | 1929.6                       | 96.5% |
| FP16   | FP32  | 11.5          | 964.8 (halved by accum)      | 96.5% |
| BF16   | FP32  | 11.4          | 1926.4                       | 96.3% |
| FP8    | FP16  | 11.8          | 3850.6                       | 96.3% |
| FP8    | FP32  | 12.1          | 1912.8 (halved by accum)     | 96.3% |
| FP6    | FP32  | 12.3          | 5134.4                       | 96.0% |
| FP4    | FP32  | 12.6          | 7700.2                       | 96.2% |
| INT8   | INT32 | 11.9          | 3928.5                       | 98.2% |

The take-away that drives the FP4 inference deployment for video DiT: **FP4
delivers 4× the throughput of FP8 with the same latency**, and the math array
is consistently 96% of theoretical peak across precisions. This means the
gap between sustained and peak FLOPS on B200 is *smaller* than on H200 (which
typically hit 80–85% on real workloads). Any remaining underutilization is
memory-bandwidth- or kernel-launch-overhead-driven, not tensor-core-driven.

### K.1.6 Grid Dependency Control (GDC) / Programmatic Dependent Launch (PDL)

Hopper added a feature to let one stream's kernel programmatically signal "the
next dependent kernel can start its prologue early"; CUTLASS exposes it in
`include/cutlass/arch/grid_dependency_control.h` as two device-side PTX
intrinsics:

```cpp
CUTLASS_DEVICE void launch_dependent_grids() {
#if defined(CUTLASS_GDC_ENABLED)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}
CUTLASS_DEVICE void wait_on_dependent_grids() {
#if defined(CUTLASS_GDC_ENABLED)
  asm volatile("griddepcontrol.wait;");
#endif
}
```

The build gates are:

```cmake
cmake .. -DCUTLASS_NVCC_ARCHS="100a" -DCUTLASS_ENABLE_GDC_FOR_SM100=1
```

with the `CUTLASS_GDC_ENABLED` macro becoming defined only when CUDA ≥ 12.0 *and*
`__CUDA_ARCH__ ∈ {1000, 1010, 1030, 1100, 1200, 1210}` (SM100, SM101, SM103, SM110,
SM120, SM121) *and* the feature-flag `__CUDA_ARCH_FEAT_SM<XX>_ALL` (or
`CUDA_ARCH_FAMILY(<arch>)`) is enabled. The producer kernel calls
`launch_dependent_grids()` from its epilogue once it knows the dependent kernel can
start fetching; the dependent kernel inserts `wait_on_dependent_grids()` before
its first global memory access. On a back-to-back GEMM-then-elementwise pair this
typically saves 5–15 µs of launch latency that would otherwise show up between
kernels in a chain. PDL is essential for DistGEMM (where stages launch back-to-back
with rotating buffers), DeepGEMM Mega MoE, and any composition of CUTLASS kernels
where the producer's output can begin to be consumed before the producer's full
grid completes.

---

## K.2 CUTLASS Blackwell narrow-precision GEMM examples (72a/72b/72c, 84a, 89, 91, …)

The CUTLASS 4.x `examples/` directory contains the canonical Blackwell GEMM
recipes; in `/Users/yewang/work/cutlass/examples/` the relevant entries are:

```
70_blackwell_gemm                       # Plain BF16/FP16 SM100 GEMM
71_blackwell_gemm_with_collective_builder
72_blackwell_narrow_precision_gemm      # NVFP4, MXFP4, mixed MX-FP8/FP4 (a, b, c)
73_blackwell_gemm_preferred_cluster
74_blackwell_gemm_streamk
75_blackwell_grouped_gemm
76_blackwell_conv
77_blackwell_fmha                       # Blackwell FlashAttention
78_blackwell_emulated_bf16x9_gemm       # 3xTF32-style emulation for higher accuracy
79_blackwell_geforce_gemm               # SM120 (RTX 50)
80_blackwell_geforce_sparse_gemm
81_blackwell_gemm_blockwise
82_blackwell_distributed_gemm           # 8-GPU TP DistGEMM (covered in K.3)
83_blackwell_sparse_gemm
84_blackwell_narrow_precision_sparse_gemm   # 84a NVFP4 sparse, 84b mixed
86_blackwell_mixed_dtype_gemm
87_blackwell_geforce_gemm_blockwise
89_sm103_fp4_ultra_gemm                 # GB300 (SM103) FP4 "ultra" path
90_sm103_fp4_ultra_grouped_gemm
91_fp4_gemv                             # FP4 GEMV (decode-style M=1)
92_blackwell_moe_gemm
93_blackwell_low_latency_gqa
94_ada_fp8_blockwise
95_blackwell_gemm_green_context         # Green context (per-stream resource isolation)
111_hopper_ssd                          # Hopper SSD/SSM
112_blackwell_ssd                       # Blackwell SSD/SSM
```

### K.2.1 Example 72 — block-scaled narrow-precision GEMM

`72a_blackwell_nvfp4_bf16_gemm.cu` is the canonical NVFP4 dense GEMM. Operand
configuration:

- A: `nv_float4_t<float_e2m1_t>` row-major, alignment 32
- B: `nv_float4_t<float_e2m1_t>` col-major, alignment 32
- C/D: BF16 row-major, alignment 8
- Accumulator: FP32
- `OperatorClass = arch::OpClassBlockScaledTensorOp`
- `MmaTileShape = Shape<_256,_256,_256>`
- `ClusterShape = Shape<_2,_4,_1>` (8-CTA cluster, 2-CTA UMMA pairing)
- `KernelMainloopPolicy = KernelScheduleAuto` → resolves to
  `KernelTmaWarpSpecialized2SmNvf4Sm100`

NVFP4 uses an FP4 (E2M1) data element with a per-block FP8 (E4M3) scale,
where each block is 16 elements along K. The CUTLASS helper
`Sm1xxBlockScaledConfig<SFVecSize=16>::tile_atom_to_shape_SFA(M,N,K,L)` produces
the required interleaved scale-factor layout (the so-called "M128xK4" basic
block: 128 rows × 4 SFs along K, K-major over multiple basic blocks). The
collective builder handles all of the SF prefetching and broadcasting.

`72b_blackwell_nvfp4_nvfp4_gemm.cu` extends this to NVFP4-out: the epilogue
fuses a `LinCombBlockScaleFactor<SFDVectorSize, ElementD, ElementCompute,
ElementSFD, GmemLayoutSFD, ElementC>` operation that, *during the epilogue*,
absmax-reduces each output block-of-16 into an E4M3 scale factor and writes
both the FP4 D tensor and its SFD scale tensor. This is exactly what the FP4
training loop wants — produce FP4 outputs that the *next* GEMM can consume
directly without a separate quantization pass.

`72c_blackwell_mixed_mxfp8_bf16_gemm.cu` is the MXFP8×MXFP4 mixed path,
demonstrating that the `mxf8f6f4` `tcgen05.mma.kind` accepts any combination of
{4, 6, 8}-bit floating types per operand provided alignments are met
(see Tables 5–8 of `blackwell_functionality.md` for the layout / tile / dispatch
matrix).

### K.2.2 Example 84a — sparse NVFP4

`84a_blackwell_nvfp4_bf16_sparse_gemm.cu` exercises the 2:4 structured-sparse
variant: the A operand is fed through a sparse compressor (already 2× compressed
along K), the dispatch policy is `KernelSparseTmaWarpSpecialized2SmNvf4Sm100`,
and the throughput is 2× of the dense NVFP4 path on the relevant tile shapes
(rows 12 and 13 of Table 3). For diffusion-transformer attention projections
this matters: the projection `x@W_qkv` where the encoder side of W is 2:4 sparse
runs at 16 PFLOPS effective throughput on B200 (vs 7.7 PFLOPS dense FP4),
matching the announced GB200 NVL72 16/32 PFLOPS specification.

### K.2.3 Examples 89, 90 — SM103 (GB300) FP4 ultra-GEMM

`89_sm103_fp4_ultra_gemm.cu` and `90_sm103_fp4_ultra_grouped_gemm.cu` target
the GB300 platform (SM103). SM103 introduces a new "ultra" variant of `tcgen05.mma`
that uses a wider data path — concretely, a `cta_group::4` form pairing four CTAs
across two TPCs, doubling effective tile dimensions vs. SM100. CUTLASS gates
these with `__CUDA_ARCH__ == 1030 && (__CUDA_ARCH_FEAT_SM103_ALL || CUDA_ARCH_FAMILY(1030))`.
For B200 reports these are forward-looking; B200 itself remains SM100 / SM100a.

### K.2.4 Example 91 — FP4 GEMV

`91_fp4_gemv.cu` is the M=1 decode case. With NVFP4 weights and BF16 outputs,
this is the kernel that drives a single-token decoder forward through a 70B/405B
LLM at minimum latency. It uses the warp-specialized 1-CTA path
(`KernelTmaWarpSpecialized1SmNvf4Sm100`) since the M dimension can't be split
across two SMs, and falls back to a SIMT-style epilogue (`NoSmemWarpSpecialized1Sm`)
to avoid the SMEM round-trip of TMA store for what is effectively a vector write.
For DiT-style video diffusion this is less directly relevant (most DiT work is
M ≫ 1), but it is the building block for any reduced-cost decoder distillation
of a video DiT.

### K.2.5 Build flags and hardware gates

Every Blackwell example must be built with:

```bash
cmake .. \
  -DCUTLASS_NVCC_ARCHS="100a" \             # SM100 + arch-features
  -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \        # PDL/GDC intrinsics
  -DCUTLASS_LIBRARY_KERNELS="all"
```

with **CUDA Toolkit ≥ 12.8** (12.9+ recommended for the FFMA-interleaving auto-pass
in `nvcc`). Runtime requires:

```
$ nvidia-smi --query-gpu=name,compute_cap --format=csv
NVIDIA B200, 10.0
…
$ nvidia-smi topo -m       # Must show NV18 between every pair for DistGEMM
```

`NV18` (NVLink 5, 18 lanes) corresponds to ≈900 GB/s unidirectional per GPU on a
GB200 NVL72, ~450 GB/s on an HGX B200 baseboard.

---

## K.3 CUTLASS Distributed GEMM (single-node TP overlap) — Hopper ex.65 → Blackwell ex.82

### K.3.1 Why DistGEMM exists — NCCL hogs SMs that should be doing math

The classical Megatron-LM tensor-parallel MLP is two back-to-back linear layers
separated by a non-linearity. With sharded activations the forward pattern is
**(AllGather → GEMM → activation → GEMM → ReduceScatter)**. The collective
operations themselves are NCCL kernels that occupy a substantial fraction
of the SMs (default 32+ for ring-AR, ring-AG, ring-RS). When NCCL and a CUTLASS
GEMM run on overlapping streams, they compete for SM occupancy, so on an HGX
H100 a fully unhidden NCCL ring-AG can lose 25–35% of GEMM throughput.

The naive fix — chunk the input along M into TP stages and pipeline
`AG_chunk_i ‖ GEMM_chunk_{i-1}` — works in principle but on GPUs the *split
GEMM* is itself slower (smaller m drives lower SM occupancy), so the gain is
limited unless the chunk size is large.

CUTLASS DistGEMM (ex. 65 for Hopper, ex. 82 for Blackwell) takes a different
approach — described in detail in the SHI Labs blog (Dec 2024) and the
CUTLASS 3.7.0 release notes. The idea:

1. **Use the GPU Copy Engine** (DMA) for inter-GPU rotation of operand A (or B)
   so that no SM is spent on communication.
2. **Use CUDA Graphs** to capture the entire pipeline so kernel-launch latency
   collapses.
3. **Use PDL (`griddepcontrol.launch_dependents`)** to start the dependent
   GEMM's prologue (TMA descriptor setup, SMEM allocation) before the previous
   GEMM has fully drained.
4. **Use peer-to-peer addressing** so the local GEMM writes its output directly
   into the next rank's input buffer, with the local rank's first stage running
   on its own pre-loaded shard so there is no exposed initial communication.

Critically, the CUTLASS GEMM kernel itself only requires a **few additional
instructions** (a `cuStreamWriteValue32` to signal "my shard is ready", and a
`wait` on the next shard's flag) — it remains otherwise unmodified, retaining
its hand-tuned TMA + persistent + warp-specialized structure. This is why
DistGEMM scales across kernels: any future `KernelTmaWarpSpecialized2SmSm100`
GEMM gets distributed for free.

### K.3.2 Hopper performance — 91/99/85/93%

The SHI Labs benchmark on 8×H100 SXM with NV18 (NVLink-4) tested four problem
sizes from Llama 70B/405B fc1/fc2 layers, all FP16, with the ping-pong warp-
specialized GEMM, 128×256×64 tile shape, 1×2×1 cluster. Reporting "best
achievable performance":

| Setting | DistGEMM efficiency vs. single-stage no-comm baseline |
|---------|--------------------------------------------------------|
| Llama 70B fc1 | **91%** |
| Llama 70B fc2 | **99%** |
| Llama 405B fc1 | **85%** |
| Llama 405B fc2 | **93%** |

Versus the 8-stage no-communication baseline (which represents an ideal split
GEMM): 92% / 93% / 89% / 91%. The conclusion: DistGEMM achieves ~70–80% of peak
single-GPU GEMM despite doing all of the inter-GPU rotation, and is faster than
the best CUTLASS-profiler-selected GEMM + NCCL by 8–15% (405B) and 12–24% (70B)
in operation-level comparison.

### K.3.3 Schedules

The DistGEMM API supports four 1D schedules (`include/cutlass/experimental/distributed/schedules/dist_gemm_1d_schedules.hpp`):

- `AllGather1D_TilingCD_RotatingA<TP>` — rotate A, tile C/D by N (column-parallel
  first-MLP layer)
- `AllGather1D_TilingCD_RotatingB<TP>` — rotate B
- `ReduceScatter1D_TilingA_RotatingC<TP>` — rotate C, accumulate into a
  partial-sum buffer, then scatter
- `ReduceScatter1D_TilingB_RotatingC<TP>`

Selection is a single line:

```cpp
using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<_8>;
```

### K.3.4 Blackwell ex.82 — same API, swap dispatch policy

Example 82 is structurally identical to ex.65 but configures:

```cpp
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ElementA = cutlass::float_e4m3_t;     // FP8 instead of FP16
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::float_e4m3_t;
using ElementD = cutlass::float_e4m3_t;
using MmaTileShape_MNK   = Shape<_256,_256,_128>;
using ClusterShape_MNK   = Shape<_2,_1,_1>;   // 2-CTA UMMA, 1 TPC pair per cluster
using PerSmTileShape_MNK = Shape<_128,_256,_128>;
using KernelMainloop = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
```

and wraps the resulting `GemmKernel` with `DistributedGemmKernelWrapper<GemmKernel,
DistSchedule>`. The default sample command line is

```
./82_blackwell_distributed_gemm --m=16384 --n=106496 --k=16384 \
   --warmup-iterations=10 --iterations=100
```

(N=106496 = 4× the LLaMA 70B fc1 hidden dim of 26624, hidden-mult-of-2 padded for
TP=8.)

### K.3.5 Limitations and what is still open

**No published efficiency numbers for Blackwell ex.82.** The example ships, the
build is parameterized correctly, but as of CUTLASS 4.4 there is no equivalent of
the SHI Labs Hopper table for Blackwell. The expected behavior (extrapolating from
the Hopper architecture, the 2× increase in NVLink BW from 450 → 900 GB/s, and
the 1.27–2× increase in tensor core throughput) is that DistGEMM remains
compute-bound for the large fc1 layers and the bandwidth-to-compute ratio
should leave the 91/99/85/93% figures *roughly intact*; for fc2 (which has the
larger reduction dimension) we would expect Blackwell DistGEMM to be slightly
more bandwidth-limited than Hopper, since FLOP/byte favors the compute side.

**No NVFP4 DistGEMM example.** All four narrow-precision dispatch policies
(`KernelTmaWarpSpecialized2SmNvf4Sm100`, …) plug into the same
`DistributedGemmKernelWrapper` mechanism, but no example yet exercises a
2-CTA NVFP4 GEMM with all-gather rotation. This is the obvious next step
for FP4 inference — a tensor-parallel decoder using NVFP4 weights would want
this exact kernel.

---

## K.4 Triton-distributed (ByteDance Seed) — distributed compiler with tile-level barrier release

### K.4.1 Motivation

Triton-distributed (arXiv:2504.19442, MLSys '25) is the first Triton-compatible
**distributed compiler**: it extends Triton with OpenSHMEM-aligned communication
primitives and lets a kernel author write fine-grained communication-computation
overlapping in pure Python. The motivating problem is the same as DistGEMM
(NCCL eats SMs, kernel-level overlapping is too coarse) but the solution scales
to inter-node, MoE expert parallel A2A, and SP all-to-all attention — patterns
that DistGEMM's CUDA-graphs + copy-engine approach cannot reach.

### K.4.2 Programming model — symmetric memory + signals + async-task

Three concepts:

1. **Symmetric memory** — every rank allocates a buffer of the same size at the
   same logical offset; remote buffers are addressed via `tdl.symm_at(buf, peer)`,
   *not* via a unified VA pointer. The Python helpers
   `nvshmem_create_tensor`, `nvshmem_create_tensors`,
   `nvshmem_free_tensor_sync` wrap NVSHMEM's
   `nvshmem_malloc` / `nvshmem_team_my_pe` / `nvshmem_free`.
2. **Signal exchange** — every cross-rank synchronization is a one-sided
   write/wait on a signal residing in symmetric memory. Primitives include
   `tdl.notify(sig, peer)`, `tdl.wait(sig, value)`, `tdl.consume_token(token)`,
   `tdl.atomic_cas`, `tdl.ld_acquire`, `tdl.red_release`, `tdl.multimem_st`,
   `tdl.multimem_ld_reduce` (the latter two driving NVSwitch in-network
   reduction).
3. **Async-task** — different threadblocks run in parallel; each is assigned
   one role (intra-node dispatch, inter-node send, GEMM, …). Signals
   coordinate them.

Two sets of primitives are exposed (Table 1 of the paper):

- **OpenSHMEM**: `my_pe`, `n_pes`, `int_p`, `remote_ptr`, `barrier_all`,
  `sync_all`, `quiet`, `fence`, `getmem(_nbi)`, `putmem(_nbi)`,
  `putmem_signal(_nbi)`, `signal_op`, `signal_wait_until`, `broadcast`.
- **non-OpenSHMEM**: `wait`, `consume_token`, `notify`, `atomic_cas`, `atomic_add`,
  `ld_acquire`, `red_release`, `multimem_ld_reduce`, `multimem_st`.

The compiler lowers OpenSHMEM primitives to NVSHMEM bitcode at JIT time
(or to ROCSHMEM on AMD); non-OpenSHMEM primitives compile directly to PTX
inline asm.

### K.4.3 Key kernels

The repository at `python/triton_dist/kernels/nvidia/` contains 30+ kernels.
The most important for video DiT and LLM inference:

- **`allgather_gemm.py`** — intra-node and inter-node AG+GEMM. Producer GEMM
  embeds a `tdl.notify` after each output tile; consumer waits via `tdl.wait` +
  `tdl.consume_token` (the token primitive forces a true data dependency on the
  signal so the Triton compiler doesn't reorder loads ahead of the wait).
  Tile-coordinate swizzling shifts each rank's GEMM start position so the
  earliest tiles are the ones whose dependent communication will arrive last
  — a transformation that is straightforward in Triton but extremely fiddly
  in CUTLASS.
- **`gemm_reduce_scatter.py`** — pushes partial outputs to peer ranks via
  `putmem_signal_nbi`, then a separate stream's reduction kernel waits and
  accumulates with `red_release` semantics so the next iteration sees the
  reduced value.
- **`gemm_allreduce.py`** — uses `multimem_st` (NVSwitch broadcast write) and
  `multimem_ld_reduce` (NVSwitch in-network sum-reduce) so the reduction
  happens *inside the switch fabric*, not on any GPU's SMs.
- **`low_latency_all_to_all.py`** — small-message LL protocol: 8-byte data+flag
  pairs (since 8-byte writes are atomic across NVLink), inline-broadcast via
  `multimem_st`. The paper estimates 13.5 µs for a small LL all-to-all on 4
  ranks vs. ~25 µs for naive looped P2P.
- **`sp_ulysess_qkv_gemm_all2all.py`** + **`sp_ulysess_o_all2all_gemm.py`** —
  the QKV-projection-fused-with-A2A and the O-projection-fused-with-A2A halves
  of DeepSpeed-Ulysses sequence parallelism. These are the kernels FastVideo
  / xDiT use for SP attention in video DiT.
- **`sp_ag_attention_intra_node.py`** + **`sp_ag_attention_inter_node.py`** —
  ring-attention-style fused AllGather + FlashAttention.
- **`ep_a2a.py`** + **`ep_all2all_fused.py`** — MoE expert-parallel dispatch /
  combine.

### K.4.4 Performance vs the field

Reported on H800 + CX7 IB clusters:

| Kernel | Speedup vs. baseline |
|--------|----------------------|
| AG+GEMM intra-node (8 H800) | 1.42× vs PyTorch+NCCL, 1.09× vs FLUX |
| GEMM+RS intra-node (8 H800) | 1.28× vs PyTorch+NCCL, 1.30× vs FLUX |
| AG+MoE intra-node (15 shapes) | 44.97× vs PyTorch+NCCL (avg) |
| MoE+RS intra-node (10 shapes) | 15.55× vs PyTorch+NCCL (avg) |
| AG+GEMM inter-node (16 H800) | 1.33× vs PyTorch+NCCL, 95.6% of FLUX |
| GEMM+RS inter-node (16 H800) | 1.42× vs PyTorch+NCCL, 96.4% of FLUX |
| AllToAll dispatch (32 H800)  | 1.18× vs DeepEP |
| AllToAll combine (32 H800)   | 1.44× vs DeepEP |
| AllGather low-latency (8 L20 PCIe) | 1.40–1.48× vs NVSHMEM, 3.0–3.1× vs NCCL |

The MoE A2A numbers explicitly: dispatch ~120 µs / combine ~75 µs at 32 H800,
versus DeepEP at ~140 µs / ~110 µs (the relevant figures in the paper's Figure 16,
matching the 1.18× and 1.44× speedups respectively). At 64 H800 DeepEP starts
to win on dispatch because DeepEP uses IBGDA while Triton-distributed uses
IBRC — a known limitation gated by NVSHMEM bitcode IBGDA support.

For inference SP: single-shape examples in the public benchmark show
Triton-distributed achieving 137 µs MoE A2A on 32 H800 vs. DeepEP's 182 µs on
the same shape (configured for low-latency inference, not high-bandwidth
training). For training-style high-bandwidth A2A on 8 H20 the picture
inverts — DeepEP wins (20.45 µs dispatch vs 47 µs) because Triton-distributed's
peripheral D2D copies expose more straggler latency. The kernels are
complementary, and DeepGEMM Mega MoE in fact uses both.

### K.4.5 Resource partitioning

For inter-node GEMM+RS on H800 the paper gives the canonical SM split:

- GEMM: 116 SMs
- intra-node scatter: 0 SMs (copy engine)
- cross-node P2P: 1 SM (one threadblock, ~45 GB/s saturating)
- first local reduction: 16 SMs
- second local reduction: 132 SMs

producing a "perfect overlap" timeline where every async-task finishes at
~the same time. This kind of explicit resource budget is exactly what is
*missing* from NCCL/NVSHMEM-only flows.

### K.4.6 Concrete kernel structure — `allgather_gemm.py` walkthrough

The intra-node AG+GEMM kernel follows a producer-consumer pattern in which
the *consumer* (GEMM) and the *producer* (gather) are *separate Triton kernels*
launched on separate streams, coordinated through a symmetric signal pad.
Sketched:

```python
# Allocate symmetric memory tensors (one per rank, identical layout)
gather_buf = nvshmem_create_tensor((world_size * M_per_rank, K), torch.bfloat16)
sig_pad    = nvshmem_create_tensor((world_size,), NVSHMEM_SIGNAL_DTYPE)

# 1. Each rank copies its local shard into its own slot of gather_buf
#    using either copy_engine or a kernel; sets sig_pad[rank] = 1
local_copy_and_barrier_all(local_rank, rank, world_size,
                            local_data=A_local,
                            global_data=gather_buf,
                            barrier_ptr=sig_pad,
                            comm_buf=symm_sync, ...)

# 2. Producer (gather) kernel on stream 0:
#    For each peer p, push our shard to peer's slot using copy engine
#    or an internal copy kernel. Set remote sig_pad[rank] = 1 on completion.
cp_engine_producer_all_gather_intra_node(rank, world_size, gather_buf,
                                          sig_pad, stream=stream_comm)

# 3. Consumer (GEMM) kernel on stream 1:
#    For each tile (pid_m, pid_n):
#      - Determine which peer's slot of gather_buf this tile draws from
#      - tdl.wait(sig_pad[that_peer], 1) ← spin until producer set it
#      - tile_token = tdl.consume_token(...) ← force a true dependency
#      - tile_a = tdl.load(gather_buf + offset, ...) [under tile_token]
#      - tile_b = tdl.load(B + offset, ...)
#      - acc = tl.dot(tile_a, tile_b, acc) [tcgen05.mma on B200, wgmma on H800]
#      - At end of K loop, tl.store(C, acc.to(bf16))
gemm_kernel_with_wait[grid](A=gather_buf, B=B_local, C=C_local,
                             sig_pad=sig_pad, ...)

torch.cuda.synchronize()  # wait for both streams
```

The `tdl.consume_token` is critical: without it the Triton compiler is free to
hoist the `tdl.load(gather_buf, ...)` above the `tdl.wait`, breaking
correctness. With it, the load becomes data-dependent on the wait's completion.

Tile-coordinate **swizzling** is the second optimization: the consumer's tile
order is permuted so the first tile each rank computes draws from the rank
*after* itself in the rotation order, since that rank is what every other
rank is gathering first. This staggered start removes the bookend
no-overlap phase that naive split-GEMM suffers from.

---

## K.5 PyTorch SymmetricMemory 2.12 + NVSHMEM4Py + Meta Kraken

### K.5.1 The torch.distributed._symmetric_memory API (PT 2.12 alpha)

PyTorch landed `torch.distributed._symmetric_memory` (still in alpha as of 2.12)
to expose NVLink/NVSHMEM-style symmetric memory at the framework level. The
core API:

```python
import torch.distributed._symmetric_memory as symm_mem

t = symm_mem.empty(128, device=torch.device("cuda", rank))
hdl = symm_mem.rendezvous(t, group)

# Three pointer arrays + signal pads
hdl.buffer_ptrs        # len = world_size, peer-mapped HBM addresses
hdl.multicast_ptr      # one address; writes broadcast via NVSwitch multicast
hdl.signal_pad_ptrs    # len = world_size, used for sync barriers

# Built-in op
torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group)
```

The `empty` + `rendezvous` calls *must* be ordered identically across ranks.
Once rendezvoused, kernels can index any rank's buffer directly. Kernels can
be CUDA, CUTLASS, or Triton — and the latter is the workhorse, since
SymmMem composes with a Triton kernel as just a few extra `tl.load`/`tl.store`
on a peer pointer plus a `ptx_utils.symm_mem_sync` synchronization.

PyTorch 2.12 adds three substantial features on top:

1. **MemPool support** — `symm_mem.get_mem_pool(device)` returns a CUDA mem-pool
   that, when used inside `torch.cuda.use_mem_pool(mempool)`, allocates *every*
   tensor (including outputs of `torch.mm`) from symmetric memory. This is the
   easy on-ramp: take an existing model, wrap a forward pass in the mem-pool
   context, and now all the activations are symmetric-memory-backed and can be
   collectively reduced in-place.
2. **Copy Engine (CE) Collectives** — when NCCL is configured with
   `cta_policy = NCCL_CTA_POLICY_ZERO` and the buffers are symm-mem-backed,
   `dist.all_gather_into_tensor()` / `dist.all_to_all_single()` automatically
   use the GPU's DMA Copy Engines instead of SMs, freeing 100% of SMs for compute.
   Requires NCCL ≥ 2.28, P2P access enabled, and `async_op=True`.
3. **Higher-precision reduction** — when symm-mem-backed BF16/FP16 tensors are
   reduced via `reduce_scatter` or `all_reduce`, NCCL's symmetric kernel
   accumulates internally in FP32 (BF16/FP16 in → FP32 accumulate → BF16/FP16
   out). This reproduces the PyTorch CUDA semantics of `tensor.sum()` and
   eliminates a class of training instability when scaling reductions over
   large groups.

### K.5.2 NVSHMEM 3.5.19 / 3.6.5 — tile RMA + NVLink-SHARP

The NVSHMEM 3.5/3.6 release adds two features the upper layers depend on:

- **Tile-RMA primitives** — `nvshmem_tile_get` / `nvshmem_tile_put` operate on
  a 2-D tile descriptor (similar to TMA's `cp.async.bulk.tensor.2d`), letting
  device kernels initiate one-sided 2-D transfers without manually computing
  per-row offsets. ParallelKittens' `store_async`/`store_add_async` and DeepEP V2
  both build on tile-RMA.
- **NVLink-SHARP** — the NVSwitch in-network reduction (multimem) is now
  exposed through standard NVSHMEM collective APIs as well as via the explicit
  `multimem.ld_reduce` / `multimem.red` PTX instructions. NVSwitch on
  GB200 NVL72 implements NVLink-SHARP for sum/min/max in BF16/FP16/FP32.
  The big win: an all-reduce that uses NVLink-SHARP no longer scales linearly
  with N (the number of GPUs) — the switch fabric does the reduction once,
  in flight.

NVSHMEM 3.6.5 adds Python bindings (`nvshmem4py`) and a Triton plugin
(`torch.distributed._symmetric_memory._nvshmem_triton`) that decorates a Triton
JIT kernel with `@requires_nvshmem` and exposes `nvshmem.put`, `nvshmem.get`,
`nvshmem.ld`, `nvshmem.st` primitives that operate in the cross-node setting
(ConnectX-7 RDMA / GPUDirect ASYNC).

### K.5.3 Meta Kraken — the cookbook

`meta-pytorch/kraken` is a Triton library of symmetric-memory operators built
on top of `torch.distributed._symmetric_memory`. Initial kernels were adapted
from Yifu Wang's `yifuwang/symm-mem-recipes`. The structure:

- `kraken.comm` — communication-only kernels: `one_shot_all_reduce`,
  `two_shot_all_reduce`, `all_gather_w_progress`, (coming) `multimem_all_reduce`.
- `kraken.fused` — comm-comp fusions: `gemm_one_shot_all_reduce_fused`,
  `gemm_reduce_scatter`, `gemm_reduce_scatter_ce_persistent`,
  `all_gather_matmul`, `one_shot_all_reduce_bias`,
  `one_shot_all_reduce_bias_rms_norm` (and two-shot variants).
- `kraken._ptx_utils` — the inline-PTX synchronization primitive
  `symm_mem_sync(signal_pad_ptrs, _, rank, world_size,
  hasPreviousMemAccess=True/False, hasSubsequenceMemAccess=True/False)` that
  Triton itself doesn't yet expose.

The blanket rule is "this is not a framework, it's a hackable cookbook". That
matches the deployment style: Cursor uses it directly for its in-house training,
embedded into FlashAttention and TorchInductor-generated kernels.

### K.5.4 Quantitative — SymmMem on Llama-3 8B/70B/105B

PyTorch's own benchmarking on the SymmMem-backed all-reduce path (from the
PyTorch 2.12 release notes) shows end-to-end speedups for tensor-parallel
training:

| Model      | TP=8 setting | SymmMem speedup vs. NCCL ring-AR |
|------------|--------------|-----------------------------------|
| Llama-3 8B | TP=8 H100    | **1.54×** |
| Llama-3 70B | TP=8 H100   | **1.47×** |
| Llama-3 105B | TP=8 H100  | **1.20×** |

The decline at 105B is consistent with the larger model becoming more compute-
bound (relative comm time falls), so the fraction of time SymmMem can hide is
smaller. On B200 with NVLink-5 SHARP the absolute communication time falls
roughly 2×; the relative speedup stays in roughly the same band but at higher
absolute throughput.

---

## K.6 ByteDance FLUX — CUTLASS + NVSHMEM fused AG-GEMM and GEMM-RS

### K.6.1 The fine-grained kernel-fusion idea

FLUX (arXiv:2406.06858, ByteDance + PKU) attacks the same TP overlap problem as
DistGEMM but with a different mechanism: **fuse** the communication tile-loop
into the GEMM kernel itself. Instead of N stages of `AG_chunk_i ‖ GEMM_chunk_{i-1}`,
FLUX over-decomposes both into many tiles and writes a single CUTLASS kernel
where each thread block:

1. Issues a small intra-cluster wait on the previous tile's signal,
2. Performs a tile of the GEMM,
3. Writes the output (or — for RS — atomically adds it) directly into the
   *destination* rank's buffer via NVSHMEM `nvshmem_putmem_nbi`,
4. Sets the destination signal.

Because the tile is the natural unit of CUTLASS GEMM execution, this is a
single fused kernel — no kernel-launch latency, no stream synchronization
overhead, no split GEMM penalty. The cost is some kernel design complexity
and a dependency on NVSHMEM symmetric memory, both of which FLUX absorbs.

### K.6.2 Quantitative results

FLUX is built on CUTLASS 4.0 + NVSHMEM (3.2.5 / 3.3.9 tested) + NCCL. On 128
H800 GPUs (16 nodes × 8) for tensor parallelism in training:

- Up to **96% communication overlap** in the fused kernel — i.e. effective
  communication time → 4% of unhidden time.
- Up to **1.24× training speedup** vs. Megatron-LM baseline (PyTorch + NCCL).
- Up to **1.66× prefill** and **1.30× decode** speedup vs. vLLM on 8 H800.
- Vs. the prior overlapping technique TransformerEngine: **1.38× train,
  2.06× prefill, 2.10× decode**.

For AG+GEMM specifically, FLUX is faster than Triton-distributed at small/medium
shapes (where the scheduler-warp overhead matters more) and ties with it at large
shapes; for GEMM+RS the Triton-distributed paper measures FLUX at ~5% higher
throughput than its own kernel. The two are essentially co-equal at the
state of the art, with PK matching or exceeding both at the cost of writing
in C++ rather than Python.

The 96% overlap claim comes from the *Effective Communication Time* (ECT)
metric the FLUX paper introduces:

```
ECT = OverallTime - GEMM_non-split
E_overlap = 1 - ECT_overlap / ECT_non-overlap
```

For a non-overlapping baseline `E_overlap = 0`; for a perfect overlap
`E_overlap = 100%`. The PyTorch + NCCL baseline is the strongest
non-overlapping reference (NCCL is the fastest ring-AR/ring-AG/ring-RS
library available). FLUX's 96% overlap means the wall time for an
AG+GEMM is essentially `GEMM_non-split + 4% × NCCL_AG_time` — i.e., the
communication is almost entirely hidden. By comparison, PyTorch + NCCL
has ECT = NCCL_AG_time (everything is exposed), and prior overlap methods
like TransformerEngine achieve roughly 50–70% overlap on the same shapes.

Critically, FLUX achieves this without making the user write a different
GEMM kernel — the FLUX framework ships a compatible-with-CUTLASS GEMM with
extra primitives prepended/inserted, and any future CUTLASS GEMM kernel
that matches the same template can be wrapped similarly. This is the
*architectural* reason FLUX scales better than hand-tuned overlap
implementations: it doesn't fork the GEMM, it composes with it.

### K.6.3 What FLUX adds on top — MoE and PCIe

Beyond dense AG+GEMM and GEMM+RS, FLUX implements:

- **MoE AG-scatter** and **MoE gather-RS** — fused token-dispatch grouped GEMM
  (these are the kernels the FLUX README's `test_moe_ag.py` and
  `test_moe_gather_rs.py` exercise).
- **PCIe support** — for L20 GPUs (no NVLink) the same primitive structure
  works over PCIe peer-to-peer; FLUX is the only framework in Table 2 of
  the Triton-distributed paper that does both PCIe and NVLink with full
  support for OpenSHMEM, low-latency protocol, copy engine, and fusion.

The companion paper **COMET** (next subsection) evolves FLUX's MoE kernels
into a more general MoE-overlap system.

---

## K.7 COMET — fine-grained computation-communication overlap for MoE (MLSys '25)

### K.7.1 Motivation

In Mixtral-8x7B / DeepSeek-V3 training on 8 H800, MoE communication
(dispatch + combine) accounts for **47% of forward time** with the standard
Megatron-LM stack. Existing coarse-grained pipelining (the chunk-the-input
strategy) reduces this only modestly because:

1. Splitting an expert GEMM into K chunks reduces per-chunk arithmetic intensity
   (smaller M ⇒ lower SM occupancy ⇒ longer per-chunk time → t1+t2 > t).
2. The first-chunk dispatch and last-chunk combine are *non-overlappable*
   bookend phases.
3. Token-level data dependency (each output tile of an expert GEMM needs tokens
   that come from many different ranks at different times) creates a
   data-dependence graph that block-level kernels cannot follow.

### K.7.2 COMET's two mechanisms

**Shared-tensor dependency resolving**: COMET identifies the buffer that is
both the producer's output and the consumer's input (the "shared tensor")
— in MoE this is the per-expert token buffer — decomposes it along
specific dimensions, and reschedules the consumer's tile-iteration order so
the tiles whose data arrives *first* are processed first. Concretely, the
consumer GEMM loops over expert × tile-of-N × tile-of-K with the iteration
order chosen so that within each expert, the rank whose tokens arrived first
contributes the first K-chunks.

**Adaptive workload assignment**: COMET fuses dispatch + Group-GEMM into a
single CUDA kernel using thread-block specialization (à la Hopper warp-spec):
some thread blocks issue communication (NVSHMEM put/get), others do GEMM, and
the relative count is auto-tuned per problem shape and GPU model. This
isolation prevents communication-side memory stalls from blocking the
compute-side TMA pipeline — which is otherwise a real failure mode on Hopper
where TMA pipelines have multi-stage latency.

### K.7.3 Results

On 8 H800 with Mixtral-8x7B / Qwen2-MoE / Phi3.5-MoE:

- **1.96× speedup on a single MoE layer** (averaged over models)
- **1.71× end-to-end** speedup vs. Megatron-LM
- Production deployment: **millions of GPU hours saved** at ByteDance ten-thousand-
  GPU scale.

COMET is open-sourced as part of the FLUX repository (the FLUX README cites
both the FLUX 2024 paper and the COMET 2025 paper). It is also the strongest
EP A2A baseline that ParallelKittens compares against (PK reaches 0.92–1.22×
Comet on EP token-dispatch + first-MLP).

---

## K.8 ThunderKittens 2.0 + ParallelKittens

### K.8.1 ThunderKittens 2.0 (Jan 2026) — Blackwell support

ThunderKittens (TK) is a header-only CUDA tile-DSL that has been the de-facto
"hand-written but ergonomic" baseline since 2024. The 2.0 release (Jan 11, 2026)
brings:

- **Full Blackwell SM100 support** — `tcgen05.mma` wrappers, TMEM allocator,
  2-CTA UMMA helpers.
- **MXFP8 / NVFP4 precision** — `st_e4m3`, `st_e5m2`, `st_fp4_e2m1`, `rt_fp8`
  data types and corresponding tile MMA functions.
- **Repository restructure** — kernels now live in `kernels/` with
  per-kernel Makefiles; the top-level `setup.py` is gone. Compile each kernel
  individually.
- **Drop Ampere** — Volta/Turing/Ampere are no longer actively supported.

For DiT-style attention the relevant entry point remains
`csrc/attention/st_attn_h100.cu` (the H100 ST-attention kernel that achieves
~855 TFLOPS, 86% of theoretical peak) plus its Blackwell sibling under
`kernels/attention/st_attn_b200.cu`. On B200 the kernel uses 2-CTA UMMA at
`MmaTile = <_256,_128,_64>`, keeps Q/K/V tiles in TMEM throughout
softmax(QK^T)·V, and demonstrates the chained-MMA-keeps-intermediate-in-TMEM
pattern from K.1.2.

### K.8.2 ParallelKittens (PK) — multi-GPU primitives (arXiv:2511.13940)

PK extends TK to multi-GPU. The motivation comes from a clean cost-model
analysis (Section 3.1 of the paper):

```
T_kernel = T_launch + max(T_comp, T_mem, T_comm) + T_non-overlap + T_sync
```

with three hardware-driven design axes: **transfer mechanism**, **scheduling**,
**design overheads**. The paper measures each:

- **Transfer mechanism** (NVLink BW, 1 GB transfer):
  - Copy Engine: 368.82 GB/s on H100 (82%), 726.13 GB/s on B200 (81%)
  - TMA Op: 350.01 GB/s H100 (78%), 669.12 GB/s B200 (74%)
  - Register Op: 342.68 GB/s H100 (76%), 628.35 GB/s B200 (70%)
  Copy Engine needs 256 MB messages to saturate; TMA needs only 2 KB.
  Register-level needs ~76 SMs to saturate vs ~15 for TMA.
- **Synchronization**:
  - intra-SM mbarrier: 64 ns
  - inter-SM HBM-routed: 832 ns (13× higher)
- **Functionality matrix**:
  | Functionality          | CE | TMA | Reg |
  |------------------------|----|-----|-----|
  | P2P transfer           | ✓  | ✓   | ✓   |
  | In-fabric broadcast    | ✓  | ✓   | ✓   |
  | P2P reduction          | ✗  | ✓   | ✓   |
  | In-fabric reduction    | ✗  | ✗   | ✓   |
  | Elementwise transfer   | ✗  | ✗   | ✓   |

The conclusion: TMA is the right default (low SM cost, low message size to
saturate, supports atomic add for P2P reduction); register-level is needed
only for in-network reduction (multimem) and elementwise transfer.

### K.8.3 PK abstractions — eight primitives

```
// P2P
store_async(dst, src, coord)            // shared tile → multicast memory
store_add_async(dst, src, coord)        // atomic add to peer

// In-network
reduce(dst, dst_coord, src, src_coord)  // multimem.ld_reduce → local HBM
all_reduce(dst_and_src, coord)          // multimem all-reduce in place

// Synchronization
signal(bar, coord, dev_idx, val)
signal_all(bar, coord, val)
wait(bar, coord, dev_idx, expected)
barrier(bar, coord, dev_idx)
```

…and a four-component program template: **loader / storer / consumer / communicator**
(Figure 1 of the paper). Each warp-group plays one role; SM-level partitioning
between consumer and communicator is auto-tuned at runtime.

### K.8.4 Quantitative results (8×H100, BF16)

| Workload                    | PK speedup vs. baseline |
|-----------------------------|--------------------------|
| AG+GEMM (DP/TP)             | 1.06–2.33× over cuBLAS+NCCL |
| GEMM+RS (DP/TP)             | up to 1.20× from intra- vs inter-SM choice |
| GEMM+AR (DP/TP)             | 3.62× from in-network reduction |
| Ring-Attention (SP)         | **1.07–4.08×** vs. xDiT |
| DeepSpeed-Ulysses A2A (SP)  | 1.01–1.39× vs. YunChang |
| Expert-parallel A2A + GEMM  | **0.92–1.22× vs. COMET** (matching SOTA) |
| Pure all-reduce             | 1.79× vs. NCCL, 4.5× lower latency vs NVSHMEM |

Headline summary used in the report's intro: PK delivers **2.33× DP/TP, 4.08× SP,
1.22× EP** over the strongest published baselines, in fewer than 50 lines of
device code per kernel beyond the original single-GPU TK kernel. PK matches or
exceeds Flux, Comet, and CUTLASS DistGEMM on the workloads where they each
specialize, and outperforms compiler-based approaches (Triton-distributed) by
1.07–5.63× across problem shapes — the latter is partially because Triton-
distributed is tuned for H800, but PK retunes per-architecture trivially.

PK is fully compatible with B200 (Appendices A and B of the paper) and shows
similar relative performance characteristics; absolute numbers scale with the
2× NVLink-5 BW and 1.27–2× tensor-core throughput.

---

## K.9 DeepGEMM, DeepEP, FP4 Mega MoE — DeepSeek's kernel stack

### K.9.1 DeepGEMM — clean FP8/FP4/BF16 GEMM with JIT

DeepGEMM is "a unified, high-performance tensor core kernel library that brings
together the key computation primitives of modern LLMs into a single, cohesive
CUDA codebase." Compared to CUTLASS it is intentionally smaller — a few core
kernels, no template explosion — and entirely **JIT-compiled** at runtime via
NVCC or NVRTC (no CUDA compilation during pip install).

Highlights as of v3 (April 2026, PR #304):

- **SM90 + SM100** support (Hopper + Blackwell)
- **FP8 GEMM** with both FP32-format scales (SM90) and packed-UE8M0 scales (SM100)
- **FP4 GEMM** (FP8×FP4 mixed) on SM100
- **Mega MoE** — fused-and-overlapped EP-dispatch / linear-1 / SwiGLU /
  linear-2 / EP-combine into a single mega-kernel with NVLink communication
  overlapping tensor-core computation. FP8×FP4 MoE only; requires PyTorch ≥ 2.9
  and symmetric memory.
- **MQA Logits** — paged and non-paged FP8 MQA logits kernels for the lightning
  indexer used in DeepSeek-V3.2.
- **PDL** support — `deep_gemm.set_pdl(True)` enables programmatic dependent
  launch globally.
- **Up to 1550 TFLOPS on H800** for the dense FP8 GEMM (90% of 1958 TFLOPS
  theoretical peak; SM90 PRs #74, #78, #81, #86).
- **NVRTC support** for 10× faster JIT compilation (`DG_JIT_USE_NVRTC=1`,
  with some performance loss in edge cases since NVRTC misses some NVCC
  optimizations like FFMA interleaving — though NVCC 12.9 does FFMA
  interleaving automatically now).
- **DeepEPv2 MoE GEMM layout** — coordinates with the new DeepEP V2 dispatch
  format.

The library uses the convention `D = C + A @ B` with input layout NT (non-
transposed A, transposed B). SM90 supports only NT; SM100 supports NT/TN/NN/TT.
The LHS scaling factor is required to be TMA-aligned and transposed.

### K.9.2 Mega MoE — fused multi-stage MoE with overlap

Mega MoE is the marquee feature of DeepGEMM v3. The pipeline:

```
EP dispatch  ── linear 1 (FP8×FP4) ── SwiGLU ── linear 2 (FP8×FP4) ── EP combine
       ↑                                                                    ↓
       └── overlapped: NVLink communication while tensor cores compute ─────┘
```

This is fused into a single mega-kernel that operates on a symmetric-memory
buffer:

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden
)
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

# Per-iteration:
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

Mega MoE benchmarks (PR #316) on B200: substantially better than separate
{DeepEP dispatch + DeepGEMM linear-1 + SwiGLU + DeepGEMM linear-2 + DeepEP combine}
because the kernel fusion eliminates SMEM round-trips of the intermediate
activations, exploits the fact that the intermediate-hidden stays in TMEM
across linear-1 → SwiGLU → linear-2 (the chained `D=(A·B)·C` pattern from
K.1.2), and overlaps NVLink dispatch/combine with tensor-core compute through
2-CTA UMMA persistent kernels.

It has been integrated into SGLang as the MoE layer runner backend in NVLink
domain (PR #23167), and TRT-LLM PR #6486 wires DeepGEMM into TRT-LLM's MoE
runtime — pulling DeepGEMM into the broader inference-server ecosystem.

The TRT-LLM PR #6486 is particularly important because it shows how DeepGEMM
plugs into a production inference server: the PR adds a "DeepGEMM" backend
alongside the existing TensorRT plugin and the CUTLASS-based MoE runner, and
the user selects via a runtime config. For SM100 inference workloads where the
tile sizes match DeepGEMM's optimization targets (BS=64–256, hidden=4K–8K,
intermediate=11K–14K), TRT-LLM with DeepGEMM backend regularly outperforms
TRT-LLM with CUTLASS backend by 5–15% — without changing the user-facing API.

Environment variables that matter for production deployment:

| Variable | Purpose |
|---------|---------|
| `DG_JIT_CACHE_DIR` | Directory for compiled kernel cache (default `$HOME/.deep_gemm`) |
| `DG_JIT_USE_NVRTC` | Use NVRTC instead of NVCC (10× faster JIT, mild perf loss) |
| `DG_JIT_USE_RUNTIME_API` | Use CUDA Runtime API for kernel loading (CUDA ≥ 12.8) |
| `DG_PRINT_CONFIGS` | Print kernel config selection per shape (debugging) |
| `DG_JIT_DUMP_PTX` / `DG_JIT_DUMP_SASS` | Dump generated PTX/SASS for kernel inspection |
| `DG_COMM_KERNEL_DEBUG` | Zero symmetric buffer between Mega MoE calls (debug) |

### K.9.3 DeepEP V2 — ElasticBuffer, NCCL Gin backend

DeepEP is DeepSeek's expert-parallel communication library. V2 (April 2026)
is a complete refactor:

- **NCCL Gin backend** replaces NVSHMEM (lightweight, header-only, can reuse
  existing NCCL communicators).
- **ElasticBuffer** unifies the high-throughput (training) and low-latency
  (inference) APIs into a single allocation.
- **Analytical SM count** — `buffer.get_theoretical_num_sms(num_experts, num_topk)`
  returns the optimal SM allocation; no auto-tuning needed.
- **EP up to 2048** — much larger scale-out domains than V1.
- **0-SM Engram (RDMA), 0-SM PP (RDMA), 0-SM CP (Copy Engine)** — experimental
  primitives that move data via the dedicated DMA path with zero SM occupation.
- **For DeepSeek V3-style training: SM usage drops from 24 to 4–6** with
  equivalent or better performance.

Performance on an SM100 cluster (8K tokens/batch, 7168 hidden, top-8 experts,
FP8 dispatch / BF16 combine):

| Arch  | NIC | Topo    | Dispatch BW (logical) | Combine BW (logical) | #SMs |
|-------|-----|---------|------------------------|------------------------|------|
| SM90  | CX7 | EP 8×2  | 90 GB/s (RDMA)         | 81 GB/s (RDMA)         | 12   |
| SM100 | CX7 | EP 8×2  | 90 GB/s (RDMA)         | 91 GB/s (RDMA)         | 12   |
| SM100 | —   | EP 8    | **726 GB/s (NVLink)**  | **740 GB/s (NVLink)**  | 64 (max perf) |
| SM100 | —   | EP 8    | 643 GB/s (NVLink)      | 675 GB/s (NVLink)      | **24 (min #SM)** |

V2 achieves up to 1.3× peak performance over V1 and saves up to 4× in SM count.
The 740 GB/s combine BW on B200 NVLink-5 is 82% of the 900 GB/s unidirectional
ceiling — i.e., DeepEP V2 + Mega MoE saturates NVLink while leaving 144−24 = 120
SMs free for MMA work. That is the key reason MoE inference latency on B200 is
fundamentally lower than on H200 by more than the 2× compute and 2× BW would
predict.

---

## K.10 FlashInfer + sgl-kernel + q8_kernels

### K.10.1 FlashInfer — the inference kernel library

FlashInfer (`flashinfer-ai/flashinfer`, arXiv:2501.01005) is the kernel library
that powers SGLang, vLLM, TensorRT-LLM, TGI, MLC-LLM, LightLLM, ScaleLLM, and
others. It provides unified APIs for attention / GEMM / MoE with multiple
backend implementations (FA-2, FA-3, cuDNN, CUTLASS, TRT-LLM, custom CUTLASS-
free kernels) and automatically selects the best per-shape, per-arch backend.

Key Blackwell-relevant features in v0.4.0+ (Oct 2025):

- **Attention** — paged + ragged KV-cache, decode/prefill/append, MLA
  (DeepSeek's Multi-Latent Attention, native), cascade attention (memory-
  efficient hierarchical KV-cache for shared prefixes), block-sparse and
  variable block-sparse, **POD-Attention** (fused prefill+decode for mixed
  batching).
- **GEMM/Linear** — BF16 GEMM for SM10.0+, FP8 GEMM (per-tensor and groupwise
  scaling), **FP4 GEMM (NVFP4 + MXFP4 for Blackwell)**, grouped GEMM for LoRA
  and multi-expert routing.
- **MoE** — fused MoE kernels with multiple routing methods (DeepSeek-V3,
  Llama-4, standard top-k); FP8 / FP4 expert weights with block-wise scaling.
- **Sampling** — sorting-free Top-K / Top-P / Min-P; chain speculative sampling.
- **Communication** — custom AllReduce; multi-node NVLink (MNNVL) support;
  NVSHMEM integration for distributed memory ops.
- **Other** — RoPE (LLaMA / LLaMA 3.1), RMSNorm, LayerNorm, Gemma fused norms,
  SiLU/GELU with fused gating.

GPU support spans SM7.5 (Turing) → SM12.0 (RTX 50). For B200 (SM10.0) and
B300 (SM10.3) the library auto-routes to the CuTe-DSL Blackwell kernels in
the `flashinfer-python[cu13]` extra. Three install variants:

- `flashinfer-python` — core, JIT-compiles on first use
- `flashinfer-cubin` — pre-compiled cubins for all archs
- `flashinfer-jit-cache` — pre-built JIT cache for a specific CUDA version

FlashInfer's attention API is the reference example of the "single API
selects the best backend" pattern. The signature:

```python
import flashinfer

# Single decode (one query, paged or ragged KV)
output = flashinfer.single_decode_with_kv_cache(q, k, v)

# Batched decode with paged KV
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, kv_layout="NHD"
)
prefill_wrapper.plan(qo_indptr, paged_kv_indices, paged_kv_indptr,
                      paged_kv_last_page_len, num_qo_heads, num_kv_heads,
                      head_dim, page_size, causal=True)
output = prefill_wrapper.run(q, paged_kv_cache)

# MLA (DeepSeek's Multi-Latent Attention)
mla_wrapper = flashinfer.BatchMLAPagedAttentionWrapper(workspace_buffer)
mla_wrapper.plan(qo_indptr, paged_kv_indptr, paged_kv_indices, ...)
output = mla_wrapper.run(q_nope, q_pe, paged_kv_cache_nope, paged_kv_cache_pe)
```

Internally, `plan()` chooses among:

- **FA-3 SM90** (the published Hopper kernel from Dao Lab)
- **CUTLASS Blackwell FMHA** (CUTLASS ex.77, dense FA on B200)
- **TRT-LLM custom XQA / MLA kernel** (NVIDIA's hand-tuned path for
  DeepSeek-style MLA with page sizes 32/64/128)
- **cuDNN** for older arches or special head dimensions
- **Custom CuTe-DSL kernels** for SM10.0/10.3 in the `[cu13]` extra
- **JIT-compiled custom kernel** when none of the above match (variable head
  dim, custom mask, learned bias, etc.)

For a video DiT inference path the relevant choice is between FA-3 (Hopper-
optimized, works on Blackwell with reduced tensor-core utilization) and the
CuTe-DSL CUTLASS Blackwell FMHA — the latter being roughly 1.4–1.6× faster on
B200 at typical DiT shapes (M=N=video token count of 4K–32K).

### K.10.2 sgl-kernel — SGLang's auxiliary kernel set

The `sgl-kernel/` subdirectory of `sgl-project/sglang` is SGLang's local
kernel library (separate from FlashInfer). It provides Blackwell-tuned kernels
for operations that are SGLang-specific:

- **Token-level routing** — fused expert-routing kernels that cooperate with
  SGLang's RadixAttention KV-cache.
- **Position embeddings** — YaRN, NTK-aware RoPE, Llama-3.1 long-context RoPE.
- **Speculative-decoding fused kernels** — chain sampling + tree-attention
  fused, used for SGLang's speculative-decoding path.
- **MLA prefill / decode** — DeepSeek-V3 MLA paths.
- **MoE wrappers** — thin wrappers over DeepGEMM Mega MoE (PR #23167) and
  FlashInfer fused-MoE.
- **Custom AllReduce** — matches FlashInfer's custom AR but tuned to SGLang's
  cuda-graph capture.

The sgl-kernel package is built into the SGLang wheel; it follows SGLang
release cadence (2.0+ targets B200 by default).

### K.10.3 q8_kernels — Lightricks LTX-Video FP8 inference

`Lightricks/LTX-Video-Q8-Kernels` (also referenced as `LTXVideo-Q8-Kernels`)
is the FP8 inference kernel set used to drive LTX-Video / LTX-2 video
diffusion at minimum latency. Architecturally it is a small CUTLASS-based
package targeting:

- **SM89 (Ada Lovelace)** — RTX 4090, L40S, L4: uses Ada FP8 tensor cores
  (1×Ada peak with FP32 accumulator, 2× peak with FP16 accumulator). The
  `mma_sm89_fp16.hpp` file in the repo defines the relevant CUTLASS
  `MMA_TRAITS` for `m16n8k32 e4m3 e4m3 fp16` and `e4m3 e4m3 fp32` shapes.
- **SM90 (Hopper)** — H100/H200: uses `wgmma` FP8 path.

Key user-visible properties (from the LTX-Video / ComfyUI-LTXVideo integration):

- **Up to 3× speedup on Ada GPUs** vs. BF16 baseline.
- Patches the diffusion transformer (`q8_kernels.integration.patch_transformer`)
  so attention, FFN, and projection layers all run FP8 end-to-end.
- Selectable per-block: `use_fp8_attention`, `quantize_self_attention`,
  `quantize_cross_attention`, `quantize_feed_forward`.
- Quantization presets (`0.9.8`, `ltxv2`, `full_bf16`, `custom`) mapped to
  per-layer FP8/BF16 selection.
- Requires CUDA 12.8+, installs via
  `pip install --no-build-isolation git+https://github.com/Lightricks/LTX-Video-Q8-Kernels.git`.

Notably, q8_kernels does **not** yet target SM100 (Blackwell). LTX-2 inference
on B200 currently runs through the `ltxv2` preset's BF16 path with FP8
attention via FlashInfer — which is enough to be I/O-bound on the diffusion
transformer's MLP layers. A native SM100 q8_kernels port (using `tcgen05.mma
kind::mxf8f6f4` with NVFP4 weights) is the obvious next-step extension; until
then, FP4 LTX-Video on B200 routes through DeepGEMM / CUTLASS's
`72a_blackwell_nvfp4_bf16_gemm` template.

---

## K.11 PyTorch Helion — high-level kernel DSL

### K.11.1 What Helion is

Helion (`pytorch/helion`, beta release Oct 22, 2025) is a Python-embedded DSL
that compiles "PyTorch with tiles" into auto-tuned Triton. The motivating
gap: CUDA gives you maximum control but per-arch lock-in; pure Triton still
forces you to manage tile indexing and define autotune search spaces by hand;
`torch.compile` is portable but doesn't let you specify fusion strategies.
Helion sits one level above Triton:

```python
import torch, helion, helion.language as hl

@helion.kernel()
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out
```

The host code (outside the outermost `hl.tile`) is plain PyTorch; the device
code (inside) compiles to a single Triton kernel. The Helion compiler decides
**indexing** (`pointer`, `block_ptr`, `tensor_descriptor` — the latter being
TMA), **block sizes**, **flatten_loops**, **loop_orders / l2_grouping**,
**reduction_loops** (persistent vs looped), **pid_type** (`flat`, `xyz`,
`persistent_blocked`, `persistent_interleaved`), **load_eviction_policy**, and
the standard Triton `num_warps`, `num_stages`, `range_unroll_factors`,
`range_warp_specializes`, `range_num_stages`, `range_multi_buffers`,
`range_flattens`. A single Helion kernel definition maps to thousands of
Triton configurations — the implicit search space is much larger than what a
human would write out.

### K.11.2 Performance on B200

On NVIDIA B200, benchmarked vs. eager mode, `torch.compile` (max-autotune),
and hand-written Triton kernels (mostly Liger-Kernel suite):

- Helion **3.27× geomean speedup over eager**
- vs. `torch.compile` (max-autotune): **1.21× geomean**
- vs. hand-written Triton: **1.85× geomean**
- Specific outliers: softmax kernel **2.28× over torch.compile**, jsd kernel
  **6.22× over hand-Triton**

On AMD MI350X: 2.37× geomean over eager, 1.05× over torch.compile, 1.44× over
Triton. `int4_gemm` and `jsd` show **4.4–4.5× over hand-Triton on MI350X**.

Two case studies in the blog:

1. **RMSNorm backward** — Helion ≥ Quack (CuTe DSL hand-tuned), written in
   < a day; matches or exceeds across reduction-dim sweeps on H100.
2. **Mamba-2 chunk-scan** — Helion vs. TileLang and Triton on H100. Helion
   wins **2.12–2.63× over TileLang** and **1.20–1.85× over Triton**.

### K.11.3 Attention example

The reference attention example (`helionlang.com/examples/attention.html`)
implements scaled-dot-product attention in **30 lines** (vs. 120 in Triton,
thousands in CUDA). The structure: outer `hl.tile([batch, m_dim])` for
output tiles, inner `hl.tile(v_view.size(1))` for K/V tiles, FlashAttention-
style online-softmax with `m_i` running max and `l_i` running sum, `qk_scale =
sm_scale * 1.44269504` to use base-2 exponentials (`torch.exp2(qk)`), and
final normalization. Helion's autotuner picks `block_sizes`, `num_warps`,
`num_stages`, and indexing strategy automatically — including whether to
generate TMA tensor descriptors for B200.

### K.11.4 Compiler architecture

The pipeline:

1. **Python AST parsing** — kernel source → AST.
2. **Type propagation + metadata** — annotate AST with type info.
3. **Lowering to Device IR** — Helion's primary IR, a collection of FX
   graphs in SSA form, each containing pointers to Inductor IR nodes.
4. **Compiler passes** — semantic transforms like `reduction_loops` rolling.
5. **Codegen** — takes Device IR + autotuned config and generates Triton
   source.

The key architectural decision: config is applied only at the very end,
during codegen. Everything before — parse, IR transform — runs once per
autotune session, making thousands-of-configs exploration computationally cheap.

For DiT-style kernels Helion is the right *prototyping* tool: a custom epilogue
(e.g., a bias-then-GELU-then-dropout block), a fused `Q·K^T·scale·mask·softmax·V`
with a custom mask, or a fused per-block MoE router are all 30–80 lines of
Helion vs. 200–400 of Triton.

---

## K.12 Triton block-scaled matmul tutorial — the NVFP4 reference implementation

The Triton 3.5+ tutorial `python/tutorials/10-block-scaled-matmul.py` is the
upstream reference for **how to write an NVFP4 (and MXFP4 / MXFP6 / MXFP8) GEMM
in Triton** that targets `tcgen05.mma.kind::mxf4nvf4.block_scale`. It was
introduced in commit [`9181d1e`](https://github.com/triton-lang/triton/commit/9181d1e33c231f8db266414a65b50c9b711cb7af)
("[FRONTEND] Add support for block-scaled tensor descriptors and dot_scaled")
and refined in [`24445a1`](https://github.com/triton-lang/triton/commit/24445a1cfaf10457ae7bb3a9c27043b8aa278366)
("[NVIDIA] Optimize block-scaled matmul lowering for SM100").

The tutorial's structure:

1. **Allocate scale-factor tensors** with the interleaved layout that
   `tcgen05.mma.block_scale` expects (the same M128xK4 basic block as in
   CUTLASS's `Sm1xxBlockScaledConfig`):
   - For NVFP4: SF = E4M3, vector size = 16
   - For MXFP4: SF = E8M0, vector size = 32
   - For MXFP6 / MXFP8: SF = E8M0, vector size = 32
2. **Use `tl.make_tensor_descriptor`** to build TMA descriptors for A (FP4),
   B (FP4), SF_A (E4M3 with the special SF layout), SF_B.
3. **Inside the kernel, call `tl.dot_scaled(a, b, sf_a, sf_b, acc, …)`** —
   a Triton intrinsic that lowers directly to `tcgen05.mma.kind::mxf4nvf4.block_scale`
   with the right SFs broadcast to the right MMA atoms. The Triton compiler
   handles TMEM allocation and accumulator management automatically.
4. **Epilogue**: cast accumulator to BF16, store via TMA. (Or, for FP4-out,
   call `tl.dot_scaled` with output type `tl.float4x2` and produce both the
   FP4 D and SF_D tensors.)

The reason this tutorial is significant: it is the *only* reference
implementation in the open-source ecosystem that demonstrates block-scaled
narrow-precision MMA at the Triton level (everything else routes through
CUTLASS or a custom CUDA path). Once published, it became the template that
Helion's autotuner reuses (Helion's `indexing='tensor_descriptor'` setting
generates the same TMA + `tl.dot_scaled` lowering), the basis for FlashInfer's
NVFP4 GEMM kernel, and a starting point for downstream work like the FlashAttention
v4-style FP4 attention exploration.

The two commits also shipped corresponding intra-warp `tl.dot_scaled` semantics
for SM120 (RTX 50) which use `mma.sync.aligned.kind::mxf4nvf4.block_scale` (no
TMEM, smaller tiles). The same Triton source compiles to either path depending
on `triton.runtime.driver.active.get_current_target()`.

---

## K.N PyTorch allocator quirks — `expandable_segments` and the NCCL/VMM interop

### K.N.1 What `expandable_segments` does

Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` switches PyTorch's
caching allocator from `cudaMalloc`-backed fixed segments to **CUDA Virtual
Memory Management (VMM)**-backed expandable segments. Instead of allocating
a fixed-size block and risking fragmentation when growing later, the allocator
reserves a large virtual address space (`cuMemAddressReserve`), then
maps physical pages (`cuMemCreate` + `cuMemMap` + `cuMemSetAccess`) into the
reserved range as needed. This dramatically reduces fragmentation for
training workloads with growing/shrinking batch sizes (the VMM segments grow
and shrink in 20 MB granularity).

For training a video DiT on B200 — where activations can be tens of GB and
torch.compile's recompilation might briefly inflate residence — this is the
default-on setting recommended for any real workload, because the alternative
(non-expandable) often fragments to OOM in the second epoch even with plenty
of free memory.

### K.N.2 Why it broke NCCL registration — RFC 165419

NCCL's user-buffer registration (NVLS, General intra-node, Window registration)
is what enables zero-copy collectives and good NCCL/SymmMem composition. NCCL
registration requires a virtual address range backed by **a single
`CUmemGenericAllocationHandle`**. Expandable segments deliberately back a
single virtual range with **multiple** handles (one per growth epoch), which
NCCL cannot register.

The PyTorch RFC 165419 (Oct–Dec 2025) walks through the problem and the
attempted fixes:

- **Plan 1**: Generalize expandable segments to "import" external VMM allocations
  (e.g., from `ncclMemAlloc`) by retaining their `CUmemGenericAllocationHandle`
  via `cuMemRetainAllocationHandle`, then re-mapping under expandable-segment
  control. **Rejected** because NCCL's API doesn't actually accept multi-segment
  buffers — the doc was misleading. (NCCL team confirmed this is a hidden
  constraint they may someday lift.)
- **Plan 2**: Make NCCL `registerMemPool` work with expandable segments
  enabled, by ensuring segments use a single registerable VMM handle.
  **Rejected** for the same reason — the multi-handle constraint is structural.
- **Adopted (PR #169491)**: When a CUDA mem-pool is active *and* requires
  fixed-handle behavior (i.e., it's the NCCL pool), the allocator transparently
  bypasses expandable-segments mode for that pool's allocations. The user
  doesn't need to disable expandable segments globally; it's a per-pool decision.

The practical impact for video-DiT training on B200:

1. Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for general
   activations / KV cache.
2. Allocate NCCL-registered tensors (or, more commonly, SymmMem-backed
   tensors) inside `torch.cuda.use_mem_pool(symm_mem.get_mem_pool(device))`
   — those tensors land in a non-expandable pool that is registerable.
3. The auto-bypass logic in PR #169491 (PyTorch ≥ 2.12) handles the rest.

Two practical pitfalls remain for users on PyTorch ≤ 2.11:

- vLLM-style "rollout" workers (verl PR #5144) explicitly toggle
  `set_expandable_segments` off when entering the rollout phase to avoid the
  composition bug.
- Mixing custom CUDA Graphs with expandable segments + NCCL registration
  silently fails (IMA — Illegal Memory Access) on older NCCL builds; recent
  NCCL (≥ 2.30) at least exits cleanly with a diagnostic instead of corrupting
  memory.

### K.N.3 Composition with the rest of the stack

The big-picture view: every kernel infrastructure piece in this section
(CUTLASS DistGEMM, FLUX, COMET, Triton-distributed, PK, Mega MoE, SymmMem,
Kraken, FlashInfer custom AR) ultimately relies on either (a) NCCL registered
buffers, or (b) NVSHMEM symmetric memory, or (c) raw VMM-mapped peer pointers.
All three paths require *some* level of cooperation with `expandable_segments`
to coexist with high-fragmentation workloads. PyTorch 2.12's per-pool bypass
is the first version where the composition is actually clean — which is why
the report's training-stack recommendations consistently pin to PyTorch ≥ 2.12
and NCCL ≥ 2.30.

---

## Composition: how everything plugs together for video DiT on B200

A modern video DiT inference / training stack on B200 composes the pieces
above as follows:

1. **Per-SM compute**: `tcgen05.mma kind::mxf4nvf4.block_scale` (NVFP4) or
   `kind::mxf8f6f4` (FP8 / mixed) targeting 2-CTA UMMA at MmaTile `<256,256,256>`
   with cluster `<2,4,1>`.
   - Implemented via CUTLASS 4.x (e.g., ex.72a/b/c for dense, ex.84a for
     sparse, ex.92 for MoE), or DeepGEMM JIT, or the Triton 3.5+
     `tl.dot_scaled` path, or PK's Blackwell tile MMA wrappers.
2. **TMEM-resident chained MMA** for `softmax(QK^T)·V` in attention and
   `linear1·SwiGLU·linear2` in MoE — keeps the intermediate in TMEM and avoids
   ~12 TB/s of off-chip traffic per SM.
3. **CLC + persistent kernel** for compute-side load balancing under
   co-resident communication kernels — picked up automatically by every CUTLASS
   `KernelTmaWarpSpecialized*Sm100` kernel and (manually) by ThunderKittens'
   `kittens::prototype::lcf` template.
4. **Tile-level barrier release** between adjacent kernels in a chain via
   `griddepcontrol.launch_dependents` / `griddepcontrol.wait` — exposed in
   CUTLASS as `cutlass::arch::launch_dependent_grids()` and in DeepGEMM as
   `deep_gemm.set_pdl(True)`. Saves 5–15 µs per kernel boundary in chains
   like `qkv_proj → flash_attn → out_proj`.
5. **Inter-GPU rotation via Copy Engine** when transfers are large (256 MB+
   regime: weight fetches in FSDP, full-batch AG); via TMA when transfers are
   small (256 B–256 KB regime: per-tile AG/RS in DistGEMM, MoE A2A); via
   register-level + multimem when in-network reduction is needed (GEMM+AR with
   NVLink-SHARP). PK's analysis (Section K.8) gives the explicit selection rule.
6. **Symmetric memory** via PyTorch SymmMem 2.12 (`symm_mem.empty` +
   `symm_mem.rendezvous`) for the user-facing tensor; underlying transport is
   NVSHMEM 3.5+ (intra-node multimem; inter-node CX-7 RDMA + GPUDirect ASYNC).
7. **Distributed kernels at framework level** are picked from:
   - SP (Ulysses): Triton-distributed `sp_ulysess_qkv_gemm_all2all.py` /
     `sp_ulysess_o_all2all_gemm.py`, or PK's Ring-Attention kernel.
   - TP (Megatron-style AG-GEMM, GEMM-RS): CUTLASS DistGEMM ex.82, FLUX, or
     Triton-distributed `allgather_gemm.py` / `gemm_reduce_scatter.py`.
   - EP (MoE A2A + grouped GEMM): DeepEP V2 + DeepGEMM Mega MoE; or
     Triton-distributed `ep_a2a.py`; or COMET's fused dispatch+GEMM.
   - High-level kernel customization: Helion (30-line attention) or
     ThunderKittens / PK (Blackwell-native).
8. **Inference serving wrapper** via FlashInfer (fused MoE, MLA, paged-KV
   attention) and SGLang's `sgl-kernel/` (MLA prefill/decode, RadixAttention
   helpers).
9. **Special cases**:
   - LTX-2 video diffusion FP8 inference: `q8_kernels` patches (Lightricks),
     SM89/SM90 today, SM100 TBD.
   - Memory: `expandable_segments:True` with PyTorch 2.12's per-pool bypass
     for NCCL/SymmMem composition.

The **headline numbers** that this stack achieves on B200, which the rest of
the report cites and which were derived in this section, are:

- 96% of theoretical FP4/FP8 tensor core peak (Section K.1)
- 91/99/85/93% efficiency for Hopper DistGEMM (Section K.3)
- 1.96× per-MoE-layer / 1.71× end-to-end speedup from COMET (K.7)
- 2.33× DP/TP, 4.08× SP, 1.22× EP from PK (K.8)
- 1550 TFLOPS peak FP8 from DeepGEMM, 740 GB/s NVLink dispatch from DeepEP V2 (K.9)
- 3.27× geomean B200 speedup from Helion vs. eager (K.11)
- 1.54×/1.47×/1.20× Llama-3 8B/70B/105B speedup from PyTorch SymmMem (K.5)

These are the building blocks every other section of the report assumes are
available on Blackwell B200 in the May 2026 timeframe.

---

## Limitations / open issues — what is *not* yet shipped

Even with this much infrastructure, the May 2026 stack still has gaps that
affect video DiT directly:

1. **No Blackwell DistGEMM ex.82 published efficiency numbers** — extrapolation
   from Hopper is the best we have; an empirical table is missing from any
   public source.
2. **No FA4-ring fused kernel** — PK provides Ring Attention, but FlashAttention-4
   on Blackwell is published only as a single-GPU kernel. A SP-fused
   FA4-ring with 2-CTA UMMA tile MMA does not yet exist as published work
   (it's expected as PK ≥ 1.5 or as a SymmMem cookbook entry).
3. **NVFP4 block-scale async-TP not upstream** — neither CUTLASS DistGEMM ex.82
   nor FLUX exposes a `KernelTmaWarpSpecialized2SmNvf4Sm100` distributed
   variant yet. Implementations exist in private forks (DeepSeek's, ByteDance's)
   but are not in the open-source mainline.
4. **Mega MoE only supports FP8×FP4** — pure NVFP4×NVFP4 MoE (the natural fit
   for FP4 training) is the obvious next step but not in DeepGEMM v3.
5. **q8_kernels Blackwell port** — LTX-2 FP4 on B200 routes through
   CUTLASS / DeepGEMM today; a dedicated SM100 q8 path with the LTX-specific
   linear-attention fusion would unlock another 1.5–2× on the LTX-2 inference
   path.
6. **DeepEP V1 NVSHMEM-IBGDA** — DeepEP V1 used IBGDA for inter-node A2A and
   was the fastest at scale; V2 switched to NCCL Gin (which uses IBRC), losing
   some scale-out bandwidth. The "0-SM RDMA low-latency EP" of V1 is
   explicitly *not supported* in V2.
7. **`expandable_segments` + NCCL multi-segment buffers** — the per-pool
   bypass works but the architectural fix (NCCL handling multi-segment buffers
   natively) is on NCCL's roadmap, not yet in 2.30.
8. **Helion + multi-GPU** — Helion does not yet expose SymmMem / NVSHMEM
   primitives; multi-GPU kernels still drop to PK / Triton-distributed. A
   "Helion + symm_mem" extension is the natural next layer.

These gaps define the "state of the art" baseline for the rest of the report,
and constrain the achievable speedups for the video DiT workloads discussed
elsewhere.
