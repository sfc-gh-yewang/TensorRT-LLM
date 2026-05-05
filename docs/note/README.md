# Parallelism for Video DiT Inference on B200 (4 or 8 GPUs)
LTX-2 / Wan / Hunyuan-class video diffusion, NVFP4 GEMM + attention, single B200 node, comm-compute overlap available.
## Schemes & kernels
### TP (Tensor Parallelism, option B = Megatron-SP+TP)
Shards weights along output dim (column-parallel) or input dim (row-parallel). Activations stay seq-sharded between blocks.
- **Primary kernels**: CUTLASS DistGEMM example 82 (`/Users/yewang/work/cutlass/examples/82_blackwell_distributed_gemm`)
  - `AllGather1D_TilingCD_RotatingA` — for column-parallel layers (QKV, FFN-up): fuses input `AllGather` along seq with the local GEMM
  - `ReduceScatter1D_TilingA_RotatingC` — for row-parallel layers (O, FFN-down): fuses local GEMM with output `ReduceScatter` along seq
  - Build: `cmake .. -DCUTLASS_NVCC_ARCHS="100a" -DCUTLASS_ENABLE_GDC_FOR_SM100=1`
- **Alternative**: Triton-distributed `allgather_gemm.py`, `gemm_reduce_scatter.py` (~95% as fast on B200; preferred only if you can't pull in CUTLASS)
- **Why option B, not option A**: option A (Megatron-1D, full AllReduce after each row-parallel) moves `2·(P-1)/P · S` bytes per rank and has to hide behind a single GEMM. Option B splits that into AG (`(P-1)/P · S`, hides behind column-parallel GEMM) + RS (`(P-1)/P · S`, hides behind row-parallel GEMM). Same total bytes, but split across two GEMMs of different sizes — only the fraction that exceeds the *smaller* GEMM is exposed.
- **Assumed overlap behavior**: per-stage GEMM > per-stage comm → comm fully hidden. Per CUTLASS blog, DistGEMM achieves 91–99% of bare-GEMM throughput on Hopper for Llama 70B/405B fc1/fc2 shapes. On B200 with GDC + TMEM, expect **95–99%**. **Exposed comm** comes only from regions where stage GEMM < stage comm — for LTX-2 that's the **O projection** (K = hidden/TP = 1024 at TP=4, very small K) which exposes ~366 µs/layer at TP=4, seq=16k; FFN-down and column-parallel paths fully hide.
### Ulysses (Sequence Parallelism, attention via head-shard inside attn)
Activations seq-sharded between layers; A2A swaps to head-sharded inside attention so FA runs locally.
- **Primary kernels**: Triton-distributed (`/Users/yewang/work/Triton-distributed/python/triton_dist/kernels/nvidia/`)
  - `sp_ulysess_qkv_gemm_all2all.py::SpUlysessQKVGemmAll2AllKernel` — fuses QKV-projection GEMM with pre-attn A2A (heads ↔ seq swap). Output is `(q, k, v)` 4D tensors with full seq, sliced heads.
  - `sp_ulysess_o_all2all_gemm.py::SpUlysessOAll2AllGemmKernel` — fuses post-attn A2A with O-projection GEMM. Input is full-seq + sliced heads, output is seq-sharded + full hidden.
  - Lower-level alternatives: `ulysses_sp_dispatch.py::pre_attn_qkv_pack_a2a_op`, `low_latency_all_to_all.py` for the comm-only path (no GEMM fusion)
  - Layer wrapper: `python/triton_dist/layers/nvidia/ulysses_sp_a2a_layer.py::UlyssesSPAllToAllLayer`
- **Build**: `pip install triton_dist nvshmem4py-cu12 nvidia-nvshmem-cu12`
- **Setup cost**: NVSHMEM symmetric heap allocated once per `(max_bs, max_seq, q_nheads, head_dim)` via `create_ulysses_sp_pre_attn_comm_context(...)`. Reused across all denoising steps.
- **Assumed overlap behavior**: NVSHMEM puts run on the copy engine while persistent TMA GEMMs occupy SMs. A2A volume per rank is `(P-1)/P · 3 · (B · seq/SP · h/SP · d · elem_size)` — for SP=2 at LTX-2 16k: 12 MB/rank, ~80 µs at 150 GB/s effective NVLink5. **Hides completely** behind the QKV GEMM (which at `M=8k, K=4096, N=12288` takes ~340 µs at NVFP4) and behind the O GEMM (~140 µs vs ~30 µs A2A). At seq ≤ 4k both A2A and GEMM shrink, but the ratio is preserved → still hides.
### Ring (Context Parallelism, attention via seq-shard inside attn)
Activations seq-sharded throughout, including inside attention; K,V rotated around the ring during FA, partial outputs merged via online softmax.
- **Primary kernels**: Triton-distributed `sp_ag_attention_intra_node.py`. **No drop-in production kernel exists for video DiT** — you'd need to write a ring-FA wrapper around FA3 with cross-rank LSE accumulation.
- **Why this is more work than Ulysses**: stock FlashAttention can't be used directly. The kernel must accept `(O_running, lse_running)` and emit updated `(O_new, lse_new)` after each ring stage. Plus stream/event ordering for K,V rotation that must overlap with FA tile execution.
- **Assumed overlap behavior**: per-stage FA > per-stage K,V rotation → comm hides. Per-stage FA scales as `seq²/ring²`, per-stage comm scales as `seq/ring` — **the ratio degrades quadratically as seq shrinks**. At LTX-2 seq=16k, ring=2: FA per stage = 765 µs, comm per stage = 213 µs → hides. At seq=4k: FA per stage = 48 µs, comm per stage = 53 µs → exposed by 5 µs/stage. At seq=2k: badly comm-bound.
### SP+TP hybrid
2-D process grid: `(SP, TP)`. Activations sharded along SP; weights sharded along TP. Combines Ulysses A2A (within SP group) with TP collectives (within TP group).
- **Kernel mix**: CUTLASS DistGEMM (TP collectives) + Triton-distributed Ulysses (SP A2As). Each library covers what the other can't — CUTLASS has no all-to-all primitive, Triton-distributed's TP path is slightly slower than CUTLASS's on B200.
## Core mental model: contest over per-rank `M`
NVFP4 on B200 has a **higher tensor-core ridge point** than bf16 on H100 (more SMs, higher math throughput). Per-rank GEMM `M` needs ≥ 3–4k to stay compute-bound; below that, GEMMs underutilize and *no amount of comm overlap saves you*.
- **SP shrinks** per-rank `M` to `seq/SP` (good when seq is large)
- **TP keeps** per-rank `M = seq` (good when seq is small)
- **SP+TP** lets you tune `M = seq/SP` independent of total parallelism — the only knob that lets you stay above the ridge point at high P
## Design rule
Pick `(SP, TP)` so that:
1. `seq / SP ≥ 3–4k` — keep GEMMs compute-bound (the binding constraint at low res / high P)
2. `num_heads / TP ≥ 4` — keep FA efficient (LTX-2 has 32 heads → TP_max = 8)
Then minimize exposed comm.
## Recommendations — LTX-2
| Workload | Tokens | **4 GPUs** | **8 GPUs** | Why |
|---|---|---|---|---|
| 1080p 30s+ | 100k+ | Pure SP=4 | SP=8 or SP=4 × TP=2 | M huge under SP; SP A2A tiny vs FA; TP unnecessary |
| 1080p 5s | ~16k | Pure SP=4 | **SP=2 × TP=4** | `M = seq/SP = 4–8k` healthy; SP A2A hides behind QKV GEMM |
| 720p 5s | ~9k | Pure SP=4 or SP=2 × TP=2 | **SP=2 × TP=4** | Crossover region; both endpoints within a few % of hybrid |
| 480p 5s | ~4.5k | TP=4 or SP=2 × TP=2 | SP=2 × TP=4 or TP=8 | Pure SP would put `M = 1k`, below ridge → TP needed to keep `M = seq` |
| 480p 2s | ~1.8k | Pure TP=4 | TP=8 | Tiny seq; only TP keeps `M` reasonable; head/rank still ≥ 4 |
| ≤1s preview | <1k | DP across requests | DP across requests | Per-video work too small to parallelize within a request |
**Single-config defaults**: 4-GPU → `SP=2 × TP=2`. 8-GPU → `SP=2 × TP=4`.
**Why the defaults differ**: At 4 GPUs the SP×TP grid is small enough that pure SP=4 and pure TP=4 are within a few % of the hybrid — `SP=2 × TP=2` is a safe robust pick. At 8 GPUs, **pure SP=8 starves GEMMs** (`M = seq/8` below ridge at any seq ≤ 25k) and **pure TP=8 has compounding RS overhead** (post-attn RS exceeds O-GEMM by more), so the interior of the SP×TP space wins, not its boundaries.
## Skip
| Avoid | Why |
|---|---|
| Pure SP=8 (8-GPU) | `M = seq/8` below NVFP4 ridge at any reasonable seq |
| TP option A (full AllReduce) at TP ≥ 4 | AR moves 2× the bytes of AG+RS and has to hide behind one GEMM (the small O-projection); option B splits the same total comm across two GEMMs (one big, one small) so only the small fraction is exposed |
| Ring | Ulysses ties or wins everywhere for LTX-2's `(seq, head_count)`; ring's per-stage FA shrinks **quadratically** in seq while comm shrinks linearly, exposing comm at low seq. Custom ring-FA + cross-rank softmax merge is real implementation cost for zero benefit on a single node |
| Pure TP at 1080p 5s+ | Above seq ≈ 12k, post-attn RS volume exceeds O-GEMM time → exposed by ~366 µs/layer; SP doesn't have this asymmetry |
| Pure SP=4 at 480p ≤3s (4-GPU) | `M = seq/4 ≤ 1.1k`, deep underutilization; TP-heavy wins by 20–30% |
## Dataflow (`SP × TP` hybrid; one transformer block)
```
input  [B, seq/SP, hidden]
  → CUTLASS:     AG_seq + GEMM_QKV   (column-parallel TP)            [AllGather1D_TilingCD_RotatingA]
  → Triton-dist: SpUlysess pre-A2A   (within SP group)                [SpUlysessQKVGemmAll2AllKernel*]
  → FA NVFP4                          (local, h/(SP·TP) heads)
  → Triton-dist: SpUlysess post-A2A  (within SP group)                [SpUlysessOAll2AllGemmKernel*]
  → CUTLASS:     GEMM_O + RS_seq     (row-parallel TP)                [ReduceScatter1D_TilingA_RotatingC]
  → CUTLASS:     AG_seq + FFN_up     (column-parallel TP)             [AllGather1D_TilingCD_RotatingA]
  → GELU + NVFP4 quant                                                (Triton epilogue)
  → CUTLASS:     FFN_down + RS_seq   (row-parallel TP)                [ReduceScatter1D_TilingA_RotatingC]
output [B, seq/SP, hidden]
```
\* In hybrid mode, the QKV/O GEMM is performed by CUTLASS DistGEMM (TP-side), and only the SP A2A is done by the Triton-distributed kernel. The kernels named `SpUlysessQKV/OAll2AllGemmKernel` are full GEMM+A2A fusions intended for pure SP — in hybrid mode use the comm-only paths (`pre_attn_qkv_pack_a2a_op`, `post_attn_a2a`).
Pure-SP: drop CUTLASS rows; use the GEMM+A2A fused versions of the Triton-dist kernels.
Pure-TP: drop SP A2A rows; FA runs on `h/TP` heads with full local seq.
## Assumptions
The recommendations above hold under these explicit assumptions. If any breaks, the trade-offs shift:
| Assumption | Where it bites if wrong |
|---|---|
| **CUTLASS DistGEMM hides comm at 95–99% efficiency on B200** with GDC enabled | If you build without `CUTLASS_ENABLE_GDC_FOR_SM100`, expect 85–90% — TP starts losing ~5% more wall time vs the numbers in tables |
| **NVFP4 GEMM achieves ~6 PF/s sustained** at `M ≥ 4k` | If sustained drops below ~3 PF/s (e.g., due to spills or bad tile shapes), the SP small-M penalty shifts and so does the SP↔TP crossover |
| **Triton-distributed Ulysses A2A hides behind QKV GEMM** | True at any seq ≥ 1k for LTX-2 (A2A volume linear in seq, GEMM also linear). Only fails if QKV GEMM tile schedule is broken |
| **NVSHMEM intra-NVL5 effective bandwidth ~150–300 GB/s per pair** | If real bandwidth is lower (e.g., contended NVL5 with other workloads), comm exposure goes up linearly |
| **Comm-compute overlap is via GDC (CUTLASS) and NVSHMEM stream pipelining (Triton-dist)**, not a single fused kernel | Both libraries achieve ~99% of bare-GEMM in their target regime; this assumption fails only if the underlying scheduling is contended (e.g., NCCL running concurrently on the same streams) |
| **CUDA Graphs capture the full denoising step** | If shapes are dynamic per step (they shouldn't be, but VSA can change), launch overhead resurfaces and 1–3 of the priority list become weaker |
| **Elementwise / dequant ops are fused into adjacent GEMM kernels' epilogues/prologues** | If kept as separate kernels, scaling is capped at ~1.3–1.5× regardless of parallelism choice |
| **FA3 supports NVFP4 attention with hardware-accelerated path** | If forced to FP8 attention, attention compute roughly doubles; SP/TP trade-offs shift slightly toward TP (because attention compute fraction grows) |
| **Per-rank weights fit in 192 GB** | Always true for LTX-2 at NVFP4 (~6.5 GB total); only fails for >100B-class DiTs which would force TP for memory regardless of perf |
## Priority (highest leverage first)
1. **CUDA graphs** around the denoising step — *Why*: NVFP4 shrinks per-op compute ~4× but launch overhead is fixed, so launch overhead becomes 15–30% of wall time. Diffusion has 1000+ launches per step with static shapes → perfect graph target. Single biggest leverage.
2. **Elementwise / dequant fusion** (ModulatedRMSNorm, RoPE-as-epilogue, gate-mul-as-epilogue, GELU-into-FFN-up) — *Why*: ~30–50 small ops/layer, each launch-bound. Don't shrink with SP. Without this, scaling caps at ~1.3× regardless of parallelism.
3. **Native NVFP4 GEMM** with per-tile scaling — *Why*: bf16-dequant-then-GEMM kernels add an HBM round-trip per Linear and don't shrink with SP. Native FP4 path eliminates them entirely.
4. **Comm-compute overlap kernels** — CUTLASS DistGEMM ex. 82 (`CUTLASS_ENABLE_GDC_FOR_SM100`) for TP collectives + Triton-dist `SpUlysess*` kernels for SP A2As. *Why*: ~5–10% e2e on top of 1–3, takes scaling from ~3.0× to ~3.5×+.
5. *(Optional)* **Adaptive parallelism per request shape** — *Why*: SP↔TP crossover at seq ≈ 12k is sharp; one fixed config across mixed resolutions leaves 10–20% on the table.
Without 1–3, scaling caps around 1.3–1.5×. With 1–3:
- 4-GPU `SP=2 × TP=2`: ~3.5–3.7× (~88–93% efficiency)
- 8-GPU `SP=2 × TP=4`: ~6.5–7.5× (~80–95% efficiency)
## Why long-seq prefers SP, short-seq prefers TP
| | Long seq (≥ 12k) | Short seq (≤ 6k) |
|---|---|---|
| **SP per-rank M** | `seq/SP` stays large → GEMMs compute-bound | shrinks below ridge → underutilize |
| **SP A2A volume** | linear in seq, hides behind large QKV/O GEMMs | also linear, hides behind small ones too |
| **TP per-rank M** | full seq → GEMMs compute-bound | full seq → still healthy |
| **TP post-attn RS** | volume = `seq · hidden`, large → exposes vs small O-GEMM | small enough to fit behind O-GEMM |
| **Verdict** | SP wins (TP's RS is the unhideable cost) | TP wins (SP's small-M penalty is dominant) |
The structural reason TP loses at long seq: post-attn AllReduce (option A) or ReduceScatter (option B) operates on the **full hidden-dim output tensor**, while SP's all-to-all operates on the **head-sliced tensor** (which is `1/P` smaller). That `P×` factor in comm volume is why TP can't hide its post-attn comm at long seq while Ulysses can.
## TL;DR
- 4-GPU default: `SP=2 × TP=2`. 8-GPU default: `SP=2 × TP=4`.
- Long seq → more SP. Short seq → more TP. Crossover at seq ≈ 12k.
- TP always option B (CUTLASS DistGEMM ex. 82 with GDC), never option A (AR).
- Skip Ring on single-node video DiT.
- **Fix elementwise + CUDA graphs first** or none of this matters (caps at 1.3×).
