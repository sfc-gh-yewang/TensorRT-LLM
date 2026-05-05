## §3. Attention Optimization

*This section is the largest single technical pillar of the report. Attention is the *other* long pole alongside QKV/O GEMM and FFN. We organize it into two sub-pillars:*

- **§3-A — Dense attention kernels** (FA3/FA4 family + SageAttention 1/2/2++/3 family + MagiAttention/cuDNN/FlexAttention/DistFlashAttn/BurstEngine/Striped/Tree) under prefix `D.x` (within §3-A).
- **§3-B — Sparse and structured attention** (STA/SSTA, VSA, Radial, SVG/SVG2, SLA/SageSLA, BSA, NSA, SpargeAttn, VMoBA, MInference, XAttention, ΔAttention, DiTFastAttn, DraftAttention, ToMeSD/VidToMe, Astraea, db-SP/DSA) under prefix `S.x` (within §3-B).

*The two sub-pillars compose multiplicatively at the kernel level (Sage 2++ + SpargeAttn) and at the SP level (db-SP and DSA explicitly target the ring + dynamic-sparse open problem; cross-ref §1 P.4).*

---

### §3-A. Dense Attention Kernels (FA3 / FA4 / FA-FP4 / SageAttention / MagiAttention / cuDNN / FlexAttention)

This sub-section is a deep-dive expansion of the dense-attention column. The intent is that a senior CUDA / kernel engineer can read this section alone and understand exactly *which instructions* run, *which numerics* are applied, and *which utilization fraction* is left on the table for each cited attention kernel relevant to a Wan / Hunyuan / LTX-2 / Mochi / CogVideoX / Step-Video pipeline running on NVIDIA B200 (compute capability sm_100a, "Blackwell").

The dense-kernel space splits cleanly into five buckets:

1. **Hopper-era foundations** that B200 inherits as a fallback path (FlashAttention-2 / 3, cuDNN 9.13 SDPA).
2. **Blackwell-native dense kernels** (FlashAttention-4 + CuTe-DSL, cuDNN 9.21 SDPA, the FA-FP4 fork at hao-ai-lab, MagiAttention's `FFA_FA4`).
3. **Microscaling / quantized attention** (the SageAttention 1/2/2++/3 family and SageBwd 8-bit training).
4. **Distributed / cross-rank dense attention** for context-parallel video-DiT training and inference (DistFlashAttn/LightSeq, BurstAttention, BurstEngine, Striped Attention, Tree Attention).
5. **Programmable / score-mod dense kernels** (PyTorch FlexAttention, the cuDNN frontend `Attention v1.22`).

Throughout, `[VERIFIED]` flags numbers I personally re-confirmed against a primary source (paper, GitHub README, release notes, GitHub issue thread); `[CORRECTED]` flags corrections to numbers in the existing report; `[INFERRED]` flags an extrapolation. References are cited inline using the parent report's link style.

---

## D.1 FlashAttention-2 (background only — A100 / Ampere baseline)

**References.** Tri Dao, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (Jul 2023, ICLR'24); [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) GitHub (23.6k stars on the fork's main branch as of May 2026 [VERIFIED]).

FlashAttention-2 (FA2) is the kernel everything else in this section is *measured against* (every TFLOPs/s number in the report's tables uses FA2 as the 1× baseline on consumer GPUs that lack FA3/FA4, and as the *secondary* baseline on H100/B200). Three things matter for the rest of the section:

- **Algorithm.** Same online-softmax tile recurrence as FA1, but with three structural changes: (i) `non-matmul` FLOPs are minimized so that softmax rescales happen *outside* the inner GEMM loop, (ii) parallelism is added along the sequence-length axis (not just batch×heads — important when the per-rank `seq` shrinks under context parallelism, see §1 and §D.10–D.12 below), and (iii) within each thread block the work is split across warps so warp-to-warp shared-memory traffic is reduced. The end-to-end consequence is **2× faster than FA1, 50–73% of A100 peak (225 TFLOPs/s on A100 BF16, head-dim 128 [VERIFIED])**, vs FA1's 30–50% peak. The forward pass exits at $\sim$73% peak; the backward pass exits at $\sim$63% peak [VERIFIED, Section 4 of arXiv:2307.08691].
- **Why FA2 still matters on Blackwell.** FA2 is the *only* FlashAttention version supported on RTX 5090 (sm_120) consumer Blackwell as of May 2026 — FA3 was Hopper-only (`wgmma`-bound), and FA4 ships only on data-center Blackwell (sm_100a). So every per-kernel "Sage-3 5× over the fastest FlashAttention on RTX 5090" claim from [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) is a comparison against **FA2**, not FA3 [VERIFIED in Sage 3 paper Section 5: *"FlashAttention3 can only run on Hopper GPUs, so FlashAttention2 is already the fastest version for RTX5090 and RTX4090"*].
- **Where FA2 sits in the report's stack.** In the SGLang Diffusion compatibility matrix and in xDiT's USP path, FA2 is the *fallback* kernel when FA3/FA4 wheels are missing, and it is the kernel `ring_flash_attn`, `zhuzilin/ring-flash-attention`, and `feifeibear/long-context-attention` build on top of for ring-based context parallelism. The cross-rank `(O, lse)` merge interface used by ring attention (see §D.11 below) was originally designed against FA2's varlen API and was later extended to FA3/FA4.

That is the entire FA2 background needed for the rest of this section. The interesting kernel work post-FA2 is on Hopper (FA3) and on Blackwell (FA4 / FA-FP4 / MagiAttention / Sage 3) — covered in detail below.

---

## D.2 FlashAttention-3 — the Hopper baseline (sm_90a only)

**References.** Shah et al., [arXiv:2407.08608](https://arxiv.org/abs/2407.08608); [PyTorch blog](https://pytorch.org/blog/flashattention-3/); NeurIPS 2024.

### D.2.1 Motivation: why a Hopper-specific rewrite was necessary

H100 changed three things relative to A100 that FA2 did **not** exploit:

1. **WGMMA (warpgroup MMA).** Where Ampere's `mma.sync` is per-warp synchronous, Hopper introduces `wgmma.mma_async` — a *warpgroup-level* (4 warps = 128 threads) MMA that is **asynchronous**. The instruction is issued by one thread of the warpgroup, runs in the background on the Tensor Core hardware, and completes via an `mbarrier`. Without `wgmma`, FA2 on H100 only sees ~⅔ of peak Tensor-Core throughput because `mma.sync` cannot saturate Hopper's tensor cores (microbenchmark in [arXiv:2402.13499](https://arxiv.org/abs/2402.13499) cited by the FA3 PyTorch blog [VERIFIED]).
2. **TMA (Tensor Memory Accelerator).** Hopper introduces a dedicated DMA engine that lets a single thread issue a multidimensional load from HBM into SMEM, with all index calculation, padding, and out-of-bound predication done in hardware. This frees registers and removes thread-level address arithmetic from the critical path.
3. **FP8 Tensor Cores.** WGMMA on E4M3/E5M2 doubles the throughput vs FP16 (1978 vs 989 TFLOPs/s on H100 [VERIFIED]). FA3 is the first FlashAttention to expose FP8.

The FA2 algorithm assumes one synchronous MMA per tile and one in-register softmax pass; on Hopper that schedule leaves WGMMA+TMA idle waiting for softmax, the SFU `MUFU.EX2` (the special-function unit handling `exp`) idle waiting for WGMMA, and the FP8 path entirely unused. Hence FA3's three contributions: producer/consumer warp specialization, GEMM-softmax overlap, and an FP8 path with incoherent-processing-style outlier suppression.

### D.2.2 Producer/consumer warp specialization

A 384-thread CTA is split into 1 *producer* warp (32 threads, of which only 1 actually issues TMA — the others are kept idle to free registers) and 3 *consumer* warpgroups (128 threads each). The producer issues `cp.async.bulk.tensor` (TMA loads) to bring the next K,V tile from HBM into SMEM; the consumers issue WGMMA on the previous tile and run softmax on the tile before that. Synchronization is via shared-memory `mbarrier` + the Hopper `mbarrier.arrive` / `mbarrier.expect_tx.shared::cta` PTX pair (the `expect_tx` form tells the barrier to wait for a transactional byte count, which TMA naturally produces on completion). Crucially, Hopper also has the `setmaxnreg.dec.sync.aligned.u32` / `setmaxnreg.inc.sync.aligned.u32` PTX intrinsics — the producer drops to ~24 registers (it doesn't need many — TMA does the work), and the consumers reclaim those into ~232 registers each. This shifts the register budget toward where it matters [VERIFIED — FA3 paper §3 and PyTorch blog].

```
   producer warp (32 thr, 24 regs)         consumer warpgroup A (128 thr, 232 regs)
   ─────────────────────────────────       ──────────────────────────────────────────
   for tile j in [0, T):                    for tile j in [0, T):
       TMA(K_j, V_j) → SMEM[j%2]                wait(SMEM[j%2] ready)
       arrive(mbar[j])                          WGMMA: S = Q · K_j^T   (async, accum in reg)
                                                wait(WGMMA done)
                                                online_softmax(S, m, ℓ)  ← uses MUFU.EX2
                                                WGMMA: O += P · V_j     (async)
                                                arrive(mbar_done[j])
```

### D.2.3 GEMM–softmax overlap (the "ping-pong" schedule)

FA3 observes that on H100 the Tensor Core throughput (989 TFLOPs/s FP16) is **~256× higher** than the SFU throughput (3.9 TFLOPs/s for `MUFU.EX2`, derived as `16 ops/SM/clock × 132 SMs × 1830 MHz`, [VERIFIED in PyTorch blog footnote]). For head_dim 128 there are 512× more matmul FLOPs than `exp` calls per tile, so naïvely `exp` consumes ~50% of the wall time *between* WGMMA dispatches. FA3 fixes this in two ways:

- **Inter-warpgroup ping-pong.** With two consumer warpgroups (call them WG1, WG2), the warps are explicitly synchronized via `bar.sync` so that while WG1 is doing softmax (CUDA cores + MUFU), WG2 is doing WGMMA. The diagram from the PyTorch blog: `[WG1.GEMM0, WG1.GEMM1] → [WG1.softmax + WG2.GEMM0, WG2.GEMM1] → ...`. Reported lift: forward pass goes from **~570 → 620 TFLOPs/s** at hd=128, seq=8K [VERIFIED].
- **Intra-warpgroup pipelining.** Even within a single warpgroup, FA3 carries enough register state for *two* in-flight softmax + one in-flight WGMMA, so that softmax of tile $j$ runs in parallel with WGMMA of tile $j+1$. Reported lift: **620 → 640–660 TFLOPs/s** at hd=128. This costs registers — the kernel goes from $\sim$170 to $\sim$232 registers per consumer thread, which is at the limit of `setmaxnreg.inc` [VERIFIED in FA3 paper §3.2].

The full forward kernel in FA3 reaches **740 TFLOPs/s = 75% of the H100 BF16 peak**, vs FA2's 35% on the same GPU — a 2.0× lift.

### D.2.4 FP8 with incoherent processing (block-quantized rotation)

The FP8 path doubles tensor-core throughput but introduces quantization error. The FA3 paper adopts **incoherent processing** from QuIP / QuaRot ([arXiv:2307.13304](https://arxiv.org/abs/2307.13304)): multiply $Q$ and $K$ by an inverse Hadamard rotation $H$ before quantizing. Because Hadamard is orthogonal, $Q H \cdot (K H)^\top = Q \cdot K^\top$ — the rotation cancels in the score matrix — but the rotated tensors have far smaller channel-wise outliers, so per-block FP8 quantization scales no longer waste range on a handful of large entries. The Hadamard transform itself takes $O(d \log d)$ per attention head and is fused with the previous op (typically RoPE) — both are SMEM-bandwidth bound, so the cost is hidden.

Numerically, on Q,K,V drawn from $\mathcal{N}(0, 1)$ but with 0.1% of entries scaled to a large outlier value, FA3 reports **2.6× lower mean quantization error** than vanilla per-tensor FP8 attention [VERIFIED, FA3 paper Figure 6 + PyTorch blog table].

### D.2.5 Reported numbers

| GPU | dtype | Setting | TFLOPs/s | % of peak | Source |
|---|---|---|---|---|---|
| H100 SXM5 | BF16 | hd=128, seq=32k | **740** | 75% | [VERIFIED, FA3 paper] |
| H100 SXM5 | FP8 (E4M3) | hd=128, seq=32k | **~1200** | 61% | [VERIFIED, FA3 paper] |

For video DiT, the FP8 path is the relevant one when the QKV/O linear is already FP8 (modelopt, DeepGEMM-FP8, Sage 2++ residue) — FA3 then consumes the FP8 outputs without a BF16 round-trip. **FA3 does not run on Blackwell** because `wgmma` is sm_90 only and is not forward-compatible to sm_100 — Blackwell exposes `tcgen05.mma` instead. So on B200, FA3 is replaced by FA4 / cuDNN 9.21 / MagiAttention's FFA_FA4 / Sage 3 (when SM100 path is restored).

### D.2.6 Composition with the rest of the stack

- **Ulysses (head-shard SP, §1.1 of parent report).** FA3 is the unmodified inner kernel. The 4 all-to-alls happen before/after the FA3 call; the FA3 inner loop is identical to single-rank.
- **Ring (seq-shard SP, §1.2).** Ring needs `(O_partial, lse_partial)` per K,V block to merge; FA3's varlen API exposes exactly that, which is why `zhuzilin/ring-flash-attention` is the de-facto reference Ring + FA implementation on Hopper. On Blackwell the same interface is needed from FA4 — see §D.10.
- **Step-Video / Hunyuan / Wan 2.1 / Wan 2.2 on H100/H200** in production. The Wan 2.2 README's footnote — *"FlashAttention3 deployed on Hopper architecture GPUs"* — is **the production attention path on Hopper** for both A14B (high-noise + low-noise expert) and TI2V-5B [VERIFIED VERBATIM, [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) README]. On B200 this is replaced by FA4 in the same generate.py command.

---

## D.3 FlashAttention-4 — the Blackwell rewrite (sm_100a)

**References.** Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao, [arXiv:2603.05451](https://arxiv.org/html/2603.05451v1) (Mar 5, 2026); [Together AI blog](https://www.together.ai/blog/flashattention-4); [Tri Dao FA4 blog](https://tridao.me/blog/2026/flash4/); [Lambda blog](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet); code at [Dao-AILab/flash-attention/tree/main/flash_attn/cute](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute); install via `pip install flash-attn-4` [VERIFIED].

### D.3.1 Motivation: asymmetric scaling on Blackwell

The FA4 paper makes the asymmetric-scaling argument explicit and quantitative. The relevant per-SM peak rates are listed in Section 2.2 of the paper:

| Resource | H100 (sm_90a) | B200 (sm_100a) | Ratio |
|---|---|---|---|
| BF16 MMA | 4096 ops/clock | **8192 ops/clock** | **2.0×** |
| BF16 peak | 989 TFLOPs/s | **2.25 PFLOPs/s** | 2.27× |
| `MUFU.EX2` (SFU `exp`) | 16 ops/clock | **16 ops/clock** | **1.0×** |
| SMEM read bandwidth | 128 B/clock | **128 B/clock** | **1.0×** |
| TMEM | — (regs only) | **256 KB/SM** | new |

*[VERIFIED — FA4 paper Section 2.2 + microbenchmark at [arXiv:2501.12084](https://arxiv.org/abs/2501.12084) for SMEM bandwidth.]*

So tensor cores got 2× faster while the two non-matmul resources that FA3 critically depends on (SFU and SMEM) stayed flat. The roofline analysis in §3.1 of the FA4 paper, for the canonical $M=N=d=128$ tile, gives:

$$
T_{\text{MMA}} = \frac{4MNd}{8192} = 1024 \text{ cycles},\qquad
T_{\text{SMEM}} = \frac{3MNd}{8192} = 768 \text{ cycles},\qquad
T_{\text{exp}} = \frac{MN}{16} = 1024 \text{ cycles}.
$$

The forward pass on B200 is **co-bottlenecked by tensor cores and the exponential unit** — they take the same number of cycles — and the SMEM bandwidth is now slack (768 < 1024). For the larger tile $M=256, N=d=128$ used in 2-CTA mode, the cycle counts are 2048/1536/2048 — same conclusion. So FA4's job is to **(i) hide softmax `exp` behind MMA**, and **(ii) reduce the *number* of `exp` evaluations**.

For the backward pass with five MMAs the picture inverts: $T_{\text{MMA}}=2560$, $T_{\text{SMEM}}=3328$, $T_{\text{exp}}=1024$. SMEM dominates by ~30%, so backward FA4 has to *reduce SMEM traffic*, which is what 2-CTA MMA + DSMEM does (§D.3.5).

Beyond this asymmetric-scaling driver, three new architectural primitives are *only* on Blackwell:

- **TMEM (Tensor Memory).** 256 KB on-chip per SM, allocated in 32-column (16 KB) granules, structured as 512 columns × 128 lanes of FP32 cells. The MMA writes its accumulator *directly* into TMEM, freeing the entire register file. On Hopper, accumulators sat in registers, which capped the largest WGMMA atom at 64×128×16 BF16 (and forced FA3 to cap the accumulator footprint and use the full register file). On Blackwell the largest single-CTA `tcgen05.mma` atom is **128×256×16** BF16, ~2× larger.
- **`tcgen05.mma` (single-thread asynchronous MMA).** A single thread issues the MMA; the hardware completes it asynchronously and writes to TMEM. Crucially, `tcgen05.mma` can also source operand A from TMEM (not just SMEM), which lets back-to-back $S$ → $P$ → $O$ chained matmuls keep all intermediates on-chip without any SMEM round-trip. This is the *enabling primitive* for back-to-back FP4 attention (Sage 3 and FA-FP4 both rely on it — see §D.4 and §D.8).
- **2-CTA MMA.** A pair of CTAs in the same threadblock cluster cooperatively executes one MMA, with the M axis split between the two CTAs. One CTA is the leader (issues the instruction); the peer must remain active. The cluster's combined SMEM is consumed for operand B, so each CTA only stages **half of B in its own SMEM** — operand B SMEM bandwidth roughly halves. The largest 2-CTA MMA atom is 256×256×16 BF16. The CTA-group size (1 or 2) must be constant for the duration of a kernel for both TMEM and tensor-core operations.

### D.3.2 The four FA4 contributions, one by one

#### (1) Redesigned forward pipeline (ping-pong + decoupled correction warpgroup)

FA4 forward uses **4 warpgroups per CTA** instead of FA3's 3+1:

```
warpgroup 1 (128 thr): softmax for Q^H tile  ──┐
warpgroup 2 (128 thr): softmax for Q^L tile  ──┼─→ ping-pong via bar.sync
warpgroup 3 (128 thr): correction (rescale O when m shifts)
warpgroup 4 (128 thr): MMA driver + TMA
```

Two query tiles `Q^H` (high) and `Q^L` (low) of 128 rows each are kept in flight; one warpgroup does softmax on the high tile while the other does softmax on the low tile, **explicitly desynchronized so they don't both hit `MUFU.EX2` at once**. Each 128-thread softmax warpgroup uses *one thread per row* of S — hardware-imposed by `tcgen05`'s TMEM read partitioning — so the row-max reduction has no inter-warp shuffles. Each thread:

1. Reads its 128-element row of S from TMEM into registers.
2. Computes `rowmax` and `rowsum`.
3. Decides per-warp (avoiding divergence) whether to dispatch `MUFU.EX2` or the FMA-emulated `2^x` (next subsection).
4. Computes $P = e^{S - m}$.
5. Stores $P$ back to TMEM **in three quarter-tiles**; the final quarter is stored separately. This staged store is necessary because holding 128 elements of $S$ + 64 elements of $P$ (BF16) per thread blows past the 256-register budget — even a one-column delay relieves register pressure.
6. As soon as the first 3/4 of $P$ is in TMEM, kicks off the `PV` MMA (which is `tcgen05.mma` with operand A from TMEM).

The *correction warpgroup* is the qualitative break from FA3. On Hopper, when the running row-max $m_{j-1}$ shifted up to $m_j$, the rescaling $O_{j-1} \leftarrow e^{m_{j-1}-m_j} O_{j-1}$ blocked on register dependencies of the same warpgroup that was running the next softmax — i.e., it was on the critical path. On Blackwell, $O$ lives in TMEM, so a separate correction warpgroup can read it, rescale it, and write it back without contending for the softmax warpgroup's registers. **Result: rescale is off the critical path.** Per the paper this is one of the largest single contributions to forward speedup.

ASCII view of the forward pipeline (one CTA, one outer iteration):

```
   timestep →  │ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │
   wg1 sm-H    │ S  │ ex │ st │    │ S  │ ex │       <- two-tile ping-pong
   wg2 sm-L    │    │ S  │ ex │ st │    │ S  │
   wg3 corr.   │    │    │ R  │ R  │ R  │ R  │       <- decoupled, runs continuously
   wg4 mma     │ Q·K│P·V │ Q·K│P·V │ Q·K│P·V │       <- tcgen05.mma async
```

The TMEM layout — 256 KB per SM, allocated in 32-column granules — supports this: at hd=128, FA4 holds two output tiles ($O^H$, $O^L$) plus two $S$ tiles overlapping with $P$ tiles. The paper considers "1×S + 2×P" vs "2×S overlap with P" and picks the latter because it lets the kernel start computing two $S$ tiles immediately at iteration 0 without dependency on the first $P$.

#### (2) FMA-emulated $2^x$ (Cody-Waite + Sollya polynomial)

This is the *headline* FA4 numerical trick. Because `MUFU.EX2` is the bottleneck (§D.3.1), FA4 implements $2^x$ as a polynomial on the otherwise-idle **floating-point FMA units** (8192 ops/clock/SM, 512× the SFU). The decomposition is the classical [Cody-Waite range reduction](https://en.wikipedia.org/wiki/Cody%E2%80%93Waite_range_reduction):

$$
2^x = 2^{\lfloor x \rfloor} \cdot 2^{\{x\}},\qquad \{x\} = x - \lfloor x \rfloor \in [0, 1).
$$

The integer part $2^{\lfloor x \rfloor}$ is a single shift-and-add into the IEEE-754 exponent field. The fractional part is approximated by a degree-3 polynomial in Horner form:

$$
2^{\{x\}} \approx p_0 + p_1 \{x\} + p_2 \{x\}^2 + p_3 \{x\}^3,\qquad
p_0 = 1.0,\; p_1 \approx 0.6951,\; p_2 \approx 0.2276,\; p_3 \approx 0.0771,
$$

with coefficients chosen via the [Sollya](https://www.sollya.org/) numerical library to minimize relative error over $[0, 1)$. Horner's method evaluates this in 3 fused multiply-adds.

Pseudocode (one row, per thread):

```cpp
// Cody-Waite range reduction for 2^x, x in fp32, output rounds to BF16.
float fma_exp2(float x) {
    x = max(x, -127.0f);            // Clamp to avoid underflow
    // Round-down via the magic-constant trick: forces fractional bits into mantissa.
    float magic = 0x1.8p23f;        // 2^23 + 2^22 = 12582912
    float rd = (x + magic) - magic; // = floor(x), round-down
    int n = (int) rd;
    float f = x - rd;               // fractional part in [0, 1)
    // Horner: 2^f ~= p0 + f*(p1 + f*(p2 + f*p3))
    float p = fmaf(0.0771f, f, 0.2276f);
    p = fmaf(p, f, 0.6951f);
    p = fmaf(p, f, 1.0f);
    // Combine: shift integer n into the exponent field; add p's mantissa bits.
    int bits = (n + 127) << 23;
    return __int_as_float(bits) * p;
}
```

The key numerical claim from the paper, reproduced verbatim from Table 2:

| Method | Max FP32 rel err | Mean FP32 rel err | Max BF16 rel err | Mean BF16 rel err |
|---|---|---|---|---|
| Ideal (FP64 → BF16) | — | — | $3.89 \times 10^{-3}$ | $1.41 \times 10^{-3}$ |
| Hardware MUFU.EX2 | $1.41 \times 10^{-7}$ | $3.04 \times 10^{-8}$ | $3.89 \times 10^{-3}$ | $1.41 \times 10^{-3}$ |
| **Degree 3 (FA4)** | $\mathbf{8.77 \times 10^{-5}}$ | $5.43 \times 10^{-5}$ | $\mathbf{3.90 \times 10^{-3}}$ | $1.41 \times 10^{-3}$ |
| Degree 4 | $3.05 \times 10^{-6}$ | $1.84 \times 10^{-6}$ | $3.89 \times 10^{-3}$ | $1.41 \times 10^{-3}$ |
| Degree 5 | $1.44 \times 10^{-7}$ | $5.48 \times 10^{-8}$ | $3.89 \times 10^{-3}$ | $1.41 \times 10^{-3}$ |

*[VERIFIED VERBATIM — FA4 paper Table 2.]*

The qualitative read: **at FP32, polynomial degree-3 is ~600× worse than hardware**. After rounding the FP32 polynomial output to BF16 (which is what attention does — $P$ is consumed by the next MMA in BF16), the BF16 quantization error of $\sim 3.9 \times 10^{-3}$ *dominates*, and the polynomial output is indistinguishable from `MUFU.EX2`. So degree-3 is "free" numerically. Degree 5 closes the FP32 gap at the cost of two extra FMAs per evaluation; FA4 picks degree 3.

**Partial emulation.** Naïvely emulating *every* `exp` doubles register pressure (more constants and intermediates), and that pressure spills past the `setmaxnreg.inc` limit. So FA4 splits each row's `exp` evaluations: a **tunable 10–25%** go through FMA-emulation, the rest through `MUFU.EX2`. The split is decided at warp granularity (so all 32 lanes take the same path, no warp divergence), and the fraction is tuned per (M, N, d) tile shape. This is the mechanism that finally **balances `MUFU.EX2` against MMA cycles** for the forward pipeline.

#### (3) Conditional softmax rescale (~10× fewer rescales)

Online softmax normally rescales every block:

$$
O_j = e^{m_{j-1}-m_j}\, O_{j-1} + e^{S_j - m_j}\, V_j.
$$

When $m_j = m_{j-1}$ (no new maximum), the rescaling factor is 1 and the multiply by $O_{j-1}$ is wasted work. FA4 generalizes: the rescale is only physically applied when $m_j - m_{j-1} > \tau$, where $\tau = \log_2(256) = 8.0$ (i.e., a rescale factor exceeding 256). Otherwise the kernel:

$$
O_j = O_{j-1} + e^{S_j - m_{j-1}}\, V_j,\qquad m_j \leftarrow m_{j-1}\ \text{(unchanged)}.
$$

The running statistics $m, \ell$ track all the slack, and at the very last block FA4 normalizes by the *true* final $m_\text{final}$ and $\ell_\text{final}$ — so the result is bit-equivalent (within FP32 rounding) to applying every rescale. Empirically this **deletes ~90% of the rescale vector ops** [VERIFIED — FA4 paper §3.1.4 + Tri Dao Hot Chips presentation linked from Lambda blog].

To avoid warp divergence, the rescale-or-not decision is made per warp: rescale if **any** thread in the warp needs it.

#### (4) TMEM in backward + 2-CTA MMA + DSMEM exchange

Backward chains five MMAs: $S = QK^T$ recompute, $dV = P^T dO$, $dP = dO V^T$, $dQ = dS K$, $dK = dS^T Q$. On Hopper the accumulators are in registers, which forces a serialized schedule (compute $S$ then $dP$ then $dV$ then $dQ$ then $dK$). On Blackwell with TMEM, FA4 keeps two of the ten GEMM operands in TMEM, recomputes $S$ and $P$ in **transposed** tile orientation so $P^T$ and $dS^T$ land *in the operand-A layout the next MMA needs* (no separate transpose), and uses 2-CTA MMA for the backward to halve operand-B SMEM traffic:

- Tile shape: $M = 256, N = K = 128$ for $S, dP, dV, dK$.
- Each CTA stages half of operand B; hardware reads the combined operand B via the cluster's distributed SMEM.
- For $dQ$, the natural 2-CTA M-split conflicts with the reduction over the KV sequence. FA4 resolves this by exchanging half of $dS$ between the two CTAs over **DSMEM** (distributed SMEM, the cluster's combined SMEM addressable across CTAs via `ld.shared::cluster` and `mapa.shared::cluster` PTX). Each CTA then owns $M/2$ rows but holds the *full* $2N = 256$ reduction. The per-CTA $dQ$ MMA tile becomes $(M/2, 2N)(2N, d) \to (M/2, d)$ — **half as many M rows, doubled K**. Side benefit: the $dQ$ atomic-add across the global memory is now half as frequent (each CTA only writes half the $dQ$ tile), which halves a major source of nondeterminism in the backward.

ASCII of the 2-CTA $dQ$ step:

```
   CTA0 (rows 0..127)                   CTA1 (rows 128..255)
   has dS[0..127, 0..127]               has dS[128..255, 0..127]
                          ─── DSMEM exchange ───
   gets dS[0..127, 128..255]            gets dS[128..255, 128..255]
   has full dS[0..127, 0..255]          has full dS[128..255, 0..255]
   MMA (M/2=128, 2N=256, d=128)         MMA (M/2=128, 2N=256, d=128)
   atomicAdd dQ[0..127, :]              atomicAdd dQ[128..255, :]
```

**Deterministic mode.** FA4 ships an opt-in deterministic backward that serializes the global $dQ$ atomicAdd's via a semaphore lock. Naïvely this would crater throughput, but FA4 uses CTA swizzling + a "shortest-processing-time-first" (SPT) ordering for causal masking so that no CTA is stalled on its first write. Reported: **deterministic mode reaches ~75–90% of nondeterministic throughput** [VERIFIED — FA4 paper §3.2.4 + Fig. 7].

### D.3.3 CuTe-DSL implementation (compile-time ~22–32× lift)

FA4 is written entirely in **CuTe-DSL** — a Python-embedded kernel DSL that lowers to PTX, then to SASS via `ptxas`. The programming model is isomorphic to C++ CUTLASS (same `Tensor`, `Layout`, `cute::tile_to_shape` abstractions), with a PTX escape hatch for instructions not yet wrapped in the DSL. The compile-time numbers from the paper's Table 4:

| Method | Forward compile | Backward compile | Speedup |
|---|---|---|---|
| FlashAttention-3 (C++) | 55 s | 45 s | 1× |
| FlashAttention-4 (CuTe-DSL) | **2.5 s** | **1.4 s** | **22×/32×** |

*[VERIFIED VERBATIM — FA4 paper Table 4, exactly as cited in the parent report.]*

Why this matters for video DiT: every research kernel in the parent report (STA, VSA, Radial, SVG2, SLA, Sage 3, etc.) is hand-written and ships hundreds of compiled variants for different `(seq, hd, masking, dtype)` combinations. With CuTe-DSL the total compile time across all variants drops from minutes to seconds, JIT specialization is now feasible, and forking FA4 to add a new mask or score_mod is a 1-day engineering exercise. This is the structural enabler for the FA-FP4 fork (§D.4) and MagiAttention's FFA_FA4 (§D.10).

### D.3.4 Reported numbers (B200, BF16, FP16/BF16 accumulator)

Setup: HGX B200 GPU, hd=128, hidden=2048, batch×seq packed to 32k tokens, 5 warmup + 10 timed runs, average. Software stack: CUDA 13.1, FlashAttention 2.8.3, Triton 3.6, PyTorch 2.10.0, CuTe-DSL 4.4.1.

| Setting | TFLOPs/s | % of 2.25 PF/s peak | Speedup |
|---|---|---|---|
| FA4 forward, hd=128, non-causal | **1605–1613** | **71%** | 1× |
| FA4 forward vs cuDNN 9.13 | — | — | **1.1–1.3×** |
| FA4 forward vs Triton | — | — | **2.1–2.7×** |
| FA4 forward vs FA2 | — | — | larger (FA3 doesn't run on B200) |
| FA4 forward, hd=(192,128) DeepSeek-V3 shape, causal | **— faster than cuDNN across all seq** | — | — |
| FA4 deterministic backward | — | — | **75–90% of nondeterministic** |

*[VERIFIED VERBATIM — FA4 paper §5 + Figs. 4–7 + Lambda blog.]*

For the **FlexAttention + FA4 backend on GB200 NVL72** (custom score_mods compiled through FlexAttention, dispatched to the FA4 kernel under the hood):

| Pattern | Forward vs Triton | Backward vs Triton |
|---|---|---|
| Dense / causal | 1.6–3.2× | 1.85–2.3× |
| ALiBi | 1.2–2.1× | 1.9–2.9× |
| Document masking | up to 2.7× | up to 3× |
| Sliding window | 1.4–2.1× | 1.8–2.2× |

*[VERIFIED VERBATIM — Lambda blog table.]*

Per the FA4 paper, *NVIDIA has since incorporated several FA4 ideas into newer cuDNN releases*, so cuDNN 9.19+ catches up to FA4 in many BF16 cases — see §D.13 for the cuDNN side of the story.

### D.3.5 Limitations and integration caveats

- **TMEM pressure caps masked attention.** At hd=128 with the 2-tile ping-pong layout, the TMEM is full ($S^H + S^L + O^H + O^L = $ full 256 KB / SM). Adding extra metadata for arbitrary masks (e.g., a block-mask ID per query position, or per-block sparse-pattern lookup) requires reusing TMEM columns whose data is already consumed — the paper's 2-CTA backward layout exchanges $S/P$ region with $dP/dS/dQ$ region at predetermined points to make room. FlexAttention sparse-masked variants on FA4 need to follow the same pattern, which is what makes Sparge / SLA on top of FA4 nontrivial.
- **No varlen support in FA4 2.8.3 (current stable as of May 2026).** This is a *practical* blocker for distributed video-DiT training: when MagiAttention benchmarks `varlen` masks on B200, it cannot use FA4 directly — they fall back to cuDNN 9.21 for baselines and to the FFA_FA4 fork (§D.10) for their own kernel. *"we don't report the backward performance for FA4 since it currently lacks robust support for varlen masks, especially on stable version of 2.8.3"* [VERIFIED VERBATIM — MagiAttention CP benchmark blog].
- **No Sage-style INT4/INT8/FP4 path in upstream FA4.** Quantized attention requires the FA-FP4 fork (§D.4) or Sage 3 (§D.8). For head-dim ≠ 128 (e.g., 64 or 256), upstream FA4 supports it but with somewhat reduced TMEM efficiency; see paper §3.1 for the partitioning analysis.
- **Hardware constraint: sm_100a.** FA4 binaries built with `-arch=sm_100a` will *not run* on RTX 5090 (sm_120) — the consumer Blackwell. SM120 does have a tensor memory and `tcgen05.mma`, but the encoding differs and the FA4 CuTe-DSL has not been ported as of May 2026.

### D.3.6 Composition with the rest of the stack

- **Cross-ref §2 NVFP4 GEMM** (parent report). FA4 is BF16; FA-FP4 (§D.4) is the FP4 variant that *consumes* the output of the FP4 QKV/O linear without a BF16 round-trip. Both rely on `tcgen05.mma`.
- **Cross-ref §3.1 cuDNN 9.21 SDPA** (§D.13 below). cuDNN incorporated FA4's conditional rescale (`CUDNN_RESCALE_THRESHOLD=8`) and FMA-emulated `exp` (`CUDNN_USE_EX2_EMULATION=1`) in the 9.13 → 9.14 line, so the practical performance gap on BF16 has closed.
- **Cross-ref §1 Ulysses / Ring SP** (parent report). FA4 is the natural inner kernel for Ulysses on B200 (just like FA3 was for H100). For Ring, the cross-rank `(O, lse)` interface needs to be exposed — the ring driver typically re-implements the FA inner loop in Triton or Helion to expose `(O, lse)` per K,V block, since upstream FA4 returns only `(O, lse_combined)`. See §D.10–D.12 for the distributed companions.
- **Cross-ref §3.4 sparse attention.** FlexAttention with FA4 backend (§D.14) is the documented path for STA/SSTA, Radial-Attention masks, BlockMask, and document masking on B200.

---

## D.4 FA-FP4 — the hao-ai-lab Blackwell NVFP4 fork

**References.** [hao-ai-lab/flash-attention-fp4](https://github.com/hao-ai-lab/flash-attention-fp4) (9 stars on May 2026 [VERIFIED] — small repo, single-purpose fork). Downstream paper is the hao-ai-lab "Attn-QAT" recipe (referenced from FastVideo's blog at [hao-ai-lab.github.io/blogs/fastvideo](https://hao-ai-lab.github.io/blogs/fastvideo/)).

### D.4.1 Motivation: closing the FP4 GEMM ↔ attention round-trip

Once the QKV / O linear is NVFP4 (via [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)'s downstream pipeline, [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer), or LightX2V's `w4a4-nvfp4`), the natural next question is: can attention also be NVFP4? Without FA-FP4, the FP4 outputs of the QKV linear are dequantized to BF16 before FA4, BF16 attention runs, and the BF16 output of attention is requantized to FP4 for the O linear. That round-trip costs both compute and memory bandwidth.

FA-FP4 retargets the FA4 pipeline to NVFP4 for $Q$ and $K$ (and $P$, post-softmax), keeps $V$ in either NVFP4 or FP8, and accumulates in FP32. Because Blackwell's `tcgen05.mma.kind::f4nvf4.block_scale` runs at roughly **2× the BF16 path** [VERIFIED — Microbenchmarking paper [arXiv:2512.02189](https://arxiv.org/abs/2512.02189) Table IV: 5300 TF/s FP4 vs 2250 TF/s BF16 on B200], FA-FP4 can lift attention throughput from FA4's 71% utilization to ~80% of the 2.25 PF/s ceiling — and crucially deliver a *higher absolute TFLOPs/s* number even at lower fractional utilization, because the underlying FP4 ceiling is 5.3 PF/s (≈2.4× the BF16 ceiling).

### D.4.2 Mechanism

**NVFP4 numeric format.** Group size 16 along the contraction axis, element type **E2M1** (1 sign + 2 exponent + 1 mantissa, encoding 15 representable values: 0, ±0.5, ±1.0, ±1.5, ±2, ±3, ±4, ±6), **per-group FP8 (E4M3) scale**. Maximum representable value in E2M1 is 6, so the natural scale factor is `s = max(|x_group|) / 6`. The FP8 group-scale gives ~2× more dynamic range per group than MX-FP4's E8M0 group-scale (which is power-of-two only), at the cost of 2× larger scale-tensor footprint and tighter group size 16 vs MX's 32.

**On-the-fly score quantization.** After softmax produces $P \in [0, 1]^{M \times N}$, FA-FP4 quantizes each row of $P$ to NVFP4 *inside the kernel*, computing the row-max in TMEM-resident intermediate registers. The trick that makes this fit on B200 is **TMEM scale-factor packing**: the FA4 forward layout fully consumes TMEM with $S^H + S^L + O^H + O^L$ at hd=128, but FA-FP4 reuses the $S^H/S^L$ TMEM columns *after* their data is no longer live (post-`P` store) to stash the per-row NVFP4 group-scales for $P$. Without this trick FA-FP4 spills to SMEM and loses ~15% of the FP4 throughput advantage.

**Pseudocode of the FA-FP4 inner loop (one tile):**

```
1. Load Q[i] tile (NVFP4 + scales) from SMEM via tcgen05.mma operand-A path.
2. Load K[j] tile (NVFP4 + scales) from SMEM via tcgen05.mma operand-B path.
3. tcgen05.mma.kind::f4nvf4.block_scale  →  S[i,j] (FP32, in TMEM)
4. Online softmax (FMA-emulated exp + conditional rescale, same as FA4):
       m_new = max(m_old, rowmax(S[i,j]))
       P[i,j] = exp(S[i,j] - m_new)  ∈ [0, 1]^{M×N}
5. Quantize P[i,j] row-wise to NVFP4:
       per-row scale  s_P = rowmax(P) / 6.0   (E4M3, packed into TMEM scale slot)
       P_FP4 = round(P / s_P) → E2M1
6. tcgen05.mma.kind::f4nvf4.block_scale  →  O[i] += P_FP4 · V[j]  (FP32, in TMEM)
7. Apply scale: O[i] *= s_P (handled at final normalization, not per-iteration).
```

The most subtle numerical issue is *step 5*: $P$'s entries are in $[0, 1]$, with the largest element per row equal to $1$ (after the `m`-shifted softmax). Without the scale-factor expansion `× 6` (which Sage 3 calls "two-level quantization"), the FP4 quantization would only cover $[0, 6]$ and waste 6/8 of the dynamic range. FA-FP4 sidesteps this in a slightly different way than Sage 3: instead of `× (448 × 6) = 2688` two-level scaling, FA-FP4 uses *the row's own max* as the scale, i.e., `s = rowmax(P) / 6`. This is mathematically equivalent to per-row dynamic scaling but loses one level of the scale-factor's resolution. The Attn-QAT recipe (hao-ai-lab blog) bridges the gap by fine-tuning Q/K activation distributions to preserve quality.

### D.4.3 Reported numbers

Setup: B200 HGX, batch=1, seq=32768, num_heads=16, head_dim=128, causal=False:

| Kernel | TFLOPs/s | % of peak | Speedup over FA4 BF16 |
|---|---|---|---|
| FA4 BF16 (reference) | 1613 | 71% (of 2.25 PF/s) | 1× |
| **FA-FP4 NVFP4** | **1801** | ~80% (of ~2.25 PF/s; ~34% of 5.3 PF/s FP4 peak) | **1.39×** |

*[VERIFIED — README of [hao-ai-lab/flash-attention-fp4](https://github.com/hao-ai-lab/flash-attention-fp4) and reproduced verbatim in the parent report's table.]*

The 1.39× lift is below the theoretical ~2× from the FP4-vs-BF16 throughput ratio because (a) softmax `exp` is still on the FMA/MUFU path at FP32 (not FP4), (b) on-the-fly $P$ quantization adds ~5–10% overhead per tile, and (c) FA-FP4 still has to write $O$ in FP32 to TMEM (no cumulative FP4 accumulator).

**No public end-to-end Wan / Hunyuan / LTX FA-FP4 number** has been published as of May 2026 — the headline 1801 TF/s is a kernel-level microbenchmark. The hao-ai-lab Attn-QAT recipe claims quality-neutral results on Wan-2.1-class models, but those are internal numbers.

### D.4.4 Limitations and the SageAttention 2++ "first/last steps" workaround

- **No FA-FP4 published backward.** The repo only exposes the forward, so this is an inference-only kernel. Training-time attention on B200 NVFP4 is not yet served by FA-FP4 (and Sage 3's SageBwd is INT8, not FP4).
- **Quality on extreme timesteps.** The Sage 3 paper documents that FP4 attention has worst-case quality on the very first and last denoising timesteps because $P$ has the widest distribution there. The standard recipe is **Sage 2++ (FP8 PV) on the first ~3 and last ~3 timesteps + FA-FP4 (or Sage 3) in the middle**; this is the same recipe that the parent report's §3 advocates, and the FA-FP4 README mirrors it.
- **No `BlockMask`/varlen path.** FA-FP4 inherits FA4 2.8.3's `varlen` limitation. Cross-ref §D.10 (MagiAttention's FFA_FA4 *does* add varlen + block-causal masks on top of an FA4 fork — though without the FP4 path).

### D.4.5 Composition

Cross-ref **§D.3 FA4** (FA-FP4 is a strict superset of FA4's pipeline with FP4 added at the operand-data level), **§D.8 SageAttention 3** (which does the same FP4 attention but with two-level $P$ quantization for a different accuracy profile, and a different kernel design specifically for SM120 rather than SM100a), and **§2 NVFP4 GEMM** (parent report — both FA-FP4 and the FP4 QKV/O linear share `tcgen05.mma.kind::f4nvf4.block_scale`).

---

## D.5 SageAttention 1 — INT8 + smooth_k (Ada / Ampere / Hopper)

**References.** Zhang, Wei, Huang, Zhang, Zhu, Chen, [arXiv:2410.02367](https://arxiv.org/abs/2410.02367), ICLR 2025; code [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention).

### D.5.1 Motivation: why naive INT8 attention generates blurry video

Direct per-tensor INT8 quantization of $Q, K$ produces a "completely blurry video" on CogVideoX and a 25.5% MMLU accuracy on Llama-2 (random-guess level). The reason is documented in the paper's Section 4.2 and Figure 4: $K$ has *channel-wise* outliers that are roughly equal across all tokens — i.e., they're a per-channel **bias**, not high-variance signal. Per-channel quantization can't be applied to $K$ in attention (per-channel = along the inner contraction axis of $QK^\top$, which the dequantization formula doesn't support), so the per-block / per-token / per-tensor scales must absorb this bias, and the available INT8 dynamic range gets allocated to the bias instead of the legitimate signal.

### D.5.2 Core idea: smooth_k preprocessing

The SageAttention 1 trick is to subtract the **token-mean of K** *before* quantizing:

$$
K' = K - \bar{k},\qquad \bar{k} = \frac{1}{N} \sum_{t=1}^{N} K[t, :] \in \mathbb{R}^{1 \times d}.
$$

This is a *per-channel constant shift* (broadcast across all tokens). Because softmax is shift-invariant in its argument:

$$
\text{softmax}\Big(Q (K - \bar{k})^\top\Big) = \text{softmax}\Big(Q K^\top - Q \bar{k}^\top\Big) = \text{softmax}(Q K^\top),
$$

the subtraction *cancels* in attention — every row of $S$ is shifted by the same per-row constant $Q \bar{k}^\top$, which falls out under softmax's `S - max(S)`. So smooth_k changes nothing mathematically but **massively narrows $K$'s dynamic range** so per-block INT8 quantization scales are well-allocated.

Cost: **<0.2% kernel overhead** for the mean subtraction, fused with the K quantization kernel [VERIFIED — SageAttention 1 paper Section 4.2 + Table 10].

The numbers (Sage 1 paper Table 1) make the case directly:

| Model | Per-token INT8 (no smooth) | Per-token INT8 (smooth_k) | FP16 baseline |
|---|---|---|---|
| Unidiffuser FID ↓ | 221.18 | **166.52** | 163.33 |
| CogVideoX FScore ↑ | 1.924 | **3.734** | 3.768 |
| UltraPixel FID ↓ | 193.36 | **179.79** | 179.78 |
| TIMM ImageNet ↑ | 84.21% | **84.74%** | 84.79% |

*[VERIFIED — SageAttention 1 paper Table 1.]*

I.e., smooth_k recovers essentially full FP16 quality on every model studied. Notably **FlashAttention-3 FP8 without smooth_k loses badly** on CogVideoX (FScore 3.394 vs 3.768 baseline) and UltraPixel (FID 383.61!), motivating Sage 3 in Hopper FP8 land later.

### D.5.3 Sage 1 kernel structure (4 variants)

The paper ships four kernels (Sage 1 paper Table 6):

| Kernel | $\psi_Q, \psi_K$ | $\psi_P$ | $\psi_V$ | Use case |
|---|---|---|---|---|
| SAGEAttn-T | per-token, INT8 | FP16, FP16 accum | FP16, FP16 accum | Highest accuracy, fastest on RTX 4090 |
| SAGEAttn-B | per-block, INT8 | FP16, FP16 accum | FP16, FP16 accum | Fewer scales than -T |
| SAGEAttn-vT | per-token, INT8 | per-block, INT8 | per-channel, INT8 | INT8 PV, broader GPU support |
| SAGEAttn-vB | per-block, INT8 | per-block, INT8 | per-channel, INT8 | Lowest accuracy, broadest support |

**The "FP16 accumulator on PV" key trick.** On RTX 4090 and 3090, `mma.f16.f16.f16.f16` (FP16 input, FP16 accumulator) is **2× faster** than `mma.f32.f16.f16.f32` (FP16 input, FP32 accumulator). Sage 1 keeps $P, V$ in FP16 and uses the FP16 accumulator for $PV$. This is the paper's "FP16 accumulator: much more accurate and efficient" section — it's "much more accurate" because using INT8 for $\widetilde{P}$ (which has many small entries near zero) loses precision badly, and "much more efficient" because the FP16 accumulator path is 2× faster than FP32 on consumer Ada/Ampere. *Two-level accumulation*: inner block-loop accumulates in FP16; outer LSE rescale uses FP32. The key empirical finding (Sage 1 paper Table 4–5) is that this two-level scheme has **identical accuracy to a pure FP32 accumulator** at attention output [VERIFIED].

This FP16-accumulator trick **only helps on consumer Ada/Ampere** (RTX 4090, 3090). On Hopper / data-center Ada (L40, L20, H100), `mma.f16.f16` runs at the same speed as `mma.f32.f16`, so Sage 1 reverts to FP32 accumulation on those GPUs. Sage 2++ (§D.7) generalizes the FP16-accumulator trick to **FP8** input on Hopper.

### D.5.4 Reported numbers

| GPU | Kernel | Speedup | Throughput |
|---|---|---|---|
| RTX 4090 | SageAttn vs FA2 (hd=128) | **2.1×** | ~340 TOPS |
| RTX 4090 | SageAttn vs xformers | **2.7×** | ~340 TOPS |
| RTX 4090 | FA2 baseline | 1× | ~165 TOPS |
| RTX 4090, hd=64 | SageAttn | — | ~340 TOPS, *competitive with FA3 (490) on Hopper* |
| L20 | SageAttention vs FA2 | similar | — |

*[VERIFIED — Sage 1 paper §5 + Figure 1.]*

Sage 1 reaches ~52% of RTX 4090's INT8 theoretical peak. The paper notes that **Sage 1 INT8 on RTX 4090 (340 TOPS) is close to FA3 FP8 on H100 (490 TOPS)** at hd=64 — i.e., the consumer SKU comes within 1.4× of Hopper's FP8 path, while costing ~5× less.

### D.5.5 End-to-end on video DiT

CogVideoX (1.5-5B), HunyuanVideo, Mochi: smooth_k + Sage 1 produces FScore / VQA-a / VQA-t identical to FP16 baseline within rounding. **Note: directly using FA3 FP8 on these models gives FScore = ✗ (failed)** — the Sage 1 paper attributes this to FA3 FP8's lack of smooth_k; the same observation drives Sage 2's per-thread granularity refinement (next subsection).

### D.5.6 Composition

Sage 1 is the *foundation* of every later Sage variant; smooth_k carries through to Sage 2, 2++, and 3, and is **also used by FA-FP4** (the hao-ai-lab Attn-QAT recipe references Sage's K-mean-subtraction). Cross-ref **§D.6 Sage 2** (which adds smoothing on $Q$), **§D.8 Sage 3** (which keeps smooth_k unchanged), and **§3.4 sparse attention** in the parent report (Sparge attention plugs into Sage 2++ via shared kernel infrastructure).

---

## D.6 SageAttention 2 — per-thread INT4 + FP8 PV (Hopper / Ada)

**References.** Zhang, Huang, Zhang, Wei, Zhu, Chen, [arXiv:2411.10958](https://arxiv.org/abs/2411.10958), ICML 2025.

### D.6.1 Motivation: extending Sage 1 to INT4 + FP8

Sage 1's two limitations:

- **W1.** INT8 Matmul on RTX 4090 is half the speed of INT4 (which goes through `mma.m16n8k64` at 2 PFLOP/s on Ada). Going to INT4 is a free 2×.
- **W2.** Sage 1's FP16-with-FP16-accumulator PV trick only works on consumer Ada (RTX 4090, 3090). On L20, L40, H100, FP16-FP16 = FP16-FP32 in throughput. To get the 2× lift on those GPUs, attention needs to use FP8 PV — which has been the only path to >1 PFLOP/s on Hopper attention since FA3.

Sage 2's three contributions: (i) **smooth Q** (in addition to smooth K from Sage 1), (ii) **per-thread INT4 quantization** matched to `mma.m16n8k64`'s memory layout, (iii) **two-level FP32 accumulator** atop FP22 to recover lost precision.

### D.6.2 Smooth Q (the second-half of the smoothing argument)

Sage 1's smooth_k handles K. Sage 2 observes that $Q$ has the same channel-bias structure (the cross-token mean $\bar{q}$ is small but nonzero, and the per-channel mean is the dominant outlier source), so it applies the analogous transform per Q-tile:

$$
\gamma(Q_i) = Q_i - \bar{q}_i,\qquad \bar{q}_i = \text{mean}(Q_i, \text{axis=token}) \in \mathbb{R}^{1 \times d}.
$$

The decomposition is:

$$
S_{ij} = Q_i K_j^\top = (\bar{q}_i + \gamma(Q_i))(\bar{k} + \gamma(K_j))^\top
$$

$$
= \gamma(Q_i)\gamma(K_j)^\top + \underbrace{\bar{q}_i \gamma(K_j)^\top}_{\Delta S_{ij}\;(1 \times N\;\text{vec})} + \underbrace{\gamma(Q_i) \bar{k}^\top + \bar{q}_i \bar{k}^\top}_{=:\,b\;(N \times 1\;\text{vec, drops out under softmax})}.
$$

The first term is the GEMM that runs at INT4. The second is a **GEMV of $\bar{q}_i$ against $\gamma(K_j)$** — cheap, fused into the quantization kernel, runs once per Q-tile. The third term is constant within a row, so it cancels in `softmax(S - max(S))`. Empirically the smoothing order is `Q+K > Q > K > none`, with smoothing both Q and K recovering essentially full FP16 quality even at INT4 [VERIFIED — Sage 2 paper Table 4 + Appendix A.5 theoretical analysis].

### D.6.3 Per-thread INT4 quantization

The PTX `mma.m16n8k64` instruction (the fastest INT4 MMA on Hopper / Ada) takes a 16×64 A-tile and an 8×64 B-tile per warp, with a specific thread-to-element layout:

```
Per-warp A-tile, mma.m16n8k64:
  threads 0..3 own row 0, cols 0..15 (8 INT4 values × 4 threads)
  threads 4..7 own row 1, cols 0..15
  ...
  threads 28..31 own row 7, cols 0..15
  (threads 0..3 own rows 8..15, cols 16..31, etc.)
```

Sage 2's "per-thread quantization" matches the granularity: each *GPU thread* gets its own scale factor, computed over exactly the 8 tokens it sees (for $Q$) and 16 tokens (for $K$). The result is a quantization granularity strictly finer than per-block (one scale per 64-row block) but coarser than per-token (one scale per 1 row × any cols), with **zero dequantization overhead** because each thread only multiplies by *its own* single scale value during the MMA result post-processing — no scale broadcast, no inter-thread shuffle.

This is the kernel-level innovation that lets INT4 attention work without per-token scales. The corresponding `mma` call is:

```ptx
mma.m16n8k64.row.col.s32.s4.s4.s32  {acc_d0, acc_d1, acc_d2, acc_d3},
                                    {qa0, qa1, qa2, qa3, qa4, qa5, qa6, qa7},
                                    {qb0, qb1, qb2, qb3},
                                    {acc_d0, acc_d1, acc_d2, acc_d3};
```

Eight INT4 elements packed into each `qa*` register; per-thread scale post-multiply happens after the MMA returns.

### D.6.4 FP22 accumulator discovery and two-level accumulation

A subtle empirical finding (Sage 2 paper Section 3.4): the `mma.f32.f8.f8.f32` instruction on Ada / Hopper, despite *advertising* an FP32 accumulator, in practice accumulates into a **22-bit float (FP22)** with 1 sign + 8 exponent + 13 mantissa bits. The Sage 2 authors verified this by initializing $A, B$ to zero and varying $D$ — when $D$ has more than 13 mantissa bits, the result $C$ has its bottom 10 mantissa bits truncated.

This is the same reason CUTLASS's FP8 GEMM and DeepGEMM use a two-level accumulator: every $b_k = 64$ MMA accumulations, the FP22 result is *added to a separate FP32 buffer* and the FP22 register is reset. Sage 2 implements the same scheme on the PV product, accumulating partial $\tilde{P} V$ in FP22 over a 64-token block, then casting to FP32 and adding to the running $O_{ij}$.

### D.6.5 Reported numbers

| Setup | Sage 2 vs FA2 OPS | Sage 2 vs xformers | E2E | E2E baseline |
|---|---|---|---|---|
| RTX 4090, hd=128 | **3×** | **4.5×** | — | — |
| L20 | matches FA3-FP8 speed | — | — | — |
| **CogVideoX (1.5-5B)** | — | — | **555 s** | 1040 s (1.87×) |
| **HunyuanVideo** | — | — | **1435 s** | 2221 s (1.55×) |

*[VERIFIED — Sage 2 paper Table 1 + Figure 5.]*

Quality (Sage 2 paper Table 2):

| Model | Sage 2-4b VQA-t | Sage 2-8b VQA-t | FP16 VQA-t |
|---|---|---|---|
| CogVideoX (1.5-5B) | 52.99 | 74.42 | 70.93 |
| HunyuanVideo | 65.37 | 75.35 | 75.93 |
| Mochi | 43.74 | 64.90 | 65.42 |

I.e., the 8-bit variant matches FP16; the 4-bit variant has small (~5–10%) drops on some metrics. This is why production stacks default to Sage2(8+8) on most timesteps and Sage2(4+8) only on the middle steps where the activation distribution is more forgiving.

### D.6.6 Limitations

- **Smooth_v not always applied.** The paper notes that smoothing $V$ helps *only* when $V$ has channel-wise bias — present in some models (CogVideoX) but not Llama-3.1. So Sage 2 ships smooth_v as optional, off by default.
- **Smoothing requires storing the means.** $\bar{q}_i$ and $\bar{k}$ live in fast on-chip memory but are recomputed every layer; the cost is ~0.5% of the FFN+attn time.
- **No SM100 path.** Sage 2 ships only `mma.m16n8k64` / `mma.m16n8k32` (sm_89 / sm_90), not `tcgen05.mma`. The SM100 Sage 3 (next subsection) adds the Blackwell path.

### D.6.7 Composition

Cross-ref **§D.7 Sage 2++** (uses the same per-thread INT4 + smooth Q/K but switches to `mma.f16.f8.f8.f16` for PV); **§D.8 Sage 3** (extends per-thread to NVFP4 microscaling); **§3.4 SpargeAttn** in the parent report (block-sparse on top of Sage 2's INT4 + FP8 PV kernel).

---

## D.7 SageAttention 2++ — `mma.f16.f8.f8.f16` swap (Hopper / Ada)

**References.** Zhang, Xu, Wei, Huang, Zhang, Xiang, Zhu, Chen, [arXiv:2505.21136](https://arxiv.org/abs/2505.21136), ICML 2025 short.

### D.7.1 Motivation: the FP8-with-FP16-accumulator instruction

Sage 2 uses `mma.f32.f8.f8.f32` (FP8 input, FP32 accumulator) for the PV product, which is **2× faster** than the FP16-FP32 path. The Sage 2++ paper observes that **`mma.f16.f8.f8.f16` (FP8 input, FP16 accumulator) is 2× faster again** — i.e., 4× over FP16 input with FP32 accumulator [VERIFIED — Sage 2++ paper Table 1, citing the [Ada Lovelace Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)].

So Sage 2++ swaps the PV instruction from `f32.f8.f8.f32` to `f16.f8.f8.f16`. The risk: **FP16 has a max representable value of 65504**, and `mma.m16n8k32` accumulates **32 product terms** into one FP16 cell. If any single product `p × v` × 32 exceeds 65504, the accumulator overflows.

### D.7.2 The scale-narrowing trick

Sage 2++ proves a sufficient safety condition: if $P$ is quantized with scale $P_r$ (so $P$'s quantized magnitude is at most $|P_q| \le P_r$) and $V$ with scale $V_r$ (so $|V_q| \le V_r$), then a 32-term FP16 accumulation stays in range iff

$$
|32 \cdot P_q \cdot V_q| \le 65504 \quad\Longleftrightarrow\quad P_r \cdot V_r \le \frac{65504}{32} = 2047.
$$

The "delayed FP32 buffering" optimization saves the FP16-to-FP32 cast cost by accumulating *two* `mma.m16n8k32` results in FP16 before casting — that doubles the budget needed:

$$
P_r \cdot V_r \le \frac{2047}{2} \approx 1023.5.
$$

The choice $P_r = 224, V_r = 4.5$ gives $1008 < 1023.5$, satisfying the bound. Sage 2++'s on-the-fly per-block scales are computed with these targets:

$$
\delta_P = \frac{|\max(\widetilde{P})|}{P_r = 224},\qquad \delta_V = \frac{|\max(V)|}{V_r = 4.5}.
$$

(For comparison, Sage 2 used $P_r = V_r = 448$, the full E4M3 range, which would saturate the FP16 accumulator at $448 \times 448 \times 32 = 6{,}422{,}528$.)

The accuracy table (Sage 2++ paper Table 2) shows that across $(P_r, V_r) \in \{(448, 448), (448, 2.25), (224, 4.5), (112, 9)\}$ the cosine similarity to FP16 reference is identically **99.97%** and L1 error is essentially identical, so the scale narrowing has **no measurable accuracy cost** [VERIFIED VERBATIM].

### D.7.3 Reported numbers

| GPU | Sage 2++ vs FA2 OPS | Sage 2++ vs Sage 2 |
|---|---|---|
| RTX 4090, hd=128 | **3.9×** (4-bit Q,K + FP8 PV) | 1.3× |
| RTX 4090, hd=128 | **3.0×** (8-bit Q,K + FP8 PV) | 1.0× |
| RTX 5090 | similar lift |  similar |

*[VERIFIED — Sage 2++ paper Figs. 1–4.]*

End-to-end on Wan, HunyuanVideo, CogVideoX, FLUX, SD3.5: Sage 2++ matches Sage 2 across every metric (Sage 2++ paper Table 3).

### D.7.4 Composition

Sage 2++ is **the parallel option to FA-FP4 on B200** for the FP8 attention slot. The standard production recipe in the parent report's §3 stack:

- **First / last 3 timesteps**: Sage 2++ (FP8, no quality loss).
- **Middle timesteps**: FA-FP4 (NVFP4) or Sage 3 (when SM100 path is restored).

The Sage 2++ kernels are reachable via FastVideo's `SageSLAAttentionImpl` class through the `qattn.qk_int8_sv_f8_accum_f16_*` PTX symbols when `SAGE2PP_ENABLED=1`. Cross-ref **§3.4 SpargeAttn** in the parent report (Sparge plugs into Sage 2++ kernels for sparse-attention on Hopper).

---

## D.8 SageAttention 3 — NVFP4 microscaling attention (Blackwell)

**References.** Zhang*, Wei*, Wang, Zhang, Xu, Huang, Jiang, Chen, Zhu, [arXiv:2505.11594](https://arxiv.org/abs/2505.11594), NeurIPS 2025 Spotlight; code [thu-ml/SageAttention/tree/main/sageattention3_blackwell](https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell). Two parts: (1) FP4 *inference* attention, (2) INT8 *trainable* attention "SageBwd".

### D.8.1 Motivation and challenges

Sage 3 targets Blackwell's FP4 Tensor Cores via `tcgen05.mma.kind::f4nvf4.block_scale`. The instruction itself is ~8× faster than FP16 on RTX 5090 (~1600 TOPS vs ~200 TOPS [VERIFIED — Sage 3 paper §3.1]). Three challenges (paper §1):

- **C1.** FP4 has only 15 representable values (E2M1: 0, ±0.5, ±1.0, ±1.5, ±2, ±3, ±4, ±6). Per-tensor or per-token scales are inadequate to absorb outliers.
- **C2.** The post-softmax matrix $P$ has values in $[0, 1]$. The natural FP4 scale `s = max(P)/6 ∈ [1/6, 1/6]` is small (always $\le 1/6 = 0.167$), and **NVFP4's group-scale must be E4M3 FP8**. The narrow scale-factor distribution wastes E4M3's range and accumulates large errors.
- **C3.** For trainable attention (SageBwd), the gradient of $\text{softmax}$ ($dS = P \odot (dP - D)$) is acutely sensitive to quantization error in $dP = dO \cdot V^\top$, propagating into $dQ$ and $dK$.

### D.8.2 Microscaling FP4 + two-level P quantization (the C2 fix)

For $Q, K, V$, Sage 3 uses NVFP4 with $1 \times 16$ blocks and FP8 (E4M3) per-group scales — straightforward. The novel piece is the **two-level scaling for $\tilde{P}$**:

- **Level 1 — per-token expansion to E4M3's full range.** Each row of $\tilde{P}$ is multiplied by `1 / (max(P_row) / (448 × 6))` so that the resulting matrix lives in $[0, 448 \times 6 = 2688]$. This expansion uses an FP32 scale $s_{P_1}$.
- **Level 2 — standard NVFP4 microscaling.** The expanded matrix is then quantized to NVFP4 with FP8 per-group scales $s_{P_2}$.

The decoder side: $\tilde{P} \approx \hat{P}_2 \cdot s_{P_2} \cdot s_{P_1}$. The PV kernel computes $\hat{P}_2 \cdot \hat{V}$ via `tcgen05.mma.kind::f4nvf4.block_scale` (using $s_{P_2}, s_V$), then multiplies by $s_{P_1}$ at the *outer* (FP32) loop level — no per-block FP8 multiplication overhead.

The accuracy table (Sage 3 paper Table 1b):

| Method | CosSim ↑ | L1 ↓ | RMSE ↓ |
|---|---|---|---|
| Direct NVFP4 quantization of P | 93.32% | 0.193 | 1.103 |
| **Two-level NVFP4 quantization** | **99.52%** | **0.077** | **0.201** |

*[VERIFIED VERBATIM — Sage 3 paper Table 1b.]*

A **6.2 percentage-point** lift in cosine similarity from a single algorithmic trick.

### D.8.3 Hardware-level optimizations (the C1 fix in practice)

Three Blackwell-specific kernel tricks (Sage 3 paper §3.3):

- **K-permutation for layout matching.** Unlike FP16, the FP4 MMA's FP32 accumulator memory layout does not match operand A's register layout. Naïvely doing a thread shuffle to match degrades performance. Sage 3 instead **permutes the columns of $K$** during quantization (free — fused with the K quant kernel) so that the natural accumulator layout matches operand A. This costs nothing at runtime.
- **Reuse-shuffle for online softmax.** The in-kernel NVFP4 quantization of $\tilde{P}$ requires the row-max over 16 consecutive elements, which are distributed across 4 threads. Sage 3 fuses this max-finding step with online softmax's row-max reduction (which it has to compute anyway), yielding **~10% kernel speedup** by deduplicating the inter-thread shuffle.
- **Producer warp epilogue.** Conventional warp specialization has consumer warps do MMA + store, producer warps do load. Sage 3 ping-pongs *between two producer warps*: while one loads inputs for the next MMA, the other stores outputs from the previous MMA to global memory. This overlaps stores with loads despite tight register constraints.

### D.8.4 SageBwd: 8-bit *trainable* attention (the C3 fix)

The five backward MMAs in attention: $S = QK^\top$, $dV = P^\top dO$, $dP = dO V^\top$, $dQ = dS K$, $dK = dS^\top Q$. Sage 3 quantizes four to INT8 (per-block) but **keeps $dO V^\top$ in FP16** because (Sage 3 paper Table 1c):

| Strategy | $dQ$ CosSim ↑ | $dQ$ L1 ↓ | $dQ$ RMSE ↓ |
|---|---|---|---|
| All INT8 (incl. $dOV^\top$) | 97.47% | 0.171 | 2.440 |
| **Keep $dOV^\top$ in FP16** | **99.77%** | **0.039** | **0.692** |

I.e., $dOV^\top$'s accuracy bottoms out the $dP \to dS \to dQ$ chain — the recurrence accumulates errors over the sequence dimension during the backward pass, so the longer the sequence the worse the all-INT8 strategy gets.

### D.8.5 Reported numbers

| Setup | TOPS / speedup | Source |
|---|---|---|
| RTX 5090, hd=128, FP4 attention | **1038 TOPS** | [VERIFIED — Sage 3 paper Fig. 4] |
| RTX 5090, vs fastest FA on RTX 5090 | **5×** | [VERIFIED — paper §1] |
| RTX 5090, vs xformers | **11×** | [VERIFIED — paper §5] |
| B300 theoretical-throughput ceiling, Table 17 | 15,000 TOPS Sage 3 vs 5,000 TOPS FA3 FP8 | **3× ceiling** [VERIFIED — paper Table 17] |
| RTX 4090, SageBwd vs FA2 | **1.67×** forward+backward | [VERIFIED — paper §5] |

End-to-end on RTX 5090 (Sage 3 paper Figure 1):

| Model | FP16 baseline | Sage 3 (FP4) | Speedup |
|---|---|---|---|
| **CogVideoX (2B)** | 64 s | **27 s** | **2.4×** |
| **HunyuanVideo** | 489 s | **164 s** | **3.0×** |

VBench / CLIP / FID / IR essentially unchanged from FP16 across CogVideoX, HunyuanVideo, Mochi, FLUX, SD3.5 (Sage 3 paper Table 2).

[CORRECTED — see parent report] **No "37% over FA3" number exists in the Sage 3 paper.** The defensible Sage 3 ↔ FA3 framing is the Table 17 theoretical ceiling: **Sage 3 (FP4) on B300 = 15,000 TOPS vs FA3 (FP8) = 5,000 TOPS, i.e., 3× ceiling**. The 5× kernel-level lift is over **FA2 on RTX 5090**, the fastest FlashAttention on a consumer Blackwell SKU. On Hopper FP8, Sage 2 (8-bit) speed is essentially identical to FA3 (8-bit): 885 vs 890 TOPS — i.e., Sage 2 ≈ FA3 in speed [VERIFIED — Sage 3 paper §5].

### D.8.6 SageBwd training results

Sage 3 paper Table 3:

| Model | Method | GSM8K Acc ↑ | DROP F1 ↑ | MMLU Acc ↑ | HELLASWAG Acc ↑ |
|---|---|---|---|---|---|
| Qwen2.5 (1.5B) | BF16 | 0.521 | 0.733 | 0.569 | 0.905 |
|  | **SageBwd** | 0.520 | 0.734 | 0.574 | **0.911** |
| Qwen2.5 (3B) | BF16 | 0.601 | 0.785 | 0.640 | 0.944 |
|  | **SageBwd** | 0.607 | 0.782 | **0.653** | 0.943 |

SageBwd is **lossless on fine-tuning**, but **slower-converging on pretraining** — that's the explicit caveat in the paper's §5.4 ablations. So the 8-bit attention training story is "yes for finetuning, not for pretraining yet."

### D.8.7 SM100 (B200) status — the issue #322 caveat

[VERIFIED — full timeline reproduced verbatim from [thu-ml/SageAttention#322](https://github.com/thu-ml/SageAttention/issues/322), opened 2025-12-04, closed 2026-01-01.]

> **@MengYu10151 (Dec 11, 2025, 4:31 pm).** "Same questions. Hi @jt-zhang, I noticed you revert many MR including support for SM100. Why is that?"
>
> **@jt-zhang (Dec 11, 2025, 6:27 pm).** "Thanks for asking. We found that, for some unknown reason, the API state before the reverts could introduce potential accuracy issues. To be safe, we reverted the repo back to a known stable version. In the next few days, we'll selectively re-apply the commits that we've verified to be correct and reliable, including the SM100 support."
>
> **@hiahiawei (Dec 23, 2025, 3:55 am).** "Hi @jt-zhang, has SM100 not been re supported yet?"
>
> **@mobcat40 (Dec 31, 2025, 5:03 pm).** "Currently working on RTX 5090 (sm_120) with PyTorch 2.11 nightly. Got SageAttention 2.2.0 working on Blackwell (RTX 5090, sm_120). Sharing a prebuilt wheel and build instructions for anyone stuck waiting for official support. Prebuilt wheel + build instructions: <https://github.com/mobcat40/sageattention-blackwell>. ... ~35% speedup on diffusion sampling (tested with Qwen Image Edit). Note: If using with Qwen or Wan models, avoid the `--use-sage-attention` flag (Triton backend causes black output). Use KJNodes 'Patch Sage Attention' node with sageattn_qk_int8_pv_fp16_cuda backend instead."
>
> **@peepeepeepoopoopoo (Jan 1, 2026, 1:17 am).** "spam"
>
> Issue closed by author.

**Summary of the SM100 status (as of May 2026):**

- **SM120 (RTX 5090) is officially supported** in `sageattention3_blackwell` and ships in upstream wheels.
- **SM100 (B200) was on `main`, then reverted on Dec 4, 2025** for "potential accuracy issues" with no firm re-enable timeline.
- **As of Jan 1, 2026 the issue closed without a re-enable; as of May 2026 the SM100 path is still not re-enabled in upstream.**
- **mobcat40's [community SM120 wheel](https://github.com/mobcat40/sageattention-blackwell)** is the working consumer-Blackwell path. Production B200 deployments rely on **FA-FP4 (hao-ai-lab) for the FP4 slot** and **Sage 2++ for the FP8 slot**, not Sage 3 directly.

### D.8.8 Composition

- Cross-ref **§D.4 FA-FP4** — both rely on `tcgen05.mma.kind::f4nvf4.block_scale`, and both face the same TMEM allocation pressure on B200. FA-FP4 uses a single-level $P$ scale (`s = rowmax(P)/6`); Sage 3 uses two-level $P$ scaling (`s_P1 × s_P2`). Empirically Sage 3's two-level is more accurate on naïve FP4 baselines but the runtime overhead is ~10%; FA-FP4 + Attn-QAT closes the quality gap with a fine-tune.
- Cross-ref **§D.10 MagiAttention's FFA_FA4** — the actively-maintained Blackwell-native FA4 fork with varlen support. While Sage 3's SM100 path is reverted, MagiAttention is the parallel "production B200 attention" option for distributed training.
- Cross-ref **§3.4 sparse attention** in parent report — SLA / SageSLA / SpargeAttn build on Sage 2++ kernels and would extend naturally to Sage 3 once SM100 is re-enabled.

---

## D.9 MagiAttention v1.1.0 + FFA_FA4 backend (Feb 28, 2026)

**References.** [SandAI-org/MagiAttention v1.1.0 release notes](https://github.com/SandAI-org/MagiAttention/releases) (Feb 28, 2026); CP benchmark blog [sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html); FA4 fork [demonatic/flash-attention magi_attn_blackwell_support](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support); MAGI-1 backbone [arXiv:2505.13211](https://arxiv.org/abs/2505.13211); Distributed Muon QK-Clip [arXiv:2507.20534](https://arxiv.org/abs/2507.20534).

### D.9.1 Motivation: distributed B200 attention with arbitrary masks

MagiAttention is a **distributed attention library for ultra-long-context training** of DiT-style and language models, originally designed for SandAI's MAGI-1 — a 24B autoregressive video DiT generating 1-second chunks of 24 frames each, supporting **up to 4M-token context**, with chunk-causal block-sparse attention as its core mask pattern [VERIFIED — MAGI-1 paper §2 + §2.2.1]. It is in production on H100 and *just received* Blackwell support in v1.1.0.

The v1.1.0 release (Feb 28, 2026) adds three things that close the gap between dense FA4 and *distributed varlen-mask FA4* on B200:

1. **`FFA_FA4` Blackwell kernel backend** — a fork of FA4 maintained at [demonatic/flash-attention:magi_attn_blackwell_support](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support). Enabled at runtime via `export MAGI_ATTENTION_FA4_BACKEND=1` [VERIFIED — release notes].
2. **Full-native group-collective kernels** built on **DeepEP** (DeepSeek's expert-parallel communication library), replacing the prior `all-to-all-v` path. Eliminates pre/post-processing copies and dtype casts, deduplicates internode volume. Enabled with `MAGI_ATTENTION_NATIVE_GRPCOLL=1`. **Requires IBGDA** (InfiniBand GPU-Direct Async).
3. **Distributed Muon QK-Clip support** ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534), the Kimi K2 / MoonshotAI training-instability fix). `flex_flash_attn_func` now returns `meta.max_logits` when called with `return_max_logits=True`, and `calc_attn` does a distributed reduction across context-parallel ranks to compute the global max-logits per attention head per training step.

### D.9.2 FFA_FA4: Flex-Flash-Attention with Blackwell support

The `FFA_FA4` kernel = **upstream FA4** + **Flex-Flash-Attention's arbitrary-mask surface** (the HSTU function-expression mask, similar in spirit to FlexAttention's `mask_mod`) + Blackwell-specific optimizations. The fork's enabled features (per the v1.1.0 PRs [#190, #206, #209, #225]):

- **`SwapAB`** in the FFA forward kernel — reduces `kBlockM` and reduces `wgmma`/`tcgen05.mma` waste under sparse attention. Specifically, when the mask predicate is mostly-False for a given $M$ block, swapping the M/N roles of A and B at the MMA level lets the kernel skip whole tiles instead of zeroing them out.
- **`PackGQA`** — gathers Q heads sharing a KV head in GQA settings. Wan 2.2 / Hunyuan are GQA models with `nh_q : nh_k : nh_v = 64 : 8 : 8` (parent report's MagiAttention CP benchmark settings — [VERIFIED]). PackGQA halves the attention compute by dispatching one `tcgen05.mma` per shared KV head.
- **`SparseLoad`** — uses `cp.async` (single-thread async copy from GMEM) instead of TMA for ultra-sparse global memory access patterns where TMA's index calculation overhead exceeds the load itself.
- **R2P (register-to-predicate) acceleration** for predicate-mask evaluation in HSTU expressions.
- **CSR mask compression** for ultra-long sequences — keeping the dense $N \times N$ block mask in shared memory becomes prohibitive at 1M+ tokens; CSR drops the on-chip footprint to O(*number of nonzero blocks*).
- **FFI-based kernel-launch acceleration** — direct CUDA C++ FFI from PyTorch, skipping the Python launch overhead which was an issue on the v1.0.5 release (see the v1.0.5 release-note caveat about a "severe bug" loading `.so` every call without caching).
- **Both kernel-level and distributed (8-rank context-parallel) Blackwell benchmarks** are published in the [CP benchmark blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html#for-b200) for full / causal / varlen-full / varlen-causal / sliding-window / **varlen block-causal** masks (the last is the MAGI-1-style mask). I.e., they cover exactly the masks a video DiT or LLM training stack needs.

### D.9.3 Distributed Muon QK-Clip integration

Background on Muon QK-Clip [VERIFIED — Kimi K2 paper, arXiv:2507.20534, §2.1]: in MoonshotAI's training of the 1T-parameter Kimi K2 with the Muon optimizer, the team observed exploding attention logits (training instability, more frequent with Muon than AdamW). They proposed **QK-Clip**, a post-update weight-rescaling rule:

For each attention head $h$, define the per-batch max-logit:

$$
S_{\max}^h = \frac{1}{\sqrt{d}} \max_{\mathbf{X} \in B} \max_{i,j} Q_i^h K_j^{h\top}.
$$

If $S_{\max}^h > \tau$ (some target threshold), rescale $W_q^h$ and $W_k^h$ by $\gamma_h = \min(1, \tau / S_{\max}^h)$:

$$
W_q^h \leftarrow \gamma_h^\alpha W_q^h,\qquad W_k^h \leftarrow \gamma_h^{1-\alpha} W_k^h,
$$

with $\alpha = 0.5$. This bounds the growth of attention logits *between* training steps without altering the current step's forward/backward. The Muon QK-Clip combination ("MuonClip") was used to pretrain Kimi K2 on 15.5 trillion tokens **without a single loss spike**.

For *distributed* training (context-parallel + tensor-parallel), the per-head max-logit must be reduced across CP ranks. MagiAttention v1.1.0 wires this in: the FFA forward kernel returns `meta.max_logits` (a per-head, per-batch FP32 scalar) when `return_max_logits=True`, and `calc_attn` does a `dist.all_reduce(max=op)` over the CP group inside the call. The application code reads `meta.max_logits` post-step and applies the QK-Clip rescaling to its sharded $W_q, W_k$ tensors.

For MAGI-1 specifically — a 24B autoregressive video DiT trained on 4M-token contexts — the joint Muon + QK-Clip optimizer is a critical training-stability tool, and MagiAttention's distributed version is the wire-up needed for SP/CP'd training.

### D.9.4 DeepEP-based group collectives

The v1.0.5 (experimental) and v1.1.0 (production) implementation of intra/internode group collectives uses **[deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)** (the expert-parallel communication library originally for DeepSeek-V3). For DiT-style attention, the gather/scatter pattern in `flex_flash_attn_func`'s pre/post-attention shuffle can be expressed as a *group all-to-all* — different ranks have different mask shapes, so the "everybody sends one block to everybody" pattern degenerates to an irregular collective. DeepEP gives:

- **Eliminates pre/post-processing copies** — the `all-to-all-v` path requires repacking buffers before/after the collective. Native DeepEP collectives consume directly from the producer's shared memory.
- **Eliminates dtype-cast overhead** — the prior path materialized BF16 → FP32 buffers for the collective. DeepEP transfers BF16 directly.
- **Internode volume deduplication** — when multiple ranks on the same node need the same remote tile, DeepEP's pull-pattern fetches it once and broadcasts intra-node.

Installation requires **IBGDA** (InfiniBand GPU-Direct Async, available in Mellanox OFED ≥5.6 + NVIDIA HCA driver ≥35.x). On NVL72 / GB200 the equivalent is NVLINK SHARP (NVLS), see §1.3 of the parent report.

### D.9.5 Reported numbers (CP benchmark)

The [CP benchmark blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html) reports MagiAttention vs Megatron-LM context-parallel baselines on H100 and B200, GQA-64-8-8, hd=128, BF16, varlen samples ≤ ¼ of total seq. CP size scales 8 → 64 with per-device seq fixed at 8K (H100) or 16K (B200). Baselines: Ulysses, Ring P2P, Ring AllGather, USP, LoongTrain, Megatron HybridCP. **On B200, baselines use cuDNN 9.21 + 2-CTA MMA** because FA4 2.8.3 lacks robust varlen support; MagiAttention uses FFA_FA4. The benchmark figures are not text-extractable, but the qualitative result reproduced from the blog is:

- **Kernel-level on B200**: FFA_FA4 matches or exceeds FA4 baseline on full/causal masks (where FA4 has solid support); on varlen-causal and varlen-block-causal, FFA_FA4 substantially exceeds the cuDNN 9.21 baseline because FA4's varlen path is missing entirely from upstream.
- **Distributed-level on B200, 64-rank CP**: MagiAttention with native DeepEP collectives is `~2×` faster than `magi_attn-a2av` (its own AlltoAll-v-based path), and `1.5–2×` faster than the best Megatron baseline (Ring P2P) on `varlen` masks. The Ulysses path is competitive on `full` mask but degrades on varlen.

### D.9.6 Limitations

- **FFA_FA4 is a fork — not upstream FA4.** Maintenance happens in MagiAttention's contributor base; commits do not flow back to Dao-AILab/flash-attention. As of May 2026 the fork is `~2 months` behind FA4 main.
- **DeepEP IBGDA dependency** is a setup hurdle for many sites; the AlltoAll-v fallback adds ~30–40% overhead.
- **Beta caveats** in the CP benchmark blog ("experimental features ... may not be enabled by default or fully ready for production use yet").
- **No Sage / FP4 path.** FFA_FA4 is BF16; for FP4 attention on B200 you still need FA-FP4 or (eventually) Sage 3.

### D.9.7 Composition

- Cross-ref **§D.3 FA4** — FFA_FA4 is a downstream fork; cherry-picks from upstream FA4 with `varlen` and HSTU-mask additions.
- Cross-ref **§D.4 FA-FP4** — both are Blackwell-native; FA-FP4 is FP4 / single-rank, FFA_FA4 is BF16 / multi-rank. They compose orthogonally (FFA_FA4 for the distributed wrapping + FA-FP4 for per-rank FP4 attention is, in principle, the best of both worlds, though no integrated kernel exists).
- Cross-ref **§D.8 Sage 3** — MagiAttention is the parallel "production B200 distributed attention" option *while Sage 3's SM100 path is reverted* (see issue #322 caveat in §D.8.7). The FastVideo / SGLang Diffusion / vLLM-Omni / LightX2V stack all expose MagiAttention as one of their attention backends per the parent report's table (§1.7).
- Cross-ref **§5 step distillation** in parent report — Distributed Muon QK-Clip is a *training* trick, so most distillation pipelines (rCM, Phased DMD, FastWan) inherit it via standard Muon optimizers without any extra code.

---

## D.10 DistFlashAttn / LightSeq

**References.** Li, Shao, Xie, Xing, Ma, Stoica, Gonzalez, Zhang, [arXiv:2310.03294](https://arxiv.org/abs/2310.03294), COLM 2024; code [RulinShao/LightSeq](https://github.com/RulinShao/LightSeq) (221 stars [VERIFIED]).

### D.10.1 Motivation

DistFlashAttn (the kernel) and LightSeq (the framework wrapping it) target *causal* long-context LLM training (Llama-7B at 32K → 512K seq) — a different distribution from video DiT, but the *technical primitives* it introduces are foundational for the ring/CP variants used in the parent report's §1.

Three technical observations:

- **Ring Attention's causal-mask load imbalance.** In a $P$-rank Ring with causal masking, the first iteration is fully imbalanced (worker 0 has trivial work; worker $P-1$ has full lower-triangle work), and asymptotically the *idle fraction* is $\frac{P^2 - P}{2P^2} \to \frac{1}{2}$ as $P \to \infty$. Half of the workers spend half of their time idle. This is a kernel-level critique; striped attention (§D.13) is one prior attempt to fix it via permutation.
- **Ring KV communication on the critical path.** Each iteration of Ring requires P2P forward of the KV block; without overlap, this comm dominates as $P$ grows.
- **Gradient-checkpointing redundancy.** The FlashAttention backward kernel internally recomputes softmax; HuggingFace-style outer-layer gradient checkpointing then recomputes the *whole* forward pass. So FA's softmax recompute happens *twice* — once internally, once via HF gradient checkpointing.

### D.10.2 Three contributions

#### (1) Token-level workload balancing for causal masking

Instead of every worker computing only its own causal block, DistFlashAttn lets workers with light loads (early-token workers) "help" workers with heavy loads (late-token workers) by computing their attention shards in parallel and shipping the partial $(O, m, \ell)$ back to the original worker. The original worker then `rescale`'s the partials with its own using the standard online-softmax merge.

The expected idle fraction becomes:

$$
X = \begin{cases} 0 & P\text{ odd} \\ \frac{1}{2P} & P\text{ even} \end{cases}
$$

I.e., *zero idle when $P$ is odd, $O(1/P)$ otherwise*. Empirically this **doubles throughput vs unbalanced Ring**.

#### (2) KV-comm/compute overlap

Two CUDA streams: one for `attn(q, k_r, v_r, s)`, one for the P2P fetch of `(k_{r+1}, v_{r+1})` from the next-rank neighbor. Standard double-buffering, ~`1.32×` end-to-end speedup vs unoverlapped.

#### (3) Rematerialization-aware gradient checkpointing

Instead of checkpointing at *Transformer-layer* boundaries (which forces FA forward recompute during backward), checkpoint at the *FA output* boundary. The FA backward then directly uses the saved $(O, lse)$ pair without redoing the forward kernel. **`1.31×` speedup** with **zero numerical difference** (vs the HuggingFace policy of recomputing).

### D.10.3 Reported numbers

| vs. baseline | DistFlashAttn speedup | Sequence support |
|---|---|---|
| Ring Self-Attention | **4.45–5.64×** | 8× longer |
| Megatron-LM + FlashAttn | **1.24–2.01×** | 2–8× longer |
| Ring Attention | **1.67×** | same |
| DeepSpeed-Ulysses | **1.26–1.88×** | same |

Setup: Llama-7B and -GQA-7B variants, A100 80GB cluster (intra-NVLink + inter-Infiniband), seq 32K → 512K [VERIFIED — DistFlashAttn paper §4 + Table 4].

### D.10.4 Limitations and relevance to video DiT

- **Causal-mask specific.** Video DiTs are *bidirectional* — the visual tokens have no causal structure — so DistFlashAttn's load-balancing trick provides no benefit. The KV-comm/compute overlap and the rematerialization-aware checkpointing **do** transfer.
- **No Blackwell port.** As of May 2026 the LightSeq codebase is on FA2 (sm_80) — no FA4/sm_100a path.

### D.10.5 Composition

DistFlashAttn's load-balancing technique is the *causal* counterpart to **§D.13 Striped Attention**'s causal balancing. Both are subsumed by the modern `zhuzilin/ring-flash-attention`'s `zigzag_ring_flash_attn_func` for causal LLMs and are *not used* in bidirectional video-DiT ring attention. The KV-comm/compute overlap is universal — present in every Ring CP implementation in the parent report's §1 table.

---

## D.11 BurstAttention / BurstEngine

**References.** Sun, Zhao, Han, Yang, Liu, Shi, Sun, [arXiv:2403.09347](https://arxiv.org/abs/2403.09347) (BurstAttention, Mar 2024); [arXiv:2509.19836](https://arxiv.org/abs/2509.19836) (BurstEngine, Sep 2025, SC'25); [thunlp/BurstEngine](https://github.com/thunlp/BurstEngine).

### D.11.1 BurstAttention (kernel)

BurstAttention starts from RingAttention but introduces a two-level partitioning: **inter-device partition** (along the sequence into $G$ chunks for $G$ GPUs, ring-passed) **+ intra-device partition** (FlashAttention-style block tiling within each GPU's local computation). The forward pass is the standard online-softmax recurrence (Algorithm 1 in the paper), with the local $\widetilde{P}_{ij} = \exp(S_{ij} - \mathrm{LSE}(S_{ij}))$ following FlashAttention's local-softmax pattern, and the cross-device aggregation following Ring's online-softmax recurrence.

The differentiator vs vanilla Ring is the "double-buffer" overlap: comms are placed on a separate stream and double-buffered against the inner FA tile loop. Reported result: **40% lower communication overhead and 1.37× speedup vs Megatron-V3 + FA at 128K seq on 32×A100** [VERIFIED — BurstAttention paper §3 + Abstract].

### D.11.2 BurstEngine (the framework around BurstAttention)

BurstEngine adds four optimizations to extend BurstAttention to multi-million-token sequences:

- **Sequence-level selective checkpointing.** Instead of checkpointing the entire sequence's activations, checkpoint only the *first half*; recompute the first-half during backward. Trades a small amount of compute for 50% memory reduction at the activation level.
- **Fused LM-head + cross-entropy.** The LM head's $\mathbf{X} \mathbf{W}_\text{vocab}^\top$ materializes a $(N, v)$ tensor (with $v$ ~150K vocab); for $N$=1M tokens this is 150B FP32 entries = 600 GB of memory just for the *pre-softmax* logits. Fusing the matmul with the cross-entropy loss computes only the per-row argmax + selected-token logit, skipping the full $(N, v)$ materialization. Eliminates a major memory bottleneck for long-context LLMs and applies *equivalently* to long-video-DiT noise-prediction outputs (which similarly produce a per-token output that is consumed by a downstream loss).
- **Backward-direction comm optimization.** Re-orders the backward Ring iteration so the comm of $\nabla K$, $\nabla V$ (the gradient counterparts of K, V) happens *while* the local backward MMA runs, instead of serially.
- **Topology-aware ring** — splits the ring into intra-node (NVLink) and inter-node (Infiniband) sub-rings so high-bandwidth links carry the bulk of the volume.

### D.11.3 Reported numbers

| Setting | Speedup | Memory reduction |
|---|---|---|
| BurstEngine vs strongest baseline at 1M+ tokens | **1.2×** | **−26.4%** vs most memory-efficient prior |
| Linear scaling to **4M tokens on 64×A800** | yes | yes |
| BurstAttention vs Megatron-V3 + FA, 128K @ 32×A100 | **1.37×** | **−40% comm volume** |
| Mask support | causal, sliding-window, block-sparse | — |

*[VERIFIED — BurstEngine paper §1, §4 + BurstAttention paper Abstract.]*

### D.11.4 Relevance to video DiT

- Video DiTs at 720p / 81 frames hit ~100K context tokens (HunyuanVideo). Wan 2.2 / LTX-2 hit similar. BurstEngine's optimizations are designed for ≥1M tokens — not currently *needed* for video DiT inference, but for *training* longer sequences (e.g., 4-second + 2160p video at 4M tokens) it would be the natural ring-attention framework.
- **Selective checkpointing.** Directly applicable to long-video-DiT training (where activations dominate memory). Even *inference* might benefit if the MAGI-1-style chunked autoregressive scheme gets longer chunks.
- **Fused LM-head + CE.** Not directly applicable to video DiT (no large vocabulary), but the *pattern* (fuse the loss/output projection with the loss to avoid materializing intermediates) is what FastVideo's `LossFunctionalLayer` does for the noise-prediction loss.

### D.11.5 Composition

Cross-ref **§D.10 DistFlashAttn** (overlapping technique parallels DistFlashAttn's). Cross-ref **§1.3 Ring Attention** in parent report.

---

## D.12 Striped Attention (causal load balancing on Ring)

**References.** Brandon, Nrusimha, Li, Gonzalez, Chen, Mialon, Ré, [arXiv:2311.09431](https://arxiv.org/abs/2311.09431) (Nov 2023); code [exists-forall/striped_attention](https://github.com/exists-forall/striped_attention).

### D.12.1 Motivation

Same observation as DistFlashAttn (§D.10.1) but a different solution. In Ring Attention with causal masking, the per-iteration KV-block ↔ Q-block overlap pattern is highly imbalanced, and the per-iteration latency is dominated by the maximally-loaded worker.

### D.12.2 Mechanism

Instead of giving each worker a contiguous chunk of tokens (positions $[pP, (p+1)P)$ on worker $p$), **stripe** the tokens: worker $p$ gets positions $\{p, p+P, p+2P, \dots\}$. This is a *uniformly distributed strided subset* of the sequence.

The key property: with striping, *every* Q-K interaction across all ring iterations has the property that approximately half the entries are masked (above diagonal) and half are unmasked. So every worker has a roughly equal workload at every iteration.

Mathematically, attention is *permutation-equivariant*: permuting the tokens before attention and inverse-permuting after produces the same result, so striping is a free transform that doesn't change the loss. The cost is the (very fast) tensor permutation at the boundaries.

### D.12.3 Reported numbers

| Setup | Striped vs Ring speedup |
|---|---|
| 8×A100 80GB, billion-parameter LM, **256K seq** | **1.45×** |
| 16×TPU v4, **786K seq** | **1.65×** |

*[VERIFIED — Striped Attention paper §1 + §4.]*

### D.12.4 Relevance to video DiT

**Video DiTs are bidirectional** — there is no causal mask on the visual tokens — so Striped Attention's specific win does not apply. Two niche scenarios where it *would* apply:

- **Causal-temporal video models** (VideoPoet, LTX-2 with causal attention, MAGI-1's chunk-causal mask): Striped's causal balancing is the load-balanced ring variant. MagiAttention v1.1.0 (§D.9) doesn't use Striped specifically but its `varlen-block-causal` mask variants get the same effect via PackGQA + workload-aware tile schedulers.
- **Cross-attention from a causal text decoder over visual tokens**: rare but not nonexistent.

In the parent report's §1, Striped Attention is **referenced as a building block** in `zhuzilin/ring-flash-attention`'s `stripe_flash_attn_func` API. SGLang Diffusion and xDiT do not expose Striped to bidirectional video DiT.

### D.12.5 Composition

Cross-ref **§D.10 DistFlashAttn** (parallel solution to the same causal-imbalance problem). Cross-ref **§1.4 zhuzilin/ring-flash-attention** in parent report.

---

## D.13 Tree Attention (log-depth tree reduction; energy-function view)

**References.** Shyam, Pilault, Shepperd, Anthony, Millidge, [arXiv:2408.04093](https://arxiv.org/abs/2408.04093) (Aug 2024); code [Zyphra/tree_attention](https://github.com/Zyphra/tree_attention).

### D.13.1 Motivation: Ring's $O(P)$ communication scaling

Ring Attention has an $O(P)$ comm pattern — at each of $P$ iterations every rank sends one KV block to its neighbor. This linear comm scaling becomes the bottleneck for very long contexts on many devices.

Tree Attention's contribution is observing that **attention can be expressed as the gradient of a scalar energy function**, and that the energy function only requires associative reductions (`logsumexp`, `max`) over the sequence — both of which can be done in **$O(\log P)$ steps via tree reduction**.

### D.13.2 Energy-function derivation

Let $\zeta$ be an auxiliary "source" vector. Define the energy:

$$
F(\zeta) = \log \sum_{a=1}^{N} \exp\big(q \cdot k_a^\top + \zeta \cdot v_a^\top\big).
$$

Then the standard attention output for query $q$ over keys/values $\{k_a, v_a\}$ is the gradient of $F$ at $\zeta = 0$:

$$
\sum_{a=1}^{N} \mathrm{softmax}(q \cdot k_a^\top) v_a = \frac{\partial F}{\partial \zeta}\bigg|_{\zeta=0}.
$$

This identity *relates attention to a single scalar reduction*. The reduction is an associative `logsumexp` (and `max` for numerical stability), so it can be computed in $\log_2(P)$ steps via a tree allreduce of $(\max, \mathrm{logsumexp})$ pairs.

Algorithm sketch (from the paper §5.1):

```
1. Each rank p computes its local r_p = q · k_{p,:}^T + ζ · v_{p,:}^T   [in parallel]
2. Compute m = Reduce(max, {r_p})                                        [O(log P) tree]
3. Scatter m; each rank does r_p ← r_p − m                                [for stability]
4. Compute lse = Reduce(logsumexp, {r_p})                                [O(log P) tree]
5. Each rank computes its local gradient w.r.t. ζ; aggregate via tree.   [O(log P)]
```

The gradient computation (step 5) is just an `exp(r_p − lse) · v_p` per-rank product followed by a sum — the sum is again a tree reduction.

### D.13.3 Reported numbers

| Setting | Tree vs Ring | Memory |
|---|---|---|
| Decode on Llama-3.1-8B (8×H100) | **up to 4× faster** | **2× less peak** |
| Long-context decode generally | **up to 8×** | **2× less peak** |
| Hardware tested | H100 DGX, MI300x DGX, PCIe-connected RTX 4090s | — |

*[VERIFIED — Tree Attention paper §1 + Abstract.]*

### D.13.4 Relevance to video DiT

**Limited direct relevance.** Tree Attention is designed for *decoding* (one query attending to a long KV cache), where the asymmetric cost structure favors $\log P$ reductions. Video DiTs are bidirectional and don't autoregressively decode (unless you're MAGI-1 or VideoPoet, which are autoregressive over chunks). For prefill-only workloads with $Q, K, V$ of equal size, Tree Attention's win is muted.

What *does* transfer is the **energy-function formalism**: the parent report §1.3's "Ring Attention with online-softmax merge" is exactly the same merge that Tree Attention uses; Tree just wires it into a $\log P$-depth tree instead of a $P$-depth ring. Future async-Ring variants that want sub-linear comm scaling (when $P$ is very large, like full GB200 NVL72 = 72 GPUs) can adopt the Tree Attention reduction pattern.

### D.13.5 Composition

Cross-ref **§1.3 Ring Attention** (parent report). Cross-ref **§D.11 BurstEngine** (which uses topology-aware ring rather than tree; the two are alternative scaling strategies).

---

## D.14 cuDNN 9.21 SDPA on Blackwell

**References.** [cuDNN 9.21 release notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.21.0/release-notes.html); [cuDNN frontend Attention v1.22 docs](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.22.0/operations/Attention.html). The 9.13 → 9.14 line backported FA4's conditional rescale and FMA `exp` emulation; 9.19 → 9.21 incorporated 2-CTA MMA, MXFP8, MLA fused attention, and the new `DiagonalBandMaskOperation` and `SoftmaxOperation` graph ops.

### D.14.1 The cuDNN 9.21 deltas (from release notes [VERIFIED])

- **MXFP8 SDPA on Blackwell.** Forward + backward, GQA, MLA (d_qk=192, d_v=128), embedding dim d=64. *No MXFP4 / NVFP4 attention support yet* — this is the gap that FA-FP4 / Sage 3 fill.
- **2-CTA MMA SDPA forward / backward.** Same primitive as FA4 §D.3.5, halves operand-B SMEM traffic.
- **`dQ` matmul reordering.** Same as FA4's backward dQ via DSMEM.
- **MLA fused attention with 2-CTA MMA.**
- **New graph ops `DiagonalBandMaskOperation` and `SoftmaxOperation`.** The former lets users express sliding/causal-window masks at the graph level (essentially a parametric, rank-1 mask operator). The latter exposes the rescale threshold and `ex2` emulation as graph parameters — i.e., `CUDNN_RESCALE_THRESHOLD` and `CUDNN_USE_EX2_EMULATION` (default values 8 and 1, identical to FA4) are now first-class graph attributes.
- **MuonClip support** (per release notes for cuDNN 9.20+): cuDNN's SDPA outputs the per-batch×head×seq max-attention-score tensor when configured with `output_max_logits=true`, providing the same data MagiAttention's `meta.max_logits` does.
- **Sink-token support stabilized.** Sink attention forward (an attention-sink token at position 0 that all queries attend to) was previously buggy on Blackwell; 9.20+ fixes it.

### D.14.2 cuDNN frontend `Attention v1.22` API surface

The frontend exposes a Python and C++ graph-builder API: `cudnn_frontend.SDPA_attributes`. Key knobs:

```python
cudnn.sdpa(
    q, k, v,
    attn_scale=1/sqrt(d),                       # FP scalar or tensor
    is_causal=True,
    paged_attention_k_table=None,               # optional, see PagedAttention
    paged_attention_v_table=None,
    paged_attention_max_seq_len_kv=None,
    generate_stats=True,                        # output softmax LSE for backward
    diagonal_band_mask_lower=-window_size,      # new in 9.21
    diagonal_band_mask_upper=0,                 # for sliding causal
    softmax_rescale_threshold=8,                # FA4 conditional rescale
    softmax_use_ex2_emulation=True,             # FA4 FMA-emulated exp
)
```

For the FE OSS API there is also a CuTe-DSL backward path for `d=256` on SM100+ (`SDPA Backward FE OSS`), which is roughly the same primitive FA4 ships but exposed through cuDNN's operator graph.

### D.14.3 Hardware coverage matrix (cuDNN frontend Attention v1.22)

Per the documentation page, B200/B300 SDPA support breaks down as:

- **Prefill**: head dims to **256** for FP16/BF16/FP8.
- **Decode**: head dims to **128** for FP16/BF16/FP8.
- **Bprop**: head dims to **128** for FP16/BF16/FP8 (192 KQ / 128 V for GQA / MLA).

I.e., prefill heads run higher than decode/bprop — same constraint as FA4.

### D.14.4 When to prefer cuDNN 9.21 over FA4 on B200

- **Mixed precision / non-standard masks.** cuDNN 9.21 has MXFP8 forward/backward and a richer graph mask surface (sliding/causal-window via `DiagonalBandMaskOperation`, sink tokens, paged caches).
- **Production support contracts.** cuDNN ships with NVIDIA OEM / DGX support; FA4 does not.
- **PyTorch fallback.** `torch.nn.functional.scaled_dot_product_attention` dispatches to cuDNN when no faster backend is registered. Most prebuilt PyTorch wheels on B200 do *not* ship FA4 (it's a separate `pip install flash-attn-4`).
- **GQA / MLA.** cuDNN's MLA fused attention with d_qk=192, d_v=128 (DeepSeek-V3 shape) is well-tested; FA4 supports this too but Sage 3 / FA-FP4 do not.

For a B200 video-DiT inference rig the practical priority is:

1. **FA4 (BF16)** for the dense fast path on standard hd=128 attention.
2. **FA-FP4 (NVFP4)** for the FP4 fast path when the QKV linear is FP4.
3. **cuDNN 9.19+** for sliding/causal-window masks, MXFP8, paged caches, MLA shapes, and as the default `torch.compile` SDPA backend.
4. **FA3** as the *Hopper* fallback when running on H100/H200 (Wan 2.2 / Hunyuan production stacks).
5. **MagiAttention FFA_FA4** when distributed varlen masks are needed.

### D.14.5 Composition

cuDNN 9.21 is the **NVIDIA-supported, production-grade** alternative to FA4. The two are converging — FA4 → cuDNN ideas are flowing both directions. Cross-ref **§D.3 FA4** (where the parent paper documents the cuDNN backporting). Cross-ref **§D.9 MagiAttention** (which uses cuDNN as the B200 baseline for its CP benchmark since FA4 lacks varlen).

---

## D.15 FlexAttention (PyTorch programmable kernel)

**References.** [PyTorch blog](https://pytorch.org/blog/flexattention/), [arXiv:2412.05496](https://arxiv.org/abs/2412.05496) (Dong, Feng, Guessous, Liang, He, NeurIPS 2024 EMC2 workshop); [PT 2.12 docs](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html).

### D.15.1 Motivation

FA2/3/4 + cuDNN are blazing fast for the standard SDPA primitive but inflexible: every variant (ALiBi, sliding window, document masking, soft-capping, RoPE-as-score-mod, ...) requires a new hand-written kernel. Worse, *combinations* of variants (sliding window + document masking + tanh soft-capping + causal) are hyperexponential in count, and most never get a hand-written kernel.

FlexAttention's design: write attention variants in idiomatic PyTorch as a `score_mod` function that takes `(score, batch, head, q_idx, kv_idx)` and returns a modified score; FlexAttention compiles this through `torch.compile` into a fused FlashAttention-style Triton (or FA4 via the `BACKEND="FLASH"` knob in PT 2.12) kernel that materializes nothing extra.

### D.15.2 Programming model

```python
def alibi(score, b, h, q_idx, kv_idx):
    bias = alibi_bias[h] * (kv_idx - q_idx)
    return score + bias

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
out = flex_attention(q, k, v, score_mod=alibi, block_mask=block_mask)
```

`score_mod` is for *score adjustments* (relative bias, ALiBi, soft-capping, RoPE-as-score-mod). `mask_mod` (without the `score` argument) is for *masking*; the resulting `BlockMask` is consumed by the kernel to skip fully-masked tile pairs entirely (so causal masking gets a free 2× from skipped blocks, sliding-window gets `1 - window/seq` skip ratio, etc.). The default block size is 128.

### D.15.3 PT 2.12 additions (relevant to video DiT)

- **`BlockMask` as a pytree.** Survives `torch.compile` and CUDA-graph capture, so the same compiled FlexAttention kernel can be called from a captured CUDA graph with different masks.
- **`BlockMask.from_kv_blocks(...)`.** Lets users construct an arbitrary precomputed sparsity pattern from a (Q_block_id, KV_block_id) mask matrix. This is the natural lowering for **Sliding Tile Attention (STA), Radial Attention, document masking**, and any video-DiT 3D-window mask that's a function of `(t, h, w)` coordinates.
- **`BACKEND="FLASH"` flag.** Dispatches to FA4 kernels when available on B200, falling back to Triton on other GPUs. This is how FlexAttention gets the FA4-level performance on Blackwell — see the parent report's table from the Lambda blog: dense/causal **1.6–3.2× fwd, 1.85–2.3× bwd vs Triton** on GB200 NVL72.
- **`forward_only=True` mode.** Inference-only path that skips backward bookkeeping; saves ~10% latency.

### D.15.4 Why FlexAttention matters for video DiT

- **3D-window masks (STA / SSTA, Radial Attention).** STA's mask is *purely a function of (t, h, w) coordinates*; FlexAttention `mask_mod` is the natural lowering. Radial Attention's analytic mask `f(distance(q, k))` is similarly clean. As of May 2026 only the H100 STA kernel exists in FastVideo; a Blackwell STA via FlexAttention `BACKEND="FLASH"` is the open path.
- **Score-mod-based 3D RoPE.** Some video DiTs use a non-standard 3D RoPE (e.g., MAGI-1's per-(t,h,w) rotary). FlexAttention can apply this in-kernel without a separate Q,K rotation pass.
- **Online masks.** Per-frame guidance, dynamic CFG dropout, and other "compute mask at runtime" patterns are easy in FlexAttention and impossible in upstream FA4.

### D.15.5 Performance vs hand-tuned

Per the FlexAttention blog and the FA4 paper §5: FlexAttention typically lags hand-tuned FA4/Sage by **~15–20% on dense workloads** (because of `torch.compile`'s less-aggressive block-size tuning vs hand-written kernels), but is the *only* path for many sparsity patterns. For the FA4 backend on GB200, the gap shrinks because the inner kernel is FA4 itself.

### D.15.6 Composition

- Cross-ref **§D.3 FA4**, **§D.4 FA-FP4** — FlexAttention dispatches to these via `BACKEND="FLASH"`.
- Cross-ref **§3.1 FA4** in parent report (sparse-mask dispatch via FlexAttention).
- Cross-ref **§3.4 STA / Radial / SVG2** in parent report — all are natural FlexAttention `mask_mod` consumers; FlexAttention is the recommended Blackwell port path for each.
- Cross-ref **§D.9 MagiAttention** — Magi uses its own HSTU function-expression mask language rather than FlexAttention; the two are alternative DSLs for arbitrary masks.

---

## D.16 Composition with the rest of the stack

The eight items above span four orthogonal axes: (precision, mask-flexibility, single-vs-distributed, Hopper-vs-Blackwell). The full composition matrix for a B200 video DiT pipeline:

### D.16.1 Per-step kernel selection (single-rank B200 inference)

| Timestep slice | Attention precision | Recommended kernel | Fallback |
|---|---|---|---|
| First 3 timesteps (high-noise) | **FP8** (Sage 2++) | Sage 2++ via `qattn.qk_int8_sv_f8_accum_f16_*` | FA4 BF16 |
| Middle timesteps (~50–80% of total) | **NVFP4** | FA-FP4 on hd=128 standard masks | Sage 3 (when SM100 path returns) → FA4 BF16 |
| Last 3 timesteps (low-noise, fine details) | **FP8** | Sage 2++ | FA4 BF16 |
| Sliding-window masks (STA, Radial, SVG2) | **BF16** | FlexAttention with `BACKEND="FLASH"` (FA4 backend) | Triton FlexAttention |
| Cross-attention (text → visual) | **BF16** | cuDNN 9.21 SDPA with `DiagonalBandMaskOperation` if windowed; else FA4 | FA3 if on Hopper |

This is the kernel-picker matrix the parent report's §3 stack assumes. FastVideo, SGLang Diffusion, and vLLM-Omni all expose this via configurable per-step backends.

### D.16.2 Distributed (multi-rank) wrapping

| SP/CP type | Inner attention kernel | Comm pattern | Library |
|---|---|---|---|
| Ulysses (head shard) | FA4 / cuDNN / Sage 2++ / FA-FP4 | 4× a2a around attention | xDiT, FastVideo, ParaAttention |
| Ring (seq shard) | Triton-implemented FA inner loop with `(O, lse)` outputs | P2P + online-softmax merge | zhuzilin/ring-flash-attention |
| USP (Ulysses + Ring) | Same as Ring | a2a + P2P | xDiT, SGLang Diffusion |
| **MagiAttention CP** (varlen / block-causal) | FFA_FA4 (B200) / FFA + FA3 (H100) | DeepEP group collectives | MagiAttention 1.1.0 |

### D.16.3 Cross-cutting integration points

- **§2 NVFP4 GEMM.** FA-FP4 and Sage 3 (when SM100 returns) consume the FP4 outputs of the QKV/O linear without a BF16 round-trip, eliminating ~2× of memory bandwidth on the attention boundaries.
- **§3.4 sparse attention.** STA / VSA / Radial / SVG2 layer their masks on top of FA4 (via FlexAttention `BACKEND="FLASH"`) or on top of Sage 2++ kernels (via SpargeAttn / SLA / SageSLA), depending on whether the sparsity is coordinate-static (STA, Radial) or activation-dynamic (VSA, MoBA).
- **§5 step distillation.** Distilled student models (rCM, FastWan, lightx2v Lightning, Phased DMD) inherit the kernel selection from the teacher; the only adjustment is that distilled models need 3–8 NFE × 1 forward pass per NFE = far fewer attention calls per video, so the absolute attention budget shrinks ~6–12×. The **per-step** kernel choice doesn't change.
- **§1.2 Ring Attention (parent).** The cross-rank `(O, lse)` interface used by ring is identical for FA3, FA4, FA-FP4, Sage 2++, and Sage 3 (when re-enabled). Today the production `zhuzilin/ring-flash-attention` is FA2-based; the natural Blackwell upgrade is a Helion or CuTe-DSL re-implementation that exposes the FA4 inner loop with explicit `(O, lse)` per-iteration outputs (open as of May 2026).
- **§1.5 pipeline parallelism (DualParal, PipeDiT, VINs).** Orthogonal to attention-kernel choice; PipeDiT's "attention co-processing" (split attention across GPU groups by per-head workload) is a *scheduling* layer that wraps any of the kernels above.
- **§6 Muon QK-Clip** in training. Sage 2++ / FA-FP4 / FA4 all are gradient-stable enough that QK-Clip is rarely needed for *finetuning* of pretrained DiTs, but for *pretraining* a 24B autoregressive video DiT (like MAGI-1) it's a critical training-stability ingredient — and MagiAttention's `meta.max_logits` is the wired-in support.

### D.16.4 Open gaps as of May 2026

| Gap | Status |
|---|---|
| **Sage 3 SM100 (B200) path** | reverted late 2025; not re-enabled as of May 2026 [VERIFIED — issue #322 closed Jan 1 with no resolution] |
| **FA-FP4 backward** | not published — inference-only kernel |
| **FA-FP4 + varlen + block-sparse mask** | not in upstream FA-FP4 |
| **MagiAttention + FA-FP4** | no integrated kernel |
| **cuDNN 9.21 NVFP4 attention** | not yet (only MXFP8) |
| **Ring Attention with FA4 inner loop exposing `(O, lse)`** | open — `zhuzilin/ring-flash-attention` lags upstream FA4 |
| **STA Blackwell port** | sm_90a only; FlexAttention + BlockMask is the open path |
| **SLA / SageSLA on Sage 3** | open (Sage 3's SM100 revert blocks) |

### D.16.5 Recommendation tree (picking a kernel)

```
B200 inference, video DiT?
├── FP4 weights (NVFP4 QKV/O linear)?
│   ├── Standard attention (no special mask)?
│   │   └── FA-FP4 (mid-timesteps) + Sage 2++ (first/last timesteps)
│   └── Sliding-tile or Radial mask?
│       └── FlexAttention with `BACKEND="FLASH"` + custom mask_mod  (BF16, FA4 backend)
├── BF16 weights, standard attention?
│   ├── No special mask?
│   │   └── FA4 BF16  (`pip install flash-attn-4`)
│   ├── Sliding window or causal-window mask?
│   │   ├── cuDNN 9.21 with `DiagonalBandMaskOperation`
│   │   └── or FlexAttention with `BACKEND="FLASH"`
│   └── Block-sparse / varlen across context-parallel ranks?
│       └── MagiAttention v1.1.0 with FFA_FA4 backend
└── Hopper (H100/H200, not Blackwell)?
    ├── BF16 → FA3 (Sage 2++ kernels co-installed for FP8)
    └── FP8 → Sage 2++ for first/last steps; FA3-FP8 elsewhere
```

For a B200 inference rig running Wan 2.2, LTX-2, HunyuanVideo, CogVideoX, Mochi, Step-Video, the practical default is:

```bash
pip install flash-attn-4                                  # FA4 BF16
pip install git+https://github.com/hao-ai-lab/flash-attention-fp4   # FA-FP4
pip install git+https://github.com/thu-ml/SageAttention   # Sage 2++ (FP8) and Sage 3 (FP4, SM120 only as of May 2026)
pip install git+https://github.com/SandAI-org/MagiAttention   # MagiAttention CP / FFA_FA4
# cuDNN 9.21+ comes with the CUDA Toolkit / PyTorch nightly
```

with backend selection via the diffusers / FastVideo / SGLang Diffusion / vLLM-Omni / LightX2V-style `attn_backend` config knob, defaulting to **FA4 BF16** for distillation-distilled models (3–8 NFE) and switching to **FA-FP4 + Sage 2++** for full-NFE 40–50 step pipelines that benefit more from quantized attention.

---

## Appendix: References (cross-checked against parent report style)

Foundational attention kernels:

- FlashAttention-1, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
- FlashAttention-2, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (background only here).
- FlashAttention-3, [arXiv:2407.08608](https://arxiv.org/abs/2407.08608); [PyTorch blog](https://pytorch.org/blog/flashattention-3/), NeurIPS 2024.
- FlashAttention-4 (Blackwell), [arXiv:2603.05451](https://arxiv.org/html/2603.05451v1) (Mar 2026); [Together blog](https://www.together.ai/blog/flashattention-4); [Tri Dao blog](https://tridao.me/blog/2026/flash4/); [Lambda blog](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet); [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).
- FA-FP4 (hao-ai-lab), [hao-ai-lab/flash-attention-fp4](https://github.com/hao-ai-lab/flash-attention-fp4).

SageAttention family (THU-ML):

- SageAttention 1, [arXiv:2410.02367](https://arxiv.org/abs/2410.02367), ICLR 2025.
- SageAttention 2, [arXiv:2411.10958](https://arxiv.org/abs/2411.10958), ICML 2025.
- SageAttention 2++, [arXiv:2505.21136](https://arxiv.org/abs/2505.21136), ICML 2025.
- SageAttention 3, [arXiv:2505.11594](https://arxiv.org/abs/2505.11594), NeurIPS 2025 Spotlight.
- Code: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention); SM100 status: [issue #322](https://github.com/thu-ml/SageAttention/issues/322).
- Community SM120 wheel: [mobcat40/sageattention-blackwell](https://github.com/mobcat40/sageattention-blackwell).

Distributed and CP-aware attention:

- DistFlashAttn / LightSeq, [arXiv:2310.03294](https://arxiv.org/abs/2310.03294), COLM 2024; code [RulinShao/LightSeq](https://github.com/RulinShao/LightSeq).
- BurstAttention, [arXiv:2403.09347](https://arxiv.org/abs/2403.09347).
- BurstEngine, [arXiv:2509.19836](https://arxiv.org/abs/2509.19836); code [thunlp/BurstEngine](https://github.com/thunlp/BurstEngine).
- Striped Attention, [arXiv:2311.09431](https://arxiv.org/abs/2311.09431).
- Tree Attention, [arXiv:2408.04093](https://arxiv.org/abs/2408.04093).
- MagiAttention v1.1.0, [SandAI-org/MagiAttention release](https://github.com/SandAI-org/MagiAttention/releases) (Feb 28, 2026); CP benchmark blog [sandai-org.github.io/MagiAttention/.../cp_benchmark.html](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html); FA4 fork [demonatic/flash-attention `magi_attn_blackwell_support`](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support).
- MAGI-1 backbone, [arXiv:2505.13211](https://arxiv.org/abs/2505.13211).
- MuonClip / Distributed Muon QK-Clip, [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) (Kimi K2).

Vendor / programmable kernels:

- cuDNN 9.21 release notes, <https://docs.nvidia.com/deeplearning/cudnn/backend/v9.21.0/release-notes.html>.
- cuDNN frontend Attention v1.22, <https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.22.0/operations/Attention.html>.
- FlexAttention, [PyTorch blog](https://pytorch.org/blog/flexattention/), [arXiv:2412.05496](https://arxiv.org/abs/2412.05496); [PT 2.12 docs](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html).
- Microbenchmarking NVIDIA's Blackwell, [arXiv:2512.02189](https://arxiv.org/abs/2512.02189).
- Microbenchmarking NVIDIA's Hopper (SMEM bandwidth source), [arXiv:2501.12084](https://arxiv.org/abs/2501.12084).
- Sollya numerical library: <https://www.sollya.org/>.
- DeepEP (group-collective backend used by MagiAttention v1.1.0), [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP).
