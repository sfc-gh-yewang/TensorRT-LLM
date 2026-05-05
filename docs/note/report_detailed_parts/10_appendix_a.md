## Appendix A: Author's Original README — Per-Resolution Recommendations, Skip Table, SP×TP Dataflow, Assumptions

The following preserves the practical mental-model content from the user-authored `README.md` that informed this report's framing. Numbers and tables are quoted verbatim from the source.

### A.1 Five-item priority list (highest leverage first)

1. **CUDA graphs** around the denoising step — *Why*: NVFP4 shrinks per-op compute ~4× but launch overhead is fixed, so launch overhead becomes 15–30% of wall time. Diffusion has 1000+ launches per step with static shapes → perfect graph target. Single biggest leverage.
2. **Elementwise / dequant fusion** (ModulatedRMSNorm, RoPE-as-epilogue, gate-mul-as-epilogue, GELU-into-FFN-up) — *Why*: ~30–50 small ops per layer, each launch-bound. Don't shrink with SP. **Without this, scaling caps at ~1.3× regardless of parallelism choice.**
3. **Native NVFP4 GEMM** with per-tile scaling — *Why*: bf16-dequant-then-GEMM kernels add an HBM round-trip per Linear and don't shrink with SP. Native FP4 path eliminates them entirely.
4. **Comm-compute overlap kernels** — CUTLASS DistGEMM ex.82 (`CUTLASS_ENABLE_GDC_FOR_SM100`) for TP collectives + Triton-distributed `SpUlysess*` kernels for SP A2As. *Why*: ~5–10% e2e on top of 1–3, takes scaling from ~3.0× to ~3.5×+.
5. *(Optional)* **Adaptive parallelism per request shape** — *Why*: SP↔TP crossover at seq ≈ 12k is sharp; one fixed config across mixed resolutions leaves 10–20% on the table.

Without 1–3, scaling caps around 1.3–1.5×. With 1–3:
- 4-GPU `SP=2 × TP=2`: ~3.5–3.7× (~88–93% efficiency)
- 8-GPU `SP=2 × TP=4`: ~6.5–7.5× (~80–95% efficiency)

### A.2 Per-resolution recommendations (LTX-2 class video DiT)

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

### A.3 Skip / Avoid table

| Avoid | Why |
|---|---|
| Pure SP=8 (8-GPU) | `M = seq/8` below NVFP4 ridge at any reasonable seq |
| TP option A (full AllReduce) at TP ≥ 4 | AR moves 2× the bytes of AG+RS and has to hide behind one GEMM (the small O-projection); option B splits the same total comm across two GEMMs (one big, one small) so only the small fraction is exposed |
| Ring on single-node video DiT | Ulysses ties or wins everywhere for LTX-2's `(seq, head_count)`; ring's per-stage FA shrinks **quadratically** in seq while comm shrinks linearly, exposing comm at low seq. Custom ring-FA + cross-rank softmax merge is real implementation cost for zero benefit on a single node |
| Pure TP at 1080p 5s+ | Above seq ≈ 12k, post-attn RS volume exceeds O-GEMM time → exposed by ~366 µs/layer; SP doesn't have this asymmetry |
| Pure SP=4 at 480p ≤3s (4-GPU) | `M = seq/4 ≤ 1.1k`, deep underutilization; TP-heavy wins by 20–30% |

### A.4 Dataflow diagram — SP × TP hybrid (one transformer block)

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

\* In hybrid mode, the QKV/O GEMM is performed by CUTLASS DistGEMM (TP-side), and only the SP A2A is done by the Triton-distributed kernel. The kernels named `SpUlysessQKV/OAll2AllGemmKernel` are full GEMM+A2A fusions intended for *pure SP* — in hybrid mode use the comm-only paths (`pre_attn_qkv_pack_a2a_op`, `post_attn_a2a`).

- **Pure-SP**: drop CUTLASS rows; use the GEMM+A2A fused versions of the Triton-dist kernels.
- **Pure-TP**: drop SP A2A rows; FA runs on `h/TP` heads with full local seq.

### A.5 Why long-seq prefers SP, short-seq prefers TP

|  | Long seq (≥ 12k) | Short seq (≤ 6k) |
|---|---|---|
| **SP per-rank M** | `seq/SP` stays large → GEMMs compute-bound | shrinks below ridge → underutilize |
| **SP A2A volume** | linear in seq, hides behind large QKV/O GEMMs | also linear, hides behind small ones too |
| **TP per-rank M** | full seq → GEMMs compute-bound | full seq → still healthy |
| **TP post-attn RS** | volume = `seq · hidden`, large → exposes vs small O-GEMM | small enough to fit behind O-GEMM |
| **Verdict** | SP wins (TP's RS is the unhideable cost) | TP wins (SP's small-M penalty is dominant) |

The structural reason TP loses at long seq: post-attn AllReduce (option A) or ReduceScatter (option B) operates on the **full hidden-dim output tensor**, while SP's all-to-all operates on the **head-sliced tensor** (which is `1/P` smaller). That `P×` factor in comm volume is why TP can't hide its post-attn comm at long seq while Ulysses can.

### A.6 Assumptions (and where each bites if wrong)

The recommendations above hold under these explicit assumptions:

| Assumption | Where it bites if wrong |
|---|---|
| **CUTLASS DistGEMM hides comm at 95–99% efficiency on B200** with GDC enabled | If you build without `CUTLASS_ENABLE_GDC_FOR_SM100`, expect 85–90% — TP starts losing ~5% more wall time vs the numbers in tables |
| **NVFP4 GEMM achieves ~6 PF/s sustained** at `M ≥ 4k` | If sustained drops below ~3 PF/s (e.g., due to spills or bad tile shapes), the SP small-M penalty shifts and so does the SP↔TP crossover |
| **Triton-distributed Ulysses A2A hides behind QKV GEMM** | True at any seq ≥ 1k for LTX-2 (A2A volume linear in seq, GEMM also linear). Only fails if QKV GEMM tile schedule is broken |
| **NVSHMEM intra-NVL5 effective bandwidth ~150–300 GB/s per pair** | If real bandwidth is lower (e.g., contended NVL5 with other workloads), comm exposure goes up linearly |
| **Comm-compute overlap is via GDC (CUTLASS) and NVSHMEM stream pipelining (Triton-dist)**, not a single fused kernel | Both libraries achieve ~99% of bare-GEMM in their target regime; this assumption fails only if the underlying scheduling is contended (e.g., NCCL running concurrently on the same streams) |
| **CUDA Graphs capture the full denoising step** | If shapes are dynamic per step (they shouldn't be, but VSA can change), launch overhead resurfaces and 1–3 of the priority list become weaker |
| **Elementwise / dequant ops are fused into adjacent GEMM kernels' epilogues/prologues** | If kept as separate kernels, scaling is capped at ~1.3–1.5× regardless of parallelism choice |
| **FA3/FA4 supports NVFP4 attention with hardware-accelerated path** | If forced to FP8 attention, attention compute roughly doubles; SP/TP trade-offs shift slightly toward TP (because attention compute fraction grows) |
| **Per-rank weights fit in 192 GB** | Always true for LTX-2 at NVFP4 (~6.5 GB total); only fails for >100B-class DiTs which would force TP for memory regardless of perf |

### A.7 README TL;DR (verbatim)

- 4-GPU default: `SP=2 × TP=2`. 8-GPU default: `SP=2 × TP=4`.
- Long seq → more SP. Short seq → more TP. Crossover at seq ≈ 12k.
- TP always option B (CUTLASS DistGEMM ex. 82 with GDC), never option A (AR).
- Skip Ring on single-node video DiT.
- **Fix elementwise + CUDA graphs first** or none of this matters (caps at 1.3×).

---

*End of report.*
