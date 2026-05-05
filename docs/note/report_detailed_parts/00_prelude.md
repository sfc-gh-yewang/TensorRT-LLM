# Accelerating Wan / LTX / HunyuanVideo on NVIDIA Blackwell B200 — Detailed Reference (May 2026)

*An expanded technical reference report covering parallelism, quantization, attention, model-level acceleration, caching, kernel infrastructure, and production stacks for video diffusion-transformer inference on a B200 NVLink-5 node.*

*This is the long-form companion to the original 3,345-line report. Every cited paper / blog / code repository has been re-fetched (where reachable) and expanded with motivation → core technical contribution → key implementation details → quantitative results → limitations → composition with the rest of the stack. Verbatim numbers carry `[VERIFIED]` markers; corrections to the original report carry `[CORRECTED]` markers.*

## Naming convention for cross-references

Each top-level section uses a one-letter prefix throughout its subsections so that cross-references survive even after re-numbering:

| Section | Prefix | Topic |
|---|---|---|
| §1 | `P.x` | Sequence parallelism, tensor parallelism, comm-overlap |
| §2 | `Q.x` | Quantization and low-precision numerics |
| §3-A | `D.x` (within §3-A) | Dense attention kernels |
| §3-B | `S.x` (within §3-B) | Sparse and structured attention |
| §4 | `D.x` (within §4) | Step distillation and model-level acceleration |
| §5 | `C.x` | Caching and feature reuse |
| §6 | `K.x` | Blackwell B200 kernel infrastructure |
| §7 | `R.x` | Production stacks and end-to-end models |
| §8 | `G.x` | Open gaps and research frontiers |

Prefix `D.x` is intentionally reused in §3-A (dense attention) and §4 (distillation) — both predate the merge and the prefix is scoped to its parent section. Where ambiguity matters, references are spelled out as "§3-A D.x" or "§4 D.x".

---

## TL;DR

The 50× wall-clock gap between BF16 reference inference and a fully-tuned 2026 stack on Blackwell B200 is overwhelmingly explained by one lever — **step distillation** — multiplied by a small handful of B200-native foundations. Distillation alone takes the typical 50→3–8 NFE loop and folds CFG, recovering ~12.5×–25× per model. NVFP4 GEMM via the new `tcgen05.mma` block-scaled tensor cores buys another ~1.5–2× on the FFN/QKV/O projections that dominate compute. FlashAttention-4 (FA4) and its FP4 fork (FA-FP4) replace the Hopper-tuned FA3 and bring attention from 75 % to 71 % of the 2.25 PF/s BF16 ceiling and up to **1801 TF/s** at NVFP4 [VERIFIED]. Sparse attention — Sliding Tile Attention (STA), Video Sparse Attention (VSA), Radial, SVG2, SpargeAttn — delivers 2–9× attention-only speedups, and when *trained-in* via sparse-distillation (BLADE, FastWan) the resulting student often matches or *beats* its dense teacher on VBench. Caching (TeaCache / FBCache / TaylorSeer) compounds at no additional kernel cost.

The verified production stacks today are: **lightx2v Wan2.2-Lightning** (4-step Phased-DMD LoRA, BF16/FP8/INT8 pre-merged), **lightx2v Wan2.2-I2V-A14B-Moe-Distill** (2 high-noise + 2 low-noise steps, CFG=1, Self-Forcing-derived for high-noise + Wan2.1 LoRA for low-noise [VERIFIED]), **FastWan 2.1/2.2** (3-step DMD + VSA + `torch.compile`), **Krea Realtime 14B** (4-step Self-Forcing on Wan 2.1 14B, 11 FPS T2V on a single B200), **HunyuanVideo-1.5 step-distilled** (12-step recommended default, 4-step also supported), **LTX-2 distilled** (8-step, CFG=1, 7 named artifacts including `ltx-2-19b-distilled`, `ltx-2-19b-dev-fp4`, `lora-384`, plus 2 upscalers [CORRECTED — 7 artifacts, not 4]), and **Cosmos-Predict 2.5 2B distilled** (4-step DMD2 on TrigFlow parameterization, GAN omitted in NVIDIA's reference recipe).

Verified B200 numbers: PyTorch/TorchAO + Diffusers Blackwell published full LTX-2 NVFP4 tables — **B=1: 16.230 → 10.374 s (1.56×); B=4: 61.591 → 36.963 s (1.67×); B=8: 122.427 → 72.689 s (1.68×)**, with 72.77 → 80.36 GB peak memory bracketing [VERIFIED VERBATIM]. FastVideo's LTX-2.3 demo generates a **5-second 1080p TI2AV clip in 4.55 s end-to-end on a single B200**, ~3.9× the next-fastest. Baseten reports **3.2× B200 / 2.6× H100** vs default for Wan 2.2, ~40–50 s for 720p × 81f × 40 steps. SGLang's `u1r2` (Ulysses=1, Ring=2) on Wan 2.2 TI2V-5B yields **1.42× e2e (90.6 → 63.7 s) and −7.33 GB peak** with VAE decoding alone hitting 1.75× [VERIFIED]. FA4 BF16 reaches **1613 TF/s, 71 % utilization on B200**; FA-FP4 reaches **1801 TF/s, 1.39× over BF16 FA4** at the same setup. SageAttention 3 hits **1038 TOPS on RTX 5090, ~5× over the fastest FlashAttention** on that GPU (FA2; FA3 is Hopper-only) [CORRECTED: NO "37 % over FA3" number exists in the paper]. FastWan 2.1-1.3B at H200 drops the DiT loop from **95.21 s → 0.98 s (~95×)** with VSA + DMD + compile.

The biggest open gap is that **NVFP4 + step-distilled + sparse attention on Wan 2.2 has no controlled VBench/FVD published yet** — every public claim isolates one axis. The second largest gap is that **SVDQuant has no public Wan / Hunyuan / Mochi / CogVideoX checkpoint** as of May 2026 ([deepcompressor#103](https://github.com/nunchaku-ai/deepcompressor/issues/103) is open with no ETA), so the LoRA-residual error correction that gives FLUX FP4 its quality stays out of reach for video. Third: **SageAttention 3's SM100 path was reverted in late 2025** ([thu-ml/SageAttention#322](https://github.com/thu-ml/SageAttention/issues/322)) and as of January 2026 had not been re-enabled, so on B200 today the production FP4-attention slot is FA-FP4 (hao-ai-lab) plus SageAttention 2++ for first/last steps. Multiple categorical gaps remain in pipeline parallelism, multi-request serving, fused ring + dynamic-sparse kernels, and NVFP4-aware joint distillation — covered in §8 below.

---

## §0. Bottleneck Model and Mental Framework

Every wall-clock claim in this report decomposes against the same end-to-end formula:

`T_e2e ≈ T_encoder + N_steps × T_step + T_VAE_decode + T_post`

where `T_step` itself decomposes into `T_QKV_proj + T_attn + T_O_proj + T_FFN + T_norm/RoPE/elementwise + T_comm`. Each term scales differently with each optimization lever:

- `N_steps` — collapsed by **step distillation** (50 → 3–8). This is the dominant axis; nothing else competes with the 12.5×–25× headroom that comes from collapsing the denoising loop. CFG distillation folds another 2× on top. (Detailed in **§4**.)
- `T_step` GEMM components (`QKV/O/FFN`) — collapsed by **NVFP4 / MXFP8 / FP8** quantization. NVFP4 weights × NVFP4 activations on B200 saturate at ~7700 TFLOP/s per die (96.2 % of the 8000 peak); BF16 saturates at 1929 TFLOP/s (96.5 %). The geometric NVFP4-vs-BF16 speedup is ~2.3× on compute-bound GEMM and somewhat less e2e because of memory traffic and small-matmul overhead. (Detailed in **§2**.)
- `T_attn` — collapsed by **kernel improvements** (FA4, FA-FP4, cuDNN 9.21+) and by **sparse attention** (VSA, STA, Radial, SVG2, SpargeAttn). FA4 brings exact attention to 71 % of B200 BF16 peak; sparse attention cuts the *constant* of attention by another 3–9× at the cost of calibration or training. (Detailed in **§3**.)
- `T_comm` — collapsed by **Ulysses/Ring/USP**, **async-Ulysses**, **DistGEMM**, **Triton-distributed**, and **ParallelKittens**. On a DGX/HGX B200 with 1.8 TB/s NVLink5, comm is hideable for any reasonable per-rank `M = seq/SP ≥ 3–4k` token regime; below that, it stalls. (Detailed in **§1** and **§6**.)
- `T_VAE_decode` — usually treated as fixed but increasingly the long pole at high resolution. **TAEHV** and a small-decoder distillation (e.g., FLUX.2-Small-Decoder analog) can cut it by 5–10×. (Detailed in **§2** Q.9 and **§7**.)
- `T_encoder` — typically negligible after T5/CLIP FP8 weight cast plus once-per-prompt caching of K_text, V_text. (Detailed in **§7** R.8.)

### §0.1 Per-rank `M = seq/SP` is the central system-side knob

NVFP4 GEMMs need `M ≥ 3–4k` to be compute-bound on B200 — below this they go memory-bandwidth-bound and per-SM utilization drops sharply. This single ridge point decides every SP-vs-TP tradeoff: pure SP=8 starves NVFP4 GEMMs at any reasonable seq below ~25k tokens (LTX-2 at 5 s × 720p × 40 frames sits at seq ≈ 12k, putting per-rank `M ≈ 1.5k` into the starvation regime), while pure TP=8 keeps the GEMMs healthy but doubles the small-projection AR/RS exposure. This is why **hybrid `SP=2 × TP=4` is the production sweet spot** that xDiT, SGLang, FastVideo, vLLM-Omni, and Wan official converge on for 8-GPU Wan 2.2 / HunyuanVideo / LTX-2 inference. The detailed roofline derivation (`A_ridge = FLOPS / BW`, `M_ridge ≈ 0.5 · A_ridge · NK/(N+K) · (1/b_eff)`) lives in **§1 P.1**, and the SP↔TP crossover values for each resolution/GPU-count combination are in **§7 Appendix A.2**.

### §0.2 The compounding-speedup framing

Naively multiplying the per-axis wins yields a theoretical maximum of:

`distillation (~12.5×) × CFG-fold (2×) × NVFP4 GEMM (~2×) × Sage 2++ / FA-FP4 (~1.3×) × VSA (~1.5×) ≈ 100×`

Empirically, the realized e2e speedup on a representative B200 stack is closer to **30–50×** because compounding loses 50–70 % to leakage:

- **NVFP4 doesn't speed up VAE decode** — the VAE remains BF16 in every public stack as of May 2026 ([§2 Q.9](#)).
- **Sparse attention doesn't speed up the FFN** — and FFN is ~50 % of `T_step` when attention is already cut to 25 %.
- **Distillation doesn't reduce per-step memory traffic** — it cuts step count, not per-step bandwidth.
- **Cache decisions cost broadcast** — under SP, the `rel_l1_thresh` decision must be broadcast to all ranks, adding ~50 µs/step ([§5 C.2](#)).
- **Overlap rarely hits >90 % efficiency** — async-TP threshold below 40 MB on H100, fused-QKV regression at P=8 on B200 ([§1 P.6](#)).
- **CUDA-graph capture overhead at small batch** — adds 50–200 µs/step until amortized ([§7 R.1](#)).

The most-cited demonstrated example is FastWan 2.1-1.3B going from 95.21 s → 0.98 s on H200 (~95× DiT-only); FastVideo's LTX-2.3 1080p in 4.55 s on B200 is the Blackwell-side analog at full-stack scope. Both are at the *upper* end of the realized range because they specifically engineer the stack to minimize leakage (sparse attention is *trained-in* via VSA so it doesn't fight quality; the kernel is `torch.compile`-fused to amortize launch cost; the LTX-2 cascade pipeline reuses most of the encoder).

### §0.3 Asymmetric scaling on Blackwell — why FA4 had to be rewritten

The FlashAttention-4 paper [arXiv:2603.05451](https://arxiv.org/abs/2603.05451) makes the asymmetry explicit. From H100 to B200, **BF16 tensor-core throughput rose from ~1 PF/s to 2.25 PF/s (2.25×)**, but **shared-memory bandwidth stayed at 128 B/cycle/SM and the special-function-unit (SFU) `MUFU.EX2` count stayed at 16 ops/cycle/SM** — both unchanged from Hopper. So while MMA got 2.25× faster, SMEM and `exp` did not. FA3's pipeline (which assumed roughly balanced MMA/SMEM/exp) leaves both SMEM and exp under-fed at Blackwell ratios. FA4 fixes this with FMA-emulated `2^x` on the otherwise-idle FMA units (~10–25 % of softmax entries), conditional softmax rescale that fires only when `m_j − m_{j-1} > τ ≈ 8.0` (skipping ~90 % of rescales), 2-CTA UMMA in the backward to halve SMEM operand-B traffic, and TMEM-resident accumulators that decouple the correction warpgroup from the critical path. Same asymmetric-scaling logic explains why naive "FA2 baseline" sparse-attention speedup tables on B200 overstate gains — the dense baseline they're measured against is itself badly utilized. (Detailed in **§3-A D.3** and **§6 K.1**.)

### §0.4 Pre-flight checklist — the lever order for any B200 video DiT inference run

In strict order of leverage (do them top-down, A/B against a fixed BF16 baseline at each stage):

1. **Step distillation** — replace the base checkpoint with a distilled student. Choose by model: lightx2v Wan2.2-Lightning (Phased-DMD 4-step), FastWan 3-step + VSA, Krea Realtime 14B (Self-Forcing 4-step), HunyuanVideo-1.5 step-distilled (12-step default), LTX-2 distilled (8-step CFG=1), Cosmos-Predict 2.5 distilled (DMD2 4-step on TrigFlow). Single largest factor; everything else is layered on top. (**§4**)
2. **CFG-fold** — distilled checkpoints already fold CFG; verify `guidance_scale=1.0` and the inference scheduler matches the distilled recipe (LCM/Euler with the model's recommended `shift`). 2× per step. (**§4 D.12**)
3. **CUDA Graphs around the denoising step** + **regional `torch.compile(... fullgraph=True)`** — eliminates per-launch host overhead. At B=1 enable `mode='reduce-overhead'`. (**§7 R.2** and **§6 K.11**.)
4. **Native NVFP4 GEMM with selective-skip** (skip layers where `min(M, K, N) < 1024` per the TorchAO microbench heuristic). (**§2 Q.6**.)
5. **Hybrid SP × TP** — `SP=2 × TP=4` for 8-GPU 720p/1080p; pure SP=4 for 4 GPUs at 1080p+; pure TP=4 for very short clips at 480p. Check per-rank `M ≥ 3–4k`. (**§1 P.1.5** and **§7 R.10**.)
6. **FA4 / cuDNN 9.21+ as the attention backend**, then **FA-FP4** when NVFP4 GEMM is in place. (**§3-A D.3, D.4**.)
7. **TeaCache at `rel_l1_thresh=0.10`** — 1.5× near-lossless on most non-distilled checkpoints. Not used with already-distilled few-step students. (**§5 C.2**.)
8. **Sparse attention** — `Radial Attention LoRA` for trained-free length extension; `VSA` if you can fine-tune. (**§3-B S.3, S.4**.)
9. **Pipeline parallelism + multi-request serving** for production at scale. (**§1 P.10, P.11**.)

If you skip step 3 (CUDA graphs + elementwise/dequant fusion), nothing else past step 4 will scale beyond 1.3–1.5× — the ~30–50 small ops per layer will dominate launch overhead and saturate your gain.

---
