## §8. Open Gaps and Research Frontiers

This section consolidates the open-gap entries that surfaced across §1–§7. The items are grouped by category and labelled with which sub-worker / part of the report flagged them; each gap is followed by an estimate of *what would unblock it* when that's articulable. Inline cross-references use the prefix scheme defined at the top of the report.

### §8.1 Quantization gaps (G.Q)

#### G.Q.1 No public NVFP4 + step-distilled VBench/FVD on Wan 2.2

Every vendor stack quotes "20× and comparable quality" but no controlled evaluation has been published. The per-axis numbers exist in isolation: ac4k Wan 2.2 NVFP4 ships side-by-side video grids without VBench scores; FastWan publishes 95.21 → 0.98 s on H200 but at BF16 weights; lightx2v Wan 2.2 4-step + FP8 ships pre-merged but reports only filename-level metadata. The single missing data point is a head-to-head (BF16 50-step / FP8 50-step / NVFP4 50-step / NVFP4 4-step / NVFP4 4-step + sparse-distill) on the same prompt set with VBench-2.0 + FVD + LPIPS and matching seeds. *What would unblock:* one academic group (Hao AI Lab, THU-ML, lightx2v) running a controlled evaluation. Estimated cost ~80 H100-days for the full grid.

Cross-ref §2 Q.6 (TorchAO LTX-2 numbers — verbatim but at fixed step count and without sparse attention), §4 (FastWan / Wan-Lightning / Self-Forcing-Plus distilled checkpoints), §3-B S.3 (VSA — the trained-in sparse component).

#### G.Q.2 No SVDQuant for Wan / Hunyuan / Mochi / CogVideoX

[deepcompressor#103](https://github.com/nunchaku-ai/deepcompressor/issues/103): issue opened 2025-09-12 by `@zjq0455`; last public reply 2025-12-16 by `@rwfsmith`: *"support hasn't been added yet and there hasn't been any timeline given."* The closest analog is the **ac4k/Wan2.2** NVFP4 fork — community PTQ without LoRA-residual correction. Image-side FLUX.1-dev SVDQuant gets to 6.1 GB / 3.1× / Image-Reward 0.937 vs BF16 0.953 (±2 % perceptual quality at 16 GB-VRAM); without the SVD low-rank residual, the same NVFP4 path on a 14B+ video DiT loses ~5–10 % VBench in informal community tests. *What would unblock:* a published SVDQuant recipe for Wan 2.2 — the nunchaku team has flagged it for over a year. The natural next paper would be "Video-aware SVDQuant: per-timestep low-rank residuals for AdaLN-modulated DiTs," combining the SVD branch with ViDiT-Q's four-variance-level analysis (token, CFG, timestep, channel).

Cross-ref §2 Q.3 (SVDQuant + Nunchaku full algorithm), §2 Q.8 (ac4k/Wan2.2 PTQ-only fork).

#### G.Q.3 No production NVFP4 VAE for any video model

ac4k explicitly leaves the VAE in BF16; lightx2v Wan-NVFP4 likewise; FastVideo's NVFP4 LTX-2.3 demo runs the LTX-2 VAE in BF16 with tiling. At 1080p × 30+ frames the VAE is the long pole. The closest precedent is **TAEHV** ([madebyollin/taehv](https://github.com/madebyollin/taehv)) — distilled BF16/FP16 mini-VAE, 5–10× faster decode but with detail loss. **FLUX.2-Small-Decoder** is the image-side proof point (1.4× decode at 1.4× memory cut). *What would unblock:* either a quantized full VAE recipe (CUTLASS examples 76+84a have Blackwell narrow-precision conv, but production diffusers code rarely wires them) or a distilled NVFP4 mini-VAE following TAEHV's recipe but with the FLUX.2 small-decoder NVFP4 numerics. The single most obvious missing piece for B200 video DiT.

Cross-ref §2 Q.9 (VAE quantization gap walkthrough), §6 K.2 (CUTLASS conv examples).

#### G.Q.4 SageAttention 3 SM100 path reverted

[thu-ml/SageAttention#322](https://github.com/thu-ml/SageAttention/issues/322): SM100 (B200/GB200) Sage-3 path was on `main` and reverted on Dec 4, 2025; maintainer cited *"potential accuracy issues"*; the issue closed Jan 1, 2026 without a re-enable. **Only SM120 (RTX 5090) is supported in `sageattention3_blackwell` as of mid-2026.** B200 production today uses **FA-FP4 (hao-ai-lab)** at 1801 TF/s for the FP4 attention slot and **SageAttention 2++** for first/last steps. *What would unblock:* the THU-ML team re-shipping the Sage-3 SM100 build with the verified-commits subset they mentioned; failing that, MagiAttention's `FFA_FA4` Blackwell-native fork (cross-ref §3-A D.10) is the actively-maintained alternative.

#### G.Q.5 No public B200 (sm_100a) end-to-end numbers for video DiT beyond TorchAO LTX-2

Most NVFP4 video benchmarks are on **RTX 5090 (sm_120)** — lightx2v's 28× e2e on Wan 2.1 I2V-14B is RTX 5090; ac4k Wan 2.2 is RTX 5090; SageAttention 3's CogVideoX/HunyuanVideo numbers are RTX 5090. The PyTorch/TorchAO LTX-2 NVFP4 table (B=1: 1.56× / B=4: 1.67× / B=8: 1.68×) is the **only published B200 NVFP4 video number with verbatim verification**. Per-dollar / per-watt economics on B200 vs RTX 5090 are unsettled.

Cross-ref §2 Q.6 (verbatim TorchAO LTX-2 table), §7 R.1 (production blog stack).

#### G.Q.6 FP4×FP4 attention quality on long-context video

Sage 3 paper recommends running Sage 2++ at the first and last few denoising timesteps because very early/late timesteps have unusually wide P distributions that FP4 quantizes poorly. FA-FP4 has not published a hybrid policy; whether the same first/last-step caveat applies is undocumented. *What would unblock:* a controlled per-timestep ablation (FP8 attention for steps {0, 1, T-1, T}; FP4 attention otherwise) with VBench / LPIPS curves — should fit in ~5 GPU-days on a single B200.

Cross-ref §3-A D.4 (FA-FP4), §3-A D.8 (Sage 3 first/last-step recommendation).

#### G.Q.7 cuDNN block-scaled attention on Blackwell — undocumented

cuDNN 9.21 added MXFP8 SDPA forward + backward (release notes), 2-CTA MMA, and `DiagonalBandMaskOperation` + `SoftmaxOperation` graph ops, plus the FA4-backported `CUDNN_RESCALE_THRESHOLD=8` and `CUDNN_USE_EX2_EMULATION=1` flags. NVFP4 attention is *not* publicly documented in cuDNN as of May 2026; the open path remains SageAttention 3 (SM120 only) or hand-written CuTe-DSL/CUTLASS.

#### G.Q.8 No public NVFP4 + LoRA story for video DiT

Nunchaku's INT4 + LoRA-without-requantization works for FLUX. Wan-style LoRAs (e.g., Wan-Lightning rank-64) on top of an NVFP4 base have **not** been validated; the [ComfyUI-Nunchaku FP4 + multi-LoRA bug](https://github.com/nunchaku-ai/ComfyUI-nunchaku/issues/71) is still open as of May 2026. *What would unblock:* either a Nunchaku update that lands video-DiT LoRA composition, or a SVDQuant for video that includes the LoRA-without-requant trick by construction.

#### G.Q.9 2:4 sparse + NVFP4 unused in production video DiT

CUTLASS example 84a (`cutlass/examples/84a_blackwell_nvfp4_bf16_sparse_gemm.cu`) implements **2:4 structured-sparse + NVFP4 → BF16** with `KernelSparseTmaWarpSpecialized2SmNvf4Sm100`, hitting the **18 PFLOPS sparse FP4** number on HGX B200. **No video DiT pipeline ships 2:4 sparse weights** as of May 2026. *What would unblock:* an offline weight-pruning recipe (already standard for LLMs via `StructuredSparseCompressor`) plus VBench validation.

Cross-ref §6 K.2 (CUTLASS example 84a walkthrough).

#### G.Q.10 FP6 (E3M2 / E2M3) untouched by video DiT

Microbenchmark hits 5134 TFLOP/s FP6 on B200 (1.33× over FP8 memory, ~1.33× compute). FP6 is the right middle ground when FP4's 5–10 % VBench drop is unacceptable but FP8's 4.5 PF/s ceiling is constraining. **No published video-DiT recipe targets FP6** — even FP6 image-DiT is rare.

#### G.Q.11 modelopt `--quantize-mha` for FP4 attention with per-model filters

The flag inserts an FP8 (or NVFP4 with `--format fp4`) quantizer on `*[qkv]_bmm_quantizer`. For Wan/Hunyuan, cross-attention K/V replication needs per-model filter funcs that don't exist in the public modelopt. Combined with SageAttention 3 (when SM100 returns) this is the cleanest path to a fully-W4A4 + FP4-attention Wan; today it requires manual surgery.

#### G.Q.12 DeepGEMM Mega MoE doesn't directly apply to Wan 2.2-A14B

Wan 2.2-A14B is a **temporally** gated MoE (`t < t_moe = SNR_min/2` switches expert), not a **token**-gated MoE (DeepGEMM Mega MoE assumes per-token EP dispatch). Compute/memory savings from Mega MoE — fused EP-dispatch + linear-1 + SwiGLU + linear-2 + EP-combine — don't apply. The right Wan 2.2 MoE optimization is **per-expert NVFP4 with COMET-style boundary-region overlap** (the small region around `t_moe` where both experts run); no public framework implements this.

Cross-ref §6 K.7 (COMET), §6 K.9 (DeepGEMM + Mega MoE).

#### G.Q.13 Calibration-data sensitivity for video DiT — no published study

Every published recipe uses COCO Captions or MJHQ. The right calibration set should sample over **(prompt length, frame count, resolution, timesteps)** — all four axes. No published study covers all four; modelopt's `forward_loop` covers timesteps only (varying frame count and resolution at calibration is the unfilled piece).

#### G.Q.14 FlashInfer FP4 attention has no diffusion entry point

[github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) ships SM100 dense block-scaled GEMM and FP4 attention with a `flashinfer.attention.SM100` API. The public diffusers / FastVideo / lightx2v stacks use FA-FP4 (hao-ai-lab) instead. A FlashInfer SDPA backend for diffusers would be the natural unification.

### §8.2 Distributed kernel and parallelism gaps (G.P)

#### G.P.1 No published Blackwell DistGEMM ex.82 efficiency numbers

All public efficiency figures are Hopper (91 / 99 / 85 / 93 % vs single-stage GEMM on Llama 70B/405B fc1/fc2 from the SHI Labs blog). GDC + 2-CTA `tcgen05.mma` *should* yield similar or better numbers, but no third party has measured it on B200. *What would unblock:* a benchmark comparing CUTLASS DistGEMM ex.82 (with `CUTLASS_ENABLE_GDC_FOR_SM100=1`) to NCCL collective + bare GEMM at Llama-class problem sizes on 8×B200; ~1 GPU-day of measurement.

Cross-ref §6 K.3 (CUTLASS DistGEMM walkthrough), §1 P.7 (Megatron option-B context).

#### G.P.2 No fused FA4-ring kernel anywhere

FA4 publishes the forward kernel returning `(O, lse)`, which is in principle enough for cross-rank online-softmax merge. But none of the production ring libraries (`zhuzilin/ring-flash-attention`, `feifeibear/long-context-attention`, FastVideo, SGLang USP) wraps FA4 — they use FA2/FA3, with custom Triton inner loops re-implementing the FA inner loop to expose the `(O, lse)` interface incrementally. Rolling a tile across ranks while accumulating in TMEM (FA4's design) requires either a Helion-generate-then-ring-wrap path or an extension to FA4's CuTe-DSL kernel exposing dual-output mode. **DSA** (ICLR 2026) is the closest published progress on the related sparse-attention problem.

Cross-ref §3-A D.3 (FA4 design), §3-B S.18 (ring + sparse open problem and DSA).

#### G.P.3 NVFP4 block-scale async-TP not upstream in PyTorch

FP8 rowwise async-TP landed in [PR #149247](https://github.com/pytorch/pytorch/pull/149247). NVFP4 block-scale (16-element blocks with FP8 E4M3 per-block scales) async-TP is **not** upstream. The compiler-pass-based AG+matmul → fused-AG-matmul rewrite in TorchTitan PR #429 doesn't yet handle the per-block scale tensor that travels alongside activations. *What would unblock:* extending `torch.ops.symm_mem.fused_all_gather_matmul` to dispatch on NVFP4's `cublasLtMatmulDescAttributes_t` block-scaling-scheme attribute, plus the compiler-pass support for the scale tensor sharding.

Cross-ref §1 P.7 (async-TP), §6 K.5 (PyTorch SymmMem 2.12).

#### G.P.4 Async-Ulysses payload <1 MB on B200

The fal "Ulysses Unbound" blog explicitly shows fixed overheads eat the gains at 8×B200: Fused-QKV via SymmMem regresses at `P=8` (only −0.3 % e2e vs −5.0 % at P=2). The vLLM thresholds (40 MB H100, 32 MB FP8) come from an LLM-decode-shaped workload; the video-DiT regime (per-rank seq < 2k tokens at SP=8 LTX-2-class) is below most published thresholds. *What would unblock:* a runtime policy that picks transport (NCCL functional collectives, async Ulysses, fused-QKV via SymmMem, or vanilla Ulysses) based on world size × per-rank message size, with a published benchmark sweep.

Cross-ref §1 P.6 (fal blog full table), §1 P.1.5 (decision rule).

#### G.P.5 No quantitative SP+TP sweep on B200 for video DiT

Every published video benchmark on B200 (Baseten, fal, FastVideo LTX-2.3, PyTorch/TorchAO LTX-2) reports a *single* configuration. No full `(SP, TP) × seq × transport` grid measured on a B200 for any of Wan 2.2 / HunyuanVideo / LTX-2. *What would unblock:* a measurement matrix at 4 GPU and 8 GPU on (SP=1,2,4,8) × (TP=1,2,4,8 with `SP·TP ≤ 8`) × seq ∈ {3k, 9k, 16k, 32k, 75k, 100k} — ~32 configurations, ~3 GPU-days.

#### G.P.6 VAE parallelism is under-explored

SGLang reports **1.75× decode-stage speedup with Ring SP=2** on Wan 2.2 TI2V-5B [VERIFIED]. vLLM-Omni added "VAE Patch Parallelism" but no published benchmark. PipeDiT's DeDiVAE pipeline-stage separation pushes this further but again no controlled e2e number on B200.

Cross-ref §1 P.10 (PipeDiT DeDiVAE), §2 Q.9 (VAE quantization).

#### G.P.7 CUDA Graphs × dynamic-shape sparse attention

CUTLASS DistGEMM and Triton-distributed both assume static shapes inside the captured graph. **VSA's per-timestep top-K block selection** changes shape; **STA**'s mask is static so it's fine. The shape variability per timestep is a live constraint — the FastVideo `DistributedAttention_VSA` has to disable CUDA Graphs around the sparse path, losing 50–100 µs/step.

Cross-ref §3-B S.3 (VSA), §6 K.3 (CUDA Graphs in CUTLASS).

#### G.P.8 No third-party verification of 1.8 TB/s NVLink5 under realistic mixed load

DGX B200 datasheet lists 18 NVLink5 links × 100 GB/s ≈ 1.8 TB/s aggregate per GPU. Effective per-pair bandwidth in real workloads (NCCL/NVSHMEM all-to-all under simultaneous GEMM compute) is reported at 150–300 GB/s/pair (SHI Labs Hopper, fal Blackwell). No published end-to-end stress test of the 1.8 TB/s number under realistic comm/compute mix on B200.

#### G.P.9 Wan 2.2-A14B MoE expert dispatch — no COMET-style fusion

COMET ([MLSys 2025](https://arxiv.org/abs/2502.19811)) fuses MoE expert GEMMs with surrounding A2A pair using fine-grained dependency analysis: each expert's GEMM starts as soon as *its* dispatch tokens arrived. **No public framework deploys this for Wan 2.2-A14B as of May 2026** — the closest analog is the temporally-routed `_get_real_score_transformer(timestep)` in FastVideo's distillation pipeline, which doesn't fuse the boundary-region overlap.

Cross-ref §6 K.7 (COMET), §4 D.10 (Wan 2.2 MoE checkpoints).

#### G.P.10 Pipeline parallelism for long video — no public B200 benchmark

DualParal ([6.54× on 8× RTX 4090 for 1025-frame video](https://arxiv.org/abs/2505.21070)), PipeDiT (1.06–4.02× on OpenSora/HunyuanVideo), Video Interface Networks (VINs, 25–40 % FLOP cut). All measured on consumer or H100 hardware. *What would unblock:* B200 measurements on Wan 2.2 / HunyuanVideo at ≥30 s clips.

Cross-ref §1 P.10 (DualParal, PipeDiT, VINs full coverage).

#### G.P.11 Multi-request serving on B200 — no published throughput

DiT-Serve, Chorus, vLLM-Omni stream-batch RFC #2280, vLLM-Omni Batched Diffusion Serving #1511 all exist; no production B200 deployment has published throughput numbers. The right benchmark is **steady-state requests-per-second on a 4-GPU or 8-GPU B200 node** with mixed step counts (4-step distilled + 50-step base) and mixed resolutions.

Cross-ref §1 P.11 (multi-request serving primitives).

### §8.3 Distillation gaps (G.D)

#### G.D.1 MoE-aware step distillation for Wan 2.2-A14B — unsolved

FastVideo's CausalWan-MoE preview ([blog](https://hao-ai-lab.github.io/blogs/fastvideo_causalwan_preview/)) enumerates open questions:
- **Memory:** three MoE models × 28B × KV cache → exploding VRAM under Self-Forcing rollout.
- **Single-run two-expert distillation:** existing recipes (Self-Forcing-Plus' wan22 branch) do two separate distillation runs. Not viable for autoregressive video because high-noise expert needs to be conditioned on past *clean* frames generated by *low-noise* expert.
- **Timestep sampling for the loss:** with MoE, generated videos can come from either high-noise alone or cascaded high+low; right timestep range to sample is unsettled.
- **`X_0` vs `X_b` conditioning of high-noise expert** for I2V vs T2V — empirically `X_0` works for I2V, `X_b` works for T2V; the joint recipe uses `X_0` for both for consistency.

The lightx2v Wan2.2-I2V-A14B-Moe-Distill checkpoint [VERIFIED — 2 high + 2 low, Self-Forcing-derived high + Wan2.1 LoRA low] is the only published per-expert MoE distillation; no published quality numbers (no VBench / FVD / human-pref).

Cross-ref §4 D.10 (Wan 2.2 MoE checkpoints).

#### G.D.2 1-step video — unstable on Wan/LTX/Hunyuan

DOLLAR 1-step works at VBench 82.57 on a CogVideoX-class teacher but has **not** been reproduced on Wan/LTX/HunyuanVideo production backbones. Wan 2.2-Lightning recommends ≥4 steps; FastWan uses 3; rCM 1-step on Wan 2.1-1.3B reports VBench 82.65 — the lowest published Wan 1-step number. *What would unblock:* either a successful video MeanFlow at scale (currently `O(T²)` memory in mean-velocity training, blocked at video latent dimensions) or a successful video-side AAPT.

Cross-ref §4 D.13 (MeanFlow lineage), §4 D.7 (rCM), §4 D.9 (DOLLAR).

#### G.D.3 No AGD-CFG for video

AGD ([arXiv:2503.07274](https://arxiv.org/abs/2503.07274)) distills 2.6B SDXL on a single 24 GB RTX 4090 with only ~2 % of parameters trained as a CFG adapter. Image-only published. **No public video AGD checkpoint exists as of May 2026.** Direct port would give ~2 % retraining cost in exchange for 2× inference (per-step, when not already CFG-folded by step distillation). The stack composability is clear: AGD adapter on top of a step-distilled student gives a compact "mostly-frozen + tiny CFG-fold" path, useful when the user wants CFG flexibility post-distill.

Cross-ref §4 D.12 (AGD coverage).

#### G.D.4 MeanFlow at video scale — memory-prohibitive

Image-side iMF/AlphaFlow/MeanFlow achieves FID 1.72 1-NFE on ImageNet-256. The training memory cost is `O(T²)` per step due to mean-velocity backprop over `T` segments. At video latent dimensions (e.g., 24-frame Wan 2.1 1.3B = ~3k tokens × 1024 hidden × FP32 grad = ~12 MB/step × `T²` for `T=8` = ~768 MB *per layer*), this becomes prohibitive at 14B+. *What would unblock:* gradient-rematerialization-aware MeanFlow training, or a piecewise MeanFlow that sub-samples the segment graph.

Cross-ref §4 D.13 (MeanFlow lineage).

#### G.D.5 Knowledge distillation 14B → 1.3B Wan — no public checkpoint

FastVideo's roadmap mentions student-direction distillation (Wan 2.1 14B → 1.3B and Wan 2.2 28B-MoE → smaller). **No public checkpoint as of May 2026.** The closest precedent is **PPCL** (depth + width DiT pruning with teacher-student distillation; 50 % param reduction with <3 % degradation) on FLUX.1 / Qwen-Image — directly portable to a video MMDiT.

Cross-ref §4 D (general distillation), PPCL coverage in §4 D.15.

#### G.D.6 QAT + step-distillation co-training — no recipe for video

TensorRT-MO PTQ→KD-QAT recipe is validated on FLUX and LTX-Video PTQ but not on a step-distilled video model. QVGen ([arXiv:2505.11497](https://arxiv.org/abs/2505.11497)) does W4A4 QAT only at *full-step* (50 NFE). The right recipe is "step-distilled student → freeze → QAT W4A4 with auxiliary modules + rank decay" — straightforward in principle, no published instance.

Cross-ref §2 Q.7 (QVGen + Attn-QAT).

#### G.D.7 Step-Video-Turbo and Open-Sora 2 step-distilled — no published quality numbers

Step-Video-T2V-Turbo and the Open-Sora 2 community LCM/Hyper-SD distillations exist by name but lack VBench / FVD side-by-side benchmarks against their own teacher. *What would unblock:* a single-page report from StepFun + HPC-AI with controlled side-by-sides.

#### G.D.8 Combining sparsity-aware distillation with Self-Forcing on Wan 2.2 MoE

BLADE (CogVideoX-5B 8.89× / Wan-1.3B 14.10× e2e; student VBench-2.0 *beats* teacher) plus CausalWan-MoE preview. Natural next combination — **sparsity-aware Self-Forcing on Wan 2.2 MoE** — has no public checkpoint. The technical challenges are the union of G.D.1 (MoE-aware) + G.D.7 (sparse + distill).

Cross-ref §4 D.11 (BLADE).

#### G.D.9 Standardized eval for distilled+quantized video models

VBench / VBench-2.0 are used inconsistently. Many vendor numbers ("comparable quality") are human side-by-side qualitative rather than scored. The single-most-useful published number missing: VSA-vs-Sage-Sparse comparison on the **same** hardware/model with VBench-2.0 + LPIPS + human-pref under matched seeds.

Cross-ref §3-B S.18 (composability matrix open issue).

### §8.4 Composability gaps (G.C)

#### G.C.1 Ring + dynamic sparse attention has no first-class fused kernel

SVG2, SpargeAttn, VSA, XAttention all have *Ulysses* support but none ship a **Ring + sparse fused** kernel with cross-rank LSE merge in the sparse domain. **DSA** (ICLR 2026: 1.43× over distributed methods, 10.79× over single-GPU on 8 GPUs) is the only published progress. **db-SP** ([arXiv:2511.23113](https://arxiv.org/abs/2511.23113); 1.40× attention / 1.25× e2e on Wan 2.1-T2V-14B + SpargeAttn + 8× A800 vs USP) addresses sparse-attn imbalance under SP via two-level (head + block) partitioning, but is itself a Ulysses+Ring hybrid scheduler, not a true ring-sparse fused kernel.

Cross-ref §3-B S.18 (db-SP + DSA), §1 P.4 (USP).

#### G.C.2 FA-FP4 + NVFP4 GEMM round-trip layout — no public guide

Kernel-level the layouts match (NVFP4 group=16, both FA-FP4 and CUTLASS NVFP4 GEMM consume the same packed 5D `(M//32//4, K//VEC//4, 32, 4, 4)` layout). **No public guide on the end-to-end composition** — the FA-FP4 README ships kernel benchmarks only; CUTLASS examples don't show the attention-then-projection chain. The FastVideo NVFP4 LTX-2.3 stack reportedly does this round-trip but the implementation isn't published as a reference.

Cross-ref §3-A D.4 (FA-FP4), §6 K.2 (CUTLASS NVFP4 GEMM).

#### G.C.3 Speculative-then-verify for video DiT — no production deployment

SpeCa (6.34× FLUX, 6.16× HunyuanVideo at 79.84 % VBench), SDVG (1.59× at 98.1 % quality on MovieGenVideoBench), DraftAttention (1.75× e2e). No production deployment on B200. The verification step's branch-divergence interaction with CUDA Graphs is the practical blocker.

Cross-ref §5 C.10 (SpeCa, SDVG), §3-B S.15 (DraftAttention).

#### G.C.4 Static-mask sparse + 3D-aware Ring

Radial Attention's mask is purely a function of `(t, h, w)` coordinates and *should* compose with Ring under a 3D-aware sequence partition (slab-along-T or block-along-(T,H,W)). **No published implementation.** *What would unblock:* a small extension to USP that lets the seq-shard be 3D-aware, plus the Radial mask lowering to FlexAttention's `BlockMask`.

Cross-ref §3-B S.4 (Radial Attention), §1 P.4 (USP).

#### G.C.5 VSA + TP — no published combined recipe

FastVideo's `DistributedAttention_VSA` is **SP-only**. In pure TP, VSA's tile op runs on full seq with `n_heads/TP` heads — should work without modification. No published TP+VSA numbers; would need a benchmark on 8× B200 with `TP=4 × SP=2 × VSA`.

Cross-ref §3-B S.3 (VSA), §1 P.9 (FastVideo SP).

#### G.C.6 USP support for FA4 backward

USP's existing Ring backward driver still expects FA3-style `(out, lse)` layout; FA4's TMEM-backed backward changes layout assumptions. The MagiAttention `FFA_FA4` fork (cross-ref §3-A D.10) is the closest available actively-maintained Blackwell-ring-FA4 path, with explicit 8-rank context-parallel benchmarks.

#### G.C.7 HunyuanVideo 1.5 SSTA kernel API not broken out

Tencent ships HunyuanVideo 1.5 with **SSTA** (Selective Sliding Tile Attention) baked into the model checkpoint, using a `flex-block-attn` kernel. The kernel is not yet published as an independent library; the API surface to call it from outside the HunyuanVideo 1.5 repo is undocumented.

Cross-ref §3-B S.2.7 (SSTA).

#### G.C.8 Wan 2.2 attention internals — Baseten's 3× B200 speedup unbroken

Baseten reports **3.2× B200** vs default for Wan 2.2 ([blog](https://www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/)) but does not break out attention vs GEMM contributions. Without that decomposition the speedup can't be attributed to specific levers.

#### G.C.9 LTX-2 attention optimization paper or kernel release — none

LTX-2 was announced in October 2025 with 4K + audio at 50 % lower compute than competitors but no specific attention-optimization paper or kernel release. The 7-artifact catalog ships pre-quantized weights but no discussion of the attention path beyond "FlashAttention 3 was deployed on Hopper" in the official README footnote.

Cross-ref §7 R.5 (LTX-2 lineage).

#### G.C.10 STG (Spatiotemporal Skip Guidance)

Improves Mochi quality 0.524 → 0.628 and SVD FVD 151.3 → 128.7 — but compute-neutral or slightly *more* expensive (it's a sampling trick, not a speedup). Mentioned only as a quality lever; no composability evaluation against distilled or sparse stacks.

#### G.C.11 Cross-method composability matrices are largely empirical

Sage 3 + VSA, Radial + TaylorSeer, etc. — not benchmarked in any single paper. The §3-B S.18 composability matrix in this report is the closest open-published consolidation; ground-truth wall-clock × VBench tables for the full {sparse method × cache method × distill method × quant method} grid don't exist.

#### G.C.12 Backward pass in sparse video DiT under SP+TP

Most published sparse methods include a backward kernel, but **LSE-saving / recomputation tradeoffs for video DiT training under SP+TP are not documented** in detail. This is increasingly important as sparse-distillation (FastWan, BLADE) becomes the default training paradigm.

#### G.C.13 Native NVFP4 attention training kernels

SageBwd is **8-bit** (INT8 inner / FP16 inner / FP16 dO·V). **No published NVFP4 training kernel** for either forward or backward. SageBwd works for fine-tuning but converges slower than BF16 on pretraining.

Cross-ref §3-A D.8 (SageAttention 3 + SageBwd).

#### G.C.14 Radial Attention LoRA on Wan 2.2 / LTX-2

Published Radial LoRAs exist for Wan2.1-14B, HunyuanVideo, Mochi 1. **No Wan 2.2 or LTX-2 adapter is published yet.** The natural next deliverable from the Hanlab team.

#### G.C.15 First-step / last-step quality for FP4 attention

Sage 3 explicitly recommends Sage 2++ at first/last steps. Whether FA-FP4 needs the same hybrid policy is not documented. Cross-cutting with G.Q.6.

#### G.C.16 Polynomial-FMA `exp` accuracy at FP4

FA4's degree-3 FMA-emulated `2^x` has 8.77e-5 max FP32 rel err, well below BF16 quant noise. Whether the same polynomial coefficients suffice for FP4 attention scoring or need re-fitting (FP4's coarser quant grid means the relative error of the polynomial may matter differently) is open.

### §8.5 The next-natural-papers list

Twelve papers that would close the highest-leverage gaps above:

1. **"Video-aware SVDQuant: per-timestep low-rank residuals for AdaLN-modulated DiTs"** — closes G.Q.2.
2. **"NVFP4 Mini-VAE: distilled NVFP4 decoder for video diffusion"** — closes G.Q.3.
3. **"FA4-Ring: cross-rank LSE merge under TMEM-resident accumulators"** — closes G.P.2.
4. **"NVFP4 async-TP for PyTorch: the block-scale extension"** — closes G.P.3.
5. **"MoE-aware Self-Forcing for Wan 2.2-A14B"** — closes G.D.1.
6. **"Ring + dynamic-sparse: a fused kernel for SVG2 / VSA / SpargeAttn"** — closes G.C.1.
7. **"Sparse-distill BLADE + Self-Forcing on Wan 2.2 MoE"** — closes G.D.8.
8. **"AGD-Video: 2 % retraining for CFG-fold on top of step-distilled video DiT"** — closes G.D.3.
9. **"Cross-stack composability matrix: sparse × cache × distill × quant on B200"** — closes G.C.11.
10. **"NVFP4 attention QAT for video"** — closes G.Q.6 + G.C.13.
11. **"Memory-efficient MeanFlow for video"** — closes G.D.4.
12. **"Standard B200 video DiT benchmarks: VBench + FVD + LPIPS at matched compute"** — closes G.D.9.

The first three — SVDQuant for video, NVFP4 mini-VAE, FA4-ring — would together unlock the *next* 2× e2e on top of the current 30–50× headroom.

---
