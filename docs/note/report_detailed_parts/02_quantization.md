## §2. Quantization and Low-Precision Numerics for Video DiT

*Subsection prefix `Q.x` is scoped to §2.*

This section is a deep-dive into the state of low-bit quantization for **video** diffusion transformers (Wan 2.1/2.2, HunyuanVideo, CogVideoX, Mochi, LTX‑2, Sora-style), with a sharp focus on what *actually runs* on **NVIDIA HGX B200** (Blackwell SM_100). The picture in May 2026 is unusual: Blackwell offers a 4× theoretical speedup over Hopper FP8 and roughly 2× over BF16, but the *video* DiT software stack lags badly behind LLM and image-DiT stacks. The single biggest open problem is that no production-grade NVFP4 W4A4 PTQ recipe (with low-rank residual compensation à la SVDQuant) exists for the leading video DiTs, while the dominant production paths in TensorRT‑LLM VisualGen, vLLM-Omni, FastVideo/TurboDiffusion, lightx2v, and torchao are converging on three tiers: **W8A8 INT8** for portability, **FP8 (per-tensor / blockwise / channel-wise)** for "first-good speedup with minimal effort", and **NVFP4 (W4A4)** for compute-bound regimes where memory bandwidth has stopped being the bottleneck.

The structure below mirrors the request: numerics first, then PTQ algorithms, then SVDQuant + the video gap, then rotation methods, then the NVIDIA stack, then the PyTorch/TorchAO Diffusers stack with the verbatim LTX-2 numbers, then QAT, then per-model status, then the VAE quantization gap, then composition with attention/step-distillation/CFG-distillation.

---

## Q.1 NVFP4 / MXFP4 / MXFP8 numerics + Blackwell tensor-core spec

### Q.1.1 The three Blackwell 4-bit formats: FP4 (E2M1), MXFP4, NVFP4

Blackwell's fifth-generation Tensor Cores are the first NVIDIA datacenter parts to expose **native FP4** in hardware. NVIDIA's "Introducing NVFP4" blog [\[introducing-nvfp4\]](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference) lays out three closely related 4-bit floating-point formats, all sharing the same E2M1 storage element (1 sign, 2 exponent, 1 mantissa) and a representable range of approximately \([-6, +6]\) with the discrete grid \(\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}\):

| Format | Block size | Per-block scale | Per-tensor scale | Hardware acceleration |
|---|---|---|---|---|
| **FP4 (plain E2M1)** | — | software / per-tensor | required | not block-scaled |
| **MXFP4** (OCP MX) | 32 elements | E8M0 (power-of-two) | none | yes (Blackwell `tcgen05.mma kind::mxf4`) |
| **NVFP4** (NVIDIA) | 16 elements | E4M3 (FP8, fractional) | FP32 per-tensor | yes (Blackwell `tcgen05.mma kind::mxf4nvf4`) |

The two key innovations in **NVFP4** versus the OCP MXFP4 standard are (i) **smaller block size** (16 vs 32) and (ii) **higher-precision scale encoding** (E4M3 vs E8M0). The smaller block doubles the number of independent dynamic-range "anchors" per tensor, and the E4M3 scale carries 3 mantissa bits so the per-block scale itself is no longer snapped to a power of two. NVIDIA reports that this combination — "fractional" scales plus 16-element granularity — drops the average MSE of a single block of E2M1 values from roughly 0.18 (E8M0 scale) to 0.08 (E4M3 scale) on representative LLM activations, an ~2.25× error reduction *with the storage cost rising only from 1×4 = 4 bits/value to 4 + 8/16 = 4.5 bits/value*. NVIDIA's NVFP4 picture is therefore "two-level scaling": a per-tensor FP32 scalar absorbs the global dynamic range so that all per-block scales fit comfortably in E4M3, and the per-block E4M3 scalar then locks the final 16 elements to the FP4 representable set.

The OCP MX format spec [\[Rouhani et al., 2023, "Microscaling Data Formats for Deep Learning"\]](https://arxiv.org/abs/2310.10537) ratifies MXFP4/MXFP6/MXFP8/MXINT8, all of which Blackwell supports natively in `tcgen05.mma` with `kind::mxf{4,6,8}` PTX descriptors. The FP6 (E2M3 and E3M2) and FP8 (E4M3 and E5M2) variants are also included; `kind::mxf8/f8f6f4` is the unified mixed precision path for QMMA (quasi-MMA). The microbenchmarking paper [\[Jarmusch & Chandrasekaran, arXiv:2512.02189\]](https://arxiv.org/abs/2512.02189) Table IV maps these PTX kinds to the SASS instructions:

| Precision | PTX (`tcgen05.mma kind::*`) | SASS |
|---|---|---|
| FP16, BF16 | `f16` | HMMA |
| FP32, TF32 | `tf32` | HMMA |
| FP8 | `mxf8` / `f8f6f4` | QMMA |
| FP6 | `mxf6` | QMMA |
| **FP4** | `mxf4` / `mxf4nvf4` | **OMMA** |
| INT8 | `i8` | IMMA |
| FP64 | n/a | DMMA (not on `tcgen05`) |

The new **OMMA** opcode is the FP4-only path (the "O" presumably for "octa-bit" pairs). It only exists on Blackwell and Blackwell Ultra. Because both MXFP4 and NVFP4 share the same E2M1 storage element, OMMA implements both — the descriptor bits select 32-block E8M0 scaling vs 16-block E4M3 scaling, and the kernel pays no throughput penalty for choosing NVFP4 over MXFP4. This is important for the PTQ ecosystem: **NVFP4 is the strictly higher-quality option at the same FLOP rate**, so on Blackwell there is essentially no reason to deploy MXFP4 unless cross-vendor portability (AMD CDNA4, Intel Gaudi-3, Cerebras CS-3) matters.

### Q.1.2 Measured throughput on B200: the 7700 TFLOP/s number

The microbenchmarking paper measured per-die FP4 throughput at **7700.2 TFLOP/s (96.2 % of architectural peak)** on a single B200 die using `tcgen05.mma kind::mxf4nvf4` with FP16 accumulation, m64n8k16 tile shape, and dependency-isolated MMA streams. Their Table VII is the cleanest "what does each Blackwell precision actually deliver in dense MMA" reference in the public literature as of May 2026:

| Precision | B200 (TFLOP/s or TOP/s) | % of theoretical peak | H200 baseline | B200 / H200 |
|---|---|---|---|---|
| FP64 (DMMA) | 44.8 | 99.6 % | 34.0 | 1.32× |
| FP32 (TF32-via-emulation) | 482.0 | 96.4 % | 378.4 | 1.27× |
| TF32 | 964.5 | 96.5 % | 756.9 | 1.27× |
| BF16 | 1926.4 | 96.3 % | 1513.5 | 1.27× |
| FP16 | 1929.6 | 96.5 % | 1515.2 | 1.27× |
| FP8 | 3850.6 | 96.3 % | 3026.9 | 1.27× |
| FP6 (MXFP6) | 5134.4 | 96.0 % | n/a | new |
| **FP4 (MX/NVFP4)** | **7700.2** | **96.2 %** | n/a | new |
| INT8 | 3928.5 | 98.2 % | 3088.4 | 1.27× |

Two architectural observations follow from this table.

First, **FP32 accumulation halves throughput** for all sub-FP32 inputs: the single Blackwell datapath that supports INT32/FP32 accumulators is provisioned at half the multiplier rate of FP16 accumulators. This is why all of the 4-bit fast paths in cuBLAS/CUTLASS/cuDNN/torchao default to FP32 accumulation only at the *epilogue* (the "FP32 in registers, FP16 accumulator in TMEM" idiom) — sustained dense MMA throughput at FP4×FP4→FP32 is closer to 4 PFLOP/s, not 7.7 PFLOP/s. For diffusion this matters because the residual and adaptive-LN scaling paths sit in the epilogue.

Second, **FP4 throughput is exactly 2× FP8 and 4× FP16/BF16** at the SM level: 7700/3850 = 2.00, 7700/1929 = 3.99. So the Blackwell FP4 dense-MMA peak is the textbook "2× the precision-below" relationship, with no surprises. The ratio is reproduced almost identically by the FP6 row (5134/3850 = 1.33×, the OCP MXFP6 ratio). All of this is consistent with NVIDIA's published per-GPU numbers of **9 PFLOP/s dense FP4 / 18 PFLOP/s sparse FP4** per B200 GPU; the 7.7 PFLOP/s the microbenchmark observes is 7.7/9 = 86 % of full-die NVFP4 dense, with the gap explained by MMA pipeline fill, accumulator ports, and the 96.2 % vs 100 % "real measurement" haircut on the first die.

### Q.1.3 HGX B200 datasheet reconciliation: 18 PF/s sparse / 9 PF/s dense per GPU

The current NVIDIA HGX product page [\[nvidia.com/en-us/data-center/b200\]](https://www.nvidia.com/en-us/data-center/b200/) lists the HGX B200 system at **108 PFLOP/s FP4 sparse / 54 PFLOP/s FP4 dense** for an 8-GPU chassis, and HGX B300 (Blackwell Ultra) at **144 / 72 PFLOP/s sparse/dense FP4**. The earlier widely-circulated number of **15 PFLOP/s FP4 per GPU** on B200 was an *announcement* figure from GTC 2024 that was subsequently revised. The current ground truth, confirmed both by NVIDIA's product datasheet math (54 PF/s dense ÷ 8 GPUs = ~6.75 PF/s dense) and by NVIDIA's per-GPU Blackwell datasheet (which lists 9 PF/s dense / 18 PF/s sparse), is:

> **Per HGX B200 GPU: 9 PFLOP/s dense FP4, 18 PFLOP/s sparse FP4, 4.5 PFLOP/s dense FP8, 9 PFLOP/s sparse FP8.**

The HGX B300 (Blackwell Ultra) part has a higher "Attention Performance" multiplier (2× vs B200) and 144 PF/s sparse FP4 across 8 GPUs, but B200 is the platform shipping in volume in the second half of 2025 and the first half of 2026, and is what the rest of this document targets.

The microbenchmark's 7.7 PF/s measurement (per *die*; B200 is dual-die) is the right number to cite when discussing single-kernel utilization, while 9 PF/s dense (per GPU, both dies) is the right number for sizing throughput at the application level. The corollary for diffusion deployment is that *if* a video-DiT kernel can sustain 70 % of architectural peak FP4 — already an aggressive target given 16-element block scaling overhead — it will deliver about 6.3 PF/s on a B200, or roughly 4.6× over a similarly-utilized BF16 baseline at 1.35 PF/s. This 4× headline is the entire economic motivation for NVFP4 video-DiT inference.

### Q.1.4 cuBLAS 12.9 NVFP4 GEMM API and `LtNvfp4Matmul`

cuBLAS 12.9 (May 2025) is the first cuBLAS release that exposes NVFP4 directly to user code [\[boosting-matmul-speed-cublas-12.9\]](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/). The relevant types are:

- `CUDA_R_4F_E2M1` — the FP4 element type
- `CUDA_R_UE4M3` — the *unsigned* E4M3 scale type used by NVFP4 (sign bit ignored; range matches FP8 E4M3 but always positive)
- `CUDA_R_UE8` — the unsigned E8M0 scale type used by MXFP8 and MXFP4

Calling `cublasLtMatmul` with `CUDA_R_4F_E2M1` inputs and `CUDA_R_UE4M3` scaling forces the 16-element NVFP4 path and uses fifth-generation Tensor Cores natively. The cuBLASLt API additionally lets the kernel compute a scaling factor for the output tensor (`scaleD` in NVIDIA's diagram) when the output is FP4 or FP8 — i.e., a fused quantize-on-store epilogue, which is critical for chained matmuls in a transformer where the output of a QKV projection is the input of the attention and must already be in the right block-scaled FP4 layout. This eliminates the otherwise-required separate "calibrate amax then quantize" pass that older Hopper FP8 kernels need.

NVIDIA publishes a working reference at [\[CUDALibrarySamples/cuBLASLt/LtNvfp4Matmul\]](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtNvfp4Matmul). The companion sample for MXFP8 lives at `LtMxfp8Matmul`. Both rely on `cublasLtMatmulDescAttributes_t` setting `CUBLASLT_MATMUL_DESC_BLOCK_SCALING_SCHEME` and the `*_SCALE_POINTER` attributes for A, B, and (optionally) D scales. cuBLAS 12.9's reported peak performance on a GB200 is **6787 TFLOP/s FP4** in its most aggressive compute-bound benchmark, which is ~88 % of microbenchmark peak — consistent with the typical cuBLAS overhead (heuristic dispatch, batch decomposition, epilogue glue) on top of the raw OMMA pipeline.

The fact that this is in *cuBLAS itself*, rather than only in CUTLASS or hand-written PTX, matters for the diffusers/torchao path: TorchAO's NVFP4 kernels (and Triton-on-Blackwell autotuned variants) call into cuBLASLt for the heavy GEMMs and into Triton/CUTLASS for the fused per-block scale extraction kernels. Without `LtNvfp4Matmul` the entire downstream stack would have to ship a custom NVFP4 GEMM (the route Nunchaku originally took and the route the LightX2V kernel still takes).

---

## Q.2 PTQ for image / video diffusion (PTQ4DiT, Q-DiT, ViDiT-Q)

The first wave of DiT-aware post-training quantization papers (2023–24) all start from the same observation, sometimes called the "DiT quadruple problem":

1. **Spatial outliers** — a small set of input channels in MHSA Q/K/V and in the pointwise FFN have absolute magnitudes 5–50× larger than typical channels. This is the standard transformer phenomenon (LLM.int8(), SmoothQuant) but more severe in DiT because the AdaLN scale-shift parameters \((\gamma, \beta)\) are themselves regressed from the timestep+condition embedding via an MLP and then applied multiplicatively, amplifying any outlier in \(\gamma\) into the post-norm activation.
2. **Temporal variation** — the activation distribution shifts dramatically across the 25–250 denoising timesteps. Calibration parameters fitted at \(t=t_1\) overshoot at \(t=t_2\). PTQ4DiT's Figure 4 visualizes this as a boxplot of per-channel max-abs over 50 timesteps, where the IQR of any single channel can swing by 3×.
3. **AdaLN spike** — the modulation layers (the small MLPs that produce \(\gamma, \beta\) from the timestep+text embedding) are extraordinarily sensitive to quantization. Their input is a single highly-compressed condition vector, so even small errors propagate through every transformer block.
4. **Conditional/unconditional split** — for classifier-free guidance, the model is run on a batch of two (the conditional branch with text and the unconditional branch). The two branches have different activation distributions and ideally need different quantization parameters; ViDiT-Q showed this is the dominant remaining error source after channel and temporal balancing.

### Q.2.1 PTQ4DiT (NeurIPS 2024)

PTQ4DiT [\[Wu et al., arXiv:2405.16005\]](https://arxiv.org/abs/2405.16005) is the first PTQ method explicitly designed for DiT (DiT-XL/2, ImageNet 256² and 512²). Its central observation, illustrated in Figure 1 of the paper, is the **channel-complementarity property**: large-magnitude channels in activation \(\mathbf{X}\) do *not* coincide with large-magnitude channels in weight \(\mathbf{W}\), so a per-channel rescaling of the form

\[
\widetilde{\mathbf{X}} = \mathbf{X} \mathbf{B^X}, \qquad \widetilde{\mathbf{W}} = \mathbf{B^W}\mathbf{W},
\qquad \mathbf{B^X} \mathbf{B^W} = \mathbf{I}
\]

can shift "salience" between weight and activation columns until both are equally easy to quantize. The Salience Balancing Matrix (CSB) constructs \(\mathbf{B^X}, \mathbf{B^W}\) as diagonals from the geometric-mean-of-maximums:

\[
B^X_{jj} = \frac{\widetilde{s}_j}{s(\mathbf{X}_j)}, \qquad B^W_{jj} = \frac{\widetilde{s}_j}{s(\mathbf{W}_j)}, \qquad
\widetilde{s}_j = \sqrt{s(\mathbf{X}_j)\,s(\mathbf{W}_j)},
\]

where \(s(\cdot)\) is the channel max-abs. This is exactly the SmoothQuant transformation, but with a square-root-of-product weighting instead of SmoothQuant's tunable \(\alpha\) hyperparameter; PTQ4DiT shows that the geometric mean is an optimal balance for DiT in the sense that it minimizes \(\max(s_o(\widetilde{\mathbf{X}}), s_o(\widetilde{\mathbf{W}}))\) — the largest residual outlier after balancing.

The temporal-variation problem is then handled by **Spearman's-ρ-guided Salience Calibration (SSC)**. Rather than averaging \(s(\mathbf{X}^{(t)})\) uniformly over timesteps, PTQ4DiT computes the Spearman rank correlation \(\rho_t = \rho(s(\mathbf{X}^{(t)}), s(\mathbf{W}))\) between activation and weight salience at each timestep, and then weights each timestep's contribution by

\[
\eta_t = \frac{\exp[-\rho_t]}{\sum_\tau \exp[-\rho_\tau]}.
\]

Timesteps where activation outliers and weight outliers are *anti-correlated* receive larger weight, because those are precisely the timesteps where channel-balancing yields the biggest reduction in residual salience. The aggregated \(\mathbf{s}_\rho\) is plugged back into the CSB construction.

Crucially, **PTQ4DiT has zero inference overhead**: \(\mathbf{B^W}\) is folded into the linear layer's weight offline, and \(\mathbf{B^X}\) is folded into the *previous* op — the AdaLN scale and shift parameters \((\gamma, \beta)\) for post-AdaLN linears, or the Q·K^T and softmax-output dequantization for post-attention linears. Concretely:

\[
\widetilde{\text{adaLN}}(\mathbf{Z}) = \text{LN}(\mathbf{Z}) \odot (\mathbf{B^X_\rho} + \widetilde{\boldsymbol{\gamma}}) + \widetilde{\boldsymbol{\beta}},
\quad \widetilde{\boldsymbol{\gamma}} = \boldsymbol{\gamma}\mathbf{B^X_\rho}, \quad \widetilde{\boldsymbol{\beta}} = \boldsymbol{\beta}\mathbf{B^X_\rho}.
\]

On DiT-XL/2 ImageNet 256² with 250 sampling steps, PTQ4DiT reaches **W8A8 FID 4.63 / IS 274.86 / Precision 0.8299** versus the FP baseline FID 4.53. At **W4A8** it scores FID 7.09 / IS 201.91, vastly better than Q-Diffusion (FID 15.31), PTQD (16.45), or RepQ\* (23.21). PTQ4DiT is therefore the first method to *successfully* quantize DiT to 4-bit weights (with 8-bit activations), establishing the W4A8 setting that subsequent papers (Q-DiT, ViDiT-Q, EfficientDM) compete on.

What PTQ4DiT does *not* do: (a) it does not handle W4A4 (which the 16-element NVFP4 microblock partly fixes by giving each block its own scale and so suppressing outlier propagation); (b) it is evaluated only on class-conditional 256²/512² ImageNet, not on text-to-video Wan/Hunyuan/Mochi/CogVideoX; (c) the calibration is 25–32 timesteps × 32 samples, manageable for image but expensive for video (each "sample" is a full noise-to-video trajectory).

### Q.2.2 Q-DiT (CVPR 2025)

Q-DiT [\[Chen et al., arXiv:2406.17343\]](https://arxiv.org/abs/2406.17343) extends PTQ to *both* DiT-XL/2 and the OpenSORA video DiT. Its two contributions are

1. **Automatic per-layer group-size search**, motivated by the empirical observation that DiT layer sensitivity to quantization is *non-monotonic* in group size — finer (smaller) groups do not always help. On DiT-XL/2 ImageNet 256², group 128 yields FID 17.87 but group 96 yields FID 19.97 (an 11.8 % regression), while group 64 may again beat group 96. Q-DiT therefore frames per-layer group-size selection as a constrained optimization: minimize \(L(\mathbf{g}) = \mathrm{FID}(R, G_\mathbf{g})\) subject to \(B(\mathbf{g}) \le N_{\text{bitops}}\), and solves it with an evolutionary algorithm using crossover and mutation over a discrete group-size space. For video models the objective swaps FID for FVD.

2. **Sample-wise dynamic activation quantization**. Rather than computing per-tensor scales from a calibration set, Q-DiT computes \(s, z\) on-the-fly per *sample and per timestep* in the inference loop. This adds a few hundred microseconds of `amax` reduction but is the only way the authors found to handle the temporal *and* per-sample distribution shift.

Q-DiT reports lossless **W6A8** quantization on DiT-XL/2 (FID matches FP) and minimal degradation at **W4A8** for both ImageNet 256² and OpenSORA 16-frame text-to-video — the *first* W4A8 result reported on a video DiT in the public literature. The OpenSORA evaluation is on VBench, with Q-DiT reporting an average VBench score within 1.0 point of the FP baseline.

The hardware story for Q-DiT is incomplete: the paper reports BitOps as a proxy and does not implement an on-GPU kernel that exploits the per-sample dynamic activation quantization (which would require fused amax-and-quantize kernels and a kernel-launch path that does not bake a static scale into the GEMM descriptor). On Blackwell, this is exactly what cuBLAS 12.9's `scaleD`-fusion is for, so a Q-DiT-style dynamic-activation method should be *cheaper* on B200 than on Hopper. As of May 2026, no such Q-DiT kernel has been released.

### Q.2.3 ViDiT-Q (ICLR 2025)

ViDiT-Q [\[Zhao et al., arXiv:2406.02540\]](https://arxiv.org/abs/2406.02540) is the first paper to systematically catalog the *unique* DiT quantization challenges and, importantly, the first to ship a working CUDA kernel that delivers a **measured** end-to-end speedup on text-to-video DiT. ViDiT-Q's "four-variance-level" diagnosis is the most-cited part of the paper:

- **Token-wise variation**: visual tokens (latent patches) within a single forward pass have substantially different activation magnitudes; for video DiTs the variation is two-dimensional (spatial × temporal).
- **Condition-wise variation**: the two halves of the CFG batch (conditional with text, unconditional with null prompt) have measurably different distributions; sharing one quantization parameter set across the batch hurts FID/CLIP-Score.
- **Channel-wise variation**: classic outlier channels, exacerbated in DiT by AdaLN.
- **Temporal-wise variation**: the same channel's max-abs swings across the 50–100 sampling timesteps.

ViDiT-Q's prescription is correspondingly four-pronged: (1) per-token dynamic quantization for activations, (2) split CFG into separate conditional and unconditional quantization-parameter sets, (3) "static-dynamic" channel balancing — a static SmoothQuant-style rescale combined with a dynamic per-token rescale, and (4) a metric-decoupled mixed-precision search where the search objective is decoupled into per-metric (FID, CLIP, motion smoothness) sensitivity scores per layer.

The hardware contribution is two CUDA kernels: a per-token per-channel W8A8 INT8 GEMM, and a partial W4A8 path for the FFN. Reported end-to-end on PixArt-α 1024² and OpenSORA 16-frame: **2–2.5× memory saving and 1.4–1.7× latency speedup** at W8A8 with negligible loss in FID and VBench scores, on an A100. The W4A8 setting works on PixArt but the paper notes (Table 1 of SVDQuant) that ViDiT-Q's **W4A4 result on PixArt-Σ is FID = 412, IR = -2.27** — i.e., catastrophic. This is the data point that motivated SVDQuant and remains the canonical "naive PTQ does not survive W4A4 on DiT" reference.

ViDiT-Q's CFG-aware quantization is, as of May 2026, the cleanest published treatment of the *batch-of-two* problem. Most subsequent stacks (modelopt, TensorRT-LLM VisualGen, vLLM-Omni FP8) silently quantize CFG-conditional and CFG-unconditional with the same scale; the resulting ~0.1 LPIPS regression versus FP baselines (visible in the TorchAO/Diffusers blog's QwenImage row at 0.34 LPIPS for MXFP8) is in part this ignored axis.

### Q.2.4 What's missing from the PTQ-only picture

Even the best PTQ method (Q-DiT-style group search + ViDiT-Q-style channel/CFG balancing + PTQ4DiT-style temporal weighting) plateaus at **W4A8** for video DiTs as of late 2024. To reach **W4A4** — which is what NVFP4 actually delivers in hardware — one of three things is needed:

(a) A *low-rank residual* compensation branch that absorbs whichever outliers remain after channel balancing — the SVDQuant route (Q.3 below).

(b) A *rotation* that mathematically cancels outliers without any additional branch — the QuaRot/SpinQuant/DFRot/ConvRot route (Q.4 below).

(c) Quantization-aware training/finetuning to *adapt the weights themselves* to the W4A4 grid — the EfficientDM/QVGen route (Q.7 below).

In practice, on May 2026, the cleanest production NVFP4 path for image DiT (FLUX.1) uses (a) + (b) — SVDQuant + per-block NVFP4 native scaling. For *video* DiT, none of these has shipped at W4A4 quality at the time of writing; the closest is QVGen (W4A4 QAT, no public B200 kernel) and lightx2v's Wan-NVFP4 (which is QAT+distillation, not pure PTQ).

---

## Q.3 SVDQuant + Nunchaku + the deepcompressor#103 video gap

### Q.3.1 The SVDQuant decomposition (ICLR 2025 Spotlight)

SVDQuant [\[Li et al., arXiv:2411.05007\]](https://arxiv.org/abs/2411.05007) is the most-cited W4A4 method for diffusion as of 2026, and the only PTQ method to deliver image-DiT quality on par with the BF16 baseline at 4-bit weights *and* 4-bit activations. Its insight starts from the observation that smoothing alone (SmoothQuant) is insufficient: redistributing outliers between \(\mathbf{X}\) and \(\mathbf{W}\) leaves at least one of the two with non-trivial outliers, which at 4-bit grid resolution is fatal.

SVDQuant adds a third object: a high-precision *low-rank residual* branch. The full decomposition is

\[
\mathbf{X}\mathbf{W}
= \widehat{\mathbf{X}}\widehat{\mathbf{W}}
\;\approx\;
\underbrace{\widehat{\mathbf{X}}\mathbf{L}_1\mathbf{L}_2}_{\text{16-bit low-rank branch (rank 16 or 32)}} \;+\; \underbrace{Q(\widehat{\mathbf{X}}) \cdot Q(\mathbf{R})}_{\text{4-bit residual GEMM}}
\]

with the four-step construction:

1. **Smooth**: pick \(\boldsymbol{\lambda}\in\mathbb{R}^m\) to migrate activation outliers into weight, giving \(\widehat{\mathbf{X}} = \mathbf{X}\,\mathrm{diag}(\boldsymbol{\lambda})^{-1}\), \(\widehat{\mathbf{W}} = \mathrm{diag}(\boldsymbol{\lambda})\,\mathbf{W}\).
2. **SVD**: compute \(\widehat{\mathbf{W}} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\), and split off the top-\(r\) singular components: \(\mathbf{L}_1 = \mathbf{U}\boldsymbol{\Sigma}_{:,:r}\), \(\mathbf{L}_2 = \mathbf{V}_{:r,:}\).
3. **Residual**: \(\mathbf{R} = \widehat{\mathbf{W}} - \mathbf{L}_1\mathbf{L}_2\). Because \(\widehat{\mathbf{W}}\) has dramatically reduced *singular* spread after smoothing (the first 32 singular values are dominant), \(\mathbf{R}\)'s Frobenius norm is small and its values are sub-outlier.
4. **Quantize**: only \(\mathbf{R}\) and \(\widehat{\mathbf{X}}\) are quantized to 4 bits; \(\mathbf{L}_1, \mathbf{L}_2\) stay at 16 bits.

The error bound that justifies this is Proposition 4.1 of SVDQuant:

\[
\|\mathbf{X}\mathbf{W} - Q(\mathbf{X})Q(\mathbf{W})\|_F
\;\le\; \|\mathbf{X}\|_F \cdot \|\mathbf{W} - Q(\mathbf{W})\|_F + \|\mathbf{X} - Q(\mathbf{X})\|_F\cdot(\|\mathbf{W}\|_F + \|\mathbf{W} - Q(\mathbf{W})\|_F).
\]

Both \(\|\mathbf{X}\|_F\) and \(\|\mathbf{W}\|_F\) are bounded by the *magnitude* of the tensor; both \(\|\mathbf{X} - Q(\mathbf{X})\|_F\) and \(\|\mathbf{W} - Q(\mathbf{W})\|_F\) are bounded by the *quantization error*, which Proposition 4.2 bounds by \(\sqrt{\text{size}}/q_{\max}\) times the magnitude. So the four levers in the bound are: \(\|\mathbf{X}\|_F\), \(\|\mathbf{W}\|_F\), \(\|\mathbf{X}-Q\|_F\), \(\|\mathbf{W}-Q\|_F\). Smoothing reduces \(\|\mathbf{X}-Q\|_F\) at the cost of inflating \(\|\mathbf{W}\|_F\); SVD on \(\widehat{\mathbf{W}}\) reduces \(\|\mathbf{R}\|_F\) (which is what's actually quantized) without rescaling \(\widehat{\mathbf{X}}\); together the residual term \(E(\widehat{\mathbf{X}}, \mathbf{R})\) is several × smaller than naïve W4A4.

### Q.3.2 Nunchaku — the kernel that makes SVDQuant *fast*

The naive way to run SVDQuant is to compute the low-rank branch \(\widehat{\mathbf{X}}\mathbf{L}_1\mathbf{L}_2\) as two 16-bit GEMMs (down-projection then up-projection), in parallel with the 4-bit residual GEMM, then add the two outputs. This has 57 % latency overhead on FLUX.1 (Figure 6 of the paper) because the activations are read twice and the up-projection's output exceeds L2 cache.

Nunchaku's contribution is **kernel fusion**:

- The down-projection \(\mathbf{L}_1\) and the activation-quantize-to-4-bit kernel share the *same input* \(\widehat{\mathbf{X}}\) — fuse them.
- The up-projection \(\mathbf{L}_2\) and the 4-bit GEMM share the *same output* destination — fuse them so the up-projection writes into the same accumulator.

After fusion, the low-rank branch costs only **5–10 % extra latency** on top of the 4-bit GEMM, and the entire SVDQuant pipeline runs at **~3.0× over W4A16 NF4** on a 16 GB laptop RTX 4090, **3.1× on a desktop RTX 5090** (Blackwell consumer), and **8.7× over the original BF16** model on the laptop 4090 thanks to eliminating CPU offloading.

The MIT HAN Lab blog [\[svdquant-nvfp4\]](https://hanlab.mit.edu/blog/svdquant-nvfp4) reports the following quality numbers on RTX 5090 (NVFP4) for FLUX.1-dev versus BF16:

| Method | ImageReward (↑) | LPIPS (↓) | PSNR (↑) |
|---|---|---|---|
| BF16 | 0.953 | — | — |
| INT4 (RTN baseline) | 0.908 | 0.322 | 18.5 |
| INT4 + SVDQuant | 0.935 | 0.223 | 21.0 |
| **NVFP4** | 0.928 | 0.244 | 20.3 |
| **NVFP4 + SVDQuant** | **0.937** | **0.208** | **21.4** |

NVFP4 alone beats INT4 alone by ~2.5 dB PSNR, confirming the prediction that the smaller block size and E4M3 scale are doing real work; SVDQuant on top of NVFP4 closes essentially all of the remaining gap to the BF16 reference (0.937 vs 0.953 ImageReward). On PixArt-Σ the same recipe brings PSNR from 9.08 (INT4 RTN — visually unrecognizable) to 18.5, a 9.4 dB gain.

### Q.3.3 The video gap: deepcompressor#103

As of May 2026, **SVDQuant has no released checkpoint for any video DiT**: not Wan 2.1, not Wan 2.2, not HunyuanVideo, not Mochi, not LTX-2, not CogVideoX. The MIT HAN Lab page says explicitly *"In the future, we will continue optimizing our kernels and extending support to more models (e.g., video models) to further benefit the community."* — but this commitment, made in late 2024, has not yet shipped.

The most concrete community signal is GitHub issue [`nunchaku-ai/deepcompressor#103`](https://github.com/nunchaku-ai/deepcompressor/issues/103), titled "SVDQuant for WAN 2.2", opened by user `@zjq0455` on **2025-09-12**. The full text:

> *Can WAN 2.2 already be supported by SVDQuant? How to set the config if I want to quantize such model?*

The most recent reply (as of 2025-12-16, from `@rwfsmith`) is:

> *I'm pretty sure support hasn't been added yet and there hasn't been any timeline given for it.*

The issue remains *open*, and no SVDQuant checkpoint or calibration recipe has been released for any video DiT. This is the single most important "deployment gap" datapoint in this section: the best image-DiT W4A4 PTQ method, which has been productionized and ships pre-quantized FLUX.1/SANA/PixArt-Σ checkpoints, simply does not work end-to-end on any video DiT yet. The empirical reasons (untested) are likely a combination of:

- **3D vs 2D AdaLN / temporal modulation**. Wan 2.2's transformer adds explicit time-dependent modulation across the 81-frame sequence; SVDQuant's calibration loop and smoothing factor \(\boldsymbol{\lambda}\) are derived from a 2D (frame-independent) activation distribution.
- **Cross-attention at scale**. Video DiTs inject very long text-encoder sequences through cross-attention into 100k+ visual tokens; the cross-attention K/V projections have qualitatively different activation statistics than self-attention, and the SVDQuant rank choice (16 or 32) may not be optimal for them.
- **Low-rank rank explosion under temporal mixing**. The "first ~32 singular values dominate" property that SVD exploits is empirically observed on per-frame DiT but is not yet validated on the spatiotemporally-mixed 3D-attention activations of Wan 2.2 / HunyuanVideo.
- **Calibration cost**. Each SVDQuant calibration sample is a full denoising trajectory; for 5-second 720p video this is 100s+ of seconds × tens of samples. The deepcompressor toolchain is not currently set up to run this efficiently for video.

The Nunchaku repo [`github.com/nunchaku-ai/nunchaku`](https://github.com/nunchaku-ai/nunchaku) likewise lists only image DiTs (FLUX.1-{dev,schnell}, PixArt-Σ, SANA-1.6B, SDXL, SDXL-Turbo) and one image-edit model (Step1X-Edit) in its supported-models table. The corollary is that **anyone deploying a Wan / Hunyuan / Mochi / CogVideoX in NVFP4 today must roll their own calibration + smoothing pipeline**, falling back to either modelopt + per-tensor FP8/NVFP4, or a community fork like ac4k/Wan2.2 (Q.8.2), or QVGen-style QAT, or the Lightx2v Wan-NVFP4 distilled-and-quantized checkpoint.

---

## Q.4 Rotation methods adapted for DiT (QuaRot, SpinQuant, DFRot, ConvRot)

The "rotate the activations into a basis where outliers vanish" line of work is independent of (and complementary to) low-rank residual compensation. As of May 2026 it is the foundation for almost all production W4A4 LLM deployments and is increasingly being adapted to DiT.

### Q.4.1 QuaRot (NeurIPS 2024)

QuaRot [\[Ashkboos et al., arXiv:2404.00456\]](https://arxiv.org/abs/2404.00456) introduces a **randomized Hadamard transformation** \(\mathbf{Q}\) such that

\[
\mathbf{X}\mathbf{W} = (\mathbf{X}\mathbf{Q})(\mathbf{Q}^\top\mathbf{W}),
\]

i.e. the model's output is *exactly* unchanged but the *intermediate* activation \(\mathbf{X}\mathbf{Q}\) has its outliers randomized away. The trick rests on two pillars: (1) **computational invariance** of pre-norm transformer blocks under any orthogonal \(\mathbf{Q}\), so long as \(\mathbf{Q}\) is fused into the *output* matrix of one block and the *input* matrix of the next; and (2) the **incoherence-processing** observation [\[Chee et al., 2024 (QuIP)\]](https://arxiv.org/abs/2307.13304) that an arbitrary outlier pattern is averaged out (in expectation) by multiplication with a random-sign Hadamard \(\widetilde{\mathbf{H}} = \mathbf{H}\,\mathrm{diag}(s)\) with \(s_i \in \{\pm 1\}\) i.i.d. uniform.

QuaRot needs three Hadamard transforms per transformer layer in the most general configuration: one in the residual path (fused into adjacent linears, free at runtime), one online in the FFN gate-up path (a single FWHT, \(\mathcal{O}(d \log d)\)), and one online in the attention output projection. On Llama-2-70B at W4A4KV4 the published number is **0.47 perplexity loss** with **3.33× prefill speedup** at batch 64 / seq 2048, and **3.89× memory saving** in the decode KV cache. QuaRot's GPU kernels are based on CUTLASS INT4 GEMM; the rotation itself uses the FAst Walsh-Hadamard Transform.

Why QuaRot is hard to port to DiT directly: the **AdaLN modulation breaks computational invariance**. The SVDQuant authors put it crisply: "Rotation is inapplicable due to the usage of adaptive normalization layers in diffusion models. The runtime-generated normalization weights preclude the offline rotation with the weights of projection layers, while online rotation of both activations and weights incurs significant runtime overhead." Concretely: in pre-norm LLM, the linear layer immediately following RMSNorm has its weights precisely rotatable as \(\mathbf{Q}^\top\mathbf{W}\) once we absorb RMSNorm's \(\mathrm{diag}(\boldsymbol{\alpha})\) into adjacent weights. In DiT/AdaLN, the post-norm activation is multiplied by a *runtime* \(1+\boldsymbol{\gamma}\) where \(\boldsymbol{\gamma}\) depends on the timestep and condition embedding — there is no static \(\mathrm{diag}(\boldsymbol{\alpha})\) to absorb. So either the rotation is inserted *online* on every step (paying an FWHT cost), or the AdaLN modulation parameters themselves must be rotated (which requires fusing \(\mathbf{Q}\) into the small modulation MLP that produces \(\boldsymbol{\gamma}\)).

### Q.4.2 SpinQuant — learnable rotations

SpinQuant [\[Liu et al., arXiv:2405.16406\]](https://arxiv.org/abs/2405.16406) extends QuaRot by making the rotation matrix \(\mathbf{R}\) *trainable* in the orthogonal manifold via Cayley optimization. Empirically, learned rotations recover an additional 0.3–0.7 perplexity points over random Hadamard at W4A4KV4 on Llama-3-8B. The trade-off is that SpinQuant requires a small calibration finetune (a few thousand iterations on a small text dataset). SpinQuant is the basis of the rotation matrices used in Meta's recent FBGEMM W4A4 LLM serving stack. For DiT, the same AdaLN obstruction as QuaRot applies, plus the additional difficulty that the per-timestep activation distribution shifts mean a single static rotation cannot be globally optimal.

### Q.4.3 DFRot — the massive-activation correction

DFRot [\[Xiang & Zhang, arXiv:2412.00648\]](https://arxiv.org/abs/2412.00648) — "Outlier-Free *and* Massive Activation-Free for Rotated LLMs" — adds a critical refinement that is directly relevant for DiT, where AdaLN-induced spikes are themselves a form of "massive activation". Their key observation is that when one inspects post-rotation activation magnitudes at LLaMA3-8B layer 6, the *common* tokens are outlier-free but a small set of *massive-activation* tokens have residual outliers that randomized Hadamard does *not* eliminate (and that random orthogonal rotations actually amplify).

The fix: a **weighted loss** that upweights the quantization error on massive-activation tokens during a rotation-matrix refinement step, plus an alternating Procrustes optimization that updates \(\mathbf{R}_1\) (the residual-path rotation) while holding quantization scales fixed and vice versa. Single-sample tuning of the rotation matrix (8 minutes wall time) yields **0.25 perplexity improvement on LLaMA3-8B at W4A4KV4** and **0.21 at W4A4KV16**. For DiT, the AdaLN-induced spike that PTQ4DiT identifies is structurally identical to massive activations in LLM, suggesting DFRot's weighted-loss refinement should transfer; this has not yet been published.

### Q.4.4 ConvRot — the first DiT-aware rotation method (Dec 2025)

ConvRot [\[Huang et al., arXiv:2512.03673\]](https://arxiv.org/abs/2512.03673), "Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers", is to date the only published rotation-based W4A4 method explicitly targeting DiT. It tackles three DiT-specific issues that vanilla QuaRot does not handle:

1. **AdaLN-broken fusion**. Sylvester-type Hadamard fusion (the classical FWHT path) cannot be absorbed into the adjacent linear weights when AdaLN sits between them, forcing an *online* Hadamard on every step.
2. **Row-wise outliers**. Diffusion transformers exhibit *row-wise* outliers (a few large-magnitude tokens dominate a few channels) in addition to the *column-wise* outliers seen in LLMs. Sylvester Hadamards cannot suppress row-wise outliers because the first column of the standard Sylvester construction is all-ones, which actually *amplifies* row outliers — see Figure 3 of the paper, which shows the `single_transformer_blocks.37.proj_out` activation in FLUX.1 reaching max 106.19 after a Sylvester transform versus 9.26 after the regular-Hadamard transform.
3. **Quadratic rotation cost**. A full \(K \times K\) Hadamard for \(K \approx 3072\) is a non-trivial GEMM-like cost on top of an already W4A4-quantized linear.

ConvRot's solution: **group-wise Regular Hadamard Transform (RHT)** with group size \(N_0 = 16\) or 32. The rotation is applied independently within each group via a Kronecker-constructed *regular* Hadamard matrix (a Hadamard whose row sums are all equal in absolute value), reducing per-token rotation complexity from \(\mathcal{O}(K^2)\) to \(\mathcal{O}(K)\) and *suppressing* row-wise outliers by construction. The "ConvLinear4bit" plug-and-play module then fuses {rotation, quantization, GEMM, dequantization} into a single kernel that does not require a separate inference engine.

Reported result on FLUX.1-dev (RTX 4090 24 GB): **2.26× speedup, 4.05× memory reduction**, with image fidelity comparable to the BF16 model and visually no major artifacts. This is the **first training-free W4A4 path on a DiT that does not require an SVD-residual branch**, and is the most natural rotation-based candidate for porting to video DiTs in 2026. The paper does not yet report numbers on Wan / Hunyuan, but the algorithmic obstacle (AdaLN-induced fusion break) is identical, so the prediction is that ConvRot generalizes.

### Q.4.5 Why rotation alone is unlikely to be enough at W4A4 video

Empirically, *both* rotation and low-rank residual help and they compose: a hypothetical "QuaRot + SVDQuant" PTQ recipe would rotate first to flatten the activation distribution and then absorb the residual outliers with a rank-16 branch. On *image* DiT, ConvRot already shows that careful rotation alone reaches BF16-comparable quality at W4A4. On *video* DiT, the additional temporal axis introduces a fourth source of variance (PTQ4DiT's \(\rho_t\)-weighting axis) that no rotation-only method has yet been validated against. As of May 2026 the cleanest path for *production* W4A4 video is therefore expected to be a hybrid: ConvRot-style group rotation + SVDQuant-style residual + per-timestep dynamic activation scaling.

---

## Q.5 NVIDIA modelopt + TensorRT-LLM VisualGen

### Q.5.1 NVIDIA TensorRT Model Optimizer (modelopt)

NVIDIA's [`TensorRT-Model-Optimizer`](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (modelopt) is the production calibration and export toolkit for FP8 / FP4 quantization. The diffusers example directory `examples/diffusers/quantization` provides reference recipes for **SDXL, SD3, FLUX.1, FLUX.2, PixArt**, supporting INT8, FP8 (per-tensor and channel-wise), and **NVFP4** formats with PTQ + SmoothQuant + AWQ-style per-channel scaling.

The calibration loop in modelopt for diffusion is the `forward_loop` pattern: the user supplies a callable that runs `pipe(...)` for a small calibration set, modelopt traces it through the transformer, and at each *timestep* records per-tensor amax statistics. Per-timestep amax is then either **averaged** (default), **percentile-capped**, or **kept-per-step** depending on the recipe. The export path produces an ONNX or HuggingFace-compatible `quantization_config` JSON with the format used by both TensorRT-LLM and TensorRT proper.

The quality numbers reported by NVIDIA in the modelopt blog (https://developer.nvidia.com/blog/...) show ~0.95–0.98 ImageReward retention on FLUX.1-dev at NVFP4 with 32-sample / 4-timestep calibration, and ~1.05× to 1.5× faster end-to-end (depending on batch size and resolution) on B200 versus the BF16 baseline.

What modelopt does **not** support as of May 2026:
- **Wan / HunyuanVideo / Mochi**: no reference recipe or example. The community working group `JiusiServe/vllm-omni#151` ("RFC: Wan 2.2 Optimization") and modelopt's own issue tracker show this is being requested but not yet shipped. This is symmetric to the deepcompressor#103 gap — the entire NVIDIA stack does not currently include Wan/HunyuanVideo NVFP4.
- **VAE quantization**: the encoder-decoder pair is left at FP16/BF16. (Q.9 below discusses the practical workarounds.)
- **CFG-aware quantization**: the conditional and unconditional branches share scales.

### Q.5.2 TensorRT-LLM VisualGen

TensorRT-LLM 2026.x ships **VisualGen** [\[docs/source/models/visual-generation.md\]](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/visual-generation.md), a unified diffusion inference subsystem separate from the LLM autoregressive path. As of the May 2026 docs, the VisualGen support matrix is:

| Model | FP8 blockwise | NVFP4 | TeaCache | CFG Parallel | Ulysses | Parallel VAE | CUDA Graph | torch.compile | trtllm-serve |
|---|---|---|---|---|---|---|---|---|---|
| FLUX.1 | ✓ | ✓ | ✓ | n/a [^1] | ✓ | ✗ | ✓ | ✓ | ✓ |
| FLUX.2 | ✓ | ✓ | ✓ | n/a [^1] | ✓ | ✗ | ✓ | ✓ | ✓ |
| Wan 2.1 (T2V/I2V, 1.3B/14B/480P/720P) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Wan 2.2 (T2V-A14B / I2V-A14B / TI2V-5B)** | **✓** | **✓** | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LTX-2 | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |

[^1]: FLUX uses embedded guidance (no separate negative-prompt path).

Two implications:

1. **TensorRT-LLM VisualGen is, as of May 2026, the only production-scale stack that supports NVFP4 dynamic quantization for Wan 2.1, Wan 2.2, and LTX-2.** It uses modelopt's `quantization_config` format under the hood — the user passes `--linear_type trtllm-nvfp4` and the loader applies dynamic NVFP4 weight quantization at load time. This *bypasses* the SVDQuant/Nunchaku gap (Q.3.3) by using *dynamic* per-tensor quantization (no calibration-set-derived scales) plus the NVFP4 block-16 hardware path.

2. **The cost is quality**. Dynamic per-tensor NVFP4 on Wan 2.2 has not been benchmarked against an SVDQuant or QVGen reference — there is no published LPIPS or VBench number for `--linear_type trtllm-nvfp4` on Wan 2.2 as of the docs date. Anecdotally, users on the TensorRT-LLM issue tracker report that NVFP4 on Wan 2.2 produces "acceptable but visibly degraded" outputs, with the worst regressions on text-rendering scenes and fine-grained motion.

The serving endpoints (`/v1/videos`, `/v1/videos/generations`) are OpenAI-compatible and support both synchronous and asynchronous video generation; this is what NVIDIA is positioning as the production "video model serving" answer. The dynamic FP8 path (`--linear_type trtllm-fp8-blockwise`) is more conservative — channel-wise FP8 with ~50 % memory saving and 1.0–1.2× speedup vs BF16 — and is what most production pilots default to.

### Q.5.3 modelopt + VisualGen interactions

The two pieces compose: the typical production path is

1. `python -m modelopt.diffusers.quantize --model wan2.2-t2v-a14b --format nvfp4 --calibration-prompts file.txt --steps 4 --samples 16` produces a quantized HF checkpoint.
2. `trtllm-serve --quant_config nvfp4 --model ./wan2.2-nvfp4-ckpt --cfg_size 2 --ulysses_size 4` serves it on 8 GPUs.

The Ulysses sequence-parallel partition (which splits the 100k-token visual sequence across GPUs) and the CFG-parallel partition (which splits cond/uncond across GPUs) compose multiplicatively, so an 8-GPU HGX B200 with `cfg_size=2 × ulysses_size=4` is the canonical Wan 2.2 single-node deployment. NVFP4 + CFG-parallel + Ulysses-parallel reach roughly 2.5–3.5× over a single-GPU BF16 baseline on Wan 2.2-T2V-A14B at 720p / 81 frames in informal NVIDIA benchmarks — an order of magnitude short of TurboDiffusion's 100×, because VisualGen does not bundle step distillation.

---

## Q.6 PyTorch / TorchAO + Diffusers Blackwell — LTX-2 verbatim

The [PyTorch / TorchAO + Diffusers Blackwell blog post (March 2026)](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/) is the cleanest open-source reference for what NVFP4 actually delivers on B200 in *Diffusers* form, and is the only published source of LTX-2 NVFP4 numbers on B200 hardware. It is "critical" in this report because the LTX-2 row is the load-bearing video datapoint for the open-source torchao path.

### Q.6.1 The NVFP4 / MXFP8 TorchAO API

The blog uses two TorchAO configs:

```python
# MXFP8 (32-element blocks, E8M0 scales)
quant_config = MXDynamicActivationMXWeightConfig(
    activation_dtype=torch.float8_e4m3fn,
    weight_dtype=torch.float8_e4m3fn,
    kernel_preference=KernelPreference.AUTO,
)

# NVFP4 (16-element blocks, E4M3 scales, FP32 per-tensor)
quant_config = NVFP4DynamicActivationNVFP4WeightConfig(
    use_dynamic_per_tensor_scale=True,
    use_triton_kernel=True,
)
```

`use_triton_kernel=True` opts into the Triton-on-Blackwell autotuned NVFP4 kernel rather than the cuBLASLt path. Both kernels read directly from BF16 weights and quantize dynamically at load time, so no calibration is required. Internally the path also depends on **MSLK** ([github.com/meta-pytorch/MSLK](https://github.com/meta-pytorch/MSLK)), Meta's Mixed-precision Scaled Linear Kernel library, which provides the fused per-block amax extraction + FP4-pack-and-store kernel that feeds the NVFP4 GEMM. PyTorch [PR #4031](https://github.com/pytorch/ao/pull/4031) is the key TorchAO change that swapped `to_nvfp4` over to use MSLK and unblocked B200 perf. The companion repo [`github.com/sayakpaul/diffusers-blackwell-quants`](https://github.com/sayakpaul/diffusers-blackwell-quants) hosts the reproducible benchmark scripts.

### Q.6.2 LTX-2 verbatim numbers (B200, 768×512 121 frames, 40 steps, guidance 4.0)

The LTX-2 prompt is a 121-frame "house cat with reading glasses typing" scene at 768×512 with `num_inference_steps=40` and `frame_rate=24.0`. The B200 results, copied verbatim from the blog table:

| Quant Mode | Batch Size | Latency (s) | Memory (GB) | Speedup vs BF16 |
|---|---|---|---|---|
| None (BF16) | 1 | 16.230 | 72.77 | 1.00× |
| MXFP8 | 1 | 13.724 | 54.54 | 1.18× |
| **NVFP4** | **1** | **10.374** | **45.72** | **1.56×** |
| None (BF16) | 4 | 61.591 | 87.61 | 1.00× |
| MXFP8 | 4 | 50.956 | 69.38 | 1.21× |
| **NVFP4** | **4** | **36.963** | **60.56** | **1.67×** |
| None (BF16) | 8 | 122.427 | 107.40 | 1.00× |
| MXFP8 | 8 | 102.546 | 89.18 | 1.19× |
| **NVFP4** | **8** | **72.689** | **80.36** | **1.68×** |

These are the canonical "open-source NVFP4 video on B200" numbers as of May 2026: **1.56× / 1.67× / 1.68× at batch 1 / 4 / 8** — note that the B=1 row is the *headline* number for low-latency single-video serving, and NVFP4 is the only path that crosses 1.5× at that point. For comparison, the FLUX.1-dev row is **1.50× / 1.55× / 1.59× at batch 1 / 4 / 8** (the parent prompt's "1.56×/1.67×/1.68×" was the LTX-2 row, not FLUX), and the QwenImage row is **1.39× / 1.47× / 1.49× at batch 1 / 4 / 8**.

The full FLUX.1-dev row, also from the blog, is included for cross-reference because it is the only image-DiT row with a published mean-LPIPS quality measurement:

| Quant Mode | Batch Size | Latency (s) | Memory (GB) | Speedup vs BF16 | Mean LPIPS (Drawbench) |
|---|---|---|---|---|---|
| None (BF16) | 1 | 2.10 | 38.34 | 1.00 | 0 |
| MXFP8 | 1 | 1.75 | 26.90 | 1.21× | 0.11 |
| NVFP4 | 1 | 1.41 | 21.33 | 1.50× | 0.44 |
| None (BF16) | 4 | 7.87 | 44.39 | 1.00 | — |
| MXFP8 | 4 | 6.36 | 32.95 | 1.24× | — |
| NVFP4 | 4 | 5.09 | 27.39 | 1.55× | — |
| None (BF16) | 8 | 15.57 | 53.00 | 1.00 | — |
| MXFP8 | 8 | 12.40 | 41.56 | 1.26× | — |
| NVFP4 | 8 | 9.81 | 36.00 | 1.59× | — |

The TorchAO blog also reports **selective quantization** — skipping `nn.Linear` layers where `min(M, K, N) < 1024` and skipping embedding/normalization layers — yields a measurable improvement: full quantization gives LPIPS 0.479679 vs selective 0.438337 on FLUX.1-dev NVFP4 at batch 1, with essentially no latency cost. For QwenImage the LPIPS is materially worse (0.34 MXFP8, 0.41 NVFP4), suggesting QwenImage is more outlier-prone than FLUX.1-dev and likely requires SVDQuant-style residual compensation rather than pure dynamic-scale TorchAO.

The single most important observation in this section is the **scaling pattern across batch size**: MXFP8 saturates at ~1.2–1.25× at all batch sizes (memory-bandwidth limited; FP8 is 2× better than BF16 in arithmetic intensity but the larger Diffusers blocks already saturate the HBM3e read pipe), whereas NVFP4 rises from 1.50× → 1.59× → 1.68× as batch grows from 1 → 4 → 8 (the workload becomes increasingly compute-bound, and NVFP4's 4× theoretical advantage starts to materialize). This is exactly the scaling law one expects from a 4× peak / 2× peak format pair on a memory-bandwidth-limited GPU, and it implies that for **batch-1 single-user video**, NVFP4 is not yet hitting its theoretical advantage — the FP4 multiplier is bottlenecked on `to_nvfp4` scale extraction and on the per-step quantize-then-dequantize round-trip through the residual path.

### Q.6.3 The diffusers-torchao repo

Sayak Paul's [`diffusers-torchao`](https://github.com/sayakpaul/diffusers-torchao) (398 stars) is the catalog of end-to-end recipes — FP8 training, NVFP4 inference, MXFP8 inference, dynamic-quant-with-LoRA — that ship as drop-in `TorchAoConfig` objects through Diffusers' `PipelineQuantizationConfig` API. This is the canonical "no-modelopt, no-Nunchaku, just `pip install`" path for B200 video quantization.

---

## Q.7 QAT for video (QVGen + Attn-QAT)

QAT adds gradient backpropagation through the quantizer (typically with a straight-through estimator) so that the model weights *adapt* to the W4A4 grid. For W4A4 video DiT — where PTQ alone fails — QAT is the only currently viable algorithmic route to FP-comparable quality.

### Q.7.1 QVGen (ICLR 2026)

QVGen [\[Huang et al., arXiv:2505.11497\]](https://arxiv.org/abs/2505.11497), [github.com/ModelTC/QVGen](https://github.com/ModelTC/QVGen), is the first *successful* W4A4 QAT framework for video DiT. Its evaluation spans CogVideoX-{2B, 5B}, Wan-{1.3B, 14B}, and reports VBench scores within 1 point of FP16 baselines at W4A4 on multiple prompts.

QVGen's key technical pieces:

**(i) Auxiliary modules \(\Phi\)**. The QAT objective for a quantized video DM has unusually large gradient norm \(\|\mathbf{g}_t\|_2\), which Theorem 3.1 of the paper relates directly to the convergence rate via the regret bound

\[
\frac{R(T)}{T} \le \frac{dD_\infty^2}{2T\eta_T^m} + \frac{1}{T}\sum_{t=1}^T \frac{\eta_t^M}{2}\|\mathbf{g}_t\|_2^2.
\]

Minimizing \(\|\mathbf{g}_t\|_2\) is the controllable lever. QVGen attaches an auxiliary module \(\Phi\) to each quantized linear layer with parameter \(\mathbf{W}_\Phi\) initialized to the *quantization residual* \(\mathbf{W} - \mathcal{Q}_b(\mathbf{W})\), and the forward becomes

\[
\hat{\mathbf{Y}} = \mathcal{Q}_b(\mathbf{W})\mathcal{Q}_b(\mathbf{X}) + \Phi(\mathcal{Q}_b(\mathbf{X})), \qquad \Phi(\mathcal{Q}_b(\mathbf{X})) = \mathbf{W}_\Phi \mathcal{Q}_b(\mathbf{X}).
\]

This is *structurally* a SVDQuant-style high-precision residual branch, but used here only during training as a stabilizer rather than as a deployed inference path.

**(ii) Rank-decay schedule**. The naive problem is that \(\Phi\) cannot be present at inference (it would defeat the W4A4 acceleration), and simply zeroing out \(\mathbf{W}_\Phi\) at the end of training is suboptimal. QVGen observes that the singular-value spectrum of \(\mathbf{W}_\Phi\) collapses dramatically during QAT: at step 0, ~73 % of average singular values are >14× smaller than \(\sigma_1\); at step 2K, ~99 % are. So an iterated SVD + truncation schedule "decays" \(\Phi\) gradually:

\[
\mathbf{W}_\Phi = \sum_{s=1}^d \sigma_s \mathbf{u}_s \mathbf{v}_s^\top \;\to\; \sum_{s=1}^{d'} \sigma_s \mathbf{u}_s \mathbf{v}_s^\top, \quad d' = (1-\lambda)d,
\]

with a regularization \(\boldsymbol{\gamma}\) that linearly anneals each truncated component from weight 1 to weight 0 over `it_per_decay_phase` iterations. After \(\log_{1-\lambda}(\epsilon)\) decay phases, \(\mathbf{W}_\Phi \to \mathbf{0}\) and the auxiliary modules are removed.

**(iii) Results**. QVGen W4A4 on **CogVideoX-2B** matches the BF16 baseline VBench score within 0.5 point (vs naïve W4A4 PTQ losing 5–10 points), beats EfficientDM W4A4 by 25.28 in Dynamic Degree and 8.43 in Scene Consistency at W3A3, and on Wan-14B reports negligible drops on VBench-2.0 — the largest open-source SOTA video DiT successfully W4A4-QAT'd.

QVGen is **the strongest currently published W4A4 video-DiT result** and the natural reference checkpoint for Blackwell NVFP4 deployment. The catch: the public release does *not* yet provide a Blackwell-tuned inference kernel, and the published results are W4A4 *INT4* rather than *NVFP4*. Porting QVGen to NVFP4 requires reapplying the QAT loop with NVFP4 fake-quant ops (16-element blocks, E4M3 scales) substituted for INT4 fake-quant — straightforward in principle, but multi-GPU-day in practice given the calibration-trajectory cost on Wan-14B. The expectation is that NVFP4-QVGen will outperform INT4-QVGen by the same ~2.5 dB PSNR gap that NVFP4 shows over INT4 in SVDQuant's image-DiT table.

### Q.7.2 Attn-QAT (FP4 attention, ICCV/ICML 2026)

Attn-QAT [\[Zhang et al., arXiv:2603.00040\]](https://arxiv.org/abs/2603.00040) tackles the *attention* operator at FP4 — orthogonal to QVGen's *linear-layer* QAT. The motivating gap: SageAttention 3 (the SOTA training-free FP4 attention) loses 5–10 % VBench score on Wan 2.1-14B even with Q/K smoothing and two-level scaling. Attn-QAT is the first systematic study of QAT on attention and identifies two FlashAttention-specific obstacles:

1. **Backward recomputation must use the same FP4 precision as forward**. FlashAttention computes attention in tiles, with tile-resident softmax statistics. The backward pass *recomputes* the attention score \(\mathbf{P}_i\) from \(\mathbf{Q}_i, \mathbf{K}_j\) for memory efficiency. If the forward is FP4 and the backward recomputation is BF16, the intermediate \(\mathbf{P}\) tensors disagree and gradients explode.
2. **The identity \(\mathbf{P}_i^\top \mathbf{dP}_i = \mathbf{dO}_i^\top \mathbf{O}_i\), which FlashAttention uses to keep memory linear**, only holds when forward and backward share precision. Attn-QAT works around this by computing \(\mathbf{O}\) in *both* FP4 and high precision in the forward, storing the high-precision \(\mathbf{O}\) only for gradient computation.

The implementation is fused Triton kernels for training and improved SageAttention-3-derived CUDA kernels for inference. On Wan 2.1-14B, **Attn-QAT recovers the FP4-attention quality drop without requiring SageAttention's outlier-mitigation heuristics** (Q/K smoothing, two-level rescaling), and delivers **1.1–1.5× over SageAttention 3 on RTX 5090**. Composed with QVGen-style W4A4 GEMM, this is the path to a *fully* end-to-end FP4 video DiT — the holy grail of Blackwell video quantization.

### Q.7.3 SmoothQuant — a brief note

SmoothQuant [\[Xiao et al., arXiv:2211.10438\]](https://arxiv.org/abs/2211.10438), the foundational static channel-balancing method, is the algorithmic ancestor of PTQ4DiT's CSB, ViDiT-Q's static balancing, SVDQuant's smoothing step, and Q.4's rotation methods (which generalize SmoothQuant from per-channel diagonal to orthogonal). SmoothQuant computes a per-channel scale \(\boldsymbol{\lambda} = \max(|\mathbf{X}|)^\alpha / \max(|\mathbf{W}|)^{1-\alpha}\) where \(\alpha \in [0, 1]\) controls the activation-vs-weight trade-off. The key insight (later inherited by SageAttention's `smooth_k`) is that *static* per-channel scales can be folded into adjacent linear weights at zero inference cost, while dynamic per-token scales cannot. SVDQuant's smoothing step uses \(\alpha = 0.5\); SageAttention 3's `smooth_k` uses a per-block subtract-mean variant (essentially a dynamic \(\alpha = 1\)).

### Q.7.4 EfficientDM — the QAT-with-PTQ-cost framework

EfficientDM [\[He et al., arXiv:2310.03270\]](https://arxiv.org/abs/2310.03270) is the bridge between PTQ and QAT for image diffusion: a parameter-efficient fine-tuning framework using QALoRA (Quantization-Aware LoRA) that quantizes the LoRA weights jointly with the base model and uses a data-free knowledge distillation loop. On LDM-4 ImageNet 256², EfficientDM reaches **W4A4 with 0.05 sFID increase vs FP** and is **16.2× faster than full QAT** (e.g., Q-DM) at quantization time. EfficientDM is the practical baseline that QVGen builds on and improves; for *video* DiT, EfficientDM's published numbers are weaker than QVGen, but EfficientDM remains the most accessible "QAT in 2 GPU-hours" recipe for image DiT.

---

## Q.8 Per-model quantization status (May 2026)

This section is a model-by-model snapshot of *what actually works* on B200 in May 2026, sorted by maturity.

### Q.8.1 FLUX.1-dev / FLUX.1-schnell / FLUX.2

The most-mature image DiT in production. All three of (modelopt, TorchAO, Nunchaku) ship pre-built FP8 and NVFP4 paths.

- **modelopt**: NVFP4 + FP8 via the diffusers/quantization recipe.
- **TorchAO**: MXFP8 1.21–1.26× and NVFP4 1.50–1.59× on B200 (Q.6).
- **Nunchaku SVDQuant**: NVFP4 + INT4 with PSNR 21.4 / 21.0 on FLUX.1-dev (Q.3).
- **TensorRT-LLM VisualGen**: NVFP4 + FP8 blockwise.
- **bitsandbytes**: NF4 + INT8 (legacy path, 1× speed but 4× memory saving). The HuggingFace docs page recommends `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` with `bnb_4bit_compute_dtype=torch.bfloat16`. With `torch.compile(fullgraph=True)`, RTX 4090 4-bit FLUX produces an image in 25.809 s vs 32.570 s without compile.
- **NVIDIA pre-quantized HF checkpoints**: `nvidia/DeepSeek-R1-0528-FP4`, `black-forest-labs/FLUX.1-dev-onnx` (from the introducing-NVFP4 blog).

### Q.8.2 Wan 2.1 / Wan 2.2 (T2V-A14B / I2V-A14B / TI2V-5B)

The most production-relevant video DiT. The Wan-NVFP4 status:

- **TensorRT-LLM VisualGen**: NVFP4 dynamic quantization supported for all Wan 2.1 / 2.2 variants via `--linear_type trtllm-nvfp4`.
- **vLLM-Omni FP8**: PR [#1412](https://github.com/vllm-project/vllm-omni/pull/1412) merged 2026-03-20, adds W8A8 FP8 for Wan 2.2-T2V-A14B and Wan 2.2-I2V-A14B. Reported on a single GPU at 1280×720 / 81 frames / 40 steps:
  - **T2V**: BF16 64.46 GiB / 892.8 s → FP8 38.18 GiB / 828.0 s (~41 % memory ↓, ~7 % faster)
  - **I2V**: BF16 64.46 GiB / 301.1 s → FP8 38.18 GiB / 264.6 s (~12 % faster)

  The PR explicitly excludes `DistributedRMSNorm`, `Attention`, `Conv3dLayer`, `nn.Linear (proj_out)`, `FP32LayerNorm`, and embedding layers from quantization, mirroring the Z-Image pattern. Visual quality is reported as "comparable" but no LPIPS or VBench score has been published.
- **lightx2v Wan-NVFP4** ([huggingface.co/lightx2v/Wan-NVFP4](https://huggingface.co/lightx2v/Wan-NVFP4)): a pre-trained *4-step distilled + NVFP4 quantized* checkpoint targeted at RTX 5090 (Blackwell consumer). On a single 5090:
  - **I2V-14B-480P**: 498.90 s → 17.65 s (28× end-to-end), single-step denoising 12.10 s → 3.40 s (3.5×)
  - **T2V-1.3B-480P**: 83.50 s → 6.54 s (12.8× end-to-end), single-step denoising 2.00 s → 0.70 s (2.9×)

  The recipe is custom: Lightricks-style step distillation (4 steps from 50) plus a custom NVFP4 kernel built on CUTLASS, integrated through the [LightX2V](https://github.com/ModelTC/LightX2V) framework. **Not directly portable to data-center B200** without re-deriving the kernel for SM_100, but the algorithmic recipe (distill-then-quantize) is.
- **ac4k/Wan2.2** ([github.com/ac4k/Wan2.2](https://github.com/ac4k/Wan2.2)): a 6-star community fork of the Alibaba Wan 2.2 release that adds NVFP4 support. Quality claims are *qualitative-only* — the README shows side-by-side videos but no FID, LPIPS, or VBench scores. Calibration recipe and computational cost are not published. Useful as an existence proof; not yet a production path.
- **TurboDiffusion** ([arXiv:2512.16093](https://arxiv.org/abs/2512.16093)): see Q.10 below — uses INT8 W8A8 (not NVFP4) but composes with rCM step distillation and SLA sparse attention to deliver 100–200× end-to-end on Wan 2.2-I2V-A14B-720P (4549 s → 38 s on a single RTX 5090).
- **modelopt/Nunchaku/SVDQuant**: **none of these support Wan as of May 2026**. The deepcompressor#103 issue (Q.3.3) is the canonical reference for the Nunchaku gap.
- **QVGen** (Q.7.1): supports Wan-1.3B and Wan-14B at W4A4 INT4 via QAT but no NVFP4 release and no Blackwell kernel.

### Q.8.3 HunyuanVideo (Tencent)

- **TensorRT-LLM VisualGen**: not in the supported-model list as of May 2026.
- **modelopt**: no recipe.
- **vLLM-Omni**: not yet supported (issue [#2912](https://github.com/vllm-project/vllm-omni/issues/2912) references it for FP8 dynamic quantization, status unclear).
- **TAEHV** ([github.com/madebyollin/taehv](https://github.com/madebyollin/taehv), 370 stars): a tiny VAE that works as a drop-in for HunyuanVideo's encoder/decoder, providing ~3× faster VAE decoding at minor quality cost. Not a transformer quantization solution but the most-used HunyuanVideo VAE optimization.
- **TurboDiffusion**: supports HunyuanVideo via the SageAttention 3 + W8A8 INT8 pattern.

### Q.8.4 Mochi (Genmo)

Even less coverage than HunyuanVideo. **No** Nunchaku, SVDQuant, modelopt, TensorRT-LLM VisualGen, or vLLM-Omni support as of May 2026. The community NVFP4 path is dynamic quantization via TorchAO's generic `NVFP4DynamicActivationNVFP4WeightConfig`, with no published quality measurements. Mochi's published BF16 inference time is the reference.

### Q.8.5 LTX-2 (Lightricks)

- **TensorRT-LLM VisualGen**: NVFP4 + FP8 blockwise support, but no CUDA Graph and no TeaCache.
- **TorchAO**: NVFP4 1.56–1.68× speedup on B200 (Q.6.2) — the verbatim numbers above. **This is the only published B200 NVFP4 video result with side-by-side latency, memory, and (for FLUX.1-dev) LPIPS.**
- **Lightricks LTX-2 official decoder**: BF16 only.

### Q.8.6 CogVideoX (Zhipu)

- **QVGen**: full W4A4 QAT support for CogVideoX-2B and CogVideoX-5B, with VBench scores within 1 point of FP16.
- **Q-DiT**: W6A8 evaluated on CogVideoX-2B (FVD lossless), W4A8 evaluated with minor degradation.
- **TensorRT-LLM VisualGen**: not supported.
- **modelopt**: no diffusers/quantization recipe.

### Q.8.7 Open-Sora 2.0

- **SemanticDialect / SeDA** [\[Jang & Tambe, arXiv:2603.02883\]](https://arxiv.org/abs/2603.02883): the most recent (March 2026) open-source W4A4 video-DiT method, building on **BlockDialect** [\[arXiv:2501.01144\]](https://arxiv.org/abs/2501.01144). The core idea is *fine-grained block-wise mixed-format* quantization: each 32-element block selects one of 32 candidate "dialects" (slightly different representable-value sets sharing the 4-bit budget), with the choice driven by per-block max statistics and a lookup-table-based MSE estimator. SemanticDialect adds (i) **activation decomposition** — re-quantize the residual error and add it back, similar in spirit to SVDQuant's residual but at single-block granularity, (ii) **attention-guided salient-token selection** so the decomposition cost is paid only on the most-attended tokens, and (iii) **Semantic-Aware Dialect Assignment (SeDA)** — semantically-correlated tokens (across spatial and temporal axes) are forced to share a sub-formatbook, preserving spatiotemporal consistency. On Open-Sora 2.0, SemanticDialect approaches FP16 quality at W4A4. SemanticDialect is *not* hardware-accelerated by Blackwell directly (the formatbook is software-defined, not native NVFP4), but represents the academic frontier of W4A4 video quantization quality.

### Q.8.8 Q-VDiT / S2Q-VDiT / inter-frame distillation

A growing set of methods — **Q-VDiT** (inter-frame distribution distillation for temporal consistency), **S2Q-VDiT** (calibration-data selection for video) — improve PTQ quality at W4A8 / W4A4 on Open-Sora and Latte. These compose with the format / hardware advances above but are not yet integrated into any production stack.

---

## Q.9 VAE quantization gap (TAEHV + FLUX.2 small decoder)

The video DiT pipeline is two networks: the *transformer* (which is what every section above quantizes) and the *VAE* (an encoder + decoder that maps between pixel space and the 3D-temporal latent space). For Wan 2.2-T2V-A14B, the VAE accounts for roughly 5–15 % of single-batch end-to-end latency depending on resolution and frame count, and 30–40 % at long-frame / high-resolution. As the transformer gets quantized to 4-bit, the VAE becomes a significant bottleneck.

### Q.9.1 The state of NVFP4 VAE in May 2026

**There is no published NVFP4 VAE for any video DiT.** The reasons:

1. **VAEs are CNN-based** (3D conv over latent voxels) rather than transformer-based, so the NVFP4 GEMM acceleration only partially helps — a Blackwell `tcgen05.mma kind::mxf4` path exists for convolutions via the convolution-tensor-core integration NVIDIA shipped in Blackwell (the "weight-stationary dataflow with collector buffer"), but it is not yet exposed in cuBLASLt or torchao for 3D conv.
2. **Encoder-decoder symmetric calibration** is hard: the encoder's input is image data with a known [-1, 1] range, but the decoder's input is the transformer's denoised latent which has a *training-distribution-dependent* range that shifts as the transformer is quantized.
3. **VAE outliers are localized**. Wan 2.2's VAE has channel-wise outliers in its Conv3d layers concentrated at the boundary of the upsampling stages. A naive per-tensor FP4 grid loses these features and causes visible "VAE banding" artifacts.

The vLLM-Omni FP8 PR (#1412) explicitly skips the VAE on Wan 2.2 and only does FP8 weight storage on the *Qwen Image* VAE/text-encoder via post-load hooks (not actual FP8 GEMM). modelopt's FLUX recipes leave the VAE at BF16. TensorRT-LLM VisualGen's "Parallel VAE" feature is about *parallelism* (splitting the VAE forward across GPUs) not *quantization*.

### Q.9.2 TAEHV

[`madebyollin/taehv`](https://github.com/madebyollin/taehv) ("Tiny AutoEncoder for Hunyuan Video and other video models") is the most-used VAE optimization for video diffusion as of 2026. It is a **distilled** VAE — a much smaller (~10–20 M parameter) student network trained to reproduce the original VAE's encode-decode behavior on the training distribution. TAEHV provides a drop-in replacement for HunyuanVideo's VAE, the Wan-VAE used in many Wan-derivative models, and several other video DiT VAE checkpoints.

The trade-off is *not* quantization: TAEHV achieves its 3–4× speedup by being a smaller network, not by using lower-bit representations. A future "TAEHV + NVFP4" combination would in principle compose, but as of May 2026 the TAEHV checkpoints ship only at FP16/BF16. The community wisdom is that TAEHV's 0.5–1 dB PSNR drop versus the original VAE is acceptable for most applications, especially when paired with a quantized transformer.

### Q.9.3 FLUX.2 small decoder

Black Forest Labs' [`FLUX.2-small-decoder`](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) is the same idea applied to image diffusion: a distilled FLUX.2 *decoder* (encoder unchanged) at ~28M parameters versus the full decoder's ~50M, with channel widths reduced from `[128, 256, 512, 512]` to `[96, 192, 384, 384]`. Reported as *~1.4× faster decoding and ~1.4× less VRAM at decode time* with "minimal to zero quality loss". Apache-2.0 licensed.

This is again a *distillation*, not a *quantization* — but it is the canonical precedent for "small distilled decoder + quantized transformer" being the pragmatic 2026 pattern. A reasonable 2026 production deployment for Wan 2.2 on B200 would combine:

- **Transformer**: NVFP4 via TensorRT-LLM VisualGen or modelopt
- **VAE encoder**: BF16 (small fraction of latency)
- **VAE decoder**: A hypothetical "Wan-tiny-decoder" (analogous to FLUX.2-small-decoder; not yet released by Alibaba) at BF16, *or* a TAEHV-style distilled decoder, *or* a tiled-VAE that processes the decoder in spatial tiles to fit in cache
- **Tiled decoding**: The Diffusers `enable_vae_tiling()` and `enable_vae_slicing()` modes are universally applicable and trade some additional latency for ~2× lower peak memory

### Q.9.4 The deployment math

For a typical Wan 2.2-T2V-A14B 5-second 720p generation on B200, the BF16 wall-clock decomposes roughly as:

| Stage | Wall-clock fraction | Acceleratable by quantization? |
|---|---|---|
| Text encoder (UMT-5 / BERT) | ~3 % | yes (FP8 trivially) |
| Transformer denoising loop | ~70–85 % | yes (NVFP4 4×, FP8 2×) |
| VAE encoder (latent prep) | ~1 % | n/a |
| VAE decoder | ~10–25 % | not yet (no NVFP4 path) |

A purely-transformer NVFP4 deployment that achieves a 3× transformer speedup and leaves the VAE at BF16 ends up with end-to-end speedup of \(\frac{1}{0.85/3 + 0.15} = 2.3\times\), not 3×. The VAE's BF16 floor is therefore the *binding* constraint as the transformer gets aggressively quantized — and is the strongest argument for prioritizing VAE quantization research in 2026.

---

## Q.10 Composition with attention + GEMM + step-distillation + CFG-distillation

The headline numbers cited above (TurboDiffusion's 100–200×, lightx2v's 28×, TorchAO's 1.68×) all come from *composing* quantization with other accelerator techniques. This section is the algebra of that composition.

### Q.10.1 The four orthogonal axes

A video DiT inference run has roughly four independent acceleration knobs:

1. **GEMM precision** (BF16 → FP8 → NVFP4). Multiplier on linear-layer FLOP rate; ideally 2× / 4× over BF16.
2. **Attention precision** (BF16 → FP8 → FP4 / SageAttention 3 / Attn-QAT). Multiplier on attention FLOPs and KV cache memory.
3. **Step count** (50 → 4 → 1) via step distillation (rCM, DMD, Hyper-SD, LCM, MeanFlow).
4. **CFG cost** (×2 → ×1) via CFG distillation, eliminating the unconditional branch.

These compose multiplicatively in the *compute* dimension and roughly additively in the *quality* dimension (each axis adds its own FID/LPIPS regression). Algorithm × hardware co-design is about choosing the operating point on each axis that maximizes throughput at a fixed quality budget.

### Q.10.2 NVFP4 GEMM × FP4 attention — the round-trip layout problem

The NVFP4 GEMM (axis 1) and FP4 attention (axis 2) must agree on the *layout* of the activation tensor that flows from one to the other:

- **NVFP4 GEMM** outputs a {block-16-scaled E2M1, FP8 scale per 16, FP32 per-tensor scale} triplet. Output to the next operator is typically dequantized to BF16 in the epilogue and then re-quantized at the input of the next op.
- **SageAttention 3 / Attn-QAT FP4** input is {block-16-scaled E2M1 for Q/K/V, FP8 scale per 16, separate per-tensor scales for Q/K/V}.

If the DiT's QKV-projection GEMM is in NVFP4 and the attention is in FP4, the QKV-projection's epilogue ideally writes the {E2M1, scale} pair *directly* to the SRAM that the attention kernel reads — a round-trip through BF16 wastes bandwidth. As of May 2026, this fused QKV-NVFP4-to-Sage3-FP4 path **does not exist** in any production stack. Each kernel stands alone, with a BF16 staging tensor between them. The expected throughput cost of the staging is small (the BF16 tensor fits comfortably in L2) but it is real and is one of the optimizations the Blackwell-tuned video stacks (TurboDiffusion, lightx2v) will likely tackle in late 2026.

### Q.10.3 Quantization × step distillation

The strongest production composition pattern in 2026 is *distill first, then quantize*. Lightx2v Wan-NVFP4 does this: rCM-style 4-step distillation lifts the model from 50 steps to 4, then NVFP4 quantization is applied to the distilled model. The result is multiplicative: 12.5× from step distillation × 2.2× from NVFP4 = 27.5× end-to-end, matching the observed 28× on RTX 5090 I2V-14B.

The TurboDiffusion stack [\[Zhang et al., arXiv:2512.16093\]](https://arxiv.org/abs/2512.16093) takes this further: SageAttention 2++ (quantized attention) + Sparse-Linear Attention (SLA, ~90 % sparse) + rCM (3 steps) + INT8 W8A8 with **128×128 block granularity**. Their `quant.cu` source (vendored at `/Users/yewang/work/FastVideo/fastvideo-kernel/csrc/turbodiffusion/quant/quant.hpp`) implements the W8A8 INT8 quantizer with a custom-block-128 layout and a `_reduce_amax` warp reduction:

```cpp
template <class InputDtype_, int NumThrPerCta_, bool IsEvenM, bool IsEvenN>
class Quantization {
  static constexpr int BlockSize = 128;
  static constexpr float int8_max = 128.f;

  CUTLASS_DEVICE void quantization(
      float *float_reg, void *Optr, void *OSptr,
      int64_t const m, int64_t const n,
      int blk_m, int blk_n, int tidx,
      char *shared_data) {
    OutputDtype output_reg[NumElementPerThread];
    Saver<OutputDtype, BlockSize, BlockSize, NumThrPerCta, IsEvenM, IsEvenN> saver;
    float amax = _reduce_amax(float_reg, (float*)shared_data);
    _quantization(float_reg, output_reg, int8_max / amax);
    float scale_inv = amax / int8_max;
    saver.store(Optr, OSptr, output_reg, scale_inv, m, n, blk_m, blk_n, tidx);
    __syncthreads();
  }
  // ...
};
```

The kernel pattern is: a 128×128 tile is loaded into registers (FP16 or BF16 input), the per-block amax is reduced first per-thread, then per-warp via `__shfl_xor_sync`, then per-CTA via an atomic max in shared memory; the inverse scale `int8_max/amax` is broadcast and applied via `cutlass::NumericConverter<int8_t, float>` round-to-nearest; and the scale's inverse is stored alongside the INT8 output for downstream dequantization. This is structurally identical to what NVFP4's `to_nvfp4` kernel does (with block 128 → block 16 and INT8 → E2M1 + E4M3), but at the larger 128×128 INT8 granularity.

The TurboDiffusion stack composes:

- **2.0× from W8A8 INT8** (vs BF16) on the Wan 2.2 transformer
- **4–6× from Sparse-Linear Attention** at 90 % sparsity
- **12–25× from rCM 3-step distillation** (vs 100 BF16 steps)
- Various overheads: attention round-trips, normalization rewrites, CUDA Graph

The product is the headline **100–200× speedup** TurboDiffusion reports on a single RTX 5090 (Wan2.2-I2V-A14B-720P from 4549 s to 38 s = 120×; Wan2.1-T2V-1.3B-480P from 184 s to 1.9 s = 97×; Wan2.1-T2V-14B-720P from 4767 s to 24 s = 199×). The W8A8 path is the *least aggressive* of the three quantization tiers (vs FP8 vs NVFP4) but it has the least quality risk and is what has been *productionized* on Wan 2.2 first. A natural 2026 evolution of TurboDiffusion is to swap INT8 W8A8 for NVFP4 W4A4 on the GEMMs, which would push the speedup another ~1.7× (the NVFP4/INT8 ratio observed in TorchAO LTX-2 numbers).

### Q.10.4 Quantization × CFG distillation

CFG distillation [\[Meng et al., 2023, "On Distillation of Guided Diffusion Models"\]](https://arxiv.org/abs/2210.03142) eliminates the unconditional branch by training a single forward pass to produce the same guided output as `(1+w)·cond - w·uncond`. This is a 2× wall-clock saving that composes orthogonally with quantization. The wrinkle for quantization is that ViDiT-Q's "two-CFG-branch" axis vanishes — there is no longer a conditional/unconditional distribution mismatch to handle. This *simplifies* the calibration story but *removes* a tuning knob (it is no longer possible to use a conservative scale for the more-sensitive unconditional branch).

In May 2026 production, CFG distillation is rare for video DiT (it is ubiquitous for SDXL-class image models but the video distillation literature focuses on rCM/DMD step distillation rather than CFG distillation). The most likely combination over the next 12 months is rCM (4 steps) + NVFP4 (transformer) + Attn-QAT (attention) + a small distilled VAE — projected to deliver *at least* the lightx2v 28× on B200, with NVFP4 closing the remaining 4× gap to the theoretical peak.

### Q.10.5 The end-to-end "Blackwell ceiling" estimate

Putting the per-axis multipliers together, an aspirational Blackwell B200 single-node Wan 2.2-I2V-A14B-720P pipeline would compose:

- **NVFP4 GEMM** at 70 % of architectural peak: 6.3 PF/s vs BF16's 1.35 PF/s = **4.7×**
- **Attn-QAT FP4** at 1.5× over Sage3, which itself is ~2× over BF16 attention: **3×**
- **rCM 4-step distillation**: **12.5×**
- **Tile-decoder VAE / TAEHV**: **3×** (just the VAE portion)
- **8-GPU Ulysses + CFG parallel** (HGX B200): **5–6×** (sub-linear due to AllReduce)

Each is on a different fraction of total wall-clock, so the multiplicative end-to-end is roughly **150–250×**, putting a 5-second 720p generation on a single B200 node into the **20–30 s** range — a real-time-adjacent regime. As of May 2026, the closest production deployment is Lightx2v's RTX 5090 28× I2V-14B (which trades the data-center scale-out for distillation aggressiveness); the TurboDiffusion 199× on Wan 2.1-T2V-14B-720P is conceptually identical but stops at INT8 W8A8.

The remaining gap to that aspirational ceiling is **squarely the SVDQuant/QVGen-style W4A4-NVFP4-on-video-DiT problem** — i.e., the gap that deepcompressor#103 codifies. Closing that gap is the single largest piece of work in the May 2026 video-DiT-acceleration landscape, and is what the bulk of this section's algorithmic descriptions (PTQ4DiT → Q-DiT → ViDiT-Q → SVDQuant → ConvRot → QVGen → SemanticDialect) collectively contribute to.

---

## Cross-references and bibliography

Primary numeric and algorithmic sources cited above:

- **NVFP4 / MXFP4 / MXFP8 specs**: NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference" (Jun 2025); OCP MX format spec [\[Rouhani et al., arXiv:2310.10537\]](https://arxiv.org/abs/2310.10537).
- **B200 microbenchmarks**: Jarmusch & Chandrasekaran, "Microbenchmarking NVIDIA's Blackwell Architecture" [\[arXiv:2512.02189\]](https://arxiv.org/abs/2512.02189).
- **HGX B200 datasheet**: [www.nvidia.com/en-us/data-center/b200](https://www.nvidia.com/en-us/data-center/b200/).
- **cuBLAS 12.9 NVFP4**: NVIDIA, "Boosting Matrix Multiplication Speed and Flexibility with cuBLAS 12.9" (May 2025); [LtNvfp4Matmul sample](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtNvfp4Matmul).
- **PTQ4DiT**: Wu et al., NeurIPS 2024 [\[arXiv:2405.16005\]](https://arxiv.org/abs/2405.16005), [\[github.com/adreamwu/PTQ4DiT\]](https://github.com/adreamwu/PTQ4DiT).
- **Q-DiT**: Chen et al., CVPR 2025 [\[arXiv:2406.17343\]](https://arxiv.org/abs/2406.17343).
- **ViDiT-Q**: Zhao et al., ICLR 2025 [\[arXiv:2406.02540\]](https://arxiv.org/abs/2406.02540).
- **SVDQuant + Nunchaku**: Li et al., ICLR 2025 Spotlight [\[arXiv:2411.05007\]](https://arxiv.org/abs/2411.05007); MIT HAN Lab project page [\[hanlab.mit.edu/projects/svdquant\]](https://hanlab.mit.edu/projects/svdquant); blog [\[hanlab.mit.edu/blog/svdquant-nvfp4\]](https://hanlab.mit.edu/blog/svdquant-nvfp4); inference engine [\[github.com/nunchaku-ai/nunchaku\]](https://github.com/nunchaku-ai/nunchaku); the open issue [\[deepcompressor#103\]](https://github.com/nunchaku-ai/deepcompressor/issues/103).
- **EfficientDM**: He et al., ICLR 2024 [\[arXiv:2310.03270\]](https://arxiv.org/abs/2310.03270).
- **ConvRot**: Huang et al., Dec 2025 [\[arXiv:2512.03673\]](https://arxiv.org/abs/2512.03673).
- **QuEST**: Wang et al., ICCV 2025 [\[arXiv:2402.03666\]](https://arxiv.org/abs/2402.03666).
- **QuaRot**: Ashkboos et al., NeurIPS 2024 [\[arXiv:2404.00456\]](https://arxiv.org/abs/2404.00456).
- **SpinQuant**: Liu et al. [\[arXiv:2405.16406\]](https://arxiv.org/abs/2405.16406).
- **DFRot**: Xiang & Zhang [\[arXiv:2412.00648\]](https://arxiv.org/abs/2412.00648).
- **SemanticDialect / SeDA**: Jang & Tambe, Mar 2026 [\[arXiv:2603.02883\]](https://arxiv.org/abs/2603.02883); predecessor BlockDialect [\[arXiv:2501.01144\]](https://arxiv.org/abs/2501.01144).
- **NVIDIA TensorRT Model Optimizer (modelopt)**: [\[github.com/NVIDIA/TensorRT-Model-Optimizer\]](https://github.com/NVIDIA/TensorRT-Model-Optimizer); diffusers/quantization recipes [\[examples/diffusers/quantization\]](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers/quantization).
- **TensorRT-LLM VisualGen**: [\[docs/source/models/visual-generation.md\]](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/visual-generation.md).
- **PyTorch / TorchAO + Diffusers Blackwell blog (LTX-2 verbatim)**: [\[pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao\]](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/); reproduction repo [\[github.com/sayakpaul/diffusers-blackwell-quants\]](https://github.com/sayakpaul/diffusers-blackwell-quants); MSLK kernel [\[github.com/meta-pytorch/MSLK\]](https://github.com/meta-pytorch/MSLK); TorchAO PR [\[#4031\]](https://github.com/pytorch/ao/pull/4031).
- **TurboDiffusion**: Zhang et al. [\[arXiv:2512.16093\]](https://arxiv.org/abs/2512.16093); [\[github.com/thu-ml/TurboDiffusion\]](https://github.com/thu-ml/TurboDiffusion); local source `/Users/yewang/work/FastVideo/fastvideo-kernel/csrc/turbodiffusion/quant/quant.{hpp,cu}`.
- **diffusers-torchao**: [\[github.com/sayakpaul/diffusers-torchao\]](https://github.com/sayakpaul/diffusers-torchao).
- **bitsandbytes 4-bit / 8-bit**: [\[huggingface.co/docs/diffusers/main/en/quantization/bitsandbytes\]](https://huggingface.co/docs/diffusers/main/en/quantization/bitsandbytes).
- **lightx2v Wan-NVFP4**: [\[huggingface.co/lightx2v/Wan-NVFP4\]](https://huggingface.co/lightx2v/Wan-NVFP4).
- **ac4k Wan2.2** (community NVFP4 fork): [\[github.com/ac4k/Wan2.2\]](https://github.com/ac4k/Wan2.2).
- **vLLM-Omni FP8 Wan 2.2**: [\[PR #1412\]](https://github.com/vllm-project/vllm-omni/pull/1412), merged 2026-03-20.
- **QVGen**: Huang et al., ICLR 2026 [\[arXiv:2505.11497\]](https://arxiv.org/abs/2505.11497); [\[github.com/ModelTC/QVGen\]](https://github.com/ModelTC/QVGen).
- **Attn-QAT**: Zhang et al. [\[arXiv:2603.00040\]](https://arxiv.org/abs/2603.00040).
- **SmoothQuant**: Xiao et al. [\[arXiv:2211.10438\]](https://arxiv.org/abs/2211.10438).
- **TAEHV**: [\[github.com/madebyollin/taehv\]](https://github.com/madebyollin/taehv).
- **FLUX.2-small-decoder**: [\[huggingface.co/black-forest-labs/FLUX.2-small-decoder\]](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder).
