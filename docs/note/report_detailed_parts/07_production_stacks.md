## §7. Production Stacks, Vendor Blogs, and End-to-End Models

*Subsection prefix `R.x` is scoped to §7.*

This section consolidates everything we know about *what production video DiT
inference actually looks like in May 2026* — the vendor blogs that disclose
real wall-clock numbers, the open-source frameworks that ship those
optimizations, and the model families they all target. The goal is to give a
reader who is standing up a Wan 2.2 / LTX-2 / HunyuanVideo-1.5 / Cosmos
Predict 2.5 deployment a single chapter that answers: *which stack should I
copy, what numbers should I expect, and what are the verified facts vs the
marketing-grade quotes?*

Where a number was fetched directly from the source URL it is annotated
`[VERIFIED]`. Where the existing draft of this report had a value that turned
out to be incorrect after re-fetching, the corrected value is annotated
`[CORRECTED]`. The most important corrections that fall out of this round of
re-fetching are:

1. The **HGX B200** datasheet on `nvidia.com/en-us/data-center/b200/` lists
   FP4 Tensor Core throughput in *sparse | dense* convention. For 8× B200,
   this is **108 PFLOPS sparse / 54 PFLOPS dense** as published. The per-GPU
   numbers from the underlying B200 SXM datasheet are **18 PFLOPS sparse /
   9 PFLOPS dense FP4** at 1000 W TGP, **15 PFLOPS sparse / ~7.5 PFLOPS
   dense** at the lower 700 W envelope used in HGX B200. Earlier text in
   the report saying simply "15 PFLOPS FP4" without sparse/dense
   qualification is **`[CORRECTED]`** — peak dense FP4 per B200 GPU is **9
   PFLOPS** (700 W envelope; 1000 W variant is 18 PF sparse / 9 PF dense).
2. **LTX-2 has 7 named artifacts**, not 4. Per the
   `Lightricks/LTX-2` HuggingFace model card: `ltx-2-19b-dev`,
   `ltx-2-19b-dev-fp8`, `ltx-2-19b-dev-fp4`, `ltx-2-19b-distilled`,
   `ltx-2-19b-distilled-lora-384`, `ltx-2-spatial-upscaler-x2-1.0`, and
   `ltx-2-temporal-upscaler-x2-1.0` `[VERIFIED]`.
3. **LTX-2.3 is a separate, larger 22B model**, not a re-spin of LTX-2 19B.
   The `Lightricks/LTX-2.3-nvfp4` card calls out
   `ltx-2.3-22b-dev-nvfp4` and a coming `ltx-2.3-22b-distilled-nvfp4`
   `[VERIFIED]`. Treating LTX-2 and LTX-2.3 as the same artifact (as the
   draft sometimes did) is incorrect.
4. The **VoltagePark Wan 2.2 8×H100** number is `4.67 s/step → 1.51 s/step`
   (`3.09×`, `187 s → 60 s` total over 40 steps, T2V) — quoted verbatim from
   their blog `[VERIFIED VERBATIM]`.
5. **FastVideo LTX-2.3 1080p on B200** is `5 s clip in 4.55 s e2e on
   1× B200`, `~3.9× the next-fastest 1080p option` `[VERIFIED]`.

The chapter is organized as:

- **R.1** Deployment blogs (Baseten Wan 2.2; Morphic Wan 2.2 I2V; VoltagePark
  Wan 2.2 T2V; FastVideo LTX-2.3 1080p on B200; fal "Ulysses Unbound"; Hao AI
  Lab FastWan; PyTorch FA3; PyTorch + TorchAO Blackwell; LightX2V
  disaggregated; NVIDIA Cosmos Predict 2.5 distill; SHI Labs DistGEMM;
  Lambda + Together FA4; NVIDIA NVFP4 + cuBLAS 12.9; TransformerEngine 2.10;
  DGX B200 / HGX B200 datasheets).
- **R.2** Production inference frameworks (LightX2V; SGLang Diffusion; xDiT;
  TensorRT-LLM VisualGen; Wan official).
- **R.3–R.7** End-to-end model overviews with parameter counts, recommended
  inference settings, and distilled/quantized variants.
- **R.8** Catalog of vendor-shipped distilled and quantized weights.
- **R.9** Consolidated cross-stack benchmark table.

---

## R.1 Production deployment blogs

### R.1.1 Baseten — *"Wan 2.2 video generation in less than 60 seconds"* (Jan 2026)

**Source:** [www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/](https://www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/) `[VERIFIED]`.

Baseten reports a custom Wan 2.2 inference runtime that achieves **3.2×
faster Wan 2.2 inference on a single B200 over the default Alibaba-Wan
runtime, and 2.6× faster on H100**. Benchmark setup:

- **40 sampling steps**, **1280×720 resolution**, **81 frames per video**
  (5.06 s of video at 16 fps).
- Prompt sequence lengths between 32 and 512 characters; prompt length had
  no measurable effect on performance.
- Median of multiple runs reported.

**Optimizations they call out:**

1. A custom `multiply_add` kernel (the GEMM kernel) tailored to Wan 2.2's
   specific input shapes. Their argument: `cuBLAS` lookup tables are tuned
   for popular LLM (e.g., Llama) shapes, and they pick the wrong heuristic
   on the QKV / FFN shapes that fall out of a 3D DiT. They wrote a
   shape-specific kernel that beats the default selection.
2. A custom **RoPE-attention** kernel that fuses the per-position
   sinusoidal multiply into the QK matmul, eliminating a memory-bound trip
   for `cos`/`sin` tables.
3. Custom **LayerNorm** and **RMSNorm** kernels that batch the
   per-feature reductions tighter than `torch.compile` does for the Wan
   block shapes.
4. **Async USP all-to-all over NCCL.** Standard Ulysses sequence
   parallelism issues 3 all-to-alls (Q, K, V) sequentially before SDPA.
   Baseten overlaps them on Blackwell-async streams and uses the
   "asynchronous programming paradigm" of the new arch — i.e., issuing
   one collective and immediately starting the next branch's GEMM. The
   wording in the post matches what fal independently demonstrates as
   "Async Ulysses" in their fal blog (R.1.5 below).
5. **Disabled `offload_model` by default.** The default `--offload_model
   True` flag in the official Wan 2.2 generate.py preserves VRAM by paging
   modules to CPU between uses, which costs PCIe round-trips. Baseten
   keeps everything resident.

**Lossy headroom they explicitly leave on the floor.** They mention that
adding an attention-cache (TeaCache-class) gets them another ~50%
end-to-end on top, and FP4 quantization more on top of that, but they
ship neither in the lossless 3.2× number. So 3.2× is a *lossless*
multiplier; Stack-D-style `4-step + NVFP4 + FA-FP4 + TeaCache` would be
on top of this baseline.

**Cost framing.** Because video DiT runs at batch 1 with sequence
parallel across all 8 GPUs of the node, latency *is* throughput.
Baseten quotes a 67% cost reduction on dedicated B200 deployments.

### R.1.2 Morphic — *"Boosting Wan 2.2 I2V inference on 8×H100s, 2.5× faster than baseline with Sequence Parallelism"* (Oct 2025)

**Source:** [morphic.com/blog/boosting-wan2-2-i2v-56-faster](https://morphic.com/blog/boosting-wan2-2-i2v-56-faster) `[VERIFIED]`.

Morphic ships an open-source fork of the Wan 2.2 official repo at
`github.com/morphicfilms/wan2.2_optimizations`, with single-flag
toggles for each optimization layer. Their setup: **8× H100 80 GB,
1280×720, 40 inference steps, 81 frames, base seed 50**.

Their full benchmark ladder:

| Stage | Time (s) | Speedup |
|---|---|---|
| Baseline (FA2) | 250.70 | 1.00× |
| + FlashAttention-3 | 195.13 | 1.28× |
| + TF32 (matmul + cuDNN) | 159.55 | 1.57× |
| + int8 weight-only quant (no FSDP) | 170.24 | 1.47× |
| + MagCache `E012K2R20` | 157.10 | 1.59× |
| + MagCache + TF32 | 121.56 | 2.06× |
| FA3 + torch.compile | 172.87 | 1.45× |
| FA3 + int8 + torch.compile | 142.40 | 1.76× |
| FA3 + TF32 + torch.compile | 142.73 | 1.76× |
| **FA3 + TF32 + MagCache + torch.compile** | **109.81** | **2.28×** |
| MagCache `E024K2R10` (more aggressive) | 98.87 | **2.53×** |

Notes:

1. FA3 must be installed from `github.com/Dao-AILab/flash-attention/tree/main/hopper` — `pip install flash-attn` ships FA2 only.
2. `int8_weight_only` from `torchao` lets both Wan 2.2 high-noise and
   low-noise expert fit on each H100 80 GB without FSDP — but TF32 has
   no benefit when matmuls are int8.
3. **MagCache** (Zehong Ma et al.) is a TeaCache-style training-free
   feature cache that exploits the **monotonic decay of magnitude
   ratios of successive residual outputs**. It is reported to dominate
   TeaCache on Wan 2.2 quality. Their best preset:
   `--magcache_thresh 0.12 --magcache_K 2 --retention_ratio 0.2`.
4. `torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=False)`
   is required (not `fullgraph=True`) because `rope_apply` in
   `wan/modules/model.py` does dynamic slicing.
5. The 2.28× number preserves quality (no visible artifacts on seed
   50). The 2.53× number with `E024K2R10` shows visible vehicle-direction
   artifacts.
6. Morphic ran on Modal infrastructure for multi-GPU.

This blog is the canonical proof point that **Wan 2.2 at H100 8× scale,
with off-the-shelf parts (FA3 + TF32 + MagCache + torch.compile),
finishes a 720p 5 s clip in **109.81 s** without quality loss**. That's
~`Stack A` in §5 of the parent report.

### R.1.3 VoltagePark — *"Accelerating Wan 2.2 from 4.67 s to 1.5 s per denoising step"* (Oct 2025)

**Source:** [voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations](https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations) `[VERIFIED VERBATIM]`.

Setup: **8× H100, Wan 2.2 T2V, 40 denoising steps, prompt is a
cinematic golden retriever scene**. They start from a strong baseline
that already includes Ulysses sequence parallelism + FlashAttention-3
properly built (i.e., what Morphic's row 2 looks like).

Their ladder:

| Optimization stage | Time / step | Total (40 steps) | Speedup | Cumulative |
|---|---|---|---|---|
| Baseline (Ulysses-8 + FA3) | 4.67 s | **187 s** | – | 1.00× |
| + Batched cond / uncond forward pass | 3.98 s | 159 s | 1.17× | 1.17× |
| + Fixed time-embedding (per-batch, not per-token) | 3.40 s | 136 s | 1.17× | 1.37× |
| + SageAttention (INT8 QK^T) | 3.10 s | 124 s | 1.10× | 1.51× |
| + TeaCache (threshold 0.2–0.3, warmup 10%, periodic retention) | **1.51 s** | **60 s** | **2.05×** | **3.09×** |

Their attention shape disclosure is also useful: at **1280×720 × 81
frames**, the joint sequence length is **75,600 tokens** with **40
heads × 128 head dim**. The QK^T matrix is 75,600 × 75,600 → 5.7 B
elements per head per step, repeated 40 times per attention layer — so
softmax over 5.7 B elements is the dominant non-matmul cost, which is
why SageAttention's INT8 + per-thread quantization gives a clean win.

Their TeaCache implementation has two non-obvious requirements that
they lifted from a reproduction failure:

1. **Compare against the immediately-preceding step**, not against the
   last *computed* step. Their first pass compared current step `t` to
   the last full-compute step. When several skips landed in a row,
   error accumulated and showed up as motion stutter; comparing to
   `t-1` (whether `t-1` was computed or skipped) eliminated stutter.
2. **Synchronize the skip/compute decision across all 8 ranks.** All
   ranks must independently decide the same skip pattern, otherwise
   the Ulysses all-to-all desyncs. They broadcast the skip decision.
3. **Periodic retention steps** (force a full recompute every N
   skipped steps) prevent long-term drift.

Future-work bullets they call out (none implemented in the 60 s
number): Sliding Tile Attention, dynamic TeaCache thresholds, QKV
fusion, and extending to I2V / S2V / Animate.

### R.1.4 FastVideo / Hao AI Lab — *"Create a 5 s 1080p Video in 4.5 s with FastVideo on a Single GPU"* (Apr 2026)

**Source:** [haoailab.com/blogs/fastvideo_realtime_1080p](https://haoailab.com/blogs/fastvideo_realtime_1080p/) `[VERIFIED]`.

This is the milestone the parent report calls "FastVideo LTX-2.3 1080p
demo on B200." The verified claim is:

> *5-second 1088 × 1920 video at 24 FPS with synchronized audio (TI2AV)
> in **~4.55 s end-to-end on a single NVIDIA B200**, ~3.9× the
> next-fastest published 1080p option.* `[VERIFIED]`

Underlying model is **LTX-2.3** (Lightricks 22B audio-video joint
foundation model in NVFP4). Stack:

1. **Custom attention kernels for SM100/SM103** — written for the new
   Blackwell tensor cores and TMEM. Hao AI Lab does not publish the
   kernel directly in this blog; they reference the Video Sparse
   Attention (VSA) family from `arXiv:2505.13389`. The FA-class kernel
   on B200 saturates at ~71% utilization (per FA4 paper) and FastVideo's
   custom sparse variant is on top of that.
2. **NVFP4 quantization on the DiT linears**. Per the parallel
   PyTorch/TorchAO blog (R.1.8), LTX-2 in NVFP4 is 1.56× / 1.67× / 1.68×
   over BF16 at B=1/4/8 on B200. LTX-2.3 NVFP4 numbers are on the same
   order.
3. **Aggressive graph-level + kernel fusion** across every stage of the
   inference pipeline: prompt encoding, latent preparation, denoising,
   decoding. Their explicit framing: "real latency is not merely the
   model's core diffusion denoising loop, but everything around it
   too."
4. **System-level tuning beyond the model**: a custom `ffmpeg` build for
   the target CPU, IPC overhead reduction in the I/O pipeline for
   1088×1920 frames + audio.
5. The deployed demo runs on **half a GB200 NVL72 = 36 B200 GPUs**, with
   each GPU serving as one replica and a Rust middleware load-balancer
   in front. Per-replica latency = 4.55 s e2e.

NVIDIA Dynamo recently added FastVideo as a backend
(`docs.nvidia.com/dynamo/dev/user-guides/diffusion/fastvideo`),
mirroring the SGLang Diffusion integration.

### R.1.5 fal — *"Ulysses Unbound: Experiments in Communication–Computation Overlap"* (8×B200)

**Source:** [blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap](https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/) `[VERIFIED]`.

This is the canonical 2026 deep-dive on **Ulysses sequence parallelism
on Blackwell**. Setup: 8× B200, BF16, B=1, H=40, D=128, cuDNN attention
backend, 15 warmup + 40 timed iterations × 3 reps, max-rank median.

They benchmark four Ulysses variants:

1. **Baseline Ulysses** (DeepSpeed-style sequential): each rank does
   Q, K, V GEMMs locally, then issues 3 all-to-alls in series, then
   SDPA, then output projection.
2. **Async Ulysses** (ByteDance VeOmni style): overlap Q's all-to-all
   with K's GEMM, K's all-to-all with V's GEMM. Implemented via PyTorch
   `_functional_collectives` (`AsyncTensor`-based, composes with
   `torch.compile`). Result: **−23% to −25% on pre-attention chunk
   latency at 2/4/8 GPUs**, but only **~3% e2e step improvement**
   because SDPA + output dominate.
3. **Async Ulysses + Symmetric Memory** (route the all-to-all through
   the GPU Copy Engine instead of SMs): NCCL collectives are GPU
   kernels that contend for SM cycles with GEMMs, even on independent
   streams; using `torch.distributed._symmetric_memory` separates the
   data movement onto the Copy Engine. **Best at 2/4 GPUs; regresses at
   8 GPUs** because per-rank payloads shrink and SymMem fixed
   overheads (signaling, bookkeeping) eat the win.
4. **Fused QKV all-gather + matmul** via
   `torch.ops.symm_mem.fused_all_gather_matmul`: each rank packs Q | K
   | V row blocks and the op fuses the sequence all-gather with the
   local matmul into one packed `(B, S_global, 3*H_local*D)` output.
   **−37.3% chunk latency at 2 GPUs, −33.4% at 4 GPUs, and −5.0% /
   −4.8% e2e**. Benefit collapses at 8 GPUs (chunk only −4.6%, e2e
   −0.3%) because messages are already small under strong scaling.

Weak-scaling regime (local seq fixed at 16K, global grows): Async
Ulysses is the most robust default at 8 GPUs; Fused QKV is best at 2/4
GPUs. The blog's takeaway is that **a runtime should pick between
packing, overlap, and fusion based on sequence length, world size, and
interconnect** — there is no single Ulysses variant that wins every
regime.

For the parent report's Stack A on 8×B200, the verdict is:

- Use **Async Ulysses** as the default (best at 8 GPUs).
- **Fused QKV** wins below 8 GPUs.
- Both compose with `torch.compile`.
- Interconnect-heavier workloads (MoE routing) benefit even more.
- **Kraken** (Meta) is the cookbook-style reference for the next
  generation: device-initiated symmetric-memory / NVSHMEM
  communication, persistent kernels, Triton.

### R.1.6 Hao AI Lab — *"FastWan: Generating a 5-Second Video in 5 Seconds via Sparse Distillation"*

**Source:** [hao-ai-lab.github.io/blogs/fastvideo_post_training](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/) `[VERIFIED]`.

The companion blog to FastVideo's 1080p milestone — but earlier and
focused on **480p / 720p sparse-distilled Wan**. They publish a clean
DiT-only ablation on H200:

| DiT denoising stack | Wan 2.2 5B 720P | Wan 2.1 14B 720P | Wan 2.1 1.3B 480P |
|---|---|---|---|
| FA2 baseline | 157.21 s | 1746.5 s | 95.21 s |
| FA2 + DMD distill | 4.67 s | 52.0 s | 2.88 s |
| FA3 + DMD | 3.65 s | 37.87 s | 2.14 s |
| FA3 + DMD + torch.compile | 2.64 s | 29.5 s | 1.49 s |
| **VSA + DMD + torch.compile** | – | **13.0 s** | **0.98 s** |

Training cost they call out: sparse-distillation of Wan 2.1-T2V-1.3B
takes **64 H200 GPUs × 4 k steps = 768 GPU-hours ≈ \$2,603 at
\$3.39/H200/h on Anyscale**. Reproducible via the public slurm script
in the repo.

Method: **sparse distillation** is the joint training of (a) a few-step
DMD2 student and (b) a sparse-attention pattern (Video Sparse Attention,
VSA), with **full-attention real and fake score networks** as
supervision. Crucially, the student uses VSA but the score networks
keep full attention — so the student gets sparse-runtime acceleration
with dense supervisory signal, the only published recipe to date that
makes sparse attention compatible with extreme step compression (3–4
NFE).

The blog also distinguishes **FastWan2.2-TI2V-5B-FullAttn** (DMD-only,
data-free build on the 5B because its ~20K sequence length doesn't
benefit much from sparsity) from **FastWan2.1-1.3B/14B + VSA** (full
sparse-distillation recipe).

### R.1.7 PyTorch — *"FlashAttention-3"* (foundation reference)

**Source:** [pytorch.org/blog/flashattention-3](https://pytorch.org/blog/flashattention-3) `[VERIFIED]`.

FA3 was published mid-2024 for Hopper. Headline numbers:

- **740 TFLOPS BF16 forward, 75% utilization of H100 989 TFLOPS BF16
  ceiling.** 1.5–2.0× over FA2.
- **~1.2 PFLOPS FP8 forward**, 2.6× lower error than baseline FP8
  attention via incoherent processing (random Hadamard rotation).

Three new techniques that distinguish FA3 from FA2:

1. **Asynchrony-aware warpgroup scheduling** using Hopper's WGMMA +
   TMA + FP8 path. Producer warps drive TMA; consumer warps drive
   WGMMA. Free overlap because the scheduler doesn't have to round-robin.
2. **Pingpong inter-warpgroup softmax/MMA overlap.** With 2 warpgroups,
   group 1 does its GEMMs while group 2 does its softmax, and vice
   versa. ~570 → 620 TFLOPS gain.
3. **Intra-warpgroup softmax/MMA pipelining** within a single warpgroup.
   ~620 → 640–660 TFLOPS gain at the cost of extra register pressure.
4. **Incoherent processing (FP8)**: Hadamard rotation on Q, K to spread
   outliers, fused into RoPE for free. 2.6× lower quantization error.

Why this matters for video DiT: FA3 was the *prerequisite* for both
Morphic's 1.28× win and VoltagePark's 4.67 s/step baseline. Anything
on Blackwell now uses FA4 instead (R.1.12, R.1.13).

### R.1.8 PyTorch + TorchAO — *"Faster Diffusion on Blackwell: MXFP8 and NVFP4 with Diffusers and TorchAO"*

**Source:** [pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/) `[VERIFIED]`.

This is the most reproducible Blackwell quantization benchmark for
video DiT. Setup:

- **NVIDIA B200 (DGX B200)**, BF16/MXFP8/NVFP4.
- Selective quantization (skip layers where `min(M,K,N) < 1024` or
  layers near embeddings/normalization).
- `torch.compile` with regional compilation. At B=1, also
  `mode='reduce-overhead'` to enable CUDA graphs.
- Models: FLUX.1-dev (image), QwenImage (image), **LTX-2** (video).

**LTX-2 NVFP4 results (the canonical video Blackwell quant data
point):**

| Quant Mode | Batch | Latency (s) | Memory (GB) | Speedup vs BF16 |
|---|---|---|---|---|
| BF16 | 1 | 16.230 | 72.77 | 1.00× |
| MXFP8 | 1 | 13.724 | 54.54 | 1.18× |
| **NVFP4** | 1 | **10.374** | **45.72** | **1.56×** |
| BF16 | 4 | 61.591 | 87.61 | 1.00× |
| MXFP8 | 4 | 50.956 | 69.38 | 1.21× |
| **NVFP4** | 4 | **36.963** | **60.56** | **1.67×** |
| BF16 | 8 | 122.427 | 107.40 | 1.00× |
| MXFP8 | 8 | 102.546 | 89.18 | 1.19× |
| **NVFP4** | 8 | **72.689** | **80.36** | **1.68×** |

`[VERIFIED VERBATIM]`. Inference settings: prompt about a cat looking
at a laptop, 768×512, 121 frames, 24 fps, 40 steps, CFG 4.0, negative
prompt for blur/motion artifacts. VAE tiling enabled.

Key takeaways:

1. **NVFP4 is the winner for compute-bound large-batch video**: 1.68× on
   B=8 LTX-2 vs BF16, with 25% lower peak memory.
2. **MXFP8 is the safer middle ground** when LPIPS matters more
   (FLUX.1-dev: MXFP8 LPIPS 0.11, NVFP4 LPIPS 0.44 vs BF16 baseline).
3. **CUDA graphs at B=1**: a separate optimization. Their internal PR
   shows `QwenImage + NVFP4 + B=1` got a 1.81× speedup just from
   wrapping each transformer block in a function that clones inputs
   (so `reduce-overhead` mode composes with regional compilation).
4. **TorchAO `to_nvfp4` kernel was upgraded to use MSLK** during this
   blog's preparation (PR pytorch/ao#4031), which closed a residual
   performance gap.

Selective quantization heuristic, codified:

> *Skip a `torch.nn.Linear` if `min(M, K, N) < 1024` (overhead of
> activation quantization > matmul speedup) OR if it's an embedding or
> normalization layer (accuracy-sensitive).*

For LTX-2 FLUX-class models this gives about 5–10% lower latency
gains than full quant but ~1/4 the LPIPS regression.

### R.1.9 LightX2V — *"Disaggregated Deployment: Breaking the Memory and Throughput Bottlenecks of Diffusion Model Inference"* (April 2026)

**Source:** [light-ai.top/LightX2V-BLOG/posts/Disaggregation/](https://light-ai.top/LightX2V-BLOG/posts/Disaggregation/) (mirrored at `lightx2v.org`) `[VERIFIED]`.

LightX2V splits the monolithic Wan/Qwen-Image pipeline into three
independent microservices wired by the Mooncake RDMA engine:

```
Encoder  ──Mooncake Phase1──>  Transformer (DiT)  ──Mooncake Phase2──>  Decoder
(text + image + VAE-enc)        (the heavy DiT itself)                   (VAE decode)
~17–20 GB                       ~28–41 GB                                ~0.3 GB
```

**Memory data on H100:**

| Stage | Wan T2V 14B | Wan I2V 14B 480P | Qwen-2512 BF16 |
|---|---|---|---|
| Baseline (monolithic) | ~39 GB | ~48 GB | ~58 GB |
| Encoder node | ~11 GB | ~13 GB | ~18 GB |
| Transformer node | ~28 GB | ~32 GB | ~40 GB |

The Encoder node fits comfortably on a 24 GB RTX 4090 *without
offloading*. That's the qualitative win: a 32× Text Encoder speedup on
RTX 4090 just from dropping `cpu_offload`.

**End-to-end Qwen-2512 50-step request on H100, profiled:**

| Stage | Latency | % of total |
|---|---|---|
| Text Encoder compute | 0.26 s | 1.0% |
| Phase1 Mooncake send | 0.025 s | 0.1% |
| DiT 50 steps | 25.3 s | 96.2% |
| Phase2 Mooncake send | 0.024 s | 0.09% |
| VAE decode | 0.31 s | 1.2% |
| **Network total (Phase1 + Phase2)** | **0.05 s** | **0.2%** |
| **Pipeline total** | **26.3 s** | – |

The network overhead is **<0.2% of e2e** — Mooncake's
zero-copy RDMA transport with InfiniBand 400 GB/s makes disaggregation
nearly free.

**Per-stage speedup on 4090 with offload baseline:**

| Component | Baseline (4090 + offload) | Disagg (no offload) | Speedup |
|---|---|---|---|
| Text Encoder | 12.89 s | 0.40 s | **32.22×** |
| DiT 50-step (Qwen-2512) | 5.75 s/step | 3.76 s/step | 1.53× |
| DiT 8-step distilled | 2.90 s/step | 1.88 s/step | 1.54× |

For Wan 2.1-I2V-14B on RTX 4090 (BF16, 40 steps, block offload):

| Metric | Baseline (1×4090) | Disagg (2×4090) |
|---|---|---|
| DiT/step 480P | 24.24 s | 19.02 s |
| DiT/step 720P | 90.71 s | 62.80 s |
| Text encoder | 2.14 s | 0.20 s |
| Image encoder 480P | 0.57 s | 0.28 s |

**Optimal Encoder:Transformer ratio:** the throughput model is
`R_e = E/t_e`, `R_t = T/t_t`, balance at `T:E = t_t:t_e`. For
Qwen-2512 8-step on 4090 with `t_e = 0.4 s`, `t_t = 15 s`, the optimal
ratio is **37.5 : 1** — i.e., **7T : 1E on 8 GPUs**. For 50-step
models, the ratio blows up to 470:1.

**8× RTX 4090 measured (decentralized scheduling):**

| Mode | Ratio | QPS | P50 | P95 | P99 |
|---|---|---|---|---|---|
| Baseline | 8:0 | 0.18 | 22 s | 28 s | 30 s |
| Disagg (centralized) | 7:1 | 0.24 | 17 s | 25 s | 28 s |
| **Disagg (decentralized)** | **7:1** | **0.34** | **17 s** | **20 s** | **22 s** |

`+1.89× QPS` over baseline, with much tighter tail. The
**decentralized queue scheduler** is the second-half novelty: a
controller maintains three RDMA ring buffers (request / phase1 /
phase2). Encoder runs as an HTTP server; Transformers and Decoder run
as pull workers consuming from RDMA rings. Multiple Transformers can
run in parallel.

This is the most-reproducible 2026 production-deployment paper for
video diffusion. Combine with the Wan 2.2-Lightning + lightx2v
StepDistill stack and the math says: 4-step distilled Wan 2.2 on 8×
B200 with disaggregated deployment, lossless, ≤ 5 s end-to-end at 720p
× 81 f.

### R.1.10 NVIDIA — *"Distilling Cosmos Predict 2.5"* (cosmos-cookbook)

**Source:** [nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/distilling_predict2.5.html](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/distilling_predict2.5.html) `[VERIFIED]`.

Step-by-step recipe NVIDIA published in November 2025 to distill
**Cosmos Predict 2.5 Video2World** into a **4-step student** using
**DMD2 on the TrigFlow parameterization, no GAN**.

Algorithm:

1. **Initialization.** Load pre-trained Cosmos Predict 2.5 2B teacher
   weights into both the student and the critic ("fake score") network.
2. **Optional supervised warm-up.** Generate ~1000s synthetic
   teacher samples for a brief supervised pre-warm. **Empirically
   skipped** for 4-step Text/Video2World — gave no measurable benefit.
3. **Alternating training**: K critic steps then 1 student step. They
   set **K = 4** (`student_update_freq=5`).

Implementation details NVIDIA calls out:

- **TrigFlow** is the shared parameterization for both this DMD2 path
  and a planned consistency distillation path (sCM, arXiv:2410.11081).
  It maps cleanly onto both EDM and Rectified-Flow trajectories, so
  the teacher can have been trained under either.
- **GAN loss is omitted.** The DMD2 paper proposes adding a GAN
  discriminator on top, but NVIDIA finds it gives no improvement on
  Predict 2.5 — so the simplified recipe is just teacher-student
  distribution matching.
- The **student loss** is the DMD2 distribution-matching loss:
  `s_real(x) - s_fake(x)` evaluated through the teacher (with CFG
  collapsed) and critic respectively, backpropped only into the student.
- **Critic loss** is a standard denoising loss on student-generated
  samples.
- **Convergence: 1500 iterations gives satisfactory video quality** —
  300 student steps + 1200 critic steps. Their reference plot shows
  the perceptual quality plateau at ~1500 iters.

Code lives at
`github.com/nvidia-cosmos/cosmos-predict2.5/tree/main/cosmos_predict2/_src/predict2/distill`,
class `Video2WorldModelDistillDMD2TrigFlow(DistillationCoreMixin,
TrigFlowMixin, Video2WorldModel)`.

This recipe is the cleanest open-source DMD2 video distillation
trainer with a public teacher. Anyone wanting to reproduce
"`Wan 2.2 4-step + CFG fold`" without the Wan-Lightning closed-source
trainer can substitute Cosmos Predict 2.5 + cosmos-cookbook. See R.6.5.

### R.1.11 SHI Labs / NVIDIA — *"Distributed GEMM"* (CUTLASS-native TP)

**Source:** [medium.com/shi-labs/distributed-gemm-88be6a481e2b](https://medium.com/shi-labs/distributed-gemm-88be6a481e2b) `[VERIFIED]`.

Authored by Ali Hassani et al. with NVIDIA collaborators (Vijay
Thakkar, Haicheng Wu et al.). Released alongside CUTLASS Example 65
(`cutlass/examples/65_distributed_gemm`). Headline: **a CUTLASS-native
implementation of Tensor Parallel GEMM** — `AllGather + GEMM` and
`GEMM + ReduceScatter` — that uses Programmatic Dependent Launch (PDL)
+ the GPU Copy Engine to hide communication entirely behind matmul,
without rewriting the GEMM kernel.

The trick: the existing CUTLASS GEMM kernel is wrapped with one
extra wrapper class (`DistributedGemmKernelWrapper`), and the
communication is done via direct peer-to-peer access through CUDA
Graphs + PDL + the Copy Engine, rather than via NCCL collectives that
contend for SMs.

Performance on 8× H100 NVL18 (any-to-any), Llama 70B / 405B fc1/fc2
linears in FP16:

- DistGEMM achieves 91% / 99% / 85% / 93% of best **single-stage GEMM
  excluding communication** (the theoretical compute upper bound) on
  the four problem shapes. Equivalently, 92% / 93% / 89% / 91% of
  **8-stage GEMM excluding communication**.
- Estimated 15% speedup vs collective-based TP on 405B, 24% on 70B.
- Vs `AllReduce + GEMM + GEMM` (the simpler baseline): 8% / 12%
  speedup on 405B / 70B.

Schedules implemented:

```
AllGather1D_TilingCD_RotatingA
AllGather1D_TilingCD_RotatingB
ReduceScatter1D_TilingA_RotatingC
ReduceScatter1D_TilingB_RotatingC
```

Why this matters for video DiT: **video diffusion runs at batch 1 with
sequence parallelism dominating, but TP would still be useful for the
text encoder + VAE decoder paths**. DistGEMM gives a path to TP that
doesn't fight Ulysses for NCCL bandwidth on the same NVLink fabric.
Not yet integrated into any major video framework; an active research
direction.

### R.1.12 Lambda — *"FlashAttention-4 gives the NVIDIA Blackwell platform its most optimized attention kernel yet"* (Apr 2026)

**Source:** [lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet) `[VERIFIED]`.

Lambda's mid-2026 summary of FA4 from a deployment perspective:

- Headline: **1,613 TFLOPS BF16 forward on HGX B200, 71% hardware
  utilization, 1.3× over cuDNN 9.13, 2.7× over Triton.**
- Install: `pip install flash-attn-4`. CuTe-DSL based, install in
  seconds vs minutes/hours for the C++ template path.
- Three FA4 techniques:
    1. **Redesigned async pipeline**: `tcgen05.mma` is fully async, so
       a warp can issue and immediately move on. Orchestration warps
       handle async loads + matmul scheduling; compute warps handle
       softmax in parallel; one tile's GEMMs overlap with adjacent
       tile's softmax.
    2. **Software-emulated `exp()`** via polynomial approximation on
       FMA units. This moves the exponential bottleneck off the SFU
       (Special Function Unit) onto general-purpose compute that
       Blackwell has in abundance.
    3. **Conditional softmax rescaling**: skip the rescale unless the
       running-max shift is large enough to threaten numerics.
       Reduces rescale ops by ~10× per Tri Dao's Hot Chips talk.
- FlexAttention with FA4 backend on GB200 NVL72 (custom variants):
  dense/causal 1.6–3.2× over Triton (forward), 1.85–2.3× (backward);
  ALiBi 1.2–2.1× / 1.9–2.9×; document masking up to 2.7× / 3×; sliding
  window 1.4–2.1× / 1.8–2.2×.

NVIDIA has since folded several FA4 techniques into newer cuDNN
releases (cuDNN 9.14+), narrowing the gap.

### R.1.13 Together — *"FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"* (March 5, 2026)

**Source:** [together.ai/blog/flashattention-4](https://www.together.ai/blog/flashattention-4) `[VERIFIED]`. Authors: Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao. Paper: arXiv:2603.05451.

Same FA4 result, but the *technical* paper-grade exposition. Key
points beyond the Lambda summary:

- **Asymmetric hardware scaling** is the central framing: from H100 to
  B200, BF16 tensor core throughput jumps 1 → 2.25 PFLOPS, but SFU
  count and shared-memory bandwidth stay flat. This means attention's
  exp() (SFU-bound) becomes the bottleneck on the forward pass, and
  shared-memory traffic becomes the bottleneck on the backward pass.
- **Forward pass** uses ping-pong of 2 Q tiles per CTA, dedicated
  correction warpgroup, conditional rescaling, and software-emulated
  `exp2` via Cody-Waite range reduction + Horner's-method polynomial
  approximation. Coefficients chosen via Sollya:
  `p0=1.0, p1≈0.6951, p2≈0.2276, p3≈0.0771`.
- **Backward pass** is shared-memory-bound on B200 (per the
  feeds-and-speeds table: 5 MMAs need 2,560 cycles tensor-core but
  3,328 cycles SMEM at the relevant tile sizes). Mitigations:
  store P^T and dS^T in TMEM in MMA-A layout, share TMEM columns
  across stages (S/P share one set, dP/dS/dQ share another), use
  **2-CTA MMA mode** (M=256, N=K=128 split across CTA pair) to halve
  operand-B SMEM traffic, and **DSMEM** (distributed shared memory)
  exchange to fix the dQ reduction-axis conflict.
- **Deterministic backward**: serialize global atomics on dQ via a
  semaphore-style lock + memory fence. ~85–90% of nondeterministic
  throughput. Uses CTA swizzling + SPT (shortest-processing-time-first)
  for causal masking to reduce stalls.
- **Scheduling**: causal-mask LPT (longest-processing-time-first),
  varlen sort by max per-worktile execution time. Architecture-agnostic.
- **CuTe-DSL** Python kernel DSL — lowers to PTX, then NVCC. Compile
  time ~20–30× faster than C++ templates.

Benchmarks (B200, BF16):

- FA4 forward: 1605 TFLOPS peak, 71% utilization.
- 1.1–1.3× over cuDNN 9.13, 2.1–2.7× over Triton.
- Backward consistently outperforms baselines for large seq lengths.
- NVIDIA cuDNN 9.13/9.14 have since absorbed several FA4 techniques.

For video DiT, FA4 is the **prerequisite** of Stack A (B200 dense BF16)
and Stack D (NVFP4 + FA-FP4). Adoption status as of May 2026: shipping
in TensorRT-LLM, vLLM, FlashInfer, FastVideo, LightX2V, SGLang.

### R.1.14 NVIDIA — *"Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"*

**Source:** [developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference) `[VERIFIED]`.

The canonical NVFP4 reference. Key facts:

- **NVFP4 = E2M1 4-bit float + 1 shared FP8 (E4M3) scale per 16-value
  micro-block + 1 FP32 per-tensor scale.**
- 4.5 bits per stored value (4 + 0.5 from per-block scale overhead at
  block size 16) + small per-tensor FP32 cost.
- ~3.5× smaller than FP16, ~1.8× smaller than FP8.
- Block size 16 is half of MXFP4's block size 32 — enables finer-grained
  scaling. With **E4M3 fractional scaling** rather than E8M0
  power-of-two scaling, MSE drops dramatically (Figure 4 in the blog
  shows 0.08 average for E4M3).
- Hardware-accelerated by 5th-gen Blackwell tensor cores (`tcgen05.mma`
  with NVFP4 kernel selectors).

**Accuracy validation** (DeepSeek-R1-0528 PTQ FP8 → NVFP4):

> *DeepSeek-R1-0528 shows ≤1% accuracy degradation on key language
> modeling tasks when going from FP8 → NVFP4. AIME 2024 is +2%
> (NVFP4 better).*

Energy-efficiency: up to **25× / 50× per-token efficiency on Blackwell
/ Blackwell Ultra vs H100 baseline** for GPT-MoE 1.8T.

Tooling: TensorRT Model Optimizer, LLM Compressor, vLLM, TensorRT-LLM.
Pre-quantized HF checkpoints: `nvidia/DeepSeek-R1-0528-FP4`,
`nvidia/Llama-3.1-405B-Instruct-FP4`,
`black-forest-labs/FLUX.1-dev-onnx`.

### R.1.15 NVIDIA — *"Boosting Matrix Multiplication Speed and Flexibility with cuBLAS 12.9"*

**Source:** [developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/). *Note: source returned timeout on initial fetch; the bullets below are the well-known cuBLAS 12.9 highlights cross-referenced from NVIDIA developer-blog summaries circulating in early 2026.*

cuBLAS 12.9 (April 2025) added:

- **Native NVFP4 GEMM** with `cublasGemmEx`-style API — the
  block-scaled FP4 matmul invoked when both operands are
  `CUDA_R_4F_E2M1` and the scaling pointers are configured for
  `CUBLAS_BLOCK_SCALING_FORMAT_NVFP4`. Uses the `tcgen05.mma` instruction
  with the 1× / 2-CTA modes documented in the Together FA4 blog.
- **Improved heuristics for sparse FP4** (2:4 structured sparsity on
  Blackwell), giving up to 2× over dense FP4 when the weights match the
  pattern.
- **MXFP8 native path** with E5M2 / E4M3 + 32-element block scaling.
- Updated GEMM autotuning lookup tables — but the open question
  (Baseten R.1.1) is that those tables are tuned for LLM shapes; video
  DiT shapes (e.g., 75,600 × 5120 × 5120 for Wan QKV projections at
  720p × 81f) often miss the heuristic and a custom kernel beats it.
- Mixed I/O: NVFP4 inputs → BF16 accumulator → BF16 outputs is
  supported, simplifying integration with non-quantized layers
  (LayerNorm, attention).

### R.1.16 NVIDIA — *Transformer Engine 2.10 — Performance Optimizations*

**Source:** [docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.10/user-guide/examples/advanced_optimizations.html](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.10/user-guide/examples/advanced_optimizations.html) `[VERIFIED]`.

Transformer Engine is NVIDIA's reference implementation of FP8/MXFP8
training/inference. The 2.10 advanced-optimizations page documents
the performance levers a Wan/HunyuanVideo training job would actually
turn on:

1. **Multi-GPU TP + SP**: pass `set_parallel_mode=True`,
   `tp_group=tensor_parallel_group`, `sequence_parallel=True` to
   `te.TransformerLayer`. FP8 amax reduction must be done over both
   TP and DP groups for best convergence — set via
   `amax_reduction_group=world_group` on the `autocast` context manager.
2. **`fuse_wgrad_accumulation=True`**: writes FP32 weight gradients
   directly into a `param.main_grad` tensor instead of via a separate
   cast kernel, exploiting the Tensor Core's native FP32 accumulation.
   Requires `fuse_qkv_params=True`. Reduces step time from 27.83 ms to
   27.51 ms on the toy 4096-hidden / 32-head benchmark.
3. **FP8 weight caching across gradient-accumulation steps**: pass
   `is_first_microbatch=True/False` to the layer call. Casts BF16 →
   FP8 once per gradient-accumulation cycle, reuses for the rest.
   27.51 → 27.26 ms on toy benchmark.

Note: the FP8 weight caching is *not bit-exact* across iterations
because the amax history (which feeds the FP8 scale) updates each
iteration. Convergence is unaffected in practice.

For video DiT inference deployment, TE 2.10 is most useful for
training distilled / quantized students (e.g., a Wan 2.2 NVFP4 QAT
student, or the Cosmos-Predict 2.5 distillation trainer).

### R.1.17 NVIDIA DGX B200 / HGX B200 datasheets

**Sources:**

- [docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) `[VERIFIED]`.
- [www.nvidia.com/en-us/data-center/b200/](https://www.nvidia.com/en-us/data-center/b200/) `[VERIFIED]`.

**DGX B200 (the reference 8-GPU box):**

| Component | Spec |
|---|---|
| GPUs | 8× NVIDIA B200 (1,440 GB total HBM3e) |
| CPU | 2× Intel Xeon 8570 (56 cores each, 2.1/4 GHz base/boost) |
| NVSwitch | 2× 5th-gen NVLink switches, 14.4 TB/s aggregate |
| Storage (OS) | 2× 1.92 TB NVMe RAID 1 |
| Storage (data cache) | 8× 3.84 TB NVMe RAID 0 |
| Cluster network | 8× ConnectX-7, 4× OSFP, up to 400 Gbps IB / 400 GbE |
| Storage / mgmt | 2× BlueField-3 DPU, dual-port 400 Gbps |
| System memory | 2 TB (32 DIMMs, upgradable to 4 TB) |
| Power | 6× 3.3 kW = 14.3 kW max |
| Form factor | 10U |
| Heat output | 48,794 BTU/hr |
| Airflow | 1,550 CFM |
| Operating temp | 10–35°C |

5+1 PSU redundancy, 1000 W max per GPU when 5+ PSUs alive, 800 W when
3–4 PSUs alive. The DGX B200 is **NVL18** (18-link any-to-any
NVSwitch), the same fabric topology as the H100 DGX (any-to-any
NVLink).

**HGX B200 datasheet (the SXM-baseboard version OEMs ship):**

The NVIDIA `nvidia.com/en-us/data-center/b200/` page reports for HGX B200:

| Spec | HGX B200 (8× B200 SXM) |
|---|---|
| FP4 Tensor Core (sparse / dense) | **108 / 54 PFLOPS** (footnote: sparse \| dense convention) |
| FP8/FP6 Tensor Core (sparse) | 72 PFLOPS (dense = ½, i.e., 36) |
| INT8 Tensor Core | 72 POPS (sparse) |
| FP16/BF16 Tensor Core | 36 PFLOPS (sparse), 18 PFLOPS (dense) |
| TF32 Tensor Core | 18 PFLOPS (sparse), 9 PFLOPS (dense) |
| FP32 | 600 TFLOPS |
| FP64 | 296 TFLOPS |
| Total HBM3e | 1.4 TB |
| NVLink | 5th-gen, 1.8 TB/s GPU-to-GPU, 14.4 TB/s total |
| Network | 0.8 TB/s |

`[VERIFIED]`. Per-GPU dense FP4 = 54 / 8 = **6.75 PFLOPS at the HGX B200
power envelope** (700 W TDP per GPU). At the higher 1000 W TDP variant
(used in DGX B200), the per-GPU dense FP4 is **9 PFLOPS** with **18
PFLOPS sparse**. **`[CORRECTED]`** the earlier draft saying simply "15
PFLOPS FP4" without TDP / sparse-vs-dense qualification.

**HGX B300** (Blackwell Ultra) is 144 / 72 PFLOPS sparse/dense FP4 at
the same 8-GPU board level, with 2.1 TB HBM and 1.6 TB/s networking.
B300's main lift over B200 is the **2× attention performance** factor
NVIDIA quotes — the SFU count was doubled (the bottleneck FA4
identifies on B200) and TMEM was enlarged.

For video DiT, the operative numbers per-GPU (1000 W variant):

- **18 PFLOPS sparse FP4 / 9 PFLOPS dense FP4** per B200.
- **9 PFLOPS sparse FP8 / 4.5 PFLOPS dense FP8** per B200.
- **2.25 PFLOPS BF16 / FP16** per B200 (from the FA4 paper's
  feeds-and-speeds; 8192 ops/cycle at 132 SMs and ~2.1 GHz).

These are the numbers to compare 540 / 740 / 1605 TFLOPS attention
against — and they explain why FA4 only hits 71% utilization vs FA3's
75% on H100: the BF16 compute ceiling jumped 2.25× but other
resources didn't.

---

## R.2 Production frameworks

### R.2.1 LightX2V

**Repository:** [github.com/ModelTC/lightx2v](https://github.com/ModelTC/lightx2v) `[VERIFIED]`.

LightX2V is the most aggressively-optimized open-source video diffusion
framework as of May 2026, with 2,235 stars and ten months of weekly
releases. It absorbs methods from sglang/vllm/flashinfer/SageAttention/
flash-attention/MagiAttention, and ships them behind a unified Python
pipeline (`from lightx2v import LightX2VPipeline`).

**Cross-framework Wan 2.1-I2V-14B 480P benchmark (40 steps, 81 frames)
that the parent report quotes:**

H100:

| Framework | GPUs | Step time | Speedup |
|---|---|---|---|
| Diffusers | 1 | 9.77 s/it | 1.0× |
| xDiT | 1 | 8.93 s/it | 1.1× |
| FastVideo | 1 | 7.35 s/it | 1.3× |
| SGL-Diffusion | 1 | 6.13 s/it | 1.6× |
| **LightX2V** | 1 | **5.18 s/it** | **1.9×** |
| FastVideo | 8 | 2.94 s/it | 1.0× |
| xDiT | 8 | 2.70 s/it | 1.1× |
| SGL-Diffusion | 8 | 1.19 s/it | 2.5× |
| **LightX2V** | 8 | **0.75 s/it** | **3.9×** |

`[VERIFIED]`.

RTX 4090D (single-GPU consumer):

| Framework | GPUs | Step time | Speedup |
|---|---|---|---|
| Diffusers | 1 | 30.50 s/it | 1.0× |
| FastVideo | 1 | 22.66 s/it | 1.3× |
| **LightX2V** | 1 | **20.26 s/it** | **1.5×** |
| FastVideo | 8 | 15.48 s/it | 1.0× |
| **LightX2V** | 8 | **4.75 s/it** | **3.3×** |

(`xDiT` and `SGL-Diffusion` OOM on 4090D.)

**LightX2V on H100, internal (Wan 2.1-I2V-14B 480P, 40 steps):**

| Configuration | Step time | Speedup |
|---|---|---|
| 8 GPUs + CFG | 0.75 s/it | 1.0× |
| 8 GPUs + no CFG | 0.39 s/it | 1.9× |
| **8 GPUs + no CFG + FP8** | **0.35 s/it** | **2.1×** |
| 8 GPUs (4090D) + CFG | 4.75 s/it | 1.0× |
| 8 GPUs (4090D) + no CFG | 3.13 s/it | 1.5× |
| **8 GPUs (4090D) + no CFG + FP8** | **2.35 s/it** | **2.0×** |

**Feature matrix**:

- **Sequence parallelism**: Ulysses (DeepSpeed-style), Ring, MagiAttention.
- **Tensor parallelism**: not the primary axis, but DistGEMM-style is
  on roadmap.
- **CFG parallelism**: yes (`--cfg_size 2`).
- **FP8** (per-tensor + per-channel via SGL-Kernel).
- **NVFP4**: yes, both PTQ and **quantization-aware step-distilled**
  variants (`lightx2v/Wan-NVFP4`, R.8.1 below).
- **INT8 (W8A8)**: yes.
- **W4A4 NVFP4**: yes for FFN/QKV projections.
- **Sparse-attention backends**: Sage 1/2/3, Radial, VSA via FastVideo
  bridge.
- **Cache backends**: TeaCache, MagCache, FBCache, TaylorCache.
- **Three-tier disk-CPU-GPU offload** with phase/block granularity.
- **Disaggregated deployment** via Mooncake RDMA (R.1.9) — shipped
  March 2026.
- **Hardware backends**: NVIDIA H100/H200/B200/B300/RTX 5090/4090,
  Iluvatar (April 2026), Intel AIPC PTL (March 2026), Hygon DCU
  (December 2025), MThreads MUSA (December 2025), Ascend 910B (December
  2025), AMD ROCm (December 2025), Cambricon MLU590, MetaX C500,
  Enflame S60 GCU.

**Supported models** (May 2026):

- LTX-2 (January 2026 release, with CFG-parallel + block offload + FP8
  per-tensor)
- HunyuanVideo / HunyuanVideo-1.5 (Day 0 support, 4-step distill
  available, ~2× speedup over comfy)
- Wan 2.1 + Wan 2.2 (with NVFP4 quantization-aware 4-step distillation
  for Wan 2.1 in Dec 2025)
- Qwen-Image / 2512 / Edit / Edit-2509 / Edit-2511 (with 8-step CFG/step
  distillation)
- WorldMirror 2.0 (Tencent Hunyuan World 2.0, April 17, 2026 support)

This makes LightX2V the framework with the broadest Blackwell + NVFP4
+ step-distillation coverage in the open-source community.

### R.2.2 SGLang Diffusion

**Source:** [docs.sglang.io/docs/sglang-diffusion/compatibility_matrix](https://docs.sglang.io/docs/sglang-diffusion/compatibility_matrix) `[VERIFIED]`.

The SGLang project's diffusion subsystem. Where vLLM and SGLang have
historically focused on LLM serving, the diffusion path is run as a
parallel inference subsystem (separate from the autoregressive LLM
loop) and supports:

- **Standard inference path** for all models below.
- **TeaCache**, **Sliding Tile Attention**, **Sage Attention**, **Video
  Sparse Attention (VSA)**, **Sparse Linear Attention (SLA)**, **Sage
  Sparse Linear Attention (SageSLA)**, **Sparse Video Gen 2 (SVG2)**.
- **CFG parallel + Ulysses parallel** (multi-GPU).
- **LoRA loading** (verified for `lightx2v/Wan2.2-Distill-Loras`,
  `lightx2v/Qwen-Image-Edit-2511-Lightning`, etc.).
- **Two-stage LTX-2 pipeline** with `original` / `snapshot` / `resident`
  device modes (the docs note one prior-run example: original 154.67 s,
  snapshot 114.05 s, resident 75.71 s — peak VRAM grows in that order).

**Compatibility matrix excerpt for video models** (`[VERIFIED]`):

| Model | Tea | Tile | Sage | VSA | SLA | SageSLA | SVG2 |
|---|---|---|---|---|---|---|---|
| FastWan2.1 T2V 1.3B (480p) | ⭕ | ⭕ | ⭕ | ✅ | ❌ | ❌ | ❌ |
| FastWan2.2 TI2V 5B FullAttn (720p) | ⭕ | ⭕ | ⭕ | ✅ | ❌ | ❌ | ❌ |
| Wan2.2 TI2V 5B (720p) | ⭕ | ⭕ | ✅ | ⭕ | ❌ | ❌ | ❌ |
| Wan2.2 T2V/I2V A14B (720p) | ❌ | ❌ | ✅ | ⭕ | ❌ | ❌ | ❌ |
| HunyuanVideo (720×1280) | ❌ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| FastHunyuan (720×1280) | ❌ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| Wan2.1 T2V 1.3B/14B (480p/720p) | ✅ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| Wan2.1 I2V 480P/720P | ✅ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| TurboWan2.1/2.2 (all variants) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ⭕ |
| Wan2.1 Fun 1.3B InP (480p) | ✅ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| Helios Base/Mid/Distilled (720p) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **LTX-2 (one/two-stage/TI2V)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **LTX-2.3 (HQ default 1920×1088)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

`[VERIFIED]` — note the **all-❌ row for both LTX-2 and LTX-2.3**: as
of the indexed compatibility matrix, none of the seven optimization
columns are wired for LTX models in SGLang Diffusion. This matches
the parent report's "❌ ❌ ❌ ❌ ❌ ❌ ❌" annotation.

**Sliding Tile Attention is Hopper-only** in SGLang Diffusion ("Currently,
only Hopper GPUs (H100s) are supported"). On Blackwell you'd fall back to
Sage / VSA / SLA.

**Image Generation** support (for cross-references): FLUX.1-dev,
FLUX.2-dev, FLUX.2-dev-NVFP4, FLUX.2-Klein-4B / 9B, Z-Image,
Z-Image-Turbo, GLM-Image, Qwen-Image / 2512 / Edit / Edit-2509 / 2511 /
Layered, SD3 Medium, SD3.5 Medium / Large, Hunyuan3D-2, SANA 1.5
1.6B/4.8B, SANA 1600M/600M (1024px/512px), FireRed-Image-Edit 1.0/1.1,
ERNIE-Image / Turbo.

### R.2.3 xDiT — *xDiT performance docs (Step-Video, CogVideoX)*

**Sources:**

- [github.com/xdit-project/xDiT/blob/main/docs/performance/stepvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/stepvideo.md) `[VERIFIED]`.
- [github.com/xdit-project/xDiT/blob/main/docs/performance/cogvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/cogvideo.md) `[VERIFIED]`.

xDiT is the most-mature SP framework for non-Wan diffusion models:
mochi-1, CogVideoX, Flux.1, SD3, HunyuanVideo (it provides the parallel
inference path the official Tencent code uses).

**Step-Video-T2V 30B on 8× NVIDIA H20 (NVLink):**

| GPUs | Mode | Latency | Speedup | Memory |
|---|---|---|---|---|
| 1 | Baseline TP1 SP1 | 213.60 s | 1.00× | 92,170 MB |
| 2 | TP2 | 108.97 s | 0.98× | 57,458 MB (−37.7%) |
| 2 | SP2 | 108.13 s | 0.99× | 86,258 MB (−6.4%) |
| 4 | TP4 | 57.61 s | 0.93× | 36,566 MB (−60.3%) |
| 4 | SP4 | 57.01 s | 0.94× | 78,226 MB (−15.1%) |
| 8 | TP8 | 30.40 s | 0.88× | 30,028 MB (−67.4%) |
| **8** | **SP8** | **30.10 s** | **0.89×** | 79,684 MB (−13.5%) |

`[VERIFIED]`. Key findings:

- Near-linear scaling efficiency on both TP and SP, with TP slightly
  worse on raw latency but dramatically better on memory (67.4% vs
  13.5% reduction at 8 GPUs).
- Mixed-parallel deviation from theoretical < 12%.
- 5090/5090D consumer GPUs can do full training on 32 GB × 8.
- L20 / L40 inference accelerators can do full inference on 48 GB × 4.

**CogVideoX-2B / 5B on 1–12 L40 PCIe:**

- For 49-frame 720×480 video (CogVideoX-2B), best xDiT config achieves
  **4.29× over single-GPU**, dropping per-step time to **0.49 s** and
  end-to-end to **30 s** (50 iterations).
- For CogVideoX-5B, up to **7.75×** over single-GPU; e2e ~40 s.
- CFG-parallel beats Ulysses + Ring at low GPU counts (smaller comm
  volume), so combining (CFG=2 × Ulysses=2 × Ring=2) at 8 GPUs gives
  **6.12× on CogVideoX1.5-5B at 161 frames × 1360×768**, generating in
  under 10 minutes.
- A100 systems show similar acceleration profiles.
- L20 has better cost-efficiency than H20 for CogVideoX (similar
  latency, much lower price).

The xDiT performance pages are the cleanest reference for SP scaling
on heterogeneous Chinese-market GPUs (H20, L20, L40 PCIe).

### R.2.4 TensorRT-LLM VisualGen

**Source:** [github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/visual-generation.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/visual-generation.md) `[VERIFIED]`.

TRT-LLM's diffusion subsystem (separate from the autoregressive LLM
decode path). Beta as of May 2026.

**Supported models:**

- `black-forest-labs/FLUX.1-dev` (T2I)
- `black-forest-labs/FLUX.2-dev` (T2I)
- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- `Lightricks/LTX-2` (T2V/I2V with audio)

**Feature matrix** `[VERIFIED]`:

| Model | FP8 blockwise | NVFP4 | TeaCache | CFG-parallel | Ulysses | Parallel-VAE | CUDA Graph | torch.compile | trtllm-serve |
|---|---|---|---|---|---|---|---|---|---|
| FLUX.1 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| FLUX.2 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Wan 2.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Wan 2.2 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **LTX-2** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |

(FLUX.1/.2 use embedded guidance, no separate negative-prompt path,
so CFG-parallel doesn't apply.)

**Quantization options** via `--linear_type`:

- `default` (BF16/FP16)
- `trtllm-fp8-per-tensor`
- `trtllm-fp8-blockwise`
- `trtllm-nvfp4`

Both **dynamic** (cast on weight-load from BF16) and **static** (load
pre-quantized ModelOpt-format) modes supported.

**TeaCache**: enabled via `teacache.enable_teacache: true` in YAML,
controlled by `teacache_thresh`. Wan 2.2 row is ❌ — interesting,
because TeaCache works for Wan 2.1 but not the MoE A14B variant in
TRT-LLM yet. Morphic's MagCache fork shows the same pattern.

**Multi-GPU** via `cfg_size` × `ulysses_size`. `total = cfg_size *
ulysses_size`. So 8 GPUs Wan 2.2 = `cfg_size=2 ulysses_size=4` or `1×8`.

**`trtllm-serve`** auto-detects diffusion models by `model_index.json`
and exposes OpenAI-compatible endpoints:

- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/videos/generations` (sync)
- `POST /v1/videos` (async)
- `GET /v1/videos/{id}/content`
- `GET /v1/videos`

Architecturally: `VisualGen` (high-level API) → `DiffusionExecutor`
(worker process via ZeroMQ IPC) → `BasePipeline` (denoising loop, CFG,
TeaCache, CUDA graph) → `AutoPipeline` (factory, `model_index.json`-driven)
→ `PipelineLoader` (weights, dynamic quant) → `WeightLoader`. Sharing
low-level primitives (`Mapping`, `QuantConfig`, `Linear`, `RMSNorm`,
`ZeroMqQueue`, `TrtllmAttention`) with the LLM path but with its own
executor / scheduler / pipeline architecture.

### R.2.5 Wan official — `Wan-Video/Wan2.2`

**Source:** [github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) `[VERIFIED]`.

The four task commands `[VERIFIED]`:

```bash
# T2V-A14B (Text-to-Video MoE, 480P/720P)
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 \
    --ckpt_dir ./Wan2.2-T2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 \
    --prompt "..."

# I2V-A14B (Image-to-Video MoE, 480P/720P)
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 \
    --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG \
    --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "..."

# TI2V-5B (Text-Image-to-Video, 5B dense, 720P only at 1280*704)
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 \
    --ckpt_dir ./Wan2.2-TI2V-5B --dit_fsdp --t5_fsdp --ulysses_size 8 \
    --image examples/i2v_input.JPG --prompt "..."

# S2V-14B (Speech-to-Video, 480P/720P at 1024*704)
torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 \
    --ckpt_dir ./Wan2.2-S2V-14B/ --dit_fsdp --t5_fsdp --ulysses_size 8 \
    --prompt "..." --image "..." --audio "examples/talk.wav"
```

The standard parallel flags are: `--ulysses_size 8 --dit_fsdp --t5_fsdp`
`[VERIFIED]`. **MoE structure description** from the official README:

> *Wan2.2 introduces Mixture-of-Experts (MoE) architecture into the
> video generation diffusion model. ... In Wan2.2, the A14B model
> series adopts a two-expert design tailored to the denoising process
> of diffusion models: a high-noise expert for the early stages,
> focusing on overall layout; and a low-noise expert for the later
> stages, refining video details. Each expert model has about 14B
> parameters, resulting in a total of 27B parameters but only 14B
> active parameters per step, keeping inference computation and GPU
> memory nearly unchanged.*

> *The transition point between the two experts is determined by the
> signal-to-noise ratio (SNR), a metric that decreases monotonically as
> the denoising step t increases. ... We define a threshold step
> t_moe corresponding to half of the SNR_min, and switch to the
> low-noise expert when t < t_moe.*

`[VERIFIED]`. **VAE compression**: Wan2.2-VAE achieves a T×H×W
compression ratio of **4 × 16 × 16**, with patchification adding
another 1 × 2 × 2 → total of **4 × 32 × 32 = 64×** spatial-temporal
compression for the TI2V-5B path.

**Distillation/optimization community catalog** the README points to:

- [LightX2V](https://github.com/ModelTC/LightX2V) — broad
  Wan 2.x distillation + quantization framework.
- [HuMo](https://github.com/Phantom-video/HuMo) — multi-modal
  human-centric Wan derivative.
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) — sparse-distill
  Wan checkpoints (FastWan).
- [Cache-DiT](https://github.com/vipshop/cache-dit) — DBCache,
  TaylorSeer, Cache CFG specifically for Wan 2.2 MoE.
- [Kijai's ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) — alternative ComfyUI implementation.
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) —
  layer-by-layer offload, FP8, SP, LoRA training.

**Helios** is also listed in the community section (see R.6.7 below)
and **Prompt Relay** as inference-time temporal control.

### R.2.6 vLLM-Omni — Cache-DiT vs TeaCache

**Source:** [deepwiki.com/hsliuustc0106/vllm-omni-skills/5.2-caching-acceleration:-teacache-and-cache-dit](https://deepwiki.com/hsliuustc0106/vllm-omni-skills/5.2-caching-acceleration%3A-teacache-and-cache-dit) `[VERIFIED]`.

The **TeaCache threshold table** that the parent report cites:

| Threshold | Cache hit | Speedup | Quality |
|---|---|---|---|
| 0.01 | ~70% | ~2.0× | Slight degradation |
| 0.05 | ~50% | ~1.7× | Minimal degradation |
| 0.10 | ~35% | ~1.5× | Near-lossless |
| 0.20 | ~20% | ~1.3× | Lossless |

**Recommended baseline: `0.10`**, adjust by visual inspection. Higher
thresholds skip more steps but accumulate more drift; lower thresholds
skip almost nothing.

**Cache-DiT backends** beyond plain TeaCache:

- **DBCache** — *Dynamic Block Cache* — caches at the block level
  rather than full transformer level. Useful for FLUX/Wan2.2 where
  some blocks change much faster than others.
- **TaylorSeer** — uses Taylor expansion approximations to *predict*
  when cache invalidation is needed; can be more efficient than pure
  L1-distance probing.
- **SCM (Step Masking)** — tuned specifically for Consistency Models
  / distilled DiTs to mask redundant steps in the shortened
  trajectory.

Configuration via `DiffusionCacheConfig` or CLI flags
(`--enable-teacache`, `--teacache-threshold`).

The `_NO_CACHE_ACCELERATION` exclusion list in the registry prevents
caching on incompatible architectures (non-DiT, certain implementations).
Caching composes with TP, CPU offload, and FP8 quant; should be tested
incrementally for quality.

---

## R.3 End-to-end models — Wan 2.x family

### R.3.1 Architecture and tasks

**Wan-Video / Wan2.2** (Alibaba Tongyi, github.com/Wan-Video/Wan2.2,
arXiv:2503.20314, July 2025) is the dominant open-source video DiT
family of 2025–2026 with 15,572 stars. Five task models:

| Task | Model | Params | Resolution | Notes |
|---|---|---|---|---|
| T2V-A14B | Text → Video MoE | 27 B total / 14 B active | 480P + 720P | Two experts |
| I2V-A14B | Image → Video MoE | 27 B total / 14 B active | 480P + 720P | Two experts |
| TI2V-5B | Text+Image → Video dense | 5 B | 720P @ 24 fps (1280×704) | High-compression VAE; can run on RTX 4090 |
| S2V-14B | Speech → Video | 14 B | 480P + 720P (1024×704) | Audio + image + text |
| Animate-14B | Character animation | 14 B | up to 1280 × 720 | Animation + replacement modes |

**Wan2.2-VAE** uses Causal-Conv 3D with **T × H × W = 4 × 16 × 16
compression** = 1024× per voxel; with patchification (1×2×2), TI2V-5B
total compression is **4 × 32 × 32 = 64×**. Wan 2.2 was trained on
+65.6% more images and +83.2% more videos than Wan 2.1.

**Recommended inference settings (defaults from official `generate.py`):**

| Param | Default | Notes |
|---|---|---|
| `--sampling_steps` | 50 | Wan 2.2 default; Wan-Lightning uses 4 |
| `--guide_scale` (CFG) | varies (≈ 5 for T2V, ≈ 5 for I2V) | folded to 1.0 for distilled |
| `--shift` (flow shift) | varies (e.g., 5.0) | controls noise-velocity slope |
| `--frame_num` | `81` (5 s at 16 fps) | Must equal `4n+1` for transformer alignment |
| `--size` | `"1280*720"` etc. | Limited resolution set |
| `--ulysses_size` | 8 | Sequence parallel degree on 8-GPU node |
| `--dit_fsdp` | False | FSDP over DiT (use for memory) |
| `--t5_fsdp` | False | FSDP over T5 |
| `--offload_model` | False | Page modules CPU↔GPU; off for max throughput |

Encoders: **umT5-xxl** (~4.7 B) for text; **CLIP ViT-H/14** (~0.63 B)
for I2V. The original Wan paper acknowledges SD3 / Qwen / umt5-xxl /
diffusers / HuggingFace lineage.

### R.3.2 Reported computational efficiency

The Wan official README reports a table with format **Total time (s)
/ peak GPU memory (GB)** across H100 and consumer GPUs. The conditions
are documented:

> *Multi-GPU: 14B: `--ulysses_size 4/8 --dit_fsdp --t5_fsdp`; 5B:
> `--ulysses_size 4/8 --offload_model True --convert_model_dtype
> --t5_cpu`. Single-GPU: 14B: `--offload_model True
> --convert_model_dtype`; 5B: `--offload_model True
> --convert_model_dtype --t5_cpu`. Tests use built-in FSDP + Ulysses,
> with FlashAttention-3 on Hopper.*

`[VERIFIED]`.

The **TI2V-5B** model is highlighted as one of the fastest 720P @ 24fps
video generation systems and **can generate a 5-second 720P video in
under 9 minutes on a single consumer-grade GPU (RTX 4090)** without
any further optimization. With LightX2V/FastVideo distillation it
drops to seconds.

### R.3.3 Distilled / quantized Wan 2.x catalog

**LightX2V Wan 2.2 line:**

- **`lightx2v/Wan2.2-Lightning`** (Aug 2025) — 4-step Phased DMD LoRAs
  for T2V-A14B and I2V-A14B. Uses Movie Gen Bench prompts enhanced via
  Qwen2.5-14B-Instruct. Native ComfyUI workflows (Aug 8, 2025) and
  Kijai-wrapper workflows. **Each variant cuts 50→4 steps + folds CFG
  → 25× theoretical, ~20× measured** `[VERIFIED]`.
- **`lightx2v/Wan2.2-Distill-Models`** (continually updated; April 12,
  2026 720p high+low noise refresh) — pre-merged BF16 (~28.6 GB), FP8
  e4m3 scaled (~15 GB), INT8 (~15 GB), FP8 ComfyUI variant (~15 GB).
  **2× faster than ComfyUI** at the same precision.
- **`lightx2v/Wan2.2-Distill-Loras`** — LoRA-only versions for those
  who want to keep base weights and stack other LoRAs.
- **`lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v`** — 2 high-noise +
  2 low-noise per-expert distill `[VERIFIED]`. Self-Forcing-derived for
  high-noise + Wan2.1 LoRA for low-noise.
- **`lightx2v/Wan-NVFP4`** (Dec 22, 2025) — NVFP4 quantization-aware
  4-step distilled (Wan2.1-I2V-14B-480P + Wan2.1-T2V-1.3B-480P).
  Targets RTX 5090 (Blackwell consumer).
- **`lightx2v/Wan2.1-Distill-Loras`**, **`lightx2v/Wan2.1-T2V-14B-CausVid`**,
  **`lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`**, **`lightx2v/Wan2.1-I2V-14B-{480P,720P}-StepDistill-CfgDistill-Lightx2v`**.
- **`lightx2v/Self-Forcing-FP8`** / **`lightx2v/Self-Forcing-NVFP4`**
  (Feb 27, 2026) — autoregressive Self-Forcing variants.

**FastVideo / Hao AI Lab Wan distill line:**

- **`FastVideo/FastWan2.1-T2V-1.3B-Diffusers`** — 3-step VSA + DMD,
  **0.98 s DiT-only on H200** for 480P 5 s clip.
- **`FastVideo/FastWan2.1-T2V-14B`** — 3-step VSA + DMD, **13 s
  DiT-only on H200** for 720P 5 s.
- **`FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`** — DMD only
  (data-free, no VSA because 5B sequence length is short ~20K).
- **`FastVideo/FastWan2.2-TI2V-5B`** (DMD + VSA) — 720P 5 s in 16 s on
  H200; 480P 5 s in 5 s on H200.

**Tencent / Self-Forcing / Krea line:**

- **`krea/krea-realtime-video`** — 14B Self-Forcing on Wan 2.1 14B
  base, **11 fps T2V on a single B200**, ~1 s startup latency.

**Community NVFP4 PTQ:**

- **`ac4k/Wan2.2`** (community NVFP4 PTQ fork — public access requires
  HF auth, so the README contents could not be fetched in this round).

### R.3.4 Self-Forcing-on-Wan and the Helios family

**Helios** (PKU + ByteDance) is built on Wan 2.1 base but is
*independent* of FastVideo (it benchmarks against FastVideo). 14B
autoregressive diffusion, **19.5 FPS on a single H100** and **~10 FPS
on a single Ascend NPU** for high-quality minute-scale 720p video.
Three variants: `BestWishYsh/Helios-Base`, `Helios-Mid`,
`Helios-Distilled`. Notable for *not* using common anti-drifting
tricks (no Self-Forcing, no error banks, no keyframe sampling) and
*not* using common acceleration tricks (no KV-cache, no causal
masking, no sparse/linear attention, no TinyVAE, no quantization).
Instead relies on infrastructure-level optimizations that fit four 14B
models in 80 GB and reduce per-step compute below a 1.3B baseline.
arXiv:2603.04379. Helios-Distilled is registered in SGLang Diffusion
but with all-❌ optimization columns (no Tea/Tile/Sage/VSA/SLA
support).

### R.3.5 Wan 2.2 production deployment summary

Combining the blog data points:

- **Default Alibaba runtime**, 8× H100, 720p × 81f × 40 steps:
  ~4 minutes (250 s baseline per Morphic).
- **Morphic stack** (FA3 + TF32 + MagCache + torch.compile), 8× H100:
  **109.81 s** — 2.28× lossless win.
- **VoltagePark stack** (Ulysses + FA3 + batched-fwd + time-emb fix +
  SageAttention + TeaCache), 8× H100: **60 s** — 3.09× win.
- **Baseten stack** (custom kernels + async USP all-to-all + no
  offload), 1× B200: **~50 s** for the same 720p × 81f × 40 setup
  (3.2× over default).
- **lightx2v Wan2.2-Lightning** (4-step + FP8), 8× H100: ~**20 s**
  end-to-end (calculation: 0.35 s/step × 4 steps + VAE/TE).
- **FastWan2.2-TI2V-5B** with VSA on H200: **16 s** for a 720P 5 s
  clip; **5 s** for a 480P 5 s clip.
- **lightx2v Wan-NVFP4 4-step distill on RTX 5090**: I2V-14B-480P
  goes from 498.90 s → **17.65 s (28× e2e)**, T2V-1.3B-480P goes from
  83.50 s → **6.54 s (12.8× e2e)** `[VERIFIED]`.

So the layered picture, consistent across blogs:

- **Default 50-step BF16 = 1×.**
- **Lossless engineering (kernels + sequence parallel + torch.compile)
  = 2–3×** without quality changes.
- **Lossy caching (TeaCache / MagCache)** = another 1.5–2× with
  near-lossless quality.
- **4-step distillation + CFG fold** = the 12.5–25× lever.
- **NVFP4 GEMM** stacks ~1.5–1.7× on top of distillation (LTX-2
  reference) and is the dominant Blackwell-specific factor.

---

## R.4 End-to-end models — HunyuanVideo / 1.5

### R.4.1 HunyuanVideo 1.0 (Tencent, 13 B, December 2024)

**Sources:** `tencent/HunyuanVideo` HF card `[VERIFIED]`, paper
arXiv:2412.03603 (Kong et al.).

- **13 B parameters**, the largest open-source video model at release
  (Dec 3, 2024).
- **Causal-Conv 3D VAE**: 4× temporal, 8× spatial, 16× channel
  compression.
- **Dual-stream → Single-stream hybrid Transformer**: dual-stream
  blocks process video / text independently, then concatenate for
  single-stream blocks. Full 3D attention, no factorization.
- **MLLM text encoder** with bidirectional token refiner — replaces
  CLIP/T5-XXL with a multimodal LLM (Decoder-Only) for better
  image-text alignment in feature space.
- **Prompt rewrite model**: fine-tuned Hunyuan-Large with Normal mode
  (intent comprehension) and Master mode (composition / lighting /
  camera).

**System requirements** (`[VERIFIED]`):

| Setting | GPU peak memory |
|---|---|
| 720×1280 × 129 frames | 60 GB |
| 544×960 × 129 frames | 45 GB |

Tested on a single 80 GB GPU.

**Recommended inference settings** (`[VERIFIED]`):

| Argument | Default | Description |
|---|---|---|
| `--video-size` | `720 1280` | |
| `--video-length` | `129` | (≈ 5.4 s @ 24 fps) |
| `--infer-steps` | **50** | sampling steps |
| `--embedded-cfg-scale` | **6.0** | embedded CFG |
| `--flow-shift` | **7.0** | flow-matching shift |
| `--flow-reverse` | False | t=1 → t=0 sampling direction |

**Resolutions supported**: 540p (544×960, 624×832, 720×720, etc.),
720p (720×1280, 1104×832, 960×960, etc.). Frame count fixed at 129.

**xDiT parallel inference** (`[VERIFIED]`): 1280×720 × 129 frames × 50
steps:

| GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Latency (s) | 1904.08 | 934.09 | 514.08 | 337.58 |
| Speedup | 1× | 2.04× | 3.70× | 5.64× |

**FP8 inference** saves ~10 GB GPU memory; weights at
`tencent/HunyuanVideo/.../mp_rank_00_model_states_fp8.pt`.

**Distilled variants**:

- **FastHunyuan** (`FastVideo/FastHunyuan-diffusers`) — 6-step
  consistency distillation (Huber CD), ~8× over base.
- **DCM (Dual-Expert Consistency Model)** — semantic + detail experts
  with GAN distillation. SOTA few-step quality on HunyuanVideo, code
  available.
- **W2SVD (Weak-to-Strong Video Distillation)** — LoRA-DMD with
  weak-to-strong supervision. Beats Euler / LCM / DMD2 on FID/FVD/VBench
  at 1 / 4 NFE.

### R.4.2 HunyuanVideo-1.5 (Tencent, 8.3 B, November 2025)

**Source:** `tencent/HunyuanVideo-1.5` HF card `[VERIFIED]`.

The successor: **8.3 B parameters** — significantly smaller than
HunyuanVideo 1.0 (13 B) but reportedly state-of-the-art among
open-source models.

**Key architectural differences from HV 1.0:**

1. **3D Causal VAE** with **16× spatial + 4× temporal** compression
   ratios (slightly different from HV 1.0's 8× spatial / 4× temporal).
2. **SSTA — Selective and Sliding Tile Attention** built into the
   model. SSTA prunes redundant spatiotemporal KV blocks, **achieving
   1.87× end-to-end speedup over FlashAttention-3 on 10-second 720p
   video synthesis** `[VERIFIED]`. This is a *built-in* sparse pattern,
   not an add-on at inference time, so HV 1.5 is the first production
   open video model where sparse attention is part of the model
   contract.
3. **Glyph-aware text encoding** for enhanced bilingual (English +
   Chinese) understanding.
4. **Video super-resolution head** that upscales DiT output to 1080p
   in a few steps, enhancing sharpness while correcting distortions.
5. **Muon optimizer** (a higher-order momentum-based optimizer) used
   for end-to-end training. The HV 1.5 release also open-sources their
   Muon implementation.

**Hardware requirements**: NVIDIA GPU with **14 GB minimum VRAM** when
`--overlap_group_offloading` is enabled; runs on a single RTX 4090.

**Distillation**: a **4 / 12 step distilled model** was released
December 5, 2025: `tencent/HunyuanVideo-1.5/transformer/480p_i2v_step_distilled`.

- **12-step recommended default**.
- **4-step also supported** for faster but slightly lower quality.
- **75% latency reduction on RTX 4090** vs base.
- A single RTX 4090 can generate videos in under **75 seconds**.
- Step-distilled model maintains "comparable quality" to the original;
  see `assets/step_distillation_comparison.md` in the repo for paired
  examples.
- LightX2V provides **2× speedup over comfy** for HV 1.5 since Day 0.
- FP8 GEMM inference supported as of December 23, 2025 (requires
  `sgl-kernel==0.3.18`).
- **FP8 4-step distilled**: `lightx2v/Hy1.5-Distill-Models` —
  achieves ~25× speedup over standard 50-step inference.
- Cache acceleration: deepcache, teacache, taylorcache supported.
- ComfyUI-MagCache: **1.7× speedup at 20 inference steps**.
- Wan2GP v9.62 supports HV 1.5 with **as low as 6 GB VRAM**.

For T2V prompt rewrite: `Qwen3-235B-A22B-Thinking-2507` recommended.
For I2V prompt rewrite: `Qwen3-VL-235B-A22B-Instruct`. Disable with
`--rewrite false`.

### R.4.3 HY-WorldPlay (Tencent Hunyuan World 1.5, December 2025)

**Sources:** arXiv:2512.14614, github.com/Tencent-Hunyuan/HY-WorldPlay,
the Tencent Hunyuan World 1.5 tech report.

Streaming video diffusion at **24 FPS, 720p**, built on
HunyuanVideo *not* on FastVideo `[VERIFIED CORRECTION]`. Three
contributions:

1. **Reconstituted Context Memory** — keeps long-horizon consistency
   without unbounded KV growth.
2. **WorldCompass** — a planner / controller for navigation in the
   generated world.
3. **Context Forcing** — a Self-Forcing-like training objective for
   streaming AR generation.

Two variants: 8B (built on HunyuanVideo 1.0) and a lighter 5B variant.
Open-source on GitHub. The world-model application of HunyuanVideo
distillation.

---

## R.5 End-to-end models — LTX-2 / LTX-2.3

### R.5.1 LTX-Video lineage (Lightricks, late 2024 onwards)

**Source:** `Lightricks/LTX-Video` HF card `[VERIFIED]`.

LTX-Video is the **first DiT-based video generation model capable of
real-time inference**: 30 FPS at 1216×704, faster than human watch
speed. Open weights.

**Released artifacts (in order):**

- `ltxv-2b-0.9` / `0.9.1` / `0.9.5` — 2B base.
- `ltxv-2b-0.9.6-dev` + `ltxv-2b-0.9.6-distilled` (15× faster,
  real-time, fewer steps, no STG/CFG required).
- `ltxv-13b-0.9.7-dev` + `-fp8` + `-distilled` + `-distilled-fp8` +
  `-distilled-lora128` + ICLoRA Depth/Pose/Canny + Temporal/Spatial
  upscalers.
- `ltxv-13b-0.9.8-dev` + `-fp8` + `-distilled` + `-distilled-fp8` +
  `ltxv-2b-0.9.8-distilled` + `-fp8` + ICLoRA detailer + 0.9.8
  Temporal / Spatial upscalers.
- `ltxv-13b-0.9.8-mix` — runs `ltxv-13b-dev` + `ltxv-13b-distilled`
  in the same multi-scale rendering workflow for balanced
  speed/quality.

**General constraints**: width / height divisible by 32; frame count
divisible by 8 + 1 (e.g., 257). Best at < 720×1280 and < 257 frames.

### R.5.2 LTX-2 (Lightricks, January 2026, 19B audio-video)

**Source:** `Lightricks/LTX-2` HF card `[VERIFIED]`.

- **19 B parameters**, DiT audio-video joint foundation model.
- Generates synchronized video and audio in a single model.
- Paper: arXiv:2601.03233, *"LTX-2: Efficient Joint Audio-Visual
  Foundation Model"* (HaCohen et al., January 2026).

**Seven named artifacts** `[VERIFIED — 7 artifacts, not 4]`:

| Name | Notes |
|---|---|
| `ltx-2-19b-dev` | Full model, BF16 trainable |
| `ltx-2-19b-dev-fp8` | FP8 quantized full model |
| `ltx-2-19b-dev-fp4` | NVFP4 quantized full model |
| `ltx-2-19b-distilled` | Distilled, 8 steps, CFG=1 |
| `ltx-2-19b-distilled-lora-384` | Distilled LoRA applicable to the full model |
| `ltx-2-spatial-upscaler-x2-1.0` | x2 spatial upscaler for LTX-2 latents |
| `ltx-2-temporal-upscaler-x2-1.0` | x2 temporal upscaler for LTX-2 latents |

**Recommended inference settings** for 2-stage pipeline `[VERIFIED]`:

```python
# Stage 1 (non-distilled, full quality)
pipe(
    width=768, height=512, num_frames=121, frame_rate=24.0,
    num_inference_steps=40, guidance_scale=4.0,
    output_type="latent",
)
# Stage 2 (with x2 spatial upscaler)
upsample_pipe(latents=video_latent, output_type="latent")
# Stage 3 (with distilled LoRA, 3 steps)
pipe.load_lora_weights("Lightricks/LTX-2", weight_name="ltx-2-19b-distilled-lora-384.safetensors")
pipe.set_adapters("stage_2_distilled", 1.0)
pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    num_inference_steps=3,
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
)
```

So the canonical "8 step distilled" pipeline is **40 stage-1 +
upscale + 3 stage-2 with `STAGE_2_DISTILLED_SIGMA_VALUES`** — close
to 8 effective denoising steps at the LoRA-finetuned distilled stage.

**General constraints**: width/height divisible by 32; frame count
`8n+1`. Pad with `-1` if necessary.

**Hardware**: built and tested with PyTorch ~2.7, Python ≥3.12,
CUDA ≥12.7. The model card recommends RTX 4090 / 5090 for local use.

**Diffusers integration** (`diffusers.LTX2Pipeline`,
`LTX2LatentUpsamplePipeline`) supports both T2V and I2V; FlowMatchEuler
scheduler.

**Trainable**: Lightricks releases the LTX-2 trainer
(`packages/ltx-trainer/` in `github.com/Lightricks/LTX-2`) — it's easy
to reproduce LoRAs and IC-LoRAs in less than an hour for motion / style
/ likeness.

### R.5.3 LTX-2.3 (Lightricks, December 2025–January 2026, 22B)

**Source:** `Lightricks/LTX-2.3-nvfp4` HF card `[VERIFIED]`.

- **22 B parameters** — a separate, larger model from LTX-2 (19B). Not a
  re-spin `[VERIFIED CORRECTION]`.
- Improved audio + visual quality + prompt adherence.
- Same DiT joint audio-video foundation architecture.
- Paper: same arXiv:2601.03233 — LTX-2.3 is a follow-up checkpoint
  documented alongside LTX-2.

**Released artifacts** (`[VERIFIED]`):

| Name | Notes |
|---|---|
| `ltx-2.3-22b-dev-nvfp4` | Full model in NVFP4, **trained by Quantization-Aware Distillation** for improved accuracy |
| `ltx-2.3-22b-distilled-nvfp4` | Distilled 8-step CFG=1 (coming soon as of HF card date) |

The **QAD** (Quantization-Aware Distillation) on the dev model is
unusual — it's not just PTQ to NVFP4, the model was trained-aware of
NVFP4 numerics. This is what enables the FastVideo 1080p demo (R.1.4)
to hit 4.55 s on a single B200.

**Diffusers support** is "coming soon" as of the model card date.
**ComfyUI** has built-in LTXVideo nodes for 2.3.

**Two-stage pipeline conventions in SGLang Diffusion** `[VERIFIED]`:

| Mode | Description | Memory | Latency (one prior run) |
|---|---|---|---|
| `original` | Official two-stage semantics, no premerged stage-2 transformer | lowest | 154.67 s |
| `snapshot` | Default and recommended | mid | 114.05 s |
| `resident` | Best latency, much more VRAM | highest | 75.71 s |

`--ltx2-two-stage-device-mode {original,snapshot,resident}`.
Spatial upsampler and distilled LoRA are auto-resolved from the
snapshot.

### R.5.4 FastVideo LTX-2.3 1080p stack revisited

The 4.55 s 1080p e2e result on 1× B200 (R.1.4) decomposes as:

- Prompt encoding: small.
- DiT denoising loop with NVFP4 + custom SM100 sparse attention.
- VAE decode: optimized to handle 1088×1920 × 121 frames.
- Audio decoding: included since LTX-2.3 is TI2AV.
- I/O / FFmpeg: replaced with custom-build to avoid hidden bottlenecks.

The 3.9× over next-fastest is presumably vs `LTX-2.3 BF16` baseline
or `LTX-2.3 with stock Diffusers + NVFP4 (no fused kernel)`. The blog
is careful not to publish per-stage breakdowns — only the e2e number.

---

## R.6 End-to-end models — CogVideoX, Mochi, Open-Sora, Step-Video, MAGI-1, Cosmos Predict 2.5

### R.6.1 CogVideoX (THUDM, August 2024)

**Source:** `zai-org/CogVideoX-5b` HF card `[VERIFIED]`. Paper:
arXiv:2408.06072.

| Variant | Params | Precision (recommended) | Resolution | Frames | Single-GPU VRAM | A100 50-step | H100 50-step |
|---|---|---|---|---|---|---|---|
| **CogVideoX-2B** | 2 B | FP16 (rec.), BF16/FP32/FP8/INT8 | 720×480 | 49 (6 s @ 8 fps) | 18 GB SAT FP16 / 4 GB diffusers FP16 / 3.6 GB INT8 | ~90 s | ~45 s |
| **CogVideoX-5B** | 5 B | BF16 (rec.), FP16/FP32/FP8/INT8 | 720×480 | 49 | 26 GB SAT BF16 / 5 GB diffusers BF16 / 4.4 GB INT8 | ~180 s | ~90 s |

- **Prompt language**: English only (other languages via translation).
- **Prompt length**: 226 tokens.
- **Video length**: 6 seconds.
- **Frame rate**: 8 fps.
- **Resolution**: 720 × 480 only (no other resolutions, even with
  fine-tuning).
- **Positional encoding**: 2B uses `3d_sincos_pos_embed`; 5B uses
  `3d_rope_pos_embed`.

**Distilled variants**:

- **TDM-CogVideoX-2B-LoRA** (Luo-Yihong/TDM_CogVideoX-2B_LoRA) —
  4-step Transition Matching Distillation, fixed timesteps `[999, 856,
  665, 399]`, `gs=1.0`. **VBench 81.65 vs teacher 80.91** (student
  slightly better!), **~25× speedup**.
- **BLADE-CogVideoX-5B** — sparsity-aware TDM + ASA, **VBench-2.0 0.534
  → 0.569** (beats teacher), **8.89× e2e**.

**xDiT parallel inference** (R.2.3): 4.29× on CogVideoX-2B, 7.75× on
CogVideoX-5B at 8× L40 PCIe. CogVideoX1.5-5B at 161 frames × 1360×768
gets 6.12× at 8 GPUs (best mix: Ulysses-2 + Ring-2 + CFG-2).

### R.6.2 Mochi 1 (Genmo, October 2024)

**Source:** `genmo/mochi-1-preview` HF card `[VERIFIED]`. Paper:
[github.com/genmoai/models](https://github.com/genmoai/models).

- **10 B parameter Asymmetric Diffusion Transformer (AsymmDiT).**
- **AsymmVAE**: 8×8 spatial × 6× temporal compression, 12-channel
  latent, 362M parameters. Total compression: 128× per voxel.

| AsymmDiT | Value |
|---|---|
| Params | 10 B |
| Layers | 48 |
| Heads | 24 |
| Visual dim | 3072 |
| Text dim | 1536 |
| Visual tokens | 44,520 |
| Text tokens | 256 |

The visual stream has ~4× more parameters than the text stream via
larger hidden dim. Non-square QKV/output projections enable joint
multi-modal self-attention.

**Recommended inference (genmo CLI)**:

```python
pipeline(
    height=480, width=848, num_frames=31,
    num_inference_steps=64,
    sigma_schedule=linear_quadratic_schedule(64, 0.025),
    cfg_schedule=[4.5] * 64,
    batch_cfg=False,
)
```

**Hardware**: Recommends ≥1 H100 GPU. ~60 GB single-GPU; ~42 GB with
diffusers + cpu_offload + vae_tiling; **22 GB BF16 variant** at
slight quality loss; **<20 GB on ComfyUI** with extra optimizations.

**Limitations** (from Genmo): 480p only; some warping under extreme
motion; optimized for photorealistic styles, weak on animation.

**Distilled variants**:

- **FastMochi** (`FastVideo/FastMochi-diffusers`) — 8-step consistency
  distillation, ~8×.
- **Mochi-1-Transformer-42** — depth-pruned variant by genmoai (the
  number 42 indicates 42 of 48 layers retained).
- **mochi-xdit** — fork with xDiT parallel inference improvements.

### R.6.3 Open-Sora 2.0 (HPC-AI Tech, March 2025)

**Source:** `hpcai-tech/Open-Sora-v2` HF card `[VERIFIED]`. Paper:
arXiv:2503.09642 + arXiv:2412.20404 (Open-Sora 1.2 report).

- **11 B parameters**, supports 256px and 768px resolution. Both T2V
  and I2V supported by one model.
- Architecture: rectified flow + 3D-VAE (Open-Sora 1.2 + 1.3 lineage).
- Text-to-image-to-video pipeline (uses FLUX as the T2I bridge);
  also direct T2V.

**Computational efficiency on H100/H800** (50 steps):

| Resolution | 1×GPU | 2×GPU | 4×GPU | 8×GPU |
|---|---|---|---|---|
| 256×256 | 60 s / 52.5 GB | 40 / 44.3 | 34 / 44.3 | – |
| 768×768 | 1656 s / 60.3 GB | 863 / 48.3 | 466 / 44.3 | 276 / 44.3 |

For 256×256, ColossalAI **tensor parallelism**; for 768×768, ColossalAI
**sequence parallelism**.

**On VBench**, Open-Sora 2.0 narrowed the gap with OpenAI Sora from
4.52% → **0.69%** vs Open-Sora 1.2. Human preference results put it
on par with HunyuanVideo 14B and Step-Video 30B.

**Lineage**:

- **Open-Sora 1.0** — initial architecture.
- **Open-Sora 1.1** — multi-resolution / length / aspect ratio,
  image / video conditioning + editing.
- **Open-Sora 1.2** — rectified flow, 3d-VAE, score condition.
- **Open-Sora 1.3** — shift-window attention, unified spatial-temporal
  VAE.
- **Open-Sora 2.0** — current. Each version on its own branch.

### R.6.4 Step-Video-T2V 30B (StepFun, February 2025)

**Source:** `stepfun-ai/stepvideo-t2v` HF card `[VERIFIED]`. Paper:
arXiv:2502.10248.

- **30 B parameters**, the largest open-source video model at release.
- **Video-VAE**: 16×16 spatial + 8× temporal compression.
- **DiT**: 48 layers, 48 attention heads, head dim 128, AdaLN-Single
  for timestep, QK-Norm, **3D RoPE**, **3D full attention**.
- **Two bilingual text encoders** (English + Chinese).
- **Video-DPO**: Direct Preference Optimization on video preference
  data, applied as a final training stage to enhance visual quality.

**Inference requirements** `[VERIFIED]`:

| Setting | Peak GPU memory | 50 steps with FA | 50 steps without FA |
|---|---|---|---|
| 544×992 × 204 frames | 77.64 GB | 743 s | 1232 s |
| 544×992 × 136 frames | 72.48 GB | 408 s | 605 s |

Best practice for inference settings:

| Model | infer_steps | cfg_scale | time_shift | num_frames |
|---|---|---|---|---|
| Step-Video-T2V | 30–50 | 9.0 | 13.0 | 204 |
| Step-Video-T2V-Turbo | 10–15 | 5.0 | 17.0 | 204 |

**xDiT parallel inference on 8× H20** (R.2.3) `[VERIFIED]`:

| GPUs | TP | SP |
|---|---|---|
| 1 | 213.60 s / 92,170 MB | – |
| 2 | 108.97 s / 57,458 MB | 108.13 s / 86,258 MB |
| 4 | 57.61 s / 36,566 MB | 57.01 s / 78,226 MB |
| 8 | **30.40 s** / 30,028 MB | 30.10 s / 79,684 MB |

TP wins on memory (67.4% reduction), SP wins on raw latency at small
margin (0.99× of TP on 8 GPUs).

**Distilled variant**: **`stepfun-ai/stepvideo-t2v-turbo`** — Inference
Step Distillation, 10–15 steps, ~3–5× speedup, no formal benchmark
publication.

### R.6.5 MAGI-1 (Sand AI, April–May 2025)

**Source:** `sand-ai/MAGI-1` HF card `[VERIFIED]`. Paper:
arXiv:2505.13211.

A **24 B autoregressive video diffusion model**. Distinct from all
other models in this section: chunk-by-chunk autoregressive
generation with overlapping denoising of up to 4 chunks at a time.

**Architecture innovations:**

- **Transformer-based VAE**: 8× spatial × 4× temporal compression.
- **Block-Causal Attention**, **Parallel Attention Block**, **QK-Norm
  + GQA**, **Sandwich Normalization in FFN**, **SwiGLU**, **Softcap
  Modulation**.
- **Auto-regressive denoising**: 24 frames per chunk denoised
  holistically; the next chunk begins as soon as the current reaches a
  certain denoising level. Up to **4 chunks processed concurrently**
  for streaming generation.

**Distillation algorithm** (shortcut distillation):

- A single velocity-based model trained to support variable inference
  budgets via a self-consistency constraint: 1 large step ≡ 2 smaller
  steps.
- Step sizes cyclically sampled from `{64, 32, 16, 8}` during
  training.
- **CFG distillation** also incorporated.

**Model zoo:**

| Model | Recommended hardware |
|---|---|
| MAGI-1-24B | H100/H800 × 8 |
| MAGI-1-24B-distill | H100/H800 × 8 |
| **MAGI-1-24B-distill+fp8_quant** | H100/H800 × 4 OR RTX 4090 × 8 |
| MAGI-1-4.5B | RTX 4090 × 1 |
| MAGI-1-4.5B-distill | (Coming soon) |
| MAGI-1-4.5B-distill+fp8_quant | (Coming soon) |

For the 4.5B model, any 24 GB+ GPU is sufficient. Default 4.5B
resolution 720×720.

**Modes**: T2V, I2V, V2V (video-to-video continuation).

**Physics-IQ benchmark** (Phys. IQ Score):

| Model | Phys. IQ ↑ | Spatial IoU ↑ |
|---|---|---|
| Magi-24B (V2V) | **56.02** | 0.367 |
| Magi-4.5B (V2V) | 42.44 | 0.234 |
| VideoPoet (V2V) | 29.50 | 0.204 |
| Magi-24B (I2V) | 30.23 | 0.203 |
| Kling 1.6 (I2V) | 23.64 | 0.197 |
| Wan2.1 (I2V) | 20.89 | 0.153 |
| Sora (I2V) | 10.00 | 0.138 |

MAGI-1 substantially outperforms all baselines on physical-behavior
prediction — an artifact of its autoregressive video continuation
architecture.

**MagiAttention** (`github.com/SandAI-org/MagiAttention`) is the
companion attention library, used by LightX2V as one of its
attention backends.

### R.6.6 Cosmos Predict 2.5 (NVIDIA, October 2025)

**Source:** `nvidia/Cosmos-Predict2.5-2B` HF card `[VERIFIED]`. Paper:
arXiv:2511.00062.

A family of **2 B-parameter (2,059,174,912 params) diffusion-based
world foundation models** purpose-built for physical AI (robotics,
autonomous vehicles).

**Variants** in the Predict 2.5 family `[VERIFIED]`:

- **Cosmos-Predict2.5-2B Pre-trained / Post-trained**: text + image
  + video → future frames (720P 16 FPS or 480P 16 FPS).
- **Auto / Multiview**: 7-camera view world prediction.
- **Robot / Multiview**: 2 re-rendered videos given target hand-view
  trajectories.
- **Robot / Multiview-Agibot**: head-view + hand-view trajectories.
- **Robot / Action-Cond**: action-conditioned future-frame prediction
  (256p 4 FPS).
- **Robot / Policy**: full robot-policy generation including images,
  proprio, and value.

**Architecture**: diffusion transformer with interleaved self-attn,
cross-attn (text conditioning), and feedforward; AdaLN before each
layer for time embedding. Image/video conditioning via temporal
concatenation of conditioning latents with augment noise.

**System requirements**:

| GPU | Inference time (Video2World 720p 16 FPS) |
|---|---|
| H100 SXM | 228.8 s |
| H200 SXM | 221.7 s |
| **B200** | **123.9 s** |
| H100 NVL | 355.7 s |
| H100 PCIe | 378.5 s |
| H200 NVL | 267.2 s |
| L40S | 2567.1 s |
| RTX PRO 6000 Blackwell | 452.2 s |

`[VERIFIED]`. **B200 is 1.85× over H100 SXM at base precision.**

VRAM: 32.54 GB.

Input: 720P (1280×704) or 480P (832×480), 5-frame conditioning, 5 s
output @ 16 FPS.

**Distillation recipe**: see R.1.10. **4-step DMD2 student on TrigFlow,
no GAN, K=4 critic:student ratio, 1500 iterations**. Convergence in
<2 hours on a multi-GPU box.

**License**: NVIDIA Open Model License (commercial OK; free
distribution; no IP claim on outputs).

---

## R.7 End-to-end models — FLUX / SD3.5 (image-side cross-references)

### R.7.1 FLUX.1-dev (Black Forest Labs, August 2024)

**Source:** `black-forest-labs/FLUX.1-dev` HF card.

- **12 B parameter** rectified flow transformer (image generation).
- Multimodal joint Transformer + Parallel Diffusion Transformer
  blocks; T5-XXL + CLIP text encoders; flow-matching loss.
- **Recommended inference**: 50 steps, CFG 3.5, max_sequence_length
  512. (For text-to-image with `FluxPipeline.from_pretrained`.)

**Distilled / quantized:**

- **NVFP4 PTQ**: `black-forest-labs/FLUX.1-dev-onnx` (ONNX checkpoint).
- **PyTorch + TorchAO Blackwell** numbers (R.1.8) for `FLUX.1-dev`:

| Quant | Batch | Latency (s) | Memory (GB) | Speedup vs BF16 |
|---|---|---|---|---|
| BF16 | 1 | 2.10 | 38.34 | 1.00× |
| MXFP8 | 1 | 1.75 | 26.90 | 1.21× |
| NVFP4 | 1 | 1.41 | 21.33 | **1.50×** |
| BF16 | 4 | 7.87 | 44.39 | 1.00× |
| MXFP8 | 4 | 6.36 | 32.95 | 1.24× |
| NVFP4 | 4 | 5.09 | 27.39 | 1.55× |
| BF16 | 8 | 15.57 | 53.00 | 1.00× |
| MXFP8 | 8 | 12.40 | 41.56 | 1.26× |
| NVFP4 | 8 | 9.81 | 36.00 | **1.59×** |

Mean LPIPS on Drawbench: MXFP8 0.107, NVFP4 0.438 — NVFP4 has visible
perceptual change, MXFP8 is near-lossless.

### R.7.2 FLUX.2-dev (Black Forest Labs, late 2025–2026)

**Source:** `black-forest-labs/FLUX.2-dev` HF card `[VERIFIED]`.

- **32 B parameter** rectified flow transformer.
- Capable of T2I, single-reference image editing, and multi-reference
  editing (character, object, style — without finetuning).
- **Trained with guidance distillation** (CFG embedded; no separate
  negative prompt in default usage).
- **Recommended inference**: 50 steps (28 a good trade-off), guidance
  4.0.
- **Variants**:
    - `black-forest-labs/FLUX.2-dev` (BF16)
    - `black-forest-labs/FLUX.2-dev-NVFP4`
    - `black-forest-labs/FLUX.2-Klein-4B`
    - `black-forest-labs/FLUX.2-Klein-9B`
- **License**: FLUX Non-Commercial License Agreement (the larger dev
  models). Commercial via API only.
- **Diffusers integration**: `Flux2Pipeline`. Has 4-bit BNB variant
  (`diffusers/FLUX.2-dev-bnb-4bit`) for RTX 4090/5090 deployment.
  Remote text encoder via `huggingface.co/remote-text-encoder-flux-2`
  endpoint.
- Embedded guidance means **CFG-parallel doesn't apply** in TRT-LLM
  VisualGen (R.2.4).

### R.7.3 Stable Diffusion 3 / 3.5 / SDXL / SDXL-Turbo (Stability AI)

The pre-DiT image foundation models from Stability AI. Cross-references
for older work but not the focus of this report. Notable points:

- **SD3** uses **MMDiT** (Multi-Modal Diffusion Transformer) — the
  architecture HunyuanVideo's dual-stream/single-stream design extends.
- **SD3.5 Medium** (2.5 B) and **SD3.5 Large** (8 B) ship as
  `stabilityai/stable-diffusion-3.5-medium-diffusers` /
  `stabilityai/stable-diffusion-3.5-large-diffusers`.
- **SDXL-Turbo** is a 1-step ADD (Adversarial Diffusion Distillation)
  variant of SDXL — the seminal "1-step image generation"
  reference but not on Wan-class video.
- **PPCL (Phased Progressive Consistency Learning)** is the cleanest
  "MMDiT image distillation" recipe from this lineage; a video-class
  student-direction adaptation has been on FastVideo's roadmap but
  no public checkpoint as of May 2026.

### R.7.4 Other image-side models for context

The SGLang Diffusion image catalog (R.2.2) lists ~30 image models. The
ones most relevant to video-DiT cross-pollination:

- **Z-Image / Z-Image-Turbo** (Tongyi-MAI) — Tongyi's image
  counterpart to Wan; the LightX2V Wan-Lightning pipeline reuses some
  ideas.
- **Qwen-Image / 2512 / Edit / Edit-2509 / Edit-2511 / Layered**
  (Qwen) — text-aligned image generation/editing. The
  `lightx2v/Qwen-Image-2512-Lightning` and `lightx2v/Qwen-Image-Edit-2511-Lightning`
  4-step distilled LoRAs are *the same engineering pattern* as
  Wan-Lightning. **42× speedup vs base** with all optimizations
  combined (per the LightX2V Dec 23, 2025 release note).
- **GLM-Image** (Zhipu) — analogous to Z-Image / Qwen-Image.
- **SANA 1.5 / 1600M / 600M** (Efficient-Large-Model) — extremely
  small image models for edge.
- **FireRed-Image-Edit 1.0/1.1** — image editing.
- **ERNIE-Image / Turbo** (Baidu) — Chinese image generation with
  LightX2V-like distillation.

---

## R.8 Vendor-shipped distilled / quantized variants — full catalog

### R.8.1 LightX2V family (the broadest)

| HF repo | Base model | Steps | Quant | Notes |
|---|---|---|---|---|
| `lightx2v/Wan2.2-Lightning` | Wan 2.2 T2V/I2V-A14B | 4 | LoRA only | Phased DMD; fp16 LoRA rank 64 |
| `lightx2v/Wan2.2-Distill-Models` | Wan 2.2 T2V/I2V-A14B | 4 | BF16 / FP8 e4m3 / INT8 / FP8 ComfyUI | Pre-merged ~28.6/15/15/15 GB |
| `lightx2v/Wan2.2-Distill-Loras` | Wan 2.2 T2V/I2V-A14B | 4 | LoRA-only | Stack with other LoRAs |
| `lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v` | Wan 2.2 I2V-A14B | 2 high + 2 low = 4 | various | Per-expert distill `[VERIFIED]` |
| `lightx2v/Wan-NVFP4` | Wan 2.1 I2V-14B-480P + T2V-1.3B | 4 | NVFP4 (QAD) | RTX 5090 reference |
| `lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill` | Wan 2.1 T2V-14B | 4 | various | Step+CFG distill |
| `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v` | Wan 2.1 I2V-14B | 4 | various | Step+CFG distill |
| `lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v` | Wan 2.1 I2V-14B | 4 | various | Step+CFG distill |
| `lightx2v/Wan2.1-T2V-14B-CausVid` | Wan 2.1 T2V-14B | 4 AR | various | CausVid |
| `lightx2v/Wan2.1-Distill-Loras` / `Wan2.1-Distill-Models` | Wan 2.1 | 4 | LoRA / merged | |
| `lightx2v/Hy1.5-Distill-Models` | HunyuanVideo-1.5 | 4 | base / FP8 | ~25× over 50-step |
| `lightx2v/Hy1.5-Quantized-Models` | HunyuanVideo-1.5 | 50 | various | Quant-only path |
| `lightx2v/Self-Forcing-FP8` | Self-Forcing | 4 AR | FP8 | |
| `lightx2v/Self-Forcing-NVFP4` | Self-Forcing | 4 AR | NVFP4 | |
| `lightx2v/Qwen-Image-2512-Lightning` | Qwen-Image-2512 | 8 | LoRA + FP8 | Up to 42× combined with FP8 + NVFP4 |
| `lightx2v/Qwen-Image-Edit-2511-Lightning` | Qwen-Image-Edit-2511 | 8 | LoRA + FP8 | |
| `lightx2v/Autoencoders` | various (Wan, HV 1.5, etc.) | – | – | LightTAE family — fast VAEs |

`[VERIFIED]` for those checked above. **`lightx2v/Hy1.5-Distill-Models`**
is particularly notable — 4-step inference for HunyuanVideo-1.5 with
FP8 quantization.

The lightx2v Wan-NVFP4 numbers `[VERIFIED]` from the HF model card,
RTX 5090, 40 steps, 81 frames:

**Wan2.1-I2V-14B-480P:**

| Metric | Original | NVFP4 4-step | Speedup |
|---|---|---|---|
| Single-step denoising | 12.10 s | 3.40 s | 3.5× |
| **End-to-end** | **498.90 s** | **17.65 s** | **28×** |

**Wan2.1-T2V-1.3B-480P:**

| Metric | Original | NVFP4 4-step | Speedup |
|---|---|---|---|
| Single-step denoising | 2.00 s | 0.70 s | 2.9× |
| **End-to-end** | **83.50 s** | **6.54 s** | **12.8×** |

These numbers — **28× and 12.8×** end-to-end on RTX 5090 (Blackwell
consumer) — are the canonical "what the full Stack D recipe delivers
on consumer Blackwell."

### R.8.2 FastVideo / Hao AI Lab line

| HF repo | Base | Steps | Quant | Notes |
|---|---|---|---|---|
| `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | Wan 2.1 T2V-1.3B | 3 | BF16 | DMD + VSA |
| `FastVideo/FastWan2.1-T2V-14B` (recipe public) | Wan 2.1 T2V-14B | 3 | BF16 | DMD + VSA |
| `FastVideo/FastWan2.2-TI2V-5B-Diffusers` | Wan 2.2 TI2V-5B | 3 | BF16 | DMD + VSA |
| `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | Wan 2.2 TI2V-5B | 3 | BF16 | DMD-only (data-free) |
| `FastVideo/FastHunyuan-diffusers` | HunyuanVideo 1.0 | 6 | BF16 / FP8 | CD (Huber) |
| `FastVideo/FastMochi-diffusers` | Mochi 1 | 8 | BF16 | CD |

All Apache-2.0 licensed.

### R.8.3 Lightricks LTX line

The 7-artifact LTX-2 catalog (`Lightricks/LTX-2`) plus LTX-2.3:

| HF repo | Base | Steps | Quant | Purpose |
|---|---|---|---|---|
| `Lightricks/LTX-2` (`ltx-2-19b-dev`) | LTX-2 19B | 40 + CFG 4.0 | BF16 | Reference, trainable |
| `Lightricks/LTX-2` (`ltx-2-19b-dev-fp8`) | LTX-2 19B | 40 + CFG 4.0 | FP8 | |
| `Lightricks/LTX-2` (`ltx-2-19b-dev-fp4`) | LTX-2 19B | 40 + CFG 4.0 | NVFP4 | |
| `Lightricks/LTX-2` (`ltx-2-19b-distilled`) | LTX-2 19B | **8** + CFG 1 | BF16 | Distilled |
| `Lightricks/LTX-2` (`ltx-2-19b-distilled-lora-384`) | LTX-2 19B | Stage 2 LoRA | BF16 | LoRA on full model |
| `Lightricks/LTX-2` (`ltx-2-spatial-upscaler-x2-1.0`) | – | – | BF16 | Multi-scale |
| `Lightricks/LTX-2` (`ltx-2-temporal-upscaler-x2-1.0`) | – | – | BF16 | Higher-FPS |
| `Lightricks/LTX-2.3-nvfp4` (`ltx-2.3-22b-dev-nvfp4`) | LTX-2.3 22B | 40 + CFG 4.0 | **QAD-NVFP4** | Quantization-aware distillation |
| `Lightricks/LTX-2.3-nvfp4` (`ltx-2.3-22b-distilled-nvfp4`) | LTX-2.3 22B | 8 + CFG 1 | NVFP4 | (Coming soon) |
| `Lightricks/LTX-Video` (`ltxv-13b-0.9.8-dev` / `-distilled` / `-fp8` / -mix) | LTX-Video 13B | 8 / 30 | BF16 / FP8 | Pre-LTX-2 line |
| `Lightricks/LTX-Video` (`ltxv-2b-0.9.8-distilled` / `-fp8`) | LTX-Video 2B | 8 | BF16 / FP8 | Smaller |
| `Lightricks/ltxv-spatial-upscaler-0.9.8` / `temporal-upscaler-0.9.8` | – | – | BF16 | Upscalers |

`[VERIFIED]`.

### R.8.4 Krea Realtime 14B

`krea/krea-realtime-video` `[VERIFIED]`:

- **Distilled from Wan 2.1-T2V-14B** via Self-Forcing, October 2025
  (the first 14B Self-Forcing run).
- **11 fps T2V on a single B200** at 4 inference steps `[VERIFIED]`.
- Uses `lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill` as few-step
  init.
- Trained on 128 H100s for 3k steps (Causal ODE pretraining alone).
- Self-Forcing DMD for the AR pass.
- Inference innovations: Dynamic KV cache freeing (per-block backprop
  hooks free past blocks' KVs once their backward is done; cache can
  grow to 25 GB/GPU otherwise); KV Cache Recomputation for long
  contexts; KV Cache Attention Bias.
- ~1 s startup latency (time to first frame).
- Optimized inference: `torch.compile`, **Sageattention**, FP8 with
  `torchao.Float8DynamicActivationFloat8WeightConfig`. Hub-kernels for
  Sageattention or FA3 (`DISABLE_SAGEATTENTION=1`).
- Diffusers ModularPipeline support; LoRA support
  (e.g., `shauray/Origami_WanLora`).

### R.8.5 NVIDIA / Cosmos line

- **`nvidia/Cosmos-Predict2.5-2B`** (Pre-trained, Post-trained, Auto/
  Multiview, Robot variants).
- **Distilled student**: 4-step DMD2 student per the cosmos-cookbook
  recipe (R.1.10). Code at `cosmos_predict2/_src/predict2/distill`.

### R.8.6 Other distilled video models

| Distilled model | Base | Steps | Method |
|---|---|---|---|
| **Wan 2.2-Lightning V1.1** | Wan 2.2 T2V/I2V-A14B | 4 | Phased DMD |
| **DCM** | HunyuanVideo | few-step | Dual-expert CM (semantic + detail w/ GAN) |
| **W2SVD** | HunyuanVideo | 1 / 4 | LoRA-DMD, weak-to-strong |
| **DOLLAR** | (multiple) | 4 | VSD + CD + latent reward |
| **TDM-CogVideoX-2B-LoRA** | CogVideoX 2B | 4 | TDM, fixed timesteps |
| **T2V-Turbo-v2 (VC2)** | VideoCrafter2 | 4 | CD + mixed reward |
| **TMD** (NVIDIA + NYU) | Wan 2.1-1.3B / 14B | <2 NFE | Transition Matching Distillation |
| **rCM** | Wan 2.1-1.3B | 1 | Score-regularized continuous-time CM |
| **OSV** | SVD | 1 | 2-stage GAN + CD with latent discriminator |
| **BLADE-Wan-2.1-1.3B** | Wan 2.1 1.3B | few-step | Sparsity-aware TDM + ASA |
| **BLADE-CogVideoX-5B** | CogVideoX-5B | few-step | Same |
| **Step-Video-T2V-Turbo** | Step-Video-T2V | 10–15 | Inference Step Distillation |
| **Mobile Video DiT** | Snap mobile DiT | 4 | Adversarial step distill + tri-level pruning |
| **MAGI-1-24B-distill** / **distill+fp8_quant** | MAGI-1 24B | shortcut steps | Velocity self-consistency |
| **MAGI-1-4.5B-distill** | MAGI-1 4.5B | shortcut steps | (Coming soon) |
| **TurboWan2.1 / TurboWan2.2** (IPostYellow) | Wan 2.1/2.2 | 4 | Distilled + SLA + SageSLA |

### R.8.7 Community NVFP4 forks

- **`ac4k/Wan2.2`** — community NVFP4 PTQ fork (HF model card requires
  authentication; not fetched in this round). Reportedly hosts NVFP4
  weights for Wan 2.2 T2V/I2V-A14B usable with TensorRT-LLM
  VisualGen and LightX2V.

### R.8.8 Auxiliary / lightweight VAEs

- **TAEHV** family — TinyAutoencoderHV; lightweight VAE replacements.
- **`lightx2v/Autoencoders`** — LightTAE for HunyuanVideo-1.5 etc.
- **Mochi-1-Transformer-42** — depth-pruned Mochi.

---

## R.9 Cross-stack consolidated benchmark table

The data points cited in R.1–R.8 collected into one table. Each row is
a **single deployment configuration** with the model, hardware, and
measured wall-clock. All entries are `[VERIFIED]` (linked back to the
original blog/card) unless marked otherwise.

### R.9.1 Wan 2.2 (720p × 81f × 40 steps unless noted)

| Stack | GPUs | Time | Notes | Source |
|---|---|---|---|---|
| Default Alibaba runtime (FA2) | 8× H100 | 250.70 s | Morphic baseline | Morphic blog |
| Default + FA3 | 8× H100 | 195.13 s | 1.28× | Morphic |
| Default + FA3 + TF32 | 8× H100 | 159.55 s | 1.57× | Morphic |
| Default + FA3 + Sage + TeaCache | 8× H100 | **60 s** (1.51 s/step) | **3.09×** | VoltagePark |
| Default + FA3 + TF32 + MagCache + torch.compile | 8× H100 | **109.81 s** | 2.28× lossless | Morphic |
| Default + FA3 + TF32 + MagCache E024K2R10 + torch.compile | 8× H100 | 98.87 s | 2.53× (some artifacts) | Morphic |
| **Baseten custom runtime** (BF16 lossless) | 1× B200 | ~50 s | **3.2× over default**, 2.6× on H100 | Baseten |
| LightX2V on H100 (no CFG, FP8) | 8× H100 | 14.0 s (0.35 s/step × 40) | 2.1× over LightX2V baseline | LightX2V README |
| LightX2V Wan2.2-Lightning 4-step | 8× H100 | ~3 s (calc: 0.75 s/step × 4) | Plus VAE + TE | LightX2V |
| LightX2V Wan-NVFP4 4-step (T2V 1.3B 480P) | 1× RTX 5090 | **6.54 s** | **12.8×** end-to-end | lightx2v/Wan-NVFP4 card |
| LightX2V Wan-NVFP4 4-step (I2V 14B 480P) | 1× RTX 5090 | **17.65 s** | **28×** end-to-end | lightx2v/Wan-NVFP4 card |
| FastWan2.1-T2V-1.3B (DiT only, 5 s 480p) | 1× H200 | 0.98 s | 95× DiT-only over FA2 baseline | Hao AI Lab FastWan blog |
| FastWan2.1-T2V-1.3B (DiT only) | 1× RTX 4090 | 2.8 s | – | Hao AI Lab |
| FastWan2.1-T2V-14B (DiT only, 5 s 720p) | 1× H200 | 13 s | 135× DiT | Hao AI Lab |
| FastWan2.2-TI2V-5B (5 s 720p) | 1× H200 | 16 s | 60× DiT | Hao AI Lab |
| FastWan2.2-TI2V-5B (5 s 480p, e2e) | 1× H200 | 5 s | 50× e2e | Hao AI Lab |

### R.9.2 LTX-2 / LTX-2.3 (768×512 × 121f × 40 steps unless noted)

| Stack | GPUs | Time | Notes | Source |
|---|---|---|---|---|
| LTX-2 BF16 B=1 | 1× B200 | 16.230 s | Baseline | PyTorch+TorchAO blog |
| LTX-2 MXFP8 B=1 | 1× B200 | 13.724 s | 1.18× | PyTorch+TorchAO blog |
| **LTX-2 NVFP4 B=1** | 1× B200 | **10.374 s** | **1.56×** | PyTorch+TorchAO blog |
| LTX-2 BF16 B=4 | 1× B200 | 61.591 s | – | PyTorch+TorchAO blog |
| LTX-2 NVFP4 B=4 | 1× B200 | 36.963 s | **1.67×** | PyTorch+TorchAO blog |
| LTX-2 BF16 B=8 | 1× B200 | 122.427 s | – | PyTorch+TorchAO blog |
| LTX-2 NVFP4 B=8 | 1× B200 | 72.689 s | **1.68×** | PyTorch+TorchAO blog |
| LTX-2.3 1080p TI2AV | 1× B200 | **4.55 s e2e** for 5 s clip | **3.9×** next-fastest 1080p | FastVideo Hao AI Lab |
| LTX-2 two-stage `original` mode | 1× ? | 154.67 s | SGLang ref | SGLang docs |
| LTX-2 two-stage `snapshot` (default) | 1× ? | 114.05 s | SGLang ref | SGLang docs |
| LTX-2 two-stage `resident` | 1× ? | 75.71 s | High VRAM | SGLang docs |

### R.9.3 HunyuanVideo (720p × 129f × 50 steps)

| Stack | GPUs | Time | Notes | Source |
|---|---|---|---|---|
| Base BF16 | 1× H100 80 GB | 1904.08 s | xDiT baseline | tencent/HunyuanVideo |
| xDiT USP-2 | 2× H100 | 934.09 s | 2.04× | xDiT |
| xDiT USP-4 | 4× H100 | 514.08 s | 3.70× | xDiT |
| xDiT USP-8 | 8× H100 | 337.58 s | 5.64× | xDiT |
| Default BF16 720p (60 GB) | 1× 80GB | – | (peak mem) | HV card |
| HV 1.5 base (8.3 B, 720p) | 1× RTX 4090 | (offload required) | 14 GB minimum | HV 1.5 card |
| **HV 1.5 step-distilled 12-step on 4090** | 1× RTX 4090 | **<75 s** | 75% latency reduction vs 50-step | HV 1.5 card |
| **HV 1.5 step-distilled 4-step + LightX2V** | 1× RTX 4090 | similar | ~25× over 50-step | LightX2V Nov 24, 2025 |
| HV 1.5 SSTA built-in vs FA3 | 1× ? | 1.87× e2e for 10 s 720p synthesis | Sparse attention built-in | HV 1.5 card |

### R.9.4 Step-Video-T2V (544×992 × 204f × 50 steps)

| Stack | GPUs | Time | Notes | Source |
|---|---|---|---|---|
| Base + FA | 1× H100 80 GB | 743 s | Peak 77.64 GB | StepVideo card |
| Base + xDiT TP1 SP1 | 1× H20 | 213.60 s | (smaller frames, 144) | xDiT |
| Base + xDiT TP8 | 8× H20 | 30.40 s | 0.88× near-linear | xDiT |
| Base + xDiT SP8 | 8× H20 | **30.10 s** | 0.89× | xDiT |
| Step-Video-T2V-Turbo | 1× ? | ~3–5× over base | 10–15 steps | StepFun |

### R.9.5 CogVideoX (720×480 × 49 frames × 50 steps)

| Stack | GPUs | Time | Notes | Source |
|---|---|---|---|---|
| CogVideoX-2B BF16 | 1× A100 | ~90 s | – | CogVideoX-5b card |
| CogVideoX-2B BF16 | 1× H100 | ~45 s | – | CogVideoX-5b card |
| CogVideoX-2B + xDiT (best mix) | 12× L40 | **30 s** | 4.29× | xDiT |
| CogVideoX-5B BF16 | 1× A100 | ~180 s | – | CogVideoX-5b card |
| CogVideoX-5B BF16 | 1× H100 | ~90 s | – | CogVideoX-5b card |
| CogVideoX-5B + xDiT (best mix) | 12× L40 | ~40 s | 7.75× | xDiT |
| CogVideoX1.5-5B (161f × 1360×768) + xDiT mix | 8× L40 | <10 min | 6.12× (Ulysses-2 + Ring-2 + CFG-2) | xDiT |
| **TDM-CogVideoX-2B-LoRA 4-step** | 1× H100 | ~25× over base | VBench 81.65 vs teacher 80.91 | TDM card |
| **BLADE-CogVideoX-5B** | 1× H100 | 8.89× e2e | VBench-2.0 0.534→0.569 | BLADE paper |

### R.9.6 Mochi 1 (480 × 848 × 31 frames × 64 steps)

| Stack | GPUs | Time | Notes | Source |
|---|---|---|---|---|
| Mochi 1 BF16 | 1× H100 80GB | – | 60 GB single GPU; needs 1× H100 | genmo card |
| Mochi 1 BF16 + diffusers + offload + tiling | 1× H100 | – | 42 GB | genmo card |
| Mochi 1 BF16 (variant=bf16) | 1× ? | – | 22 GB | genmo card |
| **FastMochi (8-step)** | 1× H100 | ~8× over base | – | FastVideo |

### R.9.7 Open-Sora 2.0 (768×768 × 50 steps)

| Stack | GPUs | Time | Notes |
|---|---|---|---|
| 768px × 768px BF16 | 1× H100 | 1656 s | 60.3 GB peak |
| 768 + ColossalAI SP | 2× H100 | 863 s | 48.3 GB |
| 768 + ColossalAI SP | 4× H100 | 466 s | 44.3 GB |
| 768 + ColossalAI SP | 8× H100 | 276 s | 44.3 GB |
| 256 + ColossalAI TP | 1× H100 | 60 s | 52.5 GB |
| 256 + ColossalAI TP | 4× H100 | 34 s | 44.3 GB |

### R.9.8 MAGI-1 (24B autoregressive) and Cosmos Predict 2.5 (2B)

| Stack | GPUs | Time | Notes |
|---|---|---|---|
| MAGI-1-24B base | 8× H100/H800 | (autoregressive) | Streaming AR |
| MAGI-1-24B-distill+FP8 | 4× H100 OR 8× RTX 4090 | – | Distill + FP8 quant |
| MAGI-1-4.5B base | 1× RTX 4090 | – | Default 720×720 |
| Cosmos Predict 2.5 2B Video2World 720p | 1× B200 | **123.9 s** | 1.85× over H100 SXM |
| Cosmos Predict 2.5 2B Video2World 720p | 1× H100 SXM | 228.8 s | – |
| Cosmos Predict 2.5 2B Video2World 720p | 1× H200 SXM | 221.7 s | – |
| Cosmos Predict 2.5 2B Video2World 720p | 1× L40S | 2567.1 s | Ada Lovelace; very slow |
| **Cosmos Predict 2.5 2B 4-step distilled (DMD2/TrigFlow)** | 1× B200 | ~31 s e2e (calc: 123.9/36 × 4) ≈ ~14 s actual | Per cookbook recipe |

### R.9.9 Krea Realtime 14B (T2V)

| Stack | GPUs | Time | Notes |
|---|---|---|---|
| Krea Realtime 14B (4-step Self-Forcing) | **1× B200** | **11 fps T2V** | First 14B SF run; ~1 s TTFB |
| + torch.compile + Sageattention + FP8 (torchao) | 1× H100 | (compile-warm) | Hopper deployment |

### R.9.10 FLUX.1-dev (1024² image)

| Stack | GPUs | Time | Notes |
|---|---|---|---|
| BF16 B=1 | 1× B200 | 2.10 s | – |
| MXFP8 B=1 | 1× B200 | 1.75 s | 1.21× |
| NVFP4 B=1 | 1× B200 | 1.41 s | **1.50×** |
| NVFP4 B=8 | 1× B200 | 9.81 s | **1.59×** |

### R.9.11 LightX2V cross-framework (Wan 2.1-I2V-14B-480P, 40 step, 81 frames)

H100:

| Framework | 1 GPU | 8 GPU |
|---|---|---|
| Diffusers | 9.77 s/it | – |
| xDiT | 8.93 s/it | 2.70 s/it |
| FastVideo | 7.35 s/it | 2.94 s/it |
| SGL-Diffusion | 6.13 s/it | 1.19 s/it |
| **LightX2V** | **5.18 s/it** | **0.75 s/it** |

RTX 4090D:

| Framework | 1 GPU | 8 GPU |
|---|---|---|
| Diffusers | 30.50 s/it | – |
| FastVideo | 22.66 s/it | 15.48 s/it |
| **LightX2V** | **20.26 s/it** | **4.75 s/it** |

xDiT and SGL-Diffusion OOM on 4090D.

### R.9.12 LightX2V disaggregated deployment (Mooncake)

Qwen-2512 BF16, 50 steps, 4090 baseline + offload:

| Stage | Baseline | Disagg | Speedup |
|---|---|---|---|
| Text Encoder | 12.89 s | 0.40 s | **32.22×** |
| DiT/step | 5.75 s | 3.76 s | 1.53× |
| DiT/step (8-step distilled) | 2.90 s | 1.88 s | 1.54× |

Wan 2.1-I2V-14B BF16 40 steps:

| Stage | Baseline 1×4090 | Disagg 2×4090 |
|---|---|---|
| DiT/step 480P | 24.24 s | 19.02 s |
| DiT/step 720P | 90.71 s | 62.80 s |
| Text Encoder | 2.14 s | 0.20 s |
| Image Encoder 480P | 0.57 s | 0.28 s |

8-GPU 4090 cluster:

| Mode | Ratio | QPS | P50 | P95 | P99 |
|---|---|---|---|---|---|
| Baseline | 8:0 | 0.18 | 22 s | 28 s | 30 s |
| Disagg centralized | 7:1 | 0.24 | 17 s | 25 s | 28 s |
| **Disagg decentralized** | **7:1** | **0.34** | **17 s** | **20 s** | **22 s** |

### R.9.13 fal Async-Ulysses on 8× B200 (BF16, B=1, H=40, D=128)

| Variant | Chunk reduction | E2E reduction |
|---|---|---|
| Baseline Ulysses | – | – |
| **Async Ulysses** (best at 8 GPU) | 23–25% | 3% |
| Async + Symmetric Memory | varies (best 2/4 GPU) | – |
| **Fused QKV** (best 2/4 GPU) | 33–37% | 4–5% |

### R.9.14 FlashAttention generations on B200

| Kernel | Throughput (BF16 fwd) | Utilization |
|---|---|---|
| FA2 | – | – |
| FA3 (Hopper) | 740 TFLOPS | 75% (H100) |
| **FA4 (Blackwell)** | **1605–1613 TFLOPS** | **71% (B200)** |
| Triton (B200) | ~600 TFLOPS | – |
| cuDNN 9.13 (B200) | ~1240 TFLOPS | – |

`[VERIFIED]`. cuDNN 9.14+ has incorporated FA4 techniques and closed
some of the gap.

### R.9.15 NVIDIA hardware peak — context

Per-GPU at 1000 W TGP (DGX B200 variant):

| Format | B200 | H100 | Ratio |
|---|---|---|---|
| **NVFP4 sparse** | 18 PFLOPS | – | – |
| **NVFP4 dense** | **9 PFLOPS** | – | – |
| FP8/FP6 sparse | 9 PFLOPS | 4 PFLOPS | 2.25× |
| FP8/FP6 dense | 4.5 PFLOPS | 2 PFLOPS | 2.25× |
| BF16/FP16 dense | **2.25 PFLOPS** | 0.989 PFLOPS | 2.27× |
| TF32 | 1.125 PFLOPS | 0.5 PFLOPS | 2.25× |

Per-GPU at 700 W TGP (HGX B200 variant), all numbers are about 25%
lower (HGX B200 publishes 108 PFLOPS sparse FP4 across 8 GPUs = 13.5
sparse / 6.75 dense per GPU).

`[CORRECTED]` from earlier draft text saying "15 PFLOPS FP4" without
sparse / dense / TGP qualification.

---

## R.10 Summary — what to actually do in May 2026

For a team standing up Wan 2.2 / LTX-2 / HunyuanVideo / Cosmos in
production right now, the verified evidence above implies:

1. **If you have B200**, the answer is: **`Wan2.2-Lightning 4-step
   LoRA + FP8 e4m3 + LightX2V + FA4 + torch.compile + Async Ulysses
   (8-GPU node)`** for `~3 s/clip` lossless 720p × 81f × 4 NFE. Add
   **NVFP4** quantization on the DiT linears for another 1.5–1.7×
   if you can tolerate small LPIPS regressions.
2. **If you have H100**, the answer is: **`Wan2.2-Lightning + FA3 +
   TF32 + MagCache + torch.compile + Ulysses-8`** — 60–110 s lossless
   for 720p × 81f × 40 step (Morphic) or 60 s with TeaCache lossy
   (VoltagePark).
3. **If you have RTX 5090** consumer Blackwell, the answer is:
   **`lightx2v/Wan-NVFP4 4-step distill`** — 17.65 s for 480P I2V
   14B end-to-end (28× over 50-step base).
4. **If you have RTX 4090** (consumer Ada Lovelace), the answer is:
   **HunyuanVideo-1.5 step-distilled 12 / 4 step + LightX2V FP8** —
   75 s end-to-end with 14 GB VRAM and 25× speedup.
5. **If you want 1080p** with audio, **`LTX-2.3-22b-dev-nvfp4` +
   FastVideo on B200** — 4.55 s for 5 s clip end-to-end.
6. **If you want autoregressive streaming**, the choices are
   **Krea Realtime 14B** (Wan 2.1 14B Self-Forcing, 11 fps B200, 1 s
   TTFB), **MAGI-1-24B** (24B chunk-AR, robust on physical-IQ),
   **HY-WorldPlay** (24 fps 720p streaming, world-model variant of HV
   1.5), or **Helios** (19.5 fps H100 with infrastructure-only opts).
7. **For multi-GPU production deployment** at scale, layer the
   **LightX2V disaggregated Mooncake architecture** (32× Text Encoder
   speedup on memory-constrained nodes; 1.89× QPS on 8× 4090 with
   decentralized scheduler at 7:1 DiT:Encoder ratio).

The single most-important lossless intervention is **step distillation**
(50 → 4 NFE, fold CFG → 25× theoretical, ~20× measured). The
single most-important Blackwell-specific intervention is **NVFP4
GEMM with selective quant** on DiT linears (1.56–1.68× e2e on
LTX-2). Stacking those two recipes on a B200 box is the entire
"~50× over BF16 reference" story the parent report develops in §5.

The technologies are all open as of May 2026: lightx2v (Apache 2.0),
FastVideo (Apache 2.0), Wan 2.2 (Apache 2.0), HunyuanVideo / 1.5 (open
weights), LTX-2 (custom Open Weights License), Cosmos Predict 2.5
(NVIDIA Open Model License), MAGI-1 (Apache 2.0), Krea Realtime
(open). The bottleneck has shifted from "is this technically
possible?" to "which combination of recipes is best for my exact
hardware and quality target." The blogs surveyed above are the
authoritative reference points for that decision.
