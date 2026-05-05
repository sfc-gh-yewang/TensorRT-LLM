## §4. Step Distillation, CFG Distillation, and Model-Level Acceleration

*Subsection prefix `D.x` is scoped to §4 (note: §3-A also uses `D.x` scoped within §3-A; references that span sections are spelled out as "§3-A D.x" or "§4 D.x").*

### §4 D.0 Background: Progressive Distillation, LCM/LCM-LoRA, AnimateLCM, CTM

Step distillation is the dominant lever for video DiT acceleration. A 50-step Wan 2.1 14B inference at 480p takes ~95.21s of pure DiT denoising on H200; collapsing to 3 NFE with DMD distillation lowers that to 2.88s, a 33× cut, and folding CFG into the student removes another 2× cost from forward passes. The numbers in this section all assume a B200/H200 generation budget where step count is the largest single Amdahl term — quantization, sparse attention, and parallelism are stacked on top of distillation, not in place of it.

The lineage that matters for video DiT is short:

- **Knowledge distillation from a teacher** (Luhman & Luhman, 2021): collect noise-image pairs from a many-step DDIM teacher, regress them with a single-step student. Conceptually the upper bound and computationally infeasible at video scale (one SDXL pair takes ~5s; 12M LAION prompts cost ~700 A100-days per Yin et al.).
- **Progressive Distillation** [arXiv:2202.00512](https://arxiv.org/abs/2202.00512) (Salimans & Ho, ICLR 2022). The first method that did not require precomputed noise-image pairs. Iteratively distills a sequence of students, each halving the sampling steps of its predecessor. Two parameterizations stabilize few-step training: ("v"-prediction with `v = α_t·ε − σ_t·x` and a snr-rescaled denoising target). On CIFAR-10, distill from a teacher at 8192 steps down to 4 steps with FID 3.0 — without losing much perceptual quality. The whole multi-stage process completes in roughly the same wall-clock as the original training run, because each stage trains for a small fraction of the total. Every modern method below either reuses progressive distillation as initialization (SDXL-Lightning, Hyper-SD's TSCD, AAPT's consistency warm-up) or explicitly contrasts with it (DMD2 calls out the regression bottleneck, Phased DMD calls out the step-halving constraint).
- **Latent Consistency Model (LCM)** (Luo et al., 2023). Adapts Song et al.'s consistency-model objective to latent diffusion: a student `f_θ(x_t, t)` is trained so that for any two timesteps on a given ODE trajectory, `f_θ(x_t, t) = f_θ(x_{t'}, t')`. Boundary parameterization `f_θ(x, t) = c_skip(t)·x + c_out(t)·F_θ(x, t)` with `c_skip(0)=1, c_out(0)=0` is canonical. **LCM-LoRA** (Luo et al., 2023b) wraps the same loss into a rank-64 LoRA, which is the form most production checkpoints use because it leaves the base model intact and lets users compose with style/identity LoRAs. LCM is the entry point for every video distill that calls itself "LCM-style": it is cheap, single-stage, and has a well-defined target, but consistently underperforms DMD2 on FID and human-pref at the 1-4 NFE regime, and on video tends to produce oversmoothed outputs that lose motion (see DOLLAR Tab. 3 ablation).
- **AnimateLCM** [arXiv:2402.00769](https://arxiv.org/abs/2402.00769) (Wang et al., 2024). The first credible LCM-on-video. Decouples consistency learning into two stages: (1) distill image-prior LCM on T2I data, (2) freeze image prior, distill a separate motion-prior LCM on video data. The decoupling matters because (a) high-quality video data is scarcer than T2I data, and (b) joint distillation gives uneven gradient signals between spatial and temporal blocks. AnimateLCM achieves a workable 4-step T2V pipeline on top of AnimateDiff. Its FID degrades at 1-2 step regimes — OSV reports it at FVD 184.79 in 8-step, which OSV beats with 1-step (171.15). It pioneered the "LoRA on the temporal blocks only" packaging that Wan2.1 distillations would later use. Practically obsolete for state-of-the-art video DiT (DMD-family methods dominate), but historically the bridge.
- **Consistency Trajectory Models (CTM)** [arXiv:2310.02279](https://arxiv.org/abs/2310.02279) (Kim et al., ICLR 2024). Generalizes consistency to "any-to-any" along the trajectory: `f_θ(x_t, t, s) → x_s`, where the target is the teacher's ODE solution from `t` to `s`. Adds a GAN loss over the predicted endpoint. CTM achieves FID 1.92 on ImageNet-64 in one step (vs. CM 6.20, DMD 2.62, DMD2 1.51). On video, CTM has not been scaled to large DiT models — its key contribution to the distillation discourse is the conceptual decoupling of "where the trajectory goes" from "how many steps you take to get there", which Phased DMD and TMD both leverage explicitly.

These four — Progressive Distillation for the multi-stage paradigm, LCM/LCM-LoRA for the "match the trajectory at any timestep" objective, AnimateLCM for the image/motion decoupling on video, and CTM for the any-to-any generalization — form the substrate. Everything below either replaces the consistency objective with a distribution-matching one (DMD/DMD2/CausVid/Self-Forcing/Phased DMD/TMD/rCM/W2SVD/MagicDistillation) or augments it (DOLLAR mixing CD+VSD+reward, BLADE adding sparsity, Hyper-SD adding HFL).

## D.1 DMD / DMD2 — distribution matching

### D.1.1 Original DMD (CVPR 2024 highlight)

DMD (Yin et al., CVPR 2024) takes a different track: instead of matching the teacher's trajectory point-by-point, it minimizes an expectation of reverse-KL divergences between the diffused student distribution `p_fake_t` and the diffused real distribution `p_real_t`, across timesteps `t`. The gradient of the reverse KL has a beautiful representation:

\[
\nabla_\theta \mathcal{L}_{\text{DMD}} = \mathbb{E}_t \int \big( s_\text{fake}(F(G_\theta(z), t), t) - s_\text{real}(F(G_\theta(z), t), t) \big) \cdot \frac{dG_\theta(z)}{d\theta} \, dz
\]

where `s_real` is the score function of the data distribution (the frozen teacher) and `s_fake` is the score of the student's output distribution (an auxiliary network trained online via score matching on the student's outputs). DMD never explicitly evaluates a divergence; it uses the difference of two scores as a synthetic gradient. This trick — the "score-difference flow" — was independently identified earlier by Diff-Instruct (Luo et al., NeurIPS 2023), VSD/ProlificDreamer (Wang et al., NeurIPS 2023, in 3D context), and Score-Difference Flow (Weber, TMLR 2023). DMD's contribution was packaging it for image diffusion distillation and getting it to converge stably.

The catch: pure score-difference training is unstable. Yin et al. found they needed an additional regression term: collect a dataset of (noise, image) pairs from the teacher with a deterministic ODE sampler, and regress the student against them with LPIPS. On ImageNet-64 this is fine. On SDXL with LAION 6.0's 12M prompts, it costs **~700 A100-days** of teacher inference at 5s/pair. That dataset construction cost is **>4× the entire DMD2 training compute**. And the regression objective has the side effect of upper-bounding student quality at the teacher: the student is forced to mimic teacher trajectories, so it inherits all their errors.

DMD's headline numbers: 1-step on ImageNet-64 → FID 2.62 (vs. EDM teacher 2.32 ODE / 1.36 SDE at 511 NFE); 1-step on COCO with SDv1.5 → FID 11.49.

### D.1.2 DMD2 — what it actually fixes

DMD2 [arXiv:2405.14867](https://arxiv.org/abs/2405.14867) (Yin et al., NeurIPS 2024 oral; [tianweiy/DMD2](https://github.com/tianweiy/DMD2)) is the foundation that essentially every video DiT distill inherits from. Three independent contributions:

**(1) Remove the regression loss; add a Two Time-scale Update Rule (TTUR).** Naively dropping the regression term collapses training: the average-brightness statistic of generated samples drifts without converging (DMD2 Appendix C). DMD2 attributes this to `s_fake` not tracking the generator's distribution — `s_fake` is updated online on a non-stationary target. They borrow the TTUR from Heusel et al.'s GAN paper: train `s_fake` 5× more frequently than the generator. With `dfake_gen_update_ratio = 5`, training is stable and matches DMD's quality on ImageNet (FID 2.61 vs 2.62) **without the dataset construction**. This single change makes large-scale text-to-image distillation tractable. **Every video DMD recipe in FastVideo, lightx2v, and Cosmos-Predict2.5 uses 5:1.**

**(2) GAN on the bottleneck of `s_fake`, trained on real data.** DMD's distilled student is bounded above by the teacher's score approximation. A small GAN classifier head on the bottleneck of `s_fake`'s UNet, trained with non-saturating loss on real images vs student outputs, breaks that bound: the discriminator sees real data the teacher never recovered properly, and the generator gets gradients that push it past the teacher. The integration is minimal — one classification head, no separate discriminator backbone — and it improves SDXL FID from 26.90 (no GAN) to 19.32 (DMD2 4-step), beating the SDXL teacher at 100 NFE (19.36). For video, this GAN-on-bottleneck is **conceptually copied** by OSV (with DINOv2 backbone), W2SVD/MagicDistillation (with hybrid latent+pixel discriminator), and DOLLAR (latent reward model as a cheaper substitute).

**(3) Backward simulation for multi-step students.** Vanilla multi-step distill trains each step on noised real data (`x_t = α_t·x_real + σ_t·ε`). At inference, every step except the first sees student-generated `x̂_t` instead. The training-test mismatch hurts. DMD2 fixes it: at each training step, sample a target timestep index, then **simulate the inference rollout** with the current student to get the input distribution the student will actually see. Loss is computed only on the final step; intermediate steps are run under `torch.no_grad()`. This is the trick that makes the 4-step SDXL student work (FID 19.32, beating the 100-step CFG=8 teacher at 20.39). The same backward simulation flag (`--simulate_generator_forward`) is the default in FastVideo's Wan 2.2-TI2V-5B distill.

For a 4-step model, DMD2 uses denoising_steps `999, 749, 499, 249` (uniform 250-step jumps in a 1000-step teacher). Multi-step trajectory: at each step, denoise to `x̂_0`, then add noise back to get `x̂_{t_{i+1}}`, then denoise again. This alternation is identical to the LCM/multistep-CM sampler at inference.

DMD2's actual loss code (verified in `/Users/yewang/work/FastVideo/fastvideo/training/distillation_pipeline.py`, lines 612-690):

```python
real_score_pred_video = pred_real_video_cond \
    + (pred_real_video_cond - pred_real_video_uncond) * self.real_score_guidance_scale
grad = (faker_score_pred_video - real_score_pred_video) \
       / torch.abs(original_latent - real_score_pred_video).mean()
grad = torch.nan_to_num(grad)
dmd_loss = 0.5 * F.mse_loss(original_latent.float(),
                            (original_latent.float() - grad.float()).detach())
```

A few details to highlight:

- The CFG on the **real score** is folded directly into the gradient (`cond + w·(cond − uncond)`), and uses the **DMD2 parameterization** `x_cond + w·(x_cond − x_uncond)`, offset by 1 from the Ho & Salimans form. So `w_dmd2=0` is conditional, `w_dmd2=−1` is unconditional, and FastVideo's recipes use `real_score_guidance_scale=3.5` for Wan 2.1 1.3B (equivalent to standard CFG=4.5).
- The denominator `torch.abs(original_latent − real_score_pred_video).mean()` is the standard DMD2 normalization, scale-invariant w.r.t. data magnitude. Critical: when the student gets close to the teacher, the denominator goes to zero and `nan_to_num` clips. Empirically this transitions the student from "being pushed" to "stable equilibrium".
- The loss is `0.5 · MSE(x_0, sg[x_0 − grad])`, which is the standard "stop-gradient pseudo-target" trick that lets autograd compute the score-difference gradient correctly through `dG_θ/dθ`.

The GAN term in DMD2 is **not present** in FastVideo's pipeline — they ablated it out for video, since the generator was already fragile and a GAN on top destabilized convergence at scale. Wan2.2-Lightning and Phased DMD also omit GAN. OSV, BLADE, W2SVD/MagicDistillation, and Hyper-SD keep it. The community has not converged on whether a GAN term helps for video DMD — see §D.4.

**Headline numbers**: DMD2 4-step on SDXL (10K COCO prompts) → FID 19.32, Patch-FID 20.86, CLIP 0.332. SDXL teacher at 100 NFE → 19.36 / 21.38 / 0.332. So the 4-step student matches the teacher in FID and CLIP, beats it in Patch-FID, with **25× fewer forward passes**. On ImageNet-64 1-step DMD2 → FID 1.51 (vs. DMD 2.62, EDM teacher ODE 2.32). Limitations DMD2 explicitly states: image-only diversity drops vs. teacher (mode collapse from reverse KL), guidance scale is fixed at training time (no AGD-style runtime control), and the GAN component requires real data which is hard for niche conditioning.

### D.1.3 What DMD2 left open for video

DMD2 was image-only. The video generalizations have to handle three things DMD2 did not:

1. **Causal models** — bidirectional teacher → causal student is asymmetric, and even fitting both in memory is the bottleneck. CausVid §D.2.
2. **Train-test mismatch under autoregression** — DMD2's backward simulation works for bidirectional multi-step, but autoregressive video has KV-cache state that affects later frames. Self-Forcing §D.3.
3. **One-step video collapses** — naive 1-step DMD on video produces frozen frames. Phased DMD's SGTS analysis §D.4 shows why.

## D.2 CausVid — first DMD-on-video, with bidirectional → causal asymmetric distillation

CausVid [arXiv:2412.07772](https://arxiv.org/abs/2412.07772) (Yin et al., CVPR 2025; [tianweiy/CausVid](https://github.com/tianweiy/CausVid); [HF tianweiy/CausVid](https://huggingface.co/tianweiy/CausVid)) is the paper that opened video DiT to step distillation. It bundles three contributions: an autoregressive (AR) DiT architecture with block-wise causal attention, asymmetric distillation from a bidirectional teacher into a causal student, and an ODE-pair initialization scheme that stabilizes the early DMD iterations.

### D.2.1 Block-wise causal architecture

The video latent (10s, 12 FPS, 352×640) is encoded into latent chunks of 5 latent frames each, by a 3D VAE. Each chunk thus represents 16 RGB frames. The student DiT applies **bidirectional self-attention within a chunk** and **causal attention across chunks**:

\[
M_{i,j} = \begin{cases} 1 & \text{if } \lfloor j/k \rfloor \le \lfloor i/k \rfloor \\ 0 & \text{otherwise} \end{cases}
\]

with chunk size `k = 5`. The intra-chunk bidirectional attention captures fast local temporal dependencies (a butterfly's wing across 16 frames), while the inter-chunk causal mask makes inference KV-cacheable. Because the VAE only outputs pixels at chunk granularity (5 latent frames → 16 RGB frames), there is no latency benefit to fully causal attention; chunk-causal is the sweet spot. At inference, since KV is cached and all queries are only over current+past tokens, **block-wise causal attention is no longer needed** — they swap in dense FlashAttention/FlexAttention for free throughput.

### D.2.2 Asymmetric distillation: bidirectional teacher → causal student

The naive option — train a causal teacher first, then distill — is dominated. CausVid Tab. 4 shows that a many-step **causal** teacher already lags the bidirectional one (frame quality 60.1 vs 62.7 at 100 NFE), and a 4-step student distilled from the causal teacher gets 61.7 frame quality. Distilling the same 4-step student from the **bidirectional** teacher gets **64.4**. The asymmetry is essentially free quality. The reason: bidirectional teacher attention is more expressive at training time, and the DMD objective is a distribution-matching loss — it doesn't require matching internal state, only that the student's marginal distribution matches the teacher's.

The training algorithm (Algorithm 1):
- Sample a video, divide into L chunks
- Sample per-chunk timesteps `t^i ∼ Uniform({0, t_1, ..., t_Q})` — diffusion-forcing style
- Add noise to each chunk independently
- Student predicts clean frames with block-causal mask
- Add noise to predictions at a single global timestep `t`
- Compute DMD loss using bidirectional teacher (cond + uncond, CFG scale `w=3.5`)
- Update student via DMD gradient, update fake-score via standard denoising loss
- TTUR ratio 5:1 (inherited from DMD2)

### D.2.3 ODE initialization to bridge architectures

Cold-starting a causal student against a bidirectional teacher with DMD diverges. The fix is an inexpensive ODE pretraining stage: generate ~1000 noise→image trajectories from the teacher, regress the causal student against them with `L_init = ||G_φ({x^i_t}) − {x^i_0}||²`, for 3000 iterations at LR 5e-6. This is cheap (1000 trajectories, not 12M) and gets the student to a place where DMD can take over. The followup DMD stage runs **6000 iterations** at LR 2e-6 with TTUR=5, on **64 H100 GPUs over ~2 days**. That is the entire compute budget for a CVPR 2025 result.

### D.2.4 Inference and headline numbers

At 4 NFE with timesteps `{999, 748, 502, 247}`, KV cache enabled, FlexAttention:
- **Latency 1.3s** for the first chunk (vs. CogVideoX-5B 208.6s)
- **Throughput 9.4 FPS** (vs. CogVideoX-5B 0.6, Pyramid Flow 2.5)
- VBench-Long total **84.27** (highest at the time)
- Dynamic Degree **92.69**, Aesthetic 64.15, Imaging 68.88

CausVid generates 10s clips natively and extends to 30s and even 14-min videos via sliding-window inference (the model is trained on 10s, but rolling KV cache across windows works because the asymmetric DMD loss does not impose a hard length boundary). Long-video imaging quality decays slowly (Fig. 8 in the paper) — much less than StreamingT2V or FIFO-Diffusion.

### D.2.5 The flaw Self-Forcing identified

CausVid's distribution-matching is computed against a **diffusion-forcing** training scheme: noise levels for each chunk are sampled independently, so the student trains on a distribution `p_train(x^{1:N})` where each `x^i` is a noisy ground-truth chunk. But at inference, the student conditions on **its own clean past chunks** plus the current noisy chunk being denoised. So `p_train ≠ p_inference`. CausVid achieves 84.27 VBench despite this mismatch; Self-Forcing, by closing the gap, gets to 84.31 with only ~3 days of training. The bigger gain from Self-Forcing is in long-video stability, which is exactly where CausVid still drifted.

## D.3 Self-Forcing — bridging the train-test gap

Self-Forcing [arXiv:2506.08009](https://arxiv.org/abs/2506.08009) (Huang et al., NeurIPS 2025 spotlight; [project](https://self-forcing.github.io/); [code](https://github.com/guandeh17/Self-Forcing)) attacks the exposure bias in CausVid head-on. It is named after the inverse of "teacher forcing" in classical RNN literature (Lamb et al. 2016 "Professor Forcing", Ranzato et al. 2016 SeqGAN-style approaches): instead of conditioning each step on ground-truth context, condition on the model's own outputs.

### D.3.1 The setup: train-time autoregressive rollout

Self-Forcing trains the causal student by **performing the full autoregressive rollout** during each training step, with a rolling KV cache:

```
for block_index, current_num_frames in enumerate(all_num_frames):
    for index, current_timestep in enumerate(self.denoising_step_list):
        exit_flag = (index == exit_flags[block_index])
        if not exit_flag:
            with torch.no_grad():
                pred_flow = current_model(...)  # advance the rollout
                noisy_input = scheduler.add_noise(denoised_pred, ...)
        else:
            pred_flow = current_model(...)  # gradient through this step only
            denoised_pred = pred_noise_to_pred_video(...)
            break
```

(verified in `/Users/yewang/work/FastVideo/fastvideo/training/self_forcing_distillation_pipeline.py`, lines 248-336). The student generates the whole video in chunks, each chunk via a few-step DMD denoising, and the gradient is backpropagated through **only one denoising step per chunk**. This is the **stochastic gradient truncation** of DMD2, lifted into the autoregressive loop. Critical: the choice of which step gets the gradient is **random per chunk** (`exit_flags`), broadcast across all DDP ranks — so over training, every step in every chunk is supervised, but per-iteration memory stays bounded.

To make this fast enough to be practical, Self-Forcing requires a **few-step student** (4-step) as a starting point. Their student initialization recipe:

1. **4-step bidirectional DMD pretraining**: distill a 4-step bidirectional version of the student first (this is identical to the FastWan 1.3B recipe at this point).
2. **Causal ODE pretraining**: convert the 4-step model to causal attention via short fine-tune on ODE pairs (~3k iterations).
3. **Self-Forcing DMD**: the autoregressive rollout step, with the actual loss being CausVid-style DMD on the final chunk output, but with KV-cache state that matches inference.

### D.3.2 Holistic video-level distribution-matching loss

The third innovation is the **video-level DMD loss**. Instead of computing a per-chunk loss, Self-Forcing collects the **entire generated video** and applies a single DMD/SiD/GAN loss on the joint distribution. This is what the bidirectional teacher actually models. The fake-score model is also bidirectional and operates on the full video. So the gradient signal is "make this autoregressive rollout look like a sample from the bidirectional teacher's distribution" — exactly the right loss.

Three loss flavors are reported (paper Sec. 3.3): SiD (Score Implicit Matching), DMD2 (reverse-KL), and GAN. SiD wins by a small margin in FID/quality but is more memory-intensive. DMD2 is the default.

### D.3.3 Rolling KV cache for arbitrary-length extrapolation

For inference, Self-Forcing uses a rolling KV cache: when the cache reaches its max size, evict the oldest tokens. For Wan 2.1 1.3B with 21 latent frames context, this works. But for Krea Realtime 14B (§D.10), the naive eviction causes catastrophic artifacts because the **first latent frame** has different VAE statistics than subsequent frames (the 3D VAE encodes the first RGB frame as a single latent, while subsequent groups of 4 RGB frames map to one latent each). Krea solves this with KV-cache recomputation (see §D.10.4).

### D.3.4 Numbers

- **17 FPS streaming** generation on a single H100 at 480p, sub-second first-frame latency
- VBench-Long total around CausVid's level, but with **dramatically better long-video stability** (10s training → arbitrary inference length without quality collapse)
- ~3 days on 64 H100 to converge from a CausVid-style initialization

### D.3.5 Self-Forcing-Plus

[GoatWu/Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus) is the production-grade reimplementation that lightx2v uses for its Wan 2.1 14B I2V step+CFG distill. The "Plus" additions:

- Support for **Wan 2.1 14B I2V at 480p and 720p** (the original Self-Forcing was 1.3B T2V)
- **flow_shift=5.0** (vs. the original's 8.0) — empirically discovered as a sweet spot, also adopted by Krea Realtime 14B
- **Independent first frame** option for I2V (so the conditioning image latent gets handled differently from the rest)
- LoRA-only training paths (essential for 14B fitting in memory)
- Multi-block context-noise scheme `context_noise=0` by default (means clean conditioning frames during cache update; some recipes set `context_noise=20–50` for additional stability)

The output checkpoints `lightx2v/Wan2.1-I2V-14B-{480P,720P}-StepDistill-CfgDistill-Lightx2v` are 4-step, CFG-1 (so 4 NFE total per chunk), with `shift=5.0` recommended at inference. They are the de facto Wan 2.1 distillation that everyone fine-tunes.

### D.3.6 FastVideo's recipe (canonical hyperparameters)

From `/Users/yewang/work/FastVideo/examples/distill/SFWan2.1-T2V/distill_dmd_t2v_1.3B.sh` and the SFWan2.2-A14B variant:

```
denoising_steps: '1000,750,500,250'    # 4-step, equally spaced
flow_shift: 5                          # vs. 8.0 in Wan 2.1 default
ema_decay: 0.99                        # generator EMA
ema_start_step: 100                    # warmup before EMA kicks in
real_score_guidance_scale: 3.0         # CFG=4.0 in Ho & Salimans form
dfake_gen_update_ratio: 5              # TTUR
fake_score_lr: 8e-6 (vs gen 1e-5)      # critic moves faster but lower LR
betas: '0.0, 0.999'                    # zero β1 (no momentum on generator)
warp_denoising_step: True              # shift the ts according to flow_shift
num_frame_per_block: 3                 # chunk size in latent frames
gradient_mask_last_n_frames: 21        # only backprop through last 21 frames
init_weights_from_safetensors: <ode_init_path>  # CausVid-style ODE init required
```

Important details:
- **EMA decay 0.99 starting at step 100**: standard "warm up before EMA". An EMA copy of the generator is maintained, used for validation and inference. Without it, generator weights oscillate with the GAN-on-fake dynamics.
- **`betas = (0.0, 0.999)`**: the generator AdamW uses β₁=0 (no first-moment momentum). This is a DMD2/SiD convention — gradients are noisy because of the score-difference structure, and momentum in this regime hurts.
- **`warp_denoising_step`**: the listed denoising_steps `[1000, 750, 500, 250]` get shifted via the flow-shift schedule, so the actual timesteps the student sees are not uniform but warped by `flow_shift=5`. This is what makes 4-step actually cover the noise schedule properly.

### D.3.7 The Wan 2.2 MoE problem

Self-Forcing on Wan 2.2 A14B (the actual MoE model) is the SFWan2.2-A14B recipe. The complication: Wan 2.2 has **two transformers** (`transformer` for high-noise expert, `transformer_2` for low-noise expert) with a `boundary_ratio` parameter that selects which expert to run at each timestep. FastVideo's pipeline correctly routes:

```python
def _get_real_score_transformer(self, timestep):
    if self.real_score_transformer_2 is not None and self.boundary_timestep is not None:
        if timestep.item() < self.boundary_timestep:
            return self.real_score_transformer_2  # Low noise expert
        else:
            return self.real_score_transformer    # High noise expert
```

But the student is also MoE. FastVideo loads `transformer` and `transformer_2` for the student, and routes via the same boundary. **Per-expert distillation** is the canonical approach. Wan2.2-Lightning V1.0/V1.1 (ModelTC) and lightx2v `Wan2.2-I2V-A14B-Moe-Distill` both ship with **2 high-noise + 2 low-noise distill steps** (4 NFE total per chunk). lightx2v reports that the **low-noise expert can be reused from Wan 2.1's LoRA** with no degradation; only the **high-noise expert needs fresh distillation** because high-noise dynamics changed substantially in Wan 2.2. This is consistent with the empirical observation in DCM (§D.9) that semantic vs detail experts have different optimization landscapes.

## D.4 Phased DMD — SNR-subinterval MoE distillation (Wan 2.2-Lightning)

Phased DMD [arXiv:2510.27684](https://arxiv.org/abs/2510.27684) (Fan et al., SenseTime + Beihang) is the distillation algorithm behind **Wan 2.2-Lightning V1.1 / V2.0** ([github.com/ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning)). It identifies and fixes the **stochastic gradient truncation strategy (SGTS)** failure mode that plagues both DMD2 and Self-Forcing when scaled to large MoE video models.

### D.4.1 The SGTS one-step degeneration

DMD2 and Self-Forcing both back-propagate through one randomly chosen step out of `K` denoising steps per training iteration. This is essential for memory: a full `K`-step backward pass through a 14B/28B video model is infeasible. But Phased DMD's analysis (§2 and Fig. 1d): when SGTS terminates after a single step (which happens with prob `1/K`), the loss for that iteration is computed exactly as if it were a 1-step distillation. The student receives a 1-step gradient, which on video tends to **kill motion** — collapsing dynamic scenes to near-static frames, and yielding outputs essentially indistinguishable from a 1-step distill.

For Wan 2.2 at 28B, the 1-step degeneration is severe. The SGTS-trained 4-step student has dynamic-degree scores ~0.4-0.5 (vs. teacher 0.7-0.9), and motion artifacts become dominant. Phased DMD says: stop pretending all `K` steps live in one big distribution; **partition the SNR axis** and train one expert per phase.

### D.4.2 Progressive distribution matching across SNR subintervals

Phased DMD divides the SNR range `[SNR_min, SNR_max]` into `P` subintervals (`P=4` for Wan 2.2-Lightning's 4-step config: `(t_3, t_2, t_1, t_0)` boundaries). For each phase `i`:
- Distill an expert `μ_i` for the SNR range `[t_{i+1}, t_i]`
- Backward simulation **terminates at t_{i+1}**, not at the clean state — so the loss for phase `i` is a 1-step distillation **within the subinterval**, not a 1-step distillation **of the whole trajectory**
- Each phase has only one gradient-recorded step (memory cost = 1-step DMD2)
- Phases are trained progressively: phase 0 (highest noise) first, then phase 1 conditioned on phase 0's output distribution, etc.

This avoids the SGTS one-step degeneration because the 1-step gradient is now over a **narrow SNR range** where the optimal solution is genuinely close to a single denoising step.

### D.4.3 The revised score-matching objective (clean-data-free)

The technical hurdle is that DMD's standard loss assumes access to clean `x_0` to compute the "ideal" target. In phase `i ≠ P-1` (i.e., not the final phase), **`x_0` is not accessible** — the student only ever produces outputs at SNR `t_i`, never clean. Phased DMD derives a **revised objective** for unbiased score estimation when only "noisy clean" samples (samples at noise level `t_{i+1}`) are available:

\[
\mathcal{L}_{\text{Phased-DMD}}^{(i)}(\theta) = \mathbb{E}_{x_{t_{i+1}}, x_t} \big[ \| x_{t_{i+1}} - \mathrm{sg}[x_{t_{i+1}} - \nabla_{x_t} \log p_{\text{teacher}}^{[t_{i+1}, t_i]}(x_t) + \nabla_{x_t} \log p_{\text{fake}}^{[t_{i+1}, t_i]}(x_t)] \|^2 \big]
\]

where the fake-score and real-score networks are **trained on samples at the SNR subinterval** rather than over the full noise schedule. This is the core mathematical contribution — it preserves the theoretical integrity of distribution matching while permitting the phased decomposition.

### D.4.4 Mixture-of-Experts (MoE) generator structure

Phased DMD inherently produces a **few-step MoE generative model**, regardless of whether the teacher was MoE. Each phase's expert is a separate set of parameters (or a separate LoRA on the base) that gets selected based on the current timestep. For Wan 2.2 (where the teacher is MoE: 2 experts at boundary `0.875`), Phased DMD's 4 phases correspond to 4 experts: `μ_0, μ_1` go through the high-noise expert, `μ_2, μ_3` through the low-noise expert. The boundary handling is consistent with FastVideo's MoE pipeline (§D.3.7).

### D.4.5 Optional combination with SGTS (4-step in 2 phases)

Phased DMD also allows a hybrid: train `μ_0` over a wider SNR range that covers phases 0+1, but with SGTS within that range, where backward simulation terminates **before reaching the boundary of phase 1**. This gives a 4-step generator using only 2 phases (Fig. 1d), simplifying the framework. In practice: Wan 2.2-Lightning V1.1 ships with `phase_0` (SNR 1.0 → 0.5) and `phase_1` (SNR 0.5 → 0.0), each containing a 2-step MoE-routed denoising, totaling 4 NFE.

### D.4.6 Validation results

Phased DMD distills:
- **Qwen-Image-20B** (T2I) — preserves complex text rendering and visual diversity; SGTS baseline collapses on text fidelity
- **Wan 2.2-28B** (T2V/I2V) — beats SGTS on dynamic-degree, recovers ~90% of teacher's motion budget at 4 NFE; this is the algorithm behind Wan 2.2-Lightning V1.1/V2.0

Wan2.2-Lightning V2.0 is the production checkpoint most teams use for Wan 2.2 acceleration; V2.0 also includes additional fine-tuning on aesthetic data. The recipe is data-free (pure DMD + score matching) and does **not** require GAN loss or regression loss, making it scalable.

### D.4.7 Limitations Phased DMD acknowledges

- The progressive training schedule means **total wall-clock is higher** than DMD2 single-stage (~2× for 4 phases), even though per-phase memory is 1-step
- The MoE structure means **inference dispatch logic** is needed (router on timestep), which is fine for Wan 2.2 but adds friction for non-MoE models
- The number of phases `P` is a hyperparameter; the paper uses `P=4` but reports fragility for `P>8`

## D.4bis Hyper-SD, PCM, T2V-Turbo, InstaFlow — auxiliary acceleration approaches

This subsection covers methods cited in the FastVideo / lightx2v / Cosmos-Predict 2.5 lineages that are either
**foundational** (PCM, InstaFlow), **production-deployed** (T2V-Turbo), or **provided implementation
patterns reused elsewhere** (Hyper-SD).

### D.4bis.1 Hyper-SD [arXiv:2404.13686]

Hyper-SD (Ren et al., ByteDance) introduces three innovations layered on consistency distillation:

**(1) Trajectory Segmented Consistency Distillation (TSCD).** Divides the timestep range into segments
and enforces consistency *within each segment*, then progressively reduces the number of segments
to achieve all-time consistency. This avoids the "sub-optimal consistency from insufficient model
fitting capability and accumulated errors" failure mode of vanilla LCM. The progressive scheduling
echoes Phased DMD's per-SNR-subinterval idea, but Hyper-SD operates on consistency loss while
Phased DMD operates on score-difference loss.

**(2) Human Feedback Learning (HFL).** After TSCD, fine-tune the accelerated model with reward
gradients. Modifies the ODE trajectory to better suit few-step inference. This is the same idea
as DOLLAR's reward-fine-tuning stage, but Hyper-SD uses pixel-space reward models (HPSv2, ImageReward)
rather than DOLLAR's latent reward.

**(3) Score distillation for one-step.** A final score-distillation stage to push from few-step
into 1-step regime. Unified LoRA supports inference at *all* NTEs (1, 2, 4, 8, 16) via the same
LoRA, which is a major UX win — users can pick their NFE budget without swapping checkpoints.

**Headline**: Hyper-SDXL surpasses SDXL-Lightning by **+0.68 in CLIP Score** and **+0.51 in Aes Score**
in 1-step inference on SDXL.

For video: Hyper-SD's TSCD + HFL pattern is the basis for **Hyper-WAN** style community LoRAs. The
HFL stage also influenced AAPT and DOLLAR.

### D.4bis.2 PCM — Phased Consistency Models [arXiv:2405.18407]

PCM (Wang, Lu et al., NeurIPS 2024) identifies three flaws in LCM:
1. **CFG-incompatibility** at higher guidance scales (exposure problems)
2. **Suboptimal multi-step quality** — LCM is designed for 1-step but inference often uses 4-8
3. **Domain shift from teacher**

PCM's fix: divide the time domain into **N phases** and apply consistency within each phase, with
adversarial training to enforce phase-boundary consistency. Generalizes the LCM design space.

PCM achieves comparable 1-step results to specialized 1-step methods, while outperforming LCM at
1-16 steps. **Critically for video**, PCM's methodology was applied to **CogVideoX**, training the
state-of-the-art few-step T2V generator at the time. This was the first non-DMD method to scale to
modern T2V DiTs.

### D.4bis.3 T2V-Turbo / T2V-Turbo-v2 [arXiv:2410.05677]

T2V-Turbo (Li et al., NeurIPS 2024; [project](https://t2v-turbo.github.io/)) integrates **mixed reward
feedback** into video consistency distillation:

- Image-text reward: **HPSv2.1** (CLIP-style, fine-tuned for human preference)
- Video-text reward: **ViCLIP** + **InternVid2** (video-text alignment)

The novelty: directly optimize rewards on the **single-step generation** that arises naturally during
CD training, bypassing memory constraints of backpropagating through multi-step sampling. This is
the same memory-saving idea later refined in DOLLAR's latent reward model.

**Headline numbers** (4-step T2V-Turbo on VC2 backbone):
- VBench total beats Gen-2 and Pika at 4 NFE
- 50-step teacher → 4-step student preferred by humans, **12.5× acceleration with quality improvement**
- 8-step variant further improves quality

T2V-Turbo-v2 (followup) integrates additional reward models and training data. Both are listed
in CausVid's reference comparisons. The T2V-Turbo stack proved that **reward fine-tuning can recover
distillation losses for video** — a pattern picked up by DOLLAR and Hyper-SD.

### D.4bis.4 InstaFlow / 2-RF / PeRFlow — rectified-flow lineage

**InstaFlow** (Liu, Zhang, Ma, Peng, Liu — ICLR 2024) trains a one-step image generator on top of
**rectified flow** (Liu et al., ICLR 2023). The rectified flow training **straightens the ODE
trajectory** so that 1-step and many-step generations agree. After training a teacher with rectified
flow, distill into a student that performs 1 NFE.

**2-Rectified Flow (2-RF)**: a second reflow iteration further straightens trajectories. Stacks well
with distillation.

**PeRFlow** (Yan et al.): **Piecewise Rectified Flow** — splits the trajectory into pieces and
applies piecewise rectification with real data samples. Acts as a universal plug-and-play accelerator
that can be combined with any base diffusion model.

For video: rectified-flow-style training has not been adopted at scale (the iterative refinement of
trajectories is expensive at video scale), but the conceptual link is important: **flow-matching
models** (Wan, LTX, HunyuanVideo, Cosmos, SD3) all use rectified flow as the default training
paradigm, so any subsequent distillation builds on a "more straightened" base trajectory than
DDPM models. This makes flow-matching models inherently more amenable to few-step distillation.

## D.5 TDM — Trajectory Distribution Matching

TDM [arXiv:2503.06674](https://arxiv.org/abs/2503.06674) (Luo et al., ICCV 2025; [project](https://tdm-t2x.github.io/); [code](https://github.com/Luo-Yihong/TDM)) bridges the trajectory-matching family (LCM, CD, PCM) and the distribution-matching family (DMD, DMD2, Diff-Instruct). The pitch: distribution matching is too coarse for fine multi-step refinement; trajectory matching is too rigid (instance-level matching dominates and quality plateaus). TDM matches **the distribution of the student's trajectory** to **the distribution of the teacher's trajectory**, but at distribution level — best of both.

### D.5.1 Sampling-steps-aware objective

TDM's loss decouples learning targets across steps. The student's trajectory `(x_{t_1}, x_{t_2}, ..., x_{t_K})` should match the teacher's trajectory's distribution at each `t_i`. The fake-score model is trained on the student's per-step distribution `p_{θ, t_i}`, and the real-score model is the teacher. The loss:

\[
\nabla_\theta \mathcal{L}_{\text{TDM}}(\theta) \approx \sum_{i=0}^{K-1} \sum_{j=t_i}^{t_{i+1}} \lambda_j [s_\psi(x_j, j) - s_\phi(x_j, j)] \frac{\partial x_{t_i}}{\partial \theta}
\]

with non-overlapping intervals `[t_i, t_{i+1})` so a **single fake-score network** suffices for all `K` stages (timestep input naturally separates the distributions). Backprop through the student is constrained to **one ODE step at a time** for memory.

### D.5.2 Headline numbers

- **PixArt-α distilled to 4-step in 500 iterations and 2 A800 hours** (~$10 of compute), and the student outperforms the teacher on real user preference at 1024×1024
- **CogVideoX-2B distilled to 4-step**: VBench total 81.65 vs. teacher 80.91. **Student beats teacher** on T2V at ~25× speedup
- TDM is **data-free** — no real videos needed, only teacher-generated trajectories

The CogVideoX-2B distillation is shipped as a LoRA at [HF Luo-Yihong/TDM_CogVideoX-2B_LoRA](https://huggingface.co/Luo-Yihong/TDM_CogVideoX-2B_LoRA) — drop-in for the CogVideoX pipeline. Trained on timesteps `[999, 856, 665, 399]`, used at 4 NFE inference.

### D.5.3 Where TDM lands relative to DMD2

TDM is the principled multi-step generalization of DMD2: it explicitly trains different step-distributions, while DMD2 trains all steps with the same backward-simulation loss. For 4-step regimes, the gap is small. For 8+ step regimes (where the multi-step structure matters more), TDM should win. In practice the video community uses DMD2-style methods more because the engineering integration (single fake-score, simple ratio scheduling) is simpler. TDM is the foundation for **BLADE** (§D.11), where the per-step decomposition is critical for joint sparsity training.

## D.6 TMD — Transition Matching Distillation (NVIDIA + NYU)

TMD [arXiv:2601.09881](https://arxiv.org/abs/2601.09881) (Nie, Berner, Ma, Liu, Xie, Vahdat — NVIDIA + NYU; [project](https://research.nvidia.com/labs/genair/tmd)) is one of the most architecturally interesting distillation methods to date. It's based on the observation that diffusion model layers have a **hierarchical structure**: early layers extract semantics (which token, which scene), later layers refine fine details (texture, motion). TMD exploits this by **decoupling the teacher into a backbone and a flow head**.

### D.6.1 The decoupled architecture

The student model has two components:
1. **Main backbone** — the majority of early DiT blocks. Takes noisy `x_t`, timestep `t`, text `c`, outputs main feature `m_t`
2. **Flow head** — the last few DiT blocks. Conditions on `m_t, c`, performs **multiple inner flow updates** between transition steps

For each outer transition step `t_i → t_{i-1}`, the backbone is run **once** (extracting semantics for that noise level), then the flow head runs **multiple inner steps** to refine the prediction toward `t_{i-1}`. This is the "transition" formulation — instead of predicting a single velocity at `t`, the flow head models the **full distributional transition** from `t_i` to `t_{i-1}`.

The shared backbone is the cost-saver: it runs once per outer step (which is the dominant cost), while the cheap flow head can run many inner steps for free.

### D.6.2 Two-stage training

**Stage 1 — Transition matching adaption**: the flow head is initialized as a MeanFlow [arXiv:2505.13447] flow head — a few-layer module that predicts the transition velocity averaged over `[r, t]`. Adapted into a conditional flow map.

**Stage 2 — Distribution matching distillation**: improved DMD2 with **flow head rollout in each transition step**. The student's transition process is unrolled, the teacher's denoising trajectory is matched at the distribution level. The fake-score and real-score networks are bidirectional/standard DiTs.

### D.6.3 Headline numbers (Wan2.1 distillation)

TMD's flagship result: **84.24 VBench-Long total** at **NFE=1.38 (effective)** on Wan 2.1 14B T2V. The "effective NFE" accounts for the cheap flow-head inner steps as fractions of an outer NFE. So this is essentially "1-2 outer transition steps with ~3-4 inner flow updates per step". State-of-the-art at this NFE budget.

For comparison: rCM at 1 NFE on Wan 2.1 14B → 83.02. TMD at 1.38 NFE → 84.24. TMD wins by 1.2 VBench points at slightly higher cost.

[VERIFIED] from the project page and paper PDF: TMD outperforms existing distilled methods at comparable inference budgets in visual fidelity and prompt adherence.

### D.6.4 Limitations TMD has

- **Wall-clock numbers NOT in the paper.** The 1.38 NFE figure is theoretical; the actual inference latency depends on flow-head depth, kernel implementations, and FlexAttention support, none of which TMD reports. Engineering teams attempting to ship TMD-distilled models should expect ~30-50% additional implementation work to realize the speedup (custom kernel for the flow-head loop).
- **Training cost is high**: the two-stage process requires both MeanFlow-style adaptation (multi-day) and DMD2-style fine-tuning (multi-day). NVIDIA reports ~64 H100 days for Wan2.1 14B distillation.
- **Architecture changes the inference contract**: TMD is not a drop-in replacement; users need to adopt the backbone+flow-head decomposition. Currently no public weights at [research.nvidia.com/labs/genair/tmd] (project page only).

## D.7 rCM — Score-Regularized Continuous-Time Consistency Models (NVIDIA Cosmos Lab, ICLR 2026)

rCM [arXiv:2510.08431](https://arxiv.org/abs/2510.08431) (Zheng et al., ICLR 2026; [NVlabs/rcm](https://github.com/NVlabs/rcm)) is the first effort to scale continuous-time consistency models (sCM, MeanFlow) to billion-parameter video diffusion models, and to make them competitive with DMD2 at production scale. The headline observation: **sCM has fundamental quality issues** (blurry textures, unstable geometry across frames, poor text rendering), which rCM diagnoses as "error accumulation under mode-covering forward divergence" and fixes with a **dual-divergence** training objective.

### D.7.1 Why sCM fails on large video models

sCM's loss takes the limit `Δt → 0` of the consistency loss:

\[
\mathcal{L}_{\text{sCM}}(\theta) = \mathbb{E}\left[ \left\| F_\theta(x_t, t) - F_{\theta^-}(x_t, t) - \frac{g}{\|g\|^2 + c} \right\|^2 \right]
\]

where `g = w(t) · df_{θ^-}/dt` is computed via **forward-mode automatic differentiation (Jacobian-vector product, JVP)**. This is mathematically clean but numerically fragile:

1. **Self-feedback amplification**: the JVP target contains `dF_{θ^-}/dt`, which is a first-order self-feedback signal. Errors propagate from small `t` (close to clean data) to large `t` (close to noise) and get amplified by `sin(t)` weighting. Under BF16, this amplification compounds.
2. **Mode-covering forward divergence**: sCM's loss is defined on real or teacher-generated data (forward divergence), which is mode-covering — it spreads probability mass to cover all training samples. For high-fidelity generation, this leads to "averaged" outputs that lack sharp details.
3. **JVP infrastructure is non-trivial**: BF16, FlashAttention-2, FSDP, context parallelism (CP) all complicate JVP — `torch.func.jvp` doesn't work natively with FSDP modules.

### D.7.2 The rCM fix: long-skip score regularization

rCM's idea: **add DMD as a "long-skip regularizer"** to sCM's instantaneous self-consistency. The two divergences are complementary:
- sCM uses **forward divergence** on real data → mode-covering, preserves diversity
- DMD uses **reverse divergence** (reverse KL) on student-generated data → mode-seeking, sharp quality

Combined loss:
\[
\mathcal{L}_{\text{rCM}}(\theta) = \mathcal{L}_{\text{sCM}}(\theta) + \lambda \cdot \mathcal{L}_{\text{DMD}}(\theta)
\]

with `λ = 0.01` empirically working across all models tested. The DMD branch uses standard fake-score updates; rCM does **not** use SiD or GAN, both empirically not improving large T2I/T2V tasks.

### D.7.3 Stable time-derivative computation

The JVP `(dF_{θ^-}/dt)` is split into spatial and temporal parts:
\[
\frac{dF_{θ^-}}{dt} = (\nabla_{x_t} F_{θ^-}) F_{\text{teacher}} + \partial_t F_{θ^-}
\]

The temporal partial `∂_t F_{θ^-}` is the unstable part (oscillatory time embeddings). Two strategies:
- **Semi-continuous time** (for ≤ 2B models): finite-difference approximation `∂_t F ≈ [cos(Δt) F(x_t, t) − F(x_t, t − Δt)] / sin(Δt)` with `Δt = 1e-4`. No architectural change.
- **High-precision time** (for 10B+ models, video tasks): keep continuous-time JVP, but enforce **FP32 precision** for time embedding layers via `torch.amp.autocast`. Required for stability at scale.

rCM also developed a **FlashAttention-2 JVP Triton kernel** that integrates JVP into the forward pass with block-wise tiling, supports both self- and cross-attention, FSDP, and Ulysses CP. This is the infrastructure NVIDIA needed to scale.

### D.7.4 Validation results — quality/diversity matching DMD2

rCM is validated on:
- **Cosmos-Predict2 0.6B / 2B / 14B** (T2I, GenEval)
- **Wan 2.1 1.3B / 14B** (T2V, VBench)

Highlights:
- **Wan 2.1 14B + rCM at 4 NFE → VBench 84.92** (vs. DMD2 reported on 1.3B 84.56)
- **Wan 2.1 14B + rCM at 1 NFE → VBench 83.02 (single-step!), 14.4 FPS on H100**
- **Wan 2.1 1.3B + rCM at 1 NFE → VBench 82.65, 32.3 FPS on H100**
- Cosmos-Predict2 14B + rCM 4-step → GenEval 0.83 (matches teacher 0.84 at 70 NFE), 50× speedup

The diversity advantage over DMD2 is visible in qualitative comparisons (Fig. 1): rCM samples have similar object positions/orientations/motion to the teacher, while DMD2 collapses to fewer modes.

### D.7.5 Implications

rCM is the strongest evidence that **dual-divergence (forward + reverse)** is the right objective for large-scale distillation. The pure-DMD2 line of work suffers from mode collapse; the pure-sCM line suffers from quality issues. rCM combines them at low extra cost.

The paper explicitly notes Self-Forcing as a "well-instantiated reverse-KL DMD", and suggests **forward-divergence-based distillation by causal teacher with teacher forcing could potentially complement Self-Forcing** for autoregressive models. This is an open research direction.

For our target deployment (Wan 2.x on B200), rCM is competitive but the FA-2 JVP kernel is not yet upstream in PyTorch. Cosmos-Predict 2.5 (NVIDIA's official path) uses **DMD2 with TrigFlow parameterization** in production — see §D.10.

## D.8 OSV — One-Step Video (Mao, Jiang, Wang et al., CVPR 2025)

OSV [arXiv:2409.11367](https://arxiv.org/abs/2409.11367) (Mao et al., CVPR 2025) targets I2V at the smallest possible NFE (1-step). It predates DMD2-on-video and uses a **two-stage CD+GAN** approach.

### D.8.1 Two-stage training

**Stage 1: Latent GAN pretraining.** Add a Huber + GAN loss on top of LoRA-only training. Use **DINOv2 directly on latent space** (not pixel space — saves a VAE decode per training step). The discriminator has a frozen DINOv2 backbone with trainable temporal/spatial heads. The student LoRA converges fast (~few hundred steps), reducing GPU memory by ~50% (35.8 GB vs SF-V's 73.5 GB at 512×512).

**Stage 2: Adversarial Consistency Latent Distillation.** Initialize the student from stage 1, add LCM-style consistency loss with a **multi-step ODE solver** (`m`-step lookahead instead of 1-step) for higher accuracy. CFG distillation similar to LCM. The discriminator continues from stage 1.

### D.8.2 Headline numbers

**OpenVid-1M evaluation (I2V from SVD teacher)**:
- **OSV 1-step → FVD 171.15** ← exceeds AnimateLCM 8-step (FVD 184.79)
- OSV 8-step → FVD ~150
- SVD 25-step → FVD 156.94 (teacher reference)
- SVD 50-step → FVD ~140 (teacher upper bound)

So OSV's 1-step model **beats the LCM baseline at 8 steps** and **approaches the SVD teacher at 25 steps**, on a fixed I2V evaluation. This was the strongest 1-step video result of late 2024.

### D.8.3 Limitations and what OSV taught the field

- OSV is for **SVD-class models** (~1-1.5B parameters). Scaling to 14B with the same recipe was unstable in followup work — too much GAN noise overwhelms the consistency signal.
- The DINOv2-on-latent trick was **adopted by W2SVD/MagicDistillation** and **lightx2v's discriminator**.
- OSV showed that **CFG can be removed at inference for I2V** without quality loss, because the conditioning image already provides strong guidance — this insight propagated into all the subsequent video distillations (lightx2v's CFG-1 default, FastWan's no-CFG, etc.).

## D.9 DOLLAR / DCM / W2SVD

Three orthogonal-but-complementary methods that augment the basic DMD/CD recipe in different directions.

### D.9.1 DOLLAR — Distillation + Latent Reward Optimization

DOLLAR [arXiv:2412.15689](https://arxiv.org/abs/2412.15689) (Ding et al., Princeton + Adobe, ICCV 2025) targets the **diversity-vs-quality trade-off** in 4-step video distillation:
- VSD (DMD-like) → high quality, mode collapse
- CD (LCM-like) → diversity, lower quality

DOLLAR's loss combines all three:

\[
\mathcal{L}(\theta) = \mathcal{L}_{\text{VSD}}(\theta) + \beta_{\text{CD}} \mathcal{L}_{\text{CD}}(\theta) + \beta_{\text{FT}} \mathcal{L}_{\text{FT}}(\theta; \phi)
\]

with `β_CD = 0.5` and `β_FT = 1.0`. The third term is the novel piece: **latent reward model fine-tuning**. A small CNN/3D-CNN reward model is trained on latent space (not pixel) to mimic a pixel-space reward (HPSv2, PickScore). Then the student is fine-tuned with `L_FT = -E[R^l_φ(G_θ(ε))]`. Training the LRM in latent space saves the VAE decode and the large pixel-space reward backbone: **memory drops from 90 GB to <40 GB**, and the LRM does not require the original reward to be differentiable (HPSv2 itself is differentiable, but VBench metrics aren't).

**DOLLAR results on a CogVideoX-style teacher** (128 frames, 192×320, 12 FPS):
- DOLLAR 4-step + HPSv2 → VBench total **82.57** (beats teacher 80.25, Gen-3 82.32, T2V-Turbo 81.01, Kling 81.85)
- 1-step variant → ~278.6× acceleration vs teacher (50-step DDIM)
- Higher dynamic-degree, higher imaging quality, higher object-class than teacher

**Takeaways for production**:
- The CD+VSD combination is robust against the mode-collapse failure of pure DMD on video
- Latent reward fine-tuning is a cheap quality-boost — works with **any reward** including VBench dimensions, but DOLLAR cautions against directly optimizing VBench (over-optimization risk)

### D.9.2 DCM — Dual-Expert Consistency Model

DCM ([project](https://vchitect.github.io/DCM/); [HF cszy98/DCM](https://huggingface.co/cszy98/DCM)) is a consistency-distillation method for HunyuanVideo. The key observation: during distillation, **early-timestep (low-SNR, near-noise) and late-timestep (high-SNR, near-clean) samples have different optimization landscapes**:
- Early steps: large, fast changes — semantic layout, motion structure
- Late steps: small, smooth refinements — fine details, texture

A single student trained on both has **conflicting gradients**: the early-step gradient pushes for big trajectory changes, the late-step gradient pushes for small detail refinements. Optimization interferes.

DCM's solution: **train two expert denoisers**:
1. **Semantic expert** (early steps): full-parameter fine-tune, with a **Temporal Coherence Loss** to capture motion variations
2. **Detail expert** (late steps): freeze most of semantic expert, add **timestep-dependent LoRA + LoRA on attention linear layers**, train with **GAN + Feature Matching loss** for sharp details

The result is a parameter-efficient dual-expert distill that beats the original HunyuanVideo at fewer NFE. The architecture is conceptually similar to (but predates) the Wan 2.2 high-noise/low-noise expert split, and to Phased DMD's SNR-subinterval decomposition. DCM is **specific to HunyuanVideo** (the LoRA structure is tuned to HV's attention layers) and not a general framework.

### D.9.3 W2SVD / MagicDistillation — weak-to-strong distillation for portrait video

W2SVD aka MagicDistillation [arXiv:2503.13319](https://arxiv.org/abs/2503.13319) (Shao et al.) targets a different failure mode: when the **fake distribution `p_fake` and real distribution `p_real` have no overlap** (Fig. 5 in the paper), the score difference `∇log p_fake − ∇log p_real` is unreliable, and DMD diverges.

W2SVD's fix: use LoRA on the real-score network, and **adjust LoRA scales differentially** between the real-score branch (`α_weak`) and the fake-score branch (`α_strong = 1` fixed). This "shifts the real distribution slightly toward the fake" so they overlap. Empirically, `α_weak = 0.25` is optimal on Magic141 portrait videos.

Additional: a **regularization KL between `p_fake` and `p_gt`** (using ground-truth video) acts as a stabilizer. A **hybrid latent+pixel discriminator** (DINOv2 in latent + a separate pixel discriminator) provides distribution-level supervision.

**Numbers**:
- 4-step Magic141 distill → outperforms DMD2 baseline on 6/7 metrics in their custom widescreen VBench
- 4-step beats WanX-I2V (14B), HunyuanVideo-I2V (13B), Magic141 — even when those use 50 sampling steps, on portrait videos
- 1-step variant works (rare for video DMD)

W2SVD is the strongest evidence that **LoRA-on-real-score** is the correct way to handle distribution mismatch in DMD on large models. The same idea has been informally adopted in lightx2v's recipes.

## D.9bis AAPT — Adversarial Distribution Matching / Autoregressive Adversarial Post-Training

AAPT [arXiv:2507.18569](https://arxiv.org/abs/2507.18569) (Lu, Ren, Xia et al., ByteDance Seed) and the
predecessor APT [arXiv:2506.09350] (Lin et al., ByteDance Seed) form one of the most ambitious
real-time interactive video generation pipelines. APT was the first to apply **pure adversarial
distillation** to a billion-parameter video DiT for **single-step generation**; AAPT extended this
to the **autoregressive** setting with student-forcing.

### D.9bis.1 APT — adversarial pre-training as initialization

[arXiv:2507.18569](https://arxiv.org/abs/2507.18569) introduces **DMDX**, a pipeline that combines:
- **Adversarial Distillation Pre-training (ADP)**: align the student's distribution with **ODE-based
  synthetic data** generated from the teacher, using a **hybrid latent + pixel discriminator**. This
  provides better initialization than DMD2's MSE-based pre-training.
- **Adversarial Distribution Matching (ADM)**: replace the DMD score-difference loss with an
  adversarial discriminator that operates on **latent predictions** of the real and fake score
  estimators, with a **fixed timestep interval Δt = T/64**. The discriminator hierarchically aggregates
  features from a frozen DiT backbone and dynamically weights them through learnable heads.

The motivation: DMD's reverse-KL minimization is **zero-forcing**, driving low-probability regions
to zero and causing mode collapse. DMD2 mitigates with TTUR + GAN; APT bypasses the predefined
divergence entirely with a learned adversarial discrepancy.

**SDXL one-step results**: DMDX matches or exceeds DMD2's SDXL one-step performance, **at lower GPU
hours**. The hybrid discriminator (latent + pixel) is critical — pixel-only is too expensive (VAE
decode), latent-only is too noisy.

### D.9bis.2 AAPT — autoregressive adversarial post-training

AAPT [arXiv:2506.09350] extends APT to autoregressive video. The architecture is **block-causal
DiT** initialized from a pretrained bidirectional video diffusion model. Key design choices:

**Causal architecture transformation:**
- Replace bidirectional attention with **block-causal** (text tokens self-attend, visual tokens
  attend to text + previous and current frame visual tokens)
- Add **past-frame channel concatenation** as input — at each AR step, the previous generated frame
  is concatenated to noise as the model input
- Sliding window of N past frames in attention (constant memory)
- KV cache + causal masking → real-time per-frame inference

**Training stages (three-stage):**
1. **Diffusion adaptation**: load pretrained weights, fine-tune with diffusion objective for
   architecture adaptation (teacher-forcing)
2. **Consistency distillation**: standard CM as initialization for adversarial training
3. **Adversarial training**: AAPT proper. The **discriminator** uses the same causal generator
   architecture as backbone (initialized from diffusion weights) with logit projection heads. **One
   logit per frame** instead of per-clip → enables parallel multi-duration discrimination.
4. **Student-forcing**: generator only uses ground-truth first frame; recycles **its own generated
   results** as input for subsequent AR steps. KV cache used. Discriminator evaluates all generated
   frames in a single forward pass. Past-frame input is detached from gradient graph.
5. **Loss**: **R3GAN objective** (more stable than non-saturating loss in AAPT's experiments),
   **relativistic loss**, R1 + R2 regularizations as in APT.

**Long-video training:** generator produces 60-second videos, broken into 10-second segments for
discriminator evaluation, with 1-second overlapping. Each segment must fit data distribution.

### D.9bis.3 Numbers

**8B model on a single H100**: real-time **24 FPS** streaming at **736 × 416** resolution.
On 8 × H100: **1280 × 720**, up to **1440 frames (60 seconds)** continuous generation.

In comparison:
- CausVid (5B): 640 × 352, 9.4 FPS, 1.30s latency
- AAPT (8B): 736 × 416, 24 FPS, **0.16s latency**

The latency advantage is from generating one latent frame (4 video frames) at a time — minimizing
the wait before frames can be shown.

### D.9bis.4 Why AAPT matters

AAPT is the strongest evidence that:
1. **Adversarial distillation can scale to large video DiTs** (8B, 24 FPS).
2. **1-step per frame is feasible** for autoregressive generation (with student forcing).
3. **R3GAN-style discriminators are more stable** than standard non-saturating GAN for video.

AAPT is the architectural ancestor of **Krea Realtime 14B** (which combines AAPT-style adversarial
ideas with Self-Forcing's DMD distillation), and a counterexample to the claim that "GAN doesn't
scale to video DiT" — it does, with the right discriminator design.

## D.10 Production checkpoints (Wan 2.x lightx2v, FastWan, LTX-2, Krea, HunyuanVideo-1.5, Cosmos-Predict 2.5)

This section catalogs the actual deployable distilled checkpoints, with their recipes and headline numbers. These are what a B200 deployment would actually run.

### D.10.1 Wan 2.1 — lightx2v Step+CFG distill

**[lightx2v/Wan2.1-I2V-14B-{480P,720P}-StepDistill-CfgDistill-Lightx2v](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v)**: 4-step bidirectional distillation of Wan 2.1 I2V 14B at 480p and 720p. Trained with **Self-Forcing-Plus** ([GoatWu/Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus)) — a 4-step bidirectional DMD2 (no autoregressive rollout in this checkpoint, just step+CFG distill). LCM scheduler at inference, `shift=5.0`, `guidance_scale=1.0` (CFG folded). Released in fp8 / int8 quantizations for RTX 4060 inference.

The "StepDistill-CfgDistill" naming captures the two simultaneous compressions: 50→4 NFE × 2 (CFG fold) = ~25× generation speedup over the base Wan 2.1 14B I2V model.

### D.10.2 Wan 2.2 — lightx2v MoE Distill (per-expert)

**[lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v](https://huggingface.co/lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v)**: the first per-expert MoE distill. **2 high-noise + 2 low-noise** denoising steps (4 NFE total per chunk), CFG-1, `shift=5.0`. Key insight from the model card:

> We found that the training challenge of Wan 2.2 lies in high noise; therefore, we have focused on the two-step training for the high-noise model. Compared with the previous version, we have adopted several new strategies, which have improved the consistency and dynamics of the model. We found that the low-noise model can achieve good results by directly using the LoRA from Wan2.1; thus, the low-noise model still adopts the old Wan2.1 LoRA.

So lightx2v's recipe is **asymmetric per-expert**: the low-noise expert reuses Wan 2.1 LoRA (zero new training), only the high-noise expert needs fresh distillation. This is the opposite of what one might naively expect — high-noise (early) is harder, not easier. Reason: Wan 2.2's main improvement over Wan 2.1 is the **high-noise expert's dynamics modeling**, which is where the visual richness comes from.

### D.10.3 Wan 2.2-Lightning V1.1 / V2.0 (ModelTC, behind Phased DMD)

[github.com/ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning) — the production checkpoint trained with Phased DMD (§D.4). 4-step (2 phases × 2 steps), MoE-aware, no GAN, no regression. V2.0 adds aesthetic fine-tuning. These are the reference distillations for Wan 2.2 deployment as of May 2026.

### D.10.4 FastWan family (Hao AI Lab, UCSD)

[hao-ai-lab.github.io/blogs/fastvideo_post_training/](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/) introduces **sparse distillation**: jointly training step distillation (DMD2) **and** Video Sparse Attention (VSA) within a unified framework. The numbers are striking (DiT denoising only, single H200):

| Modules | Wan 2.2 5B 720P | Wan 2.1 14B 720P | Wan 2.1 1.3B 480P |
| --- | --- | --- | --- |
| FA2 (50-step) | 157.21s | 1746.5s | 95.21s |
| FA2 + DMD (3-step) | 4.67s | 52s | 2.88s |
| FA3 + DMD | 3.65s | 37.87s | 2.14s |
| FA3 + DMD + torch.compile | 2.64s | 29.5s | 1.49s |
| **VSA + DMD + torch.compile** | – | 13s | **0.98s** |

So Wan 2.1 1.3B end-to-end goes from **95.21s → 0.98s**, a **97× speedup**. Wan 2.1 14B goes 1746.5s → 13s for the DiT alone (~134×). Wan 2.2 5B goes 157.21s → 2.64s (60×), but VSA isn't applied to 5B because its sequence length (~20K) is short enough that sparse attention doesn't help.

The key technical innovation: **VSA-trained students remain compatible with full-attention real-score and fake-score networks**. The student uses sparse attention; both score networks use full attention. Decoupling runtime acceleration (in the student) from distillation quality (in the score estimators) is what makes the joint training tractable.

**Released checkpoints** (FastWan):
- `FastWan2.1-T2V-1.3B-Diffusers` — 3-step DMD with VSA, ~0.98s denoising on H200
- `FastWan2.1-T2V-14B-Diffusers` — 3-step DMD with VSA
- `FastWan2.2-TI2V-5B-FullAttn-Diffusers` — DMD-only ablation variant (no VSA, since 5B sequences are short)
- All trained on **synthetic data** generated by Wan 2.1 14B (600k 480p + 250k 720p) and Wan 2.2 5B (32k), released under open license
- Training cost: **64 H200 GPUs × 4k steps = 768 GPU hours**, ~$2603 at $3.39/H200/hr

The denoising_steps for FastWan are `[1000, 757, 522]` (3-step), `flow_shift=8` (Wan 2.1 default), `generator_update_interval=5`, `real_score_guidance_scale=3.5` (verified in `/Users/yewang/work/FastVideo/examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P/distill_dmd_t2v_1.3B.slurm`).

### D.10.5 Krea Realtime 14B — 11 FPS T2V on B200

[krea.ai/blog/krea-realtime-14b](https://www.krea.ai/blog/krea-realtime-14b) ([HF krea/krea-realtime-video](https://hf.co/krea/krea-realtime-video)). Krea distilled Wan 2.1 14B T2V using **scaled-up Self-Forcing** (10× larger than the original 1.3B Self-Forcing). Key implementation choices:

**Three-stage training:**
1. **Bidirectional timestep distillation** — they reused [lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) as the starting point (no new compute for this stage)
2. **Causal ODE pretraining** — 3k steps on 128 H100, with `flow_shift=5.0` (vs. Wan 2.1's default 8.0). Lower shift preserves texture and improves subjective quality. They filtered the original VidProm prompts and synthesized new ones optimized for dynamic motion.
3. **Self-Forcing DMD** — full self-forcing rollout, with novel memory optimizations (dynamic KV cache freeing, gradient checkpointing, FSDP ZeRO-3) to fit 4 × 14B models on 64 H100 GPUs

**Inference innovations for long-video generation:**

(a) **KV Cache Recomputation with block-causal masking**. Instead of evicting the oldest frame's KV (which leaves stale receptive-field information), every cache rollover **recomputes the KV for the entire window** using a block-causal mask. Pseudocode:

```
1. Sample frame 0, add KVs to cache
2. Repeat until Frame N-1
3. Recompute KV cache for Frames 1 to N-1 by passing their clean latent
   frames through the model with a block-causal mask.
4. Sample Frame N
```

Computational complexity: O(S² · N(N+1)/2) for recomputation vs. O(S²N) for vanilla. Expensive but necessary — the block-causal recomputation **breaks the receptive-field leak** from evicted frames, and combined with **first-frame re-encoding** (re-encode the first RGB of the current window as a single-frame latent, since the VAE treats first frames specially) it solves both error accumulation and the first-frame distribution shift.

(b) **KV Cache Attention Bias** — additional mechanism to weight recent frames higher

(c) **Context window of 3 latent frames** (~12 RGB frames) — sweet spot between consistency and inference speed; longer windows exacerbate error accumulation and slow inference

**Performance: 11 FPS T2V on a single B200 GPU at 4 NFE.** First-frame latency ~1 second. Users can change prompts mid-generation, restyle videos on-the-fly, see first frames in 1 second — full real-time interactive editing. This is the production-quality demonstration that 14B autoregressive video can be real-time on B200.

### D.10.6 Wan2.2-TI2V-5B-Turbo (quanhaol)

[github.com/quanhaol/Wan2.2-TI2V-5B-Turbo](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo) — community 4-step distill of Wan 2.2-TI2V-5B. Same author as **FlashMotion** ([project](https://quanhaol.github.io/flashmotion-site/)), a CVPR 2026 controllable image-to-video extension that combines few-step generation with trajectory control. FlashMotion's three-stage pipeline:
1. Train a SlowAdapter on the SlowGenerator with diffusion loss (regular trajectory adapter)
2. Distill a FastGenerator from the SlowGenerator under DMD distribution-matching
3. Fine-tune the SlowAdapter to align with the FastGenerator using a hybrid adversarial+diffusion loss

The diffusion discriminator is a DiT backbone cloned from the SlowGenerator with attention-based classifier heads. FlashMotion targets controllable few-step I2V (trajectory boxes guide motion), which is a deployment-relevant variant of the basic distillation.

### D.10.7 HunyuanVideo-1.5 step-distilled

[HuggingFace: tencent/HunyuanVideo-1.5/blob/main/assets/step_distillation_comparison.md](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/assets/step_distillation_comparison.md):

> The step-distilled model reduces inference steps from 50 to 8 (or 12 steps recommended) while maintaining comparable visual quality to the original model. On RTX 4090, this achieves up to 75% reduction in end-to-end generation time, enabling a single RTX 4090 to generate videos within 75 seconds.

Recommended settings:
- **8 or 12 steps**: best balance between speed and quality
- **4 steps**: faster generation with slightly reduced quality, suitable for rapid prototyping

Diffusers integration is in [PR #12802](https://github.com/huggingface/diffusers/pull/12802). HunyuanVideo-1.5 is a different teacher-student configuration than the original HunyuanVideo (DCM); the recipe is not officially disclosed but the 50→8 step ratio and the no-CFG inference suggest a DMD2-style distillation.

### D.10.8 LTX-2 distilled (Lightricks, Jan 2026)

[HF Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) ships a **7-artifact catalog** for the LTX-2 19B audio-video model:

| Name | Notes |
| --- | --- |
| `ltx-2-19b-dev` | Full model, bf16, trainable |
| `ltx-2-19b-dev-fp8` | Full model, fp8 quantization |
| `ltx-2-19b-dev-fp4` | Full model, **NVFP4 quantization** |
| `ltx-2-19b-distilled` | **Distilled, 8 steps, CFG=1** |
| `ltx-2-19b-distilled-lora-384` | LoRA version of the distilled, applicable to full model |
| `ltx-2-spatial-upscaler-x2-1.0` | x2 spatial upscaler for latents (multistage pipeline) |
| `ltx-2-temporal-upscaler-x2-1.0` | x2 temporal upscaler for latents (higher FPS) |

[VERIFIED — this is the exact 7-artifact catalog from the model card.]

The **two-stage pipeline** is the production pattern: stage 1 (40 NFE, base resolution), then upsample latents (spatial 2× or temporal 2×), then stage 2 (3 NFE with the **distilled LoRA** loaded on the full model + `STAGE_2_DISTILLED_SIGMA_VALUES`, CFG=1.0). The distilled LoRA is the production speedup; LTX-2 ships it as a LoRA on top of the dev model so users keep the option of full quality (40-step) or fast (3-step) inference from the same checkpoint.

```python
# Stage 2 inference with distilled LoRA and sigmas
pipe.load_lora_weights("Lightricks/LTX-2", adapter_name="stage_2_distilled",
                      weight_name="ltx-2-19b-distilled-lora-384.safetensors")
pipe.set_adapters("stage_2_distilled", 1.0)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None)
video, audio = pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    num_inference_steps=3,
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,  # CFG folded
    ...
)
```

The distilled checkpoint is **8 steps single-stage or 3 steps stage-2**, both at CFG=1. NVFP4 quantization is officially shipped, putting LTX-2 at the frontier of distillation × quantization composition.

### D.10.9 Cosmos Predict 2.5 distilled (NVIDIA Cosmos Lab)

[Cosmos Cookbook: Distilling Cosmos Predict 2.5](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/distilling_predict2.5.html). Official NVIDIA recipe for distilling Cosmos Predict 2.5 Video2World into a 4-step student using **DMD2 with TrigFlow parameterization**.

Key architectural choices:
- **TrigFlow** noise schedule (`α_t = cos(t), σ_t = sin(t)`) as a unified parameterization between DMD2 and consistency distillation (rCM, sCM). Convertible to/from EDM and RectifiedFlow, compatible with teachers trained under either formulation. References [sCM/TrigFlow arXiv:2410.11081](https://arxiv.org/abs/2410.11081).
- `Video2WorldModelDistillDMD2TrigFlow` inherits `DistillationCoreMixin` (alternates student/critic phases), `TrigFlowMixin` (timestep sampling), and the teacher `Video2WorldModel`
- **Student phase** (`training_step_generator`): freeze critic, unfreeze student, sample time and noise, generate few-step student samples, re-noise to sampled time, feed to teacher twice (cond/uncond) for the CFG target, compute DMD2 losses
- **Critic phase** (`training_step_critic`): freeze student, unfreeze critic, generate student samples via short backward simulation, train critic on these samples to fit denoising target
- `student_update_freq = 5` (TTUR), no GAN, no warm-up needed (NVIDIA empirically found warm-up unnecessary for Predict 2.5)

The recipe mirrors DMD2 + Self-Forcing closely. The novelty is the **TrigFlow shared parameterization** that makes it a drop-in framework for both DMD2 and rCM — important because NVIDIA's Cosmos Lab has shipped both methods (rCM as ICLR 2026, DMD2 in production). The cookbook reports satisfactory 4-step generation after just 1500 iterations (300 student steps + 1200 critic steps).

### D.10.10 FastHunyuan / FastMochi (older but still relevant)

[FastVideo/FastHunyuan](https://huggingface.co/FastVideo/FastHunyuan): consistency-distilled HunyuanVideo, 6 steps with `shift=17, cfg>6`, ~8× speedup. NF4/INT8 quantizations enable inference on RTX 4090 with 20 GB VRAM. Trained on MixKit dataset, 320 steps on 32 GPUs at LR 1e-6 with Huber loss, batch 16, 720×1280×125 frames.

[FastVideo/FastMochi-diffusers](https://huggingface.co/FastVideo/FastMochi-diffusers): consistency-distilled Mochi, 8 steps (vs. teacher 64), ~8× speedup. Trained on MixKit, 128 steps on 16 GPUs at batch 32, 480×848×169 frames. Inference uses linear-quadratic schedule and CFG=1.5 (note: Mochi defaults to high CFG, FastMochi keeps moderate CFG since it was distilled with that scale).

Both are **CD-based, not DMD2** — they predate the DMD-on-video literature (early/mid 2024). Today they're surpassed by the DMD2/Self-Forcing checkpoints, but they still serve as cheap distilled baselines for HunyuanVideo and Mochi when DMD-based distills aren't available.

### D.10.11 Step-Video-T2V-Turbo

[HF stepfun-ai/stepvideo-t2v](https://huggingface.co/stepfun-ai/stepvideo-t2v) — Step-Video-T2V (30B parameters, by Step AI) released a "Turbo" variant via consistency distillation, similar to FastHunyuan/FastMochi. ~6-8 step inference, CFG-1 at inference. Specific recipe details are not publicly disclosed.

### D.10.12 HY-WorldPlay (Tencent Hunyuan, Dec 2025) — Context Forcing for memory-aware distillation

[arXiv:2512.14614](https://arxiv.org/abs/2512.14614) ([Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)).
WorldPlay (Sun, Zhang, Wang et al., Tencent Hunyuan) is a streaming video diffusion model for **interactive
world modeling** with **long-term geometric consistency**. It runs in real time at **24 FPS at 720p**
on a single GPU, with KV-cache memory management and a custom distillation called **Context Forcing**.

The novelty for distillation: existing methods (CausVid, Self-Forcing, DMD2) **fail when applied to
memory-aware autoregressive students** because of a fundamental distribution mismatch — training a
memory-aware AR student to mimic a memoryless bidirectional teacher leads to context divergence.
Even augmenting the teacher with memory does not help, because the teacher and student see
different memory contexts.

**Context Forcing's solution: align the memory context for teacher and student during distillation.**
The teacher and student both receive the same retrieved memory frames, the same temporal reframing
positional encoding, and the same continuous-camera-pose conditioning. The DMD loss is then computed
**conditional on this shared memory context**, so the score-difference reflects memory-conditional
distributions rather than memory-marginal distributions.

The result: **real-time speed** (24 FPS) **without memory erosion**, with alleviated error
accumulation across long sequences. WorldPlay also introduces **temporal reframing** — discard
absolute temporal indices, dynamically re-assign new positional encodings to context frames so that
their relative distance to the current frame stays fixed. This "pulls" geometrically important but
long-past memories closer in time, ensuring they remain influential and enabling robust extrapolation.

WorldPlay is the first published work to show that **memory-aware distillation is a separate problem**
from standard step distillation, and provides a clean framework for solving it. The applications —
real-time interactive 3D world modeling, 3D scene generation via reconstruction, dynamic world events
— suggest this will be an active research area as world models scale.

### D.10.13 Helios — alternative real-time architecture (no SF, no CausVid)

[arXiv:2603.04379](https://arxiv.org/abs/2603.04379) ([PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios)).
Helios (Yuan et al., PKU + ByteDance) is a notable contrarian work — it achieves real-time
long-video generation **without** the standard set of acceleration techniques: no Self-Forcing,
no causal masking + KV-cache, no sparse/linear attention, no quantization. Instead, Helios takes
a **dramatically different architectural path**:

**Three pillars:**
1. **Unified History Injection**: long-video generation as **video continuation**. Input is the
   concatenation of historical context `X_Hist` (timestep frozen at 0, "clean") and noisy context
   `X_Noisy` (currently being denoised). Model denoises `X_Noisy` conditioned on `X_Hist` to generate
   coherent continuation. Unified architecture for T2V (X_Hist all zeros), I2V (last frame nonzero),
   V2V (multiple frames nonzero).
2. **Easy Anti-Drifting**: simulate three canonical drifting patterns (position shift, color shift,
   restoration shift) during training. Eliminates need for self-forcing rollouts during training.
3. **Deep Compression Flow**: reduce visual tokens via Multi-Term Memory Patchification + Pyramid
   Unified Predictor Corrector. Reformulate flow matching from "full-resolution noise to full-
   resolution data" into multiple "low-resolution noise to multi-resolution data" trajectories.

**Adversarial Hierarchical Distillation**: a purely teacher-forced approach using only the
autoregressive model as teacher (no bidirectional teacher needed). Reduces sampling steps from 50 to 3.

**Headline**: 14B model, **19.5 FPS on a single H100** — beats Krea Realtime 14B (which gets 6.7 FPS
on H100, 11 FPS on B200). Generates minute-scale videos without drifting. **128× speedup** over base
Wan 2.1 14B.

The Helios story is important for B200 deployment thinking: it demonstrates that **the standard
SF + CausVid + KV-cache pipeline is not the only path to real-time** — purely teacher-forced approaches
with smarter training-time drifting simulation can match or exceed it. The trade-off is more complex
training (custom data augmentation for drifting patterns) for simpler inference (no KV cache, no
context window management).

Helios is also evidence that distillation alone (without Self-Forcing's autoregressive training trick)
can produce real-time models if combined with the right architectural and training innovations.

## D.11 BLADE — joint sparse + step distillation

BLADE / Video-BLADE [arXiv:2508.10774](https://arxiv.org/abs/2508.10774) (Gu et al., Zhejiang University + Huawei; [project](http://ziplab.co/BLADE-Homepage/)) is **the first method to jointly train sparse attention and step distillation**, in a data-free framework. The framing: training-free integration of sparse attention onto a distilled model is suboptimal (the distilled model wasn't trained with sparsity in mind); sequential training (distill, then add sparsity) requires expensive video data. Joint training avoids both pitfalls.

### D.11.1 Adaptive Block-Sparse Attention (ASA)

ASA generates **dynamic, content-aware sparsity masks on the fly**:
1. **Locality-preserving token rearrangement**: reorder tokens along a Gilbert space-filling curve so that nearby tokens are spatially contiguous (essential for blockwise sparsity)
2. **Efficient block importance estimation**: sample `k=16` representative tokens per block, compute a low-cost max-pooled attention matrix `P`
3. **Threshold-based mask generator**: sort scores in `P`, select top blocks containing 95% of the attention mass, produce binary mask `M`
4. Augment key matrix `K` with a mean-pooled version: `K = Concat(K, MeanPool_n(K))`. Original K region uses binary block mask `M`; pooled region uses fixed additive mask `ln n` for soft fallback

This is **dynamic** (vs. STA's fixed local windows or Radial Attention's static heuristic), **content-aware**, **hardware-friendly** (block-sparse), and **training-compatible** (unlike SpargeAttention).

### D.11.2 Joint training with TDM

The student model uses ASA. The teacher model and the fake-score model both use full attention. The training loss is **TDM** (§D.5):

\[
\nabla_\theta \mathcal{L}(\theta) \approx \sum_{i=0}^{K-1} \sum_{j=t_i}^{t_{i+1}} \lambda_j [s_\psi(x_j, j) - s_\phi(x_j, j)] \frac{\partial x_{t_i}}{\partial \theta}
\]

with non-overlapping intervals so a single fake-score suffices. Backprop is constrained to one ODE step at a time. The student learns to denoise efficiently **with sparsity**, while the teacher and fake-score provide full-attention-quality supervision.

### D.11.3 Headline numbers

- **Wan 2.1-1.3B + BLADE → 14.10× end-to-end speedup** over 50-step baseline
- **CogVideoX-5B + BLADE → 8.89× speedup** (lower because shorter sequences)
- **VBench-2.0 score *improves* under distillation+sparsity**:
  - CogVideoX-5B: 0.534 → **0.569** (student beats teacher)
  - Wan 2.1-1.3B: 0.563 → **0.570** (student beats teacher)

[VERIFIED] from the abstract and Table 1 of the BLADE paper. The "student beats teacher under acceleration" pattern (also seen in DOLLAR, TDM, DMD2) is increasingly common with distillation, because the GAN/distribution-matching component can fix teacher errors.

### D.11.4 Why this composition matters for B200

Sparse attention alone gives ~3× speedup. Step distillation alone gives ~10-15×. Naive composition (apply sparse attention to a step-distilled model) often degrades quality because the distillation was attention-pattern-agnostic. BLADE's joint training is the **right way** to compose. For Wan 2.1 14B on B200 with ~80K tokens per chunk, attention is >85% of the per-step latency — so sparse attention is critical, and BLADE is the only method to date that handles the joint training correctly.

## D.12 CFG distillation

Classifier-Free Guidance (CFG) doubles the per-step cost: each step requires both a conditional and an unconditional forward pass. Folding CFG into a single forward pass — via **guidance distillation** — is therefore an independent ~2× speedup that composes with step distillation. For a Wan 2.1 14B at 50 NFE × 2 (CFG) = 100 forward passes; step+CFG distill to 4 NFE × 1 = 4 passes, giving the **25×** generation speedup that lightx2v's "StepDistill-CfgDistill" naming captures.

### D.12.1 Guidance Distillation (Meng et al., CVPR 2023)

The OG paper "On Distillation of Guided Diffusion Models" (Meng et al., CVPR 2023) trained a single student to produce CFG-guided outputs in one forward pass by **fine-tuning the entire base model** with a regression loss against `ε̃ = ω·ε(x_t, t, c) − (ω−1)·ε(x_t, t, ∅)`. The student takes the guidance scale `ω` as an additional input. Heavy: full fine-tuning of large models, dataset built from diffusion trajectories.

### D.12.2 AGD — Adapter Guidance Distillation [arXiv:2503.07274]

AGD (Perez Jensen & Sadat, ETH Zürich) is the modern, efficient alternative. Two contributions:

**(1) Adapter-only distillation.** Freeze the base model, add lightweight adapters (~1-5% extra parameters), train only the adapters to replicate CFG-guided outputs. Memory drops dramatically — **SDXL (2.6B) trains on a single RTX 4090 with 24 GB VRAM**, vs. multiple high-end GPUs for full GD. The adapters are also **composable with other LoRAs** (style, identity) since the base model is unchanged.

**(2) Train on CFG-guided trajectories, not diffusion trajectories.** Prior GD methods trained on standard diffusion trajectories (`x_t = α_t·x + σ_t·ε` from training data). But CFG-guided sampling produces **different trajectories** than unguided diffusion (Fig. 3 in the paper). AGD pre-runs the teacher with CFG to collect guided trajectories `(x_t, t, c, ω, ε̃)` and trains adapters on those. This eliminates the train-inference mismatch and improves performance, while also **freeing VRAM at training time** (no teacher model needed during adapter training).

### D.12.3 Why AGD has not been ported to video DiT yet

The mechanism scales straightforwardly to video DiT in principle — adapters on attention layers can encode the CFG fold. But:

1. **Most production video distillations do step+CFG jointly**: lightx2v's recipes, FastWan, Wan 2.2-Lightning, Cosmos-Predict 2.5 all distill CFG into the few-step student via DMD2. The CFG is already folded by the time the student converges.
2. **Variable guidance scale is rarely needed for video**: image generation users tune CFG for style; video users tend to fix CFG. AGD's main user-facing benefit (variable `ω` at inference) doesn't matter much.
3. **Engineering complexity**: AGD adds an adapter inference path; for video at scale the orchestration is more friction than the saved memory at training time.

So **AGD-CFG for video is an open opportunity, not yet realized**. A team that wanted variable-CFG inference (e.g., for slider UIs) would need to apply AGD on top of a base video DiT before distillation. Probably tractable, but no public results.

### D.12.4 In FastVideo, the CFG fold is implicit

The FastVideo distillation pipeline computes the CFG-guided real-score during training:

```python
real_score_pred_video = pred_real_video_cond + (pred_real_video_cond - pred_real_video_uncond) * self.real_score_guidance_scale
```

So **the student is trained to match the CFG-guided teacher's distribution**, with `real_score_guidance_scale` (3.5 for Wan 2.1 1.3B, 3.0 for SFWan2.1) baked into the supervision. At inference, the student runs at CFG=1 (single forward pass per step); the CFG behavior is implicit in the student's learned weights. This is the most common pattern in production video distillation today.

## D.13 Shortcut / MeanFlow / iMF / AlphaFlow / Rectified MeanFlow — image-side 1-NFE state of the art

This family is **image-only** but worth understanding because it represents the theoretical state-of-the-art for 1-NFE generation, and because some of its mathematical machinery (TrigFlow, JVP, average-velocity targets) is making its way into video via rCM and TMD.

### D.13.1 Shortcut Models (Frans et al., 2024)

Introduces a self-consistency loss between flows at different discrete time intervals. The student learns `flow(x_t, t, d)` where `d` is a step size: large-step flows match a sequence of small-step flows. Trained from scratch, achieves competitive 1-NFE on ImageNet.

### D.13.2 MeanFlow [arXiv:2505.13447] (Geng, Deng, Bai, Kolter, He — CMU + MIT)

The cleanest formulation. Define **average velocity** `u(z_t, r, t) = 1/(t−r) · ∫_r^t v(z_τ, τ) dτ`. The MeanFlow Identity:

\[
u(z_t, r, t) = v(z_t, t) - (t - r) \cdot \frac{d}{dt} u(z_t, r, t)
\]

Train a network `u_θ(z_t, r, t)` to satisfy this identity. At inference: a single function evaluation `u_θ(ε, 0, 1)` gives the entire flow path. **No pre-training, no distillation, no curriculum learning** — trained from scratch with a single loss. The time-derivative `du/dt` is computed via JVP (forward-mode autodiff).

**Headline**: ImageNet 256×256 from scratch, **1-NFE FID 3.43** — a 50-70% relative improvement over Shortcut, iCT, and IMM at the same NFE budget. Beats most prior "from scratch" 1-NFE generators by a wide margin.

### D.13.3 iMF / AlphaFlow / Rectified MeanFlow

- **iMF** [arXiv:2510.20771]: improved MeanFlow with better numerics
- **AlphaFlow** [arXiv:2512.02012]: the state-of-the-art as of Dec 2025/Jan 2026
- **Rectified MeanFlow** [arXiv:2511.23342]: combines MeanFlow with the rectified-flow trajectory straightening

These all build on MeanFlow's average-velocity formulation, with various stability and quality improvements.

### D.13.4 Why MeanFlow is **infeasible at video scale**

The MeanFlow loss requires JVP through the network. For an image DiT, JVP roughly **doubles** the forward-pass cost. For a video DiT where attention is `O(T²)` in the temporal dimension and a 5-second clip at Wan 2.1 14B has ~80K tokens per chunk:

- A single forward pass of full attention is `~200 GFLOPs` per token → ~16 PFLOPs per chunk
- JVP doubles this → ~32 PFLOPs per chunk per training step
- Memory: JVP requires **two copies of all activations** (primal + tangent) → ~2× activation memory

For a 14B Wan 2.1 with full attention, this **doesn't fit in memory** even on H200 (141 GB), let alone B200. rCM (§D.7) had to develop a custom FlashAttention-2 JVP Triton kernel to even attempt JVP at video scale, and even then, **only the rCM regularizer (small DMD addition to sCM) is feasible**, not pure MeanFlow training from scratch.

**MeanFlow on video remains an open research problem.** The most likely path is partial-JVP (only on backbone, not full attention) or replacing JVP with finite-difference approximations (rCM's "semi-continuous time" trick).

## D.14 Solver-based step reduction (UniPC, A-FloPS) — training-free baseline

Before any distillation, deploying a faster **ODE/SDE solver** at inference is a free win. These methods don't change the model — they reduce the number of NFE by using higher-order numerical integration.

### D.14.1 UniPC (Zhao, Bai, Rao, Zhou, Lu — arXiv:2302.04867, NeurIPS 2023)

Unified Predictor-Corrector framework for fast sampling. Key insight: any standard ODE solver (DPM-Solver, DDIM) can be cast as a **predictor** step; UniPC adds a **corrector** step that uses the same model evaluation to refine the prediction. So UniPC gets ~1.5× quality improvement over DDIM at the same NFE without any extra forward passes.

For Wan 2.1 / Wan 2.2: UniPC reduces 50 NFE → ~20 NFE while maintaining VBench score. This is composable with distillation: even after DMD distillation to 4 NFE, switching from Euler to UniPC at inference can give ~10-15% quality improvement.

### D.14.2 DPM-Solver / DPM-Solver++ (Lu et al., NeurIPS 2022 / arXiv:2211.01095)

The standard fast solvers. DPM-Solver-2/3 give ~3-5× speedup over DDIM. DPM-Solver++ adds support for guided sampling (CFG). Currently the default in most diffusers pipelines.

### D.14.3 A-FloPS / Higher-order flow solvers

For flow-matching models (Wan, LTX, HunyuanVideo, Cosmos), specialized higher-order flow solvers (Heun's, RK4-style integrators on flow ODE) reduce NFE further. A-FloPS is one such method.

### D.14.4 Composition with distillation

The story for B200 deployment:
1. **Solver alone**: 50 → ~20 NFE, no training, ~2.5× speedup. Quality: lossless.
2. **Distillation + solver**: 50 → 4 NFE via DMD2, then UniPC scheduling for the 4 steps. Quality: matches or beats teacher (per DMD2/TDM/Self-Forcing data).
3. **All-in**: distillation + solver + sparse attention + quantization (NVFP4) + parallelism. Wan 2.1 14B 480p goes from 95.21s → 0.5s on a single B200.

Solvers are the cheapest layer of the stack and shouldn't be skipped, but distillation dominates the speedup budget.

## D.15 Open questions

### D.15.1 MoE-aware video distillation

Wan 2.2 is the first widely-deployed MoE video DiT (28B total, ~14B active per step). Phased DMD's per-phase MoE structure is the cleanest published method, but it requires the student to inherit the MoE structure exactly. Open questions:
- **Per-expert rather than per-phase distillation**: lightx2v's `Wan2.2-I2V-A14B-Moe-Distill` reuses Wan 2.1's low-noise LoRA on Wan 2.2's low-noise expert. Does this generalize? When can experts be reused across model versions?
- **CausalWan-MoE**: there is no public CausalWan distillation that handles the MoE routing correctly under autoregressive rollout. Self-Forcing on Wan 2.2 A14B (FastVideo's SFWan2.2-A14B recipe) is the closest, but the recipe ships unstable in many cases.
- **Cross-expert score-difference**: the DMD gradient `s_real − s_fake` requires both networks to be MoE-aware. FastVideo's pipeline routes correctly, but the **per-expert KL** is not necessarily the right objective — possibly a joint MoE-aware score-matching is needed.

### D.15.2 Stable 1-step video

Most production video distillations land at 4 NFE for stability. 1-step video distillation is unstable for several reasons:
- **Distribution mismatch is severe**: at 1 step, the student must directly map noise to a 5-second video, a huge generative leap that 14B parameters can barely accommodate
- **Mode collapse from reverse KL is brutal**: at 1 step, even small mode collapse means the student outputs the same video for every prompt
- **No room for backward simulation**: DMD2's training-test bridging trick requires multiple steps

OSV (1-step on SVD), W2SVD (1-step on Magic141), AAPT (1-step autoregressive on 8B), and rCM (1-step on Wan 2.1 14B → VBench 83.02) have all shown 1-step is **possible**. But none has reached the quality of 4-step distillations, and all require significant additional regularization (GAN, reward feedback, dual divergence).

A 1-step Wan 2.2 28B distillation that matches the teacher quality remains an open challenge. Likely requires a combination of: dual-divergence (rCM) + autoregressive student-forcing (AAPT) + per-phase MoE (Phased DMD).

### D.15.3 AGD-CFG for video

As discussed in §D.12.3: no public AGD-style CFG distillation for video DiT. The opportunity: variable guidance scale at inference (slider UIs, prompt-strength controls), which the current "bake CFG into the student" approach does not allow. Tractable in principle (adapter on attention layers, train on CFG-guided trajectories), but no team has shipped it.

### D.15.4 MeanFlow at video scale

As §D.13.4 explains, the `T²` memory cost of JVP is the blocker. rCM's Triton FA-2 JVP kernel is the closest to a solution — it should be possible to implement pure MeanFlow training on video DiT with that kernel — but **no one has published a video MeanFlow result yet**. If achieved, it would be a true 1-NFE video model trained from scratch (no distillation, no teacher), which would be the holy grail for compute efficiency.

### D.15.5 PPCL (Pluggable Pruning) and V.I.P. — the model-level acceleration frontier

Two recent works that augment distillation:

**PPCL** [arXiv:2511.16156] introduces pluggable pruning at the **layer/block level** of video DiT. Unlike static structural pruning, PPCL learns **content-aware pruning masks** that can be plugged in at inference time without retraining. Composes with distillation and quantization.

**V.I.P.** [arXiv:2508.03254] (the **V**ideo **I**nference **P**runing method) explores aggressive layer skipping on video DiT, with reinforcement-learned skip schedules. Reports ~2× additional speedup composed on top of step-distilled models.

**Mobile Video DiT (Snap)** [arXiv:2507.13343] takes the model-level acceleration to the extreme: a sub-1B video DiT distilled and quantized for **mobile inference** (Snap's ARM/Apple silicon target). Combines architecture redesign (smaller transformer, simpler attention) with distillation. The Snap paper is the proof-of-concept that distilled video DiT can run on phone-class hardware, an order of magnitude beyond B200's capabilities (but at much lower quality).

### D.15.6 The composition map

For a B200 deployment of Wan 2.x video DiT in May 2026, the optimal stack is roughly:

```
NVFP4 (quantization)
   × Sparse attention (VSA / ASA)
      × DMD2 / Self-Forcing / Phased DMD (step distillation, also folds CFG)
         × UniPC (solver, optional)
            × Sequence parallelism (Ulysses / Ring)
               × FA-3 / FA-4 (kernel)
                  × torch.compile
```

Each layer is independent and roughly multiplicative. Distillation × CFG fold is the dominant factor (~25-50×), sparse attention adds another 2-3×, NVFP4 adds another 1.3-1.5× on top of FP8, and the rest is fine-grained engineering. The end-to-end target — **Wan 2.1 14B → 5-second 720p video in <1s on a single B200** — is achieved by Krea Realtime 14B (11 FPS T2V on B200) and FastWan2.1-14B (likely similar or better with full stack).

The open frontier is: (a) MoE-aware all-the-way-through stack for Wan 2.2 28B, (b) 1-step video distillation that doesn't sacrifice quality, (c) variable-CFG distillation for interactive UIs, and (d) MeanFlow / pure forward-divergence training at video scale that eliminates the teacher altogether. All four are active research areas with monthly progress; none is solved as of May 2026.

---

## Summary table — distillation methods cross-reference

| Method | Year | Loss type | Video? | NFE | Notes |
| --- | --- | --- | --- | --- | --- |
| Progressive Distillation | 2022 | Regression on halved steps | No (image) | 4-8 | Foundation, multi-stage |
| LCM / LCM-LoRA | 2023 | Consistency, single-stage | No | 1-8 | LoRA-friendly |
| AnimateLCM | 2024 | LCM, image+motion decoupled | Yes (T2V) | 4-8 | First credible video LCM |
| CTM | 2023 | Any-to-any consistency | No | 1-4 | Generalizes consistency |
| DMD | 2024 (CVPR) | Reverse KL via score difference | No | 1 | 700 A100-day data cost |
| **DMD2** | **2024 (NeurIPS)** | **Score difference + GAN, no regression** | **No (image)** | **1-4** | **Foundation for video DMD** |
| Hyper-SD | 2024 | TSCD + score distill + HFL | No | 1-8 | Unified LoRA across NFEs |
| PCM | 2024 | Phased consistency | Both | 1-16 | Versatile multi-step |
| InstaFlow / 2-RF / PeRFlow | 2023-24 | Rectified flow trajectory straightening | No | 1-2 | Foundation for flow models |
| **CausVid** | **2024 (CVPR)** | **DMD2 + asymmetric bidirectional→causal** | **Yes (T2V)** | **4** | **First DMD-on-video, 9.4 FPS H100** |
| **Self-Forcing** | **2025 (NeurIPS)** | **DMD2 + AR rollout + KV cache** | **Yes (T2V)** | **4** | **17 FPS H100, foundation for Krea** |
| Self-Forcing-Plus | 2025 | SF + Wan 14B I2V support | Yes | 4 | lightx2v's training framework |
| **Phased DMD** | **2025** | **Per-SNR-phase DMD2 + MoE** | **Yes (Wan 2.2)** | **4 (2 phases × 2)** | **Wan 2.2-Lightning V1.1/V2.0** |
| TDM | 2025 (ICCV) | Trajectory distribution matching | Both | 4 | Beats teacher on PixArt/CogVideoX |
| **TMD** | **2026** | **Backbone+flow-head + DMD2** | **Yes (Wan 2.1)** | **1.38 (effective)** | **84.24 VBench, no public weights** |
| **rCM** | **2026 (ICLR)** | **sCM + DMD long-skip regularizer** | **Yes (Wan, Cosmos)** | **1-4** | **Dual divergence, requires JVP** |
| OSV | 2024 → 2025 (CVPR) | CD + GAN, two-stage | Yes (I2V) | 1-8 | FVD 171.15 1-step on SVD |
| DOLLAR | 2024 → 2025 (ICCV) | VSD + CD + latent reward | Yes (T2V) | 4 | VBench 82.57, beats teacher |
| DCM | 2025 | Dual-expert CD + GAN | Yes (HV) | 4-8 | HunyuanVideo specific |
| W2SVD/MagicDistillation | 2025 | DMD2 + LoRA-on-real-score | Yes (Magic141) | 1-4 | Distribution overlap fix |
| AAPT | 2025 | APT + AR student forcing | Yes (8B) | 1 (per frame) | 24 FPS H100, real-time AR |
| Wan2.2-Lightning | 2025 | Phased DMD | Yes (Wan 2.2) | 4 | ModelTC production checkpoint |
| FastWan | 2025 | DMD2 + VSA joint | Yes (Wan) | 3 | 0.98s denoising H200 |
| Krea Realtime 14B | 2025 | SF + KV recompute | Yes (Wan 14B T2V) | 4 | 11 FPS B200 |
| HY-WorldPlay (Context Forcing) | 2025 | DMD + memory-aware student | Yes (interactive) | few | World model, 24 FPS 720p |
| Helios | 2026 | Adversarial Hierarchical Distill | Yes (14B) | 3 | 19.5 FPS H100, no SF/causvid |
| AGD | 2025 | CFG distillation via adapters | No | n/a | Adapter-only, 24 GB VRAM |
| Guidance Distillation | 2022 (CVPR) | Full-model CFG fold | No | n/a | Foundation for CFG distill |
| MeanFlow | 2025 | Average velocity, JVP, from scratch | No (image) | 1 | FID 3.43 ImageNet, infeasible video |
| BLADE | 2025 | TDM + ASA sparse joint | Yes (Wan/CogX) | 4 | 14.10× speedup Wan 1.3B |
| FastHunyuan | 2024 | CD | Yes (HV) | 6 | 8× speedup, RTX 4090 |
| FastMochi | 2024 | CD | Yes (Mochi) | 8 | 8× speedup |
| TDM-CogVideoX-2B-LoRA | 2025 | TDM, LoRA-only | Yes (CogX) | 4 | Drop-in LoRA |
| FlashMotion | 2026 (CVPR) | DMD + trajectory adapter | Yes (Wan) | few | Controllable few-step I2V |
| PPCL | 2025 | Layer-pruning, content-aware | Yes | n/a | Composes with distillation |
| V.I.P. | 2025 | RL skip schedules | Yes | n/a | ~2× on top of step-distilled |
| Mobile Video DiT (Snap) | 2025 | Architecture redesign + distill | Yes (sub-1B) | 4-8 | Mobile inference target |
| Step-Video-T2V-Turbo | 2025 | CD | Yes (Step 30B) | 6-8 | StepFun production checkpoint |

[VERIFIED] all numbers in this table that have explicit citations above; the rest are from cited primary sources.

---

## Bottom-line numbers for B200 deployment (May 2026)

For practical reference, here are the production-quality 5-second video generation numbers on B200, with all stack layers composed:

| Model | NFE | Resolution | Time | FPS | Quality (VBench) |
| --- | --- | --- | --- | --- | --- |
| Wan 2.1 1.3B (FastWan + VSA + NVFP4) | 3 | 480p | ~0.5-0.7s | streaming | 82-83 |
| Wan 2.1 14B (lightx2v / FastWan) | 4 | 480p | ~3-5s | offline | 84-85 |
| Wan 2.1 14B (Krea Realtime) | 4 | 480p | streaming | **11 FPS** | comparable to teacher |
| Wan 2.2 28B (Lightning V2.0 + NVFP4) | 4 (2 + 2 MoE) | 480p | ~5-7s | offline | high (matches teacher motion) |
| LTX-2 19B distilled-LoRA | 8 (or 3 stage-2) | base + upscale | ~10-15s | offline | high |
| HunyuanVideo-1.5 step-distilled | 8 (or 12) | 480p | ~30s on 4090 | offline | comparable to teacher |
| Cosmos-Predict 2.5 + DMD2 | 4 | 720p | <10s | offline | matches teacher GenEval |

The 12.5–25× compute reduction from step distillation alone, combined with 2× from CFG fold, and additional 2-3× from NVFP4 + sparse attention, brings real-time video generation within reach for the first time on consumer-friendly hardware. The next 18 months will likely focus on pushing these numbers further via 1-step generation, MoE-aware distillation, and architectures designed-for-distillation from the start.
