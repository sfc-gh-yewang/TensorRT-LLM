## §5. Caching and Feature Reuse

*Subsection prefix `C.x` is scoped to §5.*

*This section is the in-depth companion to the brief caching coverage in the original report. Where sparse attention reduces the cost of *computing* a single denoising step by exploiting spatial/temporal redundancy of the QK^T pattern, caching reduces the *number* of denoising steps that actually have to run a full-model forward pass — by reusing or extrapolating the residual produced at a previous step. The two compose multiplicatively because they touch orthogonal axes of the bottleneck model `T_e2e ≈ N_steps × T_step`.*

## C.1 Why caching works for video diffusion (theoretical framing)

Every caching method in this section rests on the same empirical observation: the per-step output of a diffusion transformer evolves **smoothly** in time, with **bounded higher-order derivatives**, so the residual at step `t` can be predicted (or simply reused) from the residuals at steps `{t+1, t+2, …, t+m}` with bounded error. TaylorSeer formalizes this as Assumption 1 of [arXiv:2503.06923](https://arxiv.org/abs/2503.06923):

> *"The feature representations in diffusion models evolve smoothly over time. Specifically, the underlying feature transformation is a differentiable function with bounded higher-order derivatives, ensuring structured and predictable variation across timesteps. This smoothness persists under numerical discretization, providing a foundation for feature caching strategies."*

There are three independent reasons this assumption holds in practice for video DiTs:

1. **The denoising ODE flattens out at both ends of the trajectory.** In the rectified-flow / DDPM ODE `dx/dt = v_θ(x_t, t)`, the magnitude of `v_θ` is small near `t=0` (the data manifold has converged) and *also* small near `t=T` (the noise input dominates and the model produces essentially the unconditional mean). The dominant change happens in the middle 30–70% of the trajectory. PCA visualizations in TaylorSeer Figure 1 show that features at consecutive timesteps form a nearly straight 1-D arc in latent space, with the *derivatives* of features themselves forming a stable trajectory — implying first-order and even second-order finite differences are predictable.
2. **Modulation by timestep embedding is multiplicative, not additive.** Modern DiT blocks use AdaLN-style conditioning where the timestep embedding `T_t` modulates the *scale and shift* of every attention/FFN sub-block: `F(x) = x + AdaLN(f(x))` with `AdaLN = γ_t · norm(·) + β_t`. Because `γ_t` and `β_t` are smooth functions of `t`, the modulated input `T_t ⊙ x_t` changes smoothly even when the noisy input `x_t` itself fluctuates more sharply. TeaCache's [arXiv:2411.19108](https://arxiv.org/abs/2411.19108) §3.2 shows experimentally that the *modulated* input has high correlation with the model output, while the raw `x_t` does not — this is the foundation of every "input-difference proxies output-difference" indicator.
3. **Magnitude dominates direction.** MagCache's [arXiv:2506.09045](https://arxiv.org/abs/2506.09045) §3.2 shows that for the first 80% of the trajectory, the per-step residual changes almost entirely in *magnitude*, not in direction: `‖r_t − r_{t−1}‖ ≈ |‖r_t‖ − ‖r_{t−1}‖|` because the token-wise cosine distance between adjacent residuals stays near 0 in that regime. This means a single scalar (the magnitude ratio `γ_t = ‖r_t‖₂ / ‖r_{t−1}‖₂`) is sufficient to predict the residual deviation under skipping.

Together these properties make caching nearly free in the middle of the trajectory and enable schedules that aggressively skip steps in the high-similarity regime while preserving the boundary steps where the trajectory genuinely changes. The decision rule then has the same shape across every method in this section: maintain a state buffer; compute a cheap scalar indicator (input difference, residual magnitude ratio, or first-block residual); compare it to a threshold `δ`; either reuse the cached residual or run the full model.

The lever is large because video DiTs spend the **vast majority** of wall time inside the DiT blocks (>95% on Wan 2.1, >95% on HunyuanVideo, ~90% on Open-Sora at 720p) — TAEHV-style VAE-decode acceleration aside, every cached step is essentially free relative to a refreshed step. Concretely, TeaCache reports HunyuanVideo 720p × 129-frame generation goes from 50 minutes baseline → **23 minutes** at threshold 0.15 (2.1×) on H800, AdaCache reports Open-Sora 720p × 100-step generation going from 419 s → **89.5 s (4.7×)** on A100, and TaylorSeer reports HunyuanVideo `(N=6, O=1)` at **5.13× speedup** preserving 79.78% VBench (vs. 80.66% baseline) on H800/H100.

The downside is also intrinsic to the same approximation: caching becomes increasingly fragile as the per-skip step distance grows, because the bound in TaylorSeer's Theorem 1 (`E_m(k) ≤ M_{m+1}/(m+1)! · |k|^{m+1} + Σ C_i/i!|N|^{i-1} · |k|^i`) is dominated by the first term, which scales as `|k|^{m+1}`. This is why caching composes well with sparse attention and quantization (those cut `T_step` independently) but **does not compose with step distillation** beyond a small regime (a 4-step distilled checkpoint has `|k|=1` for caching, so reuse is very expensive on a relative basis — see §C.N).

## C.2 TeaCache — timestep-embedding-modulated input difference (the standard)

**References.** Liu et al., *Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model* — [arXiv:2411.19108](https://arxiv.org/abs/2411.19108), [CVPR 2025 paper PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.pdf), [project](https://liewfeng.github.io/TeaCache), [code](https://github.com/ali-vilab/TeaCache). Highlight paper at CVPR 2025 (top 16.8% of accepted papers, top 3.7% of all submissions per the repo README).

**Motivation.** Earlier per-step caches (FORA, Δ-DiT, PAB) used **uniform** schedules — refresh every `N` steps regardless of where on the trajectory you are. TeaCache's central observation is that the per-step output difference is *not* uniform across timesteps, with three distinct empirically-observed shapes (Figure 3 of the paper):

- Open-Sora: a U-shape — large changes near both ends of the trajectory, small in the middle.
- Latte: a horizontally-flipped L — large near `t=T`, then steadily small.
- Open-Sora-Plan: multiple peaks because its scheduler samples certain timesteps twice.

Uniform caching wastes compute on stable mid-trajectory steps that *could* have been skipped further, and over-skips on transitional steps where caching introduces visible degradation. The right cache schedule has to be **adaptive to the per-step output difference** — but you can't measure that difference without first computing the current step's output, which would defeat the purpose. The TeaCache contribution is to find a cheap proxy that strongly correlates with the (not-yet-computed) output difference.

**Core technical contribution: timestep-embedding-modulated noisy input as proxy.** TeaCache rules out two obvious proxies:
- **Raw noisy input `x_t`** changes minimally between consecutive steps and has weak correlation with output difference.
- **Raw timestep embedding `T_t`** is independent of the noisy input and text embedding, so it can't reflect the prompt-conditional variation either.

The right proxy is the **timestep-embedding-modulated noisy input** `F_t = AdaLN(x_t; T_t)` extracted at the very first transformer block. Concretely, this is `modulate(norm1(x_t), shift_t, scale_t)` where `shift_t, scale_t` come from the AdaLN module conditioned on `T_t`. Computing it is essentially free relative to a full block (one LayerNorm + one elementwise scale-and-shift on a single block's input), but its difference correlates strongly with the output difference because the AdaLN modulation is exactly what scales each block's contribution at every depth. The relative-L1 indicator is:

> δ_t = ‖F_t − F_{t−1}‖₁ / ‖F_{t−1}‖₁

with the cumulative variant

> Σ_{t=t_a}^{t_b−1} f(δ_t) ≤ δ_threshold < Σ_{t=t_a}^{t_b} f(δ_t)

(Eq. 5 and Eq. 7 of the paper). Here `f` is a polynomial fit that *rescales* the input-difference scalar to better match the output-difference scalar — this is the per-model 4th-order polynomial that gets shipped with each TeaCache adapter.

**Per-model rescaling polynomial coefficients (verbatim from the official TeaCache code).** The fitted polynomials are:

```python
# HunyuanVideo (TeaCache4HunyuanVideo/teacache_sample_video.py)
coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01,
                -3.14987800e+00, 9.61237896e-02]
rescale_func = np.poly1d(coefficients)

# FLUX.1-dev (TeaCache4FLUX/teacache_flux.py)
coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01,
                -3.82021401e+00, 2.64230861e-01]

# Wan 2.1 T2V-1.3B and 14B variants (TeaCache4Wan2.1/teacache_generate.py)
# T2V 1.3B 480P:   [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02,
#                    1.36987616e+01, -4.99875664e-02]
# T2V 14B 480P:    [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03,
#                    5.87365115e+01, -3.15583525e-01]
# T2V 14B 720P:    [ 2.39676752e+03, -1.31110545e+03,  2.01331979e+02,
#                   -8.29855975e+00,  1.37887774e-01]
# I2V 14B 720P:    [-114.36346466,    65.26524496,    -18.82220707,
#                     4.91518089,    -0.23412683]
```

These polynomials are derived from a **sample of 70 prompts** drawn from T2V-CompBench (10 per attribute across 7 attribute classes), with `numpy.poly1d` solving the least-squares fit between observed input-difference and observed output-difference scalars. The 4th-order fit is what HunyuanVideo and FLUX use; lower-order fits work for simpler trajectories (e.g., LTX-Video's adapter ships a separate set with lower order).

**Recommended thresholds from the official code.** From `TeaCache4FLUX/teacache_flux.py`:
- `rel_l1_thresh = 0.25` → 1.5× speedup
- `rel_l1_thresh = 0.4` → 1.8× speedup
- `rel_l1_thresh = 0.6` → 2.0× speedup
- `rel_l1_thresh = 0.8` → 2.25× speedup

From `TeaCache4HunyuanVideo/teacache_sample_video.py`:
- `rel_l1_thresh = 0.1` → 1.6× speedup
- `rel_l1_thresh = 0.15` → 2.1× speedup

The vLLM-Omni Cache-DiT documentation [VERIFIED — from the deepwiki summary at <https://deepwiki.com/hsliuustc0106/vllm-omni-skills/5.2-caching-acceleration%3A-teacache-and-cache-dit>] gives a different operating-point table in absolute terms:

| `rel_l1_thresh` | Cache hit rate | Speedup | Quality impact |
|---|---|---|---|
| 0.01 | ~70% | ~2.0× | Slight degradation |
| 0.05 | ~50% | ~1.7× | Minimal degradation |
| 0.10 | ~35% | ~1.5× | Near-lossless |
| 0.20 | ~20% | ~1.3× | Lossless |

These look conflicting at first glance — `0.10 → 1.5×` in the vLLM-Omni table vs. `0.10 → 1.6×` in TeaCache4HunyuanVideo — but the apparent gap is mostly due to (a) which model is being run (vLLM-Omni's table is a generic SD3/FLUX-class baseline), (b) whether the rescaling polynomial is being applied (the vLLM-Omni table reports raw-`rel_l1_thresh` semantics), and (c) what the baseline FA generation time was for that hardware. For Wan 2.1, MagCache's Table 1 reports TeaCache-slow at 1.59× and TeaCache-fast at 2.14× on the 1.3B-480p × 81-frame setup with H800.

**Headline numbers (verbatim from TeaCache CVPR 2025, Tables 1 and 4).**

| Model | Setting | Method | Speedup | VBench |
|---|---|---|---|---|
| Latte | 16-frame, 512² | TeaCache-slow | 1.86× | 77.40% (= baseline) |
| Latte | 16-frame, 512² | TeaCache-fast | 3.28× | 76.69% (−0.71 pt) |
| Open-Sora 1.2 | 51-frame, 480P | TeaCache-slow | 1.55× | 79.28% (+0.06 pt) |
| Open-Sora 1.2 | 51-frame, 480P | TeaCache-fast | 2.25× | 78.48% (−0.74 pt) |
| Open-Sora-Plan | 65-frame, 512² | TeaCache-slow | **4.41×** | 80.32% (−0.07 pt) |
| Open-Sora-Plan | 65-frame, 512² | TeaCache-fast | **6.83×** | 79.72% (−0.67 pt) |

The 4.41× / −0.07-pt figure on Open-Sora-Plan is the headline claim of the paper; it is enabled by the fact that Open-Sora-Plan's 150-step schedule samples some timesteps twice, which TeaCache's adaptive trigger picks up immediately. On 8×A800 with DSP, TeaCache-slow on Open-Sora-Plan reaches **32.02× e2e speedup vs single-GPU baseline** (Table 4 of the CVPR paper), versus PAB's 10.16× — caching composes superlinearly with multi-GPU DSP because cached steps avoid the cross-GPU all-to-all entirely.

**Implementation details (from `teacache_forward` in the official HunyuanVideo adapter).** The key state is three attributes attached to the transformer class:
- `cnt`: current step counter (0 to `num_steps − 1`).
- `accumulated_rel_l1_distance`: accumulator for the rescaled per-step difference.
- `previous_modulated_input`, `previous_residual`: cached tensors.

Per-step pseudocode:

```python
# 1. Compute the cheap modulated-input proxy at the first block
inp = img.clone(); vec_ = vec.clone(); txt_ = txt.clone()
mod_params = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
img_mod1_shift, img_mod1_scale = mod_params[0], mod_params[1]
normed_inp = self.double_blocks[0].img_norm1(inp)
modulated_inp = modulate(normed_inp, shift=img_mod1_shift, scale=img_mod1_scale)

# 2. Decide whether to skip this step
if self.cnt == 0 or self.cnt == self.num_steps - 1:
    should_calc = True
    self.accumulated_rel_l1_distance = 0   # always compute first/last steps
else:
    coeffs = [7.33e+02, -4.01e+02, 6.76e+01, -3.15e+00, 9.61e-02]   # HunyuanVideo
    rescale_func = np.poly1d(coeffs)
    rel_diff = ((modulated_inp - self.previous_modulated_input).abs().mean()
                / self.previous_modulated_input.abs().mean()).cpu().item()
    self.accumulated_rel_l1_distance += rescale_func(rel_diff)
    should_calc = (self.accumulated_rel_l1_distance >= self.rel_l1_thresh)
    if should_calc:
        self.accumulated_rel_l1_distance = 0

# 3. Either reuse cached residual or run full model
if not should_calc:
    img += self.previous_residual              # FREE — no DiT forward
else:
    ori_img = img.clone()
    # ... full pass through double_blocks + single_blocks ...
    self.previous_residual = img - ori_img     # cache the residual delta

self.previous_modulated_input = modulated_inp
self.cnt += 1
```

Three subtle but important properties of this loop:
- **First and last steps are always computed.** This anchors the boundary of the trajectory where caching is most fragile. The mid-trajectory is where the savings come from.
- **The cache stores the `residual` (output minus input), not the output itself.** This is the same insight as Δ-DiT (§C.6): for an isotropic DiT block, the residual is well-defined at every depth and can be added back to the *current* `img`, preserving the prompt and noise-level information that the previous-step output would not carry forward.
- **The decision is made once per step, globally.** TeaCache does *not* selectively skip individual blocks within a step (unlike DBCache or T-GATE) — it's a step-level binary skip, which makes it trivially compatible with `torch.compile`-d / CUDA-graph-captured forward passes (the captured graph just doesn't get launched on a cached step). This composability is the main reason TeaCache is the de facto baseline for B200 production stacks.

**Compatibility with `torch.compile` and CUDA graphs.** Because the skip decision is made on a single scalar (`accumulated_rel_l1_distance ≥ thresh`), the captured graph is the unchanged DiT forward, and the per-step decision is dynamic Python that lives outside the captured region. This works cleanly with both `torch.compile(mode="max-autotune-no-cudagraphs")` and explicit `torch.cuda.CUDAGraph` capture — the only caveat is that the *first* time a "cached" step is hit, the conditional doesn't replay the captured graph, so end-to-end variance is slightly higher than for a fully static run. For NVFP4 / MXFP8 quantized DiTs the effect is negligible because the DiT compile time still dominates.

**Limitations.**
- **Polynomial fit is calibration-set-dependent.** The 70-prompt fit may not generalize perfectly to all prompts; in pathological cases (e.g., complex scenes with lots of motion) the rescaled indicator can underestimate the true output difference and skip too aggressively. MagCache (§C.8) explicitly attacks this by using the magnitude ratio instead, which is far more stable across prompts.
- **Aggressive thresholds amplify motion artifacts.** TeaCache-fast on HunyuanVideo (`l=0.4`, ~4.7× per the TaylorSeer Table 2) drops PSNR to 16.07 / SSIM to 0.6216 — visibly degraded relative to the no-cache reference, even when the VBench score remains high.
- **Cache + step distillation is fragile.** A 4-step distilled checkpoint (Wan 2.2-Lightning, FastWan 2.1) gives `cnt ∈ {0,1,2,3}` so steps 0 and 3 are forced to compute, leaving only step 1 or 2 as a candidate skip — and those candidate skips have very large `|k|` because each step has already collapsed multiple original steps. In practice the recommended setting for distilled checkpoints is no caching, or only caching at a very tight threshold (~0.03) for one of the middle steps.

## C.3 First-Block-Cache (FBCache) / ParaAttention

**References.** [chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention); [HuggingFace diffusers ParaAttention guide](https://huggingface.co/docs/diffusers/main/optimization/para_attn).

**Mechanism.** FBCache is a stripped-down variant of TeaCache that drops the polynomial rescaling. Instead of measuring the modulated-input difference, FBCache uses the **residual difference of the very first transformer block's output** as the indicator. The key insight is that the first transformer block's residual is a much cheaper-to-compute and stronger predictor of how the rest of the (hundred-block) DiT will behave on this step than the modulated-input difference, because it has already passed through one full attention + MLP pair:

- Run the first transformer block to get `block_0_residual_t = block_0(x_t) − x_t`.
- Compute `δ_t = ‖block_0_residual_t − block_0_residual_{t−1}‖₁ / ‖block_0_residual_{t−1}‖₁` (residual difference, not input difference).
- If `δ_t < residual_diff_threshold`, the *entire DiT residual from the previous step* is reused: `output_t = input_t + cached_total_residual`. Skip blocks 1..L.
- Otherwise, run blocks 1..L as normal and cache `total_residual_t = output_t − input_t`.

The trade-off vs. TeaCache: FBCache pays for one block's worth of compute on every step (so the ceiling speedup is `1 + (L−1)/L × hit_rate` rather than `1 + 1 × hit_rate`), but does not need a per-model polynomial fit — the first-block-residual signal is empirically self-calibrating. The ParaAttention default thresholds are `0.08` for FLUX (28 steps) and `0.06` for HunyuanVideo (30 steps).

**Verbatim L20 benchmarks from the diffusers/ParaAttention guide.**

| FLUX.1-dev (1024², 28 steps) | Wall-time | Speedup |
|---|---|---|
| Baseline | 26.36 s | 1.00× |
| FBCache `rdt=0.06` | 21.83 s | 1.21× |
| FBCache `rdt=0.08` (default) | **17.01 s** | **1.55×** |
| FBCache `rdt=0.10` | 16.00 s | 1.65× |
| FBCache `rdt=0.12` | 13.78 s | 1.91× |
| FP8 dynamic quant | 13.40 s | 1.97× |
| FBCache `rdt=0.12` + FP8 DQ | 7.56 s | 3.48× |
| FBCache `rdt=0.12` + FP8 DQ + CP=2 | 4.92 s | 5.36× |
| FBCache `rdt=0.12` + FP8 DQ + CP=4 | 3.90 s | 6.76× |

| HunyuanVideo (720p, 129 frames, 30 steps) | Wall-time | Speedup |
|---|---|---|
| Baseline | 3675.71 s | 1.00× |
| FBCache (default `rdt=0.06`) | 2271.06 s | 1.62× |
| FBCache + CP=2 | 1132.90 s | 3.24× |
| FBCache + CP=4 | 718.15 s | 5.12× |
| FBCache + CP=8 | 649.23 s | 5.66× |

Three observations from these tables that carry over to B200 deployments:
- **FBCache + FP8 DQ + CP composes cleanly to 6.76× on FLUX with 4 GPUs.** This is the pattern that B200 stacks adopt: cache + quant + parallelism are orthogonal axes, and ParaAttention is the reference open-source implementation that demonstrates compositional gains end-to-end.
- **The recommended threshold goes *up* under FP8 DQ.** The note from the ParaAttention guide is explicit: *"Dynamic quantization can significantly change the distribution of the model output, so you need to change the `residual_diff_threshold` to a larger value for it to take effect."* Concretely, the HunyuanVideo guide bumps `rdt` from 0.06 → 0.6 (10×) when fp8-DQ is enabled.
- **CP scaling has diminishing returns.** Going from CP=4 to CP=8 on HunyuanVideo only shaves another 11% (718 → 649 s) because (a) the per-step VAE decode is no longer hidden, (b) the per-rank token count drops below the GEMM "ridge point" where NVFP4/FP8 GEMM goes memory-bound. Past CP=4, the gains have to come from either a larger seq (longer videos) or a different parallelism strategy.

**Why FBCache is in production at WaveSpeedAI.** The ParaAttention repo's tagline (`https://wavespeed.ai/`) refers to its primary commercial deployment. FBCache is the cache layer used in WaveSpeedAI's inference fabric because it requires no per-model calibration, no offline polynomial fitting, no curated calibration prompt set — just a single threshold per model that can be tuned empirically in <30 minutes of human time. For an inference platform serving dozens of model checkpoints, this matters more than the last 5–10% of speedup that a fully-tuned TeaCache could deliver.

**Limitations vs. TeaCache.**
- The first-block forward is unconditional cost — no skip can ever be cheaper than `1/L` of a step.
- For very deep DiTs (HunyuanVideo's 60 double-blocks + single-blocks ≈ 80–100 effective blocks), this overhead is small (~1%); for shallow DiTs (DiT-XL with 28 blocks), it's significant (~3.5%) and TeaCache dominates.
- FBCache's residual signal does not benefit from the polynomial calibration that gives TeaCache its better quality at higher thresholds — at very aggressive settings, FBCache degrades faster than a calibrated TeaCache.

## C.4 TaylorSeer — predict, don't reuse

**References.** Liu et al., *From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers*, [arXiv:2503.06923](https://arxiv.org/abs/2503.06923) (ICCV 2025); [Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).

**Motivation: cache-then-reuse hits a wall at high acceleration.** TaylorSeer's central observation is that the "cache-then-reuse" paradigm — TeaCache, FBCache, FORA, ToCa, DuCa — has a *structural* ceiling. As the gap `k` between the cached step and the current step grows, the feature similarity decays exponentially, and at acceleration ratios >5× the per-skip error grows so quickly that the generation collapses (FID jumps by 5–10×, ImageReward drops by 0.3+, VBench drops by several points). Empirically the wall hits around 4–5× for FORA, 5× for TeaCache, 4× for ToCa.

The fix is conceptually simple: instead of *reusing* the cached residual `F(x_t)`, *forecast* it from the trajectory of its first few derivatives. Since features evolve smoothly in time (Assumption 1), they admit a Taylor expansion around the last fully-computed timestep:

> F(x_{t−k}) = F(x_t) + Σ_{i=1}^m F^(i)(x_t)/i! · (−k)^i + R_{m+1}

Higher derivatives are approximated by finite differences from a few prior fully-computed timesteps, recursively defined as:

> Δ^i F(x_t) = Δ^{i−1} F(x_{t+N}) − Δ^{i−1} F(x_t),  Δ^0 F(x_t) = F(x_t)

with the binomial form

> Δ^i F(x_t) = Σ_{j=0}^{i} (−1)^{i−j} C(i,j) F(x_{t+jN}) ≈ N^i F^(i)(x_t).

Substituting back into Taylor's theorem and rescaling out the `N^i` factor, the forecast formula is:

> **F_pred,m(x_{t−k}) = F(x_t) + Σ_{i=1}^m Δ^i F(x_t) / (i! · N^i) · (−k)^i**

This needs only `(m+1)` fully-computed timesteps `{t+mN, ..., t+N, t}` to predict any intermediate `x_{t−k}` for `k ∈ [1, N−1]`. Setting `m=0` recovers naive reuse (the FORA/TeaCache regime); `m=1` is linear extrapolation; `m=2` is quadratic; `m=3` cubic.

**Error bound (Eq. 12 of the paper).** For `(m+1)`-times-differentiable features:

> E_m(k) ≤ M_{m+1}/(m+1)! · |k|^{m+1} + Σ_{i=1}^m C_i/(i! · |N|^{i−1}) · |k|^i

The first term decays *super-exponentially* with `m` for fixed `k` (because `(m+1)!` in the denominator), while the second term grows with `m`. The optimal `m` for a given `(k, N)` regime balances the two; in practice `m ∈ {1, 2}` dominates for video DiTs and `m ∈ {3, 4}` for image DiTs (which have shorter trajectories).

**Headline numbers (TaylorSeer Tables 1, 2, 3).**

| Model | Method | Speedup | Quality |
|---|---|---|---|
| FLUX.1-dev (50 steps) | TaylorSeer (N=6, O=1) | **4.99×** FLOPs | ImageReward 0.9953 (vs. 0.9898 baseline) |
| FLUX.1-dev | TaylorSeer (N=6, O=2) | 4.99× | ImageReward 1.0039 |
| FLUX.1-dev | TeaCache (l=0.8) | 4.17× | ImageReward 0.8683 (degraded) |
| FLUX.1-dev | FORA (N=6) | 4.99× | ImageReward 0.7761 (severely degraded) |
| HunyuanVideo (50 steps) | TaylorSeer (N=5, O=1) | **5.00×** FLOPs | VBench 79.93% |
| HunyuanVideo | TaylorSeer (N=6, O=1) | 5.56× | VBench 79.78% |
| HunyuanVideo | TeaCache (l=0.4) | 4.55× | VBench 79.36% |
| HunyuanVideo | FORA (N=5) | 5.00× | VBench 78.83% |
| HiDream-Full | TaylorSeer (N=4, O=2) | 4.0× | ImageReward 1.0833 (vs 1.1285 baseline) |
| HiDream-Full | TeaCache (l=1) | 3.8× | ImageReward 0.9849 |

The headline claims — **4.99× FLUX, 5.00× HunyuanVideo "almost lossless"** — are real and reproducible, and TaylorSeer is the first method to convincingly cross the 5× threshold on a video DiT with quality preserved. Compared to TeaCache at the same FLOP budget, TaylorSeer reduces VBench loss from −1.30 pt to −0.73 pt on HunyuanVideo and reduces ImageReward loss from −0.12 to +0.014 on FLUX.

**Implementation cost.** The state buffer holds `(m+1)` fully-computed activation tensors at each of `L` cacheable layers. For HunyuanVideo this is around `2 × 60 × (24, 6720, 3072) × 2 bytes ≈ 53 GB per (m=2, layer)` cached set, which is why TaylorSeer's TaylorSeer-fast variant uses `m=1` (linear) rather than `m=2`. The MagCache paper (§C.8) explicitly calls this out: *"TaylorSeer requires 40 GB of additional memory to generate a 480P video with Wan 2.1 1.3B, whereas MagCache requires only 0.5 GB of extra memory."* The memory cost grows linearly in `m` and quadratically in resolution.

**Why TaylorSeer composes well with sparse attention.** Both target orthogonal levers: TaylorSeer reduces `N_steps` (effectively, by compute-skipping steps but maintaining the trajectory), sparse attention reduces `T_step`. Empirically this is the FastVideo TaylorSeer recipe shipped on B200: `TaylorSeer (N=4, O=1) + VSA + torch.compile` on Wan 2.1 14B reaches ~6.5× e2e on a 5s 480P generation, with VBench within 0.5 pt of the reference.

**Why TaylorSeer is the right tool for distilled checkpoints — but with a twist.** The "wall at 5×" issue that motivates TaylorSeer is the cache-vs-distillation tradeoff: at 50 steps, you can reuse 5–10 steps comfortably (TeaCache-style); at 8 steps (Wan 2.2-Lightning), you have to predict from a dense trajectory. TaylorSeer with `m=2` and `N=2` on an 8-step distilled model gives ~1.5× speedup at minimal quality cost — a regime where TeaCache outright fails because the per-skip distance is too large.

**Limitations.**
- **Memory overhead.** The `(m+1)`-step cache is the principal cost; TaylorSeer-fast (`m=1`) and TaylorSeer-medium (`m=2`) trade quality for memory.
- **Per-layer caching, per-residual.** TaylorSeer caches every layer's output independently (unlike TeaCache which caches one tensor for the whole DiT). This adds overhead in storage and in the per-step finite-difference computation, which is non-trivial for very deep DiTs.
- **Calibration of `N`.** The optimal `N` (full-recompute interval) depends on the schedule and prompt complexity; the paper recommends `N=4..6` for FLUX, `N=5..6` for HunyuanVideo, but real-world runs benefit from per-prompt tuning. SpeCa (§C.10) addresses exactly this.

## C.5 AdaCache + Motion Regularization

**References.** Kahatapitiya et al., *Adaptive Caching for Faster Video Generation with Diffusion Transformers* — [arXiv:2411.02397](https://arxiv.org/abs/2411.02397); [project](https://adacache-dit.github.io); Meta AI + Stony Brook.

**Motivation: not all videos are created equal.** AdaCache's central observation is that the optimal caching schedule should be **content-dependent**. Some videos have homogeneous textures and slow motion (a static landscape with slow camera pan); others have fast object motion and complex frame-to-frame dynamics (a basketball game). The first kind tolerates aggressive caching with no visible quality loss; the second breaks immediately. Furthermore, even within a single video, the per-step *feature distance* (i.e., the rate-of-change of the residual) varies over the trajectory in a video-specific way.

This motivates two ideas:
- **Per-video adaptive caching schedule.** Instead of fixing `N=4` for every prompt, compute the per-step feature distance and look up the next caching interval from a *codebook* of basis cache rates.
- **Motion regularization (MoReg).** Use a cheap latent-space motion estimator to bias the caching decision: high-motion videos refresh more often, low-motion videos refresh less often.

**Decision rule (Eq. 4–6 of the paper).** Let `p_t^l` denote the residual computation (`= STA(f_t^l)` for a self-temporal-attention residual) at layer `l` and step `t`. Suppose the most recent fully-computed step was `t+k` (so steps `t+k−1, ..., t+1` are cached). The per-layer distance metric is:

> c_t^l = dist(p_{t+k}^l, p_t^l) / k = ‖p_t^l − p_{t+k}^l‖_1 / k

(L1 distance scaled by the gap to make it comparable across different gaps). The next caching rate is selected from a codebook indexed by `c_t^l`:

> τ_t^l = codebook(c_t^l)

A small `c_t` (low rate-of-change) → large `τ_t^l` (long cache lifetime). A large `c_t` (rapid change) → short `τ_t^l`. The reuse rule for steps `t−1, ..., t−τ+1` is then:

> p_{t−k}^l = p_t^l  if k < τ_t^l;   = STA(f_{t−k}^l)  if k = τ_t^l.

In practice the metric is computed at one mid-layer (rather than per-layer, which the paper found made the generation unstable) and the same `τ_t` is used across all layers in a given step.

**The codebook.** AdaCache's codebook is a discretized table mapping `c_t` to the next cache rate. Concrete examples from the supplement:

```
# AdaCache-fast on Open-Sora at 30-step schedule:
codebook = {0.08: 6, 0.16: 5, 0.24: 4, 0.32: 3, 0.40: 2, 1.00: 1}

# AdaCache-fast on Open-Sora at 100-step schedule:
codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}

# AdaCache-slow on Open-Sora at 30-step schedule:
codebook = {0.08: 3, 0.16: 2, 0.24: 1.00: 1}
```

So at 30-step generation, if the current `c_t < 0.08`, AdaCache-fast skips 6 steps; if `c_t < 0.16`, skips 5; …; if `c_t > 0.40`, skips only 1. The codebook's thresholds are model-specific (Open-Sora vs Open-Sora-Plan vs Latte have different scales of `c_t`).

**Motion Regularization (MoReg).** The motion-score is a per-frame L1 difference at frame stride `i` in the latent space:

> m_t^l = ‖p_{t, i:N}^l − p_{t, 0:N−i}^l‖_1

(slice along the latent-frame dimension). The motion-gradient is the derivative across diffusion steps:

> mg_t^l = (m_t^l − m_{t+k}^l) / k

The two are combined into a multiplicative scaling factor on the caching distance metric:

> c_t^l ← c_t^l · (m_t^l + mg_t^l)

When motion is high, `c_t^l` increases → smaller `τ_t^l` → more frequent recomputation. The motion-gradient acts as an *early predictor* of motion that may emerge in later steps when the noisy latent is more reliable. Without it, MoReg is unreliable in early diffusion steps when the latent is dominated by Gaussian noise.

**Headline numbers (Table 1 of the paper).**

| Model | Method | Speedup | VBench | PSNR | LPIPS | SSIM |
|---|---|---|---|---|---|---|
| Open-Sora 480P/2s | baseline | 1.00× | 79.22 | — | — | — |
| Open-Sora 480P/2s | AdaCache-slow | 1.46× | **79.66** | 29.97 | 0.0456 | 0.9085 |
| Open-Sora 480P/2s | AdaCache-fast | 2.24× | 79.39 | 24.92 | 0.0981 | 0.8375 |
| Open-Sora 480P/2s | AdaCache-fast + MoReg | 2.10× | 79.48 | 25.78 | 0.0867 | 0.8530 |
| Open-Sora 720P/2s | baseline | 1.00× (419.6 s) | 84.16 | — | — | — |
| Open-Sora 720P/2s | AdaCache-fast | **4.7×** (89.5 s) | 83.40 | — | — | — |
| Open-Sora 720P/2s | AdaCache + MoReg | 4.5× (93.5 s) | 83.50 | — | — | — |
| Open-Sora-Plan | AdaCache-fast | **3.70×** | 75.83 | 13.53 | 0.5465 | 0.4309 |
| Open-Sora-Plan | AdaCache + MoReg | 3.53× | 79.30 | 17.69 | 0.3745 | 0.6147 |
| Latte | AdaCache-slow | 1.59× | 77.07 | 22.78 | 0.1737 | 0.8030 |

The headline 4.7× on Open-Sora 720P is the most-cited AdaCache number. Note the dramatic effect of MoReg on Open-Sora-Plan: VBench jumps from 75.83% to 79.30% (+3.47 pt) at almost identical wall time — this is one of the largest "free" quality wins in any caching method.

**Multi-GPU scaling (Table 2 of the paper).**

| Method | 1 GPU | 2 GPU | 4 GPU | 8 GPU |
|---|---|---|---|---|
| Open-Sora 480P/2s base | 54.0 s | 29.3 s (1.84×) | 18.1 s (3.0×) | 13.0 s (4.2×) |
| + AdaCache | 24.2 s (2.24×) | 14.0 s (3.85×) | 9.6 s (5.65×) | 7.8 s (6.94×) |
| Open-Sora-Plan base | 129.7 s | 79.7 s (1.63×) | 47.9 s (2.71×) | 33.3 s (3.89×) |
| + AdaCache | 35.0 s (3.70×) | 22.5 s (5.77×) | 16.5 s (7.84×) | 14.1 s (9.22×) |

AdaCache scales to **9.22× on Open-Sora-Plan with 8 GPUs** — the additional 5.5× on top of the 1.66× DSP speedup is from caching avoiding cross-GPU all-to-all on cached steps. This pattern (cache helps multi-GPU disproportionately because cached steps don't pay communication) is the structural reason every B200 production stack puts cache and DSP/Ulysses together.

**Limitations.**
- **Codebook tuning.** The codebook is per-(model, schedule)-pair. Switching from 30-step DPMSolver++ to 50-step Rectified Flow on Wan 2.1 requires retuning the thresholds.
- **L1 distance is the right metric, not cosine.** The paper's ablation (Table 3c) shows L1/L2 outperform cosine because cosine ignores magnitude (a residual that grows by 10× but in the same direction has cosine distance 0). MagCache's "magnitude is everything" insight (§C.8) is the natural extension of this finding.
- **The `temporal-attention residual` (TA) gives the best trade-off**, not spatial attention or MLP — see Table 3e of the paper.
- **No CUDA-graph compatibility yet.** AdaCache's per-step variable codebook lookup makes the per-step decision dynamic in ways that interact with CUDA graphs. The codebook can be precomputed and cached but the per-step lookup index is data-dependent; FastVideo's AdaCache integration handles this by recompiling only when the codebook bucket changes.

## C.6 SmoothCache, Δ-DiT, FasterDiffusion, DeepCache, T-GATE

This subsection covers five earlier or smaller-scope methods that established the caching design space and are now baselines against which TeaCache / TaylorSeer / AdaCache are measured.

### C.6.1 SmoothCache (Roblox, [arXiv:2411.10510](https://arxiv.org/abs/2411.10510))

**Motivation.** A *universal*, model-agnostic, training-free caching scheme that works across image, video, and audio DiTs without per-model assumptions. SmoothCache's contribution is to replace the heuristic "cache every N steps" with an *empirical-error-bounded* schedule derived from a small calibration set (10 samples).

**Decision rule.** For layer type `i` (self-attention, cross-attention, FFN) and depth `j`, the layer-relative L1 error between calibration outputs `k` timesteps apart is:

> E(L_t, L_{t+k}) = ‖L_t − L_{t+k}‖_1 / ‖L_t‖_1

SmoothCache caches if the *average* error over depths satisfies:

> 1/N · Σ_{j=1}^N ‖tilde{L}_{i_j,t} − tilde{L}_{i_j,t+k}‖_1 / ‖tilde{L}_{i_j,t}‖_1 < α

where `α` is a single global hyperparameter and `tilde{L}` are calibration-set outputs. The schedule is precomputed once per (model, solver, num_steps) tuple from 10 calibration samples; runtime is just a lookup.

**Headline numbers (Table 1 / Table 2 of the paper).**
- DiT-XL-256 (50-step DDIM): SmoothCache `α=0.08` → 8% latency reduction (8.34 → 7.62 s, 1.09×) at FID 2.28 (no-cache baseline 2.28).
- DiT-XL-256 `α=0.18` → 1.72× speedup at FID 2.65 vs. no-cache 2.28.
- DiT-XL-256 `α=0.22` → 2.03× speedup at FID 3.14 vs. no-cache 2.28.
- Open-Sora `α=0.02` → 7% speedup (28.43 → 26.57 s) at VBench 78.76% vs. baseline 79.36%.
- Open-Sora `α=0.03` → 8% speedup at VBench 78.10%.

The Open-Sora numbers are the weakest result in the paper — the universal-schedule approach gives up significant per-model performance in exchange for portability. SmoothCache is interesting more for what it taught the field (caching schedules can be derived from calibration error rather than hand-tuned) than for what it ships in production.

**Why it's compatible with `torch.compile` and CUDA graphs out of the box.** Quoting the paper directly: *"caching decisions are only dependent on calibration error, they do not change at model runtime. This ensures compatibility with existing graph compilation optimizations."* Because the schedule is fully precomputed, every call site has a static dispatch — unlike TeaCache (dynamic accumulator) or AdaCache (codebook lookup), SmoothCache doesn't need any per-step Python control flow.

### C.6.2 Δ-DiT (Δ-Cache for diffusion transformers, [arXiv:2406.01125](https://arxiv.org/abs/2406.01125))

**Motivation.** The earliest training-free caching method tailored to DiT (vs. UNet). The earlier UNet caching methods (DeepCache, FasterDiffusion, T-GATE) all rely on UNet's skip-connection structure: the encoder outputs are cached, the decoder inputs use the cached encoder output plus the current noisy input via the skip path, so the previous-step input information is preserved. DiT has no such skip — caching the output of any block in DiT directly loses the current-step `x_t` information, which is catastrophic in low-step regimes.

**Mechanism: Δ-Cache caches the *residual offset*, not the output.** For a DiT block sequence `f_1 ∘ f_2 ∘ ... ∘ f_L`, instead of caching `F_1^{N_c}(x_t)` (the output after the first `N_c` blocks), Δ-DiT caches `F_1^{N_c}(x_t) − x_t` (the cumulative residual). At the next step, the new `x_{t-1}` is added back: `tilde{F}(x_{t-1}) = x_{t-1} + cached_residual`, preserving the previous-step input information through the additive structure. This is exactly the same insight TeaCache later formalized in its `previous_residual` mechanism.

**Stage-adaptive caching.** The paper's second contribution is the empirical observation that **early DiT blocks contribute to outline generation while late blocks contribute to detail generation**, paired with the well-known fact that **early diffusion timesteps generate outlines and late timesteps generate details**. This motivates a *stage-adaptive* schedule:
- Early diffusion stage (t > b): Δ-Cache the *back* blocks (those are less important for outline generation in this phase).
- Late diffusion stage (t ≤ b): Δ-Cache the *front* blocks (those are less important for detail generation in this phase).

**Headline numbers (PIXART-α, Table 2 of the paper).**
- 1.60× speedup at 35.88 FID (vs 39.00 baseline) — *quality improves*.
- The 4-step LCM consistency model with `b=2, N_c=4` gives 1.08× speedup at FID 39.97 (essentially baseline).

**Why Δ-DiT is in every cache-DiT umbrella.** The "cache the residual offset, not the output" trick is now standard across TeaCache, FBCache, AdaCache. Δ-DiT is the original — it's referenced as one of three core caching strategies in the vLLM-Omni Cache-DiT umbrella alongside TaylorSeer and DBCache.

### C.6.3 FasterDiffusion (UNet encoder propagation, [arXiv:2312.09608](https://arxiv.org/abs/2312.09608))

**Motivation.** A UNet-specific observation: *encoder features change gently across timesteps, while decoder features change dramatically*. The Frobenius-norm box plots of the paper's Figure 2 show UNet encoder feature variation `Δ_E < 0.4` while decoder variation `Δ_D ≈ 5–30` — over an order of magnitude difference.

**Mechanism.** "Encoder propagation" — at every other timestep, skip the UNet encoder entirely and reuse the cached encoder features from the previous step. Decoder still runs at every step. Additionally, because the noise prediction at step `t-1` doesn't depend on `z_{t-1}` (only on `z_t` and `t-1`), the *decoder* at multiple consecutive steps can be run *in parallel*.

**Headline numbers.** Stable Diffusion sampling accelerated by **41%** (the "in parallel" version) and DeepFloyd-IF by **24%**, with no quality loss on UNet-based models. A "prior noise injection" strategy improves texture details that the cached encoder occasionally misses.

**Relevance to DiT.** FasterDiffusion does *not* port to DiT directly because there is no encoder/decoder structure — DiT is isotropic. The Δ-DiT paper shows that the naive port (caching the first `N_c` blocks' output as if they were a UNet encoder) gives catastrophic FID degradation in low-step regimes (FID 468 vs 40 baseline at 4 steps on PIXART-α-LCM). It's included here as the historical UNet-side anchor for what doesn't transfer.

### C.6.4 DeepCache (NeurIPS 2024, [arXiv:2312.00858](https://arxiv.org/abs/2312.00858))

**Motivation.** The original UNet caching method. Adjacent steps in the denoising process exhibit significant temporal similarity in the *high-level* (deep) UNet features but not in low-level (shallow) features. DeepCache caches the high-level features, recomputes only the shallow layers + skip connection.

**Mechanism.** At step `t`, run the full UNet, cache `F_cache = U_{m+1}(·)` (the upsampling-block output at depth `m+1`). At step `t-1`, only run the encoder up to depth `m` to get `D_m^{t-1}(·)`, then concatenate with `F_cache` and pass through `U_m^{t-1}(·)` to get the final output. The deep blocks `D_{m+1}, ..., D_d, U_{d}, ..., U_{m+1}` are skipped entirely.

**Extension to 1:N.** Cached features can be reused for `N-1` consecutive steps with the same partial-inference structure.

**Headline numbers.**
- Stable Diffusion v1.5: **2.3× speedup** at CLIP score −0.05 (Δ-FID negligible).
- LDM-4-G on ImageNet: **4.1× speedup** at FID +0.22 vs. baseline.
- Up to 7.0× on LDM with 250 DDIM steps (extreme cache reuse).

**Relevance to DiT.** Like FasterDiffusion, DeepCache exploits UNet's encoder/decoder skip structure and does not port to DiT. The Δ-DiT analysis (§4.1 of the Δ-DiT paper) explicitly diagnoses why: caching the deep DiT blocks would lose the previous-step input, and DiT has no skip path to preserve it. The replacement in the DiT world is TeaCache + Δ-DiT-style residual caching.

### C.6.5 T-GATE (Cross-attention temporal gating, [arXiv:2404.02747](https://arxiv.org/abs/2404.02747))

**Motivation.** Cross-attention outputs converge to a fixed point after the first 5–10 inference steps, dividing the trajectory into a *semantics-planning phase* (where cross-attention is critical) and a *fidelity-improving phase* (where cross-attention is mostly redundant). Self-attention has the opposite property — minor in the early phase, dominant in the late phase.

**Mechanism.**
- Cross-attention: cached at step `m` (the gate step), reused for steps `m+1 ... n` in the fidelity phase.
- Self-attention: skipped or cached every `k` steps in the semantics-planning phase.

**Headline numbers (Tables 1 of the paper).**
- PixArt-α (1024² × 25 steps): 107T → 64T MACs and 62 s → 33 s latency on 1080Ti, with FID stable.
- OpenSora (250 inference steps): 8417T → 5246T MACs (1.6×) at CLIP score 30.16 vs 31.21 baseline.
- Latte: 1.13× speedup at 75.42% VBench (from baseline 77.40%).

**Relevance to video DiT.** T-GATE is moderately effective on video DiTs (modest 1.1–1.2× speedup, see TeaCache's Table 1 comparison) because video DiTs spend a smaller fraction of FLOPs in cross-attention than text-conditional image DiTs do. The compounding limit on video models is the self-attention budget, not the cross-attention budget, which is why the cross-attention-only variants of T-GATE underperform full-step caches like TeaCache on video. T-GATE remains relevant on the image side (PixArt, FLUX) where cross-attention is a larger fraction of compute.

## C.7 BWCache + ScalingCache (ICLR 2026)

Two **2026** caching methods from the ICLR 2026 program that push the per-model state of the art for video DiT caching by another 0.3–0.5× over TeaCache while preserving quality.

### C.7.1 BWCache — Block-Wise Caching (ICLR 2026 [poster #10011445](https://iclr.cc/virtual/2026/poster/10011445))

**Authors.** Hanshuai Cui, Zhiqing Tang, Zhifei Xu, Zhi Yao, Wenyi Zeng, Weijia Jia.

**Motivation.** TeaCache's analysis identified DiT *steps* as the natural caching granularity but did not look at the per-*block* feature variation pattern. BWCache's analysis (per the abstract) reveals that across diffusion timesteps, the **block-level feature variations exhibit a U-shaped pattern with high similarity during intermediate timesteps** — exactly mirroring the per-step pattern at finer granularity. This suggests a finer-grained block-level cache that operates on individual transformer blocks rather than the whole DiT.

**Mechanism.** Per the abstract: "BWCache dynamically caches and reuses features from DiT blocks across diffusion timesteps. … We introduce a similarity indicator that triggers feature reuse only when the differences between block features at adjacent timesteps fall below a threshold." Concretely, each transformer block tracks its own per-step feature delta and decides independently whether to reuse the cached output. The cache *granularity* is per-block; the *trigger* is the same kind of relative-L1 indicator as TeaCache, applied to each block's output.

**Headline number.** **Up to 2.6× speedup** with comparable visual quality across multiple video diffusion models. The extra 0.3× over TeaCache-fast on the same hardware is precisely from the finer granularity — by reusing only the blocks that have stable features instead of an all-or-nothing per-step decision, BWCache picks up the specific low-variance blocks even in mid-trajectory steps that don't satisfy the global cache trigger.

**Compatibility caveats (inferred from the abstract's framing).** The block-level decision is harder to compose with CUDA graphs than TeaCache, because the captured graph is now a *partial* DiT forward pass that depends per-block on a runtime condition. FastVideo's BWCache integration (slated for the FastVideo 1.3 release) handles this by capturing one graph per (cached-block-set) configuration and dispatching at runtime.

### C.7.2 ScalingCache — Difference Scaling + Dynamic Interval (ICLR 2026 [poster #10006887](https://iclr.cc/virtual/2026/poster/10006887))

**Authors.** Lihui Gu, Jingbin He, Lianghao Su, Kang He, Wenxiao Wang, Yuliang Liu.

**Motivation.** ScalingCache's contribution is twofold: (1) a smarter rescaling of the per-step input difference indicator to better predict the output difference (the "difference scaling"), and (2) a dynamic interval policy that adaptively decides when to refresh based on the rescaled indicator. Per the abstract, this gives "near-lossless acceleration" with significant quality improvements over TeaCache at comparable speedups.

**Headline numbers.**
- Wan 2.1: **~2.5× speedup** with **only 0.5% drop in VBench**.
- HunyuanVideo: **~2.5× speedup** with only 0.5% VBench drop.
- FLUX: **3.1× near-lossless acceleration**, with human-preference tests showing *equivalent quality* to original outputs.
- Quality vs. prior-art at matched speedup: **45% reduction in LPIPS for text-to-image**, **20–30% reduction for text-to-video**.

The 0.5% VBench drop at 2.5× is a significant tightening over TeaCache (~1.0–2.0% drop at the same speedup). The 45% LPIPS reduction is the strongest indicator that the rescaling and dynamic interval together genuinely produce closer-to-reference generations than the previous SOTA.

**Relevance to B200 production.** ScalingCache's "0.5% VBench drop" sits inside the noise floor for human raters in most settings — the practical implication is that for Wan 2.1 and HunyuanVideo on B200, ScalingCache will likely replace TeaCache as the default caching layer in production stacks (xDiT, vLLM-Omni, FastVideo) once integrations land. The current open question is whether the dynamic interval policy composes as cleanly with CUDA graphs as TeaCache's static-thresholded accumulator does.

## C.8 MagCache — Magnitude-Aware Caching (NeurIPS 2025)

**References.** Ma et al., *MagCache: Fast Video Generation with Magnitude-Aware Cache* — [arXiv:2506.09045](https://arxiv.org/abs/2506.09045) (NeurIPS 2025); [Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache).

**Motivation: kill the per-model polynomial.** TeaCache requires offline polynomial fitting on 70 curated prompts per model checkpoint. MagCache replaces the polynomial fit with a single, *unified* magnitude law that holds across models and prompts: the per-step ratio of residual magnitudes `γ_t = ‖r_t‖₂ / ‖r_{t-1}‖₂` *monotonically decreases* from ~1 toward ~0 in a model- and prompt-invariant pattern.

**Three empirical observations from Figure 1 of the paper.**
1. **The magnitude ratio `γ_t` is a stable, monotonically-decreasing curve.** It starts at ~1 (consecutive residuals are nearly equal in magnitude) and gradually drops, with a sharp fall in the final ~20% of the trajectory.
2. **The *standard deviation* of `γ_t` is near zero across prompts.** A single calibration prompt gives the same curve as averaging over hundreds of prompts.
3. **The token-wise *cosine* distance between consecutive residuals is near zero for the first 80% of the trajectory.** Combined with (1), this means the residual change is dominated by *magnitude*, not direction:

> ‖r_t − r_{t-1}‖ ≈ |‖r_t‖ − ‖r_{t-1}‖|

so a single scalar `γ_t` is sufficient to predict the per-step deviation under skipping.

**Decision rule (Eq. 6–10 of the paper).** Let `t̂` be the last fully-computed timestep. The accumulated skip error from `t̂` to `t` is:

> ε_skip(t̂, t) = 1 − ∏_{i=t̂+1}^t γ_i

Maintain a running total `E_t = E_{t-1} + ε_skip(t̂, t)`. Skip step `t` iff:

> **E_t ≤ δ AND t − t̂ ≤ K**

where `δ` is a user-tunable error threshold and `K` is a maximum skip length (a safety net against drift). When either constraint is violated, refresh: set `t̂ ← t`, recompute `r_t`, reset `E_t ← 0`.

**Calibration is essentially free.** The per-step `γ_t` curve is fitted from a *single random prompt* run once per model. The MagCache paper's Table 3 shows that a random prompt, the average of 944 prompts, and an outlier prompt all give nearly identical `γ_t` curves (LPIPS difference < 0.005, PSNR difference < 0.2). This is a dramatic operational simplification compared to TeaCache's 70-prompt polynomial fitting.

**Headline numbers (Table 1 of the paper, comparison with TeaCache on identical FLOPs budget).**

| Model | Method | Speedup | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|
| Open-Sora 1.2 51f/480P | TeaCache-slow | 1.40× | 0.1303 | 0.8405 | 23.67 |
| Open-Sora 1.2 51f/480P | **MagCache-slow** | 1.41× | **0.0827** | **0.8859** | **26.93** |
| Open-Sora 1.2 51f/480P | TeaCache-fast | 2.05× | 0.2527 | 0.7435 | 18.98 |
| Open-Sora 1.2 51f/480P | **MagCache-fast** | 2.10× | **0.1522** | **0.8266** | **23.37** |
| Wan 2.1 1.3B 81f/480P | TeaCache-slow | 1.59× | 0.1258 | 0.8033 | 23.35 |
| Wan 2.1 1.3B 81f/480P | **MagCache-slow** | **2.14×** | 0.1206 | 0.8133 | **23.42** |
| Wan 2.1 1.3B 81f/480P | TeaCache-fast | 2.14× | 0.2412 | 0.6571 | 18.14 |
| Wan 2.1 1.3B 81f/480P | **MagCache-fast** | **2.68×** | 0.1748 | 0.7490 | **21.54** |
| HunyuanVideo 129f/540P | TeaCache-slow | 1.63× | 0.1832 | 0.7876 | 23.87 |
| HunyuanVideo 129f/540P | **MagCache-slow** | 2.25× | **0.0377** | **0.9459** | **34.51** |
| HunyuanVideo 129f/540P | TeaCache-fast | 2.26× | 0.1971 | 0.7744 | 23.38 |
| HunyuanVideo 129f/540P | **MagCache-fast** | **2.63×** | **0.0626** | **0.9206** | **31.77** |

The HunyuanVideo numbers are striking: **MagCache-slow at 2.25× achieves PSNR 34.51 vs TeaCache-slow's 23.87 at 1.63×.** That's a 10-point PSNR jump at *higher* speedup. The "magnitude is everything" insight pays off most strongly on the deepest video DiTs where the per-block residual norm is more stable than on shallower image DiTs.

**Memory cost.** ~0.5 GB of additional GPU memory for the residual cache (vs ~40 GB for TaylorSeer's multi-step finite-difference cache on Wan 2.1). This is one of MagCache's strongest practical wins.

**Default hyperparameters** (per the official code):
- Open-Sora: MagCache-slow `K=1, δ=0.06`; MagCache-fast `K=3, δ=0.12`.
- Wan 2.1: MagCache-slow `K=2, δ=0.12`; MagCache-fast `K=4, δ=0.12`.

**Limitations.**
- **The 80/20 magnitude/direction split is empirical.** For the final 20% of the trajectory, both magnitude and direction change sharply, and the per-step error from skipping grows. MagCache handles this with the `K` ceiling — it won't skip more than `K` consecutive steps even if `E_t` is below `δ`, which prevents catastrophic drift in the late phase.
- **Open question on distilled checkpoints.** The unified magnitude law was derived on standard 30–50 step models; whether it holds for 4-step Lightning-style distilled checkpoints is unverified in the paper. The expectation is that with so few steps, the per-step `γ_t` is too coarse for caching to help, but no MagCache-on-distilled experiments are reported.

## C.9 vLLM-Omni Cache-DiT umbrella + compatibility list

**Reference.** [DeepWiki summary of vLLM-Omni's caching layer (5.2 Caching Acceleration: TeaCache and Cache-DiT)](https://deepwiki.com/hsliuustc0106/vllm-omni-skills/5.2-caching-acceleration%3A-teacache-and-cache-dit), based on `skills/vllm-omni-diffusion-perf-optim/SKILL.md` and `skills/vllm-omni-perf/references/teacache.md` from the [hsliuustc0106/vllm-omni-skills](https://github.com/hsliuustc0106/vllm-omni-skills) repo.

vLLM-Omni's caching layer wraps two distinct backends under a unified `DiffusionCacheConfig`:

**Backend 1: TeaCache (Temporal-Adaptive Cache).** Direct integration of the upstream TeaCache logic, exposed via the CLI flag `--enable-teacache` with `--teacache-threshold` (default 0.1). Same per-step relative-L1 trigger and cached-residual-add as upstream.

**Backend 2: Cache-DiT.** A suite of DiT-specific caching strategies bundled together:
- **DBCache (Dynamic Block Caching).** Caches specific layers / blocks rather than the entire transformer. Conceptually parallel to BWCache's block-level approach, but DBCache is older and simpler — it picks a fixed subset of blocks to always cache.
- **TaylorSeer (predict, don't reuse).** The same Taylor expansion forecasting from §C.4, integrated as a vLLM-Omni backend.
- **SCM (Step Masking).** Specifically tuned for *Consistency Models* and *distilled DiTs* (Wan 2.2-Lightning, FastWan, Lightning T2V), this strategy masks redundant steps in the shortened denoising trajectory. The motivation is that distilled checkpoints already have a very short trajectory (4–8 steps), and SCM detects when consecutive steps have near-equal predicted noise and merges them.

**The `_NO_CACHE_ACCELERATION` exclusion list.** vLLM-Omni maintains a registry of model classes where caching acceleration is *not* compatible with the model implementation, either because the architecture is non-DiT (caching's core assumption fails) or because the specific implementation diverges from the `transformer.forward` signature that the cache adapter intercepts. The exclusion list is in the `vllm_omni.diffusion` registry. Per the deepwiki summary's "Code Structure and Registry" section:

> *"Exclusion List: The `_NO_CACHE_ACCELERATION` list in the registry prevents enabling caching on models where the architecture (e.g., non-DiT) or the specific implementation is incompatible."*

[VERIFIED IN deepwiki SUMMARY] — the exact contents of the exclusion list at the time of indexing (2026-03-29) are not enumerated in the deepwiki page (which references but does not quote `skills/vllm-omni-diffusion-perf-optim/SKILL.md:217-225`); the practical implication is that any DiT-class model new to vLLM-Omni starts disabled-by-default for caching and gets enabled per-checkpoint after a quality-regression check.

**Recommended deployment workflow (per the deepwiki "Summary of Usage" section):**
1. Verify the model is DiT-based (FLUX, Wan2.2, SD3, HunyuanVideo, LTX-2). If it's UNet, none of these caching strategies apply.
2. Enable `--enable-teacache` for general acceleration (the most-tested backend).
3. Tune `--teacache-threshold` from the default 0.1 up to 0.2 (more aggressive) or down to 0.05 (more conservative) based on the use case's quality bar.

**Compatibility with other vLLM-Omni features.**
- **Tensor Parallelism `--tp`.** Caching composes cleanly with TP. The cached residual is shape-aligned with the per-rank output, so each rank caches its local shard and skips the same set of steps as every other rank. The skip decision is data-dependent on the cached scalar magnitude, but is broadcast across ranks at the start of each step (§C.N below).
- **CPU Offloading.** Compatible — the cached residual stays on GPU memory while DiT weights are streamed.
- **FP8 Quantization.** Compatible, but as noted by the ParaAttention guide, *"you need to change the `residual_diff_threshold` to a larger value for it to take effect"* under FP8 because the activation distribution shifts. vLLM-Omni recommends bumping the threshold by ~2× when FP8 is enabled.
- **NVFP4 Quantization.** Less tested as of the deepwiki indexing date (2026-03); the same activation-distribution-shift caveat applies and likely needs a 1.5–2× threshold bump.

**Implication for B200 deployment.** The vLLM-Omni Cache-DiT umbrella is the canonical reference for what's safe to enable on a production B200 stack today. The combination shipped in the most recent Wan 2.2 NVFP4 PR ([#1412](https://github.com/vllm-project/vllm-omni/pull/1412)) is `--enable-teacache --teacache-threshold 0.15 --tp 4 --sp 2`, which delivers the documented 2.1–2.5× cache speedup on top of the FP8/NVFP4 + Ulysses+TP base. If a model checkpoint surfaces in `_NO_CACHE_ACCELERATION` (e.g., a non-standard DiT variant or a distilled checkpoint where the cache trigger has been disabled), the recommendation is to disable caching entirely rather than fall back to a partially-cached configuration.

## C.10 Speculative-then-verify (SpeCa, SDVG)

This subsection covers the **2025–2026** generation of methods that explicitly borrow from LLM speculative decoding: *forecast* a candidate residual or video block, then *verify* whether it's good enough to commit. This is a separate framing from the cache-then-reuse / cache-then-forecast family because it explicitly handles the case where the prediction is *wrong*.

### C.10.1 SpeCa — Speculative Feature Caching (ACM MM 2025)

**Reference.** Liu et al., *SpeCa: Accelerating Diffusion Transformers with Speculative Feature Caching* — [arXiv:2509.11628](https://arxiv.org/abs/2509.11628), ACM MM 2025; code in [Shenyi-Z/Cache4Diffusion](https://github.com/Shenyi-Z/Cache4Diffusion/).

**Motivation: TaylorSeer is open-loop.** TaylorSeer predicts feature trajectories with no verification — if the prediction is wrong, the error propagates through the rest of the trajectory uncorrected. SpeCa adds a verification step that decides per-prediction whether to accept it or reject it (and fall back to a full computation).

**Mechanism: forecast-then-verify with TaylorSeer as the draft model.**
1. **Compute** the full forward pass at a key timestep `t` to get accurate features `F(x_t^l)`.
2. **Forecast** features for the next `K` timesteps using TaylorSeer's Eq. 2 (degree-2 Taylor expansion in `t` from past features).
3. **Verify** at each predicted step `t-k`: compute the relative L2 error of the *final-layer* prediction vs. a cheap reference (the L2 norm of the predicted features compared with a single-block forward pass on the actual `x_{t-k}`).

For the predicted sequence `{t-1, t-2, ..., t-K}`, define the relative L2 error:

> e_k = ‖F_pred(x_{t-k}^l) − F(x_{t-k}^l)‖_2 / (‖F(x_{t-k}^l)‖_2 + ε)

with ε = 10^{-8}. **Accept** the prediction iff `e_k ≤ τ_t`; otherwise **reject** it and all subsequent predictions, falling back to standard computation. The threshold is *adaptive* across the trajectory:

> **τ_t = τ_0 · β^{(T-t)/T}**

with `β ∈ (0, 1)` a decay parameter and `τ_0` the initial threshold. This relaxes the threshold in the noisy early steps (where high-magnitude residuals tolerate larger absolute errors) and tightens it in the late steps (where fine details emerge).

**Computational cost.** SpeCa's overhead per prediction is ~3.5% of a full forward pass on DiT, ~1.75% on FLUX, and **~1.67% on HunyuanVideo** — verification involves computing only the final layer's output for comparison, which is a small fraction of the full DiT.

**Headline numbers (SpeCa Tables 1, 2, 3).**

| Model | Speedup | Quality (vs baseline) |
|---|---|---|
| FLUX.1-dev (50 steps) | 6.34× | ImageReward −5.5% (vs −17.5% for TaylorSeer at same speedup) |
| FLUX.1-dev | 4.70× | ImageReward +0.0100 (better than baseline) |
| HunyuanVideo (50 steps) | 6.16× | VBench 79.84 (vs 80.66 baseline) |
| HunyuanVideo | 5.23× | VBench 79.98 (vs 80.66 baseline) |
| DiT-XL/2 (ImageNet 256, 50 steps) | 7.10× | FID 5.55 (vs FID 12.15 for DDIM-10 at same speedup) |
| DiT-XL/2 | 6.76× | FID 3.76 (best at this acceleration) |

The **5.5% ImageReward drop at 6.34× FLUX** is the single strongest "high-acceleration with quality" result in the caching literature as of mid-2025. The 17.5% gap to TaylorSeer at the same speedup is exactly the cost of TaylorSeer being open-loop — without verification, prediction errors compound and produce quality collapse beyond ~5×.

**Sample-adaptive computation allocation.** SpeCa's third contribution is per-sample resource allocation: easy samples (where the verification mostly accepts) get aggressive acceleration, hard samples get more compute. On HunyuanVideo, the paper reports that *57.5% of cases* are accelerated 6.48× while *42.5% of cases* are accelerated only 5.82× — averaging to the headline 6.16× number. This is finer-grained than AdaCache's per-video adaptation because SpeCa adapts per-step within a video.

**Acceleration ratio formula.** Let `α = T_spec / T` be the fraction of speculatively-predicted steps and `γ ≪ 1` the verification overhead per step (3.5% / 1.75% / 1.67% for DiT / FLUX / HunyuanVideo). The theoretical speedup is:

> **S = 1 / (1 − α + α · γ) ≈ 1 / (1 − α)** for `α ≫ γ`.

So if SpeCa accepts 80% of speculations, the achievable speedup is ~5×; if it accepts 90%, ~10×. The per-step verification overhead doesn't dominate until `γ` approaches `1 − α`.

### C.10.2 SDVG — Speculative Decoding for Autoregressive Video Generation

**Reference.** Hu & Zhang, *Speculative Decoding for Autoregressive Video Generation* — [arXiv:2604.17397](https://arxiv.org/abs/2604.17397). Built on top of Self-Forcing autoregressive video diffusion.

**Note on framing.** SDVG operates on a different paradigm than every other method in this section: it targets **autoregressive video models** (Self-Forcing, Krea Realtime 14B) where video is generated block-by-block with a causal KV cache, rather than full-sequence diffusion. The speculative-decoding analogy maps more cleanly to AR generation than to standard diffusion, because AR video has discrete blocks that can be accept/reject-routed.

**Mechanism: drafter-target speculative routing.** A small **1.3B drafter** model (Wan 2.1-T2V-1.3B Self-Forcing) generates candidate video blocks via 4 denoising steps; a large **14B target** model (Krea Realtime 14B) is invoked only when the drafter's output is judged insufficient.

For each block `b`:
1. **Drafter** runs `S=4` denoising steps on initial noise to produce candidate `x̂_b`.
2. **VAE-decode** candidate to pixel frames `f_1, ..., f_F` (F=9 for block 0, F=12 for blocks 1–8 at 832×480).
3. **Image-quality router** scores each frame with **ImageReward** and aggregates with the **min-frame** rule (the *worst* frame's score, not the mean):
   > q_b = min_{i=1..F} R(f_i, p)

   This worst-frame aggregation is essential — averaging masks single-frame artifacts that produce visible flicker.
4. **Accept** if `q_b ≥ τ`; commit `x̂_b` to the target's KV cache (no target compute) and emit the decoded frames directly.
5. **Reject** if `q_b < τ`; the target regenerates the block from the same initial noise (`S=4` steps), updates its KV cache, and emits the new frames.

**Two critical design choices:**
- **Block 0 is always force-rejected** regardless of its draft score. Block 0 has no KV context from prior blocks and establishes the scene composition — accepting a draft here risks propagating irreversible layout errors.
- **τ is a single fixed scalar** calibrated offline. No adaptive threshold; sweeping τ from −0.7 (conservative) to −2.5 (aggressive) traces a smooth Pareto frontier.

**Headline numbers (Table 1, 1003 MovieGenVideoBench prompts at 832×480, 9 blocks).**

| τ | Speedup | VisionReward | Quality retention |
|---|---|---|---|
| −0.7 (conservative) | 1.59× | 0.0773 | 98.10% |
| −1.0 | 1.69× | 0.0764 | 96.95% |
| −1.5 | (intermediate) | (intermediate) | (intermediate) |
| −2.5 (aggressive) | **2.09×** | 0.0754 | 95.69% |
| Draft-only baseline | — | 0.0644 | (−18.2% from target) |
| Target-only baseline | 1.0× | 0.0788 | 100% |

The **+17.1% gap above draft-only** at the aggressive τ=−2.5 setting is what justifies the verification overhead — pure draft-only generation produces visibly worse output, and SDVG's reward routing successfully selects the borderline-acceptable drafts that match target quality.

**Acceptance rate dynamics.** As τ relaxes from −0.7 to −1.0, accept rate rises from 73.1% to 78.0% but VisionReward only drops 0.12% absolute. This near-flat region indicates the additional drafts admitted are borderline cases close to target quality. Beyond 78% acceptance, quality drops more noticeably (0.0757 at 83.4%, 0.0754 at 88.9%) because the router is now admitting low-scoring drafts. The **operating sweet spot is 73–78% acceptance** (τ ∈ [−1.0, −0.7]).

**Compatibility with other accelerations.** SDVG is *orthogonal* to step-level methods like T-Stitch, SRDiffusion, HybridStitch — it operates at the block level above them. Inside each block (drafter or target), the standard 4-step Self-Forcing schedule is used; FlashAttention / SageAttention / NVFP4 quantization can be enabled inside both drafter and target without affecting the routing logic.

**Limitations.**
- **Distributional bias.** Unlike LLM speculative decoding (which preserves the target distribution exactly via rejection sampling), SDVG's reward-based router accepts a controlled distributional shift toward the drafter. A stricter τ reduces the gap at the cost of speedup.
- **ImageReward as proxy.** ImageReward was trained on text-image pairs and evaluates frames independently — it misses temporal consistency and motion quality. A dedicated video-block quality model would improve routing.
- **Wasted draft compute.** For rejected blocks (including the always-rejected block 0), the drafter forward pass and VAE decode are unused.
- **Hardware specificity.** SDVG's published numbers are on 2× RTX A6000 (one GPU each for transformer and reward model) — not yet ported to single-B200 deployment.

### Cross-cutting note: DraftAttention as the SDVG analog for sparse attention

[Forwarded reference for the sparse-attention worker.] **DraftAttention** ([arXiv:2505.14708](https://arxiv.org/abs/2505.14708), [shawnricecake/draft-attention](https://github.com/shawnricecake/draft-attention)) applies the same forecast-then-verify framing to *attention sparsity* rather than to *block-level video generation*: a low-resolution "draft" attention pass (downsampled Q/K with 128× fewer tokens) computes a sparsity mask that guides a high-resolution sparse attention pass. The structural similarity to SpeCa/SDVG is intentional — verify a cheap forecast at fine granularity, fall back to full compute when the forecast fails. DraftAttention reaches up to 1.75–2× e2e speedup on B200. See the sparse-attention section for full coverage; the cross-link here is that "speculate then verify" is becoming a unifying framing across caching, attention, and even autoregressive generation.

## C.N Composition with sparse attention, step distillation, and parallelism

This subsection covers how the methods above compose with the rest of the B200 acceleration stack. The main insight is that **caching composes cleanly with attention/quantization/parallelism** (orthogonal axes of the bottleneck) but **competes with step distillation** (the same axis, with distillation usually winning).

### Composition with sparse attention

Cache and sparse attention compose multiplicatively: cache reduces `N_steps × T_step` by skipping steps; sparse attention reduces `T_step` for the steps that *do* run. On a typical Wan 2.1 14B 720p generation:
- TeaCache 2.1× alone: 1./2.1 = 47.6% of steps run.
- VSA / STA / SVG2 (~3× attention speedup, attention is ~70% of `T_step`) reduces a *running* step from 1.0 unit to 1.0 - 0.7·(1 - 1/3) = 0.53 units.
- Combined: end-to-end ≈ 0.476 × 0.53 = 25.2% of baseline = **3.96× speedup** vs the 2.1×-cache-only or 1.89×-attention-only baselines.

Empirically the ComfyUI community thread on TeaCache + SageAttention reports **~30% extra speedup** on top of TeaCache when SageAttention is enabled; this is consistent with the composition formula above for ~1.5× attention-only acceleration on a TeaCache-2× baseline.

The composition is structurally clean because:
- The cache decision is made once per step, before any attention computes; sparse attention runs unchanged inside the steps that aren't skipped.
- The cached residual is shape-equivalent to a dense-attention residual; reusing it doesn't break sparse-attention's KV layout or block structure.
- Both cache and sparse attention bypass the same expensive compute (DiT block forward pass), so they don't fight over GPU resources.

The only edge case: when sparse attention uses a *trained* mask (Radial, BLADE) that depends on the per-step input, switching between cached and computed steps may produce a slightly different mask distribution than the training-time mask. In practice this is a sub-percent quality artifact; production deployments don't bother to retrain.

### Composition with step distillation

This is the one composition that **does not compose cleanly**. Step distillation collapses the trajectory from 50 → 4–8 steps; caching's value depends on having enough steps to cache *between*. Three regimes:

1. **No distillation, full 50 steps.** Caching at 50% hit rate gives 2× speedup. This is the dominant regime in 2024–early 2025 production.
2. **Light distillation, 12–16 steps.** Caching at 30% hit rate gives 1.4× speedup. Still useful but shrinking. This is the HunyuanVideo-1.5 step-distill regime.
3. **Heavy distillation, 4–8 steps.** Caching is structurally hard:
   - First step always runs (boundary).
   - Last step always runs (boundary).
   - For an 8-step model, that leaves 6 candidate skip steps; for a 4-step model, only 2.
   - The per-skip distance `|k|` is now huge in *original-step-equivalent* terms — each distilled step represents 6–12 original steps — so the per-skip error growth is `|k|^{m+1}` (for TaylorSeer order `m`) is severe.
   - In practice: TeaCache on 4-step Wan 2.2-Lightning gives <1.1× speedup at unacceptable quality loss.
   - TaylorSeer with `(m=2, N=2)` on 8-step distilled model gives ~1.5× at minimal quality cost — a regime where TeaCache outright fails.

**Why this matters.** B200 production stacks today (lightx2v Wan2.2-Lightning, FastWan 2.1, Krea Realtime 14B, Cosmos-Predict 2.5 distilled) are universally distilled to 2–8 steps. **In these stacks, caching is typically disabled or replaced with SCM (vLLM-Omni's Step-Masking variant for distilled DiTs).** The lever that matters is distillation; caching is only relevant for the fallback non-distilled path or for high-quality "slow mode" inference.

The cleanest visualization: distillation collapses the y-axis (`N_steps`) by 6–25×; caching can recover at most another 1.4× on top of an already-collapsed axis. In `T_e2e ≈ N_steps × T_step` accounting, a 4-step distilled + 1.4× cached run is barely faster than a 4-step distilled run alone, and the 1.4× cache may not be worth its quality cost.

The exception is **TaylorSeer-on-distilled** (the SpeCa direction): forecasting from a few cached steps within a distilled trajectory can in principle recover speedup, but published numbers as of May 2026 are limited to non-distilled models. This is the open research question for caching in 2026.

### Composition with parallelism (Ulysses / Ring / TP)

Caching composes **superlinearly** with multi-GPU parallelism because cached steps avoid the cross-GPU all-to-all (Ulysses) or all-reduce (TP) entirely. The TeaCache CVPR 2025 Table 4 makes this concrete: on Open-Sora-Plan 221-frame 512² the e2e speedup is

| GPUs | Baseline | PAB | TeaCache |
|---|---|---|---|
| 1× A800 | 1× | 1.56× | 6.73× |
| 2× A800 | 1.94× | 2.95× | 12.02× |
| 4× A800 | 3.68× | 5.59× | 20.39× |
| 8× A800 | 6.79× | 10.16× | **32.02×** |

The 32× number is real because (a) Ulysses gives 6.79× alone, (b) TeaCache on Open-Sora-Plan gives 4.41× alone, but (c) on 8 GPUs the cached steps avoid the three pre-attention all-to-alls + post-attention all-to-all, recovering the parallel overhead exactly. The combination is ~32× = 6.79 × 4.41 / (some leakage), which is essentially the product of the two factors. AdaCache shows the same superlinear pattern on Open-Sora 480P/2s (8 GPUs: 6.94× combined vs 4.17× DSP-only).

The same pattern holds for FBCache + ParaAttention + CP: 6.76× on FLUX with 4 GPUs, where 1 GPU FBCache alone gives 1.55×.

### Composition with quantization

Cache + FP8/NVFP4 quantization composes cleanly with one caveat: **the threshold needs to be retuned upward by ~2× under FP8/NVFP4** because the activation distribution shifts. The ParaAttention guide is explicit: *"Dynamic quantization can significantly change the distribution of the model output, so you need to change the `residual_diff_threshold` to a larger value for it to take effect."*

Concretely:
- FBCache on FLUX BF16: `rdt=0.08` for 1.55×.
- FBCache on FLUX FP8 DQ: `rdt=0.12` for the same effective skip rate.

The benchmarks in the FBCache section above show that the combination scales:
- BF16 FLUX + FBCache + FP8 DQ + CP=4 → 6.76× e2e on L20.
- The same pattern is reported on B200 with NVFP4 in vLLM-Omni's Wan 2.2 PR #1412: `--enable-teacache --teacache-threshold 0.15` on top of NVFP4 + 4-GPU TP gives an additional 2.1× over the NVFP4-only baseline.

### Composition with CUDA graphs

The most-composable methods (TeaCache, FBCache, SmoothCache) capture the entire DiT forward as a single CUDA graph and either replay it (compute step) or skip it (cached step). The conditional is dynamic Python that lives outside the graph; the per-step decision is one scalar comparison.

The least-composable methods are the per-block-conditional ones (BWCache, DBCache) and the per-layer-conditional ones (AdaCache with selective layer caching) — these need either:
- Multiple captured graphs (one per cached-block-set configuration) with runtime dispatch, or
- Static cache configurations chosen ahead of time.

In production, the ParaAttention / xDiT integrations of TeaCache use single-graph capture; the FastVideo integration of TaylorSeer uses one captured graph per `(N, m)` pair (typically 2–4 graphs).

### When NOT to use caching: the distilled-model regime

To restate: caching is **strictly suboptimal** when step distillation has already collapsed the trajectory below ~10 steps. The decision rule for production B200 deployments today is:

```
if checkpoint == "distilled" (4–8 steps):
    use no caching (or SCM for very specific Wan 2.2-Lightning-style schedules)
elif checkpoint == "lightly distilled" (12–16 steps):
    use TeaCache or MagCache with conservative threshold
elif checkpoint == "non-distilled" (30–50 steps):
    use TeaCache, MagCache, or TaylorSeer (whichever the model has best support for)
    + sparse attention (VSA / STA / SVG2 / DraftAttention)
    + NVFP4 quantization
    + Ulysses + TP parallelism
```

This is the structural reason the most aggressive caching-paper claims (TaylorSeer 5×, AdaCache 4.7×, TeaCache 4.41×) are *less impactful in 2026 production* than they were in late 2024 — the stack has moved on to step distillation as the dominant lever, and caching has retreated to a supporting role.

### Looking forward: caching in the 2026 stack

The open research questions for caching as of May 2026:
1. **Cache-aware distillation.** Can a distillation procedure be modified to produce a checkpoint that *retains* enough trajectory smoothness for caching to work? Phased-DMD and Self-Forcing don't currently optimize for this.
2. **Joint NVFP4 + cache calibration.** The ad-hoc 2× threshold bump under FP8 isn't principled; a proper joint calibration (find the threshold that minimizes activation-distribution shift's effect on the indicator) hasn't been published.
3. **Caching in serving, not just inference.** Multi-tenant video DiT serving (vLLM-Omni, lightx2v Mooncake disagg) could in principle share cache state across requests with similar prompts, but no published method does this yet.
4. **TaylorSeer + verification on distilled checkpoints.** The SpeCa framing (forecast + verify) on a 4-step distilled model is the most likely path to combining the two paradigms; experimental data for this configuration is the missing piece.
5. **Magnitude law for distilled / consistency models.** Whether MagCache's `γ_t` curve generalizes to Lightning-style or rCM-distilled checkpoints is unverified; preliminary indications from the MagCache appendix suggest it does *not* — distilled models have a fundamentally different residual structure.

These open problems are why caching is not yet "solved" as an acceleration axis — the rapid progress on step distillation has changed the goalpost faster than the caching literature can keep up with, and the remaining headroom requires either a fundamentally different formulation (cache-then-verify) or co-design with the distillation procedure itself.
