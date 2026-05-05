### §3-B. Sparse and Structured Attention (STA, VSA, Radial, SVG/SVG2, SLA/SageSLA, BSA, NSA, SpargeAttn, VMoBA, MInference, XAttention, ΔAttention, DiTFastAttn, DraftAttention, ToMeSD, Astraea, db-SP, DSA)

*Subsection prefix `S.x` is scoped to §3-B.*

This sub-section is the deepest single column of the May 2026 video-DiT report. Sparse attention is the *largest single non-distillation lever* in the stack — it routinely cuts attention by 3–17× kernel-side and 1.5–3.5× e2e, and when *trained-in* via sparse-distillation (FastWan VSA, BLADE) it not only matches but often *beats* the dense teacher on VBench. Every production stack on Blackwell B200 today uses at least one method from this column; FastVideo, SGLang, vLLM-Omni, TurboDiffusion, lightx2v, and Hao AI Lab all ship sparse-attention backends as first-class citizens.

The section is organized as: a taxonomy (§S.1), then per-work deep dives in dependency order (STA → VSA → Radial → SVG/SVG2 → SLA → BSA → NSA → SpargeAttn → MoBA/VMoBA → MInference → XAttention → ΔAttention → DiTFastAttn → DraftAttention → ToMeSD/VidToMe → Astraea/SSTA), and a closing composability matrix (§S.18) that summarizes which methods compose with Ulysses, Ring, db-SP, DSA, TeaCache, BLADE, and FA4 on B200.

## S.1 Taxonomy of sparse attention for video DiT

It is helpful to fix vocabulary before diving in. A sparse attention method is determined by **(a) how the mask `M ∈ {0,−∞}^{N×N}` is generated**, **(b) the granularity of the mask** (token, tile, block), **(c) whether the method is trainable**, and **(d) whether the kernel matches GPU hardware**. The video-DiT family splits cleanly along axes (a) and (c).

**Axis A — mask source.**

| Pattern | Examples | When generated | Cost of generation |
|---|---|---|---|
| **Static analytic** | Radial Attention, STA, DiTFastAttn (window) | Once per resolution; never per step | 0 (precomputed) |
| **Static profiled** | MInference offline-calibrated | Once at install per checkpoint | One full-attention probe |
| **Dynamic top-k (cheap proxy)** | VSA, MoBA/VMoBA, MInference-V/S, XAttention, SpargeAttn, SLA, BSA, NSA | Per call, per layer, per timestep | <1% of full attention |
| **Online classification** | SVG (spatial-vs-temporal head), SVG2 (k-means permutation), AdaSpa | Per call with small probe (1–5% of Q tokens) | 1–5% of full attention |
| **Speculative draft-and-verify** | DraftAttention, SpeCa, SDVG | Per call; coarse-attention pass picks blocks | 5–10% of full attention |
| **Hybrid sparse + linear** | SLA, NSA, SageSLA | Per call; sparse for critical, linear for marginal | <0.5% (linear branch is O(N)) |
| **Bidirectional pruning** | BSA (queries *and* KV) | Per call; query-prune via cosine, KV via cum-mass | <0.5% |

**Axis C — trainability.**

- **Training-free (drop-in, all heads, all timesteps):** STA training-free, SVG, SVG2, SpargeAttn, MInference, XAttention, ΔAttention, DiTFastAttn, DraftAttention, BSA training-free, Radial Attention default.
- **Adapter-tuned (LoRA or short fine-tune):** Radial Attention length-extending LoRA, STA fine-tuning recipe (8 H100 × 8 hours), SVG2 (none required, but PR #50 recommends warm-up adapter for highest sparsity).
- **Trained-from-scratch / sparse-distill:** VSA, BSA training-based, SLA fine-tune (2000 steps), NSA, MoBA-pretrained Kimi/K2, FastWan VSA-distill, BLADE.

**The key 2026 observation: the sparse-distill cell at (dynamic top-k, trained) is strictly better than dense.** All of VSA Wan-1.3B (50.9× e2e), VSA Wan-14B (>2×), SLA Wan-1.3B (95% sparsity, 2.2× e2e at zero quality loss), BSA Wan-1.3B 14B (17.79× attention), and FastWan show that *the student with sparse attention beats the dense teacher on VBench*, not just matches it. This invalidates the conventional intuition that sparsity costs quality and reframes sparse attention as the primary scaling axis for video DiT going forward.

**Axis D — block-sparse layout in 2026.** Modern Tensor Cores (Hopper SM90, Blackwell SM100/SM120) require ≥16-token tiles to achieve ≥40% MFU. Every method in the survey aligns its mask to tiles of 64 (= 4·4·4 cubes for VSA / BSA / SSTA), 128, or non-square (64×128 for SageSLA SM90, 128×64 for SM89). Below 16 tokens per tile, FlashAttention drops to <10% MFU; above 256 tokens per tile the granularity blurs critical-token recall by ≥10pt. The 64-token sweet spot (4×4×4 cubes for video latents) is shared by VSA, BSA, SSTA, and the SLA SageSLA SM89 path.

## S.2 Sliding Tile Attention (STA) and Selective Sliding Tile Attention (SSTA)

**Sources:** STA — [arXiv:2502.04507](https://arxiv.org/abs/2502.04507) (Hao AI Lab, ICML 2025). FastVideo kernel at `/Users/yewang/work/FastVideo/fastvideo-kernel/csrc/attention/st_attn_h100.cu`. SSTA — [arXiv:2511.18870](https://arxiv.org/abs/2511.18870) (Tencent HunyuanVideo 1.5).

### S.2.1 The "mixed-block tax" problem STA solves

Pre-STA, the natural translation of *3D sliding window attention* to GPU was **Tiled NATTEN** ([Hassani 2023](https://arxiv.org/abs/2308.13494)). The problem: an `N×N` FlashAttention block-mask has three cell types — **dense** (all 1s, computed in full), **empty** (all 0s, skipped), and **mixed** (some 1s, some 0s). Mixed blocks carry two costs: (i) the kernel computes `Q_i K_jᵀ` for the *whole* block before applying the mask, so FLOPs aren't actually saved; (ii) per-block mask evaluation is itself non-trivial (FlexAttention reports ~15% overhead for a *causal* mask, and for 3D-sliding far worse). The STA paper measures this at scale (HunyuanVideo, 720p, 5s, 115K tokens, sparsity ≈ 90%):

| Method | Implementation | Sparsity | TFLOPS | Latency (ms) | MFU | Speedup |
|---|---|---|---|---|---|---|
| FA3 ThunderKittens (dense) | TK | 0% | 164.03 | 265.28 | 62.5% | 1.00× |
| CLEAR r=16 | FlexAttention | 90.5% | 15.65 | 307.44 | 5.2% | 0.86× *(slower!)* |
| NATTEN (19,25,25) | FlexAttention | 89.7% | 16.91 | 313.92 | 5.4% | 0.85× |
| Tiled NATTEN | CUDA | 89.7% | 16.91 | 458.36 | 3.7% | 0.58× |
| Swin (24,32,32) | FlexAttention | 87.4% | 20.64 | 47.90 | 43.6% | 5.54× |
| **STA (18,24,24) ThunderKittens** | TK | **91.0%** | 14.76 | **25.38** | **58.8%** | **10.45×** |

**[VERIFIED — Table 2 of arXiv:2502.04507]**: a sparser pattern (CLEAR 90.5%) actually runs *0.86×*, and a coarser tiled-NATTEN runs *0.58×*. STA recovers the proportionality: 91% sparsity → 10.45× speedup, 58.79% MFU vs 62.5% dense ceiling.

### S.2.2 Core technical contribution: tile-aligned 3D windows

Where Tiled NATTEN slides the *window center* per *token*, STA slides per *tile*. Formally, given video latent shape `(L, L, L)` and FlashAttention tile size `B`, set the spatial-temporal tile size `T` such that `B = T³`. Reorder tokens so each `(T, T, T)` tile occupies contiguous indices. Then for each query tile, the window covers exactly `(W/T)³` key tiles, *all of which become dense FA blocks* — zero mixed blocks ever.

The accounting (Theorem 3.2 of the paper):
$$
S_{\mathrm{dense}} = \left(\frac{W}{T}\right)^{3} \cdot \left(\frac{L}{T}\right)^{3}
$$
For `(T,T,T) = (4,4,4)` and window `(W,W,W) = (12,12,12)`, only 1.56% of the FA blocks are dense; the rest are *empty* (literally skipped). For window `(20,20,20)`, dense fraction is 7.23%. Compare Tiled NATTEN at the same configs: 0.06% dense, 7.17% mixed (i.e., FA still has to *evaluate* the mask on 7.17% of blocks).

### S.2.3 Kernel design (ThunderKittens, sm_90a only)

```c++
// from fastvideo-kernel/csrc/attention/st_attn_h100.cu
constexpr int CONSUMER_WARPGROUPS = 3;   // 3 wgs compute attention
constexpr int PRODUCER_WARPGROUPS = 1;   // 1 wg loads via TMA
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int qo_height = 4*16;     // 64 query rows per tile
    constexpr static int kv_height = 8*16;     // 128 KV rows per tile
    constexpr static int stages    = 2;         // 2-stage TMA pipeline
};
template<int D, bool is_causal, bool text_q, bool text_kv,
         int DT, int DH, int DW, int CT, int CH, int CW>
__global__ void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g);
```

Template parameters `<DT, DH, DW>` pin the *video* tile geometry, `<CT, CH, CW>` pin the *cube* (window) geometry. Both `text_q` and `text_kv` toggle separate code paths because HunyuanVideo prepends text tokens that must always attend to all video tokens regardless of window — the kernel handles this with a special branch (`seq_idx = CT*CH*CW*6.0 + ...` for the text-shifted query stream).

**Producer–consumer split.** The producer warpgroup runs the *inter-block mask logic*: given the query tile coordinate `(t, h, w)`, computes which KV tiles are inside the 3D window, and TMA-loads exactly those into shared memory. The consumer warpgroups are *oblivious to sparsity* — they see only dense blocks in SRAM and run a stock FA3 inner loop. This is the trick that lets STA hit 58.79% MFU at 91% sparsity: the mask logic is hidden behind asynchronous TMA.

### S.2.4 Mask search (training-free) and fine-tuning

**Algorithm 1 of the paper** picks per-head window sizes:
```
for t = T_0+1 .. T:
    for each (layer, head):
        O_dense ← full attention(layer, head)
        for each candidate window p in P:
            O_p ← STA with mask p
            loss[p] ← MSE(O_dense, O_p)
        record argmin(loss) for (t, l, h)
```
Profile across 16 prompts; head-specialization is prompt-agnostic (Fig 3 of the paper shows std-across-prompts is much lower than std-across-heads). First `T_0` timesteps are kept dense (HunyuanVideo: 12 of 50, 6 of 25, 3 of 10).

**Fine-tuning** (Section 3.2) uses three losses:
$$
\mathcal{L}_\text{attn} = \frac{1}{N}\sum_i \|f^{(i)}_\phi(x_t,t,c) - f^{(i)}_\psi(x_t,t,c)\|_2^2 \qquad\text{(intermediate-layer distillation)}
$$
$$
\mathcal{L}_\text{final} = \|f_\phi(x_t,t,c) - f_\psi(x_t,t,c)\|_2^2 \qquad\text{(final-layer distillation)}
$$
$$
\mathcal{L}_\text{data} = \|(f - x_0) - f_\phi(x_t,t,c)\|_2^2 \qquad\text{(flow-matching data loss)}
$$
Combined: `min α·L_data + β·L_final + γ·L_attn` with `α=1, β=γ=0.5`. 1600 steps, batch size 2, lr 2e-5, alternate guidance scales 1 and 6 by parity, FSDP+CP on 8 H100 (8 hours total — minimal vs pretrain).

### S.2.5 Reported numbers [VERIFIED]

- HunyuanVideo 720p 5s on H100, 50-step DPM++:
  - FA2: 1496 s
  - FA3: 945 s (1.58× over FA2)
  - STA-tf-1.89× (training-free, w=(30,40,40), sparsity 58.33%): **501 s, 1.89×, VBench 82.46% vs 82.71% dense**
  - STA-tf at w=(18,24,24), sparsity 91.0%: **268 s, 3.53×, VBench 80.58%** (some quality drop)
  - STA-t-2.43× (fine-tuned at w=(30,24,40)): **388 s, 2.43×, VBench 83.00%** (slightly *higher* than dense!)
  - STA-t at w=(18,24,24): **268 s, 3.53×, VBench 82.62%** (matches dense after fine-tuning)
- Human MovieGen-bench eval: STA-tf-1.89× ties HunyuanVideo at 83% rate; STA-t-2.43× beats Δ-DiT-1.8× 70.0% to 11.0%.
- Wan 2.1 training-free: SSIM 85.81, PSNR 24.42, 730 s for 4 s 768p (vs Hunyuan because Wan default is shorter), **1.60×**.

### S.2.6 B200 status and parallelism

The kernel is `sm_90a` only — uses Hopper TMA and ThunderKittens primitives (`coord<>`, `gl<>`, `tma::load_async`). A B200 port would need TMEM-aware rewriting against the SM100 `tcgen05` MMA family. As of May 2026 no Blackwell STA kernel has been published; **the Hao AI Lab moved its DiT pipeline integration in FastVideo `main` from STA to VSA** because VSA is trainable, B200-portable through the standard FA-block-sparse Triton path, and accepts variable block sizes (which STA does not).

**Composability:** Ulysses works because STA's mask is purely a function of (t,h,w) — heads can be sharded freely. **Ring requires the chunk size to be ≥ window radius along the partitioned axis** (otherwise the local window crosses chunk boundaries and KV must be all-gathered). TP composes trivially.

### S.2.7 SSTA — Selective Sliding Tile Attention (HunyuanVideo 1.5)

[arXiv:2511.18870](https://arxiv.org/abs/2511.18870) is HunyuanVideo 1.5's 8.3B-param DiT (Tencent, Nov 2025). Architecture: 54 dual-stream blocks, 16 heads, head_dim 128, model_dim 2048, FFN dim 8192. SSTA is its sparse-attention layer; it composes STA's local 3D window with an additional *per-block selectivity gate*. **Algorithm 1** of the technical report:

1. **3D Block Partition** — split Q, K into `(tile_t, tile_h, tile_w)` blocks of size `N`.
2. **Selective Mask Generation** —
   - `Q̄_b, K̄_b ← adaptive_pool(Q_b), adaptive_pool(K_b)`
   - `Score_s = Q̄_b K̄_bᵀ` (Q–K block similarity, head-wise)
   - `Score_r = (1/(N−1))·Σ_{i≠j}[K̄_b K̄_bᵀ]_{ij}` (K–K redundancy: penalize blocks that look like all the others)
   - `Score_i = λ·Score_s − β·Score_r` (block importance)
   - `Selected_indexes = Topk(Score_i, k)` → mask `M_sel`.
3. **STA Mask Generation** — same fixed local 3D window mask `M_sta` as STA above.
4. **Combine and Compute** — `M_combined = M_sel ∧ M_sta`; run `flex_block_attention(Q, K, V, M_combined)`.

The **Score_r** term is the key novelty over STA: rather than just picking the most-attended K blocks, SSTA *also penalizes high mutual K-K similarity*. This is closely related to the determinantal-point-process intuition — picking diverse K blocks rather than redundant ones. The paper claims it eliminates the failure mode where all selected K blocks lie in the same temporal slab.

**Kernel** is `flex-block-attn` ([Tencent-Hunyuan/flex-block-attn](https://github.com/Tencent-Hunyuan/flex-block-attn)), built on ThunderKittens. Released alongside the HunyuanVideo 1.5 model weights.

**Reported numbers [VERIFIED — Tables 7 and 8 of arXiv:2511.18870]:**

| Task | Resolution | Frames | Sparse | s/step (no engineering accel) | s/step (50 steps with engineering accel) |
|---|---|---|---|---|---|
| T2V | 480p | 121 | ✗ | 0.9064 | 0.2781 (13.90 s total) |
| T2V | 480p | 241 | ✗ | 1.7015 | 0.5418 (27.08 s total) |
| T2V | 720p | 121 | ✗ | 2.0084 | 0.5667 (28.33 s total) |
| T2V | 720p | 121 | ✓ | **1.5638** | **0.5283** (26.41 s total) |
| T2V | 720p | 241 | ✗ | 5.5070 | 1.9356 (96.78 s total) |
| T2V | 720p | 241 | ✓ | **2.9475** | **1.1679** (58.39 s total) |

So at **720p × 241 frames (≈10 s clip)** SSTA delivers **5.51 → 2.95 = 1.87× over dense FA3** without any engineering accel, and **96.78 → 58.39 = 1.66× even after SageAttention + `torch.compile` + feature caching are applied on top** (the 1.66× is the *additional* speedup from sparsity beyond all other tricks). Memory: 13.6 GB peak with offloading + tiling, *enabling RTX 4090 inference of 720p 121-frame clips*. Engineering acceleration stack used in the second column: SageAttention v2 + `torch.compile` + a feature-caching mechanism that reuses cached features for non-critical steps. SSTA is sparse-distillation-trained during the distillation phase rather than from-scratch — the paper notes "sparse training is incorporated during the distillation phase, which more effectively preserves output quality while maintaining high computational efficiency" — directly aligning with the FastWan-VSA recipe.

## S.3 VSA — Video Sparse Attention (the FastVideo default)

**Source:** [arXiv:2505.13389](https://arxiv.org/abs/2505.13389), Peiyuan Zhang et al., NeurIPS 2025 Spotlight. FastVideo backend at `/Users/yewang/work/FastVideo/fastvideo/attention/backends/video_sparse_attn.py`. Kernel via `from fastvideo_kernel import video_sparse_attn`. Sparse-Distill training recipe at [hao-ai-lab.github.io/FastVideo/training/examples/Wan2.1-VSA_Wan-Syn-Data](https://hao-ai-lab.github.io/FastVideo/training/examples/Wan2.1-VSA_Wan-Syn-Data/).

VSA is the *successor* to STA — same authors (Hao AI Lab, MBZUAI, UC Berkeley), generalized from a fixed window to a *learned, data-dependent block selection*, end-to-end trainable, and ported to Triton block-sparse FA so it runs on Hopper *and* Blackwell.

### S.3.1 The chicken-and-egg problem

Identifying critical attention positions traditionally requires the full attention matrix — the very thing sparse attention wants to avoid. The VSA paper's own motivation:

> *"How can we predict critical tokens accurately, subject to hardware-aligned block structure, without paying the quadratic cost we aim to avoid?"*

VSA's answer: **a hierarchical, two-stage attention** where the coarse stage operates on cube-pooled tokens (≈64× shorter sequence, <1% of total FLOPs) and emits both an output and a top-k selection, and the fine stage attends only inside selected cubes via a block-sparse FA kernel. Every parameter is differentiable — the gating weights, the cube pooling, the top-k softmax — so the model trains end-to-end through both stages.

### S.3.2 Cube partition with hardware-aligned tile = 64

Given video latent of shape `(T, H, W)` (after VAE patchify), VSA divides into cubes of shape `(C_t, C_h, C_w) = (4, 4, 4)`. Each cube has exactly **64 tokens = 4³**, matching a Tensor Core block-sparse FA tile. Define `(N_t, N_h, N_w) = (T/C_t, H/C_h, W/C_w)` and flatten with the cube-contiguous reordering:

$$
n = \left( \left\lfloor\frac{t}{C_t}\right\rfloor N_h N_w + \left\lfloor\frac{h}{C_h}\right\rfloor N_w + \left\lfloor\frac{w}{C_w}\right\rfloor \right) \cdot 64 + (t \bmod C_t)\cdot C_h C_w + (h \bmod C_h)\cdot C_w + (w \bmod C_w)
$$

This guarantees: tokens in the same cube → contiguous indices → loaded onto the same SM in the FA kernel. The FastVideo source materializes this via three cached helpers:

```python
# fastvideo/attention/backends/video_sparse_attn.py
VSA_TILE_SIZE = (4, 4, 4)

@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(dit_seq_shape, tile_size, device):
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T*H*W).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T/ts)):
        for h in range(math.ceil(H/hs)):
            for w in range(math.ceil(W/ws)):
                ls.append(indices[t*ts:t*ts+ts, h*hs:h*hs+hs, w*ws:w*ws+ws].flatten())
    return torch.cat(ls)

@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(...):
    return torch.argsort(get_tile_partition_indices(...))

@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(dit_seq_shape, num_tiles, device):
    """Block sizes for the (possibly non-square) trailing tile."""
    ...
```

The `non_pad_index` mechanism handles ragged trailing tiles: when the latent dimensions are not exact multiples of 4, the last tile is partial; padding is inserted to make every tile literally 64 tokens, then the non-pad mask filters real tokens at the end.

### S.3.3 Coarse stage — the cube-level attention

```python
B, H, L, D = q.shape; block = 64; topk = 32
q_c = q.view(B, H, L//block, block, D).mean(dim=3)        # cube means
k_c = k.view(B, H, L//block, block, D).mean(dim=3)
v_c = v.view(B, H, L//block, block, D).mean(dim=3)
score = (q_c @ k_c.transpose(-2,-1)) / sqrt(D)             # cube×cube
score = softmax(score, dim=-1)
output_coarse = score @ v_c                                # full coarse attn output
output_coarse = repeat to L (broadcast each cube to its 64 tokens)

topk_vals, topk_idx = score.topk(topk, dim=-1)             # top-K=32 cubes per Q cube
```

So coarse stage simultaneously: (a) computes a global attention output `O_c` over pooled tokens, (b) emits the top-K cube indices that the fine stage will use. Computational cost is `(N/64)²` instead of `N²` — for HunyuanVideo's 115K tokens, that's `1798²` instead of `115200²`, a ~64× reduction.

### S.3.4 Fine stage — block-sparse FA over selected cubes

The fine stage runs the standard ThunderKittens block-sparse FA kernel with the top-K mask:
```python
hidden_states = video_sparse_attn(
    query, key, value,
    variable_block_sizes,    # for ragged final tile
    variable_block_sizes,
    cur_topk,                # top-K = (1 - sparsity) * num_cubes
    block_size=(4,4,4),
    compress_attn_weight=gate_compress
)
```
For default sparsity 0.875 (paper) on a 30K-token Wan-1.3B latent with `num_tiles ≈ 469`, `cur_topk = ceil(0.125 · 469) = 59`. For Wan-14B sparsity 0.9 with `num_tiles ≈ 4096`, `cur_topk = 410`.

### S.3.5 Output combination via learned gates

$$
O = O_c \odot G_c + O_f \odot G_f
$$
where `G_c, G_f` are learned linear projections of the input hidden states. Critically, **both** `O_c` (coarse attention output) and `O_f` (fine attention output) feed the final result — the coarse stage isn't just there to predict the mask, its output is *meaningful global context*. This is a key difference from MoBA/NSA where the pooled stage only routes.

### S.3.6 Coarse-stage kernel — fused softmax + top-K + index emit

The coarse stage operates on pooled tokens but cannot use plain FlashAttention because of the row-wise top-K. The published implementation (Section 2.4 of the paper, confirmed in the FastVideo source) fuses softmax → top-K selection → mask-to-block-index conversion into a single Triton kernel. Profile (Table 3 of arXiv:2505.13389):

| Op | w/o fusion (µs) | w/ fusion (µs) |
|---|---|---|
| QKᵀ | 0.046 | 0.046 |
| scale | 0.060 | 0.912 (fused with…) |
| softmax | 0.095 | …softmax… |
| topk | 0.869 | …and topk |
| PV | 0.045 | 0.045 |

So the fused kernel is ~37% faster on the coarse stage (1.115 → 1.003 µs measured). The full coarse stage is ~14% of total attention runtime even at 87.5% sparsity in the fine stage.

### S.3.7 Sparse-Distill training recipe [VERIFIED]

VSA is the first published *trainable* video DiT sparse attention. Two operating modes:

**Mode 1 — Sparse Adaptation.** Take pretrained Wan2.1-1.3B (full-attention checkpoint), fine-tune with the standard flow-matching loss only, gradually anneal `K = topK` from `B/L` (= full attention equivalent) down to `K=32` (≈91.2% sparsity). 4000 steps on 80K synthetic Wan-14B-generated clips at 448×832, 61 frames. 32 H200 GPUs, DDP, batch=1 per GPU, GA=2, lr 1e-5. **Total: ~10 hours** [VERIFIED Appendix C.5].
- Wan-1.3B result: VBench 82.77% vs full-fine-tuned 83.63% (statistically tied), DiT inference time **31s → 18s** (1.7×, **6× attention**).
- Wan-14B result (Appendix C.5): final sparsity 0.9, 200K synthetic 768×1280 / 77-frame clips, 64 H200 GPUs, batch=64 global, lr 1e-5, 4000 steps → **1274 s → 576 s e2e (2.21×)**, human preference matches dense.

**Mode 2 — Sparse Distillation.** Even more impressively: the student model in DMD2 (Improved Distribution Matching Distillation, [Yin 2024](https://arxiv.org/abs/2406.04103)) gets VSA *while the teacher remains dense*. Both objectives — few-step distillation *and* sparsity — are trained simultaneously with no extra hyperparameters. 64 H200, batch=1 per GPU, 12 hours, 4000 steps.
- Wan-1.3B 3-step + VSA 0.8: **50.9× e2e** [VERIFIED Appendix C.6]; **5-second video in approximately 5 seconds on a single H200**; **16 FPS on a single H100** at 3-step inference.
- This is the key result that motivates BLADE and FastWan: VSA is the *first* sparse method shown to compose with extreme step-distillation. Earlier methods (TeaCache, FBCache) that exploit timestep redundancy *fail* in the 3–4-step regime because there are barely any timesteps to redundantly cache.

### S.3.8 Sparsity per model in production

Verified from the FastVideo recipe configs:
- **Wan2.1-1.3B (FastWan):** VSA sparsity = **0.8** (top-K = 20% of cubes).
- **Wan2.1-14B:** VSA sparsity = **0.9** (top-K = 10% of cubes).
- **HunyuanVideo:** sparsity = **0.85**, top-K = 32 of `116²/64²`.

The paper's general-architecture ablations use 87.5% sparsity (top-K = 32 / 256 cubes for the 410M scaling-law model), while production sparse-distill uses model-size-tuned values (larger models tolerate higher sparsity better — counterintuitive but consistent with the 410M-vs-1.4B scaling curves).

### S.3.9 Scaling law (paper's main scientific claim)

VSA pretrained DiTs from 60M to 1.4B with up to 4×10²¹ FLOPs on 128 H200 (≈90K H200-hours). Loss curves (Figure 2 of the paper):
- VSA 87.5%-sparse 1.4B model trained at 4×10²¹ FLOPs achieves the **same diffusion loss** as a dense 1.4B model at the same FLOP budget while spending **2.53× fewer training FLOPs**.
- The Pareto frontier (loss vs FLOPs) is shifted by a constant 2.53× factor across all model sizes from 60M to 1.4B.
- Optimal top-K depends on both sequence length *and* training compute. For 16K sequence length, K=32 is optimal up to 4×10²⁰ FLOPs; at 1×10²¹ FLOPs and 61K seq_len, K=32 starts beating K=16. Conjecture: optimal K → full attention as compute → ∞ but the gap shrinks log-slowly.

### S.3.10 Kernel performance

ThunderKittens block-sparse FA achieves **~85% of FA3's MFU at long seq_len with head_dim=64** (Figure 4 of the paper). On Hunyuan-style 115K seq_len with 87.5% sparsity, the fine stage delivers ~7× over dense FA3, and the combined VSA stack (coarse + fine) ~6×. FlexAttention with the same 64×64 block-sparse mask achieves only ~2×, illustrating the value of the hand-written ThunderKittens kernel.

### S.3.11 Distributed `DistributedAttention_VSA` for Ulysses

```python
# fastvideo/attention/dit_attention.py (paraphrased)
class DistributedAttention_VSA:
    def forward(self, qkvg):
        # qkvg: [B, L_local, num_heads, D] post-A2A would be [B, L, H_local, D]
        # 1. head-shard A2A
        qkvg = sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)
        # 2. cube-aligned reorder (per-rank, since each rank now sees the full sequence)
        qkvg = self.attn_impl.preprocess_qkv(qkvg, attn_metadata)
        # 3. local VSA forward (coarse + fine)
        out = self.attn_impl.forward(*qkvg.unbind(...), attn_metadata)
        # 4. reverse cube reorder
        out = self.attn_impl.postprocess_output(out, attn_metadata)
        # 5. sequence A2A back to original layout
        out = sequence_model_parallel_all_to_all_4D(out, scatter_dim=1, gather_dim=2)
        return out
```

The `preprocess_qkv` call (lines 220–227 of `video_sparse_attn.py`) does the tile-padding and reorder; `postprocess_output` reverses it. The pattern composes cleanly with **Ulysses** because after the head-shard A2A each rank holds the full sequence for its head subset — VSA's tile partition operates per-rank on the full sequence with no cross-rank dependency.

**Ring + VSA does not exist**: each cube's top-K can pick anywhere in the sequence, so a ring rotation would have to all-gather KV at every step, defeating ring's overlap. The two known partial workarounds in production are (i) **db-SP** (cross-ref §S.18) which dynamically rebalances head-level imbalance under Ulysses, and (ii) **DSA** (ICLR 2026) which is a fused distributed sparse attention purpose-built for this gap.

### S.3.12 Ablations [VERIFIED Table 1 of the paper]

| Exp | Variant | Loss-Optimal | Loss-Overtrained | Notes |
|---|---|---|---|---|
| 1 | Compress KV (pool KV by 2×2×2, no Q pool) | 0.15281 | 0.14282 | Baseline pooling-only |
| 2 | Spatial-Temporal alternating | 0.13574 | 0.13034 | Open-Sora 1.x style |
| 3 | Spatial-Full (4 spatial + 1 full layer per group) | 0.13555 | 0.12811 | |
| 4 | Strided window | 0.13271 | 0.12716 | Swin-derived |
| 5 | **Full attention** | 0.13877 | **0.12703** | Dense baseline |
| 6 | **VSA** | **0.13162** | **0.12687** | **Wins both** |

Critically, with sufficient training (overtrained column), VSA beats full attention with 87.5% sparsity *and* 2.53× lower FLOPs. The "Compress KV" baseline (which is essentially the H2O / SnapKV pooling without the top-K stage) loses badly — pure pooling without selective recovery is insufficient.

**Locality priors are unnecessary** (Exp 11–13 of Table 1b): adding an explicit local stage to VSA's C+F architecture provides minimal benefit. This is striking — Radial Attention is *built* on the locality prior, but VSA learns it implicitly when needed.

**Mean pooling beats max and conv** (Exp 19–21): 0.13929 (max) and 0.27787 (conv, training instability) vs 0.13162 (mean).

### S.3.13 Comparison with NSA, MoBA, BiFormer (Appendix E)

- **vs MoBA:** MoBA only uses pooled attention for routing; VSA uses pooled output `O_c` directly. MoBA's variable-length FA implementation forces tile_size=512 (too coarse); VSA uses 64×64 from the beginning.
- **vs NSA:** NSA targets *causal* LLM decoding, which forces grouped-query attention to make per-query KV pooling viable. Video DiT is bidirectional → no GQA constraint → VSA pools both Q and K freely. NSA's sliding-window branch is unnecessary for video.
- **vs BiFormer:** BiFormer also has tile-to-tile coarse routing but discards the coarse output. VSA showed in ablation that the coarse output is critical.
- **vs DSV** ([Tan 2025](https://arxiv.org/abs/2505.07120)): DSV uses dedicated low-rank attention predictors trained in a multi-stage process. VSA achieves the same goal with simple pooling and end-to-end training. DSV's complexity is a major operational disadvantage; VSA is a single Triton kernel + a small additional cube-pooling step.

## S.4 Radial Attention — static O(N log N) energy decay

**Source:** [arXiv:2506.19852](https://arxiv.org/abs/2506.19852), Xingyang Li et al. (MIT/NVIDIA/Princeton/Berkeley/Stanford), NeurIPS 2025. [hanlab.mit.edu/projects/radial-attention](https://hanlab.mit.edu/projects/radial-attention). Code at [github.com/mit-han-lab/radial-attention](https://github.com/mit-han-lab/radial-attention). Length-extending LoRAs released for Wan2.1-14B, HunyuanVideo, Mochi 1.

### S.4.1 Spatiotemporal Energy Decay — the empirical observation

The Radial paper documents a clean physical-decay analogy for video DiT attention. Define spatial attention map = each token attends mainly to nearby tokens within adjacent frames; temporal attention map = each token attends mainly to same-spatial-position tokens across many frames. Both decay **exponentially** in the relevant distance (regression `R² > 0.985`):

$$
p_{js+l} \;\le\; C_{\text{rel}} \cdot e^{-\alpha|j-i_0| - \beta|l-k_0|} \cdot p_{i_0 s + k_0}
$$

where `(i_0, k_0)` is the query's `(frame, spatial_position)`, `α` is the temporal decay rate, `β` the spatial decay rate. Spatial heads have high `α`, low `β`; temporal heads have low `α`, high `β`. **Spatiotemporal Energy Decay** is the umbrella term that unifies both.

### S.4.2 Translating decay to compute density

Radial Attention's mask is purely a function of `(t, h, w)` with **two interlocking decay schedules**:

1. **Temporal density decay (in bands):** the attention map is divided into `2⌈log₂ max(f, 2)⌉ − 1` diagonal bands centered on the main diagonal (band 0). Bands are indexed `0, ±1, ±2, ...`. The compute density of band `±r` is `(1/2)^r` — the band immediately off-diagonal has half the density, the next outer has quarter, etc. Each successive outer band also *doubles* in width to keep total computation constant per band.
2. **Spatial density decay within each band:** for the frame-pair `(i, j)` falling into band `±r`, the spatial diagonal width is `floor(s / 2^r)`. When this would fall below 1, instead of a thinner diagonal, only every `ceil(2^r / s)`-th frame in the band is computed (subsampling diagonals).

Formally, the 4D mask `M̃ ∈ {−∞, 0}^{f×f×s×s}` satisfies `M̃[i,j,k,l] = 0` iff one of the conditions in the paper's Eq 4:

```
M̃[i,j,k,l] = 0  if  2^⌊log₂ max(|i−j|,1)⌋ ≤ s  AND  |k−l| + 1 ≤ s / 2^⌊log₂ max(|i−j|,1)⌋
M̃[i,j,k,l] = 0  if  |i−j| mod ⌈2^⌊log₂ max(|i−j|,1)⌋ / s⌉ = 0  AND  k = l
```

A first-frame **attention sink** (every token attends to all tokens in frame 0) is added on top, following StreamingLLM intuition.

### S.4.3 Complexity proof — `O(N log N) in time, quadratic in resolution`

Counting zeros in `M̃` (Appendix A.1 of the paper):
$$
\text{\#zeros in } \tilde M \;\le\; \underbrace{4s^2 f}_{\text{central band + sink}} \;+\; \underbrace{\sum_{r=1}^{\lfloor\log_2 s\rfloor} 2^{r+1}\,f \cdot \frac{2s^2}{2^r}}_{\text{wide bands}} \;+\; \underbrace{\sum_{r=\lfloor\log_2 s\rfloor + 1}^{\lceil\log_2 f\rceil - 1} 2^{\lfloor\log_2 s\rfloor + 1}\,f\,s}_{\text{narrow bands}}
$$
$$
\;\le\; 4s^2 f \log_2 f \;=\; 4s n (\log_2 n - \log_2 s)
$$
For fixed spatial resolution `s` and growing frame count `f`, this is **`O(N log N) in N = sf`** with respect to the number of frames. The complexity is still quadratic in spatial resolution `s`.

### S.4.4 Approximation error bound

Theorem from Section 4.2:
$$
\|\tilde p - p\|_1 \;\le\; C_{\text{rel}} \left[ \frac{8\, e^{-\beta(s/2 + 1)}}{(1 - e^{-\alpha})(1 - e^{-\beta})} \;+\; 4 \cdot \frac{1 + e^{-\beta}}{1 - e^{-\beta}} \cdot \frac{e^{-\alpha(s+1)}}{1 - e^{-\alpha}} \right] \;=\; O\!\left( C_{\text{rel}}\, e^{-\min(\beta/2, \alpha)\,s} \right)
$$
The total-variation L1 error decays *exponentially* with the spatial resolution `s` and the smaller of the two decay rates. Doubling `s` exponentially shrinks the error. Empirical measurement on Wan2.1-14B: Radial L1 error 3.9×10⁻³ vs SVG 4.4×10⁻³ vs STA 1.5×10⁻². Radial is the tightest sparse approximation among the three on Wan.

### S.4.5 Length-extending LoRAs (the operational killer feature)

Radial's static mask + a small LoRA on Q/K/V/O is enough to extend pretrained models to videos `2× to 4×` longer than their training horizon. Adapter rank 128, batch=1 per GPU + sequence parallelism (HunyuanVideo, Mochi) or batch=8 (Wan-14B), 8 H100, 16–21 hours wall clock for Hunyuan, 8–17 h Mochi, 15 h Wan2.1.

**Reported extension results [VERIFIED Table 2 of paper]:**

| Model | Length | Method | Sparsity | Train Time | Train Speedup | Inference Time | Inference Speedup | Vision Reward | VBench (S.C. / A.Q. / I.Q.) |
|---|---|---|---|---|---|---|---|---|---|
| Hunyuan | 125 (1×) | Original | 0% | – | – | 225s | – | 0.119 | 0.959 / 0.643 / 0.672 |
| Hunyuan | 253 (2×) | RIFLEx (training-free) | 0% | – | – | 797s | 1.00× | 0.128 | 0.969 / 0.622 / 0.614 |
| Hunyuan | 253 (2×) | Full LoRA (dense) | 0% | 45.0h | 1.00× | 797s | 1.00× | 0.124 | 0.955 / 0.616 / 0.648 |
| **Hunyuan** | **253 (2×)** | **Radial LoRA** | **80.8%** | **16.2h** | **2.78×** | **339s** | **2.35×** | **0.126** | **0.968 / 0.623 / 0.663** |
| Hunyuan | 509 (4×) | Full LoRA (dense) | 0% | 93.6h | 1.00× | 2895s | 1.00× | 0.133 | 0.977 / 0.590 / 0.635 |
| **Hunyuan** | **509 (4×)** | **Radial LoRA** | **88.3%** | **21.4h** | **4.37×** | **781s** | **3.71×** | **0.134** | **0.973 / 0.623 / 0.672** |
| Wan2.1-14B | 161 (2×) | Full LoRA | 0% | 28.0h | 1.00× | 5735s | 1.00× | 0.150 | 0.966 / 0.590 / 0.689 |
| **Wan2.1-14B** | **161 (2×)** | **Radial LoRA** | **73.6%** | **14.5h** | **1.93×** | **2847s** | **2.01×** | **0.145** | **0.981 / 0.607 / 0.677** |

In every row the Radial-LoRA student *matches or beats* the Full-LoRA dense baseline on Vision Reward and VBench while training in ~half the time and inferring in ~2–3.7× less time. The 4× length extension is essentially free.

### S.4.6 Default-length training-free results [VERIFIED Table 1]

At default length, Radial is competitive with SVG (PSNR 27.3 vs 27.2 on Hunyuan; 23.9 vs 23.2 on Wan2.1-14B) and beats STA's 26.7 PSNR; STA wins on raw speed (2.29× vs 1.88×) only because it's on FA3 while Radial is still on FA2 (the Radial paper notes upgrading to FA3 is orthogonal future work).

### S.4.7 LoRA compatibility with style LoRAs

A practically-important property: Radial's length-extending LoRA can be merged with existing style LoRAs (e.g., character or art-style LoRAs published on Civitai for HunyuanVideo) by simple weight-summing. Subtle style bias was observed (Appendix C.3) due to Radial's relatively small training set; a richer dataset would mitigate.

### S.4.8 B200 / parallelism

The mask is purely analytic in `(t, h, w)` — no per-call top-K, no per-call softmax. This makes Radial **the most parallelism-friendly sparse pattern in the survey**:

- **Ulysses:** trivially compatible (mask is identical across heads except where head-specialized; head sharding is independent).
- **Ring:** compatible if the chunk boundary respects the *temporal* axis — the mask is dense within ring chunks for the central band, and the sparse pattern in outer bands can be pre-routed.
- **B200:** the natural lowering is FlexAttention BlockMask. The mit-han-lab repo currently uses FlashInfer for inference and Block-Sparse-Attention (with FA2 backend) for training. **No Blackwell-specific Radial kernel published as of May 2026**, but the BlockMask path is straightforward — the mask precomputes once per resolution, then FlexAttention compiles it to a CUTLASS-DSL or CUTE-DSL kernel.
- **CUDA Graphs:** Radial's static mask is graph-capturable, unlike VSA / SVG which have per-step variability.

### S.4.9 Limitations

- Quadratic in spatial resolution `s` (only sublinear in temporal). For 4K+ frame side, additional spatial sparsification needed.
- The exponential-decay assumption is empirical — for unusual content (rapid scene cuts, panning) the energy may not actually decay at the assumed rate. Paper notes this as future work.
- Radial vs the NSA/MoBA family (which are dynamic and trainable): Radial is fixed and analytic, so it can't adapt to head-specialization the way VSA can. The trade-off is that Radial doesn't need the coarse-stage cost.

## S.5 SVG / SVG2 — online classification + semantic permutation

**Sources:** SVG — [arXiv:2502.01776](https://arxiv.org/abs/2502.01776), Haocheng Xi et al. (UC Berkeley/MIT/NVIDIA), ICML 2025. SVG2 — [arXiv:2505.18875](https://arxiv.org/abs/2505.18875), Shuo Yang et al., NeurIPS 2025 Spotlight. Code at [github.com/svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen).

### S.5.1 SVG — the spatial / temporal head taxonomy

SVG identifies that 3D full attention heads in pretrained video DiTs split cleanly into two sparse types (Figure 3 of the paper):

- **Spatial head:** focuses attention scores on spatially-local tokens within the same frame. Layout: block-wise along the diagonal of the attention matrix. Each token attends mainly to its `c_s` neighboring frames around itself. Sparsity ratio ≈ `c_s / N_frames`.
- **Temporal head:** focuses on tokens at the *same spatial position* across all frames. Layout: slash-wise with stride `L = tokens_per_frame`. Sparsity ≈ `c_t / L`.

In addition, both head types include **the text prompt tokens** and the **first frame** as always-attended (echoing StreamingLLM's attention sink hypothesis).

### S.5.2 Online profiling for head classification [VERIFIED Algorithm 1]

```python
# Sample 1% of input rows
indices = sample_indices(S, t)           # t ≈ S/100
Q_i = Q[:, :, indices, :]
mask_spatial = gen_spatial_mask()[:, :, indices, :]
mask_temporal = gen_temporal_mask()[:, :, indices, :]
# Compute three outputs on the sampled subset
O_full     = mask_attention(Q_i, K, V, None)
O_spatial  = mask_attention(Q_i, K, V, mask_spatial)
O_temporal = mask_attention(Q_i, K, V, mask_temporal)
MSE_s = (O_full - O_spatial).norm().mean(dim=(2,3))
MSE_t = (O_full - O_temporal).norm().mean(dim=(2,3))
best_mask_config = (MSE_s < MSE_t)         # per-head spatial vs temporal vote
```

Cost: `~3% of full attention` per call [VERIFIED Table 3]. Sensitivity: 1% sample → 31.1 PSNR on CogVideoX-v1.5-I2V; 100% sample (oracle) → 31.3 PSNR. So 1% is essentially lossless.

Sparse pattern is **highly dynamic across timesteps and prompts** — a head can be spatial for one prompt and temporal for the next. This is why an *offline* calibration (à la MInference) does not work well for video; SVG's runtime per-step classification is necessary.

### S.5.3 Hardware-efficient layout transformation

The temporal-head pattern is *not* immediately Tensor-Core-friendly because the attended tokens are non-contiguous (stride `L`). SVG's solution: transpose the token-major tensor into a frame-major one before running attention. After transpose, the temporally-aligned tokens are contiguous and FA can run normally. After attention, transpose back. Math is associative so the result is identical, and the transpose is cheap (data movement).

**Speedup (Figure 8 of the paper):** at 10% sparsity, layout transformation gives an additional **1.7× over naive sparse attention**, achieving **3.63× total** (close to the theoretical 10×).

### S.5.4 Reported numbers [VERIFIED Table 1]

| Model | Method | PSNR↑ | SSIM↑ | LPIPS↓ | FLOPs (PFLOPs) | Latency | Speedup |
|---|---|---|---|---|---|---|---|
| CogVideoX-v1.5 I2V (720p, 10s, 80f) | DiTFastAttn | 24.59 | 0.836 | 0.167 | 78.86 | 338s | 1.56× |
| | MInference | 22.49 | 0.743 | 0.264 | 84.89 | 357s | 1.48× |
| | PAB (cache) | 23.23 | 0.842 | 0.145 | 105.88 | 374s | 1.41× |
| | **SVG** | **28.17** | **0.915** | **0.104** | **74.57** | **237s** | **2.23×** |
| HunyuanVideo T2V (720p, 5.33s, 128f) | DiTFastAttn | 21.42 | 0.646 | 0.331 | 260.48 | 1238s | 1.82× |
| | Temporal-only | 25.85 | 0.857 | 0.175 | 259.10 | 1231s | 1.83× |
| | MInference | 23.16 | 0.823 | 0.163 | 293.87 | 1417s | 1.59× |
| | **SVG** | **29.55** | **0.907** | **0.127** | **259.79** | **1171s** | **1.92×** |
| | **SVG + FP8** | 29.45 | 0.906 | 0.128 | 259.79 | **968s** | **2.33×** |

Note SVG's PSNR on Hunyuan is **29.5** — well into the visually-indistinguishable regime — at **1.92×** speedup, beating MInference (23.2 PSNR / 1.59×) and DiTFastAttn (21.4 PSNR / 1.82×). With FP8 quantization stacked on top, **2.33× e2e at 29.5 PSNR**.

### S.5.5 SVG2 — semantic-aware permutation via k-means

SVG2 ([arXiv:2505.18875](https://arxiv.org/abs/2505.18875), Shuo Yang et al., NeurIPS 2025 Spotlight) addresses **two failure modes of SVG**:

1. **Inaccurate identification:** position-based clustering (e.g., adjacent tokens grouped into blocks for SpargeAttn) does *not* guarantee semantic similarity. Two physically-close objects (an apple and a cake) can have wildly different latent activations. Mean/max pooling of such a block produces a poor cluster representative → noisy block-level attention scores → bad top-k selection.
2. **Computation waste:** even with a *perfect* top-k oracle, scattered critical tokens require padding to fill 16-token Tensor Core tiles. Up to 80% wasted computation in pathological cases.

**SVG2's fix:** *cluster Q and K independently with k-means before identification*. After clustering, permute the tensor so each cluster's tokens are contiguous. Then block-sparse attention operates on dense clusters, no padding waste.

**Algorithm:**
```
For each (head, layer):
    Q = q_proj(x); K = k_proj(x); V = v_proj(x)
    Q_clusters, Q_centroids = kmeans(Q, C_q=100)
    K_clusters, K_centroids = kmeans(K, C_k=500)
    π_q = permutation that contiguates Q clusters
    π_k = permutation that contiguates K clusters
    Q', K', V' = π_q · Q, π_k · K, π_k · V
    # Centroid-based attention score estimate
    S_ij = centroid(Q_i) · centroid(K_j)ᵀ / sqrt(d)
    P̂_ij = |K_j| · exp(S_ij) / sum_k(|K_k| · exp(S_ik))
    # Top-p selection (target attention recall, not fixed K)
    sort P̂_ij descending; keep clusters until cumulative ≥ p
    O = π_qᵀ · BlockSparseAttn(Q', K', V', selected_clusters)
```

**Centroid cache for fast k-means.** k-means++ converges in 100+ iterations the first time, taking nearly as long as full attention. SVG2 caches centroids across denoising steps (consecutive steps have nearly identical activations [Liu 2024 TeaCache, Zhao 2024 PAB]). Re-initializing from the previous step's centroids reduces convergence to 2–4 iterations: **76× total k-means speedup** [Figure 7 of paper].

**Custom dynamic-block-size attention kernel.** Traditional FlashAttention requires fixed `(BLOCK_Q, BLOCK_K)`. SVG2's clusters are variable-size → need a custom kernel. The repo implements this for both FA2 (A100) and FA3 (H100) using sparse loading + dense compute (wgmma m64n64k16): per-token address offsets to scatter-load K/V into shared-memory contiguous form, then run dense mma. Achieves >85% of theoretical max performance.

**Reported numbers [VERIFIED Table 1]:**

| Model | Method | PSNR↑ | SSIM↑ | LPIPS↓ | VBench↑ | Density↓ | Speedup |
|---|---|---|---|---|---|---|---|
| Wan 2.1-I2V 14B 720p | SVG | 24.06 | 0.813 | 0.174 | 0.836 | 30.25% | 1.56× |
| | **SVG2** | **26.56** | **0.861** | **0.138** | **0.838** | **31.28%** | **1.58×** |
| | SVG2-Turbo | 24.51 | 0.812 | 0.179 | 0.836 | **14.13%** | **1.84×** |
| Hunyuan T2V 13B 720p | SVG | 29.16 | 0.905 | 0.120 | 0.845 | 29.86% | 1.91× |
| | XAttention | 28.89 | 0.898 | 0.120 | 0.839 | 39.32% | 1.56× |
| | **SVG2** | **30.45** | **0.910** | **0.117** | **0.852** | **25.45%** | **2.30×** |
| | **SVG2 + FP8** | 30.39 | 0.908 | 0.118 | 0.851 | 25.45% | **2.55×** |

SVG2 at **30.45 PSNR / 2.30×** on Hunyuan is the current 2026 SOTA for *training-free* sparse attention on video DiT.

### S.5.6 PR #50: SVG / SVG2 + Ulysses

The svg-project repo PR #50 added DeepSpeed-Ulysses support. The k-means runs on the full sequence per rank (post-A2A head shard), so each rank does its own k-means independently. This works because Ulysses A2A makes each rank see the full sequence for its head subset.

**Composability:** SVG + Ulysses ✓; SVG + Ring ✗ (k-means partitioning would have to span ranks); SVG + FP8 ✓ (verified additional 1.3× boost); SVG + TeaCache works in principle (TeaCache decides whether to *call* attention; SVG runs faster when called). SGLang's `compatibility_matrix` lists SVG2 as ✅ for Wan 2.1, Wan 2.2, HunyuanVideo, but ❌ for LTX-2 and ❌ for HunyuanVideo 1.5 (which uses SSTA instead).

## S.6 SLA — Sparse-Linear Attention (THU-ML, the FastVideo SageSLA)

**Source:** [arXiv:2509.24006](https://arxiv.org/abs/2509.24006), Jintao Zhang et al. (Tsinghua / UC Berkeley / SLA team), code at [github.com/thu-ml/SLA](https://github.com/thu-ml/SLA). FastVideo backend at `fastvideo/attention/backends/sla.py` (verified read above). The SageSLA variant uses SpargeAttn quantized kernels.

### S.6.1 The "decoupled rank" key observation

The SLA paper's central insight: **video DiT attention weights `P` decompose into two parts of distinct rank**:

- A small fraction (<10% of entries) with rank comparable to full attention — these are the *critical* weights, dominated by the largest entries of softmax.
- A large fraction (>90%) with extremely *low* rank — the *marginal* weights, well-approximated by linear attention.

This naturally suggests a *hybrid*:
$$
P = \underbrace{P \odot M}_{\text{sparse component (high-rank, small)}} + \underbrace{P \odot (1 - M)}_{\text{low-rank component (low-rank, large)}}
$$
Apply *exact O(N²) FA* to the first part, *O(N) linear attention* to the second, skip nothing. This explains the long-standing failure of pure linear attention on video DiT: the high-rank tail doesn't fit a low-rank approximation.

### S.6.2 The three-bucket classification

For each `(query block, key block)` pair, compute the compressed attention matrix:
$$
P_c = \mathrm{softmax}\!\left( \mathrm{pool}(Q) \cdot \mathrm{pool}(K)^\top / \sqrt{d} \right) \in \mathbb{R}^{(N/b_q) \times (N/b_{kv})}
$$
Classify each entry into one of three categories via the *compressed mask* `M_c`:
$$
M_c[i,j] = \begin{cases} +1 & \text{(critical) — top } k_h\% \text{ of row } i \\ 0 & \text{(marginal) — neither top nor bottom} \\ -1 & \text{(negligible) — bottom } k_l\% \text{ of row } i \end{cases}
$$
Then:
- **Critical (`+1`)**: full sparse FlashAttention — `O(N²)` complexity but only on `k_h%` blocks.
- **Marginal (`0`)**: linear attention with feature map `φ` — `O(N·d²)` complexity.
- **Negligible (`-1`)**: skip entirely.

For Wan2.1: paper recommends `k_h = 5%`, `k_l = 10%`, leaving 85% as marginal. Sparsity = 1 − 5% = 95%, but linear attention costs only ~0.5% of full attention's FLOPs, so **net efficiency ≈ 19.3× attention compute reduction**.

### S.6.3 Linear attention branch — softmax feature map

The linear attention branch uses a learnable activation function `φ`. Ablation in Table 2 of the paper:

| φ (feature map) | VA↑ | VT↑ | IQ↑ | OC↑ | AQ↑ | SC↑ | VR↑ |
|---|---|---|---|---|---|---|---|
| **softmax** | **76.96** | **83.92** | **62.2** | **23.6** | **55.9** | **93.1** | **0.048** |
| elu+1 | 75.50 | 81.01 | 62.8 | 23.5 | 55.3 | 92.9 | 0.034 |
| hedgehog | 74.59 | 82.62 | 61.9 | 22.5 | 54.3 | 93.2 | 0.035 |

Softmax `φ` wins. The linear branch computes `(φ(K)ᵀV)`, then `Z = rowsum(φ(K)ᵀ)`, then `φ(Q) (φ(K)ᵀV) / (φ(Q) · Z)`.

A learnable linear projection `proj_l` blends:
$$
O = O^s + \mathrm{proj}_l(O^l)
$$
**`proj_l` is initialized to zero** so SLA equals pure sparse attention at training start; the linear branch is gated in gradually. This is the same trick VSA uses for its coarse output gate.

### S.6.4 Quantitative results [VERIFIED Table 1]

| Method | VA↑ | VT↑ | IQ↑ | OC↑ | AQ↑ | SC↑ | VR↑ | FLOPs↓ | Sparsity↑ |
|---|---|---|---|---|---|---|---|---|---|
| Full Attention | 76.78 | 82.88 | 62.5 | 23.3 | 56.1 | 93.0 | 0.059 | 52.75 T | 0% |
| SpargeAttn-F (training-free) | 0.002 | 0.026 | 26.0 | 4.6 | 35.7 | 85.1 | -0.216 | 7.91 T | 85% |
| SpargeAttn-T (trainable) | 73.83 | 77.87 | 61.9 | 22.7 | 55.4 | 93.1 | 0.014 | 7.38 T | 84% |
| VMoBa | 32.33 | 35.79 | 58.0 | 18.8 | 46.2 | 89.9 | -0.175 | 7.91 T | 85% |
| VSA | 55.37 | 64.61 | 60.6 | 22.4 | 51.9 | 83.6 | -0.069 | 5.92 T | 89% |
| **SLA** | **76.96** | **83.92** | **62.2** | **23.6** | **55.9** | **93.1** | **0.048** | **2.74 T** | **95%** |

**SLA reaches 95% sparsity with quality matching dense full attention**, while VSA at 89% sparsity loses 21pt VA. The hybrid design enables this.

### S.6.5 Kernel performance

[Figure 6 of the paper, RTX 5090 + Wan2.1-1.3B]:
- Forward: SLA **13.7×** faster than FlashAttention2; **1.93×** faster than VSA at 95% sparsity; **3.36×** faster than VMoBa at 95%.
- Backward: SLA **6.8×** faster than FA2.
- E2E: attention 97s → 11s (**8.8× attention reduction**), e2e generation **2.2×** speedup.
- Fine-tuning cost: 2000 steps, batch=64, <0.1% of pretrain cost.

### S.6.6 Optimization tricks (Appendix A.3)

1. **Lookup table for highly-sparse `M_c`**: when sparsity > 90%, scanning whole rows of `M_c` becomes memory-bound. Pre-compute the nonzero positions per row/column → access only the LUT.
2. **Pre-aggregation for linear attention**: pre-compute `Σ_j h_j` and `Σ_j z_j`, then *subtract* the contributions from `M_c[i,j] ≠ 0`. With 90% sparsity, 90% of the additions become 10% of subtractions.
3. **Method of Four Russians**: for moderate sparsity (~50%), group `{h_j, z_j}` into segments of `g` blocks, precompute all `2^g` subset sums. Theoretical compute reduction by `1/g`.

### S.6.7 SageSLA — FastVideo's quantized fusion (VERIFIED from source)

The FastVideo `SageSLAAttentionImpl` class (lines 376–557 of `sla.py`) fuses SLA with SpargeAttn's quantized kernels. The flow:

```python
# 1. Compute block-sparse pattern (same as plain SLA)
sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=128, BLKK=64)

# 2. Quantize Q, K to INT8
q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, BLKQ, BLKK)
lut_triton, valid_block_num = block_map_lut_triton(sparse_map)

# 3. Quantize V to FP8 with transpose-pad-permute fusion
v_transposed_permutted = fused.transpose_pad_permute_cuda(v, ...)
v_fp8 = fused.scale_fuse_quant_cuda(v_transposed_permutted, ...)

# 4. Sparse attention on quantized tensors (architecture-dispatched)
if arch == "sm90":
    qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
        q_int8, k_int8, v_fp8, o_s, lut_triton, valid_block_num,
        q_scale, k_scale, v_scale, 1, False, 1, scale)
else:  # sm89 / 5090
    if SAGE2PP_ENABLED:
        # f16 accumulator with PV threshold (Sage 2++)
        qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q_int8, k_int8, v_fp8, o_s, lut_triton, valid_block_num, pvthreshold,
            q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
    else:
        qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(...)

# 5. Linear branch in BF16
o_l = self._calc_linear_attention(self.feature_map_q(q), self.feature_map_k(k), v)

# 6. proj_l blend
output = (o_s + self.proj_l(o_l)).to(original_dtype)
```

Block sizes are arch-dispatched: SM90 uses `(BLKQ, BLKK) = (64, 128)`; SM89/SM120 uses `(128, 64)`. The SM89 path additionally uses the **PV threshold** trick from Sage 2++: a per-head threshold above which a block's PV contribution is skipped because it's dominated by already-accumulated blocks.

**Reported speedups** (TurboDiffusion technical report):
- Wan2.1-T2V-1.3B-480P: **184 s → 1.9 s (~97×)** with the full TurboDiffusion stack (rcm + SageSLA + W8A8 + offload + final-version polish).
- Wan2.2-I2V-A14B-720P: **4549 s → 38 s (~120×)**.
- Wan2.1-T2V-14B-720P: ~200×.
- The "rcm + SageSLA + final-version → 33.3× speedup" result (TurboDiffusion Fig 4) is the SageSLA-only contribution (33.3× of the ~100× total).

**TurboDiffusion `layerwise_offload` mode [VERIFIED naming]:** Cross-reference §2 of the parent report. Keeps at most 2 transformer layers resident on GPU; prefetches next layer over a CUDA stream while current runs; releases previous to CPU. SGLang PR #17693: **88.79 → 82.48 s (1.08×)** and peak VRAM **40.22 → 10.81 GB** on a Wan setup. Configurable via `--dit-offload-prefetch-size`.

## S.7 BSA — Bidirectional Sparse Attention (ByteDance)

**Source:** [arXiv:2509.01085](https://arxiv.org/abs/2509.01085), Chenlu Zhan et al. (ByteDance), Wen Li project leader, Hao Zhang advisor. FastVideo backend at `fastvideo/attention/backends/bsa_attn.py` (verified read above).

### S.7.1 The bidirectional pruning insight

Existing sparse attention prunes only one side: VSA / MoBA / NSA all prune *KV blocks* per Q. BSA observes that *queries also have redundancy* — within a 3D block, many queries are spatially-similar to the block center (e.g., uniform background pixels) and produce nearly-identical attention rows. **Pruning queries within blocks gives an additive sparsity gain on top of KV pruning** (orthogonal axes), and the resulting attention is 17.79× faster on training, 6.2× on training-free inference at 153K tokens.

The Figure 2 visualization in the paper is striking: in a video where two adjacent frames share a static background, full attention's important-query distribution is identical between frames (lots of duplicate work). BSA's query-sparse attention focuses on different *informative* tokens in each frame, primarily moving foreground regions.

### S.7.2 Three-component algorithm (verified against `bsa_attn.py`)

**Step 1: 3D block partition** — `BSA_TILE_SIZE = (4, 4, 4)`, identical to VSA. The same `get_tile_partition_indices` and `get_reverse_tile_partition_indices` cached helpers (compare to the VSA source — same structure).

**Step 2: Query sparsification (`_prune_queries`)** — for each block:
```python
center_idx = S // 2                                    # block-center query
center = q_blocks[:, :, :, center_idx:center_idx+1, :]
similarity = (F.normalize(q_blocks, dim=-1)
              * F.normalize(center, dim=-1)).sum(dim=-1)
# Keep the LEAST similar tokens (most informative)
_, indices = similarity.topk(keep_size, dim=-1, largest=False)
indices, _ = indices.sort(dim=-1)
sparse_q = torch.gather(q_blocks, 3, indices.unsqueeze(-1).expand(-1,-1,-1,-1,D))
```
With `keep_ratio = 0.5`, half the queries per block survive. The kept queries are precisely those *least similar* to the center — i.e., the most informative for that local region.

**Step 3: KV block selection (`_select_kv_blocks`)** — mean-pool to block level, compute block-level attention scores, admit blocks until cumulative softmax mass exceeds threshold:
```python
q_repr = sparse_q.mean(dim=3)                          # [B, H, N, D]
k_repr = k_blocks.mean(dim=3)
scores = (q_repr @ k_repr.transpose(-1, -2)) / (D**0.5)
block_attn = F.softmax(scores, dim=-1)
sorted_attn, sorted_idx = block_attn.sort(dim=-1, descending=True)
cumsum = sorted_attn.cumsum(dim=-1)
keep_sorted = cumsum < cumulative_threshold            # threshold default 0.9
# Enforce minimum num blocks (at least 4)
kv_mask = scatter back to original indexing
```

The threshold `p` is computed dynamically (not a fixed `K`):
$$
p = \mathrm{mean}(S_b) + \mathrm{std}(S_b) \cdot U(1 - k/n)
$$
where `U(·)` is the quantile function. Then admit the minimal index set whose cumulative softmax mass `≥ p`. The **dynamic threshold** is the key advantage over fixed top-K — different queries need different numbers of KV blocks. From the FastVideo source, `bsa_kv_cumulative_threshold = 0.9` and `bsa_min_kv_blocks = 4` are defaults.

**Step 4: Sparse attention (`_compute_sparse_attention`)** — per (batch, head, q-block), gather only the selected KV blocks and run dense attention. The FA-varlen path (`_flash_attn_single_mask` in the source) handles the case where all heads in a batch share the same KV mask (fast path); `_flash_attn_single_head` handles per-head masks (slower correct path). Both paths flatten into `(num_kv_tokens, num_heads, D)` and call `flash_attn_varlen_func`.

**Step 5: Reconstruct pruned positions (`_reconstruct_pruned`)** — pruned query positions get nearest-kept-token's output (vectorized nearest-neighbor scatter). This restores the original sequence length for downstream layers.

### S.7.3 Reported numbers [VERIFIED Tables 1, 2, 4]

**Training-based BSA on Wan2.1-1.3B:**

| Seq_len | Method | Sparsity | TextC | BGC | IQ | SC | FLOPs (T) | SpeedUp |
|---|---|---|---|---|---|---|---|---|
| 23K | Full Attn | – | 32.71% | 95.12% | 64.33% | 92.34% | 1.51×10¹² | – |
| 23K | **BSA** | **0.93** | 32.79% | 95.22% | 64.29% | 92.39% | **1.05×10¹¹** | **12.85×** |
| 153K | Full Attn | – | 34.76% | 93.26% | 65.91% | 93.79% | 6.99×10¹³ | – |
| 153K | **BSA** | **0.95** | 34.93% | 93.41% | 66.03% | 94.13% | **3.49×10¹²** | **17.79×** |

**Training-free BSA (zero fine-tuning):** 4.6× speedup at 23K, 6.2× at 153K, with VBench *exceeding* full attention on 3 of 4 metrics. The dynamic threshold mechanism makes it adapt to each new prompt without per-prompt calibration.

**Comparison with MoBA and VSA at the same sequence length:**

| Seq_len | Method | Sparsity | TextC | BGC | IQ | SC | SpeedUp |
|---|---|---|---|---|---|---|---|
| 23K | MoBA | 0.80 | 32.56% | 95.14% | 64.14% | 92.05% | 1.2× |
| 23K | VSA | 0.87 | 32.65% | 95.03% | 64.25% | 92.21% | 4.5× |
| 23K | **BSA** | **0.93** | **32.79%** | **95.22%** | **64.29%** | **92.39%** | **12.85×** |
| 153K | MoBA | 0.80 | 34.34% | 93.05% | 65.34% | 93.49% | 2.3× |
| 153K | VSA | 0.87 | 34.72% | 93.22% | 65.87% | 93.72% | 6.2× |
| 153K | **BSA** | **0.95** | **34.93%** | **93.41%** | **66.03%** | **94.13%** | **17.79×** |

BSA is the current sparsity champion among trained-from-scratch methods at long sequence lengths.

**Annealing schedule:** training begins with full attention, then *every 30 steps* sparsity is increased by 0.03 until reaching 0.9. Top-K reduces from `n_blocks` to `0.1·n_blocks`. Window size for query selection: `(2,2,2)` within `(4,4,4)` block. 30000 steps total. NVIDIA H100.

### S.7.4 Ablation [VERIFIED Table 3]

| Component | Sparsity | VBench Quality | SpeedUp |
|---|---|---|---|
| Sparse Query (50% keep) only | 0.5 | matches full | 1.96× |
| + Window-based key selection | 0.5 | slight ↑ | 1.98× |
| Sparse KV (fixed threshold) only | 0.86 | matches full | 6.05× |
| Sparse KV (dynamic threshold) only | 0.89 | matches full | 6.12× |
| **Sparse Q + Sparse KV** (default) | **0.93** | **matches/exceeds** | **12.85×** |

The two axes are *additive* (12.85× ≈ 1.98× × 6.12× under noise), a clean orthogonal-decomposition result.

### S.7.5 User study (BSA vs VSA)

[Figure 8 of the paper] User preference study on 210 randomly-drawn MovieGen-bench prompts, both 1.3B and 14B Wan2.1 models. **Users consistently prefer BSA over VSA at both scales**, with the gap widening for the 14B model. This is the only study that puts a published trained-sparse method ahead of VSA on subjective video quality at high sparsity.

### S.7.6 Limitations / status

- The published BSA backend (`bsa_attn.py` in FastVideo) is a **pure-PyTorch reference** with FlashAttention varlen as the inner kernel. There is no custom Triton or CUTLASS BSA kernel yet — the implementation is correct but slow relative to a hand-written sparse kernel. The 6.2× training-free / 17.79× training claims above are achieved with a custom Triton kernel from the BSA repo (not yet open-sourced as of May 2026).
- BSA does not yet support GQA — `if num_kv_heads is not None and num_kv_heads != num_heads: raise ValueError(...)` (see source).
- BSA is bidirectional only — `if causal: raise ValueError(...)`.

## S.8 NSA — Native Sparse Attention (DeepSeek's design template)

**Source:** [arXiv:2502.11089](https://arxiv.org/abs/2502.11089), Jingyang Yuan et al. (DeepSeek-AI / Peking U / U Washington), ACL 2025 Best Paper Honorable Mention.

NSA is the DeepSeek paper that *invented* the trainable two-stage sparse attention pattern that VSA, BSA, SLA, and MoBA-derivatives all draw on. It is LLM-targeted (causal decoding) but the architectural template translates directly to video DiT.

### S.8.1 Three parallel attention branches sharing Q

For each query token `q_t`, NSA computes three different `(K, V)` representations and combines them with learned gates `g_t^c`:

$$
o_t^* = \sum_{c \in \{cmp, slc, win\}} g_t^c \cdot \mathrm{Attn}(q_t, \tilde K_t^c, \tilde V_t^c)
$$

1. **Compressed (cmp)** — `(K, V)` mean-pooled (or learned-MLP-pooled with intra-block position encoding) to a coarse sequence:
   $$
   \tilde K_t^{cmp} = \{\varphi(k_{id+1:id+l}) \mid 0 \le i \le \lfloor (t-l)/d \rfloor\}
   $$
   where `l` is block length, `d` is sliding stride, `φ` is a learnable MLP.
2. **Selected (slc)** — top-k tokens from the compressed branch's attention scores, then *fine-grained* attention on those selected tokens:
   $$
   \tilde K_t^{slc} = \{k_i : i \in \mathrm{TopK}(q_t \cdot \tilde K_t^{cmp})\}
   $$
3. **Sliding-window (win)** — fixed local window for recency bias.

Each branch is `O(N)` per query. Total cost is sub-quadratic.

### S.8.2 Hardware-aligned design

NSA enforces **arithmetic-intensity-balanced** access: in GQA, all query heads in a group share the same selected blocks, so the per-group KV-cache bandwidth is the union of all heads' selections. NSA constrains all heads in a group to make a *common* selection (not per-head), trading some expressiveness for a 1.5–2× decoding bandwidth win. This is the key engineering decision that makes NSA *actually fast* in deployment, not just in theory.

### S.8.3 Reported numbers (LLM)

64K-length training and inference on a 27B transformer pretrained on 260B tokens:
- **Decoding: 11.6× speedup** vs full attention.
- **Forward: 9.0×** speedup.
- **Backward: 6.0×** speedup.

**Critical performance claim**: pre-training with NSA achieves **equal or better** average score on general benchmarks, long-context tasks, and chain-of-thought reasoning vs full attention. This contradicts the conventional wisdom that sparsity costs quality — *natively trained* sparsity actually improves it, presumably because the sparse structure constrains the model toward a better inductive bias.

### S.8.4 Why NSA matters for video DiT

VSA's design draws three things from NSA:
1. The two-stage (coarse compress / fine select) architecture.
2. End-to-end trainability through the top-K via differentiable gating.
3. Hardware-aligned tile sizes.

But NSA's per-group GQA constraint *isn't* needed for video DiT (no causal decoding), which is why VSA can use per-head pooling freely. NSA's sliding-window branch is also unnecessary for video (Radial / STA already cover that pattern with a static mask). VSA effectively *simplifies* NSA by removing both constraints.

DeepSeek's productionization of NSA in DeepSeek-V3.2 (Feb 2025) demonstrated that trainable sparse attention is now a credible LLM scaling path. The video-DiT community absorbed this lesson within 3 months (VSA NeurIPS 2025 was on arXiv by May 2025).

## S.9 SpargeAttn — universal training-free sparse + quantized

**Source:** [arXiv:2502.18137](https://arxiv.org/abs/2502.18137), Jintao Zhang et al. (THU-ML), ICML 2025. Code at [github.com/thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn). The trainable variant `SpargeAttn-T` is referenced in the SLA paper.

### S.9.1 Universal training-free sparse attention

SpargeAttn's claim: **a single sparse-attention algorithm that works on language, image, and video models without per-task patterns**. Pre-2025 sparse methods were all model-specific (StreamingLLM for streaming LLMs, MInference for long-prefill LLMs, DiTFastAttn for image DiT, ...). SpargeAttn unifies via two ideas:

### S.9.2 Stage 1: selective token compression for sparse prediction

The key insight: **most blocks in the QK matrix have high *self-similarity* across rows/columns** (Fig 4 of the paper). For *self-similar blocks* (e.g., uniform background patches), mean-pooling to a single representative token is accurate. For *non-self-similar blocks* (sharp transitions), mean-pooling loses critical info — so SpargeAttn keeps these blocks dense ("fix blocks"), only compresses self-similar ones ("selective blocks").

Algorithm:
```
For each (Q_block, K_block):
    s_qi = CosSim(Q_i)               # average cosine similarity within block
    s_kj = CosSim(K_j)
    q_i = mean(Q_i, axis=0)            # block representative
    k_j = mean(K_j, axis=0)
    Ŝ[i,j] = q_i · k_jᵀ / sqrt(d)
    if s_kj < θ: Ŝ[:,j] = -∞          # skip non-self-similar K blocks (force compute)
    P̂[i] = softmax(Ŝ[i])
    M_g[i,:] = TopCdf(P̂[i], τ)        # cumulative-mass top selection
    if s_qi < θ: M_g[i,:] = 1          # force compute for non-self-similar Q
    if s_kj < θ: M_g[:,j] = 1          # force compute for non-self-similar K
```
The "TopCdf" function: sort `P̂[i]` descending, accumulate until reaching `τ · sum(P̂[i])`, mark those positions as 1. Default `τ = 0.95`, `θ = 0.5`.

### S.9.3 Stage 2: sparse warp-level online softmax

Inside the sparse FA kernel, after computing `S_ij = Q_i K_jᵀ` for an admitted block, check at warp granularity whether `max(m_local − m_running) > λ` for the warp's row range. If not, skip the `P_ij V_j` matmul too — the block's contribution is dominated by already-accumulated blocks.

This second stage yields an *extra* skip on top of Stage 1's mask, only at warp granularity (so no global mask materialization needed). Default `λ = 0.5–1.0` depending on the model.

### S.9.4 Quantized integration

Both stages are integrated into the SageAttention framework:
- `spas_sage2_attn_meansim_topk_cuda` — the Stage-1 prediction kernel
- `block_sparse_sage2_attn_cuda` — the Stage-2 sparse-warp kernel

For Hopper SM90, the default kernel symbol called by FastVideo is `qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90` (verified in `sla.py` line 528). For SM89 (RTX 5090, RTX 4090) and Sage 2++ paths, the f16-accum variant `qk_int8_sv_f8_accum_f16_*` is used with the `with_pv_threshold` suffix that bakes the Stage-2 PV check into the kernel.

### S.9.5 Reported numbers

- **Mochi (L40 GPU): 2.5–5× attention, 1.83× e2e** (Fig 1 of the paper, [VERIFIED]).
- With Sage 2++ kernel swap: additional ~30% on top.
- Universal: works on Stable Diffusion 3, FLUX, Mochi, Hunyuan, language models with no per-model pattern tuning.

### S.9.6 Composability

- **Pure Ulysses:** ✓ (each rank does its own Stage-1 prediction).
- **Ring:** ✗ — blocks span the full sequence; admitting blocks requires global cumulative softmax. **No published Ring + SpargeAttn fused kernel.**
- **TP:** ✓ trivially.
- **db-SP measured the head-imbalance under SpargeAttn + Ulysses to be `ρ_s = 1.428`** — the highest in their study, indicating pronounced workload imbalance across heads even after Ulysses head-shard. db-SP's head-level greedy partitioning brings this to ~1.05.

### S.9.7 Limitations as a video-DiT default

- SpargeAttn-F (training-free variant) collapses on Wan-1.3B: VA = 0.002, VR = -0.216 [VERIFIED Table 1 of the SLA paper] — i.e., the non-similar-block detection misclassifies for video and Stage 1 over-prunes. The SLA paper notes this and recommends the trainable SpargeAttn-T which recovers VA = 73.83 (close to dense 76.78).
- SVG2's k-means semantic permutation is the modern improvement: instead of relying on intra-block similarity, *create* high-similarity blocks by explicit clustering. SVG2 + FP8 currently dominates SpargeAttn-T on every Hunyuan benchmark.

## S.10 MoBA / VMoBA — block-sparse top-k gating for video

### S.10.1 MoBA (LLM background)

**Source:** [arXiv:2502.13189](https://arxiv.org/abs/2502.13189), Enzhe Lu et al. (Moonshot AI / Tsinghua / Zhejiang Lab/U). Code at [github.com/MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA).

MoBA = "Mixture of Block Attention" — applies the MoE top-k gating principle to attention. Each query attends only to its top-k *blocks* (not top-k tokens), where block selection is via mean-pooled affinity:
$$
s_i = \langle q,\, \mathrm{mean\_pool}(K[I_i]) \rangle
$$
$$
g_i = 1 \text{ if } s_i \in \mathrm{TopK}(\{s_j\}, k) \text{ else } 0
$$
$$
I = \bigcup_{g_i > 0} I_i
$$

Causal version: `s_i = -∞` for any block index `i > pos(q)`. The current block (containing `q`) is always selected (forced). Block size `B = N/n` is a hyperparameter; Moonshot's Kimi production uses 512 for >1M-token contexts.

**Key advantage over fixed sliding-window:** zero-bias structure, model learns where to attend. **Key advantage over MInference's offline calibration:** dynamic per-query, not pre-decided per-head.

**Production deployment at Kimi:** supports >1M-token requests in production. Hybrid full/MoBA layers train from scratch for best results, but pretrained dense models can also switch to MoBA at inference with light calibration.

### S.10.2 VMoBA — video adaptation

**Source:** [arXiv:2506.23858](https://arxiv.org/abs/2506.23858), Jianzong Wu et al. (Kling Team Kuaishou / Peking U), ICLR 2026. Code at [github.com/KwaiVGI/VMoBA](https://github.com/KwaiVGI/VMoBA). FastVideo backend at `fastvideo/attention/backends/vmoba.py` (verified read above), using kernels from `fastvideo_kernel.moba_attn_varlen`.

Kuaishou's Kling team adapted MoBA for video DiT with **three changes** based on three observations of pretrained Wan2.1-1.3B:

**Observation 1 → Layer-wise Recurrent 1D-2D-3D Block Partition.** Different layers in a video DiT have different attention patterns:
- Layer 27 of Wan2.1-1.3B: **temporal-1D** — queries primarily attend along the time axis.
- Layer 3: **spatial-2D** — queries attend within their frame.
- Layer 20: **spatio-temporal-3D** — local 3D volumes.

VMoBA cycles its block partition through `{1D, 2D, 3D}` across layers:
$$
\mathbf{K}^\prime = \begin{cases}
\text{rearrange to } (N_{b_1}^T)(s_{b_1}^T \cdot H \cdot W), & l \bmod 3 = 0 \quad\text{(temporal)} \\
\text{rearrange to } (N_{b_2}^H \cdot N_{b_2}^W)(T \cdot s_{b_2}^H \cdot s_{b_2}^W), & l \bmod 3 = 1 \quad\text{(spatial)} \\
\text{rearrange to } (N_{b_3}^T \cdot N_{b_3}^H \cdot N_{b_3}^W)(s_{b_3}^T \cdot s_{b_3}^H \cdot s_{b_3}^W), & l \bmod 3 = 2 \quad\text{(3D)}
\end{cases}
$$
Loop is `temporal_layer + spatial_layer + st_layer` (default `1+1+1`). Cycling through dimensions is more compute-efficient than always-3D and matches the natural layer-specialization of pretrained DiTs.

**Observation 2 → Global Block Selection.** Different queries have different importance — some background-region queries inherently have *low* affinity with all keys; allocating the same `K` blocks per query wastes compute. VMoBA computes the full `(s × N_b)` similarity matrix for the head, then takes the global top-K from it (instead of per-query top-K):
$$
M_i = \mathrm{TopkMask}(\mathbf{q}_i \mathbf{b}_i^T, k)
$$
This automatically allocates more KV blocks to "important" queries.

**Observation 3 → Threshold-based Block Selection.** Different attention heads have different concentration levels (Fig 5). Some heads concentrate on a few blocks (sharp); others diffuse across many. VMoBA picks `K` per-head dynamically:
$$
k = \min\!\left\{ k^\prime \,\middle|\, \sum_{j=1}^{k^\prime} \mathrm{Sorted}(\hat S_j) \ge \tau \right\}
$$
Default `τ = 0.25`. The first few timesteps (25%) use full attention as warmup — same convention as SVG and Radial.

### S.10.3 Reported numbers [VERIFIED Tables 1, 2]

**Training-free inference:**

| Video Size | Method | Sparsity | PSNR | TextC | BGC | IQ | SC | FLOPs | Latency |
|---|---|---|---|---|---|---|---|---|---|
| 81×480×832 (33K) | FullAttn | – | – | 27.85% | 94.47% | 62.06% | 90.70% | 282.64T | 103s |
| 81×480×832 (33K) | DiTFastAttn | 0.50 | 22.67 | 27.34% | 91.68% | 60.27% | 90.57% | 182.73T (1.54×) | 89s (1.18×) |
| 81×480×832 (33K) | SVG | 0.50 | 12.64 | 27.30% | 93.67% | 60.98% | 89.84% | 182.92T (1.54×) | 90s (1.17×) |
| 81×480×832 (33K) | MoBA | 0.25 | 15.54 | 28.47% | 94.84% | 60.06% | 89.90% | 133.50T (2.12×) | 126s (0.83×) |
| 81×480×832 (33K) | **VMoBA** | **0.31** | **16.00** | 27.62% | 93.74% | 60.05% | 89.65% | 149.00T (1.90×) | **104s (1.01×)** |
| 81×720×1280 (76K) | FullAttn | – | – | 27.99% | 93.74% | 63.87% | 92.56% | 1246.78T | 406s |
| 81×720×1280 (76K) | **VMoBA** | **0.31** | 18.80 | 28.06% | 92.85% | 64.39% | 92.08% | 519.75T (2.40×) | **300s (1.35×)** |

VMoBA's inference speedup over MoBA (1.01× → 1.35×) comes from the layer-recurrent partition reducing the *number* of total blocks (1D and 2D partitions have far fewer blocks than 3D), which speeds up the gating computation.

**Training-based (fine-tuning Wan2.1-1.3B):**

| Video Size | Method | Sparsity | TextC | Dynamic | BGC | IQ | SC | FLOPs | Train Time |
|---|---|---|---|---|---|---|---|---|---|
| 93×576×1024 (55K) | FullAttn | – | 24.61% | 61.58% | 94.69% | 69.49% | 90.86% | 705.02T | 276 GPU-hrs |
| 93×576×1024 (55K) | MoBA | 0.25 | 23.06% | **5.80%** ⚠️ | 97.60% | 63.73% | 94.30% | 282.69T (2.49×) | 226 GPU-hrs |
| 93×576×1024 (55K) | **VMoBA** | **0.19** | 25.88% | 56.91% | 96.76% | 67.45% | 94.72% | 248.68T (2.83×) | **187 GPU-hrs (1.48×)** |
| 141×480×832 (56K) | **VMoBA** | **0.18** | 23.71% | 31.36% | 93.06% | 67.66% | 93.78% | 248.39T (2.92×) | **182 GPU-hrs (1.44×)** |

⚠️ MoBA's training-based result is striking: **Dynamic Degree drops to 5.80%** — the generated videos are nearly static. The 1D partition cannot capture spatio-temporal motion, so the model collapses to a static-image generator. VMoBA's 1-2-3D rotation rescues this entirely (56.91% Dynamic, comparable to dense full attention's 61.58%).

### S.10.4 Composability (FastVideo source)

The FastVideo `VMOBAAttentionImpl` (verified at `fastvideo/attention/backends/vmoba.py`) processes per-layer based on `self.layer_idx = self._get_layer_idx(prefix)`:

```python
moba_layer = self.layer_idx - attn_metadata.first_full_layer
loop_layer_num = (attn_metadata.temporal_layer
                  + attn_metadata.spatial_layer
                  + attn_metadata.st_layer)
if moba_layer % loop_layer_num < attn_metadata.temporal_layer:
    moba_chunk_size = attn_metadata.temporal_chunk_size
    moba_topk = attn_metadata.temporal_topk
elif moba_layer % loop_layer_num < attn_metadata.temporal_layer + attn_metadata.spatial_layer:
    moba_chunk_size = attn_metadata.spatial_chunk_size
    moba_topk = attn_metadata.spatial_topk
else:
    moba_chunk_size = attn_metadata.st_chunk_size
    moba_topk = attn_metadata.st_topk

query, chunk_size = process_moba_input(query, attn_metadata.patch_resolution, moba_chunk_size)
key, chunk_size = process_moba_input(key, ...)
value, chunk_size = process_moba_input(value, ...)
hidden_states = moba_attn_varlen(query, key, value, cu_seqlens=..., max_seqlen=...,
                                  moba_chunk_size=chunk_size, moba_topk=moba_topk,
                                  select_mode='threshold', simsum_threshold=0.25,
                                  threshold_type='query_head')
```

`first_full_step = 12, first_full_layer = 0` defaults: first 12 denoising steps and layer 0 use full attention; after step 12 the layer-wise rotation kicks in.

## S.11 MInference 1.0 / 2.0 — pattern library for prefill

**Source:** [github.com/microsoft/MInference](https://github.com/microsoft/MInference). NeurIPS'24 Spotlight + ICLR'25 + ICML'25. Original arXiv [2407.02490](https://arxiv.org/abs/2407.02490).

MInference is the LLM-targeted, *offline-calibrated* sparse attention library. Per-head pattern is calibrated once at install via a small probe set, then used at runtime with negligible overhead. Three pattern types in MInference 1.0:

1. **Vertical-Slash (VS)** — predominant in long-context LLMs. Each head attends to a few specific *vertical lines* (always-attended tokens) plus *slash lines* (constant-stride patterns). Approximation: top-`k_v` vertical positions + top-`k_s` slash positions.
2. **A-shape (Sink)** — first few tokens always attended (the StreamingLLM-discovered sink pattern), plus a sliding window.
3. **Block-Sparse** — generic block pattern with pre-computed mask.

Per-head pattern is selected offline from the three types via an MSE-vs-full-attention probe. Reported: **up to 10× pre-fill speedup on A100, 1.64× at 64k → 15× at 1M tokens**.

**MInference 2.0** (ICLR'25 / ICML'25):
- **Modality-aware Permutation Sparse Attention (MMInference)** for vision-language models — extends to multi-modal long-context [arXiv:2504.16083].
- **Retrieval head** specialization: separate handling of attention heads identified as primarily retrieval-related.
- Native MoE-aware dispatch.

### S.11.1 Why MInference *fails* on video DiT

[Verified from SVG Table 1 and Figure 1]: MInference's offline calibration produces patterns that don't match video DiT's *dynamic* sparsity (different prompts → different head types). On HunyuanVideo: PSNR = 23.16, vs SVG = 29.55, vs Radial = 27.3. MInference's video performance is comparable to its CogVideoX result (PSNR 22.49 vs SVG 28.17) — consistently 4–6 PSNR points worse.

The lesson: **video DiT requires online (per-call) sparsity classification**. Offline calibration is sufficient for LLMs because their head specialization is essentially determined at training time and prompt-invariant. Video DiT heads' role *shifts with prompt, diffusion step, and layer* (echoing VMoBA's analysis), so offline calibration misses too much.

### S.11.2 What MInference *does* contribute to video DiT

The Triton block-sparse FA kernels in `microsoft/MInference` underlie multiple video DiT sparse methods:
- SVG2 borrows the block-sparse FA infrastructure with custom modifications.
- XAttention extends the block-sparse pattern.
- The general block-sparse "vertical-slash detection" is a building block for AdaSpa.

So MInference is *foundationally* important even though its specific patterns don't fit video DiT.

## S.12 XAttention — antidiagonal block selection

**Source:** [arXiv:2503.16428](https://arxiv.org/abs/2503.16428), Ruyi Xu et al. (Han Lab MIT/UCM/Tsinghua), ICML 2025. Code at [github.com/mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention).

XAttention's contribution: **the sum of antidiagonal entries** of each block in the attention matrix is a powerful, near-free proxy for block importance. Crucially, antidiagonal scoring **intersects every possible vertical and slash pattern** within a block (Figure 2 of the paper), so it automatically detects MInference's vertical-slash patterns *without* explicit search — eliminating the ~20-30% search overhead of MInference's vertical-slash detection.

### S.12.1 The antidiagonal trick

Within an `S × S` sub-block of the attention matrix, the antidiagonal touches each row and each column exactly once. Sum of antidiagonal:
$$
\mathrm{score}_{B} = \sum_{i=0}^{S-1} A_{i, S-1-i}
$$
This costs `S` scalar additions per block (vs `S²` for full block compute), and yet captures the block's "energy" because every row contributes once. Stride `S = 8` is the recommended default. From the paper (Section 2.1): the antidiagonal hits **every vertical line, every slash line, and every diagonal pattern** within a block — exhaustive coverage of MInference's pattern library, *for free*.

### S.12.2 Threshold block selection

After scoring all blocks via antidiagonal sums, softmax-normalize, then select blocks whose cumulative softmax mass ≥ `τ` (default 0.9). Per-head dynamic-programming threshold prediction (Section 2.3) further optimizes `τ` per head.

### S.12.3 Reported numbers [VERIFIED Table 4]

**Video generation on HunyuanVideo VBench (5-step warmup):**
| τ | PSNR↑ | SSIM↑ | LPIPS↓ | Density↓ |
|---|---|---|---|---|
| 0.90 | 21.5 | 0.767 | 0.215 | 34.4% |
| 0.95 | 23.5 | 0.822 | 0.155 | 45.5% |

**Attention speedup (Llama 3.1 8B Instruct on RULER):**
| Context | Stride 4 | Stride 8 | Stride 16 |
|---|---|---|---|
| 64K | 9.43% density | 10.98% | 11.32% |
| 256K | **6.89% density**, **13.5× speedup** | 7.32% | – |

**Pattern-selection speedup [Figure 5]:** XAttention is **24.9× faster** than MInference's vertical-slash search and **5.9× faster** than FlexPrefill's selection.

### S.12.4 Composability for video

XAttention is a drop-in plugin for any block-sparse FA backend. The mit-han-lab repo ships an integration with Hunyuan that uses 5-step full-attention warmup. SGLang/FastVideo do not yet have first-class XAttention integration, but the antidiagonal-scoring algorithm could be plugged into VSA's coarse stage as a cheaper alternative to mean-pooling when high-throughput inference is needed.

## S.13 ΔAttention — generic delta-correction patch

**Source:** [arXiv:2505.11254](https://arxiv.org/abs/2505.11254), Jeffrey Willette et al. (KAIST / DeepAuto.ai).

ΔAttention is **not** a new sparse method — it's a *quality patch* that can be stacked on any sparse method (StreamingLLM, HiP, MInference, etc.) to recover the L1 distributional shift that sparse attention introduces.

### S.13.1 The shift problem

Sparse attention computes `A* V` where `A*` is a sparse approximation of `A`. The difference `A V − A* V` is the lost mass. Even when the sparse method correctly identifies the largest entries of `A`, the *softmax normalization constant* of `A*` differs from `A` (sparse softmax only normalizes over admitted entries). This shifts the output distribution and degrades downstream task performance — RULER MultiKey-3 with sparse Streaming LLM 2K window: **0% accuracy** vs full attention's **62%** (despite the 2K window being more than enough to encode unique UUIDs).

### S.13.2 The Δ correction

For every `γ`-th query row (default `γ = 64`), compute the *full* dense attention of just that one row:
$$
\widetilde{\mathbf{A}}_{\lfloor i/\gamma \rfloor} \mathbf{V} = \sigma(\widetilde{\mathbf{Q}}_{\lfloor i/\gamma \rfloor} \mathbf{K}^\top) \mathbf{V} \qquad \widetilde{\mathbf{Q}}_{\lfloor i/\gamma \rfloor} = \mathbf{Q}_i \text{ where } i \bmod \gamma = 0
$$
Then approximate the missing attention contribution as the difference, and *broadcast* to the surrounding γ rows:
$$
(\widehat{\mathbf{A}} \mathbf{V})_i = (\mathbf{A}^* \mathbf{V})_i + \underbrace{\left[\widetilde{\mathbf{A}} \mathbf{V}_{\lfloor i/\gamma \rfloor} - (\mathbf{A}^* \mathbf{V})_{\lfloor i/\gamma \rfloor \cdot \gamma}\right]}_{\Delta\text{ correction term}}
$$
Cost: `1/γ ≈ 1.5%` extra compute (γ=64 → ≈98.5% maintained sparsity). Can be implemented with a minimally-modified flash-attention kernel that emits only every γ-th row densely.

### S.13.3 Reported numbers (LLM)

Llama 3.1 8B Instruct, RULER 131K with γ=64:
- Streaming LLM 2K window: 27.45% accuracy
- **Streaming LLM + Δ (2K window): 64.40% accuracy** (+37 percentage points)
- HiP Attention: 68.74% → HiP + Δ: **73.71%**
- MInference: 65.73% → MInference + Δ: **73.31%**

**ΔAttention recovers ~88% of full attention quality on RULER 131K with ~98.5% sparsity (32× over FA2)** [VERIFIED abstract claim].

### S.13.4 Relevance to video DiT

ΔAttention is currently LLM-targeted. The principle directly applies to video DiT — any sparse method (VSA, SVG2, BSA) introduces a softmax-renormalization shift that could be partially recovered by a 1.5% extra compute. **No published video-DiT Δ patch as of May 2026**, but it's a natural pairing for VSA distillation: the 0.5% extra Δ compute could be folded into the sparse-distill loss to further close the dense-vs-sparse quality gap. Worth flagging as an open opportunity in the FastVideo backlog.

## S.14 DiTFastAttn — three orthogonal DiT-specific cuts

**Source:** [arXiv:2406.08552](https://arxiv.org/abs/2406.08552), Zhihang Yuan et al. (Tsinghua / Infinigence AI / SJTU), NeurIPS 2024. Project at [nics-effalg.com/DiTFastAttn](http://nics-effalg.com/DiTFastAttn).

DiTFastAttn was the first systematic post-training compression for DiT attention. Three orthogonal axes of redundancy:

1. **Spatial → Window Attention with Residual Sharing (WA-RS).** Many DiT heads attend mostly within a local window. Replacing full attention with windowed attention loses some long-range dependencies. The fix: at each step `r`, cache the *residual* `R_r = O_r − W_r` (full minus window), and reuse `R_r` for several subsequent steps' window attention `O_k = W_k + R_r`. Reusing the residual (not the full output) preserves long-range dependencies that don't change much across steps.
2. **Temporal → Attention Sharing across Timesteps (AST).** Attention outputs at adjacent steps are very similar for some layers. Cache `O_k`, broadcast to next several steps. Cosine similarity between adjacent step outputs is >0.95 for many layers (Fig 4a).
3. **Conditional → Attention Sharing across CFG (ASC).** With classifier-free guidance, conditional and unconditional inferences exhibit attention-output similarity SSIM ≥ 0.95 for many layers/steps. Reuse the conditional attention output for the unconditional pass.

A greedy compression-plan search (Algorithm 1) selects per-(layer, step) which technique to apply, with a tunable threshold `δ` controlling quality–efficiency tradeoff (D1 = δ=0.025 most conservative through D6 = δ=0.15).

### S.14.1 Reported numbers (image)

- DiT-XL-2 512×512 D6: Attn FLOPs **34%** of original, IS 352.20 (vs 408.16 baseline) — quality drops noticeably at this aggressive setting.
- PixArt-Sigma 1024×1024 D6: Attn FLOPs **37%**, FID 52.73 (vs baseline 55.65, even better).
- **PixArt-Sigma 2048×2048 D6: Attn FLOPs 24%, FID 49.34 (vs 51.89), CLIP 31.28 (vs 31.47)** — the higher the resolution, the larger the speedup with minimal quality loss.

### S.14.2 Reported numbers (video, OpenSora V1.1)

D5 setting reduces ~40.52% computation with mostly-preserved quality at 240p × 16f.

### S.14.3 Why DiTFastAttn is mostly a baseline today

- **Window pattern is image-only**: 3D windowing for video would need STA-style mask geometry but DiTFastAttn uses 2D blocks. SVG / Radial / VSA outperform DiTFastAttn on video benchmarks (cross-ref §S.5 SVG Table 1 and §S.10 VMoBA Table 1).
- **Cache-based AST and ASC are subsumed** by TeaCache / FBCache / PAB / FasterCache, which are smarter about *which* steps are similar (TeaCache uses the timestep-embedding-modulated input difference as a proxy; DiTFastAttn just thresholds output similarity post-hoc).
- The three-axis taxonomy (spatial / temporal / CFG) remains conceptually useful, but each axis has been individually superseded.

## S.15 DraftAttention — speculative-then-verify sparse

**Source:** [arXiv:2505.14708](https://arxiv.org/abs/2505.14708), Xuan Shen et al. (Northeastern / CUHK / Duke / Adobe / NUIST / UCM). Code at [github.com/shawnricecake/draft-attention](https://github.com/shawnricecake/draft-attention).

DraftAttention applies the speculative-decoding pattern to attention itself. Two-pass:

1. **Draft pass:** downsample Q and K spatially via average pooling (`8×16` kernel, stride = kernel size → 128× reduction in token count). Compute *full attention* on the downsampled tensors:
   $$
   \widetilde{Q}_i = \frac{1}{|R_i|} \sum_{j \in R_i} Q_j, \qquad \widetilde{K}_i = \frac{1}{|R_i|} \sum_{j \in R_i} K_j
   $$
   $$
   A_{\mathrm{draft}} = \mathrm{softmax}(\widetilde{Q} \widetilde{K}^\top / \sqrt{d}) \in \mathbb{R}^{g \times g}
   $$
   At 128× reduction, this draft attention is essentially free.

2. **Verify pass:** select top-`r` blocks from `A_draft`, apply the resulting mask to *full-resolution* attention. To make the scattered-block layout hardware-efficient, **reorder tokens so that each region's tokens are contiguous** (Algorithm 1). Then the full-resolution sparse attention runs as a clean block-sparse FA.

### S.15.1 Theoretical analysis

Two error bounds (Section 3.2):

- **Theorem 3.3 (Draft Attention Error):** `‖S − S_draft‖_F ≤ δ · n` where `δ = max_{i,j,u∈R_i,v∈R_j} |S_uv − S̃_ij|`. For locally-smooth video, `δ` is small.
- **Theorem 3.5 (Sparsity Mask Error):** `‖S − S ⊙ M̂‖_F ≤ n(δ + t)√(1−r)` where `t = S̃_(⌈rg²⌉)` is the threshold-defining score. Concentrated draft attention → small `t` → small error.

### S.15.2 Reported numbers

- HunyuanVideo: **1.75× e2e speedup** at fixed quality.
- The 8×16 average-pooling kernel reduces tokens by 128× — this matches FA2/FA3 block sizes for hardware efficiency.

### S.15.3 Speculative pattern relevance

DraftAttention is the published instance of the "speculative-then-verify" sparse pattern in video DiT. Conceptually adjacent: SpeCa (speculative feature caching, [arXiv:2509.11628](https://arxiv.org/abs/2509.11628), 6.34× FLUX), SDVG (speculative decoding for autoregressive video diffusion, 1.59–2.09×). The draft-then-verify pattern is computationally complementary to caching (cache reuses past output; draft generates a coarse new estimate) — both can compose multiplicatively with sparse-attention-on-the-verify-pass.

## S.16 ToMeSD / VidToMe — token merging for diffusion

### S.16.1 ToMeSD ([arXiv:2303.17604](https://arxiv.org/abs/2303.17604))

Daniel Bolya & Judy Hoffman (Georgia Tech). Brings Token Merging (ToMe) to Stable Diffusion. The original ToMe operates on ViT classifiers; ToMeSD adds:

- **Unmerging operation:** ToMeSD merges tokens before each block (self-attn / cross-attn / MLP) and *unmerges* before the skip connection. For two merged tokens `x_1, x_2 → x* = (x_1 + x_2)/2`, unmerge sets `x_1' = x_2' = x*`. Loses information but tokens were similar so error is small.
- **Random 2×2 partitioning:** ToMe alternates src/dst tokens by stride; for SD, naive striding causes regular-grid artifacts. ToMeSD samples one dst token randomly per 2×2 region (with random seed *fixed across the CFG batch* — critical so conditional and unconditional samples allocate dst the same way).
- **Self-attention only:** ToMe applied to cross-attn or MLP hurts FID. Only self-attention benefits.
- Up to **60% token reduction, 2× speedup, 5.6× memory reduction** on Stable Diffusion 1.5 with FID maintained or improved.

### S.16.2 VidToMe ([arXiv:2312.10656](https://arxiv.org/abs/2312.10656), CVPR 2024)

Xirui Li et al. (SJTU / UC Merced). Video extension of ToMe targeting *zero-shot video editing*. Two-level token merging:

1. **Intra-chunk local:** within a chunk of `B` adjacent frames, merge similar tokens across frames (cosine similarity between Q tokens). Source/destination partitioning randomly samples one dst per 2×2 region.
2. **Inter-chunk global:** maintain a set of "global tokens" across chunks that participate in long-term content consistency. Merge new chunk tokens with these globals.

VidToMe is editing-focused, not pure generation acceleration. It's primarily relevant to the report as a *historical baseline* and as a reminder that token-merging at the attention-input level was tried before block-sparse attention superseded it for video DiT.

### S.16.3 Why ToMe lost to block-sparse attention for video DiT

ToMe's premise — that tokens are redundant — is real. But block-sparse attention exploits the same redundancy *at the attention pattern level* rather than the *input level*, and:
- Block-sparse keeps every token visible at every layer (no information loss).
- Block-sparse is friendly to Tensor Cores; ToMe requires per-block bipartite matching that's harder to fuse with FA.
- Block-sparse compositions with sparse-distill / TeaCache / Sage are well-understood; ToMe stacks awkwardly.

For dense image DiT (PixArt, SD3) ToMeSD is still useful — it's simple, training-free, and stacks with SageAttention. For video DiT in 2026, sparse attention dominates.

## S.17 Astraea (ICLR 2026) — token-wise budget search

**Source:** [ICLR 2026 poster #10008349](https://iclr.cc/virtual/2026/poster/10008349), Haosong Liu et al.

Astraea introduces a *search framework* for per-timestep token budgets. Core components:
1. **Lightweight token selection mechanism:** scores tokens for importance per layer.
2. **Memory-efficient GPU-parallel sparse attention strategy:** linear reduction in execution time with the budget.
3. **Evolutionary search** over timestep budgets: classic GA explores budget allocations across diffusion steps to maximize VBench score within a target latency budget.

**Reported numbers:** **2.4× single GPU**, **13.2× on 8 GPUs**, with **<0.5% VBench drop** vs the dense baseline. The 13.2× scaling on 8 GPUs is notably better than naive sparse + Ulysses (which db-SP measures at 6.09× for Wan2.1-T2V-14B) — Astraea's distributed scaling contribution is genuine.

## S.18 Composability matrix and parallelism gotchas (db-SP / DSA)

This section pulls together the cross-method composability story for B200 deployment in 2026. The two open problems flagged by the report are **sparse + ring** (no fused kernel) and **head-imbalance under sparse + Ulysses**. Both are partially addressed in late 2025 / early 2026 by **db-SP** and **DSA**.

### S.18.1 db-SP — Dual-Balanced Sequence Parallelism

**Source:** [arXiv:2511.23113](https://arxiv.org/abs/2511.23113), Siqi Chen et al. (Tsinghua), Yu Wang corresponding. Code at [github.com/thu-nics/db-SP](https://github.com/thu-nics/db-SP).

db-SP formalizes the **sparse imbalance ratio** `ρ_s`:
$$
\rho_s = \frac{\sum_{i=0}^{N-1} \max_{g \in \mathcal{G}} b_{i,g}(\mathbf{s}, \mathbf{p})}{\sum_{i,g} b_{i,g}(\mathbf{s}, \mathbf{p}) / |\mathcal{G}|}
$$
i.e., the ratio of max-loaded GPU's work to the average. Measured `ρ_s` on Wan2.1-T2V-14B and CogVideoX1.5-5B with PAROAttention and SpargeAttn under various SP strategies [VERIFIED Table 1]:

| Model | Sparse | SP | ρ_s |
|---|---|---|---|
| Wan2.1 | PAROAttention | Ulysses | 1.355 |
| Wan2.1 | PAROAttention | Ring | 1.255 |
| Wan2.1 | PAROAttention | USP (U4R2) | 1.253 |
| Wan2.1 | PAROAttention | USP (U2R4) | 1.246 |
| Wan2.1 | SpargeAttn | Ulysses | **1.428** |
| CogVideoX1.5 | PAROAttention | Ulysses | **1.447** |
| CogVideoX1.5 | PAROAttention | Ring | 1.269 |
| CogVideoX1.5 | SpargeAttn | Ulysses | **1.513** |

So even with USP at U2R4 there's **24.6%** wasted potential; with naive Ulysses + SpargeAttn on CogVideoX1.5 there's **51.3%** wasted. db-SP closes this gap with three components:

1. **Head-level greedy partitioning** (for Ulysses): sort heads by dense-block count descending, iteratively assign each to the GPU with currently-least cumulative blocks. Reuses the partitioning plan from the previous denoising step if `ρ_s < P_s` threshold (default 1.10) — partitioning needed only in 5 of 50 denoising steps for Wan2.1-T2V-14B. **`ρ_s ≤ 1.1` after head balancing.**
2. **Block-level greedy partitioning** (for Ring Attention): partitions Q chunks across GPUs and KV chunks across iterations, with a *biased greedy* algorithm using a reward factor `R_b` that artificially reduces the cost of placing Q/KV chunks on their initial GPU (mitigating inter-GPU exchange overhead). **`ρ_s ≤ 1.05` after block balancing.**
3. **Sparsity-aware parallel strategy selection** (for USP): a runtime model predicts the latency of each `(U_x R_y)` strategy:
$$
\mathcal{L}(\mathbf{s}, \mathbf{p}) = \mathcal{L}^{\text{all2all}}(x) + \left[ \max(\mathcal{L}^{\text{comp}}(\mathbf{s}), \mathcal{L}^{\text{p2p}}(y)) \cdot (y - 1) + \mathcal{L}^{\text{comp}}(\mathbf{s}) \right] \cdot \rho_s(\mathbf{s}, \mathbf{p})
$$
where the per-call sparse pattern `s` modulates `L^{comp}` and `ρ_s`. Pre-builds all `log₂|G| + 1` communication groups so switching is free at runtime. **Picks the optimal strategy per attention layer dynamically.**

**Reported numbers [VERIFIED Section 6.2]:**
- Average **end-to-end speedup: 1.25×** over USP on Wan2.1-T2V-14B and CogVideoX1.5-5B (8 A800 / 8 H800).
- Average **attention-specific speedup: 1.40×** over baseline SP methods.
- Partitioning overhead: **<5% of e2e latency** thanks to plan reuse and biased greedy.
- Compared to BurstEngine (a competing sparsity-aware Ring-only baseline): db-SP wins because BurstEngine uses tile size 32 which is 1.25× slower than tile size 64 on H800.

### S.18.2 DSA — Distributed Sparse Attention (ICLR 2026)

**Source:** [ICLR 2026 poster #10011818](https://iclr.cc/virtual/2026/poster/10011818), Shenggui Li, Runyu Lu, et al.

DSA is the closest published progress on the **fused-kernel** version of distributed sparse attention. Reported headline:
- **1.43× over existing distributed methods**.
- **10.79× over single-GPU inference** on 8 GPUs (vs naive Ulysses' 6.09× and Ring's 5.81×).

DSA "integrates sparse attention with distributed inference for diffusion-based video generation" with "carefully-designed parallelism strategies and scheduling." The poster doesn't disclose the inner-kernel approach (likely a CUTLASS-DSL or Helion-generated fused ring + sparse kernel), but the result is the **only published case in the field where (sparse + ring) actually composes faster than (sparse + Ulysses) on >4 GPUs**. This directly addresses the open problem flagged in §3 and §S.3 of the parent report.

### S.18.3 The 2026 composability matrix

| Sparse method | + Ulysses | + Ring | + db-SP | + DSA | + TeaCache | + BLADE / FastWan distill | + FA4 / FA-FP4 (B200) | + SageAttention 2/3 |
|---|---|---|---|---|---|---|---|---|
| **STA** | ✓ | ✓ if chunk≥window | ⚠ | ⚠ | ✓ | n/a (archived) | ✗ (sm90 only) | ✓ |
| **SSTA** (HY1.5) | ✓ | ✓ | ⚠ | ⚠ | ✓ | ✓ (distill phase) | ⚠ flex-block-attn TBD | ✓ (50-step + Sage = 1.66×) |
| **VSA** | ✓ DistributedAttention_VSA | ✗ no fused kernel | ✓ measured | ✓ in scope | ✓ stacks | ✓ FastWan / DMD2 | ✓ via Triton block-sparse | ⚠ via SageSLA pathway |
| **Radial** | ✓ trivially | ✓ if 3D-aware partition | ⚠ static so balanced already | ✓ | ✓ | ✓ LoRA + distill | ✓ FlexAttention BlockMask | ✓ |
| **SVG / SVG2** | ✓ PR #50 | ✗ k-means is per-rank | ⚠ | ⚠ | ✓ | ✗ training-free only | ✓ + FP8 (SVG: 2.33×) | ✓ |
| **SLA / SageSLA** | ✓ FastVideo class | ✗ | ⚠ | ⚠ | ✓ | ✓ 2000-step fine-tune | ✓ via SageAttn FP8 path | ✓ native (SageSLA) |
| **BSA** | ✓ | ✗ | ⚠ | ⚠ | ✓ | ✓ trains from scratch | ⚠ (custom Triton kernel) | ⚠ |
| **NSA** | ✓ | ✓ (LLM-tested) | n/a | n/a | n/a | n/a (LLM) | ✓ DeepSeek's own kernels | ⚠ |
| **SpargeAttn** | ✓ | ✗ | ✓ db-SP measured | ⚠ | ✓ | ✗ training-free only (Sparge-T trainable variant exists but worse than SLA) | ✓ via INT8/FP8 kernels | ✓ native |
| **VMoBA** | ✓ FastVideo | ✗ | ⚠ | ⚠ | ✓ | ✓ trains from scratch | ⚠ Triton kernel | ⚠ |
| **MInference** | ✓ | ✗ | n/a | n/a | n/a | ✗ training-free LLM | ✓ Triton block-sparse | ⚠ |
| **XAttention** | ✓ | ✗ | ⚠ | ⚠ | ✓ | ✗ training-free | ✓ via FlashInfer | ⚠ |
| **ΔAttention** | ✓ | n/a (output-space patch) | n/a | n/a | n/a | ✓ (could fold into distill loss) | ✓ standard FA | ⚠ |
| **DiTFastAttn** | ✓ | ✗ | n/a | n/a | overlaps with TeaCache | ✗ training-free | ✓ standard FA | ✓ |
| **DraftAttention** | ✓ (draft is per-rank) | ⚠ | ⚠ | ⚠ | ✓ | ✗ training-free | ✓ block-sparse FA | ✓ |
| **ToMeSD / VidToMe** | ✓ (operates on input) | ✓ | n/a | n/a | ✓ | ✗ | ✓ standard FA | ✓ |
| **Astraea** | ✓ | ⚠ | ⚠ | ✓ (built on multi-GPU sparse) | ✓ | ⚠ token-wise | ✓ | ⚠ |

Legend: ✓ = supported / known good, ✗ = not supported / known bad, ⚠ = unverified or partial.

### S.18.4 The "no fused ring + sparse" open problem (May 2026 status)

As of this report, **no published, open-source library has a single fused kernel that does (a) Ring Attention's overlapping P2P KV rotation and (b) dynamic block-sparse mask computation in the inner loop simultaneously.** The closest:

1. **Triton-distributed** has `sp_ag_attention_intra_node.py` but no sparse path.
2. **FA4 CuTe-DSL** has a sparse path but no ring.
3. **DSA (ICLR 2026)** is the most credible published candidate but the kernel isn't open yet.
4. **db-SP** mitigates the *imbalance* but doesn't fuse — it still runs separate sparse + SP kernels, just with better load balancing.

The path forward is one of:
- Helion-generate the inner ring loop and wrap it with a Python-level sparse mask + cross-rank LSE merge (FA-2 style backward).
- Extend FA4's CuTe-DSL kernel to expose `(O, lse)` dual output and compose with a Ring driver in Python.
- Wait for DSA's open-source release.

### S.18.5 Sparse + cache (TeaCache) composition

TeaCache / FBCache / TaylorSeer all decide *whether to call attention at all on a given step*; sparse methods reduce *the cost of each call*. They compose multiplicatively. ComfyUI community benchmarks report ~30% extra speedup from Sage + TeaCache on Wan 2.1 vs Sage alone.

The cross-rank gotcha: caches are per-rank decisions. With Ulysses/Ring, the "cache hit" criterion (TeaCache's `δ_t > rel_l1_thresh`) must be *broadcast from rank 0* to ensure all ranks skip the same step. Otherwise ranks desync and the pipeline deadlocks at the next collective. Verified in [vLLM-Omni Cache-DiT documentation](https://deepwiki.com/hsliuustc0106/vllm-omni-skills/5.2-caching-acceleration%3A-teacache-and-cache-dit).

### S.18.6 Sparse + step-distillation (BLADE / FastWan VSA)

The 2026 SOTA pattern is **trained-in sparse attention + step distillation** in the *same* training run:
- **FastWan 2.1-1.3B**: VSA 0.8 + DMD2 3-step → **50.9× e2e** [VERIFIED §S.3.7]; 16 FPS T2V on a single H100.
- **BLADE** (referenced in the parent report, building on similar principles): combines step-distillation, NVFP4 quantization, and trained-in sparsity.
- **SSTA in HunyuanVideo 1.5**: sparse training during the distillation phase rather than as a separate step.
- **Cosmos-Predict 2.5 / 2B distilled** (NVIDIA reference): 4-step DMD2 on TrigFlow parameterization.

The lesson: sparsity must be present *during distillation* for the student to recover the dense teacher's quality. Adding sparsity post-distillation degrades the student because the distillation loss already optimized for dense attention's specific output distribution.

### S.18.7 Sparse + B200 NVFP4 stack

The full B200 stack as of May 2026:
- **Linear layers**: NVFP4 GEMM via `tcgen05.mma` — ~2× over BF16 [VERIFIED parent §1].
- **Attention**: FA4 BF16 reaches 1613 TF/s (71% B200 peak); FA-FP4 reaches 1801 TF/s (1.39× over FA4 BF16). SageAttention 3 hits 1038 TOPS on RTX 5090 (~5× over the fastest FlashAttention on that GPU — note: FA3 is Hopper-only so the comparison is against FA2).
- **Sparse attention** layered on top: VSA 6× kernel (over FA3) at 87.5% sparsity; SLA 13.7× over FA2; SVG2 1.92× e2e on top of FA3-class baseline.

Multiplicatively: `FP4 GEMM (~2×) × FA-FP4 (~1.39×) × VSA (~1.5–6× attention-only) × distillation (~12.5×) × CFG-fold (2×) ≈ 50–200× overall`, matching the published TurboDiffusion 100–200× headline numbers.

### S.18.8 The unified picture

Sparse attention for video DiT in 2026 is no longer a single "method to swap in." It's a *training-time architectural choice* with deep system-level coupling:

1. **At training:** decide trainable-sparse (VSA / BSA / SLA) vs static-sparse (Radial / STA fine-tune) vs none. Trained-in sparse beats post-hoc by 10–20% sparsity at the same quality, and survives extreme step-distillation.
2. **At inference:** use the trained-in mask; layer caching (TeaCache) on top for additional 2× per-call savings; layer SP (Ulysses + db-SP) for 6–10× scaling on 8 GPUs; layer FA-FP4 + SageAttention 3 quantization on top for additional 1.4–2× per-kernel.
3. **At deployment:** SGLang `compatibility_matrix` is the canonical single-source-of-truth for which model × backend combinations actually work today. As of May 2026, only Wan 2.1 T2V/I2V 14B has a fully-checked row (TeaCache, STA, SageAttention, SVG2 all available; VSA n/a for the dense Wan 2.1 since VSA is trainable — FastWan-VSA is a separate distilled checkpoint). LTX-2 / LTX-2.3 still have all sparse columns ❌, awaiting integration.

**The unresolved frontier**: a single B200 kernel that fuses ring-shifted KV rotation, dynamic block-sparse mask, FP4 quantization, and trained-in distillation in one launch. As of May 2026, no such kernel exists publicly. DSA is the most credible upcoming candidate; FA4's CuTe-DSL is the most credible underlying substrate; FastVideo and SGLang are the most likely first integrators.
