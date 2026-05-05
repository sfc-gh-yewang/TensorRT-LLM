## §1. System Optimization: Sequence Parallelism, Tensor Parallelism, and Comm-Overlap

*A long-form technical reference for the parallelism layer of the Wan/LTX/HunyuanVideo B200 stack. Each subsection covers motivation → algorithm → implementation → quantitative results → limitations → composition. Subsection prefix `P.x` is scoped to §1.*

---

## P.1 Per-rank `M` ridge point and the SP↔TP crossover (framework)

Every sequence-parallel (SP) and tensor-parallel (TP) decision for a video diffusion transformer on Blackwell B200 collapses onto one number: **the per-rank GEMM `M` dimension after parallelism**. Almost all of the per-stack tradeoffs in this section can be derived from a single ridge-point inequality, so we put it first and reuse the framing throughout.

### P.1.1 The roofline of a Blackwell NVFP4 GEMM

A B200 die at full clocks delivers ~7.7 PFLOP/s NVFP4 dense (96.2% of the 8.0 PFLOP/s peak; arXiv:2512.02189 Table VII), ~3.85 PFLOP/s FP8, ~1.93 PFLOP/s BF16, and 8 TB/s HBM3e. The arithmetic intensity at which the GEMM transitions from memory-bandwidth-bound to compute-bound is `A_ridge = FLOPS / BW`. For NVFP4 with FP32 accumulation:

\[ A_{ridge}^{NVFP4} = \frac{7.7 \times 10^{12}}{8.0 \times 10^{12}} \approx 0.96 \text{ FLOP/byte} \]

Because every NVFP4 element is 0.5 bytes plus a 4-bit shared scale every 16 elements (≈0.625 bytes/element effective, including FP32 per-tensor scale amortized), and an `M × N × K` GEMM does `2·M·N·K` FLOPs against `(M·K + K·N + M·N)·b` bytes of weight + activation traffic where `b` is bytes/element, the standard derivation gives a per-rank "ridge point" on the `M` axis when `N` and `K` are large compared to `M`:

\[ M_{ridge} \approx \frac{1}{2} \cdot A_{ridge} \cdot \frac{N \cdot K}{N + K} \cdot \frac{1}{b_{\text{eff}}} \]

For typical video DiT shapes (`hidden = 4096–6144`, `intermediate = 16384–24576`), evaluating the formula gives **`M_ridge ≈ 3000–4000 tokens` in NVFP4** — *below* this `M`, the GEMM is HBM-bandwidth-starved (because the weight tile dominates traffic and isn't reused enough); *above* this `M`, the GEMM is compute-bound. This is the **per-rank `M` ridge** that anchors every other claim in this section.

Confirmation from the Mistral-7B microbench in arXiv:2512.02189: at M=4096 NVFP4 reaches ~2.5× the FP16 baseline throughput on B200, with HBM utilization dropping from 67.3% (FP16) to 47.6% (FP4) — that is, **FP4 is the first regime where Blackwell becomes compute-bound** at realistic decode-shaped LLM batches. Video DiT is even more favorable because long visual sequences make `M` natively large at SP=1.

### P.1.2 What SP and TP do to `M`

Pure SP=`P` *splits* the sequence: each rank's GEMM sees `M_{rank} = seq/P`. For Wan 2.2 14B at 720p × 81f × 40 steps, the visual token count is `seq ≈ 75k`. SP=8 → `M_{rank} ≈ 9.4k` (above ridge → compute-bound). At LTX-2 (5s × 1080p but distilled to 8 steps): `seq ≈ 12k` so SP=8 → `M_{rank} ≈ 1.5k` (well *below* ridge → bandwidth-bound, GEMMs run at 50–60% of peak NVFP4 throughput).

Pure TP=`P` does *not* shrink `M` — TP shards the contracting `K` and the column `N`, leaving `M = seq` unchanged. Each per-rank GEMM has the same `M` as a single-GPU baseline; instead it pays for AllReduce or AG+RS communication around each row-parallel step.

Pure DP doesn't shrink `M` either, but doesn't help with single-request latency at all.

This asymmetric scaling of `M` under SP versus TP is the **central system-side knob**. Whenever you read in this section that hybrid `SP=2 × TP=4` is the "production sweet spot", the underlying derivation is: at 8 GPUs and `seq ∈ [10k, 75k]`, hybrid keeps `M_{rank} = seq/2 ≥ 5k > M_{ridge}` (NVFP4 compute-bound on every per-rank GEMM) while only paying TP=4 row-parallel collectives per layer (smaller comm volume than TP=8, hideable behind the *larger* of the two GEMMs by Megatron option B).

### P.1.3 Comm-cost decomposition

For a transformer block with hidden `h`, sequence `N`, and parallelism degree `P`, the per-link communication volumes (rounded to leading `O(.)`) are:

| Method | Per-link bytes | Scales with P? | Hideable behind |
|---|---|---|---|
| Megatron option A AllReduce TP | `2·(P-1)/P · 4Nh` | constant | row-parallel GEMM |
| Megatron option B AG+RS TP (SP-TP) | `2·(P-1)/P · 4Nh` total, split into AG and RS | constant | each GEMM (asymmetric overlap) |
| Ulysses 4 a2a per attention | `4Nh/P` | shrinks with P | ❌ (vanilla, sync) |
| Async Ulysses 4 a2a per attention | `4Nh/P` | shrinks with P | ✓ (hides behind QKV proj) |
| Ring KV-rotation P stages | `4(N/P)·h` per stage | independent of P | per-stage attention compute |
| Hybrid USP (`u·r=P`) | `4Nh/u` (a2a) + `4(N/(u·r))·h·r` (ring) | shrinks with `u`, ring overlapped | both |

The constant-volume property of Ulysses (`O(N/P)` per link, `O(N/P)` per total) is the key Jacobs et al. result; Megatron-SP's `O(N)` per link makes it `P×` worse on bandwidth at scale. The Ring per-stage `O(N/P)` is the same as Ulysses but adds a per-stage *latency* term, so its goodput depends on whether per-stage compute hides per-stage comm. We work through each of these in P.2–P.4.

### P.1.4 The B200 NVLink5 budget

Per the DGX B200 datasheet: each B200 has **18 NVLink5 links delivering 1.8 TB/s aggregate bidirectional bandwidth per GPU** (≈900 GB/s unidirectional). Two NVSwitch ASICs per DGX B200 give a node-aggregate **14.4 TB/s bidirectional**. Effective per-pair bandwidth measured in real workloads (NCCL/NVSHMEM all-to-all under simultaneous compute load) is closer to **150–300 GB/s/pair** — the nominal 900 GB/s is achievable only when the link is the sole consumer. Both the SHI Labs DistGEMM blog (Hopper) and fal's "Ulysses Unbound" benchmarks (Blackwell) consistently observe the 150–300 range. We use 600 GB/s/pair as a generous upper bound for ring's per-stage budget, 300 GB/s as a realistic working number for fully-loaded all-to-all.

The implication is that Ring's `c_min = FLOPS/BW` for B200 with FA4 (~1.6 PF/s effective) and 600 GB/s working bandwidth lands at:

\[ c_{min}^{B200} = \frac{1.6 \times 10^{12}}{6 \times 10^{11}} \approx 2.6k \text{ tokens} \]

so Ring needs `seq/P ≳ 16k` (six times the block size as Ring's own `s = 6c` rule shows; arXiv:2310.01889 Table 2) for the rotation to be free. At `seq=12k, P=8` (LTX-2), each rank holds 1.5k tokens — well below `c_min` — and Ring exposes most of its KV-rotation cost. This is the *exact* regime in which Ulysses-style A2A wins, and it's why production stacks (xDiT, SGLang, FastVideo) ship Ulysses-first paths.

### P.1.5 Heuristic decision rule (the practical takeaway)

Given a video DiT inference workload `(seq, hidden, h_c, P)` on a B200 node with NVFP4 weights:

1. **Compute the SP=`P` per-rank `M_{SP}` = `seq/P`.** If `M_{SP} ≳ 4k`, pure SP is fine and minimizes comm. Examples: HunyuanVideo 720p 129f at SP=8 (`seq ≈ 100k`, `M_{SP} ≈ 12.5k` → NVFP4 happy).
2. **If `M_{SP} < M_{ridge} ≈ 3–4k`, switch to hybrid `SP=s × TP=t`** where `s = ⌈seq/M_{ridge}⌉`. For Wan 2.2 720p 81f (`seq ≈ 75k`) the answer is `s ∈ {2, 4}`; for LTX-2 1080p 5s 8-step (`seq ≈ 12k`) the answer is `s = 2, t = 4` (the SGLang/xDiT/Wan-official sweet spot).
3. **If `seq/P ≥ 16k` and `P ≤ heads`**, you can mix in Ring on the inner dimension to push past the head cap. The xDiT recipe `(ulysses_degree=2, ring_degree=2, tp_size=2)` for 8-GPU Wan 2.2 follows this rule.
4. **For the smallest-payload regime (`P=8`, `M < 1.5k`), NVFP4 isn't viable and FP8 + async-Ulysses on Symm-Mem dominates**. Below `P=8` and seq < 8k, SymmMem fused-AG-matmul is best at `P ≤ 4`; async-Ulysses is best at `P=8` (fal "Ulysses Unbound" measurement, P.6).

The remaining subsections drill into each component of this taxonomy.

---

## P.2 DeepSpeed-Ulysses (head-sharded SP via 4 all-to-all)

### P.2.1 Motivation

Pre-2023 sequence parallelism methods (ColAI-SP ring self-attention, Megatron-LM SP) shared a fundamental problem articulated in the Ulysses paper (Jacobs et al., arXiv:2309.14509, Table 1): they had per-link comm volume `O(M)` that **does not shrink with parallelism degree `P`**, plus they were either intrusive (Megatron SP requires tight integration with TP), specific to a particular attention kernel (ColAI-SP requires its own ring self-attention rather than supporting FlashAttention or sparse variants), or both. None of them could scale beyond `~16` GPUs without exposed comm dominating runtime. For training context lengths in the 100k–1M range — increasingly necessary for long-document chat, long-video understanding, and AI-for-science applications like genome modeling — a fundamentally different SP design was needed.

### P.2.2 Core algorithm — head-sharded all-to-all

Ulysses partitions input sequences across `P` devices along the sequence dimension (each rank starts with `(B, N/P, h)` activations). Right before attention, the algorithm performs **three all-to-all collectives** — one each on Q, K, V — that transpose the layout from *(seq=N/P, heads=h_c)* on each rank to *(seq=N, heads=h_c/P)*. Each rank now holds the *full sequence* but only `h_c/P` of the heads, and per-head FlashAttention runs as if on a single GPU. After local attention, **a fourth all-to-all** transposes back to *(seq=N/P, heads=h_c)* so the FFN/layernorm/projection layers run in their natural sequence-sharded layout.

Pseudocode (matching the standard 4D-all-to-all formulation used in `feifeibear/long-context-attention` and the FastVideo `sequence_model_parallel_all_to_all_4D`):

```python
# Pre-attention: scatter heads, gather sequence
Q = all_to_all_4D(Q_local, scatter_dim=2, gather_dim=1, group=sp_group)  # (B, N, h_c/P, d)
K = all_to_all_4D(K_local, scatter_dim=2, gather_dim=1, group=sp_group)
V = all_to_all_4D(V_local, scatter_dim=2, gather_dim=1, group=sp_group)

# Local attention (FA2/FA3/FA4 — completely unmodified)
O = flash_attn_func(Q, K, V)

# Post-attention: scatter sequence, gather heads
O_local = all_to_all_4D(O, scatter_dim=1, gather_dim=2, group=sp_group)  # (B, N/P, h_c, d)
```

### P.2.3 The constant-volume property

The key result of Ulysses §3.2 is the per-link bandwidth analysis. On modern clusters (NVSwitch intra-node, fat-tree IB inter-node), all-to-all of aggregate size `M` across `P` ranks moves `M/P` per link. For a transformer with hidden `h` and sequence `N`, the four all-to-alls move:

\[ \text{Total per-link bytes} = \frac{3Nh + Nh}{P} = \frac{4Nh}{P} = O(N/P) \]

When you scale `N` and `P` proportionally (the canonical "longer sequence on more GPUs" regime), this is **constant**. Megatron-SP's two-AllGather + two-ReduceScatter alternative moves `4Nh` per link **independent of P**, so it is `P×` worse on bandwidth at scale. The paper reports **2.5× faster training at 4× longer sequences** vs Megatron-SP and **>10× total comm reduction** across configurations on 64×A100.

The "weak scaling" Table 3 from Jacobs et al. is the cleanest demonstration: doubling `N` (65k → 131k → 262k) while doubling GPU count (64 → 128 → 256) keeps per-rank TFLOPs roughly constant (161 → 157 → 147), and iteration time grows roughly linearly (9.7s → 17s → 33.5s) — i.e., constant compute throughput as you scale, exactly what `O(N/P)` predicts.

### P.2.4 The `SP ≤ heads` constraint

Because Ulysses ends up with `h_c/P` heads per rank, you must have `P ≤ h_c`. For modern video DiTs (24–48 heads typical: Wan 2.2 14B has 32 heads, HunyuanVideo 13B has 24, LTX-2 has 32), SP=8 on a single B200 node never bumps against this. But the cap *is* binding for KV-grouped models — the USP follow-up is explicit: *"Llama3-8B employs GQA with a KV head number of 8, which means that when using DS-Ulysses SP, the maximum SP degree is 8."* For MQA (KV heads=1), Ulysses simply doesn't function. Two workarounds exist: (a) `repeat_interleave` the K,V tensors to inflate the head count (free per-head computation but pays comm on the inflated tensors — VeOmni's `async_ulysses.py` does exactly this when `ctx.need_repeat_kv` is True; see P.6), and (b) compose with Ring on a second axis (USP, P.4).

### P.2.5 Attention-agnosticism (the structural advantage)

Once the all-to-alls land, each rank runs an entire local SDPA over the *full sequence* on its head shard. There is **no rolling, no LSE merging across ranks, no causal-mask edge case** — it is bit-for-bit identical to single-rank FlashAttention. This is why every modern stack (xDiT, SGLang Diffusion, FastVideo, vLLM-Omni, diffusers native CP, Wan official) implements Ulysses on top of `flash_attn_func`, `cuDNN attention`, FA3, or FA4 *without any kernel modification*. By contrast, Ring requires re-implementing the FA inner loop with explicit `(O, lse)` outputs that can be merged across ranks, which is why Ring's FA4 support trails the public FA4 release (open question #2 in the parent report).

This attention-agnosticism is also why Ulysses composes cleanly with sparse attention (SageAttention 1/2/3, VSA, STA, SVG2, Radial): you swap `flash_attn_func` for `sage_attn` with no other change.

### P.2.6 Variable-length / packed sequences

For varlen the standard 4D-A2A formulation `(B, seq, heads, head_dim)` only works when `seq` is divisible by `P`. The standard practice across xDiT, FastVideo (`compute_padding_for_sp` in `fastvideo/distributed/utils.py`), and Wan official is **pad-then-shard-then-unpad**:

```python
# From fastvideo/distributed/utils.py
def compute_padding_for_sp(seq_len: int, sp_world_size: int) -> tuple[int, int]:
    if seq_len % sp_world_size == 0:
        return seq_len, 0
    padding_amount = sp_world_size - (seq_len % sp_world_size)
    padded_seq_len = seq_len + padding_amount
    return padded_seq_len, padding_amount
```

The padding wastes ~1/P fraction of compute in the worst case. For varlen *across* a packed-batch of unequal-length samples, the better recipe is to pass `cu_seqlens` (cumulative sequence lengths) into the local FA call after the all-to-all — yunchang and FastVideo both support this through `flash_attn_varlen_func`. The varlen path also handles the cross-attention case where K, V are *replicated* (text encoder, image encoder); in that case the all-to-all on K, V is wasteful and the SGLang `--skip_sp` optimization (PR #19419) skips it entirely (see P.9).

### P.2.7 ZeRO-3 integration

Ulysses doesn't reduce parameter memory by itself — it only reduces *activation* memory by `1/P`. For training, the standard recipe is to combine Ulysses with ZeRO-3: ZeRO partitions parameters, gradients, and optimizer states across `data_parallel × sequence_parallel` ranks. Jacobs et al.'s Section 3.3 makes this explicit. For **inference** (the focus of the parent report), ZeRO-3 is replaced by FSDP for parameter-only sharding — this is what `Wan-Video/Wan2.2`'s official `--dit_fsdp` and `--t5_fsdp` flags do, on top of `--ulysses_size 8`.

### P.2.8 API surface

The minimal user-facing API in the official DeepSpeed implementation is:

```python
from deepspeed.sequence.layer import DistributedAttention
dist_attn = DistributedAttention(attn, get_sequence_parallel_group())
```

The xDiT/yunchang re-implementation exposes the same ergonomic minimum and is the basis for SGLang/FastVideo/vLLM-Omni/diffusers-native-CP. The diffusers PR #11941 wraps it with a `ParallelConfig(ulysses_degree=2)` API (see P.9).

### P.2.9 Limitations recap

- `P ≤ h_c` cap (binding for GQA/MQA models, non-issue for video DiT with 24–48 heads).
- 4 a2a per attention block — large constant-volume term that becomes dominant when per-rank `M` shrinks below the NVFP4 ridge (~3-4k tokens).
- Pre-attention path is serialized (Q-proj → Q-a2a → K-proj → K-a2a → V-proj → V-a2a → SDPA) — fixed by Async Ulysses (P.6).
- No native Blackwell-specific transport optimization out-of-the-box; the SymmMem path (P.8) is what makes Ulysses keep up with FA4.

---

## P.3 Ring Attention + load-balanced variants

### P.3.1 Motivation

Ulysses's `P ≤ h_c` constraint and the structural fact that Ulysses moves `O(Nh)` *per attention block* (not amortized) limited it to small `P`. Ring Attention (Liu/Zaharia/Abbeel, arXiv:2310.01889, ICLR'24) targeted a different design point: **memory scaling proportional to device count**, with no head-count cap. The Ring claim is that you can run *device-count-times-longer* sequences on the same hardware (linear scaling of max sequence in `P`), enabling million-token contexts.

### P.3.2 Core algorithm — KV rotation + online softmax merge

Each of `P` ranks holds a contiguous shard of `Q, K, V` (block size `c = N/P`). The algorithm runs `P` outer iterations. At iteration `t`, every rank computes attention between its local Q block and its current `K, V` block, *while simultaneously* sending that current `K, V` block to the next rank in the ring and receiving the previous rank's `K, V` block. After `P` iterations every Q block has attended to every `K, V` block in the global sequence. Partial outputs are merged via the standard FlashAttention online-softmax recurrence, which only requires per-block running max `m_j` and sum-of-exp `ℓ_j`.

Key idea: **the inner-loop key-value blocks have permutation invariance** — attention can be computed in any order, as long as the running statistics (max, sumexp) are merged correctly. This lets us pipeline communication of one block with computation of another.

```python
# Pseudocode for one transformer layer (per rank i)
Q_local = X_local @ W_Q
K_local, V_local = X_local @ W_K, X_local @ W_V
m_i, l_i, O_i = -inf, 0, 0
KV_buf = (K_local, V_local)
for t in range(P):
    # Concurrent: kick off send/recv of KV_buf to ring next/prev
    next_buf = isend_irecv(KV_buf, dst=(i+1)%P, src=(i-1)%P)
    # Compute attention against the KV_buf we currently hold
    O_t, m_t, l_t = flash_attn_inner(Q_local, KV_buf[0], KV_buf[1])
    # Online merge with previous (m_i, l_i, O_i)
    m_new = max(m_i, m_t)
    l_new = exp(m_i - m_new)*l_i + exp(m_t - m_new)*l_t
    O_i = exp(m_i - m_new)*O_i + exp(m_t - m_new)*O_t
    m_i, l_i = m_new, l_new
    KV_buf = wait(next_buf)
O_final = O_i / l_i
```

### P.3.3 Per-stage compute hides per-stage comm

The Ring paper's Section 3 derives the asymmetry exactly. Per-stage attention FLOPs are `4·d·c² = 4·d·(N/P)²` (two matmuls of shape `c × d × c`); per-stage comm is `4·c·d = 4·(N/P)·d` bytes. The ratio is **`c = N/P`**, so as long as `c ≥ FLOPS/BW`, comm is fully hidden. The paper's Table 2 makes the threshold concrete:

| Hardware | FLOPS (TF) | BW (GB/s) | Min block `c` (×1k) | Min seq `N=6c` (×1k) |
|---|---|---|---|---|
| A100 NVLink | 312 | 300 | 1.0 | 6.2 |
| A100 InfiniBand | 312 | 12.5 | 24.5 | 149.5 |
| TPU v3 | 123 | 112 | 1.1 | 6.6 |
| TPU v4 | 275 | 268 | 1.0 | 6.2 |
| TPU v5e | 196 | 186 | 1.1 | 6.3 |
| **B200 (FA4 BF16, est.)** | **1613** | **~600** | **~2.7** | **~16.2** |
| **B200 (FA-FP4, est.)** | **1801** | **~600** | **~3.0** | **~18.0** |

The B200 row is *not* in the paper but follows from the same formula; the punchline is **Ring on B200 needs `seq/P ≳ 16–18k` per rank to be free**. At `seq=12k, P=8` (LTX-2 territory), each rank holds 1.5k tokens — well below `c_min` — so per-stage attention shrinks *quadratically* (`c²`) while comm only shrinks linearly (`c`), and exposed comm reappears fast. This is the regime where Ulysses unconditionally wins.

For HunyuanVideo at 720p × 129f (`seq ≈ 100k`, `c = 12.5k` at SP=8), Ring is competitive with Ulysses; for `seq ≈ 200k` (HunyuanVideo extended frame counts) Ring wins on memory while matching Ulysses on time.

### P.3.4 Online softmax merge — the FA3/FA4 incompatibility

The merge is the same algebraic identity FA1/FA2 use locally, just applied across stages instead of within a kernel. After computing `(O_local_t, m_local_t, ℓ_local_t)` from the t-th K,V block, the rank updates:

\[ m \leftarrow \max(m_{\text{old}}, m_t),\quad \ell \leftarrow e^{m_{\text{old}}-m} \ell_{\text{old}} + e^{m_t - m} \ell_t,\quad O \leftarrow e^{m_{\text{old}}-m} O_{\text{old}} + e^{m_t - m} O_t \]

Final output is `O / ℓ`. **Critically, FA3/FA4 do not natively expose this incremental update interface** — they return only `(O, lse)` from the public C++/CuTe-DSL API, with no path to the internal block-level statistics. `zhuzilin/ring-flash-attention` therefore re-implements the FA inner loop in Triton with explicit `(o, lse)` outputs that can be merged across ranks. This re-implementation lags behind the upstream FA4 release on Blackwell-specific optimizations (TMEM-resident accumulators, 2-CTA UMMA for the backward, software-emulated `2^x` on FMA units).

The right end-state — open question #2 in the parent report — is either (a) Helion-generate the inner loop and compose with a Python-level cross-rank merge, or (b) extend FA4's CuTe-DSL kernel to expose `(O, lse)` dual-output mode. Neither has shipped as of May 2026.

### P.3.5 Variants in `zhuzilin/ring-flash-attention`

The repo exposes four core variants:

- **`ring_flash_attn_func`** — canonical bidirectional Ring; what video DiT uses.
- **`zigzag_ring_flash_attn_func`** — load-balanced for *causal* attention (token reorder so each rank's chunk straddles the diagonal).
- **`stripe_flash_attn_func`** — Striped Attention variant (Brandon et al., see P.3.7).
- **`llama3_flash_attn_varlen_func`** — varlen path supporting packed sequences with `cu_seqlens`.

For bidirectional video DiTs only the first matters, and the xDiT [usp.md](https://github.com/xdit-project/xDiT/blob/main/docs/methods/usp.md) is explicit: *"Since DiT does not use Causal Attention, there is no need for load balancing operations on Ring-Attention."* xDiT and SGLang Diffusion both expose Ring via `feifeibear/long-context-attention` (yunchang) which wraps `zhuzilin/ring-flash-attention` underneath. SGLang's `--ring-degree` flag goes through this chain.

### P.3.6 Memory accounting

Ring's 6 blocks resident per rank (current Q, current K,V, send-buffer K,V, recv-buffer K,V, output O) gives `6bch` bytes max-activation per layer (paper Table 1). For the same total seq `N`, Ring achieves activation memory `6bch = 6·(N/P)·h` per rank, which is `6/P` of the unsharded baseline `6Nh`. This linear-in-`P` scaling is *the* differentiator from blockwise-only methods (`O(N)` activation memory regardless of `P`).

### P.3.7 Striped Attention (causal-mask balance)

**Reference:** Brandon et al., arXiv:2311.09431.

Vanilla Ring under causal masking has a brutal load imbalance: at iteration 0 every rank sees a full triangular workload, but at later iterations the K,V block being received is either entirely above or entirely below each Q block's diagonal — half the ranks have nothing to do while the other half do full unmasked work. The latency of Ring is determined by the *maximum* per-iteration latency across ranks, so under naive contiguous-block partitioning Ring effectively runs at the speed of unmasked attention despite needing only half the operations.

**Striped Attention** permutes tokens so that each rank holds a *uniformly distributed strided subset* of the sequence (every `P`-th token), instead of a contiguous chunk. After this permutation, each rank's Q-block and K-block always overlap on roughly half their entries by the causal mask, regardless of rotation step — so per-iteration causal masking saves close to the theoretical `~2×` factor.

**Headline numbers:** 1.45× over Ring on 8×A100 80GB at 256k seq for billion-scale models; 1.65× on 16×TPUv4 at 786k.

**Relevance to bidirectional video DiT:** ❌ — Wan, HunyuanVideo, LTX-2, CogVideoX, Mochi all use bidirectional 3D-attention on visual tokens, no causal mask. Striped's specific win does not apply. It *is* relevant if (a) a video model adopts causal-temporal attention (some autoregressive video models like VideoPoet, or self-forcing distilled Wan that becomes causal-in-time), or (b) you cross-attend a causal text decoder over visual tokens.

USP's Algorithm 2 implements an alternative load-balance strategy by reordering inputs into a zigzag pattern (chunk pairs `[k, 2P-k-1]`); Striped is the strided-permutation alternative. xDiT skips both for bidirectional DiT.

### P.3.8 Tree Attention (log-depth merge — briefly)

**Reference:** Shyam et al., arXiv:2408.04093.

Tree Attention re-derives attention as the gradient of a *scalar energy function* (the log-sum-exp of `QK^T`), then exploits that scalar reductions across `P` ranks can be done in `log₂(P)` steps via a tree allreduce. **Up to 8× faster cross-device decoding than Ring with 2× less peak memory** for very long-context cases. Less direct relevance to video DiT (DiT does not autoregressively decode), but the *formalism* (energy view + log-depth merge) is what future async-ring variants could adopt to get sub-linear-in-`P` scaling. The structural ideas overlap with `tree_all_reduce` patterns now in NCCL 2.28+.

### P.3.9 BurstEngine (THUNLP, the production Ring successor)

**Reference:** [arXiv:2509.19836](https://arxiv.org/abs/2509.19836); [thunlp/BurstEngine](https://github.com/thunlp/BurstEngine).

BurstEngine *contains* BurstAttention as its attention kernel. It adds: (i) sequence-level *selective* checkpointing for gradient recomputation, (ii) fused LM-head + cross-entropy loss, (iii) backward-direction communication optimization, (iv) fine-grained double-buffered comm-comp overlap. End-to-end, BurstEngine reports **1.2× over the strongest baseline at sequences ≥1M tokens** with **26.4% memory reduction** vs the most memory-efficient prior work; scales linearly to **4M tokens on 64×A800**. Supports causal, sliding-window, and block-sparse masks.

For video DiT inference at <100k tokens, BurstEngine's 1.2× advantage doesn't materialize — but for ultra-long video DiT context (e.g., extended HunyuanVideo or sub-minute Wan generation at 720p/1080p that pushes seq into the multi-million range), BurstEngine is the natural Ring successor.

---

## P.4 USP — Unified Sequence Parallelism (Ulysses × Ring 2-D mesh)

### P.4.1 Motivation

By late 2023 the SP landscape had calcified into "Ulysses or Ring", with the ICLR open-review rebuttal of Ring framing them as alternatives. USP (Fang & Zhao, arXiv:2405.07719) showed they are not — they decompose along orthogonal axes of a 2-D mesh, and combining them lifts both the `P ≤ heads` cap of Ulysses and the per-stage compute-bandwidth ratio constraint of Ring.

### P.4.2 The 2-D mesh

USP organizes the SP world (`sp_size = N`) into a 2-D mesh `ulysses_degree × ring_degree = sp_size`. The two axes are **orthogonal process groups**: an `ulysses_pg` containing all ranks that share a ring shard but differ in head-shard, and a `ring_pg` containing all ranks that share a head-shard but differ in seq position. The forward pass becomes:

```python
def usp_attn(ulysses_pg, ring_pg, Q, K, V, scatter_idx=2, gather_idx=1):
    # 1. Ulysses 4D-A2A on Q, K, V across ulysses_pg
    Q = AllToAll4D(Q, scatter_idx, gather_idx, group=ulysses_pg)
    K = AllToAll4D(K, scatter_idx, gather_idx, group=ulysses_pg)
    V = AllToAll4D(V, scatter_idx, gather_idx, group=ulysses_pg)
    # 2. Ring attention across ring_pg (load-balanced if causal)
    O = LoadBalanceRingAttention(Q, K, V, group=ring_pg)
    # 3. Reverse Ulysses A2A on O across ulysses_pg
    O = AllToAll4D(O, gather_idx, scatter_idx, group=ulysses_pg)
    return O
```

For an 8-rank SP world organized as `2×4` (`u=2, r=4`): each "row" of 2 ranks shares ring rotation responsibility but holds different head-shards (Ulysses A2A across them); each "column" of 4 ranks shares the head-shard but rotates K,V (Ring P2P across them). The result is full attention with 4 a2a (small `P=2`) plus 4 ring stages (each carrying half the global sequence's K,V).

### P.4.3 Why combining lifts the head-cap

With `ulysses_degree = u` and `ring_degree = r`, the Ulysses head constraint is `u ≤ heads`, **not** `u·r ≤ heads`. So Llama-3 8B with 8 KV heads runs at SP=16 via `u=8, r=2` — Ulysses over the head dim, Ring over the sequence dim. Same idea for video DiTs: even if you find a 4-head model (none of Wan, Hunyuan, LTX use that), USP scales up to SP=16 with `u=4, r=4`.

### P.4.4 Heterogeneous topology

USP's *second* benefit (less commonly cited but very real) is that the mixed pattern is *more friendly to heterogeneous networks*. Inside a node (high-bandwidth NVLink) you run the All-to-All-heavy Ulysses dimension; across nodes (lower-bandwidth IB) you run the P2P-heavy Ring dimension. With `u = 8` per node and `r = 4` across nodes, each node does its A2A locally (fast NVLink) and the cross-node Ring just moves K,V blocks at `c·d` per stage which is `(N/(u·r))·d` per rank.

For a single B200 node this matters less (NVLink5 is uniformly fast at 1.8 TB/s), but the asymmetry is exploited heavily for multi-node deployments. The USP paper reports **47% MFU on 2×8 A800 nodes for Llama-3-8B at 208k sequence** — the only meaningful published number. Note that "47% MFU" sets a low bar relative to single-node FA3 numbers (75% on H100); the cross-node IB bottleneck is real.

### P.4.5 Communication and memory accounting

USP's communication volume per layer per rank is at most `4Nh/u` (Ulysses on the `u` axis) plus `4(N/(u·r))·h·r = 4Nh/u` (Ring on the `r` axis — total per layer is `4Nh/u·(P/u)` if we sum across stages). Both terms shrink with `u`, the head-axis. As `u` grows toward `h_c`, you converge to plain Ulysses; as `u → 1` you converge to plain Ring. The "sweet spot" is `u = ⌊√P⌋` or `u = h_c, r = P/h_c`, depending on whether the binding constraint is bandwidth or head count.

Activation memory is `A/P` regardless of split (USP Table 2): each rank holds `1/P` of the activation. Parameter memory is unchanged (full replication unless you stack ZeRO/FSDP on top).

### P.4.6 Load-balanced Ring inside USP

USP's Algorithm 2 (`LocalBalanceLocalSeq`) handles causal load balance by reordering tokens into pairs `(chunk_k, chunk_{2r-k-1})` before sharding — a zigzag pattern. This adds negligible overhead (one CPU-side reorder of the integer token indices). For bidirectional video DiT, the xDiT path *skips* the zigzag re-ordering entirely (xDiT [usp.md](https://github.com/xdit-project/xDiT/blob/main/docs/methods/usp.md)).

### P.4.7 Composition with TP (the 3-D grid)

USP composes orthogonally with TP into a 3-D grid `tp × ulysses × ring`. Wan official's `--ulysses-degree`, `--ring-degree`, `--sp-degree`, `--tp-size` flags expose exactly this. For 8 GPUs the practical recipes are:

| Recipe | tp | u | r | Total | Reason |
|---|---|---|---|---|---|
| Pure Ulysses | 1 | 8 | 1 | 8 | Best for `seq/8 > c_min`, `8 ≤ h_c` |
| Pure Ring | 1 | 1 | 8 | 8 | Best for `h_c < 8`; rare in video DiT |
| USP `u4r2` | 1 | 4 | 2 | 8 | Lifts head-cap moderately, balances comm |
| USP `u2r2 × tp2` | 2 | 2 | 2 | 8 | Hybrid, used in SGLang Wan 2.2 727 example |
| USP `u1r2 × tp4` | 4 | 1 | 2 | 8 | Hybrid, FastVideo/Wan 2.2 sweet spot |
| **`SP=2 × TP=4`** | 4 | 2 | 1 | 8 | **Production sweet spot** for `seq < 25k` |
| Pure TP | 8 | 1 | 1 | 8 | Step-Video-T2V on H20 (low bandwidth) |

### P.4.8 USP as the de-facto SP backend

By May 2026, USP is the *standard* video-DiT SP backend. Stack-by-stack adoption:

- **xDiT** uses `feifeibear/long-context-attention` directly with bidirectional ring (no zigzag).
- **SGLang Diffusion** uses USP via `feifeibear/long-context-attention`. The `(--ulysses-degree, --ring-degree, --tp-size)` knobs map 1:1 to the USP mesh (PR #16532 fixed the rank-group construction so TP×SP gives strided rather than contiguous SP groups, January 2026).
- **diffusers native CP** ([PR #11941](https://github.com/huggingface/diffusers/pull/11941)) exposes `ParallelConfig(ulysses_degree=2, ring_degree=2)` over USP; supports Flux and Wan T2V as of merge.
- **vLLM-Omni** uses USP under both the standard SP path and the experimental UAA mode.
- **Wan-Video/Wan2.2 official** uses Ulysses-only by default (`--ulysses_size 8`) but supports the USP×TP mix via the same flags.
- **FastVideo** is the exception — it implements only the Ulysses path (4D-A2A NCCL all-to-all), no Ring, no USP.

### P.4.9 Limitations

- USP doesn't *help* below the head cap — if `P ≤ h_c` and `seq/P ≥ c_min`, plain Ulysses wins by avoiding the Ring overhead.
- The "47% MFU" headline is for training, not inference; published video-DiT inference USP numbers are scarcer (xDiT [stepvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/stepvideo.md), [cogvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/cogvideo.md) — see P.12).
- USP doesn't fix the FA4 `(O, lse)` exposure problem; the Ring axis still re-implements FA in Triton.

### P.4.10 MM-SP / LongVILA — multi-modal extension of USP

**Reference:** Chen et al., [arXiv:2408.10188](https://arxiv.org/abs/2408.10188) (NVIDIA, NeurIPS 2024). Code in [github.com/NVlabs/VILA/tree/main/longvila](https://github.com/NVlabs/VILA/tree/main/longvila).

**Motivation.** Long-context Vision-Language Models (VLMs) have two distinctly-shaped activation streams: **text** (short, replicated across SP ranks) and **vision** (long, shardable). A single-sequence SP backend (Ulysses, Ring, USP) treats the placeholder image token the same as text tokens, leading to imbalanced GPU loads where some ranks process vision-heavy regions and others process text-only regions. MM-SP fixes this with a multimodal-aware two-stage sharding strategy.

**Modality heterogeneity** — the placeholder problem. In text-only LLMs, sequences are processed by a single tokenizer into tokens, allowing straightforward distribution across GPUs. VLMs incorporate an encoder architecture where non-text data is initially represented by a placeholder token (e.g., `<img>`) and subsequently encoded into multiple real tokens during training (a single video frame typically requires ~256 tokens). Treating placeholder tokens the same as text tokens leads to GPU workload imbalance.

**Networking heterogeneity** — the Ring/Ulysses tradeoff. In multi-node setups, inter-node and intra-node bandwidth differ significantly (DGX H100: 900 GB/s NVLink intra-node, 50 GB/s InfiniBand inter-node — 18× asymmetry). Ring-style SP (Liu et al., Li et al.) ignores this and uses P2P everywhere, which is wasteful intra-node. Ulysses uses A2A everywhere, which is wasteful inter-node where bandwidth is scarce.

**MM-SP's two-stage sharding workflow.**
1. **Stage 1 — image encoding load balance.** Distribute video frames evenly across SP-group devices. Each rank processes its assigned frames through the vision encoder. This achieves load balance during the image-encoding stage (avoiding the worst-case where one rank gets all the frames).
2. **Stage 2 — token-level sharding for LLM backbone.** Aggregate global vision and text inputs. Pad sequences with dummy tokens to ensure even division by the ring-based SP degree (label inputs are modified to ignore padded tokens during loss). Apply a *balanced sharding strategy* that distributes context to each rank from both ends, ensuring equal computation across ranks. Then process via 2D-Attention (Fang & Zhao USP-style; Gu et al. LoongTrain-style).

**The 2D-Attention mechanism.** MM-SP uses a 2D communication mesh: intra-node All-to-All (A2A) for the head dimension (using high-bandwidth NVLink), inter-node Point-to-Point (P2P) for the sequence dimension (over IB). This is structurally identical to USP but specifically tuned to the bi-modal asymmetry. For an 8-degree SP across 2 nodes (4×2 mesh): the A2A process group of size 4 distributes Q/K/V according to the head dim within each node; the P2P process group of size 2 transfers partitioned KV chunks between nodes.

**Performance overhead of communication overlap is a real cost on H100.** MM-SP measures attention kernel forward/backward wall-clock time with and without comm-overlap (Table 2):

| Seq | forward w/o overlap | forward w/ overlap | backward w/o | backward w/ |
|---|---|---|---|---|
| 4K | 29.5 µs | 35.0 µs (+18.6%) | 77.7 µs | 82.2 µs (+5.8%) |
| 8K | 49.3 µs | 54.6 µs (+10.7%) | 123.3 µs | 129.8 µs (+5.3%) |
| 16K | 122.1 µs | 131.2 µs (+7.5%) | 362.9 µs | 367.0 µs (+1.1%) |
| 24K | 239.2 µs | 250.9 µs (+4.8%) | 730.0 µs | 743.2 µs (+1.8%) |
| 32K | 402.9 µs | 420.1 µs (+4.2%) | 1218.9 µs | 1225.3 µs (+0.5%) |

The communication-overlap design of Ring-style SP **slows down the attention kernel by occupying SM resources** at short sequences (the +18.6% forward at 4K shows the SM contention dramatically). This contention shrinks at longer sequences (where attention is naturally compute-dominant) but is non-zero. This is the *same* contention argument fal makes in the SymmMem section (P.6.4): NCCL collectives are kernels that consume SMs, and even with `async_op=True`, they throttle compute kernels. The MM-SP implication is that the benefit of comm-overlap isn't free — at short sequences (<8K), it can actually hurt.

**Reported headline numbers.** **2.1×–5.7× over ZigZag-RingAttn at 2M-token training on 256 H100 GPUs**; **1.1×–1.4× over Megatron CP+TP** at the same setup. Scales linearly to **2M context** without gradient checkpointing (the practical limit of LLM training systems before MM-SP). Achieves **99.8% accuracy in 6,000-frame video needle-in-a-haystack** (>1M context length).

**Inference mode.** MM-SP also runs in inference mode by sharding the input sequence across SP ranks (matching training-mode placement), instead of using HuggingFace's built-in pipeline-parallel inference (which only activates one GPU at a time and has memory bottlenecks at the first device). MM-SP's inference scheduler is illustrated in the paper's Figure 8: all GPUs participate in computation simultaneously, with memory evenly distributed.

**Relevance to video DiT inference.** The structural pattern (separate SP groups for text-replicated vs visually-sharded streams) **directly maps to inference for video DiTs** that have a text-encoder + vision-DiT architecture. The SGLang cross-attn skip-USP optimization (PR #19419, P.9.7) is essentially the inference-side echo of this insight — when KV is replicated (text encoder, image encoder), don't pay the SP overhead on it.

For 8-GPU B200 single-node Wan/HunyuanVideo/LTX-2 inference (no inter-node IB), the MM-SP networking-asymmetry argument doesn't apply (uniform NVLink5). But the *modality-heterogeneity* argument does: ControlNet conditioning and image-to-video prompts add short replicated-stream tokens that should not flow through the long-stream SP pipeline. This is exactly the SGLang skip-sp story.

---

## P.5 DSP — Dynamic Sequence Parallelism for axis-factorized DiTs

### P.5.1 Motivation

Multi-dimensional transformers like Open-Sora, Latte, and earlier video DiTs alternate between *temporal* and *spatial* attention rather than running full 3-D attention. Embedded SP (Ulysses, Ring, USP) shards along a single dimension and pays a re-shard penalty whenever the model switches from temporal to spatial blocks. DSP (Zhao et al., arXiv:2403.10266, ICML'25) replaces the static single-axis sharding with a *dynamic switch* between sharding dimensions, eliminating the re-shard.

### P.5.2 Core algorithm

Given input `X ∈ ℝ^{B × S₁ × S₂ × ... × C}` with `K` independent sequence dimensions, DSP partitions along the *currently inactive* dimension. When the next block uses the previously-sharded dimension, DSP re-shards via a single `all-to-all` between consecutive blocks — the same primitive as Ulysses but applied between blocks rather than within a block. The dynamic primitive set:

| Source shard | Target shard | Primitive | Comm volume | Frequency |
|---|---|---|---|---|
| `s_i` | `s_i` | (no-op) | 0 | / |
| `s_i` | `s_j` | Switch | M/N (a2a) | High |
| `ŝ` (unsharded) | `s_i` | Split | 0 | Low |
| `s_i` | `ŝ` (unsharded) | Gather | M (allgather) | Low |

So between adjacent transformer blocks, DSP performs a single All-to-All to switch between sharding dimensions. Crucially this lets each block run its attention *unchanged* on its (different) sharded dimension — the per-block cost is identical to single-GPU.

### P.5.3 Reported numbers

- **32.2% – 10× over single-axis SP** end-to-end throughput.
- **≥ 50% communication volume reduction**.
- **3× train / 2× inference speedup** for Open-Sora.

The 10× upper-bound is for sequence lengths and parallelism degrees where embedded SP forces frequent gather-and-redistribute; the lower bound (~32%) is for shorter sequences where the embedded SP's overhead is small to begin with.

### P.5.4 Why DSP doesn't help Wan / HunyuanVideo / LTX-2

Wan, HunyuanVideo, and LTX-2 all use **full 3-D attention** — there is no axis-factorization to exploit. The 3D-attention block sees a single token sequence `seq = T·H·W` (after patchify) and does `O((THW)²)` work on it. DSP's contribution is exactly to route around the temporal-spatial block alternation, but full 3-D attention has no such alternation. So DSP buys nothing for the 2025–2026 generation of full-3D-attention video DiTs.

DSP *is* the right answer for Open-Sora-style models that retain spatial-temporal block alternation. Open-Sora 2 still uses spatial-temporal blocks; Latte is the canonical case. For the Wan/HunyuanVideo/LTX family the production stacks (xDiT, SGLang, FastVideo) all use Ulysses or USP, not DSP. VideoSys (NUS-HPC-AI-Lab) is the canonical DSP integration for Open-Sora. As Open-Sora 2 evolves toward full-3D attention this gap may close.

### P.5.5 Composability

DSP is orthogonal to TP (it shards along sequence axes; TP shards along feature axis), and orthogonal to PP. It is *not* orthogonal to Ulysses on the same axis — they are alternatives. For multi-axis transformers you stack DSP on the inner sequence axes plus TP/FSDP on the model dimensions.

---

## P.6 Async / overlapped Ulysses on B200 (VeOmni + fal blog)

### P.6.1 Motivation — the structural observation

Vanilla Ulysses serializes `[compute Q] → [a2a Q] → [compute K] → [a2a K] → [compute V] → [a2a V] → [SDPA]` through 6 host-launched kernel boundaries on a single CUDA stream. The Q/K/V projections share *zero* data dependencies, so they can be reordered to overlap each branch's all-to-all with the *next* branch's compute, without changing the math. The fal blog (["Ulysses Unbound"](https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/)) phrases it: *"the Q/K/V branches are independent before attention. That lets us overlap communication from one branch with compute from the next, without changing the math."*

This is the most B200-relevant SP optimization, because the post-attention all-to-all is otherwise the second-largest pre-attn cost behind QKV projection (per fal's instrumented runs on 8×B200), and the slack it exposes is exactly what Async Ulysses turns into wall-clock savings.

### P.6.2 The `async_ulysses.py` schedule (VeOmni source)

```python
# Quoted from VeOmni's async_ulysses.py
q = F.linear(hidden_states, q_weight, q_bias).view(...)
q_res = all_to_all_tensor(q, ..., async_op=True)             # ← non-blocking

k = F.linear(hidden_states, k_weight, k_bias).view(...)
if ctx.need_repeat_kv:
    k = torch.repeat_interleave(k, dim=2, repeats=ctx.n_repeat)
k_res = all_to_all_tensor(k, ..., async_op=True)

v = F.linear(hidden_states, v_weight, v_bias).view(...)
if ctx.need_repeat_kv:
    v = torch.repeat_interleave(v, dim=2, repeats=ctx.n_repeat)
v_res = all_to_all_tensor(v, ..., async_op=True)

q = q_res(); q = unpadding_tensor_for_sp(q, ...)
k = k_res(); k = unpadding_tensor_for_sp(k, ...)
# ... QK normalization (RMSNorm/LayerNorm) and RoPE here, V's a2a still in flight ...
v = v_res(); v = unpadding_tensor_for_sp(v, ...)
return output_q, output_k, v
```

Implementation notes:
1. **Branches don't share kernels**: three separate `F.linear` calls — overlap is across boundaries (Q-a2a ↔ K-compute, K-a2a ↔ V-compute, V-a2a ↔ QK-norm-and-RoPE).
2. **GQA repeat-interleave happens *before* the a2a** so head shards are uniform across ranks. This pays the comm cost on the larger materialized tensor — unavoidable for GQA where `ulysses_size > num_kv_heads`.
3. **QK normalization sits between `k_res()` and `v_res()`** so V's a2a overlaps with QK-norm.
4. **Backward pass is symmetric**: same overlap pattern in reverse.
5. **Two transport variants:** the classic NCCL `async_op=True` (carries `Work` handles, explicit `wait`s) and PyTorch *functional collectives* (`torch.distributed._functional_collectives` returning `AsyncTensor` values, composes with `torch.compile`). fal's blog reports the functional path is the cleaner choice.

### P.6.3 fal blog — the key 8×B200 measurements

Setup: 8×B200, bf16, B=1, heads=40, head_dim=128, fixed clocks, cuDNN attention backend. 15 warmup + 40 timed iterations × 3 repeats. Reports max-rank median latency. They report both **chunk latency** (QKV projection + pre-SDPA communication, the optimization target) and **end-to-end step latency** (whole pre-attention path).

| Strategy | Chunk @2 GPU | Chunk @4 GPU | Chunk @8 GPU | E2E @2 GPU | E2E @4 GPU | E2E @8 GPU |
|---|---|---|---|---|---|---|
| Async Ulysses (NCCL functional collectives) | **−23%** | **−25%** | **−24%** | **−3%** | **−3%** | **−3%** |
| Async Ulysses + Symmetric Memory transport | better | better | regresses | (same shape) | (same shape) | regresses |
| Fused QKV (`torch.ops.symm_mem.fused_all_gather_matmul`) | **−37.3%** | **−33.4%** | **−4.6%** | **−5.0%** | **−4.8%** | **−0.3%** |

**Two takeaways:**

1. **Async Ulysses gives consistent ~23–25% chunk reduction across `P`**, but ~3% e2e because SDPA + output projection still dominate total step time. The chunk itself is roughly 12% of the step (attention SDPA + output projection are the rest), so 25% × 12% ≈ 3% e2e — exactly what the data shows.

2. **Fused-QKV via SymmMem is the strongest small-`P` optimization** (P ≤ 4) but **regresses at P=8** because the per-rank message becomes too small to amortize symm-mem fixed overheads (rendezvous, signal-pad handshaking, buffer setup). When local seq is fixed at 16k (weak scaling), Fused QKV is best at P=2,4; Async Ulysses scales more smoothly and is best at P=8.

The takeaway is **no single strategy dominates** — a runtime policy that picks based on world size and message size is the practical answer. Production stacks haven't shipped this auto-selector yet (open question for vLLM-Omni and SGLang Diffusion).

### P.6.4 SymmMem transport — why and how

The fal blog's deeper observation: even when `async_op=True` puts collectives on a *non-default stream*, NCCL collectives are *kernels* that consume SM resources. They contend with concurrent GEMMs for SM cycles, so timeline overlap doesn't always translate to throughput overlap — NCCL and GEMMs can overlap *in time* while still throttling each other on SM/L2/HBM bandwidth.

Routing pre-attention transfers through PyTorch SymmMem with the **Copy Engine** path moves data through DMA hardware (separate from SMs), freeing SM capacity for compute. Same architectural insight behind CUTLASS DistGEMM and Triton-distributed (covered in the parent report's §1.3). The `async_symm` variant in fal's data is "Async Ulysses with SymmMem transport"; it improves over plain Async Ulysses at P=2,4 but regresses at P=8 because at P=8 each rank's pre-attn payload shrinks below the threshold where copy-engine fixed overhead is amortized.

The full list of fixed overheads SymmMem pays per rendezvous: (a) symmetric heap allocation/registration, (b) signal-pad handshaking (each rank publishes a 4-byte ready flag), (c) buffer-pointer exchange, (d) IPC handle export+import on the first rendezvous, (e) per-launch `cudaStreamWaitEvent` with the comm stream. Total per-rendezvous cost on B200 is ~15–25 µs; if the aggregate transfer is < 50 µs, SymmMem loses to NCCL functional collectives.

### P.6.5 Fused QKV via `fused_all_gather_matmul`

The natural follow-up to overlap is fusion: reduce kernel and collective count. Instead of running separate Q/K/V projections and separate a2a's, each rank builds *one packed local weight shard* (`Q | K | V` rows for its local heads) and calls `torch.ops.symm_mem.fused_all_gather_matmul`. That single op performs sequence all-gather + local-head matmul together, returning a packed `(B, S_global, 3·H_local·D)` output. Then split into Q/K/V and reshape back to gathered local-head tensors for SDPA.

This is the strongest chunk optimization at P=2,4 (−37.3%, −33.4%) and gives the best e2e gains there (−5.0%, −4.8%). At P=8 the benefit mostly vanishes (−4.6% chunk, −0.3% e2e): messages are too small under strong scaling, so fixed fusion overheads take over.

### P.6.6 Weak-scaling sweep — local-seq fixed

For weak scaling at local-seq 16k (so global seq grows with `P`):
- Fused QKV is best at P=2,4
- Async Ulysses scales more smoothly and is best at P=8
- Async Ulysses with Symmetric Memory is competitive at lower scale, then degrades faster

Takeaway from fal: **overlap is the most robust high-scale default; fusion is strongest in lower/mid-scale regimes**. Communication-heavy workloads like MoE routing should benefit even more (since routing has more communication per FLOP).

### P.6.7 Recipes that actually ship

- **VeOmni** (ByteDance) ships the async path natively (`async_ulysses.py`).
- **fal's PoC** is published as recipe code; not yet productized in xDiT/SGLang.
- **xDiT** uses synchronous Ulysses; async-Ulysses lands in xDiT's roadmap as of Q1 2026.
- **SGLang Diffusion** uses the synchronous path; async-Ulysses lands as of `--enable-async-ulysses` flag (proposed PR #19xxx).
- **FastVideo** uses synchronous Ulysses; async-Ulysses is the main remaining single-step lever.
- **diffusers native CP** uses synchronous Ulysses; async support is on the diffusers PR roadmap.

### P.6.8 Limitations and open issues

- The 3% e2e win is small — the win scales with how much of step-time is actually pre-attn chunk (currently ~12%). For very-low-step distilled models (4–8 steps), each step is dominated by SDPA + output projection.
- Fused-QKV regresses at P=8 (the standard B200 single-node count). A "small-P fused QKV + large-P async Ulysses" runtime selector would be the right answer; not yet shipped.
- No published B200 numbers for `seq > 32k` end-to-end with async-Ulysses (open question for HunyuanVideo/Wan-extended at high resolution).
- `repeat_interleave` for GQA inflates the message size before a2a — for MoE-shaped models the inflation factor is large.

---

## P.7 Megatron-SP option B + PyTorch async-TP

### P.7.1 Motivation

Megatron-LM "option A" tensor parallelism (Shoeybi et al., arXiv:1909.08053) used **AllReduce after each row-parallel layer** to sum the partial outputs. Total comm volume per layer per rank: `2·(P-1)/P · S` bytes. The whole AR must hide behind the *smaller* of the two GEMMs (the row-parallel one — typically the FFN-down or O-projection), which is the bottleneck. For long sequences the activation memory under option A is also wasteful: the layernorm and dropout regions sit *outside* the AR boundaries and replicate full-`s × b × h` tensors.

Korthikanti et al. (arXiv:2205.05198, "Reducing Activation Recomputation in Large Transformer Models") introduced **option B** ("Megatron-SP"): replace the AllReduce with **AllGather before the column-parallel layer + ReduceScatter after the row-parallel layer**. Same total volume `2·(P-1)/P · S`, but split into two collectives that can hide behind two different GEMMs of asymmetric size. The region between AG and RS is sequence-sharded, which also reduces activation memory by `P×`.

### P.7.2 Detailed comm schedule

The MLP block under tensor parallelism with option B:

```
Forward:
    X_sharded (s/P, b, h)
    Y = LayerNorm(X)          # sequence-sharded
    Y = AllGather(Y)          # → (s, b, h), full sequence locally; "g"
    Z = GeLU(Y · A_col)       # column-parallel GEMM, each rank holds 1/P of output (s, b, 4h/P)
    W = Z · B_row             # row-parallel GEMM, each rank computes partial sum (s, b, h)
    W = ReduceScatter(W)      # → (s/P, b, h), sequence-sharded; "ḡ"
    V = Dropout(W)            # sequence-sharded

Backward: g̃ is reduce-scatter forward, all-gather backward; ḡ̃ is all-gather forward, reduce-scatter backward.
```

The key insight is that ring-AllReduce decomposes into ReduceScatter + AllGather. So option A's one AllReduce equals option B's RS+AG by *volume*, but option B *separates the two halves* and lets each one overlap with a different GEMM. Net: *exposed* comm is the smaller of the two, not the AR's worst-case half.

The activation-memory win is large: under option A, the full `(s, b, h)` activation between the layernorm and dropout is replicated across TP ranks. Under option B, the same activation is sequence-sharded `(s/P, b, h)`. For a 70B-class model with `s = 100k`, `b = 1`, `h = 8192`, this is a ~`P×` reduction in activation memory.

### P.7.3 The TE/Megatron-LM interface

The TE [advanced-optimizations doc](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.10/user-guide/examples/advanced_optimizations.html) confirms: *"sequence parallelism distributes along the sequence_length dimension. This can be used when tensor parallelism is enabled in order to parallelize operations that run outside the tensor-parallel region (e.g. layer norm)."* The toggle:

```python
te.TransformerLayer(
    ...,
    set_parallel_mode=True,        # enable TP
    sequence_parallel=True,        # flip from option A to option B
    context_parallel_group=cp_pg,  # USP from feifeibear/long-context-attention
)
```

Every modern stack (Megatron-LM, vLLM, SGLang, Wan official, xDiT, CUTLASS DistGEMM examples 65/82, TorchTitan) defaults to option B. Option A is essentially deprecated except for shape-mismatched layers where the AR semantics are simpler.

### P.7.4 Async-TP in PyTorch

**References:** [TorchTitan PR #429](https://github.com/pytorch/torchtitan/pull/429) (June 2024), [SymmMem PR #179918](https://github.com/pytorch/pytorch/pull/179918), FP8 rowwise PR #149247, vLLM thresholds [#27700](https://github.com/vllm-project/vllm/issues/27700).

**Mechanism.** The PyTorch `torch.ops.symm_mem.fused_all_gather_matmul(input, weight)` op replaces the AG-then-GEMM sequence with a single fused operation that pipelines per-tile copies and per-tile GEMMs through the SM-bypass copy-engine path enabled by symmetric memory. Symmetrically, `fused_scaled_matmul_reduce_scatter` fuses GEMM-then-RS. The compiler-pass version (TorchTitan PR #429) pattern-matches the AG+matmul sequence in the desugared aten IR and rewrites to the fused op when `--experimental.enable_async_tensor_parallel` is set on the run config. Trace samples in PR #429 show real overlap: the all-gather kicks in tiles right when each per-tile copy completes, and the FFN-up GEMM consumes them as they arrive.

**Fused-AG-matmul algorithm sketch:** With `P` ranks each holding `1/P` of the input rows and the full weight, the fused op:
1. Initiates a P2P-style chunked copy of `input_chunk_0` to all peers via copy engine.
2. Starts the local matmul of `input_chunk_local × W` on SMs.
3. As each peer chunk arrives, immediately schedules `input_chunk_peer × W` on SMs without blocking on full AG.
4. Concatenates partial outputs locally.

The producer-consumer pattern is per-tile (~1–4 MB tiles) so the SM-side compute is never idle, and the copy-engine side never has to wait for SM capacity. Same architectural idea as CUTLASS DistGEMM examples 65/82, just packaged at the torch op level.

### P.7.5 FP8 / NVFP4 status

FP8 *rowwise* scaling support landed in [PR #149247](https://github.com/pytorch/pytorch/pull/149247). The operator is `fused_scaled_matmul_reduce_scatter` with FP8 scales. Current status:

- **FP8 rowwise:** ✅ supported, threshold ~32 MB (smaller than BF16's 40 MB because FP8 has 2× higher arithmetic intensity).
- **FP8 blockwise (e.g., DeepSeek-V3 style 1×128 / 128×128):** not native to async-TP path; landing as part of the DeepGEMM integration roadmap.
- **NVFP4 block-scaled:** ❌ not yet upstream as of May 2026. The two scale tensors (E4M3 per-block + FP32 per-tensor) must move with activations across SP A2As and TP RS. The right scale-tensor sharding strategy is open.
- **MXFP8 (32-element blocks, E8M0 scale):** partial — supports per-block but block size 32 is awkward for the 128-element default tile.

### P.7.6 The small-batch / small-token threshold (vLLM #27700)

Real LLM production data ([vLLM #27700](https://github.com/vllm-project/vllm/issues/27700)) closed Feb 26 2026. Empirical findings on H100:

- **Async-TP regresses below ~40 MB tensor size on H100** (i.e., `dtype_bytes · hidden_size · num_tokens / tp_size < 40 MB`).
- **FP8 needs ~32 MB threshold** (lower because FP8 is more compute-bound per byte).
- **Below the threshold, host-side overheads eat the win** — fixed costs of fused-op dispatch, symmetric-memory rendezvous, and SM-launch latency dominate.
- **Multi-dimensional heuristic**: per the discussion, the proposed default is `per_gpu_tensor_size_threshold = 8 MB if hidden_size ≥ 8192 else infinity` — i.e., async-TP only auto-engages for Llama-70B-class hidden sizes; smaller models keep the synchronous TP path.

The closed PR #28672 implemented this configurable threshold; currently the user must opt in via `--enable-async-tp-threshold-num-tokens N` and hidden-size > 8192 gates the default-on logic.

### P.7.7 Why this matters less for video DiT

Video DiT inference at `seq ≥ 1k tokens × hidden 4096–6144 × bf16 = 8–12 MB minimum`. At seq = 12k the per-rank tensor on TP=4 is `12k × 5k × 2 = 120 MB` — comfortably above the 40 MB threshold. So async-TP **engages cleanly** for any reasonable video-DiT inference, unlike LLM decode where small batches sit below the threshold.

The relevant comparison is async-TP on hybrid `SP=2 × TP=4` for Wan 2.2 720p 81f (`seq = 75k`): per-rank `M = 37.5k`, way above ridge, and per-rank tensor `≈ 750 MB` — async-TP gives near-100% overlap. For LTX-2 1080p 5s 8-step distilled at `seq = 12k` and `SP=2 × TP=4`: per-rank `M = 6k`, still above NVFP4 ridge, per-rank tensor `≈ 120 MB` — async-TP still engages.

### P.7.8 Setup recipe (production)

```bash
# TorchTitan
torchrun --nproc_per_node=8 train.py \
  --parallelism.tensor_parallel_degree 4 \
  --parallelism.context_parallel_degree 2 \
  --experimental.enable_async_tensor_parallel \
  --activation_checkpoint.mode "selective"

# Or programmatically
import torch._inductor.config as inductor_cfg
inductor_cfg._micro_pipeline_tp = True
torch.compile(model)
```

For inference (vLLM/SGLang), the equivalent is `--enable-async-tp` plus the threshold flag.

### P.7.9 Limitations

- NVFP4 block-scaled async-TP is the biggest open gap (parent report's open question #3).
- The TorchTitan path requires `torch.compile` (the compiler-pass version is the only path that auto-rewrites; the manual op is the alternative if you don't want to compile).
- Async-TP is incompatible with eager-mode debugging (the compiler pass only fires under `torch.compile`).

---

## P.8 PyTorch SymmetricMemory 2.12 + NVSHMEM4Py

### P.8.1 What SymmMem unlocks

PyTorch SymmMem (alpha in 2.12, [docs](https://docs.pytorch.org/docs/2.12/symmetric_memory.html); recipe library [yifuwang/symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes)) gives three new capabilities not available through standard NCCL collectives:

1. **Customized communication patterns.** Triton/CUDA kernels can issue device-initiated puts/gets directly to peer GPUs over NVLink, no NCCL collective dispatch needed. You write the comm exactly the way you want it (e.g., per-tile barrier release in Triton-distributed; see parent report §1.3).
2. **In-kernel compute-comm fusion.** The same kernel can do compute and comm at the smallest granularity (per-tile, per-warp). This is what "async-TP" leverages on the back end.
3. **Low-latency RDMA for multi-node.** NVSHMEM RDMA bypasses the network stack and CPU; on multi-node setups this is what makes 100k+ GPU clusters feasible.

For single-node B200 video DiT, capabilities (1) and (2) are the relevant ones.

### P.8.2 The "Hello World" — symmetric tensor

```python
import torch.distributed._symmetric_memory as symm_mem

t = symm_mem.empty(128, device=torch.device("cuda", rank))
hdl = symm_mem.rendezvous(t, group)
# hdl exposes:
#   hdl.buffer_ptrs       # peer pointers (one per rank in group)
#   hdl.multicast_ptr     # NVLink-SHARP multicast pointer (Blackwell only)
#   hdl.signal_pad_ptrs   # 4-byte signal pads for handshaking
```

`empty` and `rendezvous` are *collective* — must be called in the same order on all ranks in the group. After rendezvous, each rank can read/write peer memory directly via the buffer pointers, no further collective dispatch needed.

### P.8.3 Three backends (NVSHMEM, CUDA, NCCL)

`set_backend("NVSHMEM" | "CUDA" | "NCCL")` controls how symm tensors are wired:

- **CUDA**: `cudaIpcGetMemHandle` + `cudaIpcOpenMemHandle` for intra-node direct peer-access. Simplest, intra-node only.
- **NCCL**: NCCL 2.28+ uses symmetric memory under the hood for "NCCL symmetric" collectives. Both intra-node (P2P) and inter-node (RDMA) work.
- **NVSHMEM**: device-initiated puts/gets via the NVSHMEM library. Supports both intra-node and inter-node via IBGDA. Best for fine-grained per-tile communication kernels.

### P.8.4 Op surface (`torch.ops.symm_mem`)

Common ops directly callable on symm tensors:

- `one_shot_all_reduce(input, "sum", group_name)` — sum all ranks in one shot via shared peer access.
- `two_shot_all_reduce_(input, "sum", group_name)` — two-stage AR for larger tensors / better bandwidth.
- `multimem_all_reduce_(input, "sum", group_name)` — uses NVLink-SHARP hardware multicast (B200 only).
- `multimem_all_gather_out(input, group, out)` — multimem AG.
- `fused_all_gather_matmul(input, weight)` — TP option-B AG-side fused.
- `fused_scaled_matmul_reduce_scatter(input, weight, scale, ...)` — TP option-B RS-side fused.
- `all_to_all_vdev(input, out, in_splits, out_splits_offsets, group)` — variable-length all-to-all; symm-only.
- `all_to_all_vdev_2d(...)` — 2-D all-to-all-v for MoE dispatch.
- `tile_reduce(in_tile, out_tile, root, group)` — 2-D tile reduction (used by Triton-distributed).

### P.8.5 Memory pool (PyTorch 2.12 — new)

```python
mempool = symm_mem.get_mem_pool(device)
with torch.cuda.use_mem_pool(mempool):
    x = torch.arange(128, device=device)
torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name)
```

The `MemPool` context lets you turn *any* compute output into a symmetric tensor by running it inside the mempool context. This is critical for PyTorch's auto-promotion of tensors to symm memory when async-TP fires under `torch.compile`. Without the mempool, you'd have to manually `symm_mem.empty + rendezvous` every tensor that crosses an async-TP boundary.

As of torch 2.11, CUDA and NVSHMEM backends support MemPool. NCCL backend support is in progress.

### P.8.6 Higher-precision reduction (PyTorch 2.12)

When the symmetric-memory backend is `NCCL`, BF16/FP16 reduce_scatter and all_reduce automatically accumulate in **FP32 internally** before producing BF16/FP16 outputs (BF16 in → FP32 accumulate → BF16 out). This closes the FP16 catastrophic-cancellation gap that has bitten large-cluster training for years.

Scope: applicable to `reduce_scatter` and `all_reduce` only; within the NVLink domain as of torch 2.9 (NCCL 2.27); NVLink + network for `reduce_scatter` as of torch 2.11 (NCCL 2.29). Other collectives (e.g., `all_gather`) and inter-node communication beyond reduce_scatter are not affected.

### P.8.7 Copy Engine Collectives (PyTorch 2.12 / NCCL 2.28+)

The canonical "use copy engines for collectives" setup:

```python
opts = dist.ProcessGroupNCCL.Options()
opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO   # ← critical
device = torch.device("cuda", rank)
dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)
symm_mem.set_backend("NCCL")

inp = symm_mem.empty(numel, device=device)
out = symm_mem.empty(numel * world_size, device=device)
symm_mem.rendezvous(inp, group=group_name)
symm_mem.rendezvous(out, group=group_name)

# As of NCCL 2.28, CE collectives need async_op=True + non-default stream
work = dist.all_gather_into_tensor(out, inp, async_op=True)
work.wait()
```

After this setup, standard `dist.all_gather_into_tensor()` and `dist.all_to_all_single()` *automatically* use copy engines instead of SMs when the tensors are symm-mem-backed. This is what "freeing SM capacity for compute" means in the fal blog and in CUTLASS DistGEMM — same mechanism, different layer.

The NCCL_CTA_POLICY_ZERO knob is critical: it tells NCCL to use 0 SMs for the collective (everything goes through copy engines). Old NCCL allocated some SMs by default; the new "zero" policy is the unlock for compute-only SM utilization.

Limitations:
- NCCL ≥ 2.28 required.
- GPUs must have peer-to-peer (P2P) access enabled (`nvidia-smi topo -p2p r` shows `OK`).
- Tensors must be allocated via `torch.distributed._symmetric_memory.empty()` and rendezvoused.
- Can't run on the default stream — must use `async_op=True` (activates the internal stream of `ProcessGroupNCCL`) or create a side stream yourself.

### P.8.8 NVSHMEM device-side collectives [VERIFIED]

Two waves of additions relevant to Blackwell ([NVSHMEM 3.5.19 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3519.html), [NVSHMEM 3.2.5 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3205.html)):

- **NVSHMEM 3.2.5** added one-shot and two-shot **NVLINK SHARP (NVLS) allreduce** on NVLink5/B200 platforms for fp16/bf16/fp32 datatypes. SM100 platform support enabled. This is what makes in-network reduction usable on Blackwell from CUDA kernels.
- **NVSHMEM 3.5.19** added **tile-granular RMA**: `nvshmemx_tile_put_*`, `nvshmemx_tile_get_*`, `nvshmemx_tile_broadcast_*`. Tile-based collectives are device-only (issued from inside a CUDA kernel rather than from host launches). This is the path the ParallelKittens paper benchmarks (parent report §1.3) as the primary mechanism for 1%-non-overlapped all-reduce.
- **NVSHMEM 3.6.5** added error-code returns to tile API calls and fixed an IBGDA CST bug.

CUTLASS support for the tile API was added in the same NVSHMEM 3.5.19 release.

### P.8.9 The recipe library (`yifuwang/symm-mem-recipes`)

[`symm-mem-recipes`](https://github.com/yifuwang/symm-mem-recipes) is the cookbook. Notable: `triton_all_gather_matmul.py` reports **1.54× / 1.47× / 1.20× over cuBLAS+NCCL for Llama-3-8B/70B/105B at M=16384 on 8×H100 NVSwitch**. The win shrinks at larger model sizes because the GEMM time dominates (less overlap headroom), not because the comm is slower. Same recipes apply on B200 with bigger numerator (FA4 at 1.6 PF/s vs cuDNN at 1.0 PF/s).

`meta-pytorch/kraken` (parent report §1.3) is the "production cookbook" — Triton+SymmMem with one-shot/two-shot AR, `gemm_one_shot_all_reduce_fused`, AG/RS fused, PTX synchronization helpers, used by fal as the reference for device-initiated comm on B200.

### P.8.10 Fixed overheads — when not to use SymmMem

SymmMem is *not* free: per-rendezvous cost is ~15–25 µs on B200 (signal-pad handshaking, IPC handle setup, buffer-pointer exchange). For a single transient transfer of `< 1 MB`, NCCL functional collectives win. The crossover is around 1–10 MB depending on backend; below, NCCL functional + `async_op=True` is simpler and as fast.

For the per-attention-block all-to-all in video DiT (~10–100 MB per a2a at SP=8), SymmMem wins decisively when you can amortize one rendezvous over many transfers (which the MemPool context enables).

---

## P.9 Hybrid SP×TP + production frameworks

### P.9.1 The hybrid `SP=2 × TP=4` argument

For 8-GPU video DiT inference on B200 with `seq ∈ [10k, 75k]` (which covers Wan 2.2 720p 81f, LTX-2 1080p 5s 8-step, HunyuanVideo 720p 65f), the hybrid `SP=2 × TP=4` configuration is the production sweet spot. The argument:

| Knob | Pure SP=8 | Pure TP=8 | Hybrid SP=2 × TP=4 |
|---|---|---|---|
| Per-rank `M` | seq/8 | seq | seq/2 |
| For seq=12k: per-rank `M` | 1.5k (below ridge) | 12k (above) | 6k (above ridge) |
| For seq=75k: per-rank `M` | 9.4k (above) | 75k (above) | 37.5k (above) |
| Per-layer comm | 4 a2a (Ulysses) | AG+RS pair | 4 a2a (SP=2) + AG+RS (TP=4) |
| Activation memory | 1/8 | 1/1 (replicated) | 1/2 |
| Param memory | 1/1 (FSDP can shrink) | 1/8 (sharded) | 1/4 |
| HEAD constraint | `8 ≤ h_c` (OK at 24+) | None | `2 ≤ h_c` (always OK) |

**Hybrid keeps `M = seq/2` (above the NVFP4 ridge for any reasonable seq) while only per-layer-AR-ing across 4 ranks** (smaller comm volume than TP=8, hideable behind the larger of the two GEMMs by Megatron option B). xDiT, SGLang, FastVideo, vLLM-Omni, and Wan official converge on this configuration as the recommended 8-GPU default for Wan 2.2 / HunyuanVideo / LTX-2 inference. SGLang's published `u1r2` benchmark on Wan 2.2 TI2V-5B shows the *2-GPU* expression of the same principle (SP=2; covered in P.9.7).

### P.9.2 xDiT — the multi-knob walkthrough

[xDiT](https://github.com/xdit-project/xDiT) exposes four orthogonal knobs that map to a 4-D process grid:

- **`--ulysses_degree=u`**: head-sharded SP via 4 a2a per attention block. Constraint `u ≤ heads`.
- **`--ring_degree=r`**: KV-rotated SP via P2P + online-softmax merge. No head constraint.
- **`--cfg_parallel`**: classifier-free guidance parallelism — generates the conditional and unconditional pass on different ranks. For CFG=2 this is a free `2×` for a `+1×` GPU cost. **Used in xDiT's 6-GPU CogVideoX-5B benchmark to get to 7.75×**: `cfg_parallel × ulysses_degree=2 × ring_degree=3 = 12 effective × 6 GPUs = 7.75× e2e`.
- **`--tp`**: tensor parallelism via Megatron-SP option B. Cited as the dominant mode for Step-Video-T2V 30B on 8×H20.

Published numbers in [stepvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/stepvideo.md), 8×H20 NVLink:

| GPUs | Parallel | Latency | Speedup | Memory |
|---|---|---|---|---|
| 1 | TP1 SP1 (baseline) | 213.60 s | 1.00× | 92,170 MB |
| 2 | TP2 | 108.97 s | 0.98× | 57,458 MB ▼37.7% |
| 2 | SP2 | 108.13 s | 0.99× | 86,258 MB ▼6.4% |
| 4 | TP4 | 57.61 s | 0.93× | 36,566 MB ▼60.3% |
| 4 | SP4 | 57.01 s | 0.94× | 78,226 MB ▼15.1% |
| 8 | TP8 | 30.40 s | 0.88× | 30,028 MB ▼67.4% |
| 8 | SP8 | 30.10 s | 0.89× | 79,684 MB ▼13.5% |

**Key takeaways:**
- Pure SP and pure TP have nearly identical latency (within 1%) at every scale on H20 — H20's NVLink is fast enough that the SP a2a's are nearly free.
- **TP saves dramatically more memory** — TP8 uses 33% the memory of SP8 (30 GB vs 80 GB) because TP shards parameters, while SP only shards activations.
- TP scales near-linearly to 8 GPUs (TP8 at 0.88× per-GPU efficiency).
- Memory-vs-latency is the real tradeoff: pure SP saves activation memory only; pure TP keeps parameters healthy but replicates layernorm activations.

CogVideoX numbers ([cogvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/cogvideo.md)) on 6×L40 PCIe for 49f@720×480: **7.75×** end-to-end via `cfg_parallel × ulysses=2 × ring=3 = 6 GPUs`. The CFG-parallel knob is the differential — it doubles throughput for free given a `2×` GPU budget. CogVideoX1.5-5B at 161f@1360×768 on 8 GPUs: **6.12×** with `ulysses=2 × ring=2 × cfg=2`.

Mochi-1 on 6×L40, 49f@848×480: **3.54× vs official** (74s vs 263s) with `cfg=2 × ring=3` configuration.

### P.9.3 ParaAttention — First-Block-Cache + USP

[chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention) is the WaveSpeed.ai distribution of a "context parallel attention + dynamic caching" stack. Two contributions:

1. **First-Block-Cache (FBC).** Diffusion DiTs have many blocks (Wan 2.2: 30 layers; HunyuanVideo: 60 layers); FBC caches the *output of the first block* across denoising steps when the noise change is small (heuristic: per-token L2 norm of the noise-level delta below a threshold). **~2× alone** on HunyuanVideo + Flux ([HF docs](https://huggingface.co/docs/diffusers/main/optimization/para_attn)) with no quality degradation up to ~30% of steps cached. This is *not* a sequence-parallelism contribution; it's a step-skipping caching contribution that composes orthogonally with SP.
2. **Context Parallelism via Ulysses + Ring + USP.** Same backend as xDiT.

ParaAttention does *not* support TP. For models that need TP (Step-Video-T2V 30B), xDiT is the right answer.

### P.9.4 FastVideo distributed inference (local code dive)

Reading the actual FastVideo distributed code (`/Users/yewang/work/FastVideo/fastvideo/distributed/`):

`communication_op.py` exposes:

```python
def sequence_model_parallel_all_to_all_4D(input_: torch.Tensor,
                                          scatter_dim: int = 2,
                                          gather_dim: int = 1) -> torch.Tensor:
    """All-to-all communication of 4D tensors (e.g. QKV matrices)."""
    return get_sp_group().all_to_all_4D(input_, scatter_dim, gather_dim)
```

The implementation in `device_communicators/base_device_communicator.py::DistributedAutograd.AllToAll4D.forward` shows the actual layout shuffle:

```python
if scatter_dim == 2 and gather_dim == 1:
    bs, shard_seqlen, hn, hd = input_.shape           # (B, seq/P, H, D)
    seqlen   = shard_seqlen * world_size
    shard_hn = hn // world_size                       # heads per rank after a2a
    input_  = input_.transpose(0, 2).contiguous()     # (H, seq/P, B, D)
    output  = torch.empty_like(input_)
    dist.all_to_all_single(output, input_, group=group)  # NCCL a2a
    output  = torch.cat(output.split(shard_hn), dim=1)   # (H/P, seq, B, D)
    output  = output.transpose(0, 2).contiguous()        # (B, seq, H/P, D)
    return output
```

**Key observations:**
- FastVideo uses **plain NCCL `all_to_all_single`** (not symmetric memory) under the hood. So it's the "vanilla Ulysses" path, not the async/symm-mem variants from fal/VeOmni. There's no `async_op=True`, no functional collectives, no SymmMem.
- `scatter_dim=2, gather_dim=1` is the *pre-attention* a2a; `scatter_dim=1, gather_dim=2` is the *post-attention* a2a.
- Padding via `compute_padding_for_sp` in `fastvideo/distributed/utils.py` — pad to `seq_len = next_multiple_of(seq_len, sp_world_size)`, then unpad after the gather.
- `warmup_sequence_parallel_communication` (in `communication_op.py`) runs dummy a2a's to pre-trigger NCCL's lazy communicator initialization, avoiding a slow first iteration.
- FastVideo includes both forward and backward through `DistributedAutograd.AllToAll4D` — the SP layer is differentiable via the symmetric `gather_dim ↔ scatter_dim` swap.

**Args:** `fastvideo_args.py` exposes `tp_size`, `sp_size`, `hsdp_replicate_dim`, `hsdp_shard_dim` directly. From the actual code:

```python
@dataclass
class FastVideoArgs:
    num_gpus: int = 1
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    ...

# In __post_init__:
assert self.sp_size <= self.num_gpus and self.num_gpus % self.sp_size == 0
assert self.hsdp_replicate_dim <= self.num_gpus and self.num_gpus % self.hsdp_replicate_dim == 0
assert self.hsdp_shard_dim <= self.num_gpus and self.num_gpus % self.hsdp_shard_dim == 0
```

There is **no separate `--ulysses-degree`/`--ring-degree` knob** — FastVideo's SP is *Ulysses-only* (via 4D-A2A). For users wanting Ring, the path is via SGLang Diffusion or xDiT.

**Reported wins.** **2.5× on 8×H100 for Wan 2.2 I2V** ([Morphic blog](https://morphic.com/blog/boosting-wan2-2-i2v-56-faster)); 3× single-GPU w/ SageAttention2 + TeaCache. **LTX-2.3 5s 1080p in 4.55 s on 1×B200** with NVFP4 + VSA + 2-stage distill ([FastVideo blog](https://haoailab.com/blogs/fastvideo_realtime_1080p/)).

### P.9.5 VideoSys — DSP for Open-Sora and Latte

[NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys) is the canonical DSP integration. Headline numbers: **DSP gets 3× train / 2× infer for Open-Sora** versus single-axis SP (Ulysses or Ring). Supports Open-Sora, Vchitect, CogVideoX (single-axis SP only, since CogVideoX uses full 3-D attention), Latte. Not relevant for Wan/HunyuanVideo/LTX-2 (no axis factorization).

### P.9.6 vLLM-Omni Sequence Parallel + UAA mode

[docs.vllm.ai/projects/vllm-omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/sequence_parallel/), [PR #1412](https://github.com/vllm-project/vllm-omni/pull/1412).

**Two SP modes.**

1. **Standard SP.** Same Ulysses+Ring (USP) as everywhere else. Configured via `_sp_plan` (per-module input/output sharding declaration) or intrusively (sp_shard / sp_gather inside the module).

2. **UAA mode (Ulysses Anything Attention, experimental).** Sets `ulysses_mode="advanced_uaa"`. Lets Ulysses handle (a) sequence lengths that are not evenly divisible by `ulysses_degree`, and (b) head counts that are not divisible by `ulysses_degree` (lifts the SP ≤ heads cap).

**API design (`_sp_plan`)** — the cleanest declarative API across the surveyed frameworks:

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput, SequenceParallelOutput,
)

class MyDiTBlock(nn.Module):
    _sp_plan = {
        "input.hidden_states": SequenceParallelInput(dim=1),
        "output": SequenceParallelOutput(dim=1),
    }
```

Each module declares *what* should be SP-sharded, and the framework wires up the comm primitives at runtime. The cleaner alternative to manual `sp_shard()` calls inside the forward.

**Reported numbers.** FP8 Wan 2.2 with SP=8: **−41% memory, +7% T2V / +12% I2V speed** vs. baseline ([PR #1412](https://github.com/vllm-project/vllm-omni/pull/1412)). T2V on 1×B200, 1280×720, 81f, 40 steps: BF16 baseline 892.8 s → FP8 828.0 s → FP8 + ignored_layers 826.0 s. Memory: 64.46 → 38.18 GB. I2V on 1×B200, 81f, 50 steps: BF16 301.1 s → FP8 264.6 s; ~26 GiB (~41%) memory reduction.

### P.9.7 SGLang Diffusion (the most complete USP × TP × cache stack)

**Refs:** [docs.sglang.ai/diffusion](https://docs.sglang.ai/diffusion/), [PR #16532](https://github.com/sgl-project/sglang/pull/16532), [PR #19419](https://github.com/sgl-project/sglang/pull/19419), [ring_sp_performance.html](https://docs.sglang.ai/diffusion/performance/ring_sp_performance.html).

**The `(--ulysses-degree, --ring-degree, --tp-size)` matrix.** SGLang exposes the full USP × TP grid:

```bash
sglang serve --model-type diffusion \
  --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 2 --sp-degree 2 --ulysses-degree 1 --ring-degree 2 --port 8898
```

For 8-GPU Wan 2.2-I2V-A14B with TP=2, SP=4 (ulysses=2, ring=2):

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --num-gpus 8 --tp-size 2 --ulysses-degree 2 --ring-degree 2 \
  --720p --num-frames 81 --num-inference-steps 27 \
  --enable-torch-compile
```

PR #16532 (January 2026) fixed the rank-group construction so that under TP > 1, SP groups are constructed as the *actual* SP rank groups (from `rank_generator.get_ranks("sp")`), not assuming contiguous global ranks. Before PR #16532, with TP > 1 the SP groups became *strided* in global ranks (e.g. `[0, 2, 4, 6]` instead of `[0, 1, 2, 3]`) and constructing Ulysses/Ring groups using contiguous global ranks caused **cross-group communication** mixing ranks from different SP/TP partitions, leading to incorrect outputs. The bug only appeared under TP > 1, since with TP=1 contiguous-rank-construction happened to match the real SP layout.

**Published `u1r2` benchmark on Wan 2.2 TI2V-5B** (2-GPU, 48 GB RTX40-series-class):

| Stage | u1r2 (s) | u1r1 baseline (s) | Speedup |
|---|---|---|---|
| InputValidation | 0.106 | 0.103 | 0.97× |
| TextEncoding | 1.397 | 2.226 | **1.59×** |
| LatentPreparation | 0.0002 | 0.0002 | 1.00× |
| TimestepPreparation | 0.0003 | 0.0004 | 1.33× |
| Denoising | 52.636 | 71.679 | **1.36×** |
| Decoding | 7.671 | 13.431 | **1.75×** |
| **Total** | **63.74** | **90.63** | **1.42×** |

Memory drops from 27.40 GB → 20.07 GB (**−7.33 GB / −26.7%**). VAE Decoding gets **1.75×**, *better* than the DiT denoising stage's 1.36×.

**The cross-attn USP-skip optimization (PR #19419, March 2026).** When KV is *replicated* (text encoder, image encoder, ControlNet), the local attention from `local_q` against the *full local* KV is exactly what the USP-with-comm path would have computed. Adds a `skip_sp: bool = False` parameter to `USPAttention.forward()`, set to `True` for `WanT2VCrossAttention` and `WanI2VCrossAttention`. Per-step speedup on B200:

| Config | baseline (s/step) | skip-sp (s/step) | Speedup |
|---|---|---|---|
| `ring=4` | 4.72 | 3.99 | 1.18× |
| `ring=2 ulysses=2` | 4.57 | 3.94 | 1.16× |
| `ulysses=4` | 4.01 | 3.68 | 1.09× |

The reasoning is: when KV has `S_kv = 512` (text) or `S_kv = 257` (image) tokens that are identical across all SP ranks, Ulysses' a2a inflates the sequence by `ulysses_world_size×` (e.g., 512 → 2048 on 4 GPUs), producing *duplicated data* and wasting comm + FLOPs on attention against the duplicates. Ring rotates identical KV blocks across ranks — gaining nothing. Both can be skipped, computing local attention directly with mathematically identical results.

There was a brief debate on the PR (with PR #19340 author @triple-mu) about correctness when `ulysses > 1`: under Ulysses, Q is sharded along seq dim before the cross-attention call, so each rank's local Q has only `seq_q/uly` tokens. After offline discussion the PR authors confirmed: when KV is *replicated* on each rank, attention only depends on local Q × full KV, so skipping is correct even with `ulysses > 1`. The video accuracy verification in the PR confirms.

**SGLang compatibility matrix** — the full per-model status across major attention/caching columns ([docs](https://docs.sglang.io/docs/sglang-diffusion/compatibility_matrix)):

| Model | Tea | Tile | Sage | VSA | SLA | SageSLA | SVG2 |
|---|---|---|---|---|---|---|---|
| FastWan2.1 T2V 1.3B | ⭕ | ⭕ | ⭕ | ✅ | ❌ | ❌ | ❌ |
| FastWan2.2 TI2V 5B FullAttn | ⭕ | ⭕ | ⭕ | ✅ | ❌ | ❌ | ❌ |
| Wan2.2 TI2V 5B | ⭕ | ⭕ | ✅ | ⭕ | ❌ | ❌ | ❌ |
| Wan2.2 T2V A14B (720p) | ❌ | ❌ | ✅ | ⭕ | ❌ | ❌ | ❌ |
| Wan2.2 I2V A14B (720p) | ❌ | ❌ | ✅ | ⭕ | ❌ | ❌ | ❌ |
| HunyuanVideo (544×960) | ❌ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| Wan2.1 T2V 14B (480p, 720p) | ✅ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| TurboWan2.1 T2V 14B (720P) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ⭕ |
| TurboWan2.2 I2V A14B | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ⭕ |
| **LTX-2 / LTX-2.3** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

LTX-2/LTX-2.3 has all seven optimization columns marked ❌ — both support T2V/TI2V on one-stage and two-stage pipelines but no acceleration backend is plumbed in yet. There is, however, a useful **LTX-2 two-stage device-mode** knob (`--ltx2-two-stage-device-mode`):

- `original` (no premerged stage-2 transformer) — 154.67 s.
- `snapshot` (default, recommended) — 114.05 s.
- `resident` — 75.71 s, uses much more VRAM.

So even without sparse/cache optimization, the resident-snapshot trade alone is a ~2× win on LTX-2 two-stage in SGLang.

### P.9.8 Wan-Video / Wan2.2 official

**Repo:** [github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2). Verified verbatim CLIs:

```bash
# T2V-A14B 720p
torchrun --nproc_per_node=8 generate.py --task t2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --prompt "..."

# I2V-A14B 720p
torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG \
  --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "..."

# TI2V-5B
torchrun --nproc_per_node=8 generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --dit_fsdp --t5_fsdp --ulysses_size 8 --image ... --prompt "..."

# S2V-14B (audio-driven)
torchrun --nproc_per_node=8 generate.py --task s2v-14B --size 1024*704 \
  --ckpt_dir ./Wan2.2-S2V-14B/ \
  --dit_fsdp --t5_fsdp --ulysses_size 8 --image ... --audio ...
```

Parallel modes: PyTorch FSDP (for both DiT and T5) plus DeepSpeed Ulysses ([arXiv:2309.14509](https://arxiv.org/abs/2309.14509)). The README's efficiency-table footnote: *"The distributed testing utilizes the built-in FSDP and Ulysses implementations, with FlashAttention3 deployed on Hopper architecture GPUs."*

**Single-GPU memory-saving flags:** `--offload_model True --convert_model_dtype --t5_cpu` — with all three, TI2V-5B fits on a 24 GB RTX 4090.

### P.9.9 diffusers native CP (the simplest end-user API)

[PR #11941](https://github.com/huggingface/diffusers/pull/11941), merged Sept 2025. The user-facing API:

```python
from diffusers import FluxPipeline
from diffusers import ParallelConfig, enable_parallelism

torch.distributed.init_process_group("nccl")
rank = torch.distributed.get_rank()
device = torch.device("cuda", rank % torch.cuda.device_count())
torch.cuda.set_device(device)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to(device)
pipe.transformer.parallelize(config=ParallelConfig(ulysses_degree=2, ring_degree=2))
pipe.transformer.set_attention_backend("flash")  # or "_native_cudnn" or "sage"

generator = torch.Generator().manual_seed(42)
with enable_parallelism(pipe):
    image = pipe(prompt, num_inference_steps=28, guidance_scale=4.0, generator=generator).images[0]
```

Backed by USP (`feifeibear/long-context-attention` underneath). Supports Flux and Wan T2V as of merge. Three attention backends supported: cuDNN, FA2, SageAttention.

The Wan example on the PR uses `ParallelConfig(ulysses_degree=2)` and the `flash` backend. Performance numbers haven't been published per-model for diffusers native CP, but the underlying USP performance carries over.

### P.9.10 LightX2V cross-framework H100 table

Verified at [github.com/ModelTC/lightx2v](https://github.com/ModelTC/lightx2v): cross-framework comparison on H100, Wan 2.1 I2V-14B-480P, 40 steps, 81 frames:

| Framework | GPUs | Step time | Speedup |
|---|---|---|---|
| Diffusers | 1 | 9.77 s/it | 1× |
| xDiT | 1 | 8.93 s/it | 1.1× |
| FastVideo | 1 | 7.35 s/it | 1.3× |
| SGL-Diffusion | 1 | 6.13 s/it | 1.6× |
| **LightX2V** | 1 | **5.18 s/it** | **1.9×** |
| FastVideo | 8 | 2.94 s/it | 1× |
| xDiT | 8 | 2.70 s/it | 1.1× |
| SGL-Diffusion | 8 | 1.19 s/it | 2.5× |
| **LightX2V** | 8 | **0.75 s/it** | **3.9×** |

LightX2V additionally lists **8×H100 + no-CFG + FP8 = 0.35s/step (2.1× over 8-GPU baseline)** and **8×4090D + no-CFG + FP8 = 2.35 s/step (2.0×)**. Quantization formats: `w8a8-int8`, `w8a8-fp8`, `w4a4-nvfp4`. Attention backends: SageAttention 2, FlashAttention, Radial Attention, q8_kernels, sgl-kernel, vLLM kernels, FlashInfer, MagiAttention. Mooncake disagg deployment added March 5, 2026; technical blog April 10, 2026.

### P.9.11 Choosing a stack — decision tree

```
  Need TP for parameter sharding (Step-Video 30B)?
  ├─ yes → xDiT or SGLang Diffusion (both expose --tp + --ulysses + --ring)
  │
  └─ no → SP-only options:
      ├─ Wan 2.x official only → Wan-Video/Wan2.2 (--ulysses_size 8 + --dit_fsdp)
      ├─ HunyuanVideo / multi-model → xDiT or ParaAttention
      ├─ Maximum throughput, ML-Sys research → LightX2V (3.9× on 8×H100)
      ├─ Production serving + cache integration → SGLang Diffusion
      └─ Hugging Face ergonomics → diffusers native CP (PR #11941)
```

For B200 specifically: SGLang Diffusion has the best B200 numbers (cross-attn skip-sp PR #19419 was measured on B200), Wan official is the reference recipe, xDiT handles the most models, FastVideo is the best for Wan/LTX-2 with VSA + DMD distillation.

---

## P.10 Pipeline parallelism for long video (DualParal, PipeDiT, VINs)

### P.10.1 Why pipeline parallelism

Sequence parallelism scales latency at fixed model size; tensor parallelism scales memory at fixed sequence length. **Pipeline parallelism (PP) scales both — at the cost of denoising-loop reformulation**. For minute-long video generation (>1000 frames), single-axis SP/TP either OOMs (full activation can't fit on any single GPU) or loses to PP because the per-stage compute is finally large enough to amortize stage-transition cost.

The natural reformulation is **block-wise denoising**: instead of synchronizing the noise level across all frames, divide the video into non-overlapping temporal blocks and let each block denoise asynchronously through a layer-pipelined model. This is the architectural insight that DualParal, PipeDiT, and VINs all exploit, with three different mechanisms.

### P.10.2 DualParal (AAAI 2026)

**Reference:** Wang et al., [arXiv:2505.21070](https://arxiv.org/html/2505.21070v1); code [dualparal-project/dualparal](https://github.com/dualparal-project/dualparal).

**Mechanism — the FIFO queue + device pipeline.** DualParal divides BOTH the video sequence and the model into chunks. The video is partitioned into temporal blocks `B_1, B_2, …, B_T` arranged in a FIFO queue with noise level decreasing from tail (just appended, full noise) to head (about to exit, clean). At each diffusion step, blocks flow through the device pipeline in *reverse* order (tail to head):
1. Each device handles a specific video block and a specific model chunk simultaneously.
2. Asynchronous P2P passes intermediate features between devices.
3. After each diffusion step, all blocks shift forward; tail gets a new noisy block; head is dequeued as the clean block ready for VAE decoding.

The async noise levels (different per block) are what resolves the apparent conflict between sequence-parallel splitting and pipeline-parallel layer-stack flow. At any instant, every device is busy doing meaningful work on a block at its current noise level.

**Feature cache.** Each device caches the KV features from the previous block's *self-attention*, so the next block doesn't need to re-concatenate the previous block's frames — only the subsequent block needs concatenation (`[B_{i-1}, B_i]`, not `[B_{i-1}, B_i, B_{i+1}]`). This halves communication and cuts redundant compute on Cross-Attention and FFN (which don't benefit from inter-frame information anyway).

**Coordinated noise initialization.** New blocks are initialized from a *complete noise pool* shuffled to avoid repeating noise in the last `Num_C/2` latents of the queue's final block. This avoids the performance degradation of repeated noise during whole-process denoising while preserving global consistency advantages of the complete noise space.

**Reported numbers.** **6.54× lower latency / 1.48× lower memory** on 8× RTX 4090 for 1025-frame video, with public Wan 2.1 / Wan 2.2 integration. The win scales with sequence length: at 1025 frames each GPU holds layers + a frame block, vs vanilla single-GPU which can't fit at all without offload.

| Method | Communication cost | Comm/Comp overlap | Model memory | KV activation memory |
|---|---|---|---|---|
| Ring Attention | `2·O(p·h)·L` | ✓ | `W` | `F·(1/N)·KV` |
| DeepSpeed-Ulysses | `(4/N)·O(p·h)·L` | ✗ | `W` | `F·(1/N)·KV` |
| Video-Infinity | `2·O(Num_C·H'·W'·h)·L` | ✗ | `W` | `(F·1/N + Num_C)·KV` |
| FIFO | `2·O((Num_B+Num_C)·H·W·C)` | ✓ | `W` | `(Num_B+Num_C)·KV` |
| **DualParal** | `2·O((Num_B+Num_C/2)·H'·W'·h)` | ✓ | `(1/N)·W` | `(Num_B+Num_C)·KV` |

**Bubble ratio.** With reverse-order denoising, DualParal's bubble ratio is `(N²−N−1) / (N²−N−1+T·Block_num)`. As `Block_num → ∞`, bubble → 0%. For long videos this approaches zero idle time.

**Composability with SP/TP.** DualParal's partitioning is orthogonal to SP/TP — you can stack DualParal-PP (across layers + frame blocks) on top of SP=2 or TP=4 within each pipeline stage. No published combined recipe yet, but the math indicates `(DualParal × SP=2 × TP=4)` is the right composition for >1k-frame Wan generation on a single B200 node.

### P.10.3 PipeDiT — three-component PP for video DiT

**Reference:** Wang/Wang/Shi, [arXiv:2511.12056](https://arxiv.org/html/2511.12056v1).

**Three components:**

1. **PipeSP — pipelined sequence parallelism.** Instead of running Ulysses' single-block all-to-all serially after computing all attention heads, PipeSP partitions the attention-head dimension and **issues an All-to-All immediately after each head is processed**, overlapping comm with the next head's compute. This is similar to the fal/VeOmni async-Ulysses idea but at finer granularity (per-head instead of per-Q/K/V branch). Algorithmically:

```python
for j in range(h):
    result = attention(Q[:, j, :, :], K[:, j, :, :], V[:, j, :, :], mask[:, j, :, :])
    event = record_event()
    com_stream.wait_event(event)
    hidden = AllToAll(result, group=sp_group)
    chunks.append(hidden)
hidden_states = concat(chunks, dim=1).view(-1, h, n, D).permute(0, 2, 1, 3).view(-1, h*n, D)
```

2. **DeDiVAE — decoupled DiT and VAE.** DiT denoising is compute-heavy, VAE decode is memory-heavy. Co-locating them serializes the two and creates a memory peak bigger than necessary. DeDiVAE puts the DiT on `N_denoise` GPUs and the VAE on `N − N_denoise` GPUs, with a bridge buffer so the VAE doesn't sit idle waiting for DiT denoising to finish. Multiple prompts can pipeline through: while VAE decodes prompt 1's clean latent, DiT denoises prompt 2.

   The optimal split is `N_decode ≈ ⌈T_decode / (T_decode + T_denoise) · N⌉` (balancing wall-clock).

3. **Attention co-processing (Aco).** When the denoising stage is much slower than VAE decode (the common case), the Decode GPUs idle. Aco splits each DiT block's *linear projections* (run on Denoise GPUs) and *attention kernel* (sent to Decode GPUs via P2P), parallelizing the attention work across all GPUs:

   - Denoise GPUs compute Q, K, V projections.
   - Send Q, K, V via P2P to Decode GPUs (when they're not decoding).
   - Both groups execute attention in parallel.
   - Aggregate via intra-group A2A + inter-group P2P.

**Reported numbers.** **1.06–4.02× on 8 GPUs** integrated with OpenSora-Plan and HunyuanVideo. The high end of the range comes from VAE-DiT separation; the low end from PipeSP alone. Benchmarks on 8× RTX A6000 and 8× L40 (both 48 GB):

| Resolution | OpenSoraPlan, A6000, 50 steps | HunyuanVideo, A6000, 50 steps |
|---|---|---|
| 480×352×97 | base 622 s → opt 502 s (1.24×) | base 965 s → opt 726 s (1.33×) |
| 800×592×97 | base 1994 s → opt 1766 s (1.13×) | base 2686 s → opt 2470 s (1.09×) |
| 1024×576×97 | base 2162 s → opt 1832 s (1.18×) | base 3726 s → opt 3453 s (1.08×) |

PipeSP alone (Table 2) gives 1.04–1.08× per-timestep speedup on OpenSoraPlan (some configs slightly slower because the head-by-head a2a granularity is too small for very short sequences).

### P.10.4 Video Interface Networks (VINs)

**Reference:** Dedhia et al., [arXiv:2503.17539](https://arxiv.org/abs/2503.17539).

**Mechanism — System 1 / System 2 abstraction.** VINs add a small abstraction module between video chunks. At each diffusion timestep, the VIN encodes global semantics from the noisy input of *local chunks*; the encoded representations then guide DiTs in denoising chunks **in parallel**. The architectural inspiration is dual-process cognition — fast System-1 (VIN, encodes global semantics) and slow System-2 (DiT, denoises fine-grained patches).

The VIN architecture:
- **Global tokens** `Z_init ∈ ℝ^{N_global × d}` (fixed-sized embeddings, independent of input).
- **VIN Encoder** sub-samples the input video at fixed time intervals `T_s`, then cross-attends from `Z_init` (queries) to the sub-sampled video (keys/values): `Z_t = MHA(Q=Z_init, K,V=X_t^{1:N,T_s})`.
- **VIN Processor** iteratively refines via self-attention with prompt embedding: `Z_t = MHA(Q,K,V=[Z_t, T_emb])`.

End-to-end training couples VIN and DiT via a factorized noise distribution:

\[ P_θ(\epsilon_t | X_t, t, Z_t) = \prod_{i=0}^{\lfloor N/N_s \rfloor} P_θ(\epsilon_t^{i} | X_t^{i}, t, Z_t) \]

So at training time, the DiT denoises chunks parallelly conditioned on VIN's global tokens; at inference, the same.

**Reported numbers.** **25–40% fewer FLOPs than full generation** at comparable VBench. State-of-the-art Motion Aware Warp Error (MAWE) scores. Lightweight architectural change; not yet integrated into Wan/HunyuanVideo/LTX official paths — VINs are an *architectural* contribution, not an inference recipe drop-in.

VINs preserve background consistency and subject coherence across long videos better than autoregressive chunked methods (which lose global semantics across chunks) and full-attention generation (which suffers from motion stagnation when extended past training context). On VBench, VINs surpass FreeNoise, FreeLong, and VideoDrafter.

### P.10.5 Latent Parallelism (LP) — December 2025

**Reference:** Wu et al., [arXiv:2512.07350](https://arxiv.org/html/2512.07350v1).

**Mechanism — rotating partition dimensions.** LP is the *first parallelism strategy tailored for VDM serving*. The core idea: instead of partitioning the computational graph (NMP/TP/PP/SP), apply divide-and-conquer to the **latent tensor** of individual requests. The 3-D latent space (T, H, W) is partitioned into sub-regions that can be denoised independently across GPUs, before being reconstructed.

LP rotates the partitioning dimension at each timestep: 
\[ d_i = \mathcal{M}[(i-1) \bmod 3 + 1] \]
where `M` maps indices 1, 2, 3 to temporal, height, width. Each rotation cycle (3 timesteps) ensures each partitioned region accesses the full spatio-temporal context of the entire video, maintaining global consistency.

**Patch-aligned overlapping partitioning.** To preserve generation quality, LP matches partition boundaries with the internal visual patches of VDMs and incorporates overlapping regions to prevent boundary discontinuities. The hyperparameter `r ∈ [0, K-1]` controls overlap ratio.

**Position-aware reconstruction.** Adaptive linear weights in overlap regions: `W_j = j/Δ_start` in front overlap, `1` in core, `(ℓ-j)/Δ_end` in rear overlap. Predictions from partitions farther from core regions contribute less.

**Reported numbers.** **Up to 97% communication reduction vs SP** on multi-GPU video diffusion serving. As a non-intrusive plug-in paradigm, LP can compose with conventional DP/NMP/TP/PP.

LP is the December 2025 incarnation of "shard the latent, not the model", and it's a real differentiator: while SP shards activations (which are massive at long sequences), LP only exchanges the compact latent representations, leveraging the locality of diffusion denoising. The 97% comm reduction is the headline number; quality preservation is verified on EvalCrafter, T2V-CompBench, and VBench.

### P.10.6 PP composability matrix

Crystallizing how PP variants compose with the SP/TP machinery:

| Method | Cuts seq | Cuts model | Cuts memory | SP-compatible | TP-compatible |
|---|---|---|---|---|---|
| Ulysses | ✓ | ✗ | activations | (self) | ✓ |
| Ring | ✓ | ✗ | activations | (self) | ✓ |
| USP | ✓ | ✗ | activations | (self) | ✓ |
| DSP | ✓ (multi-axis) | ✗ | activations | ⚠ | ✓ |
| DualParal | partial | ✓ | params + activations | ✓ | ✓ |
| PipeDiT | ✓ | ✓ | activations + bridge | ✓ | ✓ |
| VINs | (architectural) | (architectural) | n/a | ✓ | ✓ |
| Latent Parallelism | (latent) | (latent) | latent only | ✓ | ✓ |

For 8-GPU B200 + 1k-frame Wan generation, the right composition is roughly: `LP × DualParal × USP(SP=2)`, but no published recipe combines all three.

---

## P.11 Multi-request serving / continuous batching

Every benchmark in this report measures single-request wall-clock. For B200 deployments serving multiple users, the picture changes fundamentally — and the relevant work is recent.

### P.11.1 DiT-Serve (ICLR 2026 submission)

**Reference:** Luo, Hao, Yan, Cao, Nguyen, [OpenReview NGNRc7rZBg](https://openreview.net/forum?id=NGNRc7rZBg).

**Mechanism.** Two innovations:

1. **Step-level batching.** The scheduler preempts and swaps requests every denoising step — analogous to LLM continuous batching but at the *diffusion-step* granularity. Heterogeneous step counts (mixing 4-step distilled with 50-step base) no longer force shorter requests to wait for the longest-running request to finish; instead each denoising step batches whatever requests are at that step and swaps in/out as steps complete.

2. **Brick Attention.** A new attention algorithm that *binpacks* requests of different context lengths onto a set of GPUs. Standard cross-request batching pads heterogeneous requests to a common resolution and duration, wasting compute and memory on padding. Brick Attention binpacks instead, eliminating the padding overhead.

**Reported numbers.** **2–3× higher throughput, 3–4× lower latency** vs prior systems. Particularly strong when requests have heterogeneous step counts — e.g., a serving cluster mixing premium (50-step base Wan), standard (8-step distilled), and economy (4-step Lightning) tiers.

Two named *fundamental inefficiencies* that DiT-Serve targets:
- **Spatial underutilization**: GPUs waste compute and memory by padding heterogeneous requests to a common resolution and duration.
- **Temporal underutilization**: batching jobs with varying denoising step counts forces GPU cores to idle as shorter requests wait for the longest-running request.

DiT-Serve is the standout multi-request video DiT system from the early 2026 conference cycle.

### P.11.2 Chorus (April 2026)

**Reference:** [arXiv:2604.04451](https://arxiv.org/abs/2604.04451v1).

**Mechanism.** Three-stage *inter-request caching* that exploits **prompt similarity**: requests with overlapping prompt embeddings can share intermediate computations across requests. Particularly effective when the same prompt is sampled multiple times (variants/seeds) and when prompt prefixes are shared across requests.

**Reported numbers.** **Up to 45% speedup on industrial 4-step distilled models**. The 45% upper bound assumes high prompt similarity (e.g., A/B testing variants of the same prompt); typical mixed-prompt clusters see 10–25%.

### P.11.3 vLLM-Omni stream-batch RFC (#2280, March 2026)

**Reference:** [vllm-project/vllm-omni#2280](https://github.com/vllm-project/vllm-omni/issues/2280).

**Motivation.** vLLM-Omni's diffusion parallelism relies on Sequence Parallelism (Ulysses-SP, Ring Attention). This works for offline generation with large chunks (~33,000+ tokens), where comm overhead is amortized over heavy compute.

**Live-streaming operates in a fundamentally different regime.** A causal DiT generating 4 frames at 480p produces only ~1,536 tokens per forward pass — far below the H100 roofline ridge point (590.75 FLOP/byte), placing the workload deeply **memory-bound**. SP's all-to-all adds 40–120 ms per forward pass on NVLink H100, 20–40× higher than block-level PP's inter-stage P2P cost (~2–6 ms). The StreamDiffusionV2 paper (arXiv:2511.07399, MLSys'26) measures this directly: **SP yields essentially no speedup at 1,536 tokens, while block-level PP achieves near-linear FPS scaling**.

**Mechanism — pipeline parallelism + stream batch.** Two orthogonal parallelism dimensions:

1. **Pipeline Parallelism (Model Depth).** Partition DiT Transformer blocks along depth across GPUs. Comm pattern (from StreamDiffusionV2 source): sending uses `dist.isend` (non-blocking) on a dedicated `com_stream`; receiving uses `dist.recv` (blocking) on `com_stream`, synchronized to the compute stream via `current_stream.wait_stream(com_stream)`. When the pipeline is fully loaded, the blocking `recv` for step K+1 is pre-satisfied by the `isend` from step K — natural overlap.

2. **Stream Batch (Denoising Steps as Batch Multiplier).** With `n` denoising steps and `n` GPUs, the pipeline in steady state has `n` latents at different noise levels simultaneously in-flight:

   ```
   micro-step →    t=1           t=2           t=3           t=4
   GPU 0:       [noise_lvl 4]  [noise_lvl 3]  [noise_lvl 2]  [noise_lvl 1]
   GPU 1:                      [noise_lvl 4]  [noise_lvl 3]  [noise_lvl 2]
   GPU 2:                                     [noise_lvl 4]  [noise_lvl 3]
   GPU 3:                                                    [noise_lvl 4] → clean frame
   ```

   Every micro-step produces one clean output frame. **More steps → higher FPS** — the deeper pipeline keeps all GPUs busy. Counter-intuitive but matches StreamDiffusionV2 empirics: **42 FPS at 1 step (Wan 2.1-1.3B) vs 58.28 FPS at 4 steps (Wan 2.1-14B), both 480p on 4×H100 NVLink**.

The SLO-aware scheduler dynamically adjusts batch size `B` and step count `n` to maximize utilization while satisfying per-frame deadlines, using StreamDiffusionV2's latency model `L(T,B) ≈ (A(T,B) + P_model) / (η × BW_HBM)` giving `f ∝ B/(1+B)`.

The RFC also disambiguates from #647 (PipeFusion, spatial patches) and #268 (Async Stages, cross-model handoff). Both are orthogonal: #647 reuses stale KV across timesteps for offline high-res; #268 manages the stage graph; this RFC handles intra-DiT pipeline depth.

### P.11.4 vLLM-Omni Batched Diffusion Serving (#1511)

[#1511](https://github.com/vllm-project/vllm-omni/issues/1511): multi-request same-step batching, gated on identical sampling parameters (same step count, same scheduler, same CFG scale). Practical engineering rather than novel research, but the right primitive for production video serving where most requests are 4-step CFG=1 distilled. No quantitative numbers published as of May 2026.

### P.11.5 Comparison

| Mechanism | Primitive | Best for | Speedup |
|---|---|---|---|
| DiT-Serve | step-level batching + Brick Attention | heterogeneous step counts + heterogeneous resolutions | 2–3× tput / 3–4× latency |
| Chorus | inter-request prompt-embedding cache | prompt-variant sampling | up to 1.45× |
| vLLM-Omni stream batch | block-level PP + per-step batch | live streaming, sub-1.5k token forwards | near-linear FPS scaling |
| vLLM-Omni same-step batch | same-step request packing | uniform request profile | small (2–10%) |

For B200 deployments: DiT-Serve's mechanisms are the most general; Chorus is a targeted addition for prompt-similar workloads; vLLM-Omni stream batch is the right answer for autoregressive distilled video (where SP's all-to-all latency on small payloads dominates). All three compose with the per-request SP×TP machinery from P.9.

---

### P.11.6 CoCoDiff — collective-communication optimization for Ulysses (April 2026)

#### P.11.6.1 Why this is a separate subsection

CoCoDiff (Ma et al., [arXiv:2604.14561](https://arxiv.org/abs/2604.14561)) targets a different layer than everything else in this section: it optimizes the *collective communication primitives* underneath Ulysses, exploiting two structural properties of DiTs that NCCL/MPI/oneCCL don't know about. The paper's experiments are on **Aurora supercomputer (Intel Ponte Vecchio GPUs, NOT NVIDIA)** — 12× bandwidth asymmetry between intra-GPU tile links (185 GB/s) and inter-GPU Xe Links (15 GB/s). Two of the three CoCoDiff mechanisms generalize directly to NVIDIA HGX B200; one does not.

#### P.11.6.2 The two structural properties

**Property 1 — QKV Processing Asymmetry.** Modern DiTs (SD3, FLUX, Qwen-Image, Wan 2.x, HunyuanVideo) apply RMSNorm/LayerNorm and RoPE *only to Q and K*, not V. RoPE is positional rotation; applying it to V would alter the content being aggregated without contributing positional signal. So V's projection finishes earlier than Q/K's, creating a time gap of `T(Q,K) − T(V_p1)` that can be filled by V's communication. This is the same observation underlying VeOmni's async-Ulysses, but CoCoDiff exploits it at the *first phase* of a topology-aware decomposed all-to-all.

**Property 2 — Temporal Redundancy across denoising steps.** Adjacent denoising steps produce highly similar Q/K/V tensors. This temporal redundancy has been used for compute reduction (DeepCache, Δ-DiT, Learning-to-Cache, AttMEMO, TeaCache) but not for communication reduction. CoCoDiff communicates only the projections that have changed materially since the last step.

#### P.11.6.3 Three mechanisms

1. **TAPA (Tile-Aware Parallel All-to-All).** Decomposes each Ulysses all-to-all into two phases:
   - **Phase 1:** intra-GPU tile-to-tile exchange in parallel (185 GB/s on Aurora's internal links). All 6 GPU pairs of tiles exchange concurrently.
   - **Phase 2:** inter-GPU all-to-all as **two parallel 6-rank rings**, one per Xe Link ring (no cross-ring traffic), maximizing aggregated bandwidth.
   
   The two-phase decomposition is **topology-aligned** to Aurora's two-level hierarchy (185 GB/s intra-GPU vs 15 GB/s inter-GPU). On NVIDIA HGX B200 this property *does not transfer* — NVLink5 between any two GPUs in a node is uniformly 1.8 TB/s, no asymmetry to exploit. Phase 1 collapses to a no-op.

2. **V-First Scheduling.** Exploits Property 1 — V's projection takes 0.1 ms, Q/K each take 0.5 ms (RMSNorm + RoPE add 0.4 ms). V's Phase 1 communication launches immediately after V's projection, overlapping with Q's normalization+RoPE plus K's normalization+RoPE. Formal condition for full overlap: `T(Q,K) > T(V_p1)`. Holds for FLUX, SD3.5, Qwen-Image at all evaluated batch sizes. Risk: at very large batch sizes, V's projection time scales linearly and may exceed `T(Q,K)`, leaving a remainder exposed to the critical path.

   **This generalizes directly to NVIDIA Ulysses-SP setups.** It's a natural extension of VeOmni's async-Ulysses with the explicit observation that V is the *fastest* of the three to project, so V-First (rather than the symmetric Q/K/V parallel async) is the optimal ordering.

3. **V-Major Selective Communication.** Exploits Property 2:
   - **Active projection selection.** After Phase 1 of V, each GPU compares its V projections against the cached V from previous step using L1-norm distance. The smallest-distance fraction `r` (the "cache ratio") is *not communicated* — those tokens reuse cached values.
   - **Phase 1 (intra-GPU):** V's Phase 1 must be done at full size (for the L1-norm comparison). For Q/K, only the active `(1−r)` fraction is exchanged.
   - **Phase 2 (inter-GPU):** All three projections undergo inter-GPU all-to-all, but only the active `(1−r)` subset is communicated. This is where the largest savings occur — the message size drops by a factor of `1/r`.
   - **Post-Phase 2 reconstruction:** Each GPU caches active projections; for inactive ones, reuses the previous step's value (indexed scatter from cache).

   **Time-varying cache scheduling.** Three complementary mechanisms:
   - **Warmup**: First few denoising steps run full TAPA to populate the cache.
   - **Time-varying cache ratio**: `r` follows a linear schedule from 0 → 1 across timesteps (early steps need large global updates, later steps need only localized refinement).
   - **Periodic full sync**: At a configurable interval (default every 10 steps), force a full-size update to flush stale cache entries.
   
   **Cross-frame applicability:** the temporal-redundancy idea generalizes to NVIDIA video DiTs, but the cache size grows linearly with batch and shrinks linearly with rank count. For Qwen-Image at 1536×1536 with B=4 on 12 tiles, the per-tile cache is 4,320 MB (~6.6% of the 64 GB tile capacity); scaling to 8 nodes with B=32 keeps per-tile overhead constant.

#### P.11.6.4 Reported results

Average **3.6× over flat all-to-all, peaking 8.4×**, evaluated on **SD3.5, Qwen-Image, FLUX.1-dev, FLUX.2-dev, 1–8 nodes, up to 96 Intel GPU tiles** on Aurora. PSNR / SSIM / MSE preserved within 2 dB / 0.03 / 2e-3 of the flat baseline.

The 3.6×/8.4× numbers are **specific to Aurora's 12× bandwidth asymmetry**. On NVIDIA HGX B200 with uniform 1.8 TB/s NVLink5, TAPA collapses to a no-op (no asymmetry to exploit) and the 3.6× headline does not transfer. **V-First scheduling and V-Major selective communication generalize directly** to NVIDIA Ulysses-SP setups; the temporal-redundancy-applied-to-collectives idea is novel and applies on B200 as a 1.5–2× win on the all-to-all alone.

CoCoDiff is best understood as **the first paper to apply step-level temporal redundancy to communication rather than computation**, and that idea is portable. Open question: what's the right `r(t)` schedule for video DiTs (which have much longer step counts than image DiTs)?

---

## P.12 Cross-stack consolidated benchmarks

| Workload | Hardware | Setup | Result | Source |
|---|---|---|---|---|
| Wan 2.2 I2V, 1280×720, 81f, 40 steps | 8×H100 | FastVideo + SP (Ulysses) | **2.5× vs baseline** (~100s) | [Morphic](https://morphic.com/blog/boosting-wan2-2-i2v-56-faster) |
| Wan 2.2 1280×720 81f 40 steps | 1×B200 | Baseten optimized runtime | **3.2× vs default** (~40–50s) | [Baseten](https://www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/) |
| Wan 2.2 T2V 40 steps | 8×H100 | VoltagePark stack (Ulysses + FA3 + SageAttn + TeaCache) | **187s → 60s, 3.09×** (per-step 4.67s → 1.51s) | [VoltagePark](https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations) |
| Wan 2.2 TI2V-5B | 2×48GB GPU | SGLang u1r2 (Ring SP=2) | **1.42× e2e (90.6→63.7s)**, **−7.33 GB peak**, **Decoding 1.75×** | [SGLang](https://docs.sglang.ai/diffusion/performance/ring_sp_performance.html) |
| Wan 2.2 cross-attn skip-sp | 8×B200 | SGLang PR #19419 | **+7–15% per step** (ring=4: 4.72→3.99 s/step) | [sglang#19419](https://github.com/sgl-project/sglang/pull/19419) |
| Wan 2.1 I2V-14B-480P 40s 81f | 1×H100 vs 8×H100 | LightX2V cross-framework | **1×: 5.18 s/it (1.9× vs Diffusers)**; **8×: 0.75 s/it (3.9×)** | [lightx2v README](https://github.com/ModelTC/lightx2v) |
| Wan 2.1 I2V-14B-480P, 8×H100 + no-CFG + FP8 | 8×H100 | LightX2V | **0.35 s/step (2.1× over 8-GPU baseline)** | [lightx2v README](https://github.com/ModelTC/lightx2v) |
| HunyuanVideo 129f, 720p, 30 steps | L20 (single) | ParaAttention FBC | ~2× with first-block-cache alone | [ParaAttention HF](https://huggingface.co/docs/diffusers/main/optimization/para_attn) |
| CogVideoX-5B 49f@720×480 | 6×L40 | xDiT USP + CFG | **7.75× vs single-GPU** (~40s e2e) | [xDiT cogvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/cogvideo.md) |
| Mochi-1 49f@848×480 | 6×L40 | xDiT (USP cfg=2 ring=3) | **3.54× vs official** (74s vs 263s) | [mochi-xdit](https://github.com/xdit-project/mochi-xdit) |
| Step-Video-T2V (30B), TP scaling | 8×H20 NVLink | xDiT TP=8 vs SP=8 | TP8 30.4s SP8 30.1s; **TP saves 67% memory vs 14% for SP** | [xDiT stepvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/stepvideo.md) |
| Llama-3 8B/70B/105B GEMM-AG (M=16384) | 8×H100 NVSwitch | symm_mem Triton AG-matmul | **1.54× / 1.47× / 1.20× vs cuBLAS+NCCL** | [symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes) |
| Async-Ulysses pre-attn chunk | 2/4/8 × B200 | bf16, B=1, H=40, D=128 | −23/−25/−24% chunk, ~−3% e2e | [fal](https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/) |
| Fused QKV (`fused_all_gather_matmul`) | 2/4/8 × B200 | same as above | −37.3/−33.4/−4.6% chunk, −5.0/−4.8/−0.3% e2e | [fal](https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/) |
| Llama-3-8B 208k seq | 2×8 A800 | USP `u=8 r=2` | **47% MFU** | [USP paper](https://arxiv.org/abs/2405.07719) |
| GPT 7B/30B 32k seq | 32/64×A100 | DeepSpeed-Ulysses vs Megatron-SP | **2.5× faster, 4× longer seq, >10× comm reduction** | [arXiv:2309.14509](https://arxiv.org/abs/2309.14509) |
| Open-Sora train, 64 frames | 8×A100 | DSP vs Ulysses | **3× train, 2× infer, 50–75% comm reduction** | [arXiv:2403.10266](https://arxiv.org/abs/2403.10266) |
| Long-video VLM 2M-token train | 256×H100 | MM-SP vs Megatron CP+TP | **1.1–1.4× speedup, scales linearly to 2M tokens** | [LongVILA arXiv:2408.10188](https://arxiv.org/abs/2408.10188) |
| Long-video VLM 2M-token train | 256×H100 | MM-SP vs Ring | **2.1–5.7× speedup** | [LongVILA arXiv:2408.10188](https://arxiv.org/abs/2408.10188) |
| Striped Attention causal training | 8×A100 80GB | 256k seq, billion-scale model | **1.45× over Ring** | [arXiv:2311.09431](https://arxiv.org/abs/2311.09431) |
| Striped Attention causal training | 16×TPUv4 | 786k seq | **1.65× over Ring** | [arXiv:2311.09431](https://arxiv.org/abs/2311.09431) |
| 1025-frame Wan 2.1 video | 8×RTX 4090 | DualParal (block-PP + temporal) | **6.54× lower latency / 1.48× lower memory** | [arXiv:2505.21070](https://arxiv.org/html/2505.21070v1) |
| OpenSoraPlan + HunyuanVideo, multi-resolution | 8×A6000 / 8×L40 | PipeDiT (PipeSP + DeDiVAE + Aco) | **1.06–4.02× e2e** | [arXiv:2511.12056](https://arxiv.org/html/2511.12056v1) |
| VINs vs full-attn long video | (architectural) | parallel chunk denoising | **25–40% fewer FLOPs** at comparable VBench | [arXiv:2503.17539](https://arxiv.org/abs/2503.17539) |
| Latent Parallelism vs SP | multi-GPU video diffusion serving | dynamic dim rotation + overlap | **up to 97% comm reduction** | [arXiv:2512.07350](https://arxiv.org/html/2512.07350v1) |
| DiT-Serve vs prior systems | (multi-request serving) | step-level batching + Brick Attention | **2–3× tput / 3–4× latency** | [OpenReview](https://openreview.net/forum?id=NGNRc7rZBg) |
| Chorus inter-request cache | (4-step distilled) | prompt-similarity caching | **up to 45% speedup** | [arXiv:2604.04451](https://arxiv.org/abs/2604.04451) |
| StreamDiffusionV2 (RFC #2280 motivation) | 4×H100 NVLink | Wan2.1-1.3B 480p 1-step | **42 FPS, 0.37s TTFF** | RFC #2280 |
| StreamDiffusionV2 (RFC #2280 motivation) | 4×H100 NVLink | Wan2.1-14B 480p 4-step | **58 FPS, <0.5s TTFF** | RFC #2280 |
| CoCoDiff (Aurora — Intel) | 1–8 nodes / 96 tiles | TAPA + V-First + V-Major | **avg 3.6×, peak 8.4×** | [arXiv:2604.14561](https://arxiv.org/abs/2604.14561) |
| BurstEngine ≥1M token training | 64×A800 | ring + selective ckpt | **1.2× over baseline, 26.4% mem reduction** | [arXiv:2509.19836](https://arxiv.org/abs/2509.19836) |

---

## P.13 Where the SP↔TP crossover lives — final summary

For full-3D-attention video DiTs (Wan, HunyuanVideo, LTX-2) with 24–48 heads, hidden 4–6k, NVFP4 GEMMs on B200:

- The crossover sits roughly where per-rank `M = seq/SP` falls below **3–4k tokens** (the NVFP4 ridge point).
- Below that, TP beats SP; above, SP beats TP.
- **At 2–4 GPUs** the SP/TP grid is small enough that pure SP and pure TP are within a few % of each other for typical 5s @ 720p workloads.
- **At 8 GPUs**, pure SP=8 starves NVFP4 GEMMs at any seq ≤ ~25k, and pure TP=8 doubles RS exposure on the small O-projection. Hybrid `SP=2 × TP=4` is the sweet spot reported in production stacks.
- The fal data adds a *second* axis: for any fixed `(SP, TP)`, the optimal *transport* (NCCL functional collectives, Async-Ulysses, fused-AG-matmul via SymmMem, plain Ulysses) depends on per-rank message size. At `SP=8` on B200 with seq ~10k, async-Ulysses wins; at `SP=2` with the same seq, fused-AG-matmul wins.

The recipe lookup table:

| Model + workload | seq (tokens) | Best SP×TP grid | Best transport | Source |
|---|---|---|---|---|
| Wan 2.2 720p 81f 40-step | ~75k | SP=8 (Ulysses), or SP=2×TP=4 hybrid | NCCL functional + async | xDiT, Wan official |
| Wan 2.2 720p 81f distilled 4-step | ~75k | SP=2×TP=4 | NCCL functional | SGLang, Wan official |
| LTX-2 1080p 5s 8-step distilled | ~12k | SP=2×TP=4 (hybrid) | SymmMem fused-AG-matmul (P=2) | parent report §1.7 |
| HunyuanVideo 720p 129f | ~100k+ | SP=8 (USP `u=4 r=2`), or hybrid | NCCL functional + async | xDiT |
| Step-Video-T2V 30B | ~varies | TP=8 (param sharding dominates) | option-B AG/RS + async-TP | xDiT stepvideo.md |
| 1k-frame Wan video | very long | DualParal-PP (8 GPUs) + SP=2 inside | P2P + ring NVLink | DualParal |
| Live-stream 4f@480p chunks | ~1.5k | block PP (depth) + stream batch | dist.isend/recv | vLLM-Omni RFC #2280 |
| Multi-request mixed step counts | varies | SP×TP per request + DiT-Serve scheduler | as appropriate | DiT-Serve |

---

## P.14 Open questions / system-side gaps

These mirror the parent report's §1.11 but are sharpened to the parallelism axis:

1. **No published end-to-end CUTLASS DistGEMM ex.82 efficiency numbers on B200.** All public efficiency figures are Hopper. GDC + 2-CTA `tcgen05.mma` *should* give similar or better numbers, but no third party has measured.

2. **No fused ring-FA + cross-rank LSE merge in any production library.** Triton-distributed has `sp_ag_attention_intra_node.py` but it isn't a drop-in FA3/FA4 wrapper. Path forward is either Helion-generate the inner loop and wrap with a Python-level cross-rank merge, or extend FA4's CuTe-DSL kernel to expose `(O, lse)` dual-output mode. **db-SP** is the closest published partial workaround for the related sparse-attention imbalance problem.

3. **NVFP4 scaling exposure.** Per-tile NVFP4 scale tensors must move with activations across SP a2as / TP RS — what's the right scale-tensor sharding? FP8 rowwise async-TP support landed in PyTorch (PR #149247). **NVFP4 block-scale support in async-TP is not yet upstream**.

4. **Async-Ulysses on Blackwell with payloads <1 MB.** The fal blog explicitly shows fixed overheads eat the gains at 8×B200 with strong scaling. The vLLM thresholds (40 MB on H100, 32 MB FP8) are an LLM-decode-shaped data point; the video DiT sub-1.5k-token-forward regime (vLLM-Omni stream-batch RFC #2280) is different.

5. **Quantitative per-axis SP+TP sweep on B200 for video DiT.** Every published video benchmark on B200 reports a single configuration. There is no full `(SP, TP) × seq × transport` grid measured on a B200 for any of Wan 2.2 / HunyuanVideo / LTX-2.

6. **VAE parallelism.** Most tables measure DiT denoising; VAE decode on long-video latents is a real fraction of e2e time. SGLang reports **1.75× decode-stage speedup with Ring SP=2**. vLLM-Omni added "VAE Patch Parallelism"; no published benchmark yet.

7. **Comm-comp overlap interaction with CUDA Graphs.** Both CUTLASS (CUDA Graphs + GDC) and Triton-distributed assume static shapes inside the captured graph. VSA / sliding-tile-attention shape variability per timestep is a live constraint.

8. **No third-party verification of the 1.8 TB/s NVLink5 number under concurrent producer/consumer + GEMM load.** Effective bandwidth under realistic comm/compute mix on B200 is reported only piecewise.

9. **Wan 2.2-A14B MoE expert dispatch — is COMET-style fusion deployed anywhere?** As of May 2026 no public framework fuses the expert dispatch with the surrounding transformer blocks for Wan 2.2.

10. **No published benchmark of FA4 + ring-style SP on a video DiT.** FA4 is the canonical attention kernel for B200; ring needs `(O, lse)` access and FA4 doesn't (yet) expose this in the public C++ / CuTe-DSL API.

11. **Auto-selector for transport (NCCL functional vs SymmMem vs fused-AG-matmul)** — fal showed no single transport dominates across `P` and seq. A runtime selector based on world size and message size is the obvious win; no production stack ships it.

12. **CoCoDiff's V-First and V-Major on B200.** The Aurora-specific TAPA collapses, but V-First and V-Major selective communication should generalize. Unmeasured.

13. **DualParal × USP × NVFP4 composition.** No public recipe combines block-wise PP (DualParal), USP for in-stage parallelism, and NVFP4 quantization on B200. The combined recipe is the natural answer for >1k-frame video on a single node, but no benchmark exists.

---

*Document references: Jacobs et al. (Ulysses, [arXiv:2309.14509](https://arxiv.org/abs/2309.14509)); Liu et al. (Ring, [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)); Brandon et al. (Striped, [arXiv:2311.09431](https://arxiv.org/abs/2311.09431)); Zhao et al. (DSP, [arXiv:2403.10266](https://arxiv.org/abs/2403.10266)); Fang & Zhao (USP, [arXiv:2405.07719](https://arxiv.org/abs/2405.07719)); Korthikanti et al. (Megatron-SP, [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)); Chen et al. (LongVILA/MM-SP, [arXiv:2408.10188](https://arxiv.org/abs/2408.10188)); Ma et al. (VeOmni, [arXiv:2508.02317](https://arxiv.org/html/2508.02317v1)); Dedhia et al. (VINs, [arXiv:2503.17539](https://arxiv.org/abs/2503.17539)); Wang et al. (DualParal, [arXiv:2505.21070](https://arxiv.org/html/2505.21070v1)); Wang/Wang/Shi (PipeDiT, [arXiv:2511.12056](https://arxiv.org/html/2511.12056v1)); Wu et al. (Latent Parallelism, [arXiv:2512.07350](https://arxiv.org/html/2512.07350v1)); Luo et al. (DiT-Serve, [OpenReview](https://openreview.net/forum?id=NGNRc7rZBg)); Ma et al. (CoCoDiff, [arXiv:2604.14561](https://arxiv.org/abs/2604.14561)); Sun et al. (BurstEngine, [arXiv:2509.19836](https://arxiv.org/abs/2509.19836)); fal blog ["Ulysses Unbound"](https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/); PyTorch SymmMem 2.12 [docs](https://docs.pytorch.org/docs/2.12/symmetric_memory.html). Code repos: [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention), [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention), [ByteDance-Seed/VeOmni](https://github.com/ByteDance-Seed/VeOmni), [yifuwang/symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes), [meta-pytorch/kraken](https://github.com/meta-pytorch/kraken), [xdit-project/xDiT](https://github.com/xdit-project/xDiT), [chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention), [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo), [NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2), [docs.sglang.ai/diffusion](https://docs.sglang.ai/diffusion/), [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni), [diffusers PR #11941](https://github.com/huggingface/diffusers/pull/11941), [TorchTitan PR #429](https://github.com/pytorch/torchtitan/pull/429), [vLLM #27700](https://github.com/vllm-project/vllm/issues/27700).*
