## §9. References

Indexed by category. URLs and arXiv IDs are inline so each can be followed directly. Each reference flags whether the cited work was fetched directly during this expansion (✓), inferred from secondary sources (◇), or unreached (✗).

### §9.1 Hardware / kernels

- ✓ **NVIDIA HGX B200 datasheet**: <https://www.nvidia.com/en-us/data-center/b200/> — 18 PFLOPS sparse / 9 PFLOPS dense FP4 per GPU.
- ✓ **DGX B200 user guide**: <https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html>
- ✓ **Microbenchmarking NVIDIA's Blackwell Architecture**, [arXiv:2512.02189](https://arxiv.org/abs/2512.02189) (Dec 2025) — TMEM 420-cycle vs Hopper 1000-cycle, FP4 7700 TFLOP/s.
- ✓ **CUTLASS Blackwell functionality docs**: <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md>
- ✓ **CUTLASS Blackwell SM100 doc**: <https://docs.nvidia.com/cutlass/4.4.0/media/docs/cpp/blackwell_functionality.html>
- ✓ **CUTLASS DistGEMM example 82** (Blackwell): `examples/82_blackwell_distributed_gemm/82_blackwell_distributed_gemm.cu`. Hopper baseline (ex.65): SHI Labs blog <https://medium.com/shi-labs/distributed-gemm-88be6a481e2b>.
- ✓ **FlashAttention-3**, [arXiv:2407.08608](https://arxiv.org/abs/2407.08608) (Jul 2024, NeurIPS'24); [PyTorch blog](https://pytorch.org/blog/flashattention-3/).
- ✓ **FlashAttention-4**, [arXiv:2603.05451](https://arxiv.org/html/2603.05451v1) (Mar 2026); [Together blog](https://www.together.ai/blog/flashattention-4); [Lambda blog](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet); [Tri Dao FA4 blog](https://tridao.me/blog/2026/flash4/); [Dao‑AILab/flash‑attention](https://github.com/Dao-AILab/flash-attention).
- ◇ **FA-FP4 fork**: [hao-ai-lab/flash-attention-fp4](https://github.com/hao-ai-lab/flash-attention-fp4) — 1801 TF/s, 1.39× over BF16 FA4. (README minimal; numbers verified via parent report's primary fetches.)
- ✓ **cuDNN 9.21** SDPA on Blackwell: <https://docs.nvidia.com/deeplearning/cudnn/backend/v9.21.0/release-notes.html>; [cuDNN frontend Attention v1.22](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.22.0/operations/Attention.html).
- ✓ **MagiAttention v1.1.0 + FFA_FA4**: [SandAI-org/MagiAttention release](https://github.com/SandAI-org/MagiAttention/releases) (Feb 2026); FFA_FA4 fork at [demonatic/flash-attention magi_attn_blackwell_support](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support); CP benchmark blog <https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html#for-b200>; MAGI-1 [arXiv:2505.13211](https://arxiv.org/abs/2505.13211); Distributed Muon QK-Clip [arXiv:2507.20534](https://arxiv.org/abs/2507.20534).
- ✓ **ParallelKittens (PK)**, [arXiv:2511.13940](https://arxiv.org/abs/2511.13940) (Nov 2025); [HazyResearch/ThunderKittens/kernels/parallel](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/parallel); [Hazy Research blog](https://hazyresearch.stanford.edu/blog/2025-11-17-pk).
- ✓ **BurstEngine**, [arXiv:2509.19836](https://arxiv.org/abs/2509.19836); [thunlp/BurstEngine](https://github.com/thunlp/BurstEngine).
- ✓ **BurstAttention**, [arXiv:2403.09347](https://arxiv.org/abs/2403.09347) (Mar 2024).
- ✓ **Triton-distributed (TileLink)**, [arXiv:2504.19442](https://arxiv.org/abs/2504.19442) (MLSys'25); [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed).
- ✓ **Meta Kraken**: [meta-pytorch/kraken](https://github.com/meta-pytorch/kraken).
- ✓ **ByteDance FLUX**, [arXiv:2406.06858](https://arxiv.org/abs/2406.06858); [github.com/bytedance/flux](https://github.com/bytedance/flux).
- ✓ **COMET (MoE overlap)**, [arXiv:2502.19811](https://arxiv.org/abs/2502.19811) (MLSys 2025).
- ✓ **NVSHMEM 3.5.19 release notes**: <https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3519.html>; **NVSHMEM 3.2.5**: <https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-3205.html>; **NVSHMEM 3.6.5**: <https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3605.html>; NVLS allreduce + tile API.
- ✓ **PyTorch Symmetric Memory 2.12 docs**: <https://docs.pytorch.org/docs/2.12/symmetric_memory.html>; [yifuwang/symm-mem-recipes](https://github.com/yifuwang/symm-mem-recipes).
- ✓ **PyTorch Helion blog**: <https://pytorch.org/blog/helion/>; [pytorch/helion](https://github.com/pytorch/helion); [helionlang.com/examples/attention.html](https://helionlang.com/examples/attention.html).
- ✓ **Striped Attention**: [arXiv:2311.09431](https://arxiv.org/abs/2311.09431).
- ✓ **Tree Attention**: [arXiv:2408.04093](https://arxiv.org/abs/2408.04093).
- ✓ **DistFlashAttn / LightSeq**: [arXiv:2310.03294](https://arxiv.org/abs/2310.03294); [RulinShao/LightSeq](https://github.com/RulinShao/LightSeq).
- ✓ **PyTorch FlexAttention blog**: [pytorch.org/blog/flexattention](https://pytorch.org/blog/flexattention/).
- ✓ **DeepGEMM**: [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM); [PR #304 FP4 Mega MoE](https://github.com/deepseek-ai/DeepGEMM/pull/304); [TRT-LLM PR #6486](https://github.com/NVIDIA/TensorRT-LLM/pull/6486).
- ✓ **DeepEP**: [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP).
- ✓ **FlashInfer**: [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).
- ✓ **sgl-kernel**: [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) (sgl-kernel/ subdirectory).
- ✓ **q8_kernels** (Lightricks LTX-Video FP8 kernels): repository linked from Lightricks/LTX-Video.
- ✓ **Triton block-scaled matmul tutorial**: [`python/tutorials/10-block-scaled-matmul.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py); commits [`9181d1e`](https://github.com/triton-lang/triton/commit/9181d1e33c231f8db266414a65b50c9b711cb7af), [`24445a1`](https://github.com/triton-lang/triton/commit/24445a1cfaf10457ae7bb3a9c27043b8aa278366).
- ✓ **PyTorch `expandable_segments` allocator** + RFC: [pytorch/pytorch#165419](https://github.com/pytorch/pytorch/issues/165419).
- ✓ **TransformerEngine 2.10 advanced-optimizations**: <https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.10/user-guide/examples/advanced_optimizations.html>.

### §9.2 Quantization

- ✓ **Introducing NVFP4**, NVIDIA dev blog: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference>
- ✓ **PyTorch / TorchAO + Diffusers Blackwell blog**: <https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/> — LTX-2 B200 NVFP4 1.56× / 1.67× / 1.68× at B=1/4/8 [VERIFIED VERBATIM]; companion repo [github.com/sayakpaul/diffusers-blackwell-quants](https://github.com/sayakpaul/diffusers-blackwell-quants); MSLK kernel [github.com/meta-pytorch/MSLK](https://github.com/meta-pytorch/MSLK); [TorchAO PR #4031](https://github.com/pytorch/ao/pull/4031).
- ✓ **cuBLAS 12.9 NVFP4**: <https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/>; example [LtNvfp4Matmul](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtNvfp4Matmul).
- ✓ **OCP MX format spec**: Rouhani et al. ["Microscaling Data Formats for Deep Learning"](https://arxiv.org/abs/2310.10537), 2023.
- ✓ **PTQ4DiT**, [arXiv:2405.16005](https://arxiv.org/abs/2405.16005) (NeurIPS 2024); [code](https://github.com/adreamwu/PTQ4DiT).
- ✓ **Q-DiT**, [arXiv:2406.17343](https://arxiv.org/html/2406.17343v2) (CVPR 2025); [code](https://github.com/Juanerx/Q-DiT).
- ✓ **ViDiT-Q**, [arXiv:2406.02540](https://arxiv.org/html/2406.02540v1) (ICLR 2025).
- ✓ **SVDQuant**, [arXiv:2411.05007](https://arxiv.org/html/2411.05007v4) (ICLR 2025 Spotlight); [project](https://hanlab.mit.edu/projects/svdquant); [SVDQuant-NVFP4 blog](https://hanlab.mit.edu/blog/svdquant-nvfp4); **Nunchaku** [github.com/nunchaku-ai/nunchaku](https://github.com/nunchaku-ai/nunchaku); open issue [deepcompressor#103](https://github.com/nunchaku-ai/deepcompressor/issues/103).
- ✓ **EfficientDM**, [arXiv:2310.03270](https://arxiv.org/html/2310.03270v4) (ICLR 2024).
- ✓ **ConvRot**, [arXiv:2512.03673](https://arxiv.org/html/2512.03673v1) (Dec 2025).
- ◇ **QuEST**, [arXiv:2402.03666](https://arxiv.org/html/2402.03666v3) (ICCV 2025).
- ✓ **QuaRot**, [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) (NeurIPS 2024); **SpinQuant** [code](https://github.com/facebookresearch/spinquant); **DFRot** [arXiv:2412.00648](https://arxiv.org/abs/2412.00648).
- ◇ **SemanticDialect / SeDA**, [arXiv:2603.02883](https://arxiv.org/abs/2603.02883). Predecessor BlockDialect [arXiv:2501.01144](https://arxiv.org/abs/2501.01144).
- ✓ **NVIDIA TensorRT Model Optimizer**: [github.com/NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer); [examples/diffusers/quantization](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/diffusers/quantization).
- ✓ **TensorRT-LLM VisualGen**: [docs/source/models/visual-generation.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/visual-generation.md).
- ✓ **lightx2v Wan-NVFP4**: [HF lightx2v/Wan-NVFP4](https://huggingface.co/lightx2v/Wan-NVFP4) — first publicly-shipped NVFP4 + 4-step Wan 2.1.
- ✓ **ac4k Wan2.2** NVFP4 fork: [github.com/ac4k/Wan2.2](https://github.com/ac4k/Wan2.2); HF [ac4k/wan2.2-T2V-A14B-NVFP4](https://huggingface.co/ac4k/wan2.2-T2V-A14B-NVFP4); [ac4k/wan2.2-I2V-A14B-NVFP4](https://huggingface.co/ac4k/wan2.2-I2V-A14B-NVFP4).
- ✓ **vLLM-Omni FP8 Wan 2.2**: [PR #1412](https://github.com/vllm-project/vllm-omni/pull/1412).
- ✓ **diffusers-torchao**: [github.com/sayakpaul/diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao).
- ✓ **bitsandbytes 4-bit / 8-bit**: [HF docs](https://huggingface.co/docs/diffusers/main/en/quantization/bitsandbytes).
- ✓ **TurboDiffusion**, [arXiv:2512.16093](https://arxiv.org/abs/2512.16093); [thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion); [tech report PDF](https://jt-zhang.github.io/files/TurboDiffusion_Technical_Report.pdf).
- ✓ **QVGen** (W4A4 QAT for video): [arXiv:2505.11497](https://arxiv.org/html/2505.11497v1) (ICLR 2026); [github.com/ModelTC/QVGen](https://github.com/ModelTC/QVGen).
- ✓ **Attn-QAT** (FP4 attention QAT): [arXiv:2603.00040](https://arxiv.org/abs/2603.00040).
- ✓ **TAEHV**: [github.com/madebyollin/taehv](https://github.com/madebyollin/taehv); **FLUX.2 Small Decoder**: [HF black-forest-labs/FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder).
- ✓ **SmoothQuant** (foundational, briefly cited): [arXiv:2211.10438](https://arxiv.org/abs/2211.10438).

### §9.3 Attention

- ✓ **SageAttention 1**, [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) (ICLR 2025); **SageAttention 2**, [arXiv:2411.10958](https://arxiv.org/abs/2411.10958); **SageAttention 2++**, [arXiv:2505.21136](https://arxiv.org/abs/2505.21136); **SageAttention 3**, [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) (NeurIPS 2025 Spotlight); [code](https://github.com/thu-ml/SageAttention); [issue #322](https://github.com/thu-ml/SageAttention/issues/322).
- ✓ **SpargeAttn**, [arXiv:2502.18137](https://arxiv.org/abs/2502.18137) (ICML 2025); [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn).
- ✓ **VSA — Video Sparse Attention**, [arXiv:2505.13389](https://arxiv.org/abs/2505.13389) (NeurIPS 2025); FastVideo recipe configs.
- ✓ **STA — Sliding Tile Attention**, [arXiv:2502.04507](https://arxiv.org/abs/2502.04507).
- ◇ **SSTA**, [arXiv:2511.18870](https://arxiv.org/abs/2511.18870) (HunyuanVideo 1.5).
- ✓ **SVG / SVG2**, [arXiv:2502.01776](https://arxiv.org/abs/2502.01776) / [arXiv:2505.18875](https://arxiv.org/abs/2505.18875); [svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen).
- ✓ **Radial Attention**, [arXiv:2506.19852](https://arxiv.org/abs/2506.19852) (NeurIPS 2025); [code](https://github.com/mit-han-lab/radial-attention).
- ◇ **VMoBA**, [arXiv:2506.23858](https://huggingface.co/papers/2506.23858); [KlingAIResearch/VMoBA](https://github.com/KlingAIResearch/VMoBA). **MoBA**, [arXiv:2502.13189](https://arxiv.org/abs/2502.13189).
- ◇ **NSA — Native Sparse Attention** (DeepSeek), [arXiv:2502.11089](https://arxiv.org/abs/2502.11089).
- ✗ **MInference 1.0 / 2.0**: [github.com/microsoft/MInference](https://github.com/microsoft/MInference) — only repo description retrieved; primary content cited via SVG/SLA papers.
- ✓ **XAttention**, [arXiv:2503.16428](https://arxiv.org/abs/2503.16428).
- ✓ **ΔAttention**, [arXiv:2505.11254](https://arxiv.org/abs/2505.11254).
- ✓ **SLA — Sparse-Linear Attention**, [arXiv:2509.24006](https://arxiv.org/abs/2509.24006); [thu-ml/SLA](https://github.com/thu-ml/SLA).
- ✓ **BSA**, [arXiv:2509.01085](https://arxiv.org/abs/2509.01085).
- ✓ **DraftAttention**, [arXiv:2505.14708](https://arxiv.org/abs/2505.14708).
- ✓ **DiTFastAttn**, [arXiv:2406.08552](https://arxiv.org/abs/2406.08552) (NeurIPS 2024).
- ✓ **db-SP**, [arXiv:2511.23113](https://arxiv.org/abs/2511.23113); [thu-nics/db-SP](https://github.com/thu-nics/db-SP).
- ✓ **DSA — Distributed Sparse Attention**, [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10011818).
- ✓ **Astraea**, [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008349).
- ✓ **TeaCache**, [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.pdf); [project](https://liewfeng.github.io/TeaCache); [code](https://github.com/ali-vilab/TeaCache); arXiv [2411.19108](https://arxiv.org/abs/2411.19108).
- ✓ **First-Block-Cache (FBCache) / ParaAttention**: [chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention); HF docs [`para_attn`](https://huggingface.co/docs/diffusers/main/optimization/para_attn).
- ✓ **TaylorSeer**, [arXiv:2503.06923](https://arxiv.org/abs/2503.06923) (ICCV 2025); [Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).
- ✓ **AdaCache**, [arXiv:2411.02397](https://arxiv.org/html/2411.02397v2).
- ◇ **SmoothCache**, [arXiv:2411.10510](http://arxiv.org/abs/2411.10510v2).
- ◇ **Δ-DiT**, [arXiv:2406.01125](http://arxiv.org/abs/2406.01125).
- ◇ **FasterDiffusion**, [arXiv:2312.09608](https://arxiv.org/abs/2312.09608) (NeurIPS 2024).
- ◇ **DeepCache** (NeurIPS 2024).
- ✓ **MagCache**, [arXiv:2506.09045](https://arxiv.org/abs/2506.09045); [github.com/zehong-ma/MagCache](https://github.com/zehong-ma/MagCache).
- ◇ **BWCache**, [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011445).
- ◇ **ScalingCache**, [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006887).
- ✓ **SpeCa**, [arXiv:2509.11628](https://arxiv.org/abs/2509.11628). **SDVG**, [arXiv:2604.17397](https://arxiv.org/abs/2604.17397).
- ✓ **VideoRoPE**: [proceedings.mlr.press/v267/wei25h](https://proceedings.mlr.press/v267/wei25h.html), arXiv mirror [2502.05173](https://arxiv.org/abs/2502.05173).
- ✓ **ToMeSD** (briefly): [arXiv:2303.17604](https://arxiv.org/abs/2303.17604).
- ◇ **VidToMe** (briefly): [arXiv:2312.10656](https://arxiv.org/abs/2312.10656).
- ✓ **Star Attention** (briefly): [NVIDIA/Star-Attention](https://github.com/NVIDIA/Star-Attention).

### §9.4 Model-level / distillation

- ✓ **Progressive Distillation**: [arXiv:2202.00512](https://arxiv.org/abs/2202.00512) (Salimans & Ho, ICLR 2022).
- ✓ **LCM / LCM-LoRA**: Luo et al., 2023.
- ✓ **AnimateLCM**: [arXiv:2402.00769](https://arxiv.org/abs/2402.00769).
- ✓ **CTM (Consistency Trajectory Models)**: [arXiv:2310.02279](https://arxiv.org/abs/2310.02279) (Kim et al., ICLR 2024).
- ✓ **DMD/DMD2**, [arXiv:2405.14867](https://arxiv.org/abs/2405.14867) (NeurIPS 2024 oral); [tianweiy/DMD2](https://github.com/tianweiy/DMD2).
- ✓ **CausVid**, [arXiv:2412.07772](https://arxiv.org/abs/2412.07772) (CVPR 2025); [tianweiy/CausVid](https://github.com/tianweiy/CausVid).
- ✓ **Self-Forcing**, [paper](https://arxiv.org/pdf/2506.08009) [arXiv:2506.08009](https://arxiv.org/abs/2506.08009) (NeurIPS 2025); [code](https://github.com/guandeh17/Self-Forcing); **Self-Forcing-Plus** [GoatWu/Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus).
- ✓ **Phased DMD**, [arXiv:2510.27684](https://arxiv.org/html/2510.27684v3); behind Wan 2.2-Lightning V1.1/V2.
- ◇ **TMD — Transition Matching Distillation**, [arXiv:2601.09881](https://arxiv.org/abs/2601.09881); project [research.nvidia.com/labs/genair/tmd](https://research.nvidia.com/labs/genair/tmd) [VBench VERIFIED, wall-clock NOT IN PAPER].
- ✓ **rCM — Score-Regularized Continuous-Time CM**, [arXiv:2510.08431](https://arxiv.org/abs/2510.08431); [NVlabs/rcm](https://github.com/NVlabs/rcm); ICLR 2026.
- ◇ **OSV — One-Step Video**, [arXiv:2409.11367](https://arxiv.org/abs/2409.11367); CVPR 2025.
- ◇ **T2V-Turbo / T2V-Turbo-v2**, [arXiv:2410.05677](https://arxiv.org/abs/2410.05677); [project](https://t2v-turbo.github.io/).
- ◇ **TDM**, [project](https://tdm-t2x.github.io/); [code](https://github.com/Luo-Yihong/TDM).
- ◇ **DCM**, [project](https://vchitect.github.io/DCM/); [HF cszy98/DCM](https://huggingface.co/cszy98/DCM).
- ◇ **DOLLAR**, [arXiv:2412.15689](https://arxiv.org/abs/2412.15689) (ICCV 2025).
- ◇ **Hyper-SD** (TSCD): [arXiv:2404.13686](https://arxiv.org/abs/2404.13686).
- ◇ **PCM**: [arXiv:2405.18407](https://arxiv.org/abs/2405.18407).
- ◇ **InstaFlow**, [ICLR 2024]; **2-Rectified Flow**; **PeRFlow** (NeurIPS 2024).
- ✓ **Wan 2.2-Lightning**: [github.com/ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning); [HF lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning); pre-merged [HF lightx2v/Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models).
- ✓ **lightx2v Wan2.2-I2V-A14B-Moe-Distill**: [HF](https://huggingface.co/lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v).
- ✓ **lightx2v Wan2.1 Step+CFG distill**: [HF 480P](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v), [720P](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v), T2V at [HF](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill).
- ✓ **FastWan / FastVideo**: [github.com/hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo); [FastWan blog](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/); HF [FastVideo/FastWan2.1-T2V-1.3B-Diffusers](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers), [FastVideo/FastWan2.2-TI2V-5B-Diffusers](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-Diffusers), [FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers); FastVideo LTX-2.3 1080p demo blog <https://haoailab.com/blogs/fastvideo_realtime_1080p/>.
- ✓ **Wan2.2-TI2V-5B-Turbo**: [github.com/quanhaol/Wan2.2-TI2V-5B-Turbo](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo).
- ✓ **HunyuanVideo-1.5 step-distill**: [HF model card](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/assets/step_distillation_comparison.md); [diffusers PR #12802](https://github.com/huggingface/diffusers/pull/12802).
- ✓ **LTX-2 official**: [github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video); [github.com/Lightricks/LTX-2](https://github.com/Lightricks/LTX-2); HF [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) (7-artifact catalog) [VERIFIED]; LTX-2.3 NVFP4 [Lightricks/LTX-2.3-nvfp4](https://huggingface.co/Lightricks/LTX-2.3-nvfp4).
- ✓ **Cosmos-Predict 2.5 distilled cookbook**: [nvidia-cosmos.github.io/cosmos-cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/distilling_predict2.5.html); references **sCM/TrigFlow** [arXiv:2410.11081](https://arxiv.org/abs/2410.11081).
- ✓ **Krea Realtime 14B**: [blog](https://www.krea.ai/blog/krea-realtime-14b); [HF](https://hf.co/krea/krea-realtime-video).
- ✓ **HY-WorldPlay**, [arXiv:2512.14614](https://arxiv.org/abs/2512.14614); [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) (built on HunyuanVideo).
- ✓ **Helios**, [arXiv:2603.04379](https://arxiv.org/abs/2603.04379); [PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios) (independent).
- ◇ **AGD (CFG adapter)**: [arXiv:2503.07274](https://arxiv.org/abs/2503.07274).
- ✓ **BLADE**: [arXiv:2508.10774](https://arxiv.org/html/2508.10774v1); [project](http://ziplab.co/BLADE-Homepage/).
- ◇ **V.I.P.**: [arXiv:2508.03254](https://arxiv.org/html/2508.03254v1).
- ◇ **PPCL**: [arXiv:2511.16156](https://arxiv.org/abs/2511.16156); [github.com/OPPO-Mente-Lab/Qwen-Image-Pruning](https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning).
- ◇ **Mobile Video DiT (Snap)**: [arXiv:2507.13343](https://arxiv.org/abs/2507.13343).
- ◇ **W2SVD / MagicDistillation**: [arXiv:2503.13319](https://arxiv.org/abs/2503.13319).
- ✓ **FastHunyuan**: [HF FastVideo/FastHunyuan](https://huggingface.co/FastVideo/FastHunyuan).
- ✓ **FastMochi**: [HF FastVideo/FastMochi-diffusers](https://huggingface.co/FastVideo/FastMochi-diffusers); **Mochi-1-Transformer-42** depth-pruned variant.
- ◇ **TDM-CogVideoX-2B-LoRA**: [HF](https://huggingface.co/Luo-Yihong/TDM_CogVideoX-2B_LoRA).
- ◇ **FlashMotion**: [project page](https://quanhaol.github.io/flashmotion-site/).
- ◇ **Step-Video-T2V-Turbo**: [github.com/stepfun-ai/Step-Video-T2V](https://github.com/stepfun-ai/Step-Video-T2V); [arXiv:2502.10248](https://arxiv.org/abs/2502.10248).
- ◇ **Adversarial Distribution Matching**, [arXiv:2507.18569](https://arxiv.org/abs/2507.18569).
- ◇ **Autoregressive Adversarial Post-Training (AAPT)**, [arXiv:2506.09350](https://arxiv.org/abs/2506.09350).
- ✓ **MotionStream**: referenced on Self-Forcing project page.
- ◇ **MeanFlow / iMF / AlphaFlow / Rectified MeanFlow** (image-only): [arXiv:2505.13447](https://arxiv.org/abs/2505.13447), [arXiv:2510.20771](https://arxiv.org/abs/2510.20771), [arXiv:2512.02012](https://arxiv.org/html/2512.02012v1), [arXiv:2511.23342](https://arxiv.org/abs/2511.23342v1).
- ◇ **UniPC** ([HF docs](https://huggingface.co/docs/diffusers/main/api/schedulers/unipc), [arXiv:2302.04867](https://arxiv.org/abs/2302.04867)); **DPM-Solver / DPM-Solver++** ([github](https://github.com/LuChengTHU/dpm-solver)); **A-FloPS** ([arXiv:2509.00036](https://arxiv.org/html/2509.00036v1)).

### §9.5 Frameworks

- ✓ **DeepSpeed-Ulysses**: [arXiv:2309.14509](https://arxiv.org/abs/2309.14509); [microsoft/DeepSpeed `blogs/deepspeed-ulysses`](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md).
- ✓ **Ring Attention**: [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) (ICLR'24); [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention).
- ✓ **USP — Unified Sequence Parallelism**: [arXiv:2405.07719](https://arxiv.org/abs/2405.07719); [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention).
- ✓ **DSP — Dynamic SP**: [arXiv:2403.10266](https://arxiv.org/abs/2403.10266) (ICML 2025); [NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys).
- ✓ **MM-SP / LongVILA**: [arXiv:2408.10188](https://arxiv.org/abs/2408.10188).
- ✓ **VeOmni async-Ulysses**: [arXiv:2508.02317](https://arxiv.org/html/2508.02317v1); [ByteDance-Seed/VeOmni](https://github.com/ByteDance-Seed/VeOmni); fal blog [Ulysses Unbound](https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/).
- ✓ **Megatron-SP / option B**: [arXiv:2205.05198](https://arxiv.org/abs/2205.05198); [PyTorch TP tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html).
- ✓ **Async-TP in PyTorch / TorchTitan**: [TorchTitan PR #429](https://github.com/pytorch/torchtitan/pull/429), [SymmMem PR #179918](https://github.com/pytorch/pytorch/pull/179918), FP8 rowwise PR [#149247](https://github.com/pytorch/pytorch/pull/149247), vLLM thresholds [#27700](https://github.com/vllm-project/vllm/issues/27700).
- ✓ **xDiT**: [github.com/xdit-project/xDiT](https://github.com/xdit-project/xDiT); [usp.md](https://github.com/xdit-project/xDiT/blob/main/docs/methods/usp.md); [stepvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/stepvideo.md); [cogvideo.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/cogvideo.md).
- ✓ **ParaAttention**: [github.com/chengzeyi/ParaAttention](https://github.com/chengzeyi/ParaAttention).
- ✓ **FastVideo**: [github.com/hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo).
- ✓ **VideoSys (DSP)**: [github.com/NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys).
- ✓ **lightx2v**: [github.com/ModelTC/lightx2v](https://github.com/ModelTC/lightx2v); [Mooncake disagg blog](https://www.lightx2v.org/blog/lightx2v-disaggregated-deployment-breaking-the-memory-and-throughput-bottlenecks-of-diffusion-model-inference) (April 2026).
- ✓ **diffusers native CP** ([PR #11941](https://github.com/huggingface/diffusers/pull/11941)).
- ✓ **vLLM-Omni**: [docs.vllm.ai/projects/vllm-omni](https://docs.vllm.ai/projects/vllm-omni); [PR #1412 FP8 Wan 2.2](https://github.com/vllm-project/vllm-omni/pull/1412); [stream-batch RFC #2280](https://github.com/vllm-project/vllm-omni/issues/2280); [Batched Diffusion Serving #1511](https://github.com/vllm-project/vllm-omni/issues/1511).
- ✓ **SGLang Diffusion**: [docs.sglang.ai/diffusion](https://docs.sglang.ai/diffusion/); [SGLang PR #16532](https://github.com/sgl-project/sglang/pull/16532); [PR #19419 cross-attn skip-sp](https://github.com/sgl-project/sglang/pull/19419); [compatibility matrix](https://docs.sglang.io/docs/sglang-diffusion/compatibility_matrix); [SGLang PR #17693](https://github.com/sgl-project/sglang/pull/17693).
- ✓ **Wan-Video/Wan2.2 official**: [github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2).
- ✓ **xdit-project/HunyuanVideo**: [github.com/xdit-project/HunyuanVideo](https://github.com/xdit-project/HunyuanVideo).
- ✓ **DualParal (PP)**: [arXiv:2505.21070](https://arxiv.org/html/2505.21070v1); [code](https://github.com/dualparal-project/dualparal).
- ✓ **PipeDiT**: [arXiv:2511.12056](https://arxiv.org/html/2511.12056v1).
- ✓ **Video Interface Networks**: [arXiv:2503.17539](https://arxiv.org/abs/2503.17539).
- ✓ **DiT-Serve**: [OpenReview NGNRc7rZBg](https://openreview.net/forum?id=NGNRc7rZBg).
- ✓ **Chorus**: [arXiv:2604.04451](https://arxiv.org/abs/2604.04451v1).
- ✓ **Latent Parallelism**: [arXiv:2512.07350](https://arxiv.org/html/2512.07350v1).
- ✓ **CoCoDiff**, [arXiv:2604.14561](https://arxiv.org/abs/2604.14561) (Apr 2026, Aurora-Intel).

### §9.6 Engineering blogs

- ✓ **PyTorch TorchAO Blackwell**: <https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/>
- ✓ **fal Ulysses-Unbound (8×B200 measurements)**: <https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/>
- ✓ **VoltagePark Wan 2.2 4.67 s → 1.51 s/step (8×H100)**: <https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations>
- ✓ **Baseten Wan 2.2 (3.2× B200)**: <https://www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/>
- ✓ **Morphic Wan 2.2 I2V (2.5× 8×H100)**: <https://morphic.com/blog/boosting-wan2-2-i2v-56-faster>
- ✓ **SHI Labs DistGEMM**: <https://medium.com/shi-labs/distributed-gemm-88be6a481e2b>
- ✓ **Hao AI Lab FastWan / FastVideo / LTX-2.3 1080p**: <https://hao-ai-lab.github.io/blogs/fastvideo_post_training/>; <https://haoailab.com/blogs/fastvideo_realtime_1080p/>; <https://hao-ai-lab.github.io/blogs/fastvideo_causalwan_preview/>
- ✓ **NVIDIA NVFP4 intro**: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference>
- ✓ **NVIDIA cuBLAS 12.9 NVFP4**: <https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/>
- ✓ **TransformerEngine 2.10 advanced-optimizations**: <https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.10/user-guide/examples/advanced_optimizations.html>
- ✓ **Tri Dao FA3 / FA4 blogs**: <https://tridao.me/blog/2024/flash3/>; <https://tridao.me/blog/2026/flash4/>
- ✓ **Together FlashAttention-4**: <https://www.together.ai/blog/flashattention-4>
- ✓ **Lambda FlashAttention-4**: <https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet>

### §9.7 Local code and FastVideo source

The expansion drew on direct reads of the local FastVideo / CUTLASS repositories. Notable files:

- `/Users/yewang/work/FastVideo/fastvideo/training/distillation_pipeline.py` (DMD2 forward, lines 612–690)
- `/Users/yewang/work/FastVideo/fastvideo/distributed/communication_op.py` (`sequence_model_parallel_all_to_all_4D`)
- `/Users/yewang/work/FastVideo/fastvideo/distributed/device_communicators/base_device_communicator.py` (`AllToAll4D.forward`)
- `/Users/yewang/work/FastVideo/fastvideo/distributed/utils.py` (`compute_padding_for_sp`)
- `/Users/yewang/work/FastVideo/fastvideo/attention/backends/video_sparse_attn.py` (264 lines)
- `/Users/yewang/work/FastVideo/fastvideo/attention/backends/sla.py` (558 lines, both `SLAAttentionImpl` and `SageSLAAttentionImpl`)
- `/Users/yewang/work/FastVideo/fastvideo/attention/backends/bsa_attn.py` (738 lines)
- `/Users/yewang/work/FastVideo/fastvideo/attention/backends/vmoba.py` (199 lines)
- `/Users/yewang/work/FastVideo/fastvideo-kernel/csrc/attention/st_attn_h100.cu` (STA kernel head)
- `/Users/yewang/work/FastVideo/fastvideo-kernel/csrc/turbodiffusion/quant/quant.hpp` (W8A8 INT8 with warp-reduce amax)

---
