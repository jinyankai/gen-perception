# Reproduction Plan

## Reproduction target

The target is not an unqualified reproduction of every published headline result. The target is a traceable three-task research baseline that tests the DiGSeg-style claim that a pretrained diffusion backbone can act as a task-conditioned perception interface.

Formal targets:

- segmentation: ADE20K-150 or, if unavailable, PASCAL Context; mIoU, pixel accuracy, per-class IoU;
- depth: NYUv2; AbsRel, delta1, optional RMSE, before and after affine alignment;
- normals: NYUv2; mean/median angular error and 11.25/22.5/30 degree accuracy.

## Artifact inventory

| Artifact | Status | Notes |
| --- | --- | --- |
| DiGSeg paper/project page | available | Project page describes image/mask latents, text pathway, and noise-MSE training. |
| DiGSeg local paper PDF | available outside repository | Must not be copied into Git without license review. |
| User-supplied segmentation reference code | available outside target repository | `MarigoldSemanticSegmentation/` was inspected read-only as an engineering reference; its presence does not establish canonical upstream provenance. |
| Marigold official code | available on GitHub | Provides depth and normal reference implementation and checkpoints. |
| Stable Diffusion VAE/CLIP/U-Net weights | available and CPU-load validated | Local snapshot under `/home/jinyankai/models/stable-diffusion-2`; user-reported source `sd2-community/stable-diffusion-2` via `hf-mirror.com`; manifest-resolved revision `2511124fabf8bf30c0ca9dccd9729e7f0f2fa669`. Revision registration occurred after download and is therefore qualified provenance. |
| ADE20K | available; sample validated | ADEChallengeData2016 under `/home/jinyankai/data`; RGB image, 0-150 mask range, 150-class metadata, and codec round trip passed. Canonical split/DataLoader integration remains. |
| NYUv2 | available; sample validated | Labeled MAT under `/home/jinyankai/data/nyuv2`; images, depths, labels, and rawDepths have 1,449 entries and the depth codec passed. Normal-target preprocessing and split integration remain. |
| Discriminative baselines | planned | Prefer official pretrained inference, not retraining. |

## Paper-to-code map

| Paper claim | Current code location | Config/runtime evidence | Status |
| --- | --- | --- | --- |
| Encode targets into a VAE-compatible image domain | `perception_diffusion/codecs/`, `models/target_adapters.py` | range/round-trip unit tests; real VAE image encode/decode shape smoke | codec domain and VAE component are individually validated; real task-target VAE reconstruction remains |
| Add noise only to target latent | `training/unified_trainer.py` | scheduler-compatible structural test | basic epsilon objective implemented; annealed multi-scale noise pending |
| Condition U-Net on image, timestep, and task/text | `models/conditioning.py`, `models/adapters.py`, `models/unified_denoiser.py` | shape/gradient tests plus real SD2 CPU forward with `[1,81,1024]` conditioning | real component and 8-channel wrapper integration validated on CPU; training and CUDA pending |
| Decode task-consistent latent | `codecs/`, `evaluation/` | codec/evaluator unit tests | task protocols implemented; VAE-to-metric pipeline pending |
| Multi-step latent sampling | `inference/latent_sampler.py` | finite two-step structural sampling | core implemented; real scheduler, VAE decode, and ensemble studies pending |
| Three tasks share the backbone | `task_specs.py`, `models/builder.py`, multitask config | task switch and common denoiser evidence | structurally verified; shared trained checkpoint pending |
| Generate segmentation queries without GT leakage | `data/segmentation_vocabulary.py`, `inference/segmentation_queries.py` | ADE20K taxonomy/query tests | closed-set policy implemented; dataset evaluation pending |

## Environment and config checks

- Python 3.11 isolated environment.
- Official PyTorch CUDA 12.8 wheel and CUDA availability smoke test.
- Diffusers 0.39 stable API baseline.
- Fixed seeds and logged deterministic flags.
- AMP, gradient accumulation, DDP, checkpoint/resume, and offline logs.
- Config and data split frozen in every experiment directory.

## Run plan

1. CPU-only codec/evaluator/config tests.
2. CUDA tensor and one-GPU memory smoke test when a GPU is free.
3. Pretrained VAE/CLIP/U-Net offline load and shape check. **Completed on CPU in S002.**
4. Per-task synthetic one-batch forward/backward with tiny injected components. **Completed locally.**
5. Per-task small real-data overfit.
6. Per-task formal run only after all gates pass.
7. Baseline inference in isolated environments with protocol-aligned output conversion.

## Known gaps and risks

- The user-supplied segmentation repository is a reference snapshot, not independently verified canonical upstream code; paper-to-code details beyond inspected behavior remain provisional.
- Required SD2/ADE20K/NYUv2 assets and writable project roots are available. Further Hub artifacts must use the mirror/offline workflow; canonical splits, full dataset adapters, normals preprocessing, and GPU allocation remain unresolved.
- The server has no local CUDA toolkit (`nvcc`), so custom CUDA extensions should be avoided initially; PyTorch wheels provide the runtime needed for standard operators.

## Claims that remain unverified

- Codec/evaluator unit tests and a tiny latent-level three-task train/sample structural smoke have run locally.
- Real SD2 tokenizer/text encoder/VAE/U-Net loading and a finite project-wrapped denoiser forward have run offline on CPU; ADE20K/NYUv2 sample assets and codecs were checked.
- No real-data training/backward pass, full dataset adapter, decoded task prediction, CUDA execution, baseline, or formal benchmark has run.
- No performance comparison or reproduction success is claimed.
