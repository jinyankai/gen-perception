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
| DiGSeg paper/project page | available | Project page describes image/mask latents, text pathway, and noise-MSE training. Public code was not located in the initial search. |
| DiGSeg local paper PDF | available outside repository | Must not be copied into Git without license review. |
| Marigold official code | available on GitHub | Provides depth and normal reference implementation and checkpoints. |
| Stable Diffusion VAE/CLIP/U-Net weights | blocked | Hugging Face unreachable from server; exact checkpoint source pending. |
| ADE20K/PASCAL Context | not located | Dataset path/permission pending. |
| NYUv2 | not located | Dataset path/permission pending. |
| Discriminative baselines | planned | Prefer official pretrained inference, not retraining. |

## Paper-to-code map

| Paper claim | Planned code location | Config/runtime evidence | Status |
| --- | --- | --- | --- |
| Encode image and target into latent space | `perception_diffusion/models/vae.py`, `codecs/` | reconstruction metrics and images | planned |
| Add noise only to target latent | `models/scheduler.py` | unit test and sampled timestep log | planned |
| Condition U-Net on image, timestep, and task/text | `models/conditioner.py`, `models/denoiser.py` | shape tests and condition swap ablation | planned |
| Decode task-consistent latent | `models/unified_model.py`, codecs | inference outputs and task metrics | planned |
| Multi-step sampling and ensembling | scheduler/inference | step-latency and seed ablation | planned |
| Three tasks share the backbone | unified model/configs | task switch and shared checkpoint evidence | planned |

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
3. Pretrained VAE/CLIP/U-Net offline load and shape check.
4. Per-task synthetic one-batch forward/backward.
5. Per-task small real-data overfit.
6. Per-task formal run only after all gates pass.
7. Baseline inference in isolated environments with protocol-aligned output conversion.

## Known gaps and risks

- DiGSeg public code was not found in the initial official project-page/GitHub search; implementation details beyond the paper remain unverified.
- Hugging Face access, checkpoint paths, datasets, writable high-capacity storage, and GPU allocation are unresolved.
- The server has no local CUDA toolkit (`nvcc`), so custom CUDA extensions should be avoided initially; PyTorch wheels provide the runtime needed for standard operators.

## Claims that remain unverified

- No model component has loaded on the server.
- No codec, training, inference, evaluation, baseline, or formal benchmark has run yet.
- No performance comparison or reproduction success is claimed.
