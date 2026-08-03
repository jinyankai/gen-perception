# Real SD2 Single-GPU CUDA Forward Validation

Date: 2026-08-03 Asia/Shanghai

Experiment registry ID: `S003`

Result: `STAGE1_ASSET_VALIDATION_PASSED`

Evidence origin: user-returned output from `scripts/operator/validate_stage1_assets.py --device cuda`. The coding agent did not execute the command on the server. The validator reported `formal_experiment: false` and `network_used: false`.

## Confirmed runtime and artifacts

- Runtime: PyTorch `2.7.0+cu128`, CUDA wheel `12.8`, execution device `cuda` (GPU 0 selected by the operator command).
- Model manifest revision: `2511124fabf8bf30c0ca9dccd9729e7f0f2fa669`.
- Selected local model files: 5,159,974,448 bytes.
- ADE20K sample and `objectInfo150.txt`: readable; 150 metadata classes; binary-mask codec round trip passed.
- NYUv2 labeled MAT: readable; 1,449-sample datasets and depth codec sample passed.

## CUDA forward signal

- Token shape: `[1,77]`.
- Text hidden shape: `[1,77,1024]`.
- VAE latent shape: `[1,4,8,8]` from a 64 px input.
- Expanded U-Net input channels: 8.
- Project denoiser output shape: `[1,4,8,8]`.
- Combined conditioning shape: `[1,81,1024]`.
- Trainable policy: cross-attention, 50,375,680 U-Net parameters.
- Output and VAE decode: finite.

## Qualification

This establishes local-weight compatibility and a real single-GPU CUDA forward only. It does not establish a real-data backward pass, optimizer update, 512 px memory requirement, checkpoint/resume, decoded segmentation prediction, metric result, multi-GPU communication, or formal training result.

The separately attempted eight-rank NCCL smoke exceeded 60 seconds and returned no rank-level output. That attempt is recorded as `S004` and must not be treated as a distributed-runtime pass.
