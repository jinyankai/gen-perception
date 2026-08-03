# Stage-One Asset and Real SD2 CPU Validation

Date: 2026-08-03 Asia/Shanghai

Experiment registry ID: `S002`

Result: `STAGE1_ASSET_VALIDATION_PASSED`

Evidence origin: user-returned output from `scripts/operator/validate_stage1_assets.py` on the project server. The coding agent did not execute commands on the server. The validator reported `formal_experiment: false` and `network_used: false`.

## Environment and paths

- Environment prefix: `/home/jinyankai/miniconda3/envs/gen-perception`
- Project: `/home/jinyankai/gen-perception`
- Config: `/home/jinyankai/gen-perception/configs/multitask/stage1_shared_unet.yaml`
- Model: `/home/jinyankai/models/stable-diffusion-2`
- ADE20K: `/home/jinyankai/data/ADEChallengeData2016`
- NYUv2: `/home/jinyankai/data/nyuv2`
- Runtime reported by the validator: PyTorch `2.7.0+cu128`, CUDA wheel version `12.8`, execution device `cpu`

## Model artifact evidence

- Manifest-resolved revision: `2511124fabf8bf30c0ca9dccd9729e7f0f2fa669`
- Selected files: 5,159,974,448 bytes total

| File | Bytes |
| --- | ---: |
| `text_encoder/model.safetensors` | 1,361,597,018 |
| `unet/diffusion_pytorch_model.safetensors` | 3,463,726,498 |
| `vae/diffusion_pytorch_model.safetensors` | 334,643,276 |

The tokenizer and CLIP text encoder produced token shape `[1,77]` and hidden shape `[1,77,1024]`. The VAE produced latent shape `[1,4,8,8]` and decoded shape `[1,3,64,64]`. After the project wrapper expanded the SD2 U-Net input to eight channels, the denoiser accepted image and target latents, used conditioning shape `[1,81,1024]`, and returned a finite tensor of shape `[1,4,8,8]`. The configured trainable scope was `cross_attention`, with 50,375,680 trainable parameters.

## Dataset evidence

ADE20K sample `ADE_train_00000001` loaded as an RGB image of size 683 by 512. Its mask range was 0-150, metadata exposed 150 classes, and `SegmentationBinaryMaskCodec` passed its round-trip check.

The NYUv2 labeled MAT file was 2,972,037,809 bytes and exposed these datasets:

| Dataset | Shape |
| --- | --- |
| `images` | `[1449,3,640,480]` |
| `depths` | `[1449,640,480]` |
| `labels` | `[1449,640,480]` |
| `rawDepths` | `[1449,640,480]` |

A sample image/depth crop produced shapes `[1,3,16,16]` and `[1,16,16]`; all 256 tested depth pixels were valid and the finite sample range was 2.7170395851135254-2.759791851043701.

## Provenance qualification

The model source was user-reported as `sd2-community/stable-diffusion-2` through `https://hf-mirror.com`. Because the manifest/revision was registered after the files had already been downloaded, this record proves that the local files load and work together, but does not independently prove that every file was originally fetched from that exact upstream revision.

## Qualified conclusion

The required local SD2, ADE20K, and NYUv2 assets are readable by the project, and the real SD2 CPU component/shape integration path works without network access. This is an asset and integration smoke test only.

The following remain unverified: CUDA execution and memory use, real-data backward/training, canonical split/DataLoader integration, surface-normal target derivation, real task-target VAE reconstruction, decoded task predictions, formal metrics, latency, checkpoints/resume, and baseline comparisons.
