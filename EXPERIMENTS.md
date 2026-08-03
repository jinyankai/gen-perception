# Experiment Registry

Allowed statuses: `PLANNED`, `SMOKE_TEST`, `RUNNING`, `FAILED`, `COMPLETED`, `INVALID`.

| ID | Task | Config | Commit | Device | Status | Smoke signal | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S000 | infrastructure | `configs/smoke.yaml` | `3ea54b8` | CPU | SMOKE_TEST | repository smoke, six-config validation, dry-run passed | Harness and configuration contracts are locally executable. |
| S001 | segmentation/depth/normal structural sharing | `configs/multitask/stage1_shared_unet.yaml` | `3ea54b8` | CPU | SMOKE_TEST | 33 tests passed; finite loss, gradient, and two-step sample per task | Shared interfaces work with tiny injected components; this is not a real SD2 or benchmark result. |
| S002 | stage-one assets and real SD2 integration | `configs/multitask/stage1_shared_unet.yaml` | working tree (commit pending) | CPU | SMOKE_TEST | offline model load; ADE20K/NYUv2 sample checks; finite 8-channel denoiser forward | Required local assets and the real SD2 CPU forward path are usable; no training, GPU, or benchmark result is claimed. |
| S003 | real SD2 CUDA forward | `configs/multitask/stage1_shared_unet.yaml` | server commit unreported | 1 x RTX 4090 (GPU 0) | SMOKE_TEST | user-returned `STAGE1_ASSET_VALIDATION_PASSED`; CUDA 12.8; finite `[1,4,8,8]` output | Single-GPU CUDA component compatibility is verified at 64 px; backward, optimizer, full resolution, and metrics remain unverified. |
| S004 | single-node NCCL all-reduce | N/A | server commit unreported | 8 x RTX 4090 | FAILED | user reported the initial command exceeded 60 seconds; no rank-level output was returned | Distributed runtime is unresolved; rerun the bounded instrumented two-rank and eight-rank smokes before DDP. |
| S005 | ADE20K real-component training gate | `configs/segmentation/ade20k.yaml` | working tree (commit pending) | 1 x RTX 4090 | PLANNED | real sample, binary query mask, VAE latents, backward, finite gradient, optimizer update | Run one step at 256 px, then ten fixed-noise steps only after the one-step gate passes. |

S002 and S003 are backed by user-returned server logs recorded under `docs/validation/`. S004 records an incomplete/failed smoke rather than a successful distributed result. No formal model experiments have been run.
