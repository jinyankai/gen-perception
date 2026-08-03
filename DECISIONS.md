# Decision Log

## D001 - Stage-one task set

- Date: 2026-08-03
- Decision: semantic segmentation, monocular depth, and surface normals.
- Reason: required three-task coverage while NYUv2 lets depth and normals share data and geometric preprocessing.

## D002 - Shared framework boundary

- Date: 2026-08-03
- Decision: task differences are isolated to target codecs, dataset adapters, evaluators, conditions, and visualizers.
- Reason: the evaluation requires a real unified framework rather than three adjacent repositories.

## D003 - Python and CUDA package baseline

- Date: 2026-08-03
- Decision: use Python 3.11 and an official PyTorch CUDA 12.8 wheel; do not use the server's Python 3.14 base environment.
- Reason: Python 3.11 is widely supported by research dependencies; the official PyTorch selector supports CUDA 12.8, while Diffusers 0.39 requires PyTorch 2.6+.

## D004 - Offline-first experiment records

- Date: 2026-08-03
- Decision: local JSON/JSONL and text logs are mandatory; W&B is optional and disabled by default.
- Reason: formal evidence must not depend on external service availability.

## D005 - No data or weight artifacts in Git

- Date: 2026-08-03
- Decision: datasets, model caches, checkpoints, predictions, and logs are Git-ignored; Git stores configs, commands, summaries, and retrieval instructions.
- Reason: artifact size, privacy, reproducibility, and credential safety.
