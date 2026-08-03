# Blockers

## B001 - Writable project storage

- Status: RESOLVED FOR STAGE-ONE ARTIFACTS
- Evidence: the user approved `/home/jinyankai/data`, `/home/jinyankai/models`, and `/home/jinyankai/outputs`, reported about 200 GiB free, and the asset validator read the model and datasets from those roots.
- Remaining constraint: monitor free space before checkpoints, predictions, or additional model/dataset downloads.

## B002 - All GPUs occupied at inventory time

- Status: RESOLVED FOR SINGLE-GPU CUDA; MULTI-GPU NCCL PENDING
- Evidence: the user reported an eight-GPU availability window and returned a successful GPU 0 real-SD2 CUDA forward (`S003`). The initial eight-rank NCCL smoke exceeded 60 seconds and returned no rank-level log, so distributed execution remains unverified.
- Impact: bounded single-GPU training gates may proceed. Multi-GPU or long training remains blocked until an instrumented NCCL all-reduce passes and the remaining training gates are complete.
- Minimum resolution for multi-GPU: run `scripts/operator/distributed_cuda_smoke.py` with two ranks and then eight ranks, retaining the rank-level log.

## B003 - Default Hugging Face endpoint unreachable from server

- Status: RESOLVED FOR THE REQUIRED SD2 SNAPSHOT; DEFAULT ENDPOINT STILL UNVERIFIED
- Evidence: HTTPS to `huggingface.co` timed out during inventory, but the required `sd2-community/stable-diffusion-2` files were obtained through `hf-mirror.com`, registered locally, and loaded offline by the validator.
- Provenance qualification: revision `2511124fabf8bf30c0ca9dccd9729e7f0f2fa669` was registered after an external download, so the local artifact is operationally validated but its original upstream revision is not independently proven by this run.
- Remaining constraint: use the mirror/offline manifest workflow for further Hub artifacts and retain license/source records.

## B004 - Required datasets located

- Status: IMPLEMENTED LOCALLY; SERVER DATA-PIPELINE SMOKE PENDING
- Evidence: ADEChallengeData2016 image/mask samples and the 1,449-sample NYUv2 labeled MAT datasets passed structural and codec checks under `/home/jinyankai/data`.
- Remaining constraint: run the new canonical split/DataLoader and derived-normal preprocessing smokes against the server assets before training or benchmark claims.
