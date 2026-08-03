# Blockers

## B001 - Writable project storage

- Status: RESOLVED FOR STAGE-ONE ARTIFACTS
- Evidence: the user approved `/home/jinyankai/data`, `/home/jinyankai/models`, and `/home/jinyankai/outputs`, reported about 200 GiB free, and the asset validator read the model and datasets from those roots.
- Remaining constraint: monitor free space before checkpoints, predictions, or additional model/dataset downloads.

## B002 - GPU availability and updated-code DDP regression

- Status: NCCL PREFLIGHT PASSED; DDP OPTIMIZER STEP FAILED; ALL EIGHT GPUS CURRENTLY OCCUPIED
- Evidence: the latest user-operated regression reported `NCCL_STATUS=0`; the DDP gate emitted eight `nccl_all_reduce_passed` events, then exited with `S006_STATUS=1`, zero optimizer-step events, and zero final-pass events. The isolated smoke contained seven counted final pass lines, so its raw output still needs review.
- Impact: launcher/NCCL is no longer the primary suspected blocker. Updated real training fails after collective initialization and before completion of the first optimizer step. No GPU retry is allowed while the devices are occupied.
- Minimum resolution: retrieve the existing S005/S006 trace without using GPUs, fix and locally test the first failure, then rerun one single-GPU step before a one-step DDP gate in the next free window.

## B003 - Default Hugging Face endpoint unreachable from server

- Status: RESOLVED FOR THE REQUIRED SD2 SNAPSHOT; DEFAULT ENDPOINT STILL UNVERIFIED
- Evidence: HTTPS to `huggingface.co` timed out during inventory, but the required `sd2-community/stable-diffusion-2` files were obtained through `hf-mirror.com`, registered locally, and loaded offline by the validator.
- Provenance qualification: revision `2511124fabf8bf30c0ca9dccd9729e7f0f2fa669` was registered after an external download, so the local artifact is operationally validated but its original upstream revision is not independently proven by this run.
- Remaining constraint: use the mirror/offline manifest workflow for further Hub artifacts and retain license/source records.

## B004 - Required datasets located

- Status: ADE20K PARSER VERIFIED ON SERVER; UPDATED REAL BACKWARD FAILED LATER
- Evidence: ADEChallengeData2016 image/mask samples and the 1,449-sample NYUv2 labeled MAT datasets passed structural and codec checks. The dynamic-width ADE20K parser preserves multiword names, passes local regression tests, and returned `VOCAB_STATUS=0` on the server. The subsequent updated-code S005 run returned status 1 for an unknown later-stage error.
- Remaining constraint: retrieve the S005 traceback, resolve the later-stage failure, and run the NYUv2 real-sample/preprocessing smokes before training or benchmark claims.
