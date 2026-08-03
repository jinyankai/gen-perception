# Project Status

Last updated: 2026-08-03 Asia/Shanghai

## Stage

Stage: local latent-framework core complete; stage-one assets and real SD2 single-GPU integration validated

Status: real SD2 CUDA forward, static distributed launch, and eight-rank NCCL preflight are validated; the updated real backward/DDP gates fail before an optimizer step, and all eight GPUs are currently occupied

## Confirmed facts

- Host OS: Ubuntu 20.04.6 LTS, Linux 5.15.
- CPU: 2 x Intel Xeon Gold 6530, 64 physical cores / 128 threads.
- Memory: 503 GiB total, 283 GiB available at inventory time; swap was fully used.
- GPU: 8 x NVIDIA GeForce RTX 4090, 24,564 MiB each.
- Driver: NVIDIA 570.181.
- GPU topology: four NUMA-local pairs; no NVLink reported.
- GPU availability: a temporary eight-GPU window supported CUDA/NCCL/DDP gate attempts on 2026-08-03; the user subsequently reported that all eight GPUs are occupied again.
- Distributed launcher: bare `torchrun --standalone` attempts timed out for both NCCL and Gloo. Recovery uses the exact Conda Python as `python -m torch.distributed.run` with static `127.0.0.1` rendezvous and a unique master port. The latest isolated smoke exited zero and the DDP gate recorded eight successful NCCL all-reduces; the isolated log contained only seven counted final pass lines and still requires raw-log review.
- System filesystem: 878 GiB total, about 39 GiB free (96% used).
- Shared `/data`: about 3.2 TiB free, but its root is not writable by the project user.
- Project directory: `/home/jinyankai/gen-perception`.
- Python: base environment is Python 3.14.6 with no PyTorch or scientific stack.
- Environment managers: conda and uv are available; no sudo is available.
- Job/session tools: no Slurm; tmux and screen are available.
- Network: GitHub and PyPI are reachable; the default Hugging Face endpoint timed out during inventory. The required SD2 snapshot was subsequently obtained through `hf-mirror.com` and registered locally.
- Approved local roots: `/home/jinyankai/data`, `/home/jinyankai/models`, and `/home/jinyankai/outputs`; the user reported about 200 GiB free under `/home/jinyankai`.
- Python environment: `/home/jinyankai/miniconda3/envs/gen-perception`; the returned validation log reports PyTorch `2.7.0+cu128` with a CUDA 12.8 wheel.
- Data: ADEChallengeData2016 and NYUv2 labeled MAT assets are present under the approved data root and passed sample-level format/codec checks.
- Model: the local `stable-diffusion-2` snapshot loaded without network access; its manifest-resolved revision is `2511124fabf8bf30c0ca9dccd9729e7f0f2fa669` and the three selected weight files total 5,159,974,448 bytes.
- Existing code: no Git repository existed on the server before this project bootstrap.
- GitHub branch: `origin/codex/stage1-core` was pushed from the local workspace at commit `3ea54b8`.

## Implemented

- Repository operating contract and quality harness.
- Server inventory and risk triage.
- Stage-one scope and paper-to-code plan.
- Configurable HF-Mirror environment and revision-pinned model/dataset downloader.
- Durable collaboration boundary: the coding agent works locally and provides reviewed server scripts; the user runs them and returns logs.
- Segmentation, depth, and normal target codecs and protocol-aligned evaluators with local tests.
- Unified model skeleton: learned task tokens, optional text tokens, task-specific condition adapters, shared 8-channel U-Net wrapper, and cross-attention trainability policy.
- Recursive configuration inheritance, single-task configs, and a three-task shared-U-Net configuration.
- Unified framework technical design in `docs/unified-perception-framework.md`.
- Configuration-driven `TaskSpec` registry for the three codecs/evaluators and segmentation query policy.
- ADE20K metadata vocabulary loader plus a closed-set query planner whose API cannot inspect ground-truth masks.
- Shared mask-aware diffusion loss and latent sampling cores, exercised for all three tasks by a torch-only forward/backward/sample smoke.
- Optional zero-initialized, bounded, task-specific pre-VAE residual CNN with a dedicated ablation config.
- Local evidence at commit `3ea54b8`: 33 unit tests passed, six configs validated, the repository smoke/dry-run passed, and all three tasks produced finite structural forward/backward/sample signals.
- User-operated server evidence on 2026-08-03: tokenizer, CLIP text encoder, VAE, and SD2 U-Net loaded offline on CPU; image/target latents, 81-token conditioning, the project 8-channel U-Net wrapper, and a finite denoiser output were verified. See `docs/validation/stage1-assets-2026-08-03.md`.
- User-operated S003 evidence: the same real-SD2 integration path passed on GPU 0 with PyTorch `2.7.0+cu128`; the 64 px forward and VAE decode were finite. See `docs/validation/segmentation-cuda-forward-2026-08-03.md`.
- Bounded operator scripts now cover rank-level NCCL all-reduce diagnosis and a real ADE20K segmentation backward/optimizer gate without starting a formal run.
- A bounded real-component segmentation DDP gate covers `DistributedSampler`, DataLoader collation, rank-local ADE20K/SD2 encoding, NCCL preflight, DDP gradient synchronization, optimizer update, and post-step replica consistency. The updated-code server run reached eight successful all-reduces but failed before any optimizer-step event.
- The first real ADE20K backward attempt failed before model load because the official five-column whitespace `objectInfo150.txt` format was parsed with a hard-coded six-field split. The dynamic-width parser fix now passes 61 local tests and the latest server vocabulary check; the updated-code S005 run fails later for an as-yet-unidentified reason.
- The known-good single-node launcher pattern and its evidence qualification are recorded in `docs/validation/distributed-launcher-recovery-2026-08-03.md` and the server operator contract.
- Unified offline evaluation CLI: strict prediction/GT/mask pairing, configuration-driven segmentation/depth/normal evaluators, dataset aggregation, and JSON/CSV outputs; synthetic known-answer CLI tests cover all three tasks.
- ADE20K and NYUv2 Dataset/DataLoader adapters with canonical labels/splits, worker-safe HDF5 access, metric depth, depth-derived camera-space normals, resumable download/preprocessing CLIs, and separate real-sample smoke commands; local synthetic-layout and geometry tests pass.
- Reusable local-only SD2 loader, frozen Visual Latent Pathway, optional task-specific pre-VAE adapter, and explicit epsilon/scheduler compatibility gate.
- One configuration-driven real optimizer loop for segmentation, depth, normal, and round-robin multitask batches, with Marigold-style timestep-annealed multi-resolution noise, AMP, gradient accumulation, clipping, checkpoint/resume, JSONL, TensorBoard, and optional W&B logging.
- One decoded inference runner with explicit task/text condition switches, deterministic seed control, multi-sample ensembling, closed-set ADE20K query batching without GT-present-class access, paired artifacts, and task visualizations.
- VAE reconstruction-fidelity and condition-sensitivity diagnostic CLIs, three single-task plus one multitask overfit configuration, and a user-operated `docs/run-cookbook.md` for all real GPU/data gates.
- Current local working-tree gate: 61 unit tests (including one optimizer/log/checkpoint runner integration), 10 configs, bytecode compilation, harness smoke, training dry-run, and the three-task structural framework smoke pass; Ruff is not installed in the local environment.

## Not yet verified

- Exact traceback for updated-code S005/S006 failures under `/home/jinyankai/outputs/gates/regression_20260803_221718`; NCCL preflight passed, but no DDP optimizer step completed.
- Server execution of the new ADE20K/NYUv2 real-sample DataLoader smokes and full NYUv2 normal preprocessing.
- Server execution of the new reusable loader/runner, three-task VAE reconstruction reports, loss convergence, checkpoint resume, decoded predictions, and condition diagnostics; implementation and local structural tests are not substitutes for these runs.
- Real-data/pretrained-model backward passes, overfit evidence, benchmark evaluation, latency/memory measurements, or formal metrics.
- Unified evaluator execution over provenance-bound real prediction directories and canonical validation splits; only synthetic CLI cases have run.
- Independent proof that the post-download registered revision is the exact upstream revision from which every local file was originally fetched.

## Next actions

1. On 2026-08-04, refine the WK1 report and split long-form documentation while keeping `REPORT.md` as an evidence-qualified overview/index.
2. After the documentation work, use one explicitly free GPU to complete the three-task VAE reconstruction analysis in `docs/run-cookbook.md` section 2; retain `report.json`, `fidelity.md`, visualizations, runtime, memory, commit, command, and exit status.
3. Read `preflight.log` plus the S005/S006 traces from `/home/jinyankai/outputs/gates/regression_20260803_221718`; this is read-only and does not require free GPUs.
4. Fix the first single-GPU failure narrowly and rerun local checks. When a GPU is free, rerun one S005 step; only after it passes, rerun isolated NCCL and one S006 step using the recorded static rendezvous launcher.
5. Continue the cookbook's bounded overfit, resume, decoded inference, metrics, and condition diagnostics. Start round-robin joint training only after all three overfit gates show decreasing loss and coherent decoded outputs.
