# Project Status

Last updated: 2026-08-03 Asia/Shanghai

## Stage

Stage: local latent-framework core complete; stage-one assets and real SD2 single-GPU integration validated

Status: real SD2 CUDA forward passed on GPU 0; distributed runtime and training gates pending

## Confirmed facts

- Host OS: Ubuntu 20.04.6 LTS, Linux 5.15.
- CPU: 2 x Intel Xeon Gold 6530, 64 physical cores / 128 threads.
- Memory: 503 GiB total, 283 GiB available at inventory time; swap was fully used.
- GPU: 8 x NVIDIA GeForce RTX 4090, 24,564 MiB each.
- Driver: NVIDIA 570.181.
- GPU topology: four NUMA-local pairs; no NVLink reported.
- GPU availability: all eight GPUs were occupied at inventory time. On 2026-08-03 the user reported a new eight-GPU availability window; GPU 0 completed the real-SD2 CUDA forward, while the initial eight-rank NCCL smoke exceeded 60 seconds.
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
- Unified offline evaluation CLI: strict prediction/GT/mask pairing, configuration-driven segmentation/depth/normal evaluators, dataset aggregation, and JSON/CSV outputs; synthetic known-answer CLI tests cover all three tasks.
- ADE20K and NYUv2 Dataset/DataLoader adapters with canonical labels/splits, worker-safe HDF5 access, metric depth, depth-derived camera-space normals, resumable download/preprocessing CLIs, and separate real-sample smoke commands; local synthetic-layout and geometry tests pass.

## Not yet verified

- Single-node NCCL communication across the reported eight-GPU window; the initial smoke exceeded 60 seconds without rank-level evidence.
- Server execution of the new ADE20K/NYUv2 real-sample DataLoader smokes and full NYUv2 normal preprocessing.
- Reusable training-path integration of the validated real SD2 loader; the recorded check is an asset/integration validator, not a training run.
- Real-data/pretrained-model backward passes, decoded predictions, benchmark evaluation, latency/memory measurements, or formal metrics.
- Unified evaluator execution over provenance-bound real prediction directories and canonical validation splits; only synthetic CLI cases have run.
- Independent proof that the post-download registered revision is the exact upstream revision from which every local file was originally fetched.

## Next actions

1. Promote the validated SD2 loading sequence into the reusable model/training boundary with an explicit immutable-revision check.
2. Run the three real-sample data smokes and NYUv2 preprocessing on the server, then connect the resulting codec batches through the optional target adapter, frozen VAE, latent trainer/sampler, decode, and evaluator.
3. Run the instrumented two-rank/eight-rank NCCL smoke and the single-GPU real ADE20K one-batch backward gate without disturbing occupied devices.
4. Add annealed multi-scale noise without task branches in the shared loop.
5. Complete 10-step loss, tiny-overfit, checkpoint/resume, decoded inference, metric-known-answer, visualization, and provenance gates before formal runs.
