# Project Status

Last updated: 2026-08-03 Asia/Shanghai

## Stage

Stage: local framework bootstrap and user-operated server handoff

Status: partially complete

## Confirmed facts

- Host OS: Ubuntu 20.04.6 LTS, Linux 5.15.
- CPU: 2 x Intel Xeon Gold 6530, 64 physical cores / 128 threads.
- Memory: 503 GiB total, 283 GiB available at inventory time; swap was fully used.
- GPU: 8 x NVIDIA GeForce RTX 4090, 24,564 MiB each.
- Driver: NVIDIA 570.181.
- GPU topology: four NUMA-local pairs; no NVLink reported.
- GPU availability: all eight GPUs were occupied at inventory time. GPUs 0-3 each used about 22 GiB; GPUs 4-7 each used about 16-17 GiB.
- System filesystem: 878 GiB total, about 39 GiB free (96% used).
- Shared `/data`: about 3.2 TiB free, but its root is not writable by the project user.
- Project directory: `/home/jinyankai/gen-perception`.
- Python: base environment is Python 3.14.6 with no PyTorch or scientific stack.
- Environment managers: conda and uv are available; no sudo is available.
- Job/session tools: no Slurm; tmux and screen are available.
- Network: GitHub and PyPI reachable; Hugging Face timed out.
- Data: ADE20K/PASCAL Context and NYUv2 were not found in the user home inventory.
- Existing code: no Git repository existed on the server before this project bootstrap.

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

## Not yet verified

- PyTorch CUDA runtime compatibility on a free GPU.
- Writable high-capacity data/model/output paths.
- Dataset versions and canonical splits.
- Stable Diffusion/Marigold/DiGSeg checkpoint availability.
- Remote repository synchronization. A previous session initialized `.git`, but its fetch timed out from the client side, so the resulting branch state is unknown.
- HF-Mirror connectivity from the server.
- Training, inference, evaluation, or formal metrics.
- Real Diffusers SD2/VAE/CLIP forward-backward integration; the current structural smoke uses a torch-only tiny U-Net and scheduler.

## Next actions

1. Implement the real SD2 component loader, annealed multi-scale noise, and mask-aware diffusion loss behind the unified model boundary.
2. Add a single unified trainer/pipeline and complete three-task tiny-overfit gates before formal runs.
3. Ask the user for read-only Git status, remote, and branch output; then provide the safe first-checkout command.
4. After checkout, ask the user to run `scripts/operator/sync_server_repo.sh` and `scripts/operator/probe_hf_mirror.sh`, returning both outputs.
5. Obtain an approved writable high-capacity path and GPU window before any pretrained-component or training smoke test.
