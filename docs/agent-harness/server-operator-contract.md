# User-Operated Server Contract

Generated: 2026-08-03 Asia/Shanghai

## Stable collaboration preference

The user explicitly requested that the coding agent not spend time directly operating the research server. This is a durable project preference:

- The agent develops, reviews, and tests code in the local repository.
- For every server-side action, the agent provides an exact command or a checked-in script for the user to run.
- The user runs the command and returns its output. The agent diagnoses that evidence and prepares the next action.
- The agent never SSHes into, controls, or directly executes commands on the server.

## Required handoff format

Every server request must state:

1. The working directory and exact command.
2. Whether it is read-only or what it will change.
3. Whether it uses network, disk, CPU, RAM, or GPUs.
4. The expected success output and which output the user should return.
5. A recovery or stop condition when failure could leave partial state.

Prefer idempotent scripts under `scripts/operator/`. Do not claim a remote action succeeded without user-returned logs.

## Safety boundaries

- Never persist passwords, access tokens, private keys, host details, or personal account identifiers in Git, agent memory, scripts, or logs.
- Never request `sudo`; the account does not have it.
- Never stop, signal, or interfere with another user's GPU processes.
- Keep datasets, models, checkpoints, and outputs under the approved `/home/jinyankai/data`, `/home/jinyankai/models`, and `/home/jinyankai/outputs` roots; check free space before material growth.
- Destructive or non-recoverable operations require explicit user authorization and a verified target.

## Current remote handoff state

- The project directory `/home/jinyankai/gen-perception` was confirmed by the returned inventory.
- The isolated environment is `/home/jinyankai/miniconda3/envs/gen-perception`.
- The approved asset/output roots are under `/home/jinyankai`; the user reported about 200 GiB free.
- The local workspace pushed `origin/codex/stage1-core` at commit `3ea54b8`. The server can run project scripts, but its exact checked-out commit remains unbound in the returned evidence.
- The required `sd2-community/stable-diffusion-2` snapshot was obtained through `hf-mirror.com`, registered locally, and loaded offline. Default `huggingface.co` connectivity is still not established.
- The S002 user-operated validator passed on CPU for the SD2 components, ADE20K/NYUv2 assets, codecs, and project 8-channel denoiser forward; see `../validation/stage1-assets-2026-08-03.md`.
- The user subsequently reported an eight-GPU availability window. S003 passed the same real-SD2 integration forward on GPU 0; see `../validation/segmentation-cuda-forward-2026-08-03.md`.
- The initial eight-rank NCCL command exceeded 60 seconds without rank-level output. Distributed execution remains unresolved; rerun the bounded checked-in smoke at two ranks and then eight ranks before any DDP launch.
