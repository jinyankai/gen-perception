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
- Keep datasets, caches, checkpoints, and outputs off the nearly full system filesystem once a writable high-capacity project path is approved.
- Destructive or non-recoverable operations require explicit user authorization and a verified target.

## Current remote handoff state

- The project directory is expected to exist.
- A prior interrupted attempt initialized `.git`; the subsequent fetch timed out from the client side. Repository synchronization and branch state are therefore unknown.
- Hugging Face mirror connectivity from the server has not been verified.
- Writable high-capacity storage and a free GPU window remain unresolved.
- Before the first checkout, ask the user for read-only `git status`, remote, and branch output. Once the working tree contains this branch, use `scripts/operator/sync_server_repo.sh`, then `scripts/operator/probe_hf_mirror.sh`, and treat the returned logs as evidence.
