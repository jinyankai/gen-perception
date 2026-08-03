# Single-Node Distributed Launcher Recovery

Date: 2026-08-03 Asia/Shanghai

Experiment registry IDs: `S004`, `S004A`

## Failure signature

- The initial eight-rank NCCL command used a PATH-resolved `torchrun --standalone` launch and exceeded 60 seconds without rank-level success output.
- A bounded two-rank NCCL diagnostic ended with status 137 after the timeout wrapper escalated to `SIGKILL`.
- A CPU/Gloo control using the same standalone launch style ended with timeout status 124.
- At diagnosis time the server reported 503 GiB total RAM and 346 GiB available. The returned kernel OOM entries were dated 2026-05-02 and 2026-05-03, not the 2026-08-03 test window.

These observations localized the failure above NCCL: the standalone launcher/rendezvous path was the shared factor.

## Successful recovery pattern

The user confirmed success after both sources of launcher ambiguity were removed:

1. invoke `/home/jinyankai/miniconda3/envs/gen-perception/bin/python -m torch.distributed.run` instead of a PATH-resolved bare `torchrun`;
2. replace `--standalone` with static single-node rendezvous using `--master-addr=127.0.0.1`, `--node-rank=0`, and a unique master port;
3. set `GLOO_SOCKET_IFNAME=lo` for the Gloo control.

The exact launcher and Gloo commands are preserved in `docs/agent-harness/server-operator-contract.md`.

## Evidence qualification

Evidence origin is the user's statement that the provided static launcher/Gloo method succeeded. The complete rank logs and exact exit-status lines were not returned, so this record does not invent them. It establishes the operator workaround for local rendezvous, not NCCL all-reduce, multi-GPU model execution, DDP training, or performance.

## Required next distributed gate

Use the same exact interpreter and static loopback rendezvous pattern with `scripts/operator/distributed_cuda_smoke.py`. First run two allocated GPUs, then all eight allocated GPUs. Preserve each rank's JSON result and the wrapper exit status before declaring DDP available.
