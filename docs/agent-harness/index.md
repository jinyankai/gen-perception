# Agent Harness Index

This directory is the system of record for repository behavior that agents need.

## Documents

- `architecture.md`: codebase map and ownership boundaries.
- `quality.md`: canonical local checks, CI, evals, and evidence rules.
- `tools.md`: MCP/tool constraints, approvals, secrets, and destructive-command policy.
- `server-operator-contract.md`: durable user-operated server workflow and current remote handoff state.
- `review.md`: self-review, agent-review, human-review, and PR response loop.
- `../reproduction-plan.md`: paper-to-code map, protocol targets, and staged run plan.
- `../stage-one-plan.md`: compressed 3-4 week milestone plan and acceptance gates.
- `../unified-perception-framework.md`: task-token, shared U-Net, task-adapter architecture and engineering configuration.
- `../huggingface-mirror.md`: mirror, cache, revision, license, and secret-handling workflow.
- `../target-encoding-strategies.md`: deterministic three-task target representations and VAE fidelity metrics.
- `../real-training-and-inference.md`: reusable SD2, training, checkpoint, logging, inference, and condition-ablation boundaries.
- `../run-cookbook.md`: exact user-operated data, VAE, overfit, resume, inference, evaluation, and condition gates.
- `../../STATUS.md`: confirmed environment facts, implemented boundary, unresolved verification, and next actions.
- `../../EXPERIMENTS.md`: smoke and formal experiment registry with commit/config binding.
- `../../REPORT.md`: stage report scaffold and evidence-qualified implementation snapshot.
- `../validation/stage1-assets-2026-08-03.md`: user-returned server evidence for the stage-one asset and real SD2 CPU integration smoke.
- `../validation/distributed-launcher-recovery-2026-08-03.md`: failure signature and known-good static loopback launcher pattern for this server.

## Maintenance

- Update this directory when the repo gains a new workflow, test gate, eval, tool, or architectural invariant.
- Promote repeated review comments into mechanical checks when practical.
- Keep `AGENTS.md` as a short routing file that points here.
