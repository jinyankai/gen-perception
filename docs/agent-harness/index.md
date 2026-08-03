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
- `../huggingface-mirror.md`: mirror, cache, revision, license, and secret-handling workflow.

## Maintenance

- Update this directory when the repo gains a new workflow, test gate, eval, tool, or architectural invariant.
- Promote repeated review comments into mechanical checks when practical.
- Keep `AGENTS.md` as a short routing file that points here.
