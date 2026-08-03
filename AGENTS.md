# Agent Operating Contract

This repository is optimized for coding-agent work. Keep this file short: it is a map to deeper project knowledge, not the full manual.

## Start Here

- Read `docs/agent-harness/index.md` before large or unfamiliar changes.
- Read `STATUS.md`, `BLOCKERS.md`, `DECISIONS.md`, and `EXPERIMENTS.md` before changing research scope or launching an experiment.
- Prefer repository-local docs, tests, evals, logs, and configs over chat memory.
- Preserve existing user changes. Do not revert unrelated work.
- Before editing, identify the relevant module, owner, checks, and expected evidence.

## Research Invariants

- The required stage-one tasks are semantic segmentation, monocular depth, and surface normals.
- Task differences belong in codecs, dataset adapters, evaluators, conditions, and visualizers. Do not fork the training framework per task.
- Do not call a result formal unless it has a commit, frozen config, dataset split, seed, checkpoint, raw log, commands, GPU count, and runtime.
- Never delete failed experiments. Record the failure and status in `EXPERIMENTS.md`.
- No multi-GPU or long training before codec round-trip, one-batch forward/backward, short loss, overfit, inference, metric, and visualization gates pass.

## Canonical Checks

- Run `python evals/smoke_eval.py` for harness sanity.
- Run `python -m unittest discover -s tests -p "test_*.py"` for baseline tests.
- Run `python scripts/validate_configs.py` before committing configuration changes.
- Run `python scripts/train.py --config configs/smoke.yaml --dry-run` before changing training infrastructure.
- Run `python scripts/framework_smoke.py --config configs/multitask/stage1_shared_unet.yaml` after changing shared model, task registry, loss, or sampling code.

## Tool and MCP Policy

- Follow `docs/agent-harness/tools.md`.
- Follow `docs/agent-harness/server-operator-contract.md` for every server-side action.
- Do not connect to, control, or run commands on the research server. Prepare reviewed commands or scripts for the user to run, then continue from the returned logs.
- Prefer Markdown or LaTeX for routine written deliverables to minimize dependence on Word.
- When a task requires Microsoft Word UI operations, provide the exact steps for the user to perform locally instead of controlling Word directly. This preference was explicitly stated by the user on 2026-08-03.
- Never store server credentials in the repository, agent memory, commands, or logs.
- Do not use destructive commands or external network tools unless the task and approvals require them.
- Record important tool outputs in the final answer: tests, evals, CI, logs, screenshots, traces, or exact files.

## Review Policy

- Use `docs/agent-harness/review.md` before proposing completion.
- Claims about behavior must be backed by evidence.
- If a check cannot run, state why and identify the residual risk.
