# Quality Gates

## Canonical Local Checks

Run these before claiming completion:

```powershell
python evals/smoke_eval.py
python -m unittest discover -s tests -p "test_*.py"
python scripts/validate_configs.py
python scripts/train.py --config configs/smoke.yaml --dry-run
```

On Linux, the same commands are used with the active `gen-perception` Python environment.

## Evidence Rules

- Include exact commands run and whether they passed.
- Include relevant logs, screenshots, traces, or changed files.
- Mark unrun checks and explain the risk.

## Eval Strategy

- Keep at least one fast smoke eval.
- Codec tests establish an upper bound before diffusion training.
- Metric tests use perfect, invalid-mask, and controlled-error examples.
- A task may enter long training only after the gate sequence documented in `docs/stage-one-plan.md` passes.
- Formal experiment metrics must be reproducible from a committed config and recorded commands.
