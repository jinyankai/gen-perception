# Evals

This directory contains fast checks that make repository behavior legible to agents.

- `smoke_eval.py`: verifies the repository harness, canonical research documents, core framework modules, configs, and entry-point files exist.
- `scripts/framework_smoke.py`: performs the model-level torch-only structural gate for segmentation, depth, and normals; it is intentionally separate from formal dataset evaluation.
- `scripts/evaluate.py`: pairs prediction/GT arrays, invokes the configured task evaluator, and writes dataset/per-sample JSON plus long-form CSV. Its synthetic CLI tests validate metric wiring only; formal evidence still requires fixed splits and prediction provenance.
- Add task-specific real-model and dataset evals only after their corresponding stage gates pass.
