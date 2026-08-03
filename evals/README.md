# Evals

This directory contains fast checks that make repository behavior legible to agents.

- `smoke_eval.py`: verifies the repository harness, canonical research documents, core framework modules, configs, and entry-point files exist.
- `scripts/framework_smoke.py`: performs the model-level torch-only structural gate for segmentation, depth, and normals; it is intentionally separate from formal dataset evaluation.
- `scripts/evaluate.py`: pairs prediction/GT arrays, invokes the configured task evaluator, and writes dataset/per-sample JSON plus long-form CSV. Its synthetic CLI tests validate metric wiring only; formal evidence still requires fixed splits and prediction provenance.
- `scripts/analyze_vae_reconstruction.py`: reports target-domain VAE reconstruction fidelity and saves per-task visual panels; this is a representation gate, not model-quality evidence.
- `scripts/validate_conditions.py`: compares fixed-noise outputs under task/text condition switches; output differences prove sensitivity, not semantic correctness.
- Real-model commands and their ordered stop conditions are in `docs/run-cookbook.md`; formal evidence still requires provenance-bound canonical splits.
