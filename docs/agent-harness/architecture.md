# Architecture Map

## Domains

- `perception_diffusion/codecs`: reversible or approximately reversible task target representations.
- `perception_diffusion/data`: adapters that map datasets to the unified batch schema.
- `perception_diffusion/evaluation`: protocol-aligned metrics for segmentation, depth, and normals.
- `perception_diffusion/models`: VAE, task conditioner, denoiser, scheduler, and unified model boundaries.
- `perception_diffusion/visualization`: task-aware qualitative evidence.
- `perception_diffusion/utils`: config, experiment provenance, seed, logging, and validation utilities.
- `scripts`: user-facing data preparation, training, inference, evaluation, and reproduction entry points.
- `configs`: immutable base, task, baseline, smoke, and experiment configurations.
- `tests` and `evals`: mechanical checks and research smoke gates.
- `outputs`: generated experiment artifacts; ignored by Git except documentation.

## Boundaries

- Dataset-specific paths and transforms stay in `data`; they must not leak into model code.
- Target semantics and normalization stay in `codecs`; evaluators consume decoded task values.
- The unified model accepts the same batch contract for every task.
- Model/download paths come from config or environment variables, never hard-coded absolute paths.
- `outputs/<task>/<experiment>/` is append-oriented evidence. Failed runs are retained.
- Public reports must not contain credentials, private server addresses, or other users' private paths.

## Agent Notes

- Prefer existing patterns in nearby files.
- Add new abstractions only when they remove real duplication or enforce a repeated invariant.
