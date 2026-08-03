# Architecture Map

## Domains

- `perception_diffusion/codecs`: reversible or approximately reversible task target representations.
- `perception_diffusion/data`: adapters and dataset-owned metadata, including segmentation taxonomies, that map datasets to the unified batch schema.
- `perception_diffusion/evaluation`: protocol-aligned metrics plus strict paired-file dataset aggregation for segmentation, depth, and normals.
- `perception_diffusion/models`: VAE, task conditioner, denoiser, scheduler, and unified model boundaries.
- `perception_diffusion/training`: the task-agnostic diffusion loss and future optimizer/checkpoint loop.
- `perception_diffusion/inference`: the shared latent sampler and leakage-free task query planning.
- `perception_diffusion/tasks.py`: canonical task identifiers shared by config, model, data, and evaluation layers.
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
- A single shared U-Net consumes learned task tokens through cross-attention; task-specific adapters operate on condition tokens in v1.
- Deterministic codecs own task semantics and `[-1,1]` normalization. Optional pre-VAE CNNs may only learn bounded residual adaptation of that canonical representation.
- Closed-set segmentation query vocabularies come from dataset metadata and never from the current sample's ground truth.
- Stage-one batches are task-homogeneous. Multi-task scheduling happens between batches, never by forking trainers.
- Model configuration is inherited from `configs/base/model.yaml`; task configs may override only task/data/evaluation and intentional experiment fields.
- Model/download paths come from config or environment variables, never hard-coded absolute paths.
- `scripts/evaluate.py` is the shared offline metric entry point. File decoding and aggregation live in `evaluation/runner.py`; task-specific metric definitions remain in the individual evaluator modules.
- `outputs/<task>/<experiment>/` is append-oriented evidence. Failed runs are retained.
- Public reports must not contain credentials, private server addresses, or other users' private paths.

## Agent Notes

- Prefer existing patterns in nearby files.
- Add new abstractions only when they remove real duplication or enforce a repeated invariant.
