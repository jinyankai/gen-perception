# Decision Log

## D001 - Stage-one task set

- Date: 2026-08-03
- Decision: semantic segmentation, monocular depth, and surface normals.
- Reason: required three-task coverage while NYUv2 lets depth and normals share data and geometric preprocessing.

## D002 - Shared framework boundary

- Date: 2026-08-03
- Decision: task differences are isolated to target codecs, dataset adapters, evaluators, conditions, and visualizers.
- Reason: the evaluation requires a real unified framework rather than three adjacent repositories.

## D003 - Python and CUDA package baseline

- Date: 2026-08-03
- Decision: use Python 3.11 and an official PyTorch CUDA 12.8 wheel; do not use the server's Python 3.14 base environment.
- Reason: Python 3.11 is widely supported by research dependencies; the official PyTorch selector supports CUDA 12.8, while Diffusers 0.39 requires PyTorch 2.6+.

## D004 - Offline-first experiment records

- Date: 2026-08-03
- Decision: local JSON/JSONL and text logs are mandatory; W&B is optional and disabled by default.
- Reason: formal evidence must not depend on external service availability.

## D005 - No data or weight artifacts in Git

- Date: 2026-08-03
- Decision: datasets, model caches, checkpoints, predictions, and logs are Git-ignored; Git stores configs, commands, summaries, and retrieval instructions.
- Reason: artifact size, privacy, reproducibility, and credential safety.

## D006 - Hugging Face mirror is explicit and revision-pinned

- Date: 2026-08-03
- Decision: default China-accessible endpoint is `https://hf-mirror.com`, selected through `HF_ENDPOINT`; every download resolves to an immutable SHA and writes a manifest.
- Reason: the official endpoint timed out from the server, while environment-based endpoint selection is supported by Hugging Face clients. The mirror remains overridable and is not embedded in model code.
- Safety: no tokens in Git or logs; gated licenses must be accepted upstream; dataset provenance and benchmark splits are reviewed separately.

## D007 - Server operations are user-operated

- Date: 2026-08-03
- Decision: the coding agent must not SSH into, control, or directly run commands on the research server. It prepares reviewed, copy-pasteable commands or repository scripts; the user executes them and returns logs.
- Reason: the user explicitly prefers to operate the server and wants agent time focused on local implementation and evidence analysis.
- Evidence rule: only user-returned output establishes remote state. Interrupted or timed-out attempts remain unverified.
- Safety: credentials are never persisted in Git, agent memory, scripts, command examples, or logs.

## D008 - Task-token cross-attention with one shared SD2 U-Net

- Date: 2026-08-03
- Decision: stage one uses one Stable Diffusion 2 U-Net for segmentation, depth, and normals. Every forward pass includes learned task tokens; optional CLIP text tokens are concatenated for text-controllable tasks and injected through the U-Net cross-attention layers.
- Trainability: VAE and CLIP remain frozen; `conv_in`, cross-attention, task embeddings, and task adapters are trainable by default.
- Reason: the reference pipelines share the same latent denoising mechanism, while their empty-text conditions cannot identify the task after weights are unified.

## D009 - Task adapters operate on condition tokens in v1

- Date: 2026-08-03
- Decision: each task owns a residual bottleneck adapter in the cross-attention condition space. The adapted tokens are consumed at every U-Net scale; there are no task-specific U-Net copies.
- Reason: this is the smallest task-specific capacity that preserves a genuinely shared backbone and remains easy to test and checkpoint.
- Escalation rule: add task-specific attention processors or LoRA only after controlled multi-task experiments show negative transfer that sampling and loss balancing do not resolve.

## D010 - Config inheritance and task-homogeneous batches

- Date: 2026-08-03
- Decision: model defaults live in `configs/base/model.yaml` and are recursively deep-merged into experiment configs. Multi-task training switches tasks between homogeneous batches.
- Reason: one model source prevents configuration drift; homogeneous batches avoid mixing incompatible target semantics, resolutions, masks, and prompt policies inside one batch.

## D011 - Closed-set segmentation queries are dataset-owned

- Date: 2026-08-03
- Decision: ADE20K evaluation loads the complete 150-class taxonomy from the official `objectInfo150.txt` metadata and queries all classes when class-wise text inference is used. No query-planning API accepts a ground-truth mask.
- Reason: standard mIoU must include false positives and absent classes according to the dataset protocol; filtering by GT-present classes leaks evaluation labels and inflates results.
- Extension: user-provided or independently generated vocabularies are allowed only as separately labeled open-vocabulary experiments.

## D012 - Deterministic codecs precede optional learned target adaptation

- Date: 2026-08-03
- Decision: `TargetCodec` owns task semantics, validity, channel canonicalization, and the `[-1,1]` VAE input range. An optional per-task `ResidualPreVAEAdapter` may learn a bounded three-channel residual and is disabled by default.
- Reason: an unconstrained raw-target-to-RGB CNN can misuse discrete class IDs, collapse representations, or hide protocol errors. Zero-initialized residual adaptation gives an identity starting point and a controlled ablation.

## D013 - Marigold-style annealed multi-resolution noise

- Date: 2026-08-03
- Decision: when enabled, training samples multi-resolution Gaussian fields and linearly scales their strength by `timestep / num_train_timesteps`, following the inspected Marigold reference behavior.
- Reason: the previous YAML declared annealed multi-scale noise while the trainer sampled only standard Gaussian noise; implementing one configuration-driven sampler removes that behavior drift for all tasks.
- Evidence boundary: local tests verify the formula, shape, reproducibility, variance, and trainer integration. Only real loss curves can establish usefulness.

## D014 - Single-process real runner before DDP

- Date: 2026-08-03
- Decision: the standard runner supports real single-GPU training, checkpoint/resume, JSONL/TensorBoard, optional W&B, and end-to-end inference. It explicitly rejects DDP until the server NCCL gate passes.
- Reason: a verified single-process closure is required before adding distributed failure modes; task differences remain outside the runner.
