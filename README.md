# gen-perception

Unified generative perception research code for semantic segmentation, monocular depth, and surface normal estimation.

The stage-one modeling contract is:

```text
clean image latent + noisy target latent + timestep + task/text condition
    -> consistent denoiser
    -> decoded task prediction
```

## Current status

- Repository and research harness: implemented.
- Server inventory: completed on 2026-08-03.
- Target codecs and evaluators: implemented with local unit tests.
- Task-token + task-adapter + shared-U-Net model skeleton: implemented with local unit tests.
- Inherited single-task and multi-task engineering configs: implemented and validated.
- One shared task registry, latent-level diffusion-loss core, and latent sampler: implemented and exercised for all three tasks with tiny CPU components.
- ADE20K closed-set taxonomy query policy and optional pre-VAE residual CNN ablation: implemented.
- Pretrained tokenizer/CLIP/VAE/U-Net offline CPU load and project 8-channel denoiser forward: passed in user-operated server smoke S002.
- ADE20K and NYUv2 local asset/sample codec checks: passed in S002.
- ADE20K segmentation and NYUv2 depth/derived-normal Dataset/DataLoader, download, preprocessing, and real-sample smoke CLIs: implemented locally; server real-sample execution is pending.
- Raw dataset -> codec is implemented; codec -> VAE -> shared U-Net -> decode/evaluate is not yet integrated end to end.
- Formal training/evaluation: not yet run.

See `docs/unified-perception-framework.md` for the technical design and `STATUS.md`, `BLOCKERS.md`, and `EXPERIMENTS.md` for evidence-backed state.

## Environment

The supported research environment is Python 3.11. The server base Python 3.14 environment is intentionally not used.

```bash
conda env create -f environment.yml
conda activate gen-perception
python evals/smoke_eval.py
python -m unittest discover -s tests -p 'test_*.py'
```

PyTorch CUDA wheels are installed separately by `scripts/bootstrap_env.sh` so that the CUDA index is explicit and auditable.

Because the server cannot currently reach the official Hugging Face endpoint, source `scripts/hf_mirror_env.sh` after setting an approved model-cache path. See `docs/huggingface-mirror.md`. Tokens are never stored in the repository.

## Canonical checks

```bash
python evals/smoke_eval.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/validate_configs.py
python scripts/train.py --config configs/smoke.yaml --dry-run
python scripts/framework_smoke.py --config configs/multitask/stage1_shared_unet.yaml
```

## Unified evaluation

`scripts/evaluate.py` recursively pairs predictions and targets by relative path without
the extension, calls the configured segmentation/depth/normal evaluator, and writes both
`metrics.json` and long-form `metrics.csv`. Supported inputs are NPY, single-array NPZ,
PNG, and TIFF. Existing metric files are not overwritten unless `--overwrite` is supplied.

ADE20K predictions must use class IDs 0-149. Raw ADE20K targets use 0 as ignore and
1-150 as class IDs, so evaluate them with an explicit conversion:

```bash
python scripts/evaluate.py \
  --config configs/segmentation/ade20k.yaml \
  --predictions /path/to/predictions \
  --targets /path/to/ADEChallengeData2016/annotations/validation \
  --target-ignore-value 0 \
  --target-label-offset -1 \
  --output-dir outputs/eval/ade20k
```

Depth arrays must be in meters. For millimeter PNG targets, add
`--target-depth-scale 0.001`. Surface normals may be CHW or HWC arrays; use
`--prediction-normal-encoding uint8` or `zero_one` when they are not already XYZ values
in `[-1,1]`. A multitask config additionally requires `--task segmentation`, `depth`, or
`normal`.

## Repository map

```text
configs/                  versioned experiment configuration
perception_diffusion/     shared model, codec, data, evaluation, and utility code
scripts/                  training/evaluation entry points, config checks, downloads, and operator helpers
tests/                    fast protocol and unit tests
evals/                    repository and research smoke gates
docs/                     architecture, reproduction, and stage reports
outputs/                  ignored experiment artifacts
```

The local framework smoke is intentionally structural: it uses an injected tiny U-Net and scheduler to verify that all three task names share the same denoiser, loss, and sampling interfaces. S002 separately verifies that the real local SD2 components and dataset samples are readable and that a CPU denoiser forward is finite. Neither smoke is evidence of end-to-end real-data training, CUDA execution, decoded benchmark predictions, or formal metrics.

## Evidence policy

Every formal result must bind the Git commit, frozen config, dataset version and split, seed, checkpoint, raw log, inference/evaluation commands, GPU count, runtime, and any pretrained weights, ensembling, or extra data. Failed experiments are retained and recorded.

## Primary references

- [DiGSeg project page](https://wang-haoxiao.github.io/DiGSeg/)
- [Marigold official repository](https://github.com/prs-eth/Marigold)
- [PyTorch installation guidance](https://pytorch.org/get-started/locally/)
- [Diffusers installation guidance](https://huggingface.co/docs/diffusers/main/en/installation)
- [HF-Mirror usage](https://hf-mirror.com/)
