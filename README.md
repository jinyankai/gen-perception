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
- One shared task registry, diffusion loss, and latent sampler: implemented and exercised for all three tasks with tiny CPU components.
- ADE20K closed-set taxonomy query policy and optional pre-VAE residual CNN ablation: implemented.
- Pretrained VAE/CLIP/U-Net load test: not yet run.
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

## Repository map

```text
configs/                  versioned experiment configuration
perception_diffusion/     shared model, codec, data, evaluation, and utility code
scripts/                  train, infer, evaluate, data, and reproduction entry points
tests/                    fast protocol and unit tests
evals/                    repository and research smoke gates
docs/                     architecture, reproduction, and stage reports
outputs/                  ignored experiment artifacts
```

## Evidence policy

Every formal result must bind the Git commit, frozen config, dataset version and split, seed, checkpoint, raw log, inference/evaluation commands, GPU count, runtime, and any pretrained weights, ensembling, or extra data. Failed experiments are retained and recorded.

## Primary references

- [DiGSeg project page](https://wang-haoxiao.github.io/DiGSeg/)
- [Marigold official repository](https://github.com/prs-eth/Marigold)
- [PyTorch installation guidance](https://pytorch.org/get-started/locally/)
- [Diffusers installation guidance](https://huggingface.co/docs/diffusers/main/en/installation)
- [HF-Mirror usage](https://hf-mirror.com/)
