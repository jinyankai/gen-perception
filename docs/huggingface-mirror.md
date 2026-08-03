# Hugging Face China Mirror

The server cannot currently reach `huggingface.co`. This project supports the community HF-Mirror endpoint through standard Hugging Face client configuration:

```bash
export MODEL_CACHE=/approved/high-capacity/path/model_cache
source scripts/hf_mirror_env.sh
```

This sets:

```text
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=$MODEL_CACHE/huggingface
HF_HUB_CACHE=$HF_HOME/hub
HF_HUB_DISABLE_TELEMETRY=1
```

The endpoint is process-configurable and is not hard-coded in model modules.

## Connectivity and metadata probe

First run the dependency-free connectivity probe and return its complete output:

```bash
bash scripts/operator/probe_hf_mirror.sh
```

This performs two small API requests and downloads no model or dataset artifacts.

Resolve a repository revision without downloading files:

```bash
python scripts/hf_download.py \
  --repo-type model \
  --repo-id stabilityai/stable-diffusion-2 \
  --revision main \
  --local-dir "$MODEL_CACHE/stable-diffusion-2" \
  --metadata-only
```

## Download a pretrained model

```bash
python scripts/hf_download.py \
  --repo-type model \
  --repo-id prs-eth/marigold-depth-v1-1 \
  --revision main \
  --local-dir "$MODEL_CACHE/marigold-depth-v1-1"
```

The downloader first resolves `main` or a tag to an immutable repository SHA and writes that SHA to `download_manifest.json`. Formal experiments must cite this resolved revision.

## Download a dataset repository

Only use a dataset repository after its license, provenance, layout, and benchmark split are reviewed:

```bash
python scripts/hf_download.py \
  --repo-type dataset \
  --repo-id OWNER/DATASET \
  --revision main \
  --local-dir "$DATA_ROOT/DATASET"
```

Do not infer ADE20K or NYUv2 benchmark compliance merely from a repository name. Record the exact repository, commit, license, preprocessing, and split in the experiment config and report.

## Gated repositories and secrets

- Accept the license on the official Hugging Face site first.
- Export `HF_TOKEN` only in the active shell or use the official local credential store.
- Never place tokens in `.env`, shell scripts, Git, logs, commands, or download manifests.
- The mirror is a third-party transport path. Pin immutable revisions and retain manifests; if a file checksum or license is published upstream, verify it.

HF-Mirror documents the same `HF_ENDPOINT=https://hf-mirror.com` mechanism, and Hugging Face maintainers have confirmed that `HfApi(endpoint=...)` or `HF_ENDPOINT` can select a mirror endpoint.
