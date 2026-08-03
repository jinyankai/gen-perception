#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${GEN_PERCEPTION_ENV_NAME:-gen-perception}"

if ! command -v conda >/dev/null 2>&1; then
  printf 'conda is required but was not found.\n' >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  printf 'Environment already exists: %s\n' "$ENV_NAME" >&2
  printf 'Refusing to mutate it automatically. Remove it explicitly or choose GEN_PERCEPTION_ENV_NAME.\n' >&2
  exit 2
fi

conda env create --name "$ENV_NAME" --file "$ROOT/environment.yml"
conda run --name "$ENV_NAME" python -m pip install -r "$ROOT/requirements-torch-cu128.txt"
conda run --name "$ENV_NAME" python -m pip install --no-deps -e "$ROOT"

conda run --name "$ENV_NAME" python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY

printf 'Environment created: %s\n' "$ENV_NAME"
