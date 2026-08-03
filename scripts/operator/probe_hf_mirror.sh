#!/usr/bin/env bash
set -euo pipefail

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-15}"

probe() {
  local label="$1"
  local url="$2"
  local code
  code="$(curl --silent --show-error --location \
    --connect-timeout "$TIMEOUT_SECONDS" \
    --max-time "$TIMEOUT_SECONDS" \
    --output /dev/null \
    --write-out '%{http_code}' \
    "$url")"
  printf '%s_HTTP_STATUS=%s\n' "$label" "$code"
  [[ "$code" == "200" ]]
}

echo "HF_ENDPOINT=$HF_ENDPOINT"
probe "MODEL_API" "$HF_ENDPOINT/api/models/stabilityai/stable-diffusion-2"
probe "DATASET_API" "$HF_ENDPOINT/api/datasets/zh-plus/tiny-imagenet"
echo "HF_MIRROR_PROBE=ok"
