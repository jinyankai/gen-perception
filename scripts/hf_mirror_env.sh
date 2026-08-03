#!/usr/bin/env bash
# Source this file to configure Hugging Face clients for the China mirror.

if [[ -z "${MODEL_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  printf 'Set MODEL_CACHE or HF_HOME to an approved high-capacity path first.\n' >&2
  return 2 2>/dev/null || exit 2
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${MODEL_CACHE}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

printf 'HF_ENDPOINT=%s\n' "$HF_ENDPOINT"
printf 'HF_HOME=%s\n' "$HF_HOME"
printf 'HF_HUB_CACHE=%s\n' "$HF_HUB_CACHE"
printf 'HF token present: %s\n' "$( [[ -n "${HF_TOKEN:-}" ]] && printf yes || printf no )"
