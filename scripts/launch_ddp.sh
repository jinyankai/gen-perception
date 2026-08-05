#!/usr/bin/env bash
# Launch multi-GPU DDP full-scale training with the server-validated static
# single-node rendezvous (see docs/validation/distributed-launcher-recovery-2026-08-03.md
# and docs/agent-harness/server-operator-contract.md).
#
# Do NOT use a PATH-resolved bare `torchrun --standalone` here: on this host it
# hung past 60s. This script invokes the environment interpreter explicitly and
# pins static loopback rendezvous with a unique master port.
#
# Usage:
#   scripts/launch_ddp.sh [NPROC] [CONFIG] [-- extra args passed to train.py]
#
# Examples:
#   # 20-step smoke on 2 GPUs before the full run
#   CUDA_VISIBLE_DEVICES=0,1 scripts/launch_ddp.sh 2 configs/multitask/stage1_shared_unet_ddp.yaml -- --max-steps 20
#   # full 60k run on all 8 GPUs
#   scripts/launch_ddp.sh 8 configs/multitask/stage1_shared_unet_ddp.yaml
#   # resume from a rank-0 checkpoint
#   scripts/launch_ddp.sh 8 configs/multitask/stage1_shared_unet_ddp.yaml -- --resume /home/jinyankai/outputs/<run>/checkpoints/step-00030000.pt
set -euo pipefail

NPROC="${1:-8}"
CONFIG="${2:-configs/multitask/stage1_shared_unet_ddp.yaml}"
shift || true
shift || true
if [[ "${1:-}" == "--" ]]; then
  shift
fi

# Environment interpreter, explicit (never a PATH-resolved bare torchrun).
GP_PYTHON="${GP_PYTHON:-/home/jinyankai/miniconda3/envs/gen-perception/bin/python}"
# A unique, currently-unused port per launch; override MASTER_PORT for concurrency.
MASTER_PORT="${MASTER_PORT:-29527}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "$GP_PYTHON" ]]; then
  echo "launch_ddp: interpreter not found: $GP_PYTHON (set GP_PYTHON)" >&2
  exit 1
fi

echo "launch_ddp: nproc=$NPROC config=$CONFIG master_port=$MASTER_PORT" >&2
echo "launch_ddp: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2

exec "$GP_PYTHON" -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node="$NPROC" \
  --node-rank=0 \
  --master-addr=127.0.0.1 \
  --master-port="$MASTER_PORT" \
  scripts/train.py --config "$CONFIG" "$@"
