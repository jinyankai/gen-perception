#!/usr/bin/env python3
"""Run a bounded single-node CUDA/NCCL all-reduce smoke test.

Launch this script with torchrun. It allocates only a scalar on every visible
GPU and prints one JSON record per rank. It does not load project models or use
external network services.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = (
        int(os.environ.get("LOCAL_RANK", "0"))
        if args.local_rank is None
        else args.local_rank
    )
    result: dict[str, object] = {
        "status": "RUNNING",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }
    try:
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"visible CUDA devices {torch.cuda.device_count()} < world size {world_size}"
            )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=args.timeout_seconds),
        )
        value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            raise ValueError(f"all-reduce result {value.item()} != expected {expected}")
        dist.barrier()

        result.update(
            {
                "status": "DISTRIBUTED_CUDA_SMOKE_PASSED",
                "all_reduce_sum": value.item(),
                "device_name": torch.cuda.get_device_name(local_rank),
                "allocated_mib": round(torch.cuda.memory_allocated(local_rank) / 2**20, 2),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            }
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        result.update(
            {
                "status": "DISTRIBUTED_CUDA_SMOKE_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
