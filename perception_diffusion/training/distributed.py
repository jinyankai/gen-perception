"""DistributedDataParallel helpers for the shared training runner.

The single-process path stays untouched: with ``runtime.distributed: disabled``
every helper here is either a pure no-op or an identity function, so the runner
behaves exactly as before. Only ``runtime.distributed: ddp`` activates the real
process group, which requires a ``torch.distributed.run`` launch context (RANK /
WORLD_SIZE / LOCAL_RANK). See
``docs/validation/distributed-launcher-recovery-2026-08-03.md`` for the exact
static loopback rendezvous that is known to work on this server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class DistributedContext:
    """Rank/world topology for one training process."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def single_process_context() -> DistributedContext:
    return DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1)


def _require_env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"runtime.distributed=ddp requires the {name} environment variable; "
            "launch with `python -m torch.distributed.run` (see "
            "docs/validation/distributed-launcher-recovery-2026-08-03.md)"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def resolve_distributed_context(
    config: Mapping[str, Any], *, device_type: str
) -> DistributedContext:
    """Return the process topology and initialize the NCCL group when enabled."""

    mode = str(config.get("runtime", {}).get("distributed", "disabled"))
    if mode == "disabled":
        return single_process_context()
    if mode != "ddp":
        raise ValueError(f"unsupported runtime.distributed: {mode!r}")

    import torch.distributed as dist

    if device_type != "cuda":
        raise RuntimeError("runtime.distributed=ddp requires runtime.device=cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("runtime.distributed=ddp requires torch.cuda.is_available()")

    rank = _require_env_int("RANK")
    world_size = _require_env_int("WORLD_SIZE")
    local_rank = _require_env_int("LOCAL_RANK")
    if world_size < 2:
        raise RuntimeError(
            f"runtime.distributed=ddp expects WORLD_SIZE>=2, got {world_size}"
        )
    device_count = torch.cuda.device_count()
    if device_count < local_rank + 1:
        raise RuntimeError(
            f"visible CUDA devices {device_count} < local_rank+1 {local_rank + 1}"
        )

    timeout_seconds = int(config.get("runtime", {}).get("ddp_timeout_seconds", 1800))
    if timeout_seconds <= 0:
        raise ValueError("runtime.ddp_timeout_seconds must be positive")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", timeout=timedelta(seconds=timeout_seconds)
        )
    return DistributedContext(
        enabled=True, rank=rank, local_rank=local_rank, world_size=world_size
    )


def wrap_ddp(module: torch.nn.Module, ctx: DistributedContext) -> torch.nn.Module:
    """Wrap a module in DistributedDataParallel, or return it unchanged.

    The parameter set mirrors the server-validated DDP gate
    (``scripts/operator/segmentation_ddp_training_gate.py``):
    ``find_unused_parameters=True`` because round-robin batches exercise only one
    task's task-specific parameters per step, leaving the others without grad.
    """

    if not ctx.enabled:
        return module
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(
        module,
        device_ids=[ctx.local_rank],
        output_device=ctx.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )


def barrier(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


def destroy(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: float, ctx: DistributedContext, *, device: torch.device) -> float:
    """Average a Python scalar across ranks for logging; identity when disabled."""

    if not ctx.enabled:
        return value
    import torch.distributed as dist

    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item()) / ctx.world_size


def broadcast_run_root(run_root: str, ctx: DistributedContext) -> str:
    """Share rank 0's created run directory with the other ranks."""

    if not ctx.enabled:
        return run_root
    import torch.distributed as dist

    payload = [run_root if ctx.is_main else None]
    dist.broadcast_object_list(payload, src=0)
    resolved = payload[0]
    if not isinstance(resolved, str):
        raise RuntimeError("failed to broadcast the run root from rank 0")
    return resolved
