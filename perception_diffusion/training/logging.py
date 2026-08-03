"""Offline-first JSONL, TensorBoard, and optional W&B training logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrainingLogger:
    def __init__(
        self,
        run_root: str | Path,
        config: dict[str, Any],
        *,
        enable_tensorboard: bool | None = None,
        enable_wandb: bool | None = None,
    ) -> None:
        self.root = Path(run_root)
        self.metrics_path = self.root / "metrics.jsonl"
        self.train_log_path = self.root / "train.log"
        logging_config = config["training"].get("logging", {})
        tensorboard_enabled = bool(logging_config.get("tensorboard", True))
        if enable_tensorboard is not None:
            tensorboard_enabled = enable_tensorboard
        self.writer = None
        if tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:  # pragma: no cover - environment dependency
                raise RuntimeError("TensorBoard logging requires tensorboard") from exc
            self.writer = SummaryWriter(log_dir=str(self.root / "tensorboard"))

        wandb_config = logging_config.get("wandb", {})
        wandb_enabled = bool(wandb_config.get("enabled", False))
        if enable_wandb is not None:
            wandb_enabled = enable_wandb
        self.wandb_run = None
        if wandb_enabled:
            try:
                import wandb
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "W&B logging was enabled; install requirements-wandb.txt"
                ) from exc
            self.wandb_run = wandb.init(
                project=str(wandb_config.get("project", "gen-perception")),
                name=str(config["experiment"]["name"]),
                dir=str(self.root),
                mode=str(wandb_config.get("mode", "offline")),
                config=config,
                reinit=True,
            )

    def log(self, step: int, metrics: dict[str, float | int | str]) -> None:
        record: dict[str, Any] = {"step": int(step), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.train_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if self.writer is not None:
            for key, value in numeric.items():
                self.writer.add_scalar(key, value, step)
        if self.wandb_run is not None:
            self.wandb_run.log(numeric, step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
