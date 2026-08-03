"""Create append-oriented experiment evidence directories."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    checkpoints: Path
    predictions: Path
    visualizations: Path


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def create_experiment_directory(
    *,
    output_root: str | Path,
    task: str,
    experiment_name: str,
    config: dict[str, Any],
    command: Sequence[str],
    repo_root: str | Path,
) -> ExperimentPaths:
    if task not in {"segmentation", "depth", "normal", "infrastructure"}:
        raise ValueError(f"unsupported experiment task: {task}")
    if not experiment_name or any(part in experiment_name for part in ("/", "\\", "..")):
        raise ValueError("experiment_name must be a single safe path component")
    root = Path(output_root) / task / experiment_name
    root.mkdir(parents=True, exist_ok=False)
    checkpoints = root / "checkpoints"
    predictions = root / "predictions"
    visualizations = root / "visualizations"
    for directory in (checkpoints, predictions, visualizations):
        directory.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (root / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (root / "git_commit.txt").write_text(
        _git_commit(Path(repo_root)) + "\n", encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    (root / "environment.txt").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "train.log").touch()
    (root / "metrics.json").write_text("{}\n", encoding="utf-8")
    return ExperimentPaths(root, checkpoints, predictions, visualizations)
