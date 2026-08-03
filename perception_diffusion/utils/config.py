"""Configuration loading and structural validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the project schema."""


REQUIRED_TOP_LEVEL = {
    "experiment",
    "task",
    "model",
    "data",
    "training",
    "inference",
    "evaluation",
}
ALLOWED_TASKS = {"segmentation", "depth", "normal"}


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_config(path: str | Path, *, expand_environment: bool = True) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    config = _expand_environment(loaded) if expand_environment else loaded
    validate_config(config, source=str(config_path))
    return config


def validate_config(config: dict[str, Any], *, source: str = "<memory>") -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ConfigError(f"{source}: missing top-level keys: {missing}")
    if not isinstance(config["task"], dict):
        raise ConfigError(f"{source}: task must be a mapping")
    task_name = config["task"].get("name")
    if task_name not in ALLOWED_TASKS:
        raise ConfigError(f"{source}: task.name must be one of {sorted(ALLOWED_TASKS)}")
    experiment = config["experiment"]
    if not isinstance(experiment, dict) or not experiment.get("name"):
        raise ConfigError(f"{source}: experiment.name is required")
    seed = experiment.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"{source}: experiment.seed must be a non-negative integer")
    training = config["training"]
    if not isinstance(training, dict):
        raise ConfigError(f"{source}: training must be a mapping")
    max_steps = training.get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ConfigError(f"{source}: training.max_steps must be a positive integer")
    inference = config["inference"]
    steps = inference.get("num_steps") if isinstance(inference, dict) else None
    if not isinstance(steps, int) or steps <= 0:
        raise ConfigError(f"{source}: inference.num_steps must be a positive integer")
