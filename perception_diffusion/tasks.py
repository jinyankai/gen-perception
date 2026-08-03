"""Canonical task identifiers shared across config and model boundaries."""

from __future__ import annotations

from collections.abc import Sequence


TASK_NAMES = ("segmentation", "depth", "normal")


def validate_task_names(task_names: Sequence[str]) -> tuple[str, ...]:
    """Return validated, unique task names in their configured order."""

    names = tuple(task_names)
    if not names:
        raise ValueError("at least one task name is required")
    if len(set(names)) != len(names):
        raise ValueError(f"task names must be unique, got {names}")
    unknown = sorted(set(names) - set(TASK_NAMES))
    if unknown:
        raise ValueError(f"unsupported task names: {unknown}; expected {list(TASK_NAMES)}")
    return names
