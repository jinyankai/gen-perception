#!/usr/bin/env python3
"""Validate every versioned YAML experiment configuration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception_diffusion.utils.config import ConfigError, load_config  # noqa: E402


def main() -> int:
    paths = sorted((ROOT / "configs").rglob("*.yaml"))
    failures: list[str] = []
    validated = 0
    for path in paths:
        if path.parts[-2] == "base":
            continue
        try:
            load_config(path, expand_environment=False)
            validated += 1
        except (ConfigError, OSError, ValueError) as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    if failures:
        print("Configuration validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Validated {validated} experiment configurations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
