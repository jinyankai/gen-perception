#!/usr/bin/env python3
"""Resolve and download a Hugging Face repository through an explicit endpoint.

The requested revision is resolved to an immutable commit SHA before download.
No token value is printed or written to the generated manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("model", "dataset"), required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--allow-pattern", action="append", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true")
    mode.add_argument(
        "--register-existing",
        action="store_true",
        help=(
            "Do not download. Inventory an existing local directory, record file "
            "SHA-256 values, and write a provenance-qualified manifest."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_existing_files(
    local_dir: Path, allow_patterns: list[str] | None
) -> list[dict[str, object]]:
    """Hash selected existing files without following hidden HF cache metadata."""

    if not local_dir.is_dir():
        raise FileNotFoundError(f"existing local directory not found: {local_dir}")
    inventory: list[dict[str, object]] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        if relative == "download_manifest.json" or ".cache" in path.parts:
            continue
        if allow_patterns and not any(
            fnmatchcase(relative, pattern) for pattern in allow_patterns
        ):
            continue
        print(f"hashing {relative}", file=sys.stderr)
        inventory.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not inventory:
        raise RuntimeError("no existing files matched the requested local directory/patterns")
    return inventory


def main() -> int:
    args = parse_args()
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed in the active environment", file=sys.stderr)
        return 2

    api = HfApi(endpoint=endpoint, token=token)
    info: Any
    if args.repo_type == "model":
        info = api.model_info(args.repo_id, revision=args.revision)
    else:
        info = api.dataset_info(args.repo_id, revision=args.revision)
    resolved_sha = info.sha
    if not resolved_sha:
        raise RuntimeError("mirror did not return an immutable commit SHA")

    manifest = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "requested_revision": args.revision,
        "resolved_revision": resolved_sha,
        "endpoint": endpoint,
        "local_dir": str(args.local_dir.resolve()),
        "allow_patterns": args.allow_pattern,
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "token_used": bool(token),
        "download_completed": False,
        "download_method": "huggingface_hub_snapshot_download",
        "revision_verification": "resolved_before_download",
    }

    if args.metadata_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    if args.register_existing:
        manifest_path = args.local_dir / "download_manifest.json"
        if manifest_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing manifest: {manifest_path}"
            )
        inventory = inventory_existing_files(args.local_dir, args.allow_pattern)
        manifest.update(
            {
                "download_completed": True,
                "download_method": "external_registered_after_download",
                "revision_verification": (
                    "unverified_current_revision_resolved_after_download"
                ),
                "registration_note": (
                    "The immutable revision was queried after the files already existed; "
                    "local SHA-256 values identify these exact files, but their upstream "
                    "revision is not independently proven."
                ),
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                "local_file_count": len(inventory),
                "local_total_bytes": sum(
                    int(item["size_bytes"]) for item in inventory
                ),
                "local_files": inventory,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    args.local_dir.mkdir(parents=True, exist_ok=True)
    api.snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=resolved_sha,
        local_dir=args.local_dir,
        allow_patterns=args.allow_pattern,
        token=token,
    )
    manifest["download_completed"] = True
    manifest_path = args.local_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
