#!/usr/bin/env python3
"""Resolve and download a Hugging Face repository through an explicit endpoint.

The requested revision is resolved to an immutable commit SHA before download.
No token value is printed or written to the generated manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("model", "dataset"), required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--allow-pattern", action="append", default=None)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


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
    }

    if args.metadata_only:
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
