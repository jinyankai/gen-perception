"""Small dependency-free download/extraction helpers for dataset scripts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    expected_digest: str | None = None,
    digest_algorithm: str = "sha256",
) -> dict[str, Any]:
    """Download with a resumable .part file and optional digest verification."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        digest = file_digest(destination, digest_algorithm)
        if expected_digest is None or digest.casefold() == expected_digest.casefold():
            return {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                digest_algorithm: digest,
                "downloaded": False,
            }
        raise ValueError(
            f"existing file digest mismatch for {destination}: {digest}; "
            "move the file aside before retrying"
        )

    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "gen-perception/1.0"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    status = getattr(response, "status", None)
    append = existing > 0 and status == 206
    if existing and not append:
        existing = 0
    total_header = response.headers.get("Content-Length")
    total = (int(total_header) + existing) if total_header else None
    mode = "ab" if append else "wb"
    downloaded = existing
    next_report = downloaded + 128 * 1024 * 1024
    with response, partial.open(mode) as handle:
        while chunk := response.read(8 * 1024 * 1024):
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                suffix = f"/{total}" if total is not None else ""
                print(f"downloaded {downloaded}{suffix} bytes -> {partial}", file=sys.stderr)
                next_report = downloaded + 128 * 1024 * 1024
        handle.flush()
        os.fsync(handle.fileno())
    digest = file_digest(partial, digest_algorithm)
    if expected_digest is not None and digest.casefold() != expected_digest.casefold():
        raise ValueError(
            f"download digest mismatch for {partial}: expected {expected_digest}, got {digest}"
        )
    os.replace(partial, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        digest_algorithm: digest,
        "downloaded": True,
    }


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a zip while rejecting absolute and parent-traversal members."""

    destination.mkdir(parents=True, exist_ok=False)
    resolved_root = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                target = (destination / info.filename).resolve()
                if target != resolved_root and resolved_root not in target.parents:
                    raise ValueError(f"unsafe zip member: {info.filename}")
            handle.extractall(destination)
    except Exception:
        shutil.rmtree(destination)
        raise


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)

