#!/usr/bin/env python3
"""Download and safely extract the ADEChallengeData2016 benchmark archive."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_utils import download_file, safe_extract_zip, write_json  # noqa: E402


DEFAULT_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
TERMS_URL = "https://groups.csail.mit.edu/vision/ADE20K/terms/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data")),
        help="Parent directory that will contain ADEChallengeData2016/.",
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use an already downloaded zip instead of fetching --url.",
    )
    parser.add_argument("--sha256", default=None, help="Optional expected archive SHA-256.")
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help=f"Confirm acceptance of ADE20K terms: {TERMS_URL}",
    )
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


def validate_layout(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split, expected in (("training", 20_210), ("validation", 2_000)):
        image_root = root / "images" / split
        mask_root = root / "annotations" / split
        if not image_root.is_dir() or not mask_root.is_dir():
            raise FileNotFoundError(f"archive lacks images/annotations for {split}")
        images = sorted(image_root.rglob("*.jpg"))
        masks = sorted(mask_root.rglob("*.png"))
        if len(images) != expected or len(masks) != expected:
            raise ValueError(
                f"ADE20K {split} expected {expected} image/mask files, "
                f"found {len(images)}/{len(masks)}"
            )
        counts[split] = expected
    metadata = root / "objectInfo150.txt"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    return counts


def main() -> int:
    args = parse_args()
    if not args.accept_terms:
        print(
            f"Refusing to download/extract until --accept-terms is supplied. Read {TERMS_URL}",
            file=sys.stderr,
        )
        return 2
    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    destination = data_root / "ADEChallengeData2016"
    if destination.exists():
        counts = validate_layout(destination)
        print(f"ADE20K already prepared at {destination}: {counts}")
        return 0
    if args.archive is None:
        archive = data_root / "downloads" / "ADEChallengeData2016.zip"
        download = download_file(
            args.url, archive, expected_digest=args.sha256, digest_algorithm="sha256"
        )
    else:
        archive = args.archive.expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
        from download_utils import file_digest

        digest = file_digest(archive)
        if args.sha256 and digest.casefold() != args.sha256.casefold():
            raise ValueError(f"archive SHA-256 mismatch: {digest}")
        download = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": digest,
            "downloaded": False,
        }

    temporary = data_root / f".ade20k-extract-{os.getpid()}-{time.time_ns()}"
    safe_extract_zip(archive, temporary)
    extracted = temporary / "ADEChallengeData2016"
    if not extracted.is_dir():
        candidates = list(temporary.rglob("ADEChallengeData2016"))
        if len(candidates) != 1 or not candidates[0].is_dir():
            shutil.rmtree(temporary)
            raise FileNotFoundError("zip does not contain one ADEChallengeData2016 directory")
        extracted = candidates[0]
    counts = validate_layout(extracted)
    os.replace(extracted, destination)
    shutil.rmtree(temporary)
    write_json(
        destination / "download_manifest.json",
        {
            "dataset": "ADEChallengeData2016",
            "source_url": args.url,
            "terms_url": TERMS_URL,
            "terms_accepted_by_operator": True,
            "archive": download,
            "split_counts": counts,
        },
    )
    if not args.keep_archive and args.archive is None:
        archive.unlink()
    print(f"ADE20K prepared at {destination}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

