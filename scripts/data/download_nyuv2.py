#!/usr/bin/env python3
"""Download the official NYUv2 labeled MAT file and canonical split file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_utils import download_file, write_json  # noqa: E402


LABELED_URL = "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
SPLITS_URL = "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"
DATASET_PAGE = "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"
# Published official files are stable; MD5 catches interrupted/mirror-corrupted downloads.
LABELED_MD5 = "520609c519fba3ba5ac58c8fefcc3530"
SPLITS_MD5 = "08e3c3aea27130ac7c01ffd739a4535f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data")),
        help="Parent directory that will contain nyuv2/.",
    )
    parser.add_argument("--labeled-url", default=LABELED_URL)
    parser.add_argument("--splits-url", default=SPLITS_URL)
    parser.add_argument(
        "--skip-known-md5",
        action="store_true",
        help="Do not compare against known hashes (useful only for an intentional mirror).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data_root.expanduser().resolve() / "nyuv2"
    root.mkdir(parents=True, exist_ok=True)
    labeled = download_file(
        args.labeled_url,
        root / "nyu_depth_v2_labeled.mat",
        expected_digest=None if args.skip_known_md5 else LABELED_MD5,
        digest_algorithm="md5",
    )
    splits = download_file(
        args.splits_url,
        root / "splits.mat",
        expected_digest=None if args.skip_known_md5 else SPLITS_MD5,
        digest_algorithm="md5",
    )
    write_json(
        root / "download_manifest.json",
        {
            "dataset": "NYUv2 labeled",
            "dataset_page": DATASET_PAGE,
            "labeled": {"source_url": args.labeled_url, **labeled},
            "splits": {"source_url": args.splits_url, **splits},
        },
    )
    print(f"NYUv2 raw assets prepared at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

