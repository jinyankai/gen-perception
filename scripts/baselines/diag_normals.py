#!/usr/bin/env python3
"""Diagnose the Marigold-vs-NYUv2 surface-normal angular-error gap.

Runs three checks so the residual can be attributed to a concrete cause
instead of guessed at, all over the exported baseline artifacts:

1. GT self-consistency: recompute normals from the exported GT depth via the
   same depth_to_normals used at export, and compare to the stored GT normals.
   Near 0 deg means the GT pipeline is self-consistent; a large value means the
   GT export itself is wrong.
2. Sign/permutation search: best axis-signs (and permutations) mapping the
   Marigold prediction onto the GT convention, aggregated over N samples.
3. Stored-vector sanity: are the stored GT normals actually unit length.

Usage (baseline artifacts under $OUT, repo importable):
    OUT=/home/jinyankai/outputs/baselines \
    conda run -n gen-perception python scripts/baselines/diag_normals.py --samples 20
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _chw(a: np.ndarray) -> np.ndarray:
    a = np.squeeze(np.asarray(a, dtype=np.float32))
    if a.shape[-1] == 3 and a.shape[0] != 3:
        a = np.moveaxis(a, -1, 0)
    return a


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=0, keepdims=True)
    return a / np.clip(n, 1e-6, None)


def _paths(out: str, sid: str) -> tuple[str, str, str, str]:
    return (
        f"{out}/normal/pred_marigold/{sid}.npy",
        f"{out}/normal/gt/targets/{sid}.npy",
        f"{out}/normal/gt/valid_masks/{sid}.npy",
        f"{out}/depth/gt/targets/{sid}.npy",
    )


def _sample_ids(out: str, limit: int) -> list[str]:
    preds = sorted(glob.glob(f"{out}/normal/pred_marigold/**/*.npy", recursive=True))
    ids = [p.split("pred_marigold/", 1)[1][:-4] for p in preds]
    return ids[:limit]


def gt_self_consistency(out: str, sid: str) -> None:
    from perception_diffusion.data.nyuv2_geometry import depth_to_normals

    _, gf, mf, df = _paths(out, sid)
    if not os.path.exists(df):
        print(f"[gt-check] depth GT missing for {sid}: {df}")
        return
    z = np.load(df).astype(np.float32)
    g = _unit(_chw(np.load(gf)))
    m = np.load(mf).astype(bool)
    recomputed, valid = depth_to_normals(z)
    r = _unit(_chw(recomputed))
    both = m & valid
    cos = np.clip((g * r).sum(0), -1.0, 1.0)
    deg = float(np.degrees(np.arccos(cos))[both].mean())
    norm_mean = float(np.linalg.norm(_chw(np.load(gf)), axis=0)[m].mean())
    print(f"[gt-check] sample={sid} depth_shape={z.shape}")
    print(f"[gt-check] GT_vs_recomputed_deg={deg:.3f}  (near 0 => GT self-consistent)")
    print(f"[gt-check] stored_GT_norm_mean={norm_mean:.4f}  (should be ~1.0)")


def sign_perm_search(out: str, ids: list[str], permute: bool) -> None:
    preds, gts = [], []
    for sid in ids:
        pf, gf, mf, _ = _paths(out, sid)
        if not (os.path.exists(gf) and os.path.exists(mf)):
            continue
        p = _unit(_chw(np.load(pf)))
        g = _unit(_chw(np.load(gf)))
        m = np.load(mf).astype(bool)
        preds.append(p[:, m])
        gts.append(g[:, m])
    P = np.concatenate(preds, axis=1)
    G = np.concatenate(gts, axis=1)
    print(f"[search] samples={len(preds)} valid_pixels={P.shape[1]}")
    perms = itertools.permutations((0, 1, 2)) if permute else [(0, 1, 2)]
    results = []
    for perm in perms:
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    s = np.array([sx, sy, sz], np.float32)[:, None]
                    pp = P[list(perm), :] * s
                    cos = np.clip((pp * G).sum(0), -1.0, 1.0)
                    mae = float(np.degrees(np.arccos(cos)).mean())
                    results.append((mae, perm, (sx, sy, sz)))
    results.sort()
    for mae, perm, sgn in results[:6]:
        print(f"[search] perm={perm} sign={sgn}  mean_ang={mae:6.2f}")
    print(f"[search] BEST {results[0]}")


def _gather(out: str, ids: list[str]):
    """Return (P, G, per_sample) as 3xN stacks plus per-sample (sid, p3hw, g3hw, mask)."""
    preds, gts, per = [], [], []
    for sid in ids:
        pf, gf, mf, _ = _paths(out, sid)
        if not (os.path.exists(gf) and os.path.exists(mf)):
            continue
        p = _unit(_chw(np.load(pf)))
        g = _unit(_chw(np.load(gf)))
        m = np.load(mf).astype(bool)
        preds.append(p[:, m])
        gts.append(g[:, m])
        per.append((sid, p, g, m))
    return np.concatenate(preds, axis=1), np.concatenate(gts, axis=1), per


def kabsch_residual(P: np.ndarray, G: np.ndarray) -> None:
    """Best-fit orthogonal transform (allows reflection) mapping P->G; report residual."""
    H = P @ G.T
    U, _, Vt = np.linalg.svd(H)
    R = (Vt.T @ U.T)
    PR = R @ P
    cos = np.clip((PR * G).sum(0), -1.0, 1.0)
    mae = float(np.degrees(np.arccos(cos)).mean())
    print(f"[kabsch] det(R)={np.linalg.det(R):+.3f}")
    print("[kabsch] R=" + np.array2string(R, precision=3, suppress_small=True).replace("\n", " "))
    print(f"[kabsch] residual_mean_ang_after_best_rotation={mae:6.2f}")


def flip_search(per) -> None:
    """For each spatial flip, find best axis-sign; catches stored-orientation mismatch."""
    flips = {
        "none": lambda a: a,
        "lr": lambda a: a[:, :, ::-1],
        "ud": lambda a: a[:, ::-1, :],
        "both": lambda a: a[:, ::-1, ::-1],
        "transpose": lambda a: np.swapaxes(a, 1, 2),
    }
    for fname, fn in flips.items():
        best = None
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    s = np.array([sx, sy, sz], np.float32)[:, None, None]
                    tot_cos, tot_n = 0.0, 0
                    for _sid, p, g, m in per:
                        pp = fn(p) * s
                        if pp.shape != g.shape:
                            break
                        cos = np.clip((pp * g).sum(0), -1.0, 1.0)[m]
                        tot_cos += float(np.degrees(np.arccos(cos)).sum())
                        tot_n += int(m.sum())
                    else:
                        mae = tot_cos / max(tot_n, 1)
                        if best is None or mae < best[0]:
                            best = (mae, (sx, sy, sz))
        if best is not None:
            print(f"[flip] {fname:9s} best_sign={best[1]} mean_ang={best[0]:6.2f}")
        else:
            print(f"[flip] {fname:9s} shape-incompatible, skipped")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.environ.get("OUT"))
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--permute", action="store_true", help="also search axis permutations")
    args = ap.parse_args()
    if not args.out:
        print("set --out or $OUT to the baselines output root")
        return 2
    ids = _sample_ids(args.out, args.samples)
    if not ids:
        print("no prediction files found under normal/pred_marigold")
        return 2
    gt_self_consistency(args.out, ids[0])
    sign_perm_search(args.out, ids, permute=args.permute)
    P, G, per = _gather(args.out, ids)
    kabsch_residual(P, G)
    flip_search(per)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
