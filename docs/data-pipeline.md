# Stage-One Data Pipeline

This pipeline covers ADE20K semantic segmentation and NYUv2 monocular depth / derived surface normals. Dataset code emits task-homogeneous batches; it does not download anything during training.

## Unified batch schema

Every DataLoader returns:

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `image` | `float32 [B,3,H,W]` | RGB normalized to `[-1,1]` for the frozen VAE |
| `target` | `float32 [B,3,H,W]` | task codec output in `[-1,1]` |
| `valid_mask` | `bool [B,1,H,W]` | pixels included in target loss/evaluation |
| `native_target` | task-native tensor | 0-based ADE IDs, metric depth, or camera-space unit normals |
| `task_name` | scalar string | enforces task-homogeneous batches |
| `sample_id` | list of strings | stable source identifiers |
| `source_index` | `int64 [B]` | zero-based source index |
| `original_size` | `int64 [B,2]` | source `H,W` before resizing |
| `query_class_id` | `int64 [B]` | ADE binary-query class; `-1` otherwise |
| `text_condition` | list of strings | task/class prompt for conditioning |

ADEChallenge PNG IDs `1..150` are remapped to model IDs `0..149`; source ID `0` becomes ignore ID `255`. Binary-query training uses a configurable mixture of present-class and taxonomy-uniform queries. The evaluation batch carries a fixed, ground-truth-independent placeholder query target while formal query planning remains closed-set over all 150 metadata classes and consumes the full `native_target` label map.

NYUv2 depth is read in meters from the official labeled HDF5/MAT file and indexed by `trainNdxs` / `testNdxs` from `splits.mat` (795 train, 654 test). Surface normals are derived from depth with the official RGB intrinsics. Their convention is camera coordinates `x-right, y-down, z-forward`, channel order `xyz`, oriented toward the camera. Invalid borders, degenerate neighborhoods, and depth discontinuities are excluded.

## Download

Set the approved server data root first:

```bash
export DATA_ROOT=/home/jinyankai/data
```

Read and accept the [ADE20K terms](https://groups.csail.mit.edu/vision/ADE20K/terms/) before running:

```bash
python scripts/data/download_ade20k.py \
  --data-root "$DATA_ROOT" \
  --accept-terms \
  --keep-archive
```

If ADE20K was downloaded through the registration page, pass it without fetching again:

```bash
python scripts/data/download_ade20k.py \
  --data-root "$DATA_ROOT" \
  --archive /path/to/ADEChallengeData2016.zip \
  --accept-terms
```

Download the official NYUv2 labeled file and official split:

```bash
python scripts/data/download_nyuv2.py --data-root "$DATA_ROOT"
```

Both downloaders resume through `.part` files, validate stable source files, and write provenance manifests. They refuse to silently replace an existing prepared dataset.

## NYUv2 preprocessing

Materialize per-sample RGB, metric depth, validity masks, and derived normals:

```bash
python scripts/data/preprocess_nyuv2.py \
  --root "$DATA_ROOT/nyuv2" \
  --split all
```

Output is written to `nyuv2/processed/{train,test}/*.npz` plus `processed/manifest.json`. The command is resumable: valid existing samples are skipped. Use `--overwrite` only to intentionally regenerate files atomically.

`source: auto` in the task configs prefers complete processed samples and falls back to direct MAT reading. Set `--nyuv2-source processed` in the smoke command to require and verify the preprocessed path.

## Real-sample DataLoader tests

Run each task separately so failures are isolated:

```bash
python scripts/data/smoke_test_datasets.py \
  --task segmentation --data-root "$DATA_ROOT" --num-workers 0

python scripts/data/smoke_test_datasets.py \
  --task depth --data-root "$DATA_ROOT" --nyuv2-source processed --num-workers 0

python scripts/data/smoke_test_datasets.py \
  --task normal --data-root "$DATA_ROOT" --nyuv2-source processed --num-workers 0
```

Then exercise worker-safe HDF5/NPZ loading with the configured worker count:

```bash
python scripts/data/smoke_test_datasets.py \
  --task all --data-root "$DATA_ROOT" --nyuv2-source auto --num-workers 8 --batch-size 2
```

The command reports shapes, dtypes, ranges, valid fractions, source mode, sample IDs, and task-specific label/depth/unit-normal checks. It is a data smoke test, not a benchmark or training result.
