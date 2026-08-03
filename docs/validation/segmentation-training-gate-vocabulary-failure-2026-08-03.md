# ADE20K Training Gate Vocabulary Failure

Date: 2026-08-03 Asia/Shanghai

Experiment registry IDs: `S005`, `S005R`

## Failure

The user-operated single-GPU segmentation training gate failed during `ADE20KDataset` initialization, before model loading, CUDA backward, or optimizer execution. The returned exception was `ValueError: vocabulary class names must be unique` from `load_ade20k_object_info`.

## Root cause

The official SceneParse150 `objectInfo150.txt` uses five whitespace-separated columns with `Name` last. Names may contain spaces and comma-separated synonyms. The loader used a hard-coded `maxsplit=5`, which treated the first word of a multiword name as a complete fifth field and placed the remainder in an unused sixth field. For example, `screen door, screen` became `screen`, colliding with the distinct `screen, silver screen, projection screen` class.

The failure was a parser-induced duplicate, not a duplicate class ID, corrupt mask, excluded sample, model failure, CUDA failure, or DDP failure.

## Fix and local evidence

The parser now reads the header independently and uses `maxsplit=expected_columns-1` for whitespace rows. It also requires `Name` to be the final column when no tab delimiter is present. Tab-separated fixtures with other column orders remain supported.

A regression fixture covers `screen door`, `screen`, and `chest of drawers`. Targeted dataset/query tests passed (11 tests), followed by the full local suite (49 tests), nine config validations, and the harness smoke.

## Remaining gate

Deploy the parser change to the server and rerun the one-step 256 px single-GPU gate as `S005R`. Do not start the DDP gate until `SEGMENTATION_TRAINING_GATE_PASSED` is returned.
