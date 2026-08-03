"""Dataset-level file evaluation shared by the unified CLI."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from perception_diffusion.codecs.base import ensure_chw3
from perception_diffusion.evaluation.depth import DepthEvaluator
from perception_diffusion.evaluation.normal import NormalEvaluator
from perception_diffusion.evaluation.segmentation import SegmentationEvaluator
from perception_diffusion.task_specs import TaskSpec


SUPPORTED_SUFFIXES = {".npy", ".npz", ".png", ".tif", ".tiff"}
NORMAL_ENCODINGS = {"unit", "zero_one", "uint8"}


@dataclass(frozen=True)
class EvaluationInputs:
    prediction_root: Path
    target_root: Path
    valid_mask_root: Path | None = None
    prediction_key: str | None = None
    target_key: str | None = None
    valid_mask_key: str | None = None
    prediction_depth_scale: float = 1.0
    target_depth_scale: float = 1.0
    prediction_normal_encoding: str = "unit"
    target_normal_encoding: str = "unit"
    prediction_label_offset: int = 0
    target_label_offset: int = 0
    prediction_ignore_value: int | None = None
    target_ignore_value: int | None = None
    limit: int | None = None


@dataclass(frozen=True)
class ArrayPair:
    sample_id: str
    prediction: Path
    target: Path
    valid_mask: Path | None


def _discover(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise NotADirectoryError(root)
    discovered: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        sample_id = path.relative_to(root).with_suffix("").as_posix()
        if sample_id in discovered:
            raise ValueError(
                f"duplicate sample ID {sample_id!r}: {discovered[sample_id]} and {path}"
            )
        discovered[sample_id] = path
    if not discovered:
        suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"no supported arrays found under {root}; expected {suffixes}")
    return discovered


def pair_array_roots(inputs: EvaluationInputs) -> list[ArrayPair]:
    predictions = _discover(inputs.prediction_root)
    targets = _discover(inputs.target_root)
    prediction_ids = set(predictions)
    target_ids = set(targets)
    if prediction_ids != target_ids:
        missing_predictions = sorted(target_ids - prediction_ids)
        missing_targets = sorted(prediction_ids - target_ids)
        raise ValueError(
            "prediction/target sample IDs differ; "
            f"missing predictions={missing_predictions[:10]}, "
            f"missing targets={missing_targets[:10]}"
        )

    masks: dict[str, Path] | None = None
    if inputs.valid_mask_root is not None:
        masks = _discover(inputs.valid_mask_root)
        mask_ids = set(masks)
        if mask_ids != prediction_ids:
            missing_masks = sorted(prediction_ids - mask_ids)
            extra_masks = sorted(mask_ids - prediction_ids)
            raise ValueError(
                "valid-mask sample IDs differ; "
                f"missing masks={missing_masks[:10]}, extra masks={extra_masks[:10]}"
            )

    sample_ids = sorted(prediction_ids)
    if inputs.limit is not None:
        if inputs.limit <= 0:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[: inputs.limit]
    return [
        ArrayPair(
            sample_id=sample_id,
            prediction=predictions[sample_id],
            target=targets[sample_id],
            valid_mask=None if masks is None else masks[sample_id],
        )
        for sample_id in sample_ids
    ]


def _load_array(path: Path, key: str | None) -> NDArray[np.generic]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is not None:
                if key not in archive.files:
                    raise KeyError(f"{path}: no array named {key!r}; found {archive.files}")
                return np.asarray(archive[key])
            if len(archive.files) != 1:
                raise ValueError(
                    f"{path}: NPZ contains {archive.files}; provide the corresponding --*-key"
                )
            return np.asarray(archive[archive.files[0]])
    with Image.open(path) as image:
        return np.asarray(image)


def _prepare_mask(array: NDArray[np.generic], sample_id: str) -> NDArray[np.bool_]:
    mask = np.squeeze(np.asarray(array))
    if mask.ndim != 2:
        raise ValueError(f"{sample_id}: valid mask must be HxW, got {mask.shape}")
    if np.issubdtype(mask.dtype, np.floating) and not np.all(np.isfinite(mask)):
        raise ValueError(f"{sample_id}: valid mask contains non-finite values")
    return np.asarray(mask != 0, dtype=bool)


def _prepare_segmentation(
    array: NDArray[np.generic],
    *,
    sample_id: str,
    label_offset: int,
    input_ignore_value: int | None,
    evaluator_ignore_label: int,
) -> NDArray[np.int64]:
    labels = np.squeeze(np.asarray(array))
    if labels.ndim != 2:
        raise ValueError(f"{sample_id}: segmentation array must be HxW, got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{sample_id}: segmentation labels must use an integer dtype")
    result = labels.astype(np.int64, copy=True)
    ignored = np.zeros(result.shape, dtype=bool)
    if input_ignore_value is not None:
        ignored = result == input_ignore_value
    if label_offset:
        result[~ignored] += label_offset
    result[ignored] = evaluator_ignore_label
    return result


def _prepare_depth(
    array: NDArray[np.generic], *, sample_id: str, scale: float
) -> NDArray[np.float32]:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth scale must be finite and positive")
    depth = np.squeeze(np.asarray(array))
    if depth.ndim != 2:
        raise ValueError(f"{sample_id}: depth array must be HxW, got {depth.shape}")
    return np.asarray(depth, dtype=np.float32) * np.float32(scale)


def _prepare_normal(
    array: NDArray[np.generic], *, sample_id: str, encoding: str
) -> NDArray[np.float32]:
    if encoding not in NORMAL_ENCODINGS:
        raise ValueError(f"unsupported normal encoding: {encoding}")
    values = np.asarray(array)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    chw = ensure_chw3(values, f"{sample_id} surface normals")
    if encoding == "zero_one":
        chw = 2.0 * chw - 1.0
    elif encoding == "uint8":
        chw = chw / 127.5 - 1.0
    return np.asarray(chw, dtype=np.float32)


def _scalar_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Remove large array-valued diagnostics from per-sample records."""

    return {
        key: value
        for key, value in result.items()
        if key not in {"confusion_matrix", "per_class_iou"}
    }


def _aggregate_depth(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate_section(section: str) -> dict[str, float | int]:
        records = [sample["metrics"][section] for sample in samples]
        total = sum(int(record["valid_pixels"]) for record in records)
        if total <= 0:
            raise ValueError("depth aggregation has no valid pixels")
        abs_rel = sum(float(record["abs_rel"]) * int(record["valid_pixels"]) for record in records)
        delta1 = sum(float(record["delta1"]) * int(record["valid_pixels"]) for record in records)
        squared_error = sum(
            float(record["rmse"]) ** 2 * int(record["valid_pixels"])
            for record in records
        )
        return {
            "abs_rel": abs_rel / total,
            "delta1": delta1 / total,
            "rmse": float(np.sqrt(squared_error / total)),
            "valid_pixels": total,
        }

    summary: dict[str, Any] = {"raw": aggregate_section("raw")}
    if "affine_aligned" in samples[0]["metrics"]:
        summary["affine_aligned"] = aggregate_section("affine_aligned")
        summary["alignment"] = {
            "scale_mean": float(
                np.mean([sample["metrics"]["alignment"]["scale"] for sample in samples])
            ),
            "shift_mean": float(
                np.mean([sample["metrics"]["alignment"]["shift"] for sample in samples])
            ),
        }
    return summary


def _aggregate_normals(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(sample["metrics"]["valid_pixels"]) for sample in samples)
    if total <= 0:
        raise ValueError("normal aggregation has no valid pixels")

    def weighted(metric: str) -> float:
        return sum(
            float(sample["metrics"][metric]) * int(sample["metrics"]["valid_pixels"])
            for sample in samples
        ) / total

    return {
        "mean_angular_error": weighted("mean_angular_error"),
        "median_angular_error_macro": float(
            np.mean([sample["metrics"]["median_angular_error"] for sample in samples])
        ),
        "acc_11_25": weighted("acc_11_25"),
        "acc_22_5": weighted("acc_22_5"),
        "acc_30": weighted("acc_30"),
        "valid_pixels": total,
        "median_aggregation": "mean_of_per_sample_medians",
    }


def evaluate_array_roots(
    spec: TaskSpec,
    inputs: EvaluationInputs,
    *,
    config_path: Path,
    resolved_config: dict[str, Any],
) -> dict[str, Any]:
    pairs = pair_array_roots(inputs)
    samples: list[dict[str, Any]] = []
    evaluator = spec.evaluator
    if isinstance(evaluator, SegmentationEvaluator):
        evaluator.reset()

    for pair in pairs:
        prediction_raw = _load_array(pair.prediction, inputs.prediction_key)
        target_raw = _load_array(pair.target, inputs.target_key)
        valid_mask = (
            None
            if pair.valid_mask is None
            else _prepare_mask(
                _load_array(pair.valid_mask, inputs.valid_mask_key), pair.sample_id
            )
        )

        if spec.name == "segmentation":
            if not isinstance(evaluator, SegmentationEvaluator):
                raise TypeError("segmentation TaskSpec has the wrong evaluator")
            prediction = _prepare_segmentation(
                prediction_raw,
                sample_id=pair.sample_id,
                label_offset=inputs.prediction_label_offset,
                input_ignore_value=inputs.prediction_ignore_value,
                evaluator_ignore_label=evaluator.ignore_label,
            )
            target = _prepare_segmentation(
                target_raw,
                sample_id=pair.sample_id,
                label_offset=inputs.target_label_offset,
                input_ignore_value=inputs.target_ignore_value,
                evaluator_ignore_label=evaluator.ignore_label,
            )
            evaluator.update(prediction, target, valid_mask)
            sample_result = SegmentationEvaluator(
                evaluator.num_classes, evaluator.ignore_label
            ).evaluate(prediction, target, valid_mask)
        elif spec.name == "depth":
            if not isinstance(evaluator, DepthEvaluator):
                raise TypeError("depth TaskSpec has the wrong evaluator")
            prediction = _prepare_depth(
                prediction_raw,
                sample_id=pair.sample_id,
                scale=inputs.prediction_depth_scale,
            )
            target = _prepare_depth(
                target_raw,
                sample_id=pair.sample_id,
                scale=inputs.target_depth_scale,
            )
            sample_result = evaluator.evaluate(prediction, target, valid_mask)
        elif spec.name == "normal":
            if not isinstance(evaluator, NormalEvaluator):
                raise TypeError("normal TaskSpec has the wrong evaluator")
            prediction = _prepare_normal(
                prediction_raw,
                sample_id=pair.sample_id,
                encoding=inputs.prediction_normal_encoding,
            )
            target = _prepare_normal(
                target_raw,
                sample_id=pair.sample_id,
                encoding=inputs.target_normal_encoding,
            )
            sample_result = evaluator.evaluate(prediction, target, valid_mask)
        else:
            raise ValueError(f"unsupported task: {spec.name}")

        samples.append(
            {
                "sample_id": pair.sample_id,
                "prediction": str(pair.prediction.resolve()),
                "target": str(pair.target.resolve()),
                "valid_mask": (
                    None if pair.valid_mask is None else str(pair.valid_mask.resolve())
                ),
                "metrics": _scalar_metrics(sample_result),
            }
        )

    if isinstance(evaluator, SegmentationEvaluator):
        summary = evaluator.compute()
        aggregation = "dataset_confusion_matrix"
    elif isinstance(evaluator, DepthEvaluator):
        summary = _aggregate_depth(samples)
        aggregation = "valid_pixel_weighted; affine alignment is per sample"
    else:
        summary = _aggregate_normals(samples)
        aggregation = (
            "valid_pixel_weighted except median_angular_error_macro, which is the "
            "mean of exact per-sample medians"
        )

    resolved_json = json.dumps(
        resolved_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "status": "EVALUATION_COMPLETED",
        "evidence_level": "metrics_only_not_a_formal_experiment_claim",
        "task": spec.name,
        "dataset": spec.dataset_name,
        "config": str(config_path.resolve()),
        "resolved_config_sha256": hashlib.sha256(resolved_json).hexdigest(),
        "input": {
            "prediction_root": str(inputs.prediction_root.resolve()),
            "target_root": str(inputs.target_root.resolve()),
            "valid_mask_root": (
                None
                if inputs.valid_mask_root is None
                else str(inputs.valid_mask_root.resolve())
            ),
            "sample_count": len(samples),
        },
        "aggregation": aggregation,
        "metrics": summary,
        "samples": samples,
    }


def _flatten_scalars(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "confusion_matrix":
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_scalars(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}"
            yield from _flatten_scalars(item, child)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        yield prefix, value


def write_evaluation_outputs(
    result: dict[str, Any], output_dir: Path, *, overwrite: bool = False
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {[str(path) for path in existing]}; pass --overwrite"
        )

    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("scope", "sample_id", "metric", "value")
        )
        writer.writeheader()
        for metric, value in _flatten_scalars(result["metrics"]):
            writer.writerow(
                {"scope": "dataset", "sample_id": "", "metric": metric, "value": value}
            )
        for sample in result["samples"]:
            for metric, value in _flatten_scalars(sample["metrics"]):
                writer.writerow(
                    {
                        "scope": "sample",
                        "sample_id": sample["sample_id"],
                        "metric": metric,
                        "value": value,
                    }
                )
    return json_path, csv_path
