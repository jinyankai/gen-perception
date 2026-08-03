"""Configuration-driven task registry for codecs, evaluators, and query policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from perception_diffusion.codecs import (
    DepthCodec,
    NormalCodec,
    SegmentationBinaryMaskCodec,
    SegmentationIdCodec,
    SegmentationPaletteCodec,
)
from perception_diffusion.data import load_segmentation_vocabulary
from perception_diffusion.evaluation import (
    DepthEvaluator,
    NormalEvaluator,
    SegmentationEvaluator,
)
from perception_diffusion.inference.segmentation_queries import SegmentationQueryPlanner
from perception_diffusion.tasks import TASK_NAMES
from perception_diffusion.utils.config import configured_task_names


@dataclass(frozen=True)
class TaskSpec:
    """The task-owned pieces consumed by shared training and inference code."""

    name: str
    dataset_name: str
    codec: object
    evaluator: object
    query_config: dict[str, Any] | None = None


def _build_codec(task_name: str, config: dict[str, Any]) -> object:
    codec_name = config.get("name")
    if task_name == "segmentation":
        num_classes = int(config["num_classes"])
        ignore_label = int(config.get("ignore_label", 255))
        if codec_name == "palette":
            return SegmentationPaletteCodec(num_classes, ignore_label)
        if codec_name == "normalized_id":
            return SegmentationIdCodec(num_classes, ignore_label)
        if codec_name == "binary_query_mask":
            return SegmentationBinaryMaskCodec(
                threshold=float(config.get("threshold", 0.5))
            )
    elif task_name == "depth" and codec_name == "affine_invariant_depth":
        return DepthCodec(
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
            representation=str(config.get("representation", "linear")),
        )
    elif task_name == "normal" and codec_name == "unit_vector":
        return NormalCodec()
    raise ValueError(f"unsupported codec for {task_name}: {codec_name}")


def _build_evaluator(task_name: str, config: dict[str, Any]) -> object:
    if task_name == "segmentation":
        return SegmentationEvaluator(
            num_classes=int(config["num_classes"]),
            ignore_label=int(config.get("ignore_label", 255)),
        )
    if task_name == "depth":
        return DepthEvaluator(
            min_depth=float(config.get("min_depth", 1.0e-3)),
            max_depth=(
                None if config.get("max_depth") is None else float(config["max_depth"])
            ),
            affine_align=bool(config.get("affine_align", True)),
        )
    if task_name == "normal":
        return NormalEvaluator()
    raise ValueError(f"unsupported evaluator task: {task_name}")


def build_task_specs(config: dict[str, Any]) -> dict[str, TaskSpec]:
    """Build all task-specific boundaries without creating task-specific trainers."""

    specs: dict[str, TaskSpec] = {}
    selected_tasks = configured_task_names(config)
    is_multitask = config["task"]["name"] == "multitask"
    for task_name in selected_tasks:
        if task_name not in TASK_NAMES:
            raise ValueError(f"unsupported task: {task_name}")
        if is_multitask:
            task_data = config["data"]["datasets"][task_name]
            evaluation = config["evaluation"][task_name]
            codec_config = task_data["codec"]
            dataset_name = str(task_data["name"])
        else:
            task_data = config["data"]
            evaluation = config["evaluation"]
            codec_config = config["task"]["codec"]
            dataset_name = str(task_data["dataset"])
        specs[task_name] = TaskSpec(
            name=task_name,
            dataset_name=dataset_name,
            codec=_build_codec(task_name, codec_config),
            evaluator=_build_evaluator(task_name, evaluation),
            query_config=(
                {
                    "query": dict(evaluation["query"]),
                    "vocabulary": dict(evaluation["vocabulary"]),
                }
                if task_name == "segmentation"
                else None
            ),
        )
    return specs


def build_segmentation_query_planner(spec: TaskSpec) -> SegmentationQueryPlanner:
    """Resolve the declared dataset taxonomy only when segmentation runs."""

    if spec.name != "segmentation" or spec.query_config is None:
        raise ValueError("a segmentation TaskSpec with query configuration is required")
    vocabulary = load_segmentation_vocabulary(spec.query_config["vocabulary"])
    query_config = spec.query_config["query"]
    if query_config.get("mode") != "closed_set_all_classes":
        raise ValueError("standard evaluation requires closed_set_all_classes")
    return SegmentationQueryPlanner(
        vocabulary,
        prompt_template=str(query_config["prompt_template"]),
    )
