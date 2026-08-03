"""Configuration loading and structural validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from perception_diffusion.tasks import TASK_NAMES, validate_task_names


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the project schema."""


REQUIRED_TOP_LEVEL = {
    "experiment",
    "task",
    "model",
    "data",
    "training",
    "inference",
    "evaluation",
    "runtime",
}
ALLOWED_TASKS = set(TASK_NAMES)
ALLOWED_EXPERIMENT_TASKS = ALLOWED_TASKS | {"multitask"}


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_with_defaults(
    config_path: Path, *, stack: tuple[Path, ...] = ()
) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    resolved_path = config_path.resolve()
    if resolved_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, resolved_path))
        raise ConfigError(f"configuration defaults contain a cycle: {cycle}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    defaults = loaded.pop("defaults", [])
    if not isinstance(defaults, list) or not all(
        isinstance(item, str) for item in defaults
    ):
        raise ConfigError(f"{config_path}: defaults must be a list of YAML paths")

    merged: dict[str, Any] = {}
    for default in defaults:
        default_path = (config_path.parent / default).resolve()
        merged = _deep_merge(
            merged,
            _load_with_defaults(default_path, stack=(*stack, resolved_path)),
        )
    return _deep_merge(merged, loaded)


def load_config(path: str | Path, *, expand_environment: bool = True) -> dict[str, Any]:
    config_path = Path(path)
    loaded = _load_with_defaults(config_path)
    config = _expand_environment(loaded) if expand_environment else loaded
    validate_config(config, source=str(config_path))
    return config


def configured_task_names(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the canonical tasks selected by a single- or multi-task config."""

    task_config = config.get("task")
    if not isinstance(task_config, dict):
        raise ConfigError("task must be a mapping")
    task_name = task_config.get("name")
    if task_name == "multitask":
        tasks = task_config.get("tasks")
        if not isinstance(tasks, list):
            raise ConfigError("multitask configuration requires task.tasks as a list")
        try:
            return validate_task_names(tasks)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    if task_name in ALLOWED_TASKS:
        return (task_name,)
    raise ConfigError(f"task.name must be one of {sorted(ALLOWED_EXPERIMENT_TASKS)}")


def _require_positive_integer(value: Any, label: str, source: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{source}: {label} must be a positive integer")


def _validate_model_config(
    model: Any, task_names: tuple[str, ...], *, source: str
) -> None:
    if not isinstance(model, dict):
        raise ConfigError(f"{source}: model must be a mapping")
    for section in (
        "backbone",
        "conditioning",
        "shared_unet",
        "condition_adapter",
        "target_adapter",
    ):
        if not isinstance(model.get(section), dict):
            raise ConfigError(f"{source}: model.{section} must be a mapping")

    backbone = model["backbone"]
    if not backbone.get("pretrained_model_path"):
        raise ConfigError(f"{source}: model.backbone.pretrained_model_path is required")
    _require_positive_integer(
        backbone.get("image_latent_channels"),
        "model.backbone.image_latent_channels",
        source,
    )
    _require_positive_integer(
        backbone.get("target_latent_channels"),
        "model.backbone.target_latent_channels",
        source,
    )
    if backbone.get("conv_in_initialization") not in {
        "repeat_half",
        "preserve_image_zero_target",
    }:
        raise ConfigError(
            f"{source}: unsupported model.backbone.conv_in_initialization"
        )

    conditioning = model["conditioning"]
    if conditioning.get("type") != "task_token_cross_attention":
        raise ConfigError(
            f"{source}: model.conditioning.type must be task_token_cross_attention"
        )
    try:
        configured_condition_tasks = validate_task_names(
            conditioning.get("task_names", [])
        )
    except ValueError as exc:
        raise ConfigError(f"{source}: model.conditioning.task_names: {exc}") from exc
    missing_condition_tasks = sorted(set(task_names) - set(configured_condition_tasks))
    if missing_condition_tasks:
        raise ConfigError(
            f"{source}: conditioner is missing task tokens for {missing_condition_tasks}"
        )
    for field in ("cross_attention_dim", "text_input_dim", "num_task_tokens"):
        _require_positive_integer(
            conditioning.get(field), f"model.conditioning.{field}", source
        )
    dropout = conditioning.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
        raise ConfigError(f"{source}: model.conditioning.dropout must be numeric")
    if not 0.0 <= float(dropout) < 1.0:
        raise ConfigError(f"{source}: model.conditioning.dropout must be in [0,1)")

    if model["shared_unet"].get("trainable_scope") not in {
        "full",
        "cross_attention",
        "frozen",
    }:
        raise ConfigError(f"{source}: unsupported model.shared_unet.trainable_scope")

    condition_adapter = model["condition_adapter"]
    if condition_adapter.get("type") != "residual_bottleneck":
        raise ConfigError(
            f"{source}: model.condition_adapter.type must be residual_bottleneck"
        )
    if condition_adapter.get("placement") != "condition_tokens":
        raise ConfigError(
            f"{source}: model.condition_adapter.placement must be condition_tokens"
        )
    if condition_adapter.get("per_task") is not True:
        raise ConfigError(f"{source}: model.condition_adapter.per_task must be true")
    _require_positive_integer(
        condition_adapter.get("bottleneck_dim"),
        "model.condition_adapter.bottleneck_dim",
        source,
    )
    residual_scale = condition_adapter.get("residual_scale_init")
    if not isinstance(residual_scale, (int, float)) or isinstance(residual_scale, bool):
        raise ConfigError(
            f"{source}: model.condition_adapter.residual_scale_init must be numeric"
        )

    target_adapter = model["target_adapter"]
    if not isinstance(target_adapter.get("enabled"), bool):
        raise ConfigError(f"{source}: model.target_adapter.enabled must be boolean")
    if target_adapter.get("type") != "residual_cnn":
        raise ConfigError(f"{source}: model.target_adapter.type must be residual_cnn")
    if target_adapter.get("placement") != "pre_vae":
        raise ConfigError(f"{source}: model.target_adapter.placement must be pre_vae")
    if target_adapter.get("per_task") is not True:
        raise ConfigError(f"{source}: model.target_adapter.per_task must be true")
    try:
        target_adapter_tasks = validate_task_names(target_adapter.get("task_names", []))
    except ValueError as exc:
        raise ConfigError(f"{source}: model.target_adapter.task_names: {exc}") from exc
    missing_target_tasks = sorted(set(task_names) - set(target_adapter_tasks))
    if missing_target_tasks:
        raise ConfigError(
            f"{source}: target adapter is missing tasks {missing_target_tasks}"
        )
    for field in ("channels", "hidden_channels", "num_blocks"):
        _require_positive_integer(
            target_adapter.get(field), f"model.target_adapter.{field}", source
        )
    if target_adapter.get("channels") != 3:
        raise ConfigError(f"{source}: model.target_adapter.channels must be 3")
    target_scale = target_adapter.get("residual_scale_init")
    if not isinstance(target_scale, (int, float)) or isinstance(target_scale, bool):
        raise ConfigError(
            f"{source}: model.target_adapter.residual_scale_init must be numeric"
        )
    output_range = target_adapter.get("output_range")
    if output_range != [-1.0, 1.0]:
        raise ConfigError(
            f"{source}: model.target_adapter.output_range must be [-1.0, 1.0]"
        )


def _validate_runtime(runtime: Any, *, source: str) -> None:
    if not isinstance(runtime, dict):
        raise ConfigError(f"{source}: runtime must be a mapping")
    if runtime.get("device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError(f"{source}: runtime.device must be auto, cpu, or cuda")
    if runtime.get("distributed") not in {"disabled", "ddp"}:
        raise ConfigError(f"{source}: runtime.distributed must be disabled or ddp")
    if not isinstance(runtime.get("compile"), bool):
        raise ConfigError(f"{source}: runtime.compile must be boolean")


def _validate_segmentation_evaluation(
    config: dict[str, Any], task_names: tuple[str, ...], *, source: str
) -> None:
    if "segmentation" not in task_names:
        return
    if config["task"]["name"] == "multitask":
        evaluation = config["evaluation"].get("segmentation")
    else:
        evaluation = config["evaluation"]
    if not isinstance(evaluation, dict):
        raise ConfigError(f"{source}: segmentation evaluation must be a mapping")
    num_classes = evaluation.get("num_classes")
    _require_positive_integer(num_classes, "evaluation.num_classes", source)
    vocabulary = evaluation.get("vocabulary")
    if not isinstance(vocabulary, dict):
        raise ConfigError(f"{source}: segmentation evaluation.vocabulary is required")
    if vocabulary.get("source") not in {"ade20k_object_info", "inline"}:
        raise ConfigError(f"{source}: unsupported segmentation vocabulary source")
    if vocabulary.get("expected_num_classes") != num_classes:
        raise ConfigError(
            f"{source}: vocabulary class count must match evaluation.num_classes"
        )
    if vocabulary.get("source") == "ade20k_object_info" and not vocabulary.get("path"):
        raise ConfigError(f"{source}: ADE20K vocabulary metadata path is required")
    if vocabulary.get("source") == "inline":
        class_names = vocabulary.get("class_names")
        if not isinstance(class_names, list) or len(class_names) != num_classes:
            raise ConfigError(
                f"{source}: inline vocabulary must contain evaluation.num_classes names"
            )
    query = evaluation.get("query")
    if not isinstance(query, dict):
        raise ConfigError(f"{source}: segmentation evaluation.query is required")
    if query.get("mode") != "closed_set_all_classes":
        raise ConfigError(
            f"{source}: standard segmentation evaluation must query all dataset classes"
        )
    template = query.get("prompt_template")
    if not isinstance(template, str) or "{class_name}" not in template:
        raise ConfigError(
            f"{source}: segmentation prompt_template must contain {{class_name}}"
        )
    _require_positive_integer(
        query.get("batch_size"), "evaluation.query.batch_size", source
    )


def _validate_task_boundaries(
    config: dict[str, Any], task_names: tuple[str, ...], *, source: str
) -> None:
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    if not isinstance(data_config, dict):
        raise ConfigError(f"{source}: data must be a mapping")
    if not isinstance(evaluation_config, dict):
        raise ConfigError(f"{source}: evaluation must be a mapping")
    is_multitask = config["task"]["name"] == "multitask"
    datasets = data_config.get("datasets") if is_multitask else None
    if is_multitask and not isinstance(datasets, dict):
        raise ConfigError(f"{source}: multitask data.datasets must be a mapping")
    for task_name in task_names:
        if is_multitask:
            task_data = datasets.get(task_name)
            if not isinstance(task_data, dict):
                raise ConfigError(f"{source}: data.datasets.{task_name} is required")
            codec = task_data.get("codec")
            evaluation = evaluation_config.get(task_name)
        else:
            task_data = data_config
            codec = config["task"].get("codec")
            evaluation = evaluation_config
        if not isinstance(codec, dict):
            raise ConfigError(f"{source}: {task_name} codec must be a mapping")
        if not isinstance(evaluation, dict):
            raise ConfigError(f"{source}: {task_name} evaluation must be a mapping")

        dataset_name = task_data.get("name" if is_multitask else "dataset")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ConfigError(f"{source}: {task_name} dataset name is required")
        if dataset_name != "synthetic":
            root = task_data.get("root")
            if not isinstance(root, str) or not root:
                raise ConfigError(f"{source}: {task_name} data.root is required")
        image_size = task_data.get("image_size")
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in image_size
            )
        ):
            raise ConfigError(f"{source}: {task_name} data.image_size must be [H,W]")
        num_workers = task_data.get("num_workers", data_config.get("num_workers", 0))
        if (
            not isinstance(num_workers, int)
            or isinstance(num_workers, bool)
            or num_workers < 0
        ):
            raise ConfigError(f"{source}: {task_name} data.num_workers must be non-negative")
        flip_probability = task_data.get("horizontal_flip_probability", 0.0)
        if (
            not isinstance(flip_probability, (int, float))
            or isinstance(flip_probability, bool)
            or not 0.0 <= float(flip_probability) <= 1.0
        ):
            raise ConfigError(
                f"{source}: {task_name} horizontal_flip_probability must be in [0,1]"
            )

        if task_name == "segmentation":
            if codec.get("name") not in {
                "palette",
                "normalized_id",
                "binary_query_mask",
            }:
                raise ConfigError(f"{source}: unsupported segmentation codec")
            _require_positive_integer(
                codec.get("num_classes"), "segmentation codec.num_classes", source
            )
            if codec.get("num_classes") != evaluation.get("num_classes"):
                raise ConfigError(
                    f"{source}: segmentation codec/evaluation class counts differ"
                )
        elif task_name == "depth":
            if codec.get("name") != "affine_invariant_depth":
                raise ConfigError(f"{source}: unsupported depth codec")
            if codec.get("representation", "linear") not in {"linear", "inverse", "log"}:
                raise ConfigError(f"{source}: unsupported depth representation")
            minimum = codec.get("min_depth")
            maximum = codec.get("max_depth")
            if not isinstance(minimum, (int, float)) or not isinstance(
                maximum, (int, float)
            ):
                raise ConfigError(f"{source}: depth bounds must be numeric")
            if float(minimum) <= 0 or float(maximum) <= float(minimum):
                raise ConfigError(f"{source}: depth bounds must satisfy 0 < min < max")
        elif task_name == "normal":
            if codec.get("name") != "unit_vector":
                raise ConfigError(f"{source}: unsupported normal codec")
            if codec.get("channel_order") != "xyz":
                raise ConfigError(f"{source}: normal channel_order must be xyz")
            if not isinstance(codec.get("flip_y"), bool):
                raise ConfigError(f"{source}: normal flip_y must be boolean")

        if dataset_name == "nyuv2":
            if task_data.get("source", "auto") not in {"auto", "raw", "processed"}:
                raise ConfigError(f"{source}: unsupported NYUv2 data.source")
            if task_data.get("depth_field", "depths") not in {"depths", "rawDepths"}:
                raise ConfigError(f"{source}: unsupported NYUv2 depth_field")
        if dataset_name == "ade20k":
            if task_data.get("query_sampling", "mixed") not in {
                "mixed",
                "uniform",
                "fixed",
            }:
                raise ConfigError(f"{source}: unsupported ADE20K training query_sampling")
            if task_data.get("evaluation_query_sampling", "fixed") != "fixed":
                raise ConfigError(
                    f"{source}: ADE20K evaluation query placeholder must be GT-independent fixed"
                )
            fixed_query = task_data.get("fixed_query_class_id", 0)
            if (
                not isinstance(fixed_query, int)
                or isinstance(fixed_query, bool)
                or not 0 <= fixed_query < 150
            ):
                raise ConfigError(
                    f"{source}: ADE20K fixed_query_class_id must lie in [0,149]"
                )


def validate_config(config: dict[str, Any], *, source: str = "<memory>") -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ConfigError(f"{source}: missing top-level keys: {missing}")
    if not isinstance(config["task"], dict):
        raise ConfigError(f"{source}: task must be a mapping")
    task_names = configured_task_names(config)
    _validate_runtime(config["runtime"], source=source)
    _validate_task_boundaries(config, task_names, source=source)
    experiment = config["experiment"]
    if not isinstance(experiment, dict) or not experiment.get("name"):
        raise ConfigError(f"{source}: experiment.name is required")
    seed = experiment.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"{source}: experiment.seed must be a non-negative integer")
    training = config["training"]
    if not isinstance(training, dict):
        raise ConfigError(f"{source}: training must be a mapping")
    max_steps = training.get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ConfigError(f"{source}: training.max_steps must be a positive integer")
    task_batch_homogeneous = training.get("task_batch_homogeneous")
    if task_batch_homogeneous is not True:
        raise ConfigError(f"{source}: training.task_batch_homogeneous must be true")
    inference = config["inference"]
    steps = inference.get("num_steps") if isinstance(inference, dict) else None
    if not isinstance(steps, int) or steps <= 0:
        raise ConfigError(f"{source}: inference.num_steps must be a positive integer")
    _validate_model_config(config["model"], task_names, source=source)
    _validate_segmentation_evaluation(config, task_names, source=source)
