"""Configuration-driven Dataset and DataLoader construction."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from perception_diffusion.tasks import TASK_NAMES

from .ade20k import ADE20KDataset
from .nyuv2 import NYUv2Dataset
from .schema import collate_task_samples


def _task_data_config(config: Mapping[str, Any], task_name: str) -> Mapping[str, Any]:
    if task_name not in TASK_NAMES:
        raise ValueError(f"unsupported task: {task_name}")
    task_config = config.get("task")
    data_config = config.get("data")
    if not isinstance(task_config, Mapping) or not isinstance(data_config, Mapping):
        raise ValueError("config requires task and data mappings")
    if task_config.get("name") == "multitask":
        datasets = data_config.get("datasets")
        if not isinstance(datasets, Mapping) or not isinstance(datasets.get(task_name), Mapping):
            raise ValueError(f"multitask config has no data.datasets.{task_name}")
        return datasets[task_name]
    if task_config.get("name") != task_name:
        raise ValueError(
            f"single-task config selects {task_config.get('name')!r}, not {task_name!r}"
        )
    return data_config


def build_task_dataset(
    config: dict[str, Any],
    task_name: str,
    *,
    split: str | None = None,
    training: bool | None = None,
    strict_protocol: bool = True,
) -> Dataset[dict[str, Any]]:
    """Build one raw-data adapter using its configured task codec."""

    # Delayed import avoids a package-initialization cycle with TaskSpec metadata.
    from perception_diffusion.task_specs import build_task_specs

    task_data = _task_data_config(config, task_name)
    spec = build_task_specs(config)[task_name]
    selected_split = str(split or task_data.get("split", "train"))
    if training is None:
        training = selected_split.casefold() in {"train", "training"}
    image_size_raw = task_data.get("image_size")
    if (
        not isinstance(image_size_raw, list)
        or len(image_size_raw) != 2
        or not all(isinstance(value, int) and value > 0 for value in image_size_raw)
    ):
        raise ValueError(f"{task_name} data.image_size must be [H,W]")
    image_size = (image_size_raw[0], image_size_raw[1])
    root = task_data.get("root")
    if not isinstance(root, str) or not root:
        raise ValueError(f"{task_name} data.root must be a non-empty path")
    flip_probability = (
        float(task_data.get("horizontal_flip_probability", 0.5)) if training else 0.0
    )

    if task_name == "segmentation":
        if spec.dataset_name != "ade20k":
            raise ValueError(f"unsupported segmentation dataset: {spec.dataset_name}")
        evaluation = (
            config["evaluation"]["segmentation"]
            if config["task"]["name"] == "multitask"
            else config["evaluation"]
        )
        query_config = evaluation["query"]
        query_sampling = str(
            task_data.get("query_sampling", "mixed" if training else "first_present")
        )
        if not training:
            # The placeholder encoded target must not choose a query by inspecting GT.
            # Formal evaluation consumes native_target and the closed-set query planner.
            query_sampling = str(task_data.get("evaluation_query_sampling", "fixed"))
        return ADE20KDataset(
            root,
            split=selected_split,
            image_size=image_size,
            codec=spec.codec,  # type: ignore[arg-type]
            horizontal_flip_probability=flip_probability,
            query_sampling=query_sampling,  # type: ignore[arg-type]
            positive_query_probability=float(task_data.get("positive_query_probability", 0.5)),
            fixed_query_class_id=task_data.get(
                "fixed_query_class_id", 0 if not training else None
            ),
            prompt_template=str(query_config["prompt_template"]),
            strict_protocol=strict_protocol,
        )
    if task_name in {"depth", "normal"}:
        if spec.dataset_name != "nyuv2":
            raise ValueError(f"unsupported {task_name} dataset: {spec.dataset_name}")
        return NYUv2Dataset(
            root,
            split=selected_split,
            task=task_name,  # type: ignore[arg-type]
            image_size=image_size,
            codec=spec.codec,  # type: ignore[arg-type]
            source=str(task_data.get("source", "auto")),  # type: ignore[arg-type]
            processed_dir=str(task_data.get("processed_dir", "processed")),
            depth_field=str(task_data.get("depth_field", "depths")),
            horizontal_flip_probability=flip_probability,
            max_relative_depth_jump=float(
                task_data.get("normal_max_relative_depth_jump", 0.05)
            ),
            strict_protocol=strict_protocol,
        )
    raise ValueError(f"unsupported task: {task_name}")


def seed_data_worker(worker_id: int) -> None:
    """Seed NumPy/Python from the per-worker PyTorch seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    config: dict[str, Any],
    task_name: str,
    *,
    split: str | None = None,
    training: bool | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    shuffle: bool | None = None,
    strict_protocol: bool = True,
    max_samples: int | None = None,
) -> DataLoader[dict[str, Any]]:
    """Build a reproducibly seeded, task-homogeneous DataLoader."""

    task_data = _task_data_config(config, task_name)
    selected_split = str(split or task_data.get("split", "train"))
    if training is None:
        training = selected_split.casefold() in {"train", "training"}
    dataset = build_task_dataset(
        config,
        task_name,
        split=selected_split,
        training=training,
        strict_protocol=strict_protocol,
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    resolved_batch_size = int(batch_size or config["training"].get("batch_size", 1))
    resolved_workers = int(
        task_data.get("num_workers", config["data"].get("num_workers", 0))
        if num_workers is None
        else num_workers
    )
    if resolved_batch_size <= 0 or resolved_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if shuffle is None:
        shuffle = bool(training)
    seed = int(config["experiment"].get("seed", 0))
    generator = torch.Generator()
    generator.manual_seed(seed)
    runtime_device = str(config.get("runtime", {}).get("device", "auto"))
    pin_memory = runtime_device == "cuda" or (
        runtime_device == "auto" and torch.cuda.is_available()
    )
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        num_workers=resolved_workers,
        collate_fn=collate_task_samples,
        worker_init_fn=seed_data_worker,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=resolved_workers > 0,
        drop_last=bool(task_data.get("drop_last", False) and training),
    )
