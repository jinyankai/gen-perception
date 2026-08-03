"""Dataset adapters, metadata, geometry, and unified loading helpers."""

from .ade20k import ADE20KDataset
from .loaders import build_dataloader, build_task_dataset
from .nyuv2 import NYUv2Dataset, NYUv2RawReader, load_nyuv2_splits
from .nyuv2_geometry import (
    NYUV2_RGB_INTRINSICS,
    CameraIntrinsics,
    depth_to_camera_points,
    depth_to_normals,
)
from .schema import collate_task_samples, validate_unified_sample

from .segmentation_vocabulary import (
    ClassVocabulary,
    load_ade20k_object_info,
    load_segmentation_vocabulary,
)

__all__ = [
    "ADE20KDataset",
    "CameraIntrinsics",
    "ClassVocabulary",
    "NYUV2_RGB_INTRINSICS",
    "NYUv2Dataset",
    "NYUv2RawReader",
    "build_dataloader",
    "build_task_dataset",
    "collate_task_samples",
    "depth_to_camera_points",
    "depth_to_normals",
    "load_ade20k_object_info",
    "load_nyuv2_splits",
    "load_segmentation_vocabulary",
    "validate_unified_sample",
]
