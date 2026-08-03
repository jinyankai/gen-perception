"""NYUv2 depth and derived surface-normal dataset adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from perception_diffusion.codecs import DepthCodec, NormalCodec

from .nyuv2_geometry import NYUV2_RGB_INTRINSICS, depth_to_normals
from .schema import validate_unified_sample
from .transforms import (
    resize_depth_with_mask,
    resize_normals_with_mask,
    rgb_to_vae_tensor,
    should_horizontal_flip,
)


NYUV2_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "test": "test",
    "val": "test",
    "validation": "test",
}
NYUV2_EXPECTED_COUNTS = {"train": 795, "test": 654}
NYUV2_SAMPLE_COUNT = 1_449
NYUV2_NATIVE_SIZE = (480, 640)
NYUv2Task = Literal["depth", "normal"]


def canonical_nyuv2_split(split: str) -> str:
    try:
        return NYUV2_SPLIT_ALIASES[split.casefold()]
    except KeyError as exc:
        raise ValueError(f"unsupported NYUv2 split: {split}") from exc


def load_nyuv2_splits(
    path: str | Path,
    *,
    sample_count: int = NYUV2_SAMPLE_COUNT,
    strict_protocol: bool = True,
) -> dict[str, np.ndarray]:
    """Load official 1-based MATLAB indices and return zero-based arrays."""

    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"NYUv2 official split file is missing: {split_path}. "
            "Run scripts/data/download_nyuv2.py."
        )
    raw = loadmat(split_path)
    missing = sorted({"trainNdxs", "testNdxs"} - set(raw))
    if missing:
        raise ValueError(f"NYUv2 splits.mat is missing keys: {missing}")
    result: dict[str, np.ndarray] = {}
    for split, key in (("train", "trainNdxs"), ("test", "testNdxs")):
        one_based = np.asarray(raw[key]).reshape(-1)
        if not np.issubdtype(one_based.dtype, np.integer):
            if not np.all(np.equal(one_based, np.floor(one_based))):
                raise TypeError(f"{key} must contain integer indices")
        indices = one_based.astype(np.int64) - 1
        if indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= sample_count:
            raise ValueError(f"{key} contains indices outside 1..{sample_count}")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{key} contains duplicate indices")
        result[split] = indices
    overlap = np.intersect1d(result["train"], result["test"])
    if overlap.size:
        raise ValueError(f"NYUv2 train/test split overlap: {overlap[:10].tolist()}")
    if strict_protocol:
        for split, expected in NYUV2_EXPECTED_COUNTS.items():
            if result[split].size != expected:
                raise ValueError(
                    f"NYUv2 {split} split should contain {expected} samples, "
                    f"found {result[split].size}"
                )
        covered = np.sort(np.concatenate((result["train"], result["test"])))
        if not np.array_equal(covered, np.arange(sample_count)):
            raise ValueError("NYUv2 official splits must cover all 1,449 labeled frames")
    return result


class NYUv2RawReader:
    """Worker-safe random-access reader for the official HDF5/MAT asset."""

    def __init__(
        self,
        root: str | Path,
        *,
        depth_field: str = "depths",
        native_size: tuple[int, int] = NYUV2_NATIVE_SIZE,
        strict_protocol: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self._handle: h5py.File | None = None
        self._owner_pid: int | None = None
        self.mat_path = self.root / "nyu_depth_v2_labeled.mat"
        self.split_path = self.root / "splits.mat"
        self.depth_field = depth_field
        self.native_size = (int(native_size[0]), int(native_size[1]))
        if not self.mat_path.is_file():
            raise FileNotFoundError(
                f"NYUv2 labeled MAT file is missing: {self.mat_path}. "
                "Run scripts/data/download_nyuv2.py."
            )
        with h5py.File(self.mat_path, "r") as handle:
            missing = sorted({"images", depth_field} - set(handle.keys()))
            if missing:
                raise ValueError(f"NYUv2 MAT file is missing datasets: {missing}")
            image_shape = tuple(int(value) for value in handle["images"].shape)
            depth_shape = tuple(int(value) for value in handle[depth_field].shape)
        if len(image_shape) != 4 or len(depth_shape) != 3:
            raise ValueError(
                f"unexpected NYUv2 image/depth ranks: {image_shape}/{depth_shape}"
            )
        self.sample_count = self._infer_sample_count(
            image_shape, depth_shape, self.native_size
        )
        if strict_protocol and self.sample_count != NYUV2_SAMPLE_COUNT:
            raise ValueError(
                f"NYUv2 labeled MAT should contain {NYUV2_SAMPLE_COUNT} samples, "
                f"found {self.sample_count}"
            )
        self.image_sample_axis = self._sample_axis(image_shape, self.sample_count, "images")
        self.depth_sample_axis = self._sample_axis(depth_shape, self.sample_count, depth_field)
        self.splits = load_nyuv2_splits(
            self.split_path,
            sample_count=self.sample_count,
            strict_protocol=strict_protocol,
        )

    @staticmethod
    def _infer_sample_count(
        image_shape: tuple[int, ...],
        depth_shape: tuple[int, ...],
        native_size: tuple[int, int],
    ) -> int:
        shared = set(image_shape) & set(depth_shape)
        candidates = sorted(value for value in shared if value not in {3, *native_size})
        if NYUV2_SAMPLE_COUNT in shared:
            return NYUV2_SAMPLE_COUNT
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"cannot infer NYUv2 sample axis from shapes {image_shape}/{depth_shape}"
        )

    @staticmethod
    def _sample_axis(shape: tuple[int, ...], sample_count: int, name: str) -> int:
        axes = [axis for axis, size in enumerate(shape) if size == sample_count]
        if len(axes) != 1:
            raise ValueError(f"cannot identify sample axis for {name}: {shape}")
        return axes[0]

    def _file(self) -> h5py.File:
        pid = os.getpid()
        if self._handle is None or self._owner_pid != pid:
            self.close()
            self._handle = h5py.File(self.mat_path, "r")
            self._owner_pid = pid
        return self._handle

    @staticmethod
    def _take(dataset: h5py.Dataset, axis: int, index: int) -> np.ndarray:
        selection: list[int | slice] = [slice(None)] * dataset.ndim
        selection[axis] = int(index)
        return np.asarray(dataset[tuple(selection)])

    def _orient_depth(self, array: np.ndarray) -> np.ndarray:
        squeezed = np.squeeze(array)
        if squeezed.shape == self.native_size:
            result = squeezed
        elif squeezed.shape == self.native_size[::-1]:
            result = squeezed.T
        else:
            raise ValueError(
                f"NYUv2 depth sample has unexpected shape {squeezed.shape}; "
                f"expected {self.native_size} or {self.native_size[::-1]}"
            )
        return np.asarray(result, dtype=np.float32)

    def _orient_image(self, array: np.ndarray) -> np.ndarray:
        channel_axes = [axis for axis, size in enumerate(array.shape) if size == 3]
        if len(channel_axes) != 1:
            raise ValueError(f"NYUv2 image sample must have one RGB axis: {array.shape}")
        hwc = np.moveaxis(array, channel_axes[0], -1)
        if hwc.shape[:2] == self.native_size:
            result = hwc
        elif hwc.shape[:2] == self.native_size[::-1]:
            result = np.transpose(hwc, (1, 0, 2))
        else:
            raise ValueError(
                f"NYUv2 image sample has unexpected shape {hwc.shape}; "
                f"expected {(*self.native_size, 3)}"
            )
        if result.dtype != np.uint8:
            if not np.issubdtype(result.dtype, np.integer):
                raise TypeError(f"NYUv2 RGB values must be integers, got {result.dtype}")
            if int(result.min()) < 0 or int(result.max()) > 255:
                raise ValueError("NYUv2 RGB values lie outside uint8 range")
            result = result.astype(np.uint8)
        return np.ascontiguousarray(result)

    def read(self, source_index: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= source_index < self.sample_count:
            raise IndexError(source_index)
        handle = self._file()
        image = self._take(handle["images"], self.image_sample_axis, source_index)
        depth = self._take(handle[self.depth_field], self.depth_sample_axis, source_index)
        return self._orient_image(image), self._orient_depth(depth)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._owner_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()


class NYUv2Dataset(Dataset[dict[str, Any]]):
    """Emit depth or derived-normal targets from raw or preprocessed NYUv2."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        task: NYUv2Task,
        image_size: tuple[int, int],
        codec: DepthCodec | NormalCodec,
        source: Literal["auto", "raw", "processed"] = "auto",
        processed_dir: str | Path = "processed",
        depth_field: str = "depths",
        horizontal_flip_probability: float = 0.0,
        max_relative_depth_jump: float = 0.05,
        strict_protocol: bool = True,
        native_size: tuple[int, int] = NYUV2_NATIVE_SIZE,
    ) -> None:
        self.root = Path(root).expanduser()
        self.raw_reader: NYUv2RawReader | None = None
        self.split = canonical_nyuv2_split(split)
        if task not in {"depth", "normal"}:
            raise ValueError(f"NYUv2 task must be depth or normal, got {task}")
        if task == "depth" and not isinstance(codec, DepthCodec):
            raise TypeError("NYUv2 depth task requires DepthCodec")
        if task == "normal" and not isinstance(codec, NormalCodec):
            raise TypeError("NYUv2 normal task requires NormalCodec")
        self.task = task
        self.codec = codec
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if min(self.image_size) <= 0:
            raise ValueError("image_size must contain positive H,W")
        if source not in {"auto", "raw", "processed"}:
            raise ValueError(f"unsupported NYUv2 source: {source}")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must lie in [0,1]")
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.max_relative_depth_jump = float(max_relative_depth_jump)
        self.depth_field = depth_field
        self.processed_root = self.root / processed_dir
        self.source = self._resolve_source(source, strict_protocol, native_size)

    def _resolve_source(
        self,
        requested: str,
        strict_protocol: bool,
        native_size: tuple[int, int],
    ) -> str:
        manifest_path = self.processed_root / "manifest.json"
        processed_error: Exception | None = None
        if requested in {"auto", "processed"} and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("format") != "gen-perception-nyuv2-v1":
                    raise ValueError(f"unsupported processed NYUv2 manifest: {manifest_path}")
                indices = manifest.get("splits", {}).get(self.split)
                if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
                    raise ValueError(f"manifest has no valid {self.split} split")
                if strict_protocol and len(indices) != NYUV2_EXPECTED_COUNTS[self.split]:
                    raise ValueError(
                        f"processed NYUv2 {self.split} should contain "
                        f"{NYUV2_EXPECTED_COUNTS[self.split]} samples"
                    )
                records = [
                    self.processed_root / self.split / f"{source_index + 1:04d}.npz"
                    for source_index in indices
                ]
                missing = next((path for path in records if not path.is_file()), None)
                if missing is not None:
                    raise FileNotFoundError(f"processed NYUv2 sample is missing: {missing}")
                if self.task == "normal" and records:
                    with np.load(records[0], allow_pickle=False) as first:
                        if not {"normal", "normal_valid"}.issubset(first.files):
                            raise ValueError("processed NYUv2 samples do not contain normals")
                self.source_indices = np.asarray(indices, dtype=np.int64)
                self.processed_records = records
                return "processed"
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                processed_error = exc
                if requested == "processed":
                    raise
        elif requested == "processed":
            raise FileNotFoundError(manifest_path)

        try:
            self.raw_reader = NYUv2RawReader(
                self.root,
                depth_field=self.depth_field,
                native_size=native_size,
                strict_protocol=strict_protocol,
            )
        except Exception as raw_error:
            if processed_error is not None:
                raise RuntimeError(
                    f"processed NYUv2 is unusable ({processed_error}); raw fallback also "
                    f"failed ({raw_error})"
                ) from raw_error
            raise
        self.source_indices = self.raw_reader.splits[self.split]
        self.processed_records: list[Path] = []
        return "raw"

    def __len__(self) -> int:
        return int(self.source_indices.size)

    def _read(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self.source == "raw":
            assert self.raw_reader is not None
            image, depth = self.raw_reader.read(int(self.source_indices[index]))
            return image, depth, None, None
        with np.load(self.processed_records[index], allow_pickle=False) as sample:
            image = np.asarray(sample["image"], dtype=np.uint8)
            depth = np.asarray(sample["depth"], dtype=np.float32)
            normal = (
                np.asarray(sample["normal"], dtype=np.float32)
                if "normal" in sample.files
                else None
            )
            normal_valid = (
                np.asarray(sample["normal_valid"], dtype=bool)
                if "normal_valid" in sample.files
                else None
            )
        return image, depth, normal, normal_valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, depth, normals, normal_valid = self._read(index)
        if image.shape[:2] != depth.shape:
            raise ValueError(
                f"NYUv2 image/depth shape mismatch at source index "
                f"{int(self.source_indices[index])}: {image.shape}/{depth.shape}"
            )
        original_size = depth.shape
        if isinstance(self.codec, DepthCodec):
            depth_valid = (
                np.isfinite(depth)
                & (depth >= self.codec.min_depth)
                & (depth <= self.codec.max_depth)
            )
        else:
            depth_valid = np.isfinite(depth) & (depth > 0) & (depth <= 10.0)
        if self.task == "normal" and normals is None:
            if depth.shape != (NYUV2_RGB_INTRINSICS.height, NYUV2_RGB_INTRINSICS.width):
                raise ValueError(
                    "on-the-fly normal generation requires original NYUv2 480x640 depth"
                )
            normals, normal_valid = depth_to_normals(
                depth,
                depth_valid,
                intrinsics=NYUV2_RGB_INTRINSICS,
                max_relative_depth_jump=self.max_relative_depth_jump,
            )

        image_tensor = rgb_to_vae_tensor(image, self.image_size)
        depth_tensor, resized_depth_valid = resize_depth_with_mask(
            depth, depth_valid, self.image_size
        )
        if self.task == "normal":
            assert normals is not None and normal_valid is not None
            native_target, valid_tensor = resize_normals_with_mask(
                normals, normal_valid, self.image_size
            )
        else:
            native_target = depth_tensor[None]
            valid_tensor = resized_depth_valid

        if should_horizontal_flip(self.horizontal_flip_probability):
            image_tensor = torch.flip(image_tensor, dims=(2,))
            native_target = torch.flip(native_target, dims=(2,))
            valid_tensor = torch.flip(valid_tensor, dims=(1,))
            if self.task == "normal":
                # Mirroring image coordinates reflects the camera-space x axis.
                native_target[0] *= -1.0

        native_numpy = native_target.cpu().numpy()
        valid_numpy = valid_tensor.cpu().numpy()
        if self.task == "depth":
            encoded = self.codec.encode(native_numpy[0], valid_numpy)
            text_condition = "monocular depth estimation"
        else:
            encoded = self.codec.encode(native_numpy, valid_numpy)
            text_condition = "surface normal estimation"
        source_index = int(self.source_indices[index])
        sample: dict[str, Any] = {
            "image": image_tensor,
            "target": torch.from_numpy(np.ascontiguousarray(encoded.values)).float(),
            "valid_mask": torch.from_numpy(encoded.valid_mask.copy())[None],
            "native_target": native_target.to(torch.float32).contiguous(),
            "task_name": self.task,
            "sample_id": f"nyuv2/{self.split}/{source_index + 1:04d}",
            "source_index": source_index,
            "original_size": torch.tensor(original_size, dtype=torch.int64),
            "query_class_id": -1,
            "text_condition": text_condition,
        }
        validate_unified_sample(sample)
        return sample

    def close(self) -> None:
        if self.raw_reader is not None:
            self.raw_reader.close()

    def __del__(self) -> None:
        self.close()
