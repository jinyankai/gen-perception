"""ADEChallengeData2016 semantic-segmentation dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from perception_diffusion.codecs import (
    SegmentationBinaryMaskCodec,
    SegmentationIdCodec,
    SegmentationPaletteCodec,
)

from .schema import validate_unified_sample
from .segmentation_vocabulary import load_ade20k_object_info
from .transforms import resize_labels, rgb_to_vae_tensor, should_horizontal_flip


ADE20K_SPLITS = {
    "train": "training",
    "training": "training",
    "val": "validation",
    "validation": "validation",
}
ADE20K_EXPECTED_COUNTS = {"training": 20_210, "validation": 2_000}
SegmentationCodec = (
    SegmentationBinaryMaskCodec | SegmentationIdCodec | SegmentationPaletteCodec
)
QuerySampling = Literal["mixed", "uniform", "first_present", "fixed"]


class ADE20KDataset(Dataset[dict[str, Any]]):
    """Load ADE20K RGB/mask pairs and emit the unified dense-task schema.

    Official PNG IDs are 1..150 with 0 denoting unlabeled pixels. This adapter
    maps them to model-facing IDs 0..149 with 255 as ignore. When the binary
    query codec is selected, each item is converted into a class-conditioned
    binary mask while retaining the full ID map in ``native_target``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        image_size: tuple[int, int],
        codec: SegmentationCodec,
        horizontal_flip_probability: float = 0.0,
        query_sampling: QuerySampling = "first_present",
        positive_query_probability: float = 0.5,
        fixed_query_class_id: int | None = None,
        prompt_template: str = "segmentation mask of {class_name}",
        strict_protocol: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        try:
            self.split = ADE20K_SPLITS[split.casefold()]
        except KeyError as exc:
            raise ValueError(f"unsupported ADE20K split: {split}") from exc
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must contain positive H,W")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.codec = codec
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must lie in [0,1]")
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        if query_sampling not in {"mixed", "uniform", "first_present", "fixed"}:
            raise ValueError(f"unsupported query sampling mode: {query_sampling}")
        if not 0.0 <= positive_query_probability <= 1.0:
            raise ValueError("positive_query_probability must lie in [0,1]")
        self.query_sampling = query_sampling
        self.positive_query_probability = float(positive_query_probability)
        self.fixed_query_class_id = fixed_query_class_id
        self.prompt_template = prompt_template
        self.num_classes = int(getattr(codec, "num_classes", 150))
        if isinstance(codec, SegmentationBinaryMaskCodec):
            self.num_classes = 150
        if fixed_query_class_id is not None and not 0 <= fixed_query_class_id < self.num_classes:
            raise ValueError(f"fixed query class must lie in [0,{self.num_classes - 1}]")
        if query_sampling == "fixed" and fixed_query_class_id is None:
            raise ValueError("fixed query sampling requires fixed_query_class_id")

        image_root = self.root / "images" / self.split
        annotation_root = self.root / "annotations" / self.split
        if not image_root.is_dir() or not annotation_root.is_dir():
            raise FileNotFoundError(
                f"expected images/{self.split} and annotations/{self.split} under {self.root}"
            )
        image_paths = sorted(
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
        )
        self.records: list[tuple[Path, Path, str]] = []
        for image_path in image_paths:
            relative = image_path.relative_to(image_root)
            mask_path = annotation_root / relative.with_suffix(".png")
            if not mask_path.is_file():
                raise FileNotFoundError(f"ADE20K mask missing for {image_path}: {mask_path}")
            self.records.append((image_path, mask_path, relative.with_suffix("").as_posix()))
        if not self.records:
            raise FileNotFoundError(f"no ADE20K JPEG images found under {image_root}")
        if strict_protocol and len(self.records) != ADE20K_EXPECTED_COUNTS[self.split]:
            raise ValueError(
                f"ADE20K {self.split} should contain {ADE20K_EXPECTED_COUNTS[self.split]} "
                f"samples, found {len(self.records)}"
            )
        vocabulary = load_ade20k_object_info(
            self.root / "objectInfo150.txt", expected_num_classes=150
        )
        self.class_names = vocabulary.class_names

    def __len__(self) -> int:
        return len(self.records)

    def _select_query_class(self, labels: torch.Tensor) -> int:
        present = torch.unique(labels[labels != 255])
        if self.query_sampling == "fixed":
            assert self.fixed_query_class_id is not None
            return self.fixed_query_class_id
        if self.query_sampling == "first_present":
            if present.numel() == 0:
                raise ValueError("ADE20K sample contains no labeled class")
            return int(present.min())
        if self.query_sampling == "uniform":
            return int(torch.randint(self.num_classes, ()).item())
        choose_positive = present.numel() > 0 and bool(
            torch.rand(()) < self.positive_query_probability
        )
        if choose_positive:
            return int(present[torch.randint(present.numel(), ())].item())
        return int(torch.randint(self.num_classes, ()).item())

    def _read_source_labels(self, mask_path: Path) -> np.ndarray:
        with Image.open(mask_path) as handle:
            source_labels = np.asarray(handle)
        if source_labels.ndim != 2:
            raise ValueError(f"ADE20K mask must be HxW, got {source_labels.shape}: {mask_path}")
        if not np.issubdtype(source_labels.dtype, np.integer):
            raise TypeError(f"ADE20K mask must contain integer IDs: {mask_path}")
        if int(source_labels.min()) < 0 or int(source_labels.max()) > 150:
            raise ValueError(
                f"ADE20K source labels must lie in 0..150, got "
                f"{int(source_labels.min())}..{int(source_labels.max())}: {mask_path}"
            )
        return source_labels

    @staticmethod
    def _map_source_labels(source_labels: np.ndarray) -> np.ndarray:
        """Map official 1..150 (0=unlabeled) to model IDs 0..149 (255=ignore)."""

        labels = np.full(source_labels.shape, 255, dtype=np.int64)
        source_valid = source_labels > 0
        labels[source_valid] = source_labels[source_valid].astype(np.int64) - 1
        return labels

    def load_native_labels(self, index: int) -> np.ndarray:
        """Return native-resolution HxW int64 labels (0..149, 255=ignore).

        Unlike ``native_target`` in ``__getitem__``, these are never resized to
        the model input size, so evaluation can score at the dataset's native
        resolution rather than the 512x512 latent-decode resolution.
        """

        _, mask_path, _ = self.records[index]
        return self._map_source_labels(self._read_source_labels(mask_path))

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path, relative_id = self.records[index]
        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        source_labels = self._read_source_labels(mask_path)
        if image.shape[:2] != source_labels.shape:
            raise ValueError(
                f"ADE20K image/mask shape mismatch for {relative_id}: "
                f"{image.shape[:2]} vs {source_labels.shape}"
            )
        original_size = image.shape[:2]
        labels = self._map_source_labels(source_labels)

        image_tensor = rgb_to_vae_tensor(image, self.image_size)
        label_tensor = resize_labels(labels, self.image_size)
        if should_horizontal_flip(self.horizontal_flip_probability):
            image_tensor = torch.flip(image_tensor, dims=(2,))
            label_tensor = torch.flip(label_tensor, dims=(1,))

        valid = (label_tensor != 255).cpu().numpy()
        query_class_id = -1
        text_condition = "semantic segmentation"
        if isinstance(self.codec, SegmentationBinaryMaskCodec):
            query_class_id = self._select_query_class(label_tensor)
            binary = (label_tensor == query_class_id).to(torch.uint8).cpu().numpy()
            encoded = self.codec.encode(binary, valid)
            text_condition = self.prompt_template.format(
                class_name=self.class_names[query_class_id]
            )
        else:
            encoded = self.codec.encode(label_tensor.cpu().numpy(), valid)

        sample: dict[str, Any] = {
            "image": image_tensor,
            "target": torch.from_numpy(np.ascontiguousarray(encoded.values)).float(),
            "valid_mask": torch.from_numpy(encoded.valid_mask.copy())[None],
            "native_target": label_tensor[None].contiguous(),
            "task_name": "segmentation",
            "sample_id": f"ade20k/{self.split}/{relative_id}",
            "source_index": int(index),
            "original_size": torch.tensor(original_size, dtype=torch.int64),
            "query_class_id": int(query_class_id),
            "text_condition": text_condition,
        }
        validate_unified_sample(sample)
        return sample
