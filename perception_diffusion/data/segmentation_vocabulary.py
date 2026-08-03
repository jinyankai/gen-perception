"""Dataset-owned semantic class vocabularies for leakage-free evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClassVocabulary:
    """An ordered dataset taxonomy using model-facing zero-based class IDs."""

    dataset: str
    class_names: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("vocabulary dataset name is required")
        if not self.class_names:
            raise ValueError("vocabulary must contain at least one class")
        if any(not name.strip() for name in self.class_names):
            raise ValueError("vocabulary class names must be non-empty")
        normalized = tuple(name.casefold() for name in self.class_names)
        if len(set(normalized)) != len(normalized):
            raise ValueError("vocabulary class names must be unique")

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self.class_names)))


def _split_ade20k_row(line: str) -> list[str]:
    tab_parts = [part.strip() for part in line.rstrip("\r\n").split("\t")]
    if len(tab_parts) >= 6:
        return tab_parts
    return re.split(r"\s+", line.strip(), maxsplit=5)


def load_ade20k_object_info(
    path: str | Path, *, expected_num_classes: int = 150
) -> ClassVocabulary:
    """Load ADE20K's official objectInfo150.txt without consulting image labels."""

    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    lines = [line for line in metadata_path.read_text(encoding="utf-8-sig").splitlines() if line]
    if len(lines) < 2:
        raise ValueError(f"ADE20K metadata is empty: {metadata_path}")

    header = [field.casefold() for field in _split_ade20k_row(lines[0])]
    try:
        index_column = header.index("idx")
        name_column = header.index("name")
    except ValueError as exc:
        raise ValueError("ADE20K metadata requires Idx and Name columns") from exc

    indexed_names: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = _split_ade20k_row(line)
        if len(fields) <= max(index_column, name_column):
            raise ValueError(f"invalid ADE20K metadata row {line_number}")
        try:
            one_based_id = int(fields[index_column])
        except ValueError as exc:
            raise ValueError(f"invalid ADE20K class ID at row {line_number}") from exc
        canonical_name = fields[name_column].split(",", maxsplit=1)[0].strip()
        if not canonical_name:
            raise ValueError(f"empty ADE20K class name at row {line_number}")
        indexed_names.append((one_based_id, canonical_name))

    indexed_names.sort(key=lambda item: item[0])
    expected_ids = list(range(1, expected_num_classes + 1))
    actual_ids = [class_id for class_id, _ in indexed_names]
    if actual_ids != expected_ids:
        raise ValueError(
            "ADE20K class IDs must be contiguous and match "
            f"1..{expected_num_classes}; got {len(actual_ids)} rows"
        )
    return ClassVocabulary(
        dataset="ade20k",
        class_names=tuple(name for _, name in indexed_names),
        source=str(metadata_path),
    )


def load_segmentation_vocabulary(config: dict[str, Any]) -> ClassVocabulary:
    """Build a vocabulary only from declared metadata, never from ground truth masks."""

    source = config.get("source")
    expected = config.get("expected_num_classes")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        raise ValueError("vocabulary.expected_num_classes must be a positive integer")
    if source == "ade20k_object_info":
        path = config.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("ADE20K vocabulary requires a metadata path")
        return load_ade20k_object_info(path, expected_num_classes=expected)
    if source == "inline":
        class_names = config.get("class_names")
        if not isinstance(class_names, list) or not all(
            isinstance(name, str) for name in class_names
        ):
            raise ValueError("inline vocabulary requires class_names as a string list")
        if len(class_names) != expected:
            raise ValueError(
                f"inline vocabulary expected {expected} classes, got {len(class_names)}"
            )
        return ClassVocabulary(
            dataset=str(config.get("dataset", "custom")),
            class_names=tuple(class_names),
            source="inline",
        )
    raise ValueError(f"unsupported segmentation vocabulary source: {source}")
