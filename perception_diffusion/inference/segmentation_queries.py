"""Leakage-free semantic-segmentation query planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from perception_diffusion.data import ClassVocabulary


IGNORE_LABEL = 255


@dataclass(frozen=True)
class SegmentationQuery:
    class_id: int
    class_name: str
    prompt: str


class SegmentationQueryPlanner:
    """Create a fixed closed-set query list from dataset metadata.

    Ground-truth masks are intentionally absent from this API.
    """

    def __init__(
        self,
        vocabulary: ClassVocabulary,
        *,
        prompt_template: str = "segmentation mask of {class_name}",
    ) -> None:
        if "{class_name}" not in prompt_template:
            raise ValueError("prompt_template must contain {class_name}")
        self.vocabulary = vocabulary
        self.prompt_template = prompt_template

    def all_queries(self) -> tuple[SegmentationQuery, ...]:
        return tuple(
            SegmentationQuery(
                class_id=class_id,
                class_name=class_name,
                prompt=self.prompt_template.format(class_name=class_name),
            )
            for class_id, class_name in zip(
                self.vocabulary.class_ids,
                self.vocabulary.class_names,
                strict=True,
            )
        )

    def chunks(self, batch_size: int) -> tuple[tuple[SegmentationQuery, ...], ...]:
        if batch_size <= 0:
            raise ValueError("query batch size must be positive")
        queries = self.all_queries()
        return tuple(
            queries[start : start + batch_size]
            for start in range(0, len(queries), batch_size)
        )


def merge_query_scores(
    class_scores: NDArray[np.generic],
    queries: tuple[SegmentationQuery, ...],
    *,
    valid_mask: NDArray[np.generic] | None = None,
    confidence_threshold: float | None = None,
) -> NDArray[np.int64]:
    """Merge per-class score maps into closed-set semantic IDs.

    ``valid_mask`` forces pixels outside the mask to ``IGNORE_LABEL`` so that
    unlabeled regions are not silently absorbed by the ``argmax`` fallback.
    ``confidence_threshold`` (default off) ignores pixels whose winning score
    falls below it, mapping genuine no-confidence pixels to ``IGNORE_LABEL``.
    """

    scores = np.asarray(class_scores, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError(f"class_scores must have shape [C,H,W], got {scores.shape}")
    if scores.shape[0] != len(queries):
        raise ValueError(
            f"received {scores.shape[0]} score maps for {len(queries)} queries"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("class scores must be finite")
    class_ids = np.asarray([query.class_id for query in queries], dtype=np.int64)
    winners = np.argmax(scores, axis=0)
    labels = class_ids[winners]
    if confidence_threshold is not None:
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must lie in (0,1)")
        labels[np.max(scores, axis=0) < confidence_threshold] = IGNORE_LABEL
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != labels.shape:
            raise ValueError(
                f"valid_mask shape {valid.shape} differs from scores {labels.shape}"
            )
        labels[~valid] = IGNORE_LABEL
    return labels
