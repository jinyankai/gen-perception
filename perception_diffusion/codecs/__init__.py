"""Task target codecs."""

from .base import EncodedTarget
from .depth import DepthCodec
from .normal import NormalCodec
from .segmentation import SegmentationIdCodec, SegmentationPaletteCodec

__all__ = [
    "DepthCodec",
    "EncodedTarget",
    "NormalCodec",
    "SegmentationIdCodec",
    "SegmentationPaletteCodec",
]
