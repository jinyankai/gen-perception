"""NYUv2 camera geometry and deterministic depth-to-normal conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for image coordinates x-right, y-down, z-forward."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0 or self.width <= 2 or self.height <= 2:
            raise ValueError("invalid pinhole camera intrinsics")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


# RGB camera calibration distributed in the official NYUv2 toolbox.
NYUV2_RGB_INTRINSICS = CameraIntrinsics(
    fx=518.85790117450188,
    fy=519.46961112127485,
    cx=325.58244941119034,
    cy=253.73616633400465,
    width=640,
    height=480,
)


def depth_to_camera_points(
    depth: NDArray[np.generic], intrinsics: CameraIntrinsics = NYUV2_RGB_INTRINSICS
) -> NDArray[np.float32]:
    """Back-project a metric depth map to HxWx3 camera-space points."""

    z = np.asarray(depth, dtype=np.float32)
    if z.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"depth shape {z.shape} does not match intrinsics "
            f"{(intrinsics.height, intrinsics.width)}"
        )
    u, v = np.meshgrid(
        np.arange(intrinsics.width, dtype=np.float32),
        np.arange(intrinsics.height, dtype=np.float32),
    )
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def depth_to_normals(
    depth: NDArray[np.generic],
    valid_mask: NDArray[np.generic] | None = None,
    *,
    intrinsics: CameraIntrinsics = NYUV2_RGB_INTRINSICS,
    max_relative_depth_jump: float = 0.05,
    epsilon: float = 1.0e-6,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Derive camera-space unit normals using centered 3D tangents.

    Normals use x-right, y-down, z-forward camera coordinates and are oriented
    toward the camera. Border pixels, invalid neighborhoods, degenerate
    tangents, and neighborhoods crossing a large relative depth discontinuity
    are excluded.
    """

    if max_relative_depth_jump <= 0:
        raise ValueError("max_relative_depth_jump must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    z = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(z) & (z > 0)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != z.shape:
            raise ValueError(f"valid mask must have shape {z.shape}, got {supplied.shape}")
        valid &= supplied
    points = depth_to_camera_points(z, intrinsics)

    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    # dy x dx points toward the camera for a fronto-parallel surface.
    inner_normals = np.cross(dy, dx)
    inner_norms = np.linalg.norm(inner_normals, axis=2)

    center = np.maximum(z[1:-1, 1:-1], epsilon)
    neighbor_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )
    max_jump = np.maximum.reduce(
        (
            np.abs(z[1:-1, :-2] - center),
            np.abs(z[1:-1, 2:] - center),
            np.abs(z[:-2, 1:-1] - center),
            np.abs(z[2:, 1:-1] - center),
        )
    )
    inner_valid = (
        neighbor_valid
        & np.isfinite(inner_normals).all(axis=2)
        & (inner_norms > epsilon)
        & ((max_jump / center) <= max_relative_depth_jump)
    )
    unit = np.zeros_like(inner_normals, dtype=np.float32)
    unit[inner_valid] = inner_normals[inner_valid] / inner_norms[inner_valid, None]

    # Resolve the cross-product sign so every valid normal faces the camera.
    inner_points = points[1:-1, 1:-1]
    points_away = np.sum(unit * inner_points, axis=2) > 0
    unit[points_away] *= -1.0

    normals = np.zeros((*z.shape, 3), dtype=np.float32)
    normal_valid = np.zeros(z.shape, dtype=bool)
    normals[1:-1, 1:-1] = unit
    normal_valid[1:-1, 1:-1] = inner_valid
    normals[~normal_valid] = 0.0
    return np.moveaxis(normals, -1, 0), normal_valid

