"""Type definitions for the synthocc compositing library."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Affine2D:
    """Invertible 2D affine transform mapping source (asset) coords to canvas coords.

    Canvas point = M @ [sx, sy, 1]. Stored row-major as 6 floats.
    """

    m: Tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.m) != 6:
            raise ValueError("Affine2D requires exactly 6 coefficients")

    @property
    def a(self) -> float:
        return self.m[0]

    @property
    def b(self) -> float:
        return self.m[1]

    @property
    def c(self) -> float:
        return self.m[2]

    @property
    def d(self) -> float:
        return self.m[3]

    @property
    def e(self) -> float:
        return self.m[4]

    @property
    def f(self) -> float:
        return self.m[5]

    def inverse(self) -> "Affine2D":
        a, b, c, d, e, f = self.m
        det = a * e - b * d
        if det == 0.0:
            raise ValueError("Affine2D transform is not invertible")
        ia = e / det
        ib = -b / det
        id_ = -d / det
        ie = a / det
        ic = -(ia * c + ib * f)
        if_ = -(id_ * c + ie * f)
        return Affine2D((ia, ib, ic, id_, ie, if_))


@dataclass(frozen=True)
class Instance:
    """One placed instance, ordered back-to-front within a Scene."""

    instance_id: str
    category: str
    asset_path: Path
    transform: Affine2D


@dataclass(frozen=True)
class Scene:
    """Canvas size, opaque solid background, and back-to-front instances."""

    width: int
    height: int
    background: Tuple[int, int, int]
    instances: Tuple[Instance, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BBox:
    """Integer bounding box; right and bottom edges are exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class InstanceAnnotation:
    """Per-instance occlusion annotation."""

    instance_id: str
    category: str
    occlusion: float  # 0..100
    bbox: Optional[BBox]  # None when fully occluded


@dataclass(frozen=True)
class RenderResult:
    """Paths of the artifacts written for one rendered scene."""

    image_path: Path
    annotation_path: Path
    annotations: Tuple[InstanceAnnotation, ...]
