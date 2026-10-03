"""Core data types for scene description and annotation output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AffineTransform:
    """Invertible 2D affine transform mapping source (asset) coordinates
    to canvas coordinates:

        canvas_x = a * src_x + b * src_y + c
        canvas_y = d * src_x + e * src_y + f
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def inverse(self) -> "AffineTransform":
        det = self.a * self.e - self.b * self.d
        if det == 0.0:
            raise ValueError("transform is not invertible")
        a = self.e / det
        b = -self.b / det
        d = -self.d / det
        e = self.a / det
        c = -(a * self.c + b * self.f)
        f = -(d * self.c + e * self.f)
        return AffineTransform(a=a, b=b, c=c, d=d, e=e, f=f)


@dataclass(frozen=True)
class InstanceSpec:
    """One placed instance. Instances are ordered back-to-front."""

    instance_id: str
    category: str
    asset_path: str
    transform: AffineTransform


@dataclass(frozen=True)
class SceneSpec:
    """Canvas size, opaque solid background, back-to-front instances."""

    width: int
    height: int
    background: Tuple[int, int, int]
    instances: Tuple[InstanceSpec, ...]


@dataclass(frozen=True)
class BBox:
    """Integer rectangle; right and bottom edges are exclusive."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class InstanceAnnotation:
    """Per-instance annotation. bbox is None when fully occluded."""

    instance_id: str
    category: str
    bbox: Optional[BBox]
    occlusion_rate: float
