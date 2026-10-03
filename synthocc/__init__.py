"""synthocc: single-scene compositing with per-instance occlusion annotation."""
from .api import render_scene
from .types import (
    Affine2D,
    BBox,
    Instance,
    InstanceAnnotation,
    RenderResult,
    Scene,
)

__all__ = [
    "render_scene",
    "Affine2D",
    "BBox",
    "Instance",
    "InstanceAnnotation",
    "RenderResult",
    "Scene",
]
