"""occlugen: single-scene image compositing with occlusion annotation."""
from .types import (
    AffineTransform,
    BBox,
    InstanceAnnotation,
    InstanceSpec,
    SceneSpec,
)
from .sampling import sample_premultiplied
from .compose import render_scene
from .io import load_asset, render_to_dir

__all__ = [
    "AffineTransform",
    "BBox",
    "InstanceAnnotation",
    "InstanceSpec",
    "SceneSpec",
    "sample_premultiplied",
    "render_scene",
    "load_asset",
    "render_to_dir",
]
