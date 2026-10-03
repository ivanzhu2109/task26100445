"""Asset loading and output persistence."""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
from PIL import Image

from .compose import render_scene
from .types import InstanceAnnotation, SceneSpec


def load_asset(path: str) -> np.ndarray:
    """Load a local RGBA PNG as float array (H, W, 4), RGB 0..255, alpha 0..1."""
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float64)
    rgba[..., 3] /= 255.0
    return rgba


def _annotation_to_dict(ann: InstanceAnnotation) -> dict:
    if ann.bbox is None:
        bbox = None
    else:
        bbox = {
            "x": ann.bbox.x,
            "y": ann.bbox.y,
            "width": ann.bbox.width,
            "height": ann.bbox.height,
        }
    return {
        "instance_id": ann.instance_id,
        "category": ann.category,
        "bbox": bbox,
        "occlusion_rate": ann.occlusion_rate,
    }


def render_to_dir(scene: SceneSpec, out_dir: str) -> List[InstanceAnnotation]:
    """Render the scene, save RGB PNG and per-instance JSON into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    image, annotations = render_scene(scene, load_asset)
    Image.fromarray(image).save(os.path.join(out_dir, "image.png"))
    for ann in annotations:
        path = os.path.join(out_dir, f"{ann.instance_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_annotation_to_dict(ann), fh, indent=2)
    return annotations
