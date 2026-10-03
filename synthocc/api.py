"""Top-level scene rendering entry point used by upstream generation tasks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .annotation import compute_annotations
from .compositor import composite
from .sampler import sample_premultiplied
from .types import RenderResult, Scene


def _load_asset(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGBA"), dtype=np.uint8)


def render_scene(scene: Scene, output_dir: Path, stem: str = "scene") -> RenderResult:
    """Render a scene: write RGB PNG and per-instance JSON annotations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = []
    alphas = []
    for instance in scene.instances:
        asset = _load_asset(Path(instance.asset_path))
        rgb, alpha = sample_premultiplied(
            asset, instance.transform.inverse(), scene.width, scene.height
        )
        layers.append((rgb, alpha))
        alphas.append(alpha)

    image = composite(scene.background, layers, scene.width, scene.height)
    image_u8 = np.rint(image).clip(0, 255).astype(np.uint8)

    annotations = compute_annotations(
        alphas,
        [inst.instance_id for inst in scene.instances],
        [inst.category for inst in scene.instances],
    )

    image_path = output_dir / f"{stem}.png"
    Image.fromarray(image_u8).save(image_path)

    annotation_path = output_dir / f"{stem}.json"
    payload = {
        "width": scene.width,
        "height": scene.height,
        "background": list(scene.background),
        "instances": [
            {
                "instance_id": ann.instance_id,
                "category": ann.category,
                "occlusion": ann.occlusion,
                "bbox": (
                    None
                    if ann.bbox is None
                    else [ann.bbox.x0, ann.bbox.y0, ann.bbox.x1, ann.bbox.y1]
                ),
            }
            for ann in annotations
        ],
    }
    annotation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return RenderResult(
        image_path=image_path,
        annotation_path=annotation_path,
        annotations=tuple(annotations),
    )
