"""Scene compositing and per-instance occlusion annotation."""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .sampling import sample_premultiplied
from .types import BBox, InstanceAnnotation, SceneSpec


def render_scene(
    scene: SceneSpec,
    asset_loader: Callable[[str], np.ndarray],
) -> Tuple[np.ndarray, List[InstanceAnnotation]]:
    """Render the scene and compute per-instance annotations.

    Returns (uint8 RGB image of shape (H, W, 3), annotations).
    """
    h, w = scene.height, scene.width

    sampled = []
    for inst in scene.instances:
        asset = asset_loader(inst.asset_path)
        premul = sample_premultiplied(asset, inst.transform.inverse(), w, h)
        sampled.append(premul)

    # Back-to-front "over" compositing, fully in float; quantize only at
    # the very end. Background is opaque, so final alpha is 1 everywhere.
    rgb = np.empty((h, w, 3), dtype=np.float64)
    rgb[...] = np.asarray(scene.background, dtype=np.float64)
    for premul in sampled:
        alpha = premul[..., 3:4]
        rgb = premul[..., 0:3] + (1.0 - alpha) * rgb

    image = np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)

    # Visibility: weight_i = alpha_i * prod_{j>i} (1 - alpha_j), computed
    # with a running transmittance swept from front to back.
    annotations: List[InstanceAnnotation] = []
    front_trans = np.ones((h, w), dtype=np.float64)
    visible_weights = []
    for premul in reversed(sampled):
        alpha = premul[..., 3]
        visible_weights.append(alpha * front_trans)
        front_trans = front_trans * (1.0 - alpha)
    visible_weights.reverse()

    for inst, premul, vis in zip(scene.instances, sampled, visible_weights):
        alpha_sum = float(premul[..., 3].sum())
        if alpha_sum == 0.0:
            continue
        vis_sum = float(vis.sum())
        occlusion = 100.0 * (1.0 - vis_sum / alpha_sum)
        mask = vis > 0.0
        if np.any(mask):
            ys, xs = np.nonzero(mask)
            bbox = BBox(
                x=int(xs.min()),
                y=int(ys.min()),
                width=int(xs.max()) - int(xs.min()) + 1,
                height=int(ys.max()) - int(ys.min()) + 1,
            )
        else:
            bbox = None
        annotations.append(
            InstanceAnnotation(
                instance_id=inst.instance_id,
                category=inst.category,
                bbox=bbox,
                occlusion_rate=occlusion,
            )
        )
    return image, annotations
