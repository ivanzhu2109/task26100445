"""Per-instance visibility weights, occlusion ratios, and visible bboxes."""
from __future__ import annotations

import numpy as np

from .types import BBox, InstanceAnnotation


def compute_annotations(
    alphas: list[np.ndarray],
    instance_ids: list[str],
    categories: list[str],
) -> list[InstanceAnnotation]:
    """Compute annotations from per-instance sampled alpha maps (back-to-front)."""
    annotations: list[InstanceAnnotation] = []
    front_transmittance = None  # product of (1 - alpha) over layers in front
    for index in range(len(alphas) - 1, -1, -1):
        alpha = alphas[index]
        if front_transmittance is None:
            visible = alpha
        else:
            visible = alpha * front_transmittance
        alpha_sum = float(alpha.sum())
        if alpha_sum > 0.0:
            visible_sum = float(visible.sum())
            occlusion = 100.0 * (1.0 - visible_sum / alpha_sum)
            ys, xs = np.nonzero(visible > 0.0)
            bbox = (
                BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                if xs.size
                else None
            )
            annotations.append(
                InstanceAnnotation(
                    instance_id=instance_ids[index],
                    category=categories[index],
                    occlusion=occlusion,
                    bbox=bbox,
                )
            )
        transmittance = 1.0 - alpha
        front_transmittance = (
            transmittance
            if front_transmittance is None
            else front_transmittance * transmittance
        )
    annotations.reverse()
    return annotations
