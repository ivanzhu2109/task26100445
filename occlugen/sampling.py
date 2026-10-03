"""Premultiplied-alpha bilinear sampling of RGBA assets."""
from __future__ import annotations

import numpy as np

from .types import AffineTransform


def sample_premultiplied(
    asset: np.ndarray,
    canvas_to_src: AffineTransform,
    out_width: int,
    out_height: int,
) -> np.ndarray:
    """Sample an RGBA asset onto the canvas grid.

    Returns float array of shape (out_height, out_width, 4) with
    premultiplied RGB (0..255 scale) and alpha (0..1).
    """
    if asset.ndim != 3 or asset.shape[2] != 4:
        raise ValueError("asset must have shape (H, W, 4)")
    src_h, src_w = asset.shape[0], asset.shape[1]

    # Canvas pixel centers: (x + 0.5, y + 0.5), mapped into source space.
    xs = np.arange(out_width, dtype=np.float64) + 0.5
    ys = np.arange(out_height, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    sx = canvas_to_src.a * gx + canvas_to_src.b * gy + canvas_to_src.c
    sy = canvas_to_src.d * gx + canvas_to_src.e * gy + canvas_to_src.f

    # Bilinear weights in pixel-index space (pixel centers at i + 0.5).
    fx = sx - 0.5
    fy = sy - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    dx = fx - x0
    dy = fy - y0

    # Premultiply RGB by alpha; fully transparent pixels contribute zero.
    alpha = asset[..., 3]
    premul = np.empty_like(asset)
    premul[..., 0:3] = asset[..., 0:3] * alpha[..., None]
    premul[..., 3] = alpha

    out = np.zeros((out_height, out_width, 4), dtype=np.float64)
    corners = (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    )
    for cx, cy, w in corners:
        valid = (cx >= 0) & (cx < src_w) & (cy >= 0) & (cy < src_h)
        if not np.any(valid):
            continue
        contrib = premul[cy[valid], cx[valid]] * w[valid][:, None]
        out[valid] += contrib
    return out
