"""Premultiplied-alpha bilinear sampling of RGBA assets onto the canvas."""
from __future__ import annotations

import numpy as np

from .types import Affine2D


def sample_premultiplied(
    asset_rgba: np.ndarray, canvas_to_src: Affine2D, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample asset onto a width x height canvas.

    Returns (rgb, alpha): rgb is float64 HxWx3 premultiplied color on the
    0..255 scale, alpha is float64 HxW in 0..1. Outside the asset is
    transparent black; RGB hidden under fully transparent texels never
    contributes.
    """
    src = asset_rgba.astype(np.float64)
    alpha = src[..., 3] / 255.0
    premult = src[..., :3] * alpha[..., None]

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    cx = xs + 0.5
    cy = ys + 0.5
    sx = canvas_to_src.a * cx + canvas_to_src.b * cy + canvas_to_src.c
    sy = canvas_to_src.d * cx + canvas_to_src.e * cy + canvas_to_src.f

    fx = sx - 0.5
    fy = sy - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    wx = fx - x0
    wy = fy - y0

    sh, sw = src.shape[0], src.shape[1]
    out_rgb = np.zeros((height, width, 3), dtype=np.float64)
    out_a = np.zeros((height, width), dtype=np.float64)
    for dy in (0, 1):
        for dx in (0, 1):
            xi = x0 + dx
            yi = y0 + dy
            w = ((1.0 - wx) if dx == 0 else wx) * ((1.0 - wy) if dy == 0 else wy)
            valid = (xi >= 0) & (xi < sw) & (yi >= 0) & (yi < sh)
            xic = np.clip(xi, 0, sw - 1)
            yic = np.clip(yi, 0, sh - 1)
            wv = w * valid
            out_rgb += premult[yic, xic] * wv[..., None]
            out_a += alpha[yic, xic] * wv
    return out_rgb, out_a
