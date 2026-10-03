"""Back-to-front alpha compositing on the 8-bit RGB scale."""
from __future__ import annotations

import numpy as np


def composite(
    background: tuple[int, int, int],
    layers: list[tuple[np.ndarray, np.ndarray]],
    width: int,
    height: int,
) -> np.ndarray:
    """Composite premultiplied (rgb, alpha) layers back-to-front.

    Arithmetic stays in float on the 0..255 scale; the caller rounds once
    at output time. Returns float64 HxWx3.
    """
    result = np.empty((height, width, 3), dtype=np.float64)
    result[...] = np.asarray(background, dtype=np.float64)
    for rgb, alpha in layers:
        result = rgb + (1.0 - alpha)[..., None] * result
    return result
