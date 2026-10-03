"""Shared test helpers: procedural RGBA assets and transform builders."""
import math

import numpy as np
from PIL import Image

from occlugen import AffineTransform


def make_asset_pixels():
    """6x6 RGBA asset with a transparent hole, semi-transparent border,
    and a fully transparent pixel whose RGB is non-zero."""
    px = np.zeros((6, 6, 4), dtype=np.uint8)
    px[..., 0] = 200  # red body
    px[..., 1] = 20
    px[..., 2] = 60
    px[..., 3] = 255
    # semi-transparent border
    px[0, :, 3] = 128
    px[-1, :, 3] = 90
    px[:, 0, 3] = 64
    # transparent hole (RGB zeroed)
    px[2:4, 2:4] = 0
    # fully transparent but RGB non-zero (hidden color must not leak)
    px[4, 4] = (77, 33, 201, 0)
    return px


def save_asset(tmp_path, name, pixels):
    path = tmp_path / name
    Image.fromarray(pixels).save(path)
    return str(path)


def asset_rows(pixels):
    """Convert uint8 RGBA array to reference-format rows (alpha 0..1)."""
    return [
        [tuple(int(v) for v in px[:3]) + (px[3] / 255.0,) for px in row]
        for row in pixels
    ]


def rot_scale_translate(angle_deg, scale, tx, ty):
    """Canvas = T(tx, ty) * R(angle) * S(scale) * src, as 6-tuple."""
    rad = math.radians(angle_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    a = scale * cos
    b = -scale * sin
    d = scale * sin
    e = scale * cos
    return (a, b, tx, d, e, ty)


def to_affine(t):
    return AffineTransform(a=t[0], b=t[1], c=t[2], d=t[3], e=t[4], f=t[5])
