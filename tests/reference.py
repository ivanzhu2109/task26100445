"""Independent brute-force reference implementation (pure Python, per-pixel).

Deliberately shares no code with the synthocc package so test results are
cross-checked against an independent computation of the same spec.
"""
from __future__ import annotations

import math


def ref_sample_pixel(asset, inv, px, py):
    """Sample one canvas pixel. asset: list of rows of (r,g,b,a) tuples."""
    cx, cy = px + 0.5, py + 0.5
    a, b, c, d, e, f = inv
    sx = a * cx + b * cy + c
    sy = d * cx + e * cy + f
    fx, fy = sx - 0.5, sy - 0.5
    x0, y0 = math.floor(fx), math.floor(fy)
    wx, wy = fx - x0, fy - y0
    h = len(asset)
    w = len(asset[0])
    rgb = [0.0, 0.0, 0.0]
    alpha = 0.0
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            wt = (wx if dx else 1.0 - wx) * (wy if dy else 1.0 - wy)
            if 0 <= xi < w and 0 <= yi < h:
                r, g, bl, al = asset[yi][xi]
                af = al / 255.0
                rgb[0] += r * af * wt
                rgb[1] += g * af * wt
                rgb[2] += bl * af * wt
                alpha += af * wt
    return rgb, alpha


def ref_render(width, height, background, instances):
    """instances: list of (instance_id, category, asset, inv6) back-to-front.

    Returns (image_float, annotations) where annotations is a list of
    (instance_id, category, occlusion, bbox_or_None).
    """
    layers = []
    for _id, _cat, asset, inv in instances:
        alphas = [[0.0] * width for _ in range(height)]
        rgbs = [[(0.0, 0.0, 0.0)] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                rgb, alpha = ref_sample_pixel(asset, inv, x, y)
                rgbs[y][x] = tuple(rgb)
                alphas[y][x] = alpha
        layers.append((rgbs, alphas))

    image = [[[float(c) for c in background] for _ in range(width)] for _ in range(height)]
    for rgbs, alphas in layers:
        for y in range(height):
            for x in range(width):
                a = alphas[y][x]
                for ch in range(3):
                    image[y][x][ch] = rgbs[y][x][ch] + (1.0 - a) * image[y][x][ch]

    annotations = []
    for i, (iid, cat, _asset, _inv) in enumerate(instances):
        alpha_sum = 0.0
        visible_sum = 0.0
        min_x = min_y = None
        max_x = max_y = None
        for y in range(height):
            for x in range(width):
                own = layers[i][1][y][x]
                front = 1.0
                for j in range(i + 1, len(instances)):
                    front *= 1.0 - layers[j][1][y][x]
                vis = own * front
                alpha_sum += own
                visible_sum += vis
                if vis > 0.0:
                    min_x = x if min_x is None else min(min_x, x)
                    min_y = y if min_y is None else min(min_y, y)
                    max_x = x if max_x is None else max(max_x, x)
                    max_y = y if max_y is None else max(max_y, y)
        if alpha_sum == 0.0:
            continue
        occlusion = 100.0 * (1.0 - visible_sum / alpha_sum)
        bbox = None if min_x is None else (min_x, min_y, max_x + 1, max_y + 1)
        annotations.append((iid, cat, occlusion, bbox))
    return image, annotations
