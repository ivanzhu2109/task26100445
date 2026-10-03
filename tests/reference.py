"""Independent slow reference implementation used only by tests.

Deliberately written as plain per-pixel Python loops so it shares no code
with the vectorized library under test.
"""
import math


def invert(t):
    a, b, c, d, e, f = t
    det = a * e - b * d
    ia, ib = e / det, -b / det
    id_, ie = -d / det, a / det
    ic = -(ia * c + ib * f)
    if_ = -(id_ * c + ie * f)
    return (ia, ib, ic, id_, ie, if_)


def sample_pixel(asset, sx, sy):
    """Premultiplied bilinear sample of an RGBA asset at source point.

    asset: list of rows, each pixel (r, g, b, a) with a in 0..1.
    Returns (pr, pg, pb, alpha); outside the asset is transparent black.
    """
    h = len(asset)
    w = len(asset[0])
    fx, fy = sx - 0.5, sy - 0.5
    x0, y0 = math.floor(fx), math.floor(fy)
    dx, dy = fx - x0, fy - y0
    pr = pg = pb = pa = 0.0
    for cx, cy, wgt in (
        (x0, y0, (1 - dx) * (1 - dy)),
        (x0 + 1, y0, dx * (1 - dy)),
        (x0, y0 + 1, (1 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    ):
        if 0 <= cx < w and 0 <= cy < h:
            r, g, b, a = asset[cy][cx]
            pr += wgt * (r * a)
            pg += wgt * (g * a)
            pb += wgt * (b * a)
            pa += wgt * a
    return pr, pg, pb, pa


def render(width, height, background, instances):
    """instances: list of (instance_id, category, asset, transform6).

    Returns (image_rows, annotations) where image_rows are integer RGB
    tuples and annotations map instance_id -> (bbox_or_None, occlusion).
    """
    layers = []
    for inst_id, cat, asset, t in instances:
        ia, ib, ic, id_, ie, if_ = invert(t)
        layer = []
        for y in range(height):
            row = []
            for x in range(width):
                sx = ia * (x + 0.5) + ib * (y + 0.5) + ic
                sy = id_ * (x + 0.5) + ie * (y + 0.5) + if_
                row.append(sample_pixel(asset, sx, sy))
            layer.append(row)
        layers.append((inst_id, cat, layer))

    image = []
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = (float(v) for v in background)
            for _, _, layer in layers:
                pr, pg, pb, pa = layer[y][x]
                r = pr + (1 - pa) * r
                g = pg + (1 - pa) * g
                b = pb + (1 - pa) * b
            row.append((int(math.floor(r + 0.5)),
                        int(math.floor(g + 0.5)),
                        int(math.floor(b + 0.5))))
        image.append(row)

    annotations = {}
    n = len(layers)
    for i, (inst_id, cat, layer) in enumerate(layers):
        alpha_sum = 0.0
        vis_sum = 0.0
        min_x = min_y = None
        max_x = max_y = None
        for y in range(height):
            for x in range(width):
                pa = layer[y][x][3]
                alpha_sum += pa
                trans = 1.0
                for j in range(i + 1, n):
                    trans *= 1.0 - layers[j][2][y][x][3]
                vis = pa * trans
                vis_sum += vis
                if vis > 0.0:
                    if min_x is None:
                        min_x = max_x = x
                        min_y = max_y = y
                    else:
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
        if alpha_sum == 0.0:
            continue
        occlusion = 100.0 * (1.0 - vis_sum / alpha_sum)
        if min_x is None:
            bbox = None
        else:
            bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        annotations[inst_id] = (cat, bbox, occlusion)
    return image, annotations
