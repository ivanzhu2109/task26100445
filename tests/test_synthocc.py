"""End-to-end tests comparing synthocc against an independent reference."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import ref_render  # noqa: E402
from synthocc import Affine2D, Instance, Scene, render_scene  # noqa: E402


def make_affine(scale=1.0, angle=0.0, tx=0.0, ty=0.0, cx=0.0, cy=0.0):
    """Source->canvas: rotate/scale about asset point (cx, cy), then translate."""
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale
    a, b = cos_a, -sin_a
    d, e = sin_a, cos_a
    c = tx - (a * cx + b * cy)
    f = ty - (d * cx + e * cy)
    return Affine2D((a, b, c, d, e, f))


def save_asset(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


@pytest.fixture()
def tricky_asset(tmp_path):
    """8x8 RGBA asset: opaque interior, transparent hole, semi-transparent
    edge column, and a fully transparent pixel with non-zero hidden RGB."""
    asset = np.zeros((8, 8, 4), np.uint8)
    asset[1:7, 1:7] = [200, 40, 90, 255]  # opaque body
    asset[3, 3] = [0, 0, 0, 0]  # transparent hole
    asset[1:7, 6] = [10, 220, 30, 128]  # semi-transparent edge
    asset[0, 0] = [255, 255, 255, 0]  # hidden RGB under full transparency
    asset[7, 7] = [123, 222, 11, 0]  # another hidden-RGB corner
    return save_asset(tmp_path / "tricky.png", asset), asset


def to_asset_list(array: np.ndarray):
    return [[tuple(int(v) for v in px) for px in row] for row in array]


def assert_matches_reference(scene: Scene, assets, result, tmp_path):
    """Cross-check image + annotations against the brute-force reference."""
    instances = [
        (
            inst.instance_id,
            inst.category,
            to_asset_list(assets[inst.instance_id]),
            inst.transform.inverse().m,
        )
        for inst in scene.instances
    ]
    ref_image, ref_anns = ref_render(
        scene.width, scene.height, scene.background, instances
    )

    actual = np.asarray(Image.open(result.image_path)).astype(np.float64)
    expected = np.array(
        [[[round_half_up(c) for c in px] for px in row] for row in ref_image]
    )
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() <= 1.0

    by_id = {ann.instance_id: ann for ann in result.annotations}
    ref_ids = [a[0] for a in ref_anns]
    assert set(by_id) == set(ref_ids)
    assert len(result.annotations) == len(ref_anns)
    for iid, cat, occlusion, bbox in ref_anns:
        ann = by_id[iid]
        assert ann.category == cat
        assert ann.occlusion == pytest.approx(occlusion, abs=1e-6)
        actual_bbox = (
            None
            if ann.bbox is None
            else (ann.bbox.x0, ann.bbox.y0, ann.bbox.x1, ann.bbox.y1)
        )
        assert actual_bbox == bbox


def round_half_up(value):
    return min(255, max(0, math.floor(value + 0.5)))


def test_rotation_scale_subpixel_translation_and_clipping(tricky_asset, tmp_path):
    path, asset = tricky_asset
    # Rotate 33 deg, scale 1.6 about asset center, subpixel translate so the
    # asset is clipped by the right and bottom canvas edges.
    transform = make_affine(
        scale=1.6, angle=math.radians(33.0), tx=9.37, ty=6.13, cx=3.5, cy=3.5
    )
    scene = Scene(
        14,
        12,
        (17, 29, 41),
        (Instance("hero", "widget", path, transform),),
    )
    result = render_scene(scene, tmp_path, "rot")
    assert_matches_reference(scene, {"hero": asset}, result, tmp_path)
    # Clipped-off-canvas area must not count as occlusion.
    assert result.annotations[0].occlusion == pytest.approx(0.0, abs=1e-9)


def test_same_category_instances_annotated_separately(tricky_asset, tmp_path):
    path, asset = tricky_asset
    back = make_affine(tx=2.0, ty=2.0)
    front = make_affine(tx=4.0, ty=4.0)
    scene = Scene(
        16,
        16,
        (0, 0, 0),
        (
            Instance("a", "widget", path, back),
            Instance("b", "widget", path, front),
        ),
    )
    result = render_scene(scene, tmp_path, "samecat")
    assert len(result.annotations) == 2
    assert {a.instance_id for a in result.annotations} == {"a", "b"}
    assert all(a.category == "widget" for a in result.annotations)
    assert_matches_reference(scene, {"a": asset, "b": asset}, result, tmp_path)
    assert result.annotations[0].occlusion > 0.0
    assert result.annotations[1].occlusion == pytest.approx(0.0, abs=1e-9)


def test_multiple_foregrounds_no_double_deduction(tmp_path):
    # 6x6 opaque red square, half covered by a 50%-alpha green square,
    # the same half again covered by a 50%-alpha blue square.
    red = np.zeros((6, 6, 4), np.uint8)
    red[:] = [255, 0, 0, 255]
    green = np.zeros((6, 3, 4), np.uint8)
    green[:] = [0, 255, 0, 128]
    blue = np.zeros((6, 3, 4), np.uint8)
    blue[:] = [0, 0, 255, 128]
    rp = save_asset(tmp_path / "r.png", red)
    gp = save_asset(tmp_path / "g.png", green)
    bp = save_asset(tmp_path / "b.png", blue)
    scene = Scene(
        8,
        8,
        (9, 9, 9),
        (
            Instance("red", "sq", rp, make_affine(tx=1.0, ty=1.0)),
            Instance("green", "sq", gp, make_affine(tx=4.0, ty=1.0)),
            Instance("blue", "sq", bp, make_affine(tx=4.0, ty=1.0)),
        ),
    )
    result = render_scene(scene, tmp_path, "stack")
    assert_matches_reference(
        scene,
        {"red": red, "green": green, "blue": blue},
        result,
        tmp_path,
    )
    anns = {a.instance_id: a for a in result.annotations}
    # Red's right half keeps visibility (1-128/255)^2 of its alpha.
    keep = (1.0 - 128.0 / 255.0) ** 2
    expected_occ = 100.0 * (1.0 - (18.0 + 18.0 * keep) / 36.0)
    assert anns["red"].occlusion == pytest.approx(expected_occ, abs=1e-6)
    assert anns["green"].occlusion == pytest.approx(100.0 * (1.0 - (1.0 - 128.0 / 255.0)), abs=1e-6)
    assert anns["blue"].occlusion == pytest.approx(0.0, abs=1e-9)


def test_fully_occluded_instance_kept_with_empty_bbox(tmp_path):
    back = np.zeros((4, 4, 4), np.uint8)
    back[:] = [255, 255, 0, 255]
    front = np.zeros((6, 6, 4), np.uint8)
    front[:] = [0, 255, 255, 255]
    bp = save_asset(tmp_path / "back.png", back)
    fp = save_asset(tmp_path / "front.png", front)
    scene = Scene(
        8,
        8,
        (0, 0, 0),
        (
            Instance("hidden", "thing", bp, make_affine(tx=1.0, ty=1.0)),
            Instance("cover", "thing", fp, make_affine(tx=0.0, ty=0.0)),
        ),
    )
    result = render_scene(scene, tmp_path, "occl")
    assert len(result.annotations) == 2
    hidden = result.annotations[0]
    assert hidden.instance_id == "hidden"
    assert hidden.occlusion == pytest.approx(100.0, abs=1e-9)
    assert hidden.bbox is None
    assert_matches_reference(
        scene, {"hidden": back, "cover": front}, result, tmp_path
    )


def test_clipping_is_not_occlusion(tmp_path):
    asset = np.zeros((4, 4, 4), np.uint8)
    asset[:] = [10, 20, 30, 255]
    path = save_asset(tmp_path / "clip.png", asset)
    # Only a 2x2 corner of the asset lands inside the 6x6 canvas.
    scene = Scene(
        6, 6, (0, 0, 0), (Instance("c", "sq", path, make_affine(tx=4.0, ty=4.0)),)
    )
    result = render_scene(scene, tmp_path, "clip")
    assert result.annotations[0].occlusion == pytest.approx(0.0, abs=1e-9)
    bbox = result.annotations[0].bbox
    assert (bbox.x0, bbox.y0, bbox.x1, bbox.y1) == (4, 4, 6, 6)
    assert_matches_reference(scene, {"c": asset}, result, tmp_path)


def test_hidden_rgb_under_transparency_never_leaks(tmp_path):
    asset = np.zeros((2, 2, 4), np.uint8)
    asset[0, 0] = [255, 255, 255, 0]  # bright hidden RGB, zero alpha
    path = save_asset(tmp_path / "hidden.png", asset)
    scene = Scene(4, 4, (3, 5, 7), (Instance("h", "t", path, make_affine()),))
    result = render_scene(scene, tmp_path, "hiddenrgb")
    img = np.asarray(Image.open(result.image_path))
    assert (img == [3, 5, 7]).all()
    assert result.annotations == ()


def test_json_annotation_file_contents(tricky_asset, tmp_path):
    path, asset = tricky_asset
    scene = Scene(
        10,
        10,
        (1, 2, 3),
        (Instance("one", "cat1", path, make_affine(tx=1.0, ty=1.0)),),
    )
    result = render_scene(scene, tmp_path, "jsoncheck")
    payload = json.loads(result.annotation_path.read_text())
    assert payload["width"] == 10 and payload["height"] == 10
    assert payload["background"] == [1, 2, 3]
    entry = payload["instances"][0]
    assert entry["instance_id"] == "one"
    assert entry["category"] == "cat1"
    assert entry["occlusion"] == pytest.approx(0.0, abs=1e-9)
    x0, y0, x1, y1 = entry["bbox"]
    assert x1 > x0 and y1 > y0
