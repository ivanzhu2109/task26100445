import json

import numpy as np
import pytest
from PIL import Image

import reference
from helpers import (
    asset_rows,
    make_asset_pixels,
    rot_scale_translate,
    save_asset,
    to_affine,
)
from occlugen import (
    InstanceSpec,
    SceneSpec,
    load_asset,
    render_scene,
    render_to_dir,
)

BG = (11, 22, 33)


def run_both(tmp_path, width, height, instances):
    """Render with the library and the independent reference; return both."""
    pixels = make_asset_pixels()
    asset_path = save_asset(tmp_path, "asset.png", pixels)
    rows = asset_rows(pixels)

    specs = []
    ref_instances = []
    for inst_id, cat, t in instances:
        specs.append(
            InstanceSpec(
                instance_id=inst_id,
                category=cat,
                asset_path=asset_path,
                transform=to_affine(t),
            )
        )
        ref_instances.append((inst_id, cat, rows, t))

    scene = SceneSpec(
        width=width, height=height, background=BG, instances=tuple(specs)
    )
    image, anns = render_scene(scene, load_asset)
    ref_image, ref_anns = reference.render(width, height, BG, ref_instances)
    return image, anns, ref_image, ref_anns


def assert_matches_reference(image, anns, ref_image, ref_anns):
    ref_arr = np.array(ref_image, dtype=np.uint8)
    assert image.shape == ref_arr.shape
    assert np.array_equal(image, ref_arr), (
        f"image mismatch at {np.argwhere(image != ref_arr)[:5]}"
    )
    assert {a.instance_id for a in anns} == set(ref_anns)
    for ann in anns:
        cat, bbox, occ = ref_anns[ann.instance_id]
        assert ann.category == cat
        if bbox is None:
            assert ann.bbox is None
        else:
            assert ann.bbox is not None
            assert (ann.bbox.x, ann.bbox.y, ann.bbox.width, ann.bbox.height) == bbox
        assert ann.occlusion_rate == pytest.approx(occ, abs=1e-9)


def test_rotation_scale_subpixel_translation_and_clipping(tmp_path):
    # Rotated 30 deg, scaled 1.7x, sub-pixel translation, partly off-canvas.
    image, anns, ref_image, ref_anns = run_both(
        tmp_path, 40, 30, [("inst-1", "box", rot_scale_translate(30, 1.7, -2.3, 5.8))]
    )
    assert_matches_reference(image, anns, ref_image, ref_anns)


def test_clipped_instance_has_zero_occlusion(tmp_path):
    # Half of the instance hangs off the canvas; nothing in front of it.
    image, anns, _, _ = run_both(
        tmp_path, 8, 8, [("clipped", "box", rot_scale_translate(0, 1.0, -3.0, -3.0))]
    )
    assert len(anns) == 1
    assert anns[0].occlusion_rate == pytest.approx(0.0, abs=1e-12)
    assert (anns[0].bbox.x, anns[0].bbox.y) == (0, 0)


def test_same_category_instances_annotated_separately(tmp_path):
    image, anns, ref_image, ref_anns = run_both(
        tmp_path,
        16,
        12,
        [
            ("a", "box", rot_scale_translate(0, 1.0, 2.0, 2.0)),
            ("b", "box", rot_scale_translate(0, 1.0, 4.0, 4.0)),
        ],
    )
    assert_matches_reference(image, anns, ref_image, ref_anns)
    assert [a.instance_id for a in anns] == ["a", "b"]
    assert all(a.category == "box" for a in anns)
    # Back instance is partially covered by the front one.
    assert anns[0].occlusion_rate > 0.0
    assert anns[1].occlusion_rate == pytest.approx(0.0, abs=1e-12)


def test_multiple_foregrounds_cover_same_region(tmp_path):
    # Three stacked layers over one region: occlusion must accumulate
    # exactly once per covering layer (no double subtraction).
    opaque = np.zeros((4, 4, 4), dtype=np.uint8)
    opaque[..., 0] = 255
    opaque[..., 3] = 255
    rows = asset_rows(opaque)
    path = save_asset(tmp_path, "opaque.png", opaque)

    identity = (1.0, 0.0, 4.0, 0.0, 1.0, 4.0)
    specs = [
        InstanceSpec(f"L{i}", "layer", path, to_affine(identity)) for i in range(3)
    ]
    scene = SceneSpec(width=12, height=12, background=BG, instances=tuple(specs))
    image, anns = render_scene(scene, load_asset)
    _, ref_anns = reference.render(
        12, 12, BG, [(f"L{i}", "layer", rows, identity) for i in range(3)]
    )
    assert anns[0].occlusion_rate == pytest.approx(100.0)
    assert anns[1].occlusion_rate == pytest.approx(100.0)
    assert anns[2].occlusion_rate == pytest.approx(0.0)
    for ann in anns:
        assert ann.occlusion_rate == pytest.approx(
            ref_anns[ann.instance_id][2], abs=1e-9
        )


def test_fully_occluded_instance_keeps_empty_annotation(tmp_path):
    opaque = np.zeros((4, 4, 4), dtype=np.uint8)
    opaque[..., 1] = 255
    opaque[..., 3] = 255
    path = save_asset(tmp_path, "opaque.png", opaque)
    back = InstanceSpec("back", "box", path, to_affine((1, 0, 2, 0, 1, 2)))
    front = InstanceSpec("front", "box", path, to_affine((2, 0, 1, 0, 2, 1)))
    scene = SceneSpec(width=16, height=16, background=BG, instances=(back, front))
    _, anns = render_scene(scene, load_asset)
    assert len(anns) == 2
    assert anns[0].instance_id == "back"
    assert anns[0].occlusion_rate == pytest.approx(100.0)
    assert anns[0].bbox is None
    assert anns[1].occlusion_rate == pytest.approx(0.0)


def test_zero_alpha_instances_produce_no_annotation(tmp_path):
    pixels = make_asset_pixels()
    asset_path = save_asset(tmp_path, "asset.png", pixels)
    # Fully off-canvas: sampled alpha sums to zero over the canvas.
    off = InstanceSpec("off", "box", asset_path, to_affine((1, 0, 100, 0, 1, 100)))
    on = InstanceSpec("on", "box", asset_path, to_affine((1, 0, 1, 0, 1, 1)))
    scene = SceneSpec(width=10, height=10, background=BG, instances=(off, on))
    _, anns = render_scene(scene, load_asset)
    assert [a.instance_id for a in anns] == ["on"]


def test_hidden_rgb_of_transparent_pixels_does_not_leak(tmp_path):
    # Single fully transparent pixel with non-zero RGB, enlarged 4x.
    px = np.zeros((2, 2, 4), dtype=np.uint8)
    px[0, 0] = (250, 130, 70, 0)
    path = save_asset(tmp_path, "hidden.png", px)
    inst = InstanceSpec("h", "box", path, to_affine((4, 0, 0, 0, 4, 0)))
    scene = SceneSpec(width=8, height=8, background=(9, 8, 7), instances=(inst,))
    image, anns = render_scene(scene, load_asset)
    assert anns == []
    assert np.all(image == np.array([9, 8, 7], dtype=np.uint8))


def test_render_to_dir_writes_png_and_per_instance_json(tmp_path):
    pixels = make_asset_pixels()
    asset_path = save_asset(tmp_path, "asset.png", pixels)
    specs = (
        InstanceSpec("one", "box", asset_path, to_affine((1, 0, 1, 0, 1, 1))),
        InstanceSpec(
            "two", "box", asset_path, to_affine(rot_scale_translate(15, 1.2, 3.5, 2.5))
        ),
    )
    scene = SceneSpec(width=20, height=16, background=BG, instances=specs)
    out_dir = tmp_path / "out"
    anns = render_to_dir(scene, str(out_dir))

    img = np.asarray(Image.open(out_dir / "image.png").convert("RGB"))
    expected_image, expected_anns = render_scene(scene, load_asset)
    assert np.array_equal(img, expected_image)

    assert len(anns) == len(expected_anns) == 2
    for ann in anns:
        with open(out_dir / f"{ann.instance_id}.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["instance_id"] == ann.instance_id
        assert data["category"] == ann.category
        assert data["occlusion_rate"] == pytest.approx(ann.occlusion_rate)
        if ann.bbox is None:
            assert data["bbox"] is None
        else:
            assert data["bbox"] == {
                "x": ann.bbox.x,
                "y": ann.bbox.y,
                "width": ann.bbox.width,
                "height": ann.bbox.height,
            }
