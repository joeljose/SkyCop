"""Unit tests for the instance-seg → YOLO label pipeline (no CARLA required)."""

from pathlib import Path

import numpy as np
import pytest

from skycop.cv.dataset import (
    BBox,
    DatasetManifest,
    FrameStats,
    extract_yolo_labels_from_seg,
    write_yolo_label,
)
from skycop.cv.vehicle_classes import (
    CLASS_INDEX,
    CLASS_NAMES,
    class_index,
    classify_blueprint,
)

# ── classify_blueprint ────────────────────────────────────────────────────

def test_four_wheel_defaults_to_car():
    assert classify_blueprint("vehicle.audi.a2", 4) == "car"
    assert classify_blueprint("vehicle.tesla.model3", 4) == "car"


def test_all_two_wheelers_are_dropped():
    """2-wheelers (bikes + motorcycles) excluded — TM can't autopilot them."""
    assert classify_blueprint("vehicle.harley-davidson.low_rider", 2) is None
    assert classify_blueprint("vehicle.kawasaki.ninja", 2) is None
    assert classify_blueprint("vehicle.vespa.zx125", 2) is None
    assert classify_blueprint("vehicle.bh.crossbike", 2) is None
    assert classify_blueprint("vehicle.diamondback.century", 2) is None
    assert classify_blueprint("vehicle.gazelle.omafiets", 2) is None


def test_known_van_and_truck_patterns():
    assert classify_blueprint("vehicle.carlamotors.firetruck", 4) == "truck"
    assert classify_blueprint("vehicle.carlamotors.carlacola", 4) == "truck"
    assert classify_blueprint("vehicle.tesla.cybertruck", 4) == "truck"
    assert classify_blueprint("vehicle.ford.ambulance", 4) == "van"
    assert classify_blueprint("vehicle.mercedes.sprinter", 4) == "van"
    assert classify_blueprint("vehicle.volkswagen.t2", 4) == "van"


def test_bus_pattern():
    assert classify_blueprint("vehicle.mitsubishi.fusorosa", 4) == "bus"


def test_class_names_cover_four_classes():
    assert CLASS_NAMES == ["car", "van", "truck", "bus"]
    assert CLASS_INDEX["car"] == 0
    assert CLASS_INDEX["bus"] == 3
    assert "motorcycle" not in CLASS_INDEX


def test_class_index_raises_on_unknown():
    with pytest.raises(KeyError):
        class_index("submarine")


# ── extract_yolo_labels_from_seg ──────────────────────────────────────────

def _seg(h: int, w: int):
    """Blank seg image (H, W, 3) BGR, uint8. All actor_id = 0."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _paint(seg: np.ndarray, rect: tuple[int, int, int, int], actor_id: int) -> None:
    """Paint a rectangle x1,y1,x2,y2 with the given actor_id.

    CARLA 0.9.16 instance-seg: G = id low byte, B = id high byte.
    In BGR memory layout: index 0 = B, index 1 = G.
    """
    x1, y1, x2, y2 = rect
    seg[y1:y2, x1:x2, 0] = (actor_id >> 8) & 0xFF  # B = high
    seg[y1:y2, x1:x2, 1] = actor_id & 0xFF         # G = low


def test_single_actor_emits_one_normalized_bbox():
    seg = _seg(100, 200)
    _paint(seg, (40, 20, 80, 60), actor_id=7)  # 40×40 block

    boxes, stats = extract_yolo_labels_from_seg(
        seg,
        actor_id_to_class=lambda aid: 0 if aid == 7 else None,
        min_pixel=10,
        min_visibility=0.3,
    )

    assert len(boxes) == 1
    assert stats.emitted == 1
    b = boxes[0]
    assert b.class_idx == 0
    # cx ~ (40 + 79) / 2 / 200 ≈ 0.2975
    assert 0.29 < b.cx < 0.31
    # cy ~ (20 + 59) / 2 / 100 ≈ 0.395
    assert 0.38 < b.cy < 0.41
    # w = 40/200 = 0.2, h = 40/100 = 0.4
    assert 0.19 < b.w < 0.21
    assert 0.39 < b.h < 0.41


def test_multiple_actors_emit_multiple_boxes():
    seg = _seg(100, 200)
    _paint(seg, (10, 10, 40, 40), actor_id=1)
    _paint(seg, (150, 60, 190, 95), actor_id=2)

    boxes, stats = extract_yolo_labels_from_seg(
        seg,
        actor_id_to_class=lambda aid: {1: 0, 2: 2}.get(aid),
        min_pixel=10,
    )
    assert len(boxes) == 2
    assert stats.emitted == 2
    classes = {b.class_idx for b in boxes}
    assert classes == {0, 2}


def test_below_min_pixel_is_dropped():
    seg = _seg(100, 200)
    _paint(seg, (10, 10, 18, 18), actor_id=3)  # 8×8 box

    boxes, stats = extract_yolo_labels_from_seg(
        seg,
        actor_id_to_class=lambda aid: 0,
        min_pixel=20,
    )
    assert boxes == []
    assert stats.dropped_too_small == 1
    assert stats.emitted == 0


def test_low_visibility_is_dropped():
    """Actor pixels concentrated in a tiny fraction of a large bbox extent →
    visibility ratio low → dropped."""
    seg = _seg(100, 200)
    # Two disconnected small blobs at extreme corners → bbox spans 160×60 but
    # actor pixels = 2 × (10×10) = 200. Visibility = 200 / (160*60) ≈ 0.02.
    _paint(seg, (20, 20, 30, 30), actor_id=9)
    _paint(seg, (170, 70, 180, 80), actor_id=9)

    boxes, stats = extract_yolo_labels_from_seg(
        seg,
        actor_id_to_class=lambda aid: 0,
        min_pixel=10,
        min_visibility=0.3,
    )
    assert boxes == []
    assert stats.dropped_occluded == 1


def test_unknown_class_is_dropped_without_pixel_work():
    seg = _seg(100, 200)
    _paint(seg, (10, 10, 50, 50), actor_id=42)

    boxes, stats = extract_yolo_labels_from_seg(
        seg,
        actor_id_to_class=lambda aid: None,  # always skip
        min_pixel=10,
    )
    assert boxes == []
    assert stats.dropped_unknown_class == 1


def test_actor_id_zero_is_ignored():
    seg = _seg(100, 100)  # all-zero image — background only

    boxes, stats = extract_yolo_labels_from_seg(
        seg,
        actor_id_to_class=lambda aid: 0,
    )
    assert boxes == []
    assert stats.emitted == 0


# ── write_yolo_label ──────────────────────────────────────────────────────

def test_write_yolo_label_round_trips(tmp_path: Path):
    target = tmp_path / "labels" / "frame.txt"
    boxes = [
        BBox(class_idx=0, cx=0.5, cy=0.5, w=0.1, h=0.2),
        BBox(class_idx=3, cx=0.25, cy=0.75, w=0.05, h=0.08),
    ]
    write_yolo_label(target, boxes)

    assert target.exists()
    lines = target.read_text().strip().splitlines()
    assert len(lines) == 2
    first = lines[0].split()
    assert first[0] == "0"
    assert float(first[1]) == pytest.approx(0.5)
    assert float(first[2]) == pytest.approx(0.5)
    second = lines[1].split()
    assert second[0] == "3"


def test_write_yolo_label_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c" / "frame.txt"
    write_yolo_label(target, [BBox(1, 0.5, 0.5, 0.1, 0.1)])
    assert target.exists()


# ── DatasetManifest ───────────────────────────────────────────────────────

def test_manifest_aggregates_class_and_skip_counts(tmp_path: Path):
    manifest = DatasetManifest(
        seed=42,
        fixed_delta_seconds=0.05,
        map_name="Town10HD_Opt",
        class_names=list(CLASS_NAMES),
        min_pixel=20,
        min_visibility=0.3,
    )

    boxes = [BBox(0, 0.5, 0.5, 0.1, 0.1), BBox(2, 0.3, 0.3, 0.1, 0.1)]
    stats = FrameStats(emitted=2, dropped_too_small=1, dropped_occluded=0)
    manifest.record_frame(
        index=0, tick=10,
        camera_pose={"x": 0, "y": 0, "z": 25, "pitch": -90, "yaw": 0},
        suspect_pose={"x": 0, "y": 0, "z": 0},
        boxes=boxes,
        stats=stats,
    )

    assert manifest.class_counts == {"car": 1, "truck": 1}
    assert manifest.skip_counts["dropped_too_small"] == 1
    assert len(manifest.frames) == 1

    out = tmp_path / "manifest.json"
    manifest.save(out)
    assert out.exists()
    raw = out.read_text()
    assert "\"seed\": 42" in raw
    assert "\"class_counts\"" in raw
