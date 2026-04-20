"""Unit tests for skycop.cv.inference — class remap + Detection shape.

The YoloDetector itself isn't exercised (it would pull ultralytics and
weights). We test the remap logic that sits in front of it.
"""

from skycop.cv.inference import Detection, YoloDetector, coco_to_skycop_map


def test_coco_to_skycop_map_normalizes_keys():
    cfg_style = {2: 0, "5": 3, 7: 2}  # OmegaConf often gives str keys
    m = coco_to_skycop_map(cfg_style)
    assert m == {2: 0, 5: 3, 7: 2}


def test_coco_to_skycop_map_passes_through_none():
    """A None value for a key means 'drop this class' — must survive the conversion."""
    m = coco_to_skycop_map({2: 0, 4: None})
    assert m[2] == 0
    assert m[4] is None


def test_yolo_detector_remap_drops_unmapped_classes():
    det = YoloDetector(weights="fake.pt", class_remap={2: 0, 7: 2})
    assert det._remap(2) == 0
    assert det._remap(7) == 2
    # Class not in map → None (dropped)
    assert det._remap(99) is None


def test_yolo_detector_remap_drops_explicit_none():
    det = YoloDetector(weights="fake.pt", class_remap={2: 0, 4: None})
    assert det._remap(2) == 0
    assert det._remap(4) is None


def test_yolo_detector_identity_remap_when_none():
    """No remap provided → pass class_idx through unchanged."""
    det = YoloDetector(weights="fake.pt", class_remap=None)
    assert det._remap(0) == 0
    assert det._remap(3) == 3
    assert det._remap(99) == 99


def test_detection_cxcywh_normalized_matches_xyxy():
    d = Detection(class_idx=0, confidence=0.9, x1=10, y1=20, x2=60, y2=70)
    cx, cy, w, h = d.cxcywh_normalized
    assert cx == 35
    assert cy == 45
    assert w == 50
    assert h == 50
