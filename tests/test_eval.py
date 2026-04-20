"""Unit tests for skycop.cv.eval — YOLO label reading + class filter.

Score computation is exercised via torchmetrics; we only test the shape
and filter behaviour, not the mAP values themselves.
"""

from pathlib import Path

from skycop.cv.eval import EvalBox, filter_classes, read_yolo_labels


def test_read_yolo_labels_parses_normalized_to_pixels(tmp_path: Path):
    label = tmp_path / "frame.txt"
    label.write_text("0 0.5 0.5 0.1 0.2\n3 0.25 0.75 0.05 0.08\n")
    boxes = read_yolo_labels(label, img_w=200, img_h=100)
    assert len(boxes) == 2

    # First: cls=0, cx=0.5, cy=0.5, w=0.1, h=0.2 → x in [90, 110], y in [40, 60]
    b0 = boxes[0]
    assert b0.class_idx == 0
    assert abs(b0.x1 - 90) < 1e-6
    assert abs(b0.x2 - 110) < 1e-6
    assert abs(b0.y1 - 40) < 1e-6
    assert abs(b0.y2 - 60) < 1e-6

    # Second: cls=3, cx=0.25, cy=0.75, w=0.05, h=0.08 → x centered at 50, y centered at 75
    b1 = boxes[1]
    assert b1.class_idx == 3


def test_read_yolo_labels_returns_empty_for_missing_file(tmp_path: Path):
    assert read_yolo_labels(tmp_path / "nonexistent.txt", 100, 100) == []


def test_read_yolo_labels_returns_empty_for_empty_file(tmp_path: Path):
    label = tmp_path / "empty.txt"
    label.write_text("")
    assert read_yolo_labels(label, 100, 100) == []


def test_filter_classes_keeps_only_requested():
    boxes = [
        EvalBox(class_idx=0, x1=0, y1=0, x2=1, y2=1),
        EvalBox(class_idx=1, x1=0, y1=0, x2=1, y2=1),
        EvalBox(class_idx=2, x1=0, y1=0, x2=1, y2=1),
        EvalBox(class_idx=3, x1=0, y1=0, x2=1, y2=1),
    ]
    filtered = filter_classes(boxes, keep={0, 2, 3})
    assert {b.class_idx for b in filtered} == {0, 2, 3}
    assert len(filtered) == 3


def test_filter_classes_none_is_passthrough():
    boxes = [EvalBox(class_idx=1, x1=0, y1=0, x2=1, y2=1)]
    assert filter_classes(boxes, keep=None) == boxes
