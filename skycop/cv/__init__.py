"""Computer vision — detection, tracking, fingerprinting, re-ID."""

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

__all__ = [
    "BBox",
    "DatasetManifest",
    "FrameStats",
    "extract_yolo_labels_from_seg",
    "write_yolo_label",
    "CLASS_INDEX",
    "CLASS_NAMES",
    "class_index",
    "classify_blueprint",
]
