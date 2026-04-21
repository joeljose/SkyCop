"""Computer vision — detection, tracking, fingerprinting, re-ID."""

from skycop.cv.dataset import (
    BBox,
    DatasetManifest,
    FrameStats,
    extract_yolo_labels_from_seg,
    write_yolo_label,
)
from skycop.cv.eval import EvalBox, EvalMetrics, read_yolo_labels, score_predictions
from skycop.cv.inference import Detection, YoloDetector, coco_to_skycop_map
from skycop.cv.vehicle_classes import (
    CLASS_INDEX,
    CLASS_NAMES,
    FINGERPRINT_CLASSES,
    FINGERPRINT_INDEX,
    class_index,
    classify_blueprint,
    detector_class_for,
)

__all__ = [
    "BBox",
    "DatasetManifest",
    "FrameStats",
    "extract_yolo_labels_from_seg",
    "write_yolo_label",
    "CLASS_INDEX",
    "CLASS_NAMES",
    "FINGERPRINT_CLASSES",
    "FINGERPRINT_INDEX",
    "class_index",
    "classify_blueprint",
    "detector_class_for",
    "Detection",
    "YoloDetector",
    "coco_to_skycop_map",
    "EvalBox",
    "EvalMetrics",
    "read_yolo_labels",
    "score_predictions",
]
