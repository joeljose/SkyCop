"""YOLOv8 inference wrapper.

Thin adapter around `ultralytics.YOLO`. Keeps the rest of the codebase free
of ultralytics-specific shapes — detectors produce `Detection` dataclasses,
class indices are SkyCop indices (not COCO), and the model loads lazily so
tests that import this module don't trigger a weights download.

The class-remap map is injected rather than hardcoded — the same wrapper
serves both the pretrained baseline (COCO class ids) and the fine-tuned
model (SkyCop class ids direct, identity map).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """One detection in image coordinates (pixel-space)."""
    class_idx: int               # SkyCop class index (after remap)
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cxcywh_normalized(self) -> tuple[float, float, float, float]:
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return (self.x1 + w / 2, self.y1 + h / 2, w, h)


class YoloDetector:
    """Lazy-loaded YOLOv8 detector with optional source-class remap.

    `class_remap` maps the model's native class ids → SkyCop class ids.
    Entries mapping to None cause those predictions to be dropped.
    """

    def __init__(
        self,
        weights: str,
        class_remap: dict[int, int | None] | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cuda:0",
        fp16: bool = True,
    ) -> None:
        self.weights = weights
        self.class_remap = class_remap
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.device = device
        self.fp16 = fp16
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO
            # Let ultralytics own device + dtype setup via predict(half=...).
            # Pre-casting the weights to half breaks layer fusion on load.
            self._model = YOLO(self.weights)
        return self._model

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame, return SkyCop-indexed detections."""
        model = self._ensure_loaded()
        results = model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            half=self.fp16 and "cuda" in self.device,
            verbose=False,
        )
        return self._extract_detections(results[0])

    def _extract_detections(self, result) -> list[Detection]:
        out: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return out
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids, strict=True):
            mapped = self._remap(int(cid))
            if mapped is None:
                continue
            out.append(Detection(
                class_idx=mapped,
                confidence=float(c),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            ))
        return out

    def _remap(self, native_cls: int) -> int | None:
        if self.class_remap is None:
            return native_cls
        return self.class_remap.get(native_cls)  # None if absent or explicitly mapped to None


def coco_to_skycop_map(cfg_map: dict) -> dict[int, int | None]:
    """Build a dict usable as `class_remap` from the YAML config.

    OmegaConf yields string keys for numeric yaml keys; normalise to int.
    """
    return {int(k): (None if v is None else int(v)) for k, v in cfg_map.items()}
