"""ByteTrack adapter wrapping Ultralytics' ``model.track()``.

Design goal: hide the Ultralytics result shape behind a stable ``Track``
dataclass so downstream code (exp 11 fingerprint, mission tracer) doesn't
couple to the tracker's internal API.

ByteTrack picked over BoT-SORT per grill Q2 — CMC is paying for a problem
our scene doesn't have (drone centres camera on the suspect), and
ByteTrack's two-pass low-confidence association is a net positive for
recovering faded detections on a briefly-occluded suspect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Track:
    """One tracked object at one frame. track_id is None on unconfirmed tracks."""
    track_id: int | None
    class_idx: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class ByteTrackAdapter:
    """Stateful tracker over a YOLOv8 detector using Ultralytics' ByteTrack.

    Call ``update(frame_bgr)`` once per frame in temporal order. The adapter
    keeps tracker state across calls via ``persist=True``. Resetting between
    independent sequences requires a new adapter instance (Ultralytics' API
    doesn't expose a clean reset).
    """

    def __init__(
        self,
        weights: str,
        tracker_yaml: str = "bytetrack.yaml",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cuda:0",
        fp16: bool = True,
    ) -> None:
        self.weights = weights
        self.tracker_yaml = tracker_yaml
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.device = device
        self.fp16 = fp16
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def update(self, frame_bgr: np.ndarray) -> list[Track]:
        """Feed one frame; return the current per-frame tracks with IDs."""
        model = self._ensure_loaded()
        results = model.track(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            half=self.fp16 and "cuda" in self.device,
            tracker=self.tracker_yaml,
            persist=True,
            verbose=False,
        )
        return self._extract_tracks(results[0])

    @staticmethod
    def _extract_tracks(result) -> list[Track]:
        out: list[Track] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        # Ultralytics sets boxes.id to None on frames with no confirmed tracks
        # and to a tensor otherwise. Unconfirmed detections within a frame
        # that does have some confirmed tracks end up with NaN in .id.
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None

        for i, ((x1, y1, x2, y2), c, cid) in enumerate(
            zip(xyxy, confs, cls_ids, strict=True)
        ):
            tid: int | None = None
            if ids is not None and i < len(ids):
                raw = ids[i]
                if not np.isnan(raw):
                    tid = int(raw)
            out.append(Track(
                track_id=tid,
                class_idx=int(cid),
                confidence=float(c),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            ))
        return out


def ensure_tracker_yaml(path: Path = Path("bytetrack.yaml")) -> Path:
    """Resolve the ByteTrack config path Ultralytics will load.

    ``bytetrack.yaml`` is shipped with ultralytics; passing that string
    tells it to look up its own bundled config. Override only if we want
    custom tracker params (not needed for v1).
    """
    return path
