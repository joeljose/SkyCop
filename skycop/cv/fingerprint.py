"""HSV-histogram fingerprint for re-identifying the suspect across tracker rebinds.

Scope for Mission v0 is deliberately narrow: one attribute (colour), one
distance (histogram intersection). Additional attributes (roof shape,
apparent size, speed/heading) land when the video shows colour alone
isn't enough — we don't speculate.

Pure functions over numpy arrays; no CARLA, no YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Fingerprint:
    """Normalised HSV histogram (sum = 1) flattened across H, S, V bins."""
    hsv_hist: np.ndarray

    def is_valid(self) -> bool:
        return self.hsv_hist.size > 0 and float(self.hsv_hist.sum()) > 0.0


def _clamp_bbox(
    bbox: tuple[float, float, float, float], w: int, h: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    # Reject bboxes that don't intersect the frame at all.
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return None
    x1i = max(0, int(x1))
    y1i = max(0, int(y1))
    x2i = min(w, int(x2))
    y2i = min(h, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return None
    return (x1i, y1i, x2i, y2i)


def extract(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    bins: int = 8,
) -> Fingerprint:
    """HSV histogram over ``bbox`` crop of ``frame_bgr``, normalised to sum=1.

    Returns a Fingerprint with an all-zero histogram for empty/off-frame
    bboxes — callers should guard with ``is_valid()``.
    """
    h, w = frame_bgr.shape[:2]
    clamped = _clamp_bbox(bbox, w, h)
    empty = np.zeros(bins ** 3, dtype=np.float32)
    if clamped is None:
        return Fingerprint(empty)
    x1, y1, x2, y2 = clamped
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return Fingerprint(empty)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.flatten().astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        return Fingerprint(empty)
    hist /= total
    return Fingerprint(hist)


def score(a: Fingerprint, b: Fingerprint) -> float:
    """Histogram intersection — sum of min(a_i, b_i). Bounded 0..1 for
    sum-normalised histograms. Symmetric; ``score(a, a) == 1.0`` iff ``a`` is valid."""
    if not a.is_valid() or not b.is_valid():
        return 0.0
    if a.hsv_hist.shape != b.hsv_hist.shape:
        raise ValueError(
            f"Fingerprint shape mismatch: {a.hsv_hist.shape} vs {b.hsv_hist.shape}"
        )
    return float(np.minimum(a.hsv_hist, b.hsv_hist).sum())
