"""Overlay rendering + video writer for tracker diagnostics.

Pure-offline helpers — no CARLA, no YOLO. Consume on-disk jpgs + the
ground-truth + tracker-output lists that ``skycop.cv.tracking_eval``
already shapes, and emit frames / videos that make ID switches visible.

The point is inspection, not a product surface — kept narrow on
purpose so the caller owns the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from skycop.cv.tracking_eval import GTBox, TrackerBox

log = logging.getLogger(__name__)


_GT_OTHER_PALETTE = [
    (200, 160, 100),   # muted cyan-blue
    (140, 200, 140),   # muted green
    (180, 140, 180),   # muted purple
    (160, 180, 200),   # muted steel
    (170, 170, 120),   # muted olive
    (120, 170, 190),   # muted teal
]

_TRACKER_PALETTE = [
    (40, 140, 240),    # orange
    (40, 200, 255),    # yellow-orange
    (80, 80, 240),     # magenta-red
    (200, 100, 240),   # pink-magenta
    (60, 220, 200),    # amber-lime
    (180, 60, 220),    # violet
    (20, 200, 140),    # green-amber
]

_SUSPECT_GT_COLOR = (0, 0, 220)      # red (BGR)
_LOCKED_TRACK_COLOR = (0, 255, 255)  # yellow (BGR)
_SWITCH_BANNER_COLOR = (0, 255, 255)


def _stable_color(key: int, palette: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    return palette[key % len(palette)]


def render_overlay(
    image: np.ndarray,
    gt_boxes: list[GTBox],
    tracker_boxes: list[TrackerBox],
    suspect_actor_id: int,
    locked_track_id: int | None,
    frame_idx: int,
    switch_event: str | None = None,
) -> np.ndarray:
    """Draw GT + tracker boxes on a copy of ``image``.

    - GT suspect bbox is RED, thick (4 px).
    - Other GT bboxes get a stable palette colour keyed on ``actor_id``,
      thin (1 px).
    - Tracker bboxes get a stable palette colour keyed on ``track_id``,
      medium (2 px). If a tracker box's ``track_id`` equals
      ``locked_track_id``, overlay a yellow 4-px border to mark "the
      suspect track" — regardless of where it actually landed.
    - ``switch_event`` renders a yellow banner across the top of frame.
    - ``frame_idx`` is drawn bottom-left.
    """
    out = image.copy()
    h, w = out.shape[:2]

    for gt in gt_boxes:
        is_suspect = gt.actor_id == suspect_actor_id
        color = _SUSPECT_GT_COLOR if is_suspect else _stable_color(gt.actor_id, _GT_OTHER_PALETTE)
        thickness = 4 if is_suspect else 1
        p1 = (int(gt.x1), int(gt.y1))
        p2 = (int(gt.x2), int(gt.y2))
        cv2.rectangle(out, p1, p2, color, thickness)
        if is_suspect:
            cv2.putText(
                out, f"GT suspect a{gt.actor_id}", (p1[0], max(0, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

    for t in tracker_boxes:
        tid = t.track_id
        if tid is None:
            color = (180, 180, 180)
        else:
            color = _stable_color(tid, _TRACKER_PALETTE)
        p1 = (int(t.x1), int(t.y1))
        p2 = (int(t.x2), int(t.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        if tid is not None and locked_track_id is not None and tid == locked_track_id:
            cv2.rectangle(out, (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3),
                          _LOCKED_TRACK_COLOR, 3)
        label = f"t{tid}" if tid is not None else "t?"
        cv2.putText(
            out, label, (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    cv2.putText(
        out, f"frame {frame_idx}", (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if locked_track_id is not None:
        cv2.putText(
            out, f"locked=t{locked_track_id}", (8, h - 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _LOCKED_TRACK_COLOR, 2, cv2.LINE_AA,
        )

    if switch_event is not None:
        banner_h = 36
        cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 0), -1)
        cv2.putText(
            out, switch_event, (10, banner_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, _SWITCH_BANNER_COLOR, 2, cv2.LINE_AA,
        )
    return out


def write_video(frames: list[np.ndarray], fps: int, out_path: Path) -> Path:
    """Encode ``frames`` to an mp4 at ``out_path``. Returns the path.

    All frames must share shape; caller's responsibility.
    """
    if not frames:
        raise ValueError("write_video: frames is empty")
    h, w = frames[0].shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"write_video: failed to open VideoWriter for {out_path}")
    try:
        for f in frames:
            if f.shape[:2] != (h, w):
                raise ValueError(
                    f"write_video: frame shape mismatch — got {f.shape[:2]}, expected {(h, w)}"
                )
            writer.write(f)
    finally:
        writer.release()

    log.info("wrote %d frames to %s (%dx%d @ %d fps)", len(frames), out_path, w, h, fps)
    return out_path
