"""Dataset primitives: instance-seg → YOLO bbox extraction, label writer, manifest.

CARLA 0.9.16 instance-segmentation camera encodes each pixel as:
  R = semantic label (CityObjectLabel — 14 Car, 15 Truck, 16 Bus, 18 Motorcycle, 19 Bicycle, ...)
  G = actor ID low byte
  B = actor ID high byte

`actor_id = (B << 8) | G`, with ID 0 meaning "not an actor".

Verified empirically against a live 0.9.16 server: a spawned vehicle with
`actor.id = 238` produced pixels with (R=14, G=238, B=0). Keep this byte
order — upstream docs show conflicting conventions across versions and it
must be anchored to the running server, not inferred.

This module exposes pure-numpy helpers that take a BGR array and a
callable `actor_id_to_class(id) -> int | None` (normally backed by a
live CARLA world + blueprint classifier). The callable is injected so
the extraction logic can be unit-tested without a CARLA server.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np


class BBox(NamedTuple):
    """Normalized YOLO box — all fields in [0, 1]."""
    class_idx: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class ActorDetection:
    """Per-actor bbox in pixel coordinates, with visibility metadata.

    Separate from ``BBox`` because tracking evaluation needs the raw actor
    id + visibility info that the YOLO-normalized form strips out.
    """
    actor_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    visibility: float  # pixel_count / bbox_area ∈ (0, 1]
    pixel_count: int


def extract_actor_boxes_from_seg(seg_bgr: np.ndarray) -> list[ActorDetection]:
    """Return per-actor bboxes from an instance-seg frame. No filtering.

    The caller applies whatever min-pixel / min-visibility / class filters
    their use case needs. Tracking evaluation wants raw output (suspect
    always emitted even at low visibility); dataset collection wants
    stricter filtering.

    Encoding per ``extract_yolo_labels_from_seg``'s docstring: R=label,
    G=actor_id_low, B=actor_id_high; ``actor_id = (B << 8) | G``.
    """
    b = seg_bgr[:, :, 0].astype(np.uint32)
    g = seg_bgr[:, :, 1].astype(np.uint32)
    actor_id = (b << 8) | g

    out: list[ActorDetection] = []
    unique_ids = np.unique(actor_id)
    for aid in unique_ids:
        if aid == 0:
            continue
        mask = actor_id == aid
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        visibility = xs.size / float(bw * bh)
        out.append(ActorDetection(
            actor_id=int(aid),
            x1=x_min, y1=y_min, x2=x_max, y2=y_max,
            visibility=visibility,
            pixel_count=int(xs.size),
        ))
    return out


@dataclass
class FrameStats:
    """Per-frame accounting of what was kept vs dropped, for diagnostics."""
    emitted: int = 0
    dropped_too_small: int = 0
    dropped_occluded: int = 0
    dropped_unknown_class: int = 0


def extract_yolo_labels_from_seg(
    seg_bgr: np.ndarray,
    actor_id_to_class: Callable[[int], int | None],
    min_pixel: int = 20,
    min_visibility: float = 0.3,
) -> tuple[list[BBox], FrameStats]:
    """Extract YOLO-format bboxes from a CARLA instance-seg frame.

    Parameters
    ----------
    seg_bgr
        BGR numpy array (H, W, 3), uint8, as returned by ``carla_image_to_bgr``.
    actor_id_to_class
        Callable mapping a CARLA actor id to a YOLO class index, or
        None to drop the actor from the dataset.
    min_pixel
        Minimum bbox side length in pixels. Smaller boxes are dropped —
        protects against labels the detector cannot learn from.
    min_visibility
        Ratio of actor pixels to bbox area required for the label to be
        emitted. Filters out mostly-occluded vehicles whose bbox would
        enclose the occluder.
    """
    h, w = seg_bgr.shape[:2]
    b = seg_bgr[:, :, 0].astype(np.uint32)
    g = seg_bgr[:, :, 1].astype(np.uint32)
    actor_id = (b << 8) | g

    stats = FrameStats()
    boxes: list[BBox] = []

    unique_ids = np.unique(actor_id)
    for aid in unique_ids:
        if aid == 0:
            continue

        cls_idx = actor_id_to_class(int(aid))
        if cls_idx is None:
            stats.dropped_unknown_class += 1
            continue

        mask = actor_id == aid
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1

        if bw < min_pixel or bh < min_pixel:
            stats.dropped_too_small += 1
            continue

        visibility = xs.size / float(bw * bh)
        if visibility < min_visibility:
            stats.dropped_occluded += 1
            continue

        cx = (x_min + x_max) / 2.0 / w
        cy = (y_min + y_max) / 2.0 / h
        nw = bw / w
        nh = bh / h
        boxes.append(BBox(cls_idx, cx, cy, nw, nh))
        stats.emitted += 1

    return boxes, stats


def write_yolo_label(path: Path, boxes: list[BBox]) -> None:
    """Write YOLO-format labels to disk — one line per box."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for box in boxes:
            f.write(f"{box.class_idx} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}\n")


@dataclass
class DatasetManifest:
    """Per-run metadata for reproducibility and downstream analysis."""
    seed: int
    fixed_delta_seconds: float
    map_name: str
    class_names: list[str]
    min_pixel: int
    min_visibility: float
    frames: list[dict] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    skip_counts: dict[str, int] = field(default_factory=lambda: {
        "dropped_too_small": 0,
        "dropped_occluded": 0,
        "dropped_unknown_class": 0,
    })

    def record_frame(
        self,
        index: int,
        tick: int,
        camera_pose: dict,
        suspect_pose: dict,
        boxes: list[BBox],
        stats: FrameStats,
    ) -> None:
        self.frames.append({
            "index": index,
            "tick": tick,
            "camera": camera_pose,
            "suspect": suspect_pose,
            "emitted": stats.emitted,
            "dropped_too_small": stats.dropped_too_small,
            "dropped_occluded": stats.dropped_occluded,
        })
        for cls_idx in {b.class_idx for b in boxes}:
            cls_name = self.class_names[cls_idx]
            self.class_counts[cls_name] = self.class_counts.get(cls_name, 0) + sum(
                1 for b in boxes if b.class_idx == cls_idx
            )
        self.skip_counts["dropped_too_small"] += stats.dropped_too_small
        self.skip_counts["dropped_occluded"] += stats.dropped_occluded
        self.skip_counts["dropped_unknown_class"] += stats.dropped_unknown_class

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
