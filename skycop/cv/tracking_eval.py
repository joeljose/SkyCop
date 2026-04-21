"""Tracker evaluation — suspect-continuity as the mission-relevant primary metric.

Mission framing (grill Q1): once the tracker locks onto the suspect, success
is *maintaining that one track's identity* through the pursuit. Scene-wide
HOTA/MOTA average over 50 NPCs we don't care about and dilute the signal
on the one object that matters.

Primary outputs from ``evaluate_suspect_continuity``:

- ``continuity`` — fraction of detection-present frames post-initial-lock where
  the track_id matches the initial-lock's id. 1.0 = perfect.
- ``id_switches`` — number of times the tracker reassigned the suspect to a
  new id after the initial lock.
- ``reacquisition_frames`` — list of gap lengths after each loss, in frames.
- ``false_lock_frames`` — number of frames where the tracker's "suspect
  track" bbox actually matched a non-suspect actor.
- Plus per-frame diagnostics for debugging.

IoU-Hungarian association per frame maps tracker outputs → GT bboxes by
maximising IoU. The suspect's GT bbox is identified by its known CARLA
``actor_id`` (captured at spawn).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── Data types ──────────────────────────────────────────────────────────

@dataclass
class GTBox:
    """One ground-truth box at one frame."""
    actor_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    visibility: float = 1.0


@dataclass
class TrackerBox:
    """One tracker output box at one frame. track_id may be None (unconfirmed)."""
    track_id: int | None
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0


@dataclass
class ContinuityResult:
    """Summary of suspect-continuity evaluation for one holdout run."""
    run_id: str
    suspect_actor_id: int
    n_frames: int
    n_frames_suspect_present: int
    n_frames_suspect_detected: int  # present AND tracker returned a match
    continuity: float               # detected-frames with consistent id / n_frames_suspect_detected
    initial_lock_frame: int | None
    initial_lock_track_id: int | None
    id_switches: int
    reacquisition_frames: list[int] = field(default_factory=list)
    false_lock_frames: int = 0
    per_frame: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "suspect_actor_id": self.suspect_actor_id,
            "n_frames": self.n_frames,
            "n_frames_suspect_present": self.n_frames_suspect_present,
            "n_frames_suspect_detected": self.n_frames_suspect_detected,
            "continuity": round(self.continuity, 4),
            "initial_lock_frame": self.initial_lock_frame,
            "initial_lock_track_id": self.initial_lock_track_id,
            "id_switches": self.id_switches,
            "reacquisition_frames": self.reacquisition_frames,
            "false_lock_frames": self.false_lock_frames,
        }


# ── IoU + Hungarian matching ────────────────────────────────────────────

def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def _hungarian_match(
    tracker_boxes: list[TrackerBox],
    gt_boxes: list[GTBox],
    iou_threshold: float = 0.3,
) -> dict[int, int]:
    """Return a map {tracker_idx: gt_idx} maximising summed IoU, filtered by
    a minimum IoU. Unmatched on either side simply absent.
    """
    if not tracker_boxes or not gt_boxes:
        return {}
    cost = np.zeros((len(tracker_boxes), len(gt_boxes)))
    for i, t in enumerate(tracker_boxes):
        for j, g in enumerate(gt_boxes):
            cost[i, j] = -_iou((t.x1, t.y1, t.x2, t.y2), (g.x1, g.y1, g.x2, g.y2))

    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost)
    out: dict[int, int] = {}
    for r, c in zip(rows, cols, strict=True):
        iou = -cost[r, c]
        if iou >= iou_threshold:
            out[int(r)] = int(c)
    return out


# ── Primary metric ──────────────────────────────────────────────────────

def evaluate_suspect_continuity(
    tracker_frames: list[list[TrackerBox]],
    gt_frames: list[list[GTBox]],
    suspect_actor_id: int,
    run_id: str = "run",
    iou_threshold: float = 0.3,
) -> ContinuityResult:
    """Evaluate suspect-continuity over a sequence of matched frames.

    Parameters
    ----------
    tracker_frames
        One list of TrackerBox per frame (the tracker's outputs, in frame order).
    gt_frames
        Same-length list of ground-truth boxes per frame.
    suspect_actor_id
        CARLA actor id of the suspect, known from capture time.

    Algorithm
    ---------
    - Identify the suspect's GT bbox per frame (may be absent if suspect out
      of frame; visibility is tracked in GTBox but we don't filter on it —
      eval counts only detection-present frames anyway).
    - IoU-Hungarian match tracker outputs ↔ GT bboxes each frame.
    - Read off which tracker track_id was matched to the suspect that frame.
    - Find the initial-lock frame: first frame where the matched tracker
      track_id is non-None.
    - For each subsequent suspect-detected frame, compare the current matched
      track_id to the locked id. Mismatch = id switch; accumulate.
    - Continuity = matched-and-consistent frames / matched frames, post-lock.
    """
    assert len(tracker_frames) == len(gt_frames), "frame-count mismatch"

    n_frames = len(gt_frames)
    per_frame: list[dict] = []
    n_suspect_present = 0
    n_suspect_detected = 0
    consistent_detections = 0
    id_switches = 0
    false_lock_frames = 0
    initial_lock_frame: int | None = None
    initial_lock_track_id: int | None = None
    reacquisition_frames: list[int] = []
    last_missed_run_start: int | None = None

    for f, (t_boxes, g_boxes) in enumerate(zip(tracker_frames, gt_frames, strict=True)):
        # Which GT box (if any) is the suspect this frame?
        suspect_gt_idx = next(
            (i for i, g in enumerate(g_boxes) if g.actor_id == suspect_actor_id),
            None,
        )
        suspect_present = suspect_gt_idx is not None
        if suspect_present:
            n_suspect_present += 1

        matches = _hungarian_match(t_boxes, g_boxes, iou_threshold=iou_threshold)
        # Reverse: gt_idx → tracker_idx
        gt_to_tracker = {g: t for t, g in matches.items()}

        matched_track_id: int | None = None
        if suspect_gt_idx is not None and suspect_gt_idx in gt_to_tracker:
            t_idx = gt_to_tracker[suspect_gt_idx]
            matched_track_id = t_boxes[t_idx].track_id
            n_suspect_detected += 1

            if initial_lock_frame is None and matched_track_id is not None:
                initial_lock_frame = f
                initial_lock_track_id = matched_track_id

            if initial_lock_track_id is not None and matched_track_id is not None:
                if matched_track_id == initial_lock_track_id:
                    consistent_detections += 1
                    if last_missed_run_start is not None:
                        reacquisition_frames.append(f - last_missed_run_start)
                        last_missed_run_start = None
                else:
                    id_switches += 1
        else:
            if initial_lock_frame is not None and last_missed_run_start is None:
                last_missed_run_start = f

        # Also check whether any tracker output with initial_lock_track_id
        # landed on a non-suspect GT (false lock)
        if initial_lock_track_id is not None:
            for t_idx, t in enumerate(t_boxes):
                if t.track_id == initial_lock_track_id and t_idx in matches:
                    matched_gt = matches[t_idx]
                    if g_boxes[matched_gt].actor_id != suspect_actor_id:
                        false_lock_frames += 1

        per_frame.append({
            "frame": f,
            "suspect_present": suspect_present,
            "matched_track_id": matched_track_id,
        })

    continuity = (
        consistent_detections / n_suspect_detected
        if n_suspect_detected > 0 else 0.0
    )

    return ContinuityResult(
        run_id=run_id,
        suspect_actor_id=suspect_actor_id,
        n_frames=n_frames,
        n_frames_suspect_present=n_suspect_present,
        n_frames_suspect_detected=n_suspect_detected,
        continuity=continuity,
        initial_lock_frame=initial_lock_frame,
        initial_lock_track_id=initial_lock_track_id,
        id_switches=id_switches,
        reacquisition_frames=reacquisition_frames,
        false_lock_frames=false_lock_frames,
        per_frame=per_frame,
    )
