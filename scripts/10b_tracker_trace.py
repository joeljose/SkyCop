"""
Experiment 10b — tracker-trace diagnostic.

Pre-requisite for exp 11 (fingerprint). Replays each tracking holdout
through the tracker again, reuses ``evaluate_suspect_continuity`` to
identify ID-switch frame indices, and renders an overlay video so the
14 run_A switches are visually inspectable instead of opaque aggregate
counts.

No CARLA required — operates purely on the jpgs + ``tracks.json`` that
exp 10 already produced. Skips runs that have no ``tracks.json``
(e.g. ``run_B`` after the #25 dual-sensor teardown investigation wiped
its holdout).

Outputs:
    <holdout_dir>/<run_id>/trace.mp4
"""

import logging
import sys
from pathlib import Path

import cv2

from skycop.config import load
from skycop.cv.track import ByteTrackAdapter
from skycop.cv.tracker_viz import render_overlay, write_video
from skycop.cv.tracking_eval import (
    GTBox,
    TrackerBox,
    evaluate_suspect_continuity,
)
from skycop.logs import setup_logging

sys.stdout.reconfigure(line_buffering=True)

log = logging.getLogger("exp10b")

TRACE_FPS = 10


def _load_gt_frames(run_dir: Path) -> tuple[list[list[GTBox]], int]:
    import json
    with open(run_dir / "tracks.json") as f:
        data = json.load(f)
    suspect_aid = int(data["suspect_actor_id"])
    gt_frames: list[list[GTBox]] = []
    for frame in data["frames"]:
        boxes = [
            GTBox(
                actor_id=int(o["actor_id"]),
                x1=float(o["x1"]), y1=float(o["y1"]),
                x2=float(o["x2"]), y2=float(o["y2"]),
                visibility=float(o.get("visibility", 1.0)),
            )
            for o in frame["objects"]
        ]
        gt_frames.append(boxes)
    return gt_frames, suspect_aid


def _replay_and_track(run_dir: Path, adapter: ByteTrackAdapter) -> list[list[TrackerBox]]:
    img_dir = run_dir / "images"
    frame_paths = sorted(img_dir.glob("frame_*.jpg"))
    out: list[list[TrackerBox]] = []
    for p in frame_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            out.append([])
            continue
        tracks = adapter.update(frame)
        out.append([
            TrackerBox(
                track_id=t.track_id,
                x1=t.x1, y1=t.y1, x2=t.x2, y2=t.y2,
                confidence=t.confidence,
            )
            for t in tracks
        ])
    return out


def _iter_runs(root: Path):
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tracks_json = run_dir / "tracks.json"
        if not tracks_json.exists():
            log.warning("skipping %s — no tracks.json", run_dir.name)
            continue
        img_dir = run_dir / "images"
        if not img_dir.exists() or not any(img_dir.glob("frame_*.jpg")):
            log.warning("skipping %s — no images", run_dir.name)
            continue
        yield run_dir


def diagnose_run(run_dir: Path, adapter: ByteTrackAdapter, iou_threshold: float) -> dict:
    run_id = run_dir.name
    log.info("═══ diagnosing %s ═══", run_id)

    gt_frames, suspect_aid = _load_gt_frames(run_dir)
    tracker_frames = _replay_and_track(run_dir, adapter)
    assert len(tracker_frames) == len(gt_frames)

    result = evaluate_suspect_continuity(
        tracker_frames, gt_frames, suspect_aid,
        run_id=run_id, iou_threshold=iou_threshold,
    )
    log.info("[%s] continuity=%.4f id_switches=%d detected=%d/%d",
             run_id, result.continuity, result.id_switches,
             result.n_frames_suspect_detected, result.n_frames_suspect_present)

    locked = result.initial_lock_track_id
    switch_frames: list[int] = []
    prev_matched: int | None = None
    for pf in result.per_frame:
        matched = pf.get("matched_track_id")
        if (
            locked is not None
            and matched is not None
            and matched != locked
            and prev_matched != matched
        ):
            switch_frames.append(int(pf["frame"]))
        if matched is not None:
            prev_matched = matched

    log.info("[%s] distinct ID-switch transitions: %d at frames %s",
             run_id, len(switch_frames), switch_frames)

    img_dir = run_dir / "images"
    frame_paths = sorted(img_dir.glob("frame_*.jpg"))
    rendered: list = []
    for f, p in enumerate(frame_paths):
        image = cv2.imread(str(p))
        if image is None:
            continue
        pf = result.per_frame[f]
        matched = pf.get("matched_track_id")
        switch_event = None
        if locked is not None and matched is not None and matched != locked:
            switch_event = f"ID SWITCH — locked=t{locked} matched=t{matched}"
        rendered.append(render_overlay(
            image=image,
            gt_boxes=gt_frames[f],
            tracker_boxes=tracker_frames[f],
            suspect_actor_id=suspect_aid,
            locked_track_id=locked,
            frame_idx=f,
            switch_event=switch_event,
        ))

    trace_path = run_dir / "trace.mp4"
    write_video(rendered, fps=TRACE_FPS, out_path=trace_path)
    log.info("[%s] trace video → %s", run_id, trace_path)

    return {
        "run_id": run_id,
        "continuity": round(result.continuity, 4),
        "id_switches": result.id_switches,
        "initial_lock_track_id": result.initial_lock_track_id,
        "switch_transition_frames": switch_frames,
        "trace_path": str(trace_path),
    }


def main():
    setup_logging()
    cfg = load("default", "detector", "training_dataset", "training", "tracking")

    weights_path = Path(cfg.training.project) / cfg.training.name / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned weights missing at {weights_path}. Run exp 08 first."
        )
    log.info("using weights: %s", weights_path)

    holdout_root = Path(cfg.tracking.holdout_output_dir)
    if not holdout_root.exists():
        raise FileNotFoundError(
            f"Holdout root missing: {holdout_root}. Run exp 10 first."
        )

    per_run: list[dict] = []
    for run_dir in _iter_runs(holdout_root):
        adapter = ByteTrackAdapter(
            weights=str(weights_path),
            tracker_yaml=str(cfg.tracking.tracker_yaml),
            conf_threshold=float(cfg.detector.conf_threshold),
            iou_threshold=float(cfg.detector.iou_threshold),
            input_size=int(cfg.training.imgsz),
            device=str(cfg.training.device),
            fp16=bool(cfg.training.half),
        )
        per_run.append(
            diagnose_run(run_dir, adapter, float(cfg.tracking.match_iou_threshold))
        )

    if not per_run:
        log.warning("no runs with tracks.json + images found under %s", holdout_root)
        return

    log.info("═══ summary ═══")
    for r in per_run:
        log.info("%s: continuity=%.3f switches=%d transitions_at=%s → %s",
                 r["run_id"], r["continuity"], r["id_switches"],
                 r["switch_transition_frames"], r["trace_path"])


if __name__ == "__main__":
    main()
