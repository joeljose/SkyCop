"""
Experiment 10 — ByteTrack on fine-tuned YOLOv8s detections.

Measures **suspect track continuity** as the primary metric (per design
D-11's mission-vs-tracker-quality distinction). Scene-wide HOTA/MOTA
are secondary and optional.

Pipeline:

 1. Ensure tracking holdout exists at ``output/eval/tracking/``.
    Two 250-frame continuous captures (different seeds + weathers) with
    per-frame ground-truth ``{actor_id, bbox, visibility}``.
 2. For each holdout run:
    - Replay the saved RGB frames through YOLOv8s + ByteTrack.
    - Record per-frame tracker outputs.
    - Evaluate suspect-continuity via ``skycop.cv.tracking_eval``.
 3. Write ``output/eval/tracking_metrics.json`` summarising both runs.

Primary metric to watch:
    ``continuity`` ≥ 0.80 on at least one run. Below that = flag loudly
    in progress.md, don't claim success.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2

from skycop.config import load
from skycop.cv.capture import reset_world, run_tracking_capture
from skycop.cv.track import ByteTrackAdapter
from skycop.cv.tracking_eval import (
    GTBox,
    TrackerBox,
    evaluate_suspect_continuity,
)
from skycop.logs import setup_logging
from skycop.sim import connect

sys.stdout.reconfigure(line_buffering=True)

log = logging.getLogger("exp10")


def ensure_tracking_holdout(cfg) -> list[dict]:
    """Capture the two tracking holdout runs if not already on disk.

    Returns list of per-run metadata dicts (run_id, dir, suspect_actor_id).
    """
    root = Path(cfg.tracking.holdout_output_dir)
    root.mkdir(parents=True, exist_ok=True)
    captures: list[dict] = []

    # Connect once; reuse across both runs
    client = None

    for spec in cfg.tracking.captures:
        run_id = str(spec.run_id)
        run_dir = root / run_id
        tracks_path = run_dir / "tracks.json"

        if tracks_path.exists():
            with open(tracks_path) as f:
                data = json.load(f)
            captures.append({
                "run_id": run_id,
                "dir": run_dir,
                "suspect_actor_id": data["suspect_actor_id"],
                "n_frames": data["n_frames"],
            })
            log.info("[%s] cached — %d frames, suspect=%d",
                     run_id, data["n_frames"], data["suspect_actor_id"])
            continue

        if client is None:
            client = connect(host=cfg.carla.host, port=cfg.carla.port)

        cleared = reset_world(client, cfg.carla.tm_port)
        if cleared:
            log.info("reset: destroyed %d orphan actors/sensors", cleared)
        time.sleep(0.5)

        log.info("[%s] capturing %d frames seed=%d weather=%s",
                 run_id, spec.frames, spec.seed, spec.weather)
        result = run_tracking_capture(
            cfg,
            output_dir=run_dir,
            client=client,
            run_id=run_id,
            seed=int(spec.seed),
            weather=str(spec.weather),
            target_frames=int(spec.frames),
            min_suspect_visibility=float(cfg.tracking.min_suspect_visibility),
            min_visibility_other=float(cfg.tracking.min_visibility_other),
            min_bbox_pixel=int(cfg.tracking.min_bbox_pixel),
            jpeg_quality=int(cfg.tracking.jpeg_quality),
        )
        captures.append({
            "run_id": run_id,
            "dir": run_dir,
            "suspect_actor_id": result.suspect_actor_id,
            "n_frames": result.frames_saved,
        })

    return captures


def replay_and_track(run_dir: Path, adapter: ByteTrackAdapter) -> list[list[TrackerBox]]:
    """Replay the saved RGB frames through the tracker in order.

    Returns one list of TrackerBox per frame (matching the saved frame count).
    """
    img_dir = run_dir / "images"
    frame_paths = sorted(img_dir.glob("frame_*.jpg"))
    log.info("replaying %d frames from %s", len(frame_paths), run_dir)

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


def load_gt_frames(run_dir: Path) -> tuple[list[list[GTBox]], int]:
    """Read tracks.json; return per-frame GT list and the suspect actor id."""
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


def save_metrics_atomic(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def main():
    setup_logging()
    cfg = load("default", "training_dataset", "training", "tracking")

    # 1. Ensure holdout exists
    holdout = ensure_tracking_holdout(cfg)

    # 2. Build the tracker adapter (one instance per run so state resets)
    weights_path = Path(cfg.training.project) / cfg.training.name / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned weights missing at {weights_path}. Run exp 08 first."
        )
    log.info("using weights: %s", weights_path)

    per_run_results: list[dict] = []
    for spec in holdout:
        run_id = spec["run_id"]
        run_dir = spec["dir"]
        log.info("═══ evaluating %s ═══", run_id)

        # Fresh adapter per run so track IDs don't bleed between sequences
        adapter = ByteTrackAdapter(
            weights=str(weights_path),
            tracker_yaml=str(cfg.tracking.tracker_yaml),
            conf_threshold=float(cfg.detector.conf_threshold),
            iou_threshold=float(cfg.detector.iou_threshold),
            input_size=int(cfg.training.imgsz),
            device=str(cfg.training.device),
            fp16=bool(cfg.training.half),
        )

        gt_frames, suspect_aid = load_gt_frames(run_dir)
        tracker_frames = replay_and_track(run_dir, adapter)
        assert len(tracker_frames) == len(gt_frames), \
            f"{run_id}: tracker={len(tracker_frames)} gt={len(gt_frames)}"

        result = evaluate_suspect_continuity(
            tracker_frames, gt_frames, suspect_aid,
            run_id=run_id,
            iou_threshold=float(cfg.tracking.match_iou_threshold),
        )
        log.info("[%s] continuity=%.4f  id_switches=%d  false_lock_frames=%d",
                 run_id, result.continuity, result.id_switches, result.false_lock_frames)
        log.info("[%s] frames: %d total, %d suspect-present, %d suspect-detected",
                 run_id, result.n_frames,
                 result.n_frames_suspect_present, result.n_frames_suspect_detected)
        if result.reacquisition_frames:
            log.info("[%s] reacquisition gaps (frames): %s",
                     run_id, result.reacquisition_frames)

        per_run_results.append(result.summary())

    # 3. Summary + thresholds
    continuity_values = [r["continuity"] for r in per_run_results]
    max_cont = max(continuity_values) if continuity_values else 0.0
    mean_cont = sum(continuity_values) / len(continuity_values) if continuity_values else 0.0

    if max_cont >= 0.80:
        verdict = "GOOD: suspect continuity ≥ 0.80 on at least one run"
    elif max_cont >= 0.50:
        verdict = "PARTIAL: flag in progress.md; tune tracker thresholds"
    else:
        verdict = "POOR: tracker losing suspect too often; consider BoT-SORT or tuning"

    payload = {
        "weights": str(weights_path),
        "tracker": str(cfg.tracking.tracker_yaml),
        "match_iou_threshold": float(cfg.tracking.match_iou_threshold),
        "per_run": per_run_results,
        "continuity_max": max_cont,
        "continuity_mean": mean_cont,
        "verdict": verdict,
        "notes": [
            "Primary metric is suspect track continuity per design D-11.",
            "Scene-wide HOTA/MOTA not computed — mission cares about one track.",
            "Two 250-frame runs across different seeds + weathers for hard-moment diversity.",
        ],
    }
    save_metrics_atomic(payload, Path(cfg.tracking.metrics_out))
    log.info("metrics → %s", cfg.tracking.metrics_out)
    log.info("verdict: %s (max continuity %.3f)", verdict, max_cont)


if __name__ == "__main__":
    main()
