"""
Experiment 06 — pretrained YOLOv8s baseline (no training).

Two modes run back-to-back:

1. Offline eval on the exp 05 holdout → mAP@0.5, mAP@0.5:0.95, per-class AP
   for COCO-reachable classes (car / truck / bus). Van is excluded from
   pretrained scoring because COCO has no van class — the omission is
   recorded explicitly in the metrics JSON, not silently.

2. Live pursuit on the exp 04 scene → detection overlay on the MJPEG stream,
   sustained-FPS and peak-VRAM measurement. This is the acceptance bar for
   NFR-01 / NFR-03 at this pipeline depth (detection only; tracking and
   fingerprint still to land).

Both modes use the pretrained yolov8s.pt COCO checkpoint with the SkyCop
class remap defined in configs/detector.yaml.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2

from skycop.config import load
from skycop.cv import (
    CLASS_NAMES,
    EvalBox,
    YoloDetector,
    coco_to_skycop_map,
    read_yolo_labels,
    score_predictions,
)
from skycop.cv.inloop import measure_live_pursuit
from skycop.dashboard import MJPEGServer
from skycop.logs import setup_logging

# Unbuffer stdout so progress prints survive CARLA's SIGABRT on process exit
# (now largely mitigated by teardown_pursuit, but belt-and-braces).
sys.stdout.reconfigure(line_buffering=True)

log = logging.getLogger("exp06")


def build_detector(cfg) -> YoloDetector:
    class_remap = coco_to_skycop_map(dict(cfg.detector.coco_to_skycop))
    return YoloDetector(
        weights=cfg.detector.weights,
        class_remap=class_remap,
        conf_threshold=cfg.detector.conf_threshold,
        iou_threshold=cfg.detector.iou_threshold,
        input_size=cfg.detector.input_size,
        device=cfg.detector.device,
        fp16=cfg.detector.fp16,
    )


# ── Mode 1: offline eval on holdout ────────────────────────────────────────

def run_offline_eval(cfg, detector: YoloDetector) -> dict:
    eval_dir = Path(cfg.baseline.offline_eval_dir)
    img_dir = eval_dir / "images"
    lbl_dir = eval_dir / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(
            f"Holdout not found at {eval_dir}. Run `make exp N=05` first."
        )

    class_filter = {v for v in dict(cfg.detector.coco_to_skycop).values() if v is not None}
    frame_count = len(list(img_dir.glob("*.jpg")))
    log.info("offline eval on %d frames, class_filter=%s", frame_count, sorted(class_filter))

    preds_per_frame: list[list[EvalBox]] = []
    gts_per_frame: list[list[EvalBox]] = []

    for img_path in sorted(img_dir.glob("*.jpg")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        detections = detector.predict(frame)
        preds_per_frame.append([
            EvalBox(
                class_idx=d.class_idx,
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                confidence=d.confidence,
            )
            for d in detections
        ])
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gts_per_frame.append(read_yolo_labels(lbl_path, w, h))

    metrics = score_predictions(preds_per_frame, gts_per_frame, class_filter=class_filter)
    log.info("mAP@0.5      = %.4f", metrics.map_50)
    log.info("mAP@0.5:0.95 = %.4f", metrics.map_50_95)
    log.info(
        "per-class AP@0.5: %s",
        {CLASS_NAMES[k]: round(v, 4) for k, v in metrics.per_class_ap_50.items()},
    )

    return {
        "mAP_50": metrics.map_50,
        "mAP_50_95": metrics.map_50_95,
        "per_class_ap_50": {
            CLASS_NAMES[k]: round(v, 4) for k, v in metrics.per_class_ap_50.items()
        },
        "n_frames": metrics.n_frames,
        "n_predictions": metrics.n_predictions,
        "n_ground_truths": metrics.n_ground_truths,
        "class_filter_skycop": metrics.class_filter,
        "class_filter_names": [CLASS_NAMES[c] for c in metrics.class_filter]
                              if metrics.class_filter else None,
        "van_excluded_reason": "COCO has no 'van' class; pretrained cannot emit",
    }


def save_metrics_atomic(metrics: dict, out_path: Path) -> None:
    """Write metrics JSON atomically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


def main():
    setup_logging()
    cfg = load("default", "detector")
    detector = build_detector(cfg)

    log.info("loading detector weights: %s", cfg.detector.weights)
    detector._ensure_loaded()

    log.info("=== Mode 1/2: offline eval on exp 05 holdout ===")
    offline = run_offline_eval(cfg, detector)

    log.info("=== Mode 2/2: live pursuit ===")
    server = MJPEGServer(
        title="SkyCop — Experiment 06",
        hud="Experiment 06 · YOLOv8s COCO-pretrained baseline",
    )
    server.start(port=5000)
    time.sleep(0.3)

    live = measure_live_pursuit(
        cfg,
        detector=detector,
        server=server,
        duration_s=float(cfg.baseline.live_pursuit_seconds),
        fps_threshold_warn=float(cfg.baseline.fps_threshold_warn),
        vram_threshold_gb=float(cfg.baseline.vram_threshold_gb),
    )

    metrics = {
        "model": cfg.detector.weights,
        "pretrained_offline_eval": offline,
        "pretrained_live_pursuit": {
            **live.as_dict(),
            "fps_threshold_warn": float(cfg.baseline.fps_threshold_warn),
            "vram_threshold_gb": float(cfg.baseline.vram_threshold_gb),
        },
    }
    save_metrics_atomic(metrics, Path(cfg.baseline.metrics_out))
    log.info("metrics → %s", cfg.baseline.metrics_out)


if __name__ == "__main__":
    main()
