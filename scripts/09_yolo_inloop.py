"""
Experiment 09 — fine-tuned YOLOv8s in-loop inference.

Drops the exp 08 fine-tuned weights into the live pursuit pipeline. First
end-to-end verification of the detection milestone: all prior measurements
were offline mAP on static frames. This tells us whether the detector
actually works during a moving drone + moving suspect + moving NPCs.

Reports sustained FPS, detection latency, peak VRAM. Compares against the
exp 06 pretrained baseline's live-pursuit block by reading the baseline
metrics JSON if present.

Open http://localhost:5000 during the run to watch boxes overlaid on the
MJPEG stream — the visual check catches failure modes static-frame mAP
misses (flicker, temporal drop-outs, edge-of-frame misses on turns).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from skycop.config import load
from skycop.cv.inference import YoloDetector
from skycop.cv.inloop import measure_live_pursuit
from skycop.dashboard import MJPEGServer
from skycop.logs import setup_logging

sys.stdout.reconfigure(line_buffering=True)

log = logging.getLogger("exp09")


def save_metrics_atomic(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def load_baseline_live(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("pretrained_live_pursuit", {})
    except Exception:
        return {}


def main():
    setup_logging()
    cfg = load("default", "altitude", "detector", "training")

    weights_path = Path(cfg.training.project) / cfg.training.name / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned weights missing at {weights_path}. Run `make exp N=08` first."
        )

    log.info("loading fine-tuned weights: %s", weights_path)
    detector = YoloDetector(
        weights=str(weights_path),
        class_remap=None,                        # fine-tuned model is already single-class
        conf_threshold=float(cfg.detector.conf_threshold),
        iou_threshold=float(cfg.detector.iou_threshold),
        input_size=int(cfg.training.imgsz),
        device=str(cfg.training.device),
        fp16=bool(cfg.training.half),
    )
    detector._ensure_loaded()

    server = MJPEGServer(
        title="SkyCop — Experiment 09",
        hud="Experiment 09 · YOLOv8s fine-tuned in-loop",
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

    baseline_live = load_baseline_live(Path(cfg.baseline.metrics_out))

    fps_delta = (
        live.sustained_fps - baseline_live["sustained_fps"]
        if baseline_live.get("sustained_fps") is not None else None
    )
    vram_delta = (
        live.peak_vram_gb - baseline_live["peak_vram_gb"]
        if baseline_live.get("peak_vram_gb") is not None else None
    )
    detection_delta_ms = (
        live.detection_mean_ms - baseline_live["detection_mean_ms"]
        if baseline_live.get("detection_mean_ms") is not None else None
    )

    payload = {
        "model": str(weights_path),
        "fine_tuned_live": {
            **live.as_dict(),
            "fps_threshold_warn": float(cfg.baseline.fps_threshold_warn),
            "vram_threshold_gb": float(cfg.baseline.vram_threshold_gb),
        },
        "pretrained_baseline_live": baseline_live or None,
        "deltas": {
            "fps_delta": fps_delta,
            "vram_delta_gb": vram_delta,
            "detection_mean_ms_delta": detection_delta_ms,
        },
        "notes": [
            "Fine-tuned model is same yolov8s architecture as baseline — "
            "FPS/VRAM should be near-identical; any meaningful delta is suspect.",
            "Visual sanity check: watch http://localhost:5000 during the run.",
        ],
    }

    out_path = Path("/app/output/eval/inloop_metrics.json")
    save_metrics_atomic(payload, out_path)
    log.info("metrics → %s", out_path)

    if fps_delta is not None:
        log.info("FPS delta vs baseline: %+.2f", fps_delta)
    if vram_delta is not None:
        log.info("VRAM delta vs baseline: %+.3f GB", vram_delta)


if __name__ == "__main__":
    main()
