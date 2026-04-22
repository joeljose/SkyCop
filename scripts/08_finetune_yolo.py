"""
Experiment 08 — Fine-tune YOLOv8s on the exp 07 CARLA training dataset.

Pipeline:

 1. Prepare Ultralytics YOLO dataset YAML from exp 07's ``dataset_manifest.json``.
 2. Capture the Town03 cross-map validity probe (~100 frames) if missing.
    This runs against the live CARLA server — stop the server afterwards if
    you want to free VRAM before training.
 3. Fine-tune YOLOv8s from pretrained COCO weights (``yolov8s.pt``).
 4. Evaluate fine-tuned weights on both the Town10HD_Opt holdout (same-map)
    and the Town03 probe (cross-map).
 5. Write ``output/eval/fine_tuned_metrics.json`` with both numbers plus a
    comparison against the exp 06 pretrained baseline.

Interpretation rules (per docs/design.md D-11):
  - same_map < 30%   → pipeline broken; debug before reporting.
  - same_map ≥ 70% AND cross_map < 30%  → memorisation; caveat loudly.
  - same_map ≥ 70% AND cross_map ≥ 65%  → genuine generalisation.
  - Anything in between → judgment call; quote both numbers, flag the gap.
"""

import json
import logging
import os
import time
from pathlib import Path

import cv2

from skycop.config import load
from skycop.cv.capture import reset_world, run_capture
from skycop.cv.eval import EvalBox, read_yolo_labels, score_predictions
from skycop.cv.inference import YoloDetector
from skycop.cv.training import (
    TrainingConfig,
    best_weights_path,
    build_dataset_yaml,
    train_detector,
)
from skycop.logs import setup_logging
from skycop.sim import connect

log = logging.getLogger("exp08")


def prepare_dataset_yaml(cfg) -> Path:
    """Read exp 07's combined manifest, emit an Ultralytics YAML."""
    manifest_path = Path(cfg.training_dataset.output_dir) / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing. Run `make exp N=07` first to collect training data."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    yaml_path = Path(cfg.training_dataset.output_dir) / "dataset.yaml"
    build_dataset_yaml(
        dataset_root=Path(cfg.training_dataset.output_dir),
        train_runs=manifest["train_runs"],
        val_runs=manifest["val_runs"],
        out_path=yaml_path,
    )
    return yaml_path


def ensure_crossmap_probe(cfg) -> Path:
    """Capture the Town03 cross-map probe if not already on disk."""
    probe_dir = Path(cfg.crossmap_probe.output_dir)
    manifest_path = probe_dir / "manifest.json"
    if manifest_path.exists():
        log.info("crossmap probe already captured at %s — skipping", probe_dir)
        return probe_dir

    log.info("capturing crossmap probe on %s (seed=%d, weather=%s)…",
             cfg.crossmap_probe.map, cfg.crossmap_probe.seed, cfg.crossmap_probe.weather)

    # The probe needs a CARLA connection. Override the default map for this call.
    probe_cfg = load("default")
    probe_cfg.carla.map = cfg.crossmap_probe.map

    client = connect(host=probe_cfg.carla.host, port=probe_cfg.carla.port)
    cleared = reset_world(client, probe_cfg.carla.tm_port)
    if cleared:
        log.info("reset: destroyed %d orphan actors/sensors", cleared)
    time.sleep(0.5)

    run_capture(
        probe_cfg,
        output_dir=probe_dir,
        client=client,
        run_id="crossmap_town03",
        seed=int(cfg.crossmap_probe.seed),
        weather=str(cfg.crossmap_probe.weather),
        target_frames=int(cfg.crossmap_probe.target_frames),
        subsample_every=int(cfg.crossmap_probe.subsample_every),
        max_ticks=int(cfg.crossmap_probe.max_ticks),
        min_bbox_pixel=int(cfg.crossmap_probe.min_bbox_pixel),
        min_visibility=float(cfg.crossmap_probe.min_visibility),
        jpeg_quality=int(cfg.crossmap_probe.jpeg_quality),
    )
    return probe_dir


def evaluate_on(detector: YoloDetector, eval_dir: Path, label: str) -> dict:
    img_dir = eval_dir / "images"
    lbl_dir = eval_dir / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} missing for {label} eval")

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

    metrics = score_predictions(preds_per_frame, gts_per_frame, class_filter={0})
    log.info("[%s] mAP@0.5=%.4f  mAP@0.5:0.95=%.4f  preds=%d  gt=%d",
             label, metrics.map_50, metrics.map_50_95,
             metrics.n_predictions, metrics.n_ground_truths)
    return {
        "mAP_50": metrics.map_50,
        "mAP_50_95": metrics.map_50_95,
        "per_class_ap_50": metrics.per_class_ap_50,
        "n_frames": metrics.n_frames,
        "n_predictions": metrics.n_predictions,
        "n_ground_truths": metrics.n_ground_truths,
    }


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
    cfg = load("default", "training_dataset", "training")

    # 1. Dataset YAML
    yaml_path = prepare_dataset_yaml(cfg)

    # 2. Crossmap probe (idempotent — regenerates only if missing)
    probe_dir = ensure_crossmap_probe(cfg)

    # 3. Train
    training_cfg = TrainingConfig(
        weights=str(cfg.training.weights),
        data_yaml=yaml_path,
        epochs=int(cfg.training.epochs),
        patience=int(cfg.training.patience),
        imgsz=int(cfg.training.imgsz),
        batch=int(cfg.training.batch),
        optimizer=str(cfg.training.optimizer),
        seed=int(cfg.training.seed),
        deterministic=bool(cfg.training.deterministic),
        half=bool(cfg.training.half),
        mosaic=float(cfg.training.mosaic),
        degrees=float(cfg.training.degrees),
        hsv_h=float(cfg.training.hsv_h),
        hsv_s=float(cfg.training.hsv_s),
        hsv_v=float(cfg.training.hsv_v),
        scale=float(cfg.training.scale),
        translate=float(cfg.training.translate),
        project=str(cfg.training.project),
        name=str(cfg.training.name),
        device=str(cfg.training.device),
    )
    log.info("starting training — this may take 1–2 hours")
    t0 = time.perf_counter()
    train_detector(training_cfg)
    log.info("training done in %.1fs", time.perf_counter() - t0)

    # 4. Evaluate on both holdouts
    weights = best_weights_path(training_cfg)
    if not weights.exists():
        raise RuntimeError(f"best weights missing at {weights}")
    log.info("evaluating %s on holdouts", weights)

    detector = YoloDetector(
        weights=str(weights),
        class_remap=None,          # single-class, identity map
        conf_threshold=0.25,
        iou_threshold=0.45,
        input_size=int(cfg.training.imgsz),
        device=str(cfg.training.device),
        fp16=bool(cfg.training.half),
    )

    same_map = evaluate_on(detector, Path(cfg.eval.same_map_dir), "same-map (Town10HD)")
    cross_map = evaluate_on(detector, probe_dir, "cross-map (Town03)")

    # 5. Metrics JSON + interpretation
    gap = same_map["mAP_50"] - cross_map["mAP_50"]

    baseline = {}
    baseline_path = Path(cfg.eval.baseline_metrics)
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f).get("pretrained_offline_eval", {})

    payload = {
        "model": str(weights),
        "fine_tuned_same_map": same_map,
        "fine_tuned_cross_map": cross_map,
        "same_cross_gap_pts": gap,
        "pretrained_baseline_same_map": baseline.get("mAP_50"),
        "interpretation": _interpret(same_map["mAP_50"], cross_map["mAP_50"]),
        "notes": [
            "Training data: single map Town10HD_Opt. Cross-map probe: Town03, 100 frames.",
            "CUDA training has minor non-determinism; identical seeds yield ±1 mAP@0.5 variation.",
            "Single-class `vehicle` detector per D-09; fingerprint class sourced separately.",
        ],
    }
    save_metrics_atomic(payload, Path(cfg.eval.metrics_out))
    log.info("metrics → %s", cfg.eval.metrics_out)
    log.info("same-map=%.4f  cross-map=%.4f  gap=%.2f pts  → %s",
             same_map["mAP_50"], cross_map["mAP_50"], gap * 100,
             payload["interpretation"])


def _interpret(same_map: float, cross_map: float) -> str:
    if same_map < 0.30:
        return "BROKEN: same-map < 30%, debug data pipeline before reporting"
    if same_map >= 0.70 and cross_map < 0.30:
        return "MEMORISATION SUSPECTED: same-map high but cross-map collapsed"
    if same_map >= 0.70 and cross_map >= 0.65:
        return "GOOD: same-map and cross-map both strong, real generalisation"
    return "PARTIAL: report both numbers, flag the gap in progress.md"


if __name__ == "__main__":
    main()
