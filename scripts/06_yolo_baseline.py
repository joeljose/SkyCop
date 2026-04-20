"""
Experiment 06 — pretrained YOLOv8s baseline (no training).

Two modes run back-to-back:

1. Offline eval on the exp 05 holdout → mAP@0.5, mAP@0.5:0.95, per-class AP
   for COCO-reachable classes (car / truck / bus). Van is excluded from
   pretrained scoring because COCO has no van class — the omission is
   recorded explicitly in the metrics JSON, not silently.

2. Live pursuit on the exp 04 scene → detection overlay on the MJPEG
   stream, sustained-FPS and peak-VRAM measurement. This is the acceptance
   bar for NFR-01 / NFR-03 at this pipeline depth (detection only; tracking
   and fingerprint still to land).

Both modes use the pretrained yolov8s.pt COCO checkpoint with the SkyCop
class remap defined in configs/detector.yaml.
"""

import json
import logging
import os
import queue
import random
import time
from pathlib import Path

import carla
import cv2
import numpy as np

from skycop.config import load
from skycop.control import AdaptiveAltitudeController, AltitudeConfig
from skycop.cv import (
    CLASS_NAMES,
    EvalBox,
    YoloDetector,
    coco_to_skycop_map,
    read_yolo_labels,
    score_predictions,
)
from skycop.dashboard import MJPEGServer
from skycop.logs import setup_logging
from skycop.sim import (
    SuspectParams,
    carla_image_to_bgr,
    connect,
    destroy_all,
    spawn_aerial_camera,
    spawn_npcs,
    spawn_reckless_suspect,
    synchronous_mode,
)

log = logging.getLogger("exp06")


def ensure_map(client, map_name):
    world = client.get_world()
    if map_name in world.get_map().name:
        return world
    log.info("loading map %s", map_name)
    return client.load_world(map_name)


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


# ── Mode 2: live pursuit ───────────────────────────────────────────────────

def overlay_detections(frame, detections, class_names):
    for d in detections:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[d.class_idx]} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def overlay_stats(frame, fps: float, infer_ms: float, vram_gb: float):
    txt = f"{fps:5.1f} FPS | det {infer_ms:5.1f}ms | VRAM {vram_gb:4.2f}GB"
    cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2, cv2.LINE_AA)


def peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def run_live_pursuit(
    cfg,
    detector: YoloDetector,
    server: MJPEGServer,
    offline_metrics: dict,
) -> dict:
    rng = random.Random(cfg.seed)
    actors: list[carla.Actor] = []
    log.info("connecting to CARLA at %s:%s", cfg.carla.host, cfg.carla.port)
    client = connect(host=cfg.carla.host, port=cfg.carla.port)
    world = ensure_map(client, cfg.carla.map)

    infer_times_ms: list[float] = []
    frame_times_s: list[float] = []

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    with synchronous_mode(world, cfg.carla.fixed_delta_seconds):
        tm = client.get_trafficmanager(cfg.carla.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(cfg.seed)

        try:
            npcs, remaining = spawn_npcs(world, tm, cfg.scene.npc_count, rng)
            actors.extend(npcs)
            log.info("spawned %d/%d NPCs", len(npcs), cfg.scene.npc_count)

            suspect_params = SuspectParams(**dict(cfg.scene.suspect))
            suspect = spawn_reckless_suspect(world, tm, remaining, rng, suspect_params)
            actors.append(suspect)
            log.info("spawned suspect %s", suspect.type_id)

            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(100.0)

            camera, img_queue = spawn_aerial_camera(
                world, width=cfg.camera.width, height=cfg.camera.height, fov=cfg.camera.fov,
            )
            actors.append(camera)

            altitude_ctrl = AdaptiveAltitudeController(world, AltitudeConfig(**dict(cfg.altitude)))

            duration = float(cfg.baseline.live_pursuit_seconds)
            log.info("live pursuit %.0fs with detection overlay…", duration)

            t0 = time.perf_counter()
            frames = 0
            peak_vram = 0.0

            while True:
                frame_start = time.perf_counter()
                loc = suspect.get_transform().location
                target_z, _ = altitude_ctrl.step(loc.x, loc.y)
                camera.set_transform(carla.Transform(
                    carla.Location(loc.x, loc.y, target_z),
                    carla.Rotation(pitch=cfg.camera.pitch, yaw=0, roll=0),
                ))

                world.tick()

                try:
                    image = img_queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                frame = carla_image_to_bgr(image)

                t_inf_start = time.perf_counter()
                detections = detector.predict(frame)
                infer_ms = (time.perf_counter() - t_inf_start) * 1000
                infer_times_ms.append(infer_ms)

                overlay_detections(frame, detections, CLASS_NAMES)
                cur_vram = peak_vram_gb()
                peak_vram = max(peak_vram, cur_vram)

                frames += 1
                elapsed = time.perf_counter() - t0
                fps_so_far = frames / elapsed if elapsed > 0 else 0.0
                overlay_stats(frame, fps_so_far, infer_ms, cur_vram)
                server.push(frame)

                frame_times_s.append(time.perf_counter() - frame_start)

                if elapsed >= duration:
                    break

            total_elapsed = time.perf_counter() - t0
            sustained_fps = frames / total_elapsed
            p95_infer = float(np.percentile(infer_times_ms, 95))
            mean_infer = float(np.mean(infer_times_ms))
            mean_frame_ms = float(np.mean(frame_times_s)) * 1000

            log.info("frames          = %d", frames)
            log.info("sustained FPS   = %.2f", sustained_fps)
            log.info("detection mean  = %.2f ms (p95 %.2f ms)", mean_infer, p95_infer)
            log.info("frame total     = %.2f ms (CARLA + overlay + MJPEG push)", mean_frame_ms)
            log.info("peak VRAM       = %.2f GB", peak_vram)

            if sustained_fps < float(cfg.baseline.fps_threshold_warn):
                log.warning("sustained FPS below threshold (%s)", cfg.baseline.fps_threshold_warn)
            if peak_vram > float(cfg.baseline.vram_threshold_gb):
                log.warning("peak VRAM above threshold (%s GB)", cfg.baseline.vram_threshold_gb)

            result = {
                "frames": frames,
                "duration_s": total_elapsed,
                "sustained_fps": sustained_fps,
                "detection_mean_ms": mean_infer,
                "detection_p95_ms": p95_infer,
                "frame_total_mean_ms": mean_frame_ms,
                "peak_vram_gb": peak_vram,
                "fps_threshold_warn": float(cfg.baseline.fps_threshold_warn),
                "vram_threshold_gb": float(cfg.baseline.vram_threshold_gb),
            }

            # Save combined metrics HERE, before the cleanup finally.
            # CARLA's C++ terminate on actor-registry issues can SIGABRT the
            # process during tm teardown, bypassing Python's return path.
            metrics = {
                "model": cfg.detector.weights,
                "pretrained_offline_eval": offline_metrics,
                "pretrained_live_pursuit": result,
            }
            save_metrics_atomic(metrics, Path(cfg.baseline.metrics_out))
            log.info("metrics → %s", cfg.baseline.metrics_out)

        finally:
            # Clean up TM state BEFORE destroying the hero actor it references
            # (carla_caveats §10 — hybrid physics anchors on hero, attempting to
            # reset after hero is gone triggers "destroyed actor" SIGABRT).
            try:
                tm.set_hybrid_physics_mode(False)
                tm.set_synchronous_mode(False)
            except Exception:
                pass
            destroy_all(actors)

    return result


def save_metrics_atomic(metrics: dict, out_path: Path) -> None:
    """Write metrics JSON atomically — CARLA's SIGABRT on process exit can
    interrupt a plain write before the page cache flushes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


def main():
    setup_logging()
    cfg = load("default", "altitude", "detector")
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
    # Metrics JSON is saved inside run_live_pursuit before the CARLA cleanup
    # finally — CARLA's SIGABRT on TM teardown can bypass Python's return path.
    run_live_pursuit(cfg, detector, server, offline_metrics=offline)


if __name__ == "__main__":
    main()
