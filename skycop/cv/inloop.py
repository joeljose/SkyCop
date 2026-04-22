"""Live-pursuit detector evaluation loop.

Shared between exp 06 (pretrained baseline) and exp 09 (fine-tuned in-loop).
Runs the exp 04 scene for ``duration_s``, overlays detector predictions on
the MJPEG stream, and measures sustained FPS, detection latency, and peak
torch-process VRAM.
"""

from __future__ import annotations

import logging
import queue
import random
import time
from dataclasses import dataclass

import carla
import cv2
import numpy as np

from skycop.cv.inference import Detection, YoloDetector
from skycop.cv.vehicle_classes import CLASS_NAMES
from skycop.dashboard import MJPEGServer
from skycop.sim import (
    SuspectParams,
    carla_image_to_bgr,
    connect,
    spawn_aerial_camera,
    spawn_npcs,
    spawn_reckless_suspect,
    synchronous_mode,
    teardown_pursuit,
)

log = logging.getLogger(__name__)


@dataclass
class LivePursuitResult:
    frames: int
    duration_s: float
    sustained_fps: float
    detection_mean_ms: float
    detection_p95_ms: float
    frame_total_mean_ms: float
    peak_vram_gb: float

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "duration_s": self.duration_s,
            "sustained_fps": self.sustained_fps,
            "detection_mean_ms": self.detection_mean_ms,
            "detection_p95_ms": self.detection_p95_ms,
            "frame_total_mean_ms": self.frame_total_mean_ms,
            "peak_vram_gb": self.peak_vram_gb,
        }


def _overlay_detections(frame, detections: list[Detection]) -> None:
    for d in detections:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{CLASS_NAMES[d.class_idx]} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def _overlay_stats(frame, fps: float, infer_ms: float, vram_gb: float) -> None:
    txt = f"{fps:5.1f} FPS | det {infer_ms:5.1f}ms | VRAM {vram_gb:4.2f}GB"
    cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2, cv2.LINE_AA)


def _peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _ensure_map(client: carla.Client, map_name: str) -> carla.World:
    world = client.get_world()
    if map_name in world.get_map().name:
        return world
    log.info("loading map %s", map_name)
    return client.load_world(map_name)


def measure_live_pursuit(
    cfg,
    detector: YoloDetector,
    server: MJPEGServer,
    duration_s: float,
    fps_threshold_warn: float = 18.0,
    vram_threshold_gb: float = 5.5,
) -> LivePursuitResult:
    """Run a timed live pursuit with overlaid detections; return measurements.

    Uses the exp 04 scene (50 NPCs + hero suspect + adaptive altitude) on the
    map specified in cfg.carla.map. Always ends with a clean teardown — the
    returned result is populated before teardown begins, so CARLA's SIGABRT
    during cleanup (already mitigated by teardown_pursuit but not proven zero)
    doesn't lose the measurement.
    """
    rng = random.Random(cfg.seed)
    actors: list[carla.Actor] = []
    log.info("connecting to CARLA at %s:%s", cfg.carla.host, cfg.carla.port)
    client = connect(host=cfg.carla.host, port=cfg.carla.port)
    world = _ensure_map(client, cfg.carla.map)

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

            # Altitude pinned per D-12 — adaptive altitude controller dropped.
            pinned_altitude = float(cfg.camera.altitude)

            log.info("live pursuit %.0fs with detection overlay…", duration_s)
            t0 = time.perf_counter()
            frames = 0
            peak_vram = 0.0

            while True:
                frame_start = time.perf_counter()
                loc = suspect.get_transform().location
                camera.set_transform(carla.Transform(
                    carla.Location(loc.x, loc.y, loc.z + pinned_altitude),
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

                _overlay_detections(frame, detections)
                cur_vram = _peak_vram_gb()
                peak_vram = max(peak_vram, cur_vram)

                frames += 1
                elapsed = time.perf_counter() - t0
                fps_so_far = frames / elapsed if elapsed > 0 else 0.0
                _overlay_stats(frame, fps_so_far, infer_ms, cur_vram)
                server.push(frame)

                frame_times_s.append(time.perf_counter() - frame_start)

                if elapsed >= duration_s:
                    break

            total_elapsed = time.perf_counter() - t0
            result = LivePursuitResult(
                frames=frames,
                duration_s=total_elapsed,
                sustained_fps=frames / total_elapsed if total_elapsed > 0 else 0.0,
                detection_mean_ms=float(np.mean(infer_times_ms)) if infer_times_ms else 0.0,
                detection_p95_ms=float(np.percentile(infer_times_ms, 95)) if infer_times_ms else 0.0,
                frame_total_mean_ms=float(np.mean(frame_times_s)) * 1000 if frame_times_s else 0.0,
                peak_vram_gb=peak_vram,
            )

            log.info("frames          = %d", result.frames)
            log.info("sustained FPS   = %.2f", result.sustained_fps)
            log.info("detection mean  = %.2f ms (p95 %.2f ms)",
                     result.detection_mean_ms, result.detection_p95_ms)
            log.info("frame total     = %.2f ms", result.frame_total_mean_ms)
            log.info("peak VRAM       = %.3f GB", result.peak_vram_gb)

            if result.sustained_fps < fps_threshold_warn:
                log.warning("sustained FPS below threshold (%s)", fps_threshold_warn)
            if result.peak_vram_gb > vram_threshold_gb:
                log.warning("peak VRAM above threshold (%s GB)", vram_threshold_gb)

        finally:
            teardown_pursuit(client, world, tm, actors)

    return result
