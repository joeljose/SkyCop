"""Mission v0 — drone-above-suspect pursuit with HSV fingerprint rebind.

First vertical slice of the SkyCop application. Spawns an NPC scene and a
reckless suspect, pins the drone directly above the suspect at a fixed
altitude and pitch (bypassing the adaptive altitude controller so exp 10b's
oscillation finding doesn't interfere), runs YOLO + ByteTrack, seeds an HSV
fingerprint at initial lock, and sticky-rebinds the "suspect track_id" each
tick against that fingerprint.

Dev-mode scaffolding: CARLA's suspect ``actor_id`` is the ground truth
both for seeding the fingerprint (at the first frame where a tracker
bbox overlaps the GT-projected bbox) and for scoring mission correctness
(IoU between the locked tracker bbox and the GT bbox per tick). Dispatch/
clue-based bootstrap is deferred to a later slice.

Artifacts per run, under ``output/mission/<timestamp>/``:

- ``mission.mp4`` — overlay video written as-you-go.
- ``summary.json`` — metadata + correctness score + fingerprint config.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np

from skycop.cv.capture import weather_preset
from skycop.cv.fingerprint import Fingerprint, extract, score
from skycop.cv.gt_projection import build_camera_matrix, iou_xyxy, world_bbox_to_image
from skycop.cv.track import ByteTrackAdapter, Track
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


# ── Result types ───────────────────────────────────────────────────────

@dataclass
class MissionResult:
    run_id: str
    duration_s: float
    frames_total: int
    frames_suspect_visible: int
    frames_locked: int
    frames_iou_correct: int
    correctness: float
    initial_lock_frame: int | None
    initial_lock_track_id: int | None
    suspect_actor_id: int
    suspect_type_id: str
    seed: int
    weather: str
    video_path: str | None
    summary_path: str


# ── Helpers ────────────────────────────────────────────────────────────

def _suspect_world_vertices(vehicle: carla.Actor) -> np.ndarray:
    """Return (8, 3) world-space coords of the vehicle's bounding box."""
    verts = vehicle.bounding_box.get_world_vertices(vehicle.get_transform())
    return np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)


def _camera_world_matrix(camera: carla.Actor) -> np.ndarray:
    return np.array(camera.get_transform().get_matrix(), dtype=np.float64)


def _fmt_score(s: float) -> str:
    return f"{s:.2f}"


def _render_mission_overlay(
    frame_bgr: np.ndarray,
    gt_bbox: tuple[float, float, float, float] | None,
    tracks: list[Track],
    locked_track_id: int | None,
    track_scores: dict[int, float],
    frame_idx: int,
    running_correctness: float | None,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # GT suspect (thick green)
    if gt_bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in gt_bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(out, "GT suspect", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)

    # Tracker boxes
    for t in tracks:
        p1 = (int(t.x1), int(t.y1))
        p2 = (int(t.x2), int(t.y2))
        is_locked = t.track_id is not None and t.track_id == locked_track_id
        colour = (0, 255, 255) if is_locked else (240, 160, 40)  # yellow locked / orange otherwise
        thick = 3 if is_locked else 2
        cv2.rectangle(out, p1, p2, colour, thick)
        label = f"t{t.track_id}" if t.track_id is not None else "t?"
        fp_s = track_scores.get(t.track_id) if t.track_id is not None else None
        if fp_s is not None:
            label += f"  fp={_fmt_score(fp_s)}"
        if is_locked:
            label = "LOCKED " + label
        cv2.putText(out, label, (p1[0], max(12, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    # HUD
    cv2.putText(out, f"frame {frame_idx}", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if running_correctness is not None:
        cv2.putText(out, f"correctness {running_correctness:.3f}", (8, h - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2, cv2.LINE_AA)
    return out


def _choose_locked_track(
    track_scores: dict[int, float],
    current_lock: int | None,
    rebind_threshold: float,
    stickiness: float,
) -> int | None:
    """Sticky rebind. Keep ``current_lock`` unless a candidate beats it by ``stickiness``."""
    if not track_scores:
        return current_lock
    best_id, best_s = max(track_scores.items(), key=lambda kv: kv[1])
    if current_lock is None:
        return best_id if best_s >= rebind_threshold else None
    cur_s = track_scores.get(current_lock)
    if cur_s is None:
        # Locked track has disappeared this frame — fall back to best if it passes threshold
        return best_id if best_s >= rebind_threshold else current_lock
    if best_id != current_lock and best_s >= rebind_threshold and (best_s - cur_s) > stickiness:
        return best_id
    return current_lock


# ── Orchestrator ───────────────────────────────────────────────────────

def run_mission(cfg, mjpeg_server: MJPEGServer | None = None) -> MissionResult:
    mission_cfg = cfg.mission
    duration_s = float(mission_cfg.duration_s)
    altitude_m = float(mission_cfg.altitude_m)
    weather_name = str(mission_cfg.weather)
    iou_gate = float(mission_cfg.iou_correctness_threshold)
    save_video = bool(mission_cfg.get("save_video", False))
    seed_mode = str(mission_cfg.get("seed_mode", "fixed"))
    fp_cfg = mission_cfg.fingerprint
    rebind_threshold = float(fp_cfg.score_threshold_rebind)
    stickiness = float(fp_cfg.score_stickiness)
    hsv_bins = int(fp_cfg.hsv_bins)
    video_fps = int(mission_cfg.video_fps)

    # Seed: fixed uses cfg.seed verbatim; random derives one from the wall clock
    # and logs it, so the run is still reproducible via summary.json.
    if seed_mode == "random":
        used_seed = int(time.time_ns() & 0xFFFFFFFF)
        log.info("seed_mode=random → using seed=%d (record it to reproduce)", used_seed)
    elif seed_mode == "fixed":
        used_seed = int(cfg.seed)
        log.info("seed_mode=fixed → using cfg.seed=%d", used_seed)
    else:
        raise ValueError(f"seed_mode must be 'fixed' or 'random', got {seed_mode!r}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(mission_cfg.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / "mission.mp4"
    summary_path = run_dir / "summary.json"

    weights_path = Path(cfg.training.project) / cfg.training.name / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Fine-tuned weights missing: {weights_path}. Run exp 08.")

    client = connect(host=cfg.carla.host, port=cfg.carla.port)
    world = client.get_world()
    if cfg.carla.map not in world.get_map().name:
        world = client.load_world(cfg.carla.map)
    world.set_weather(weather_preset(weather_name))

    K = build_camera_matrix(int(cfg.camera.width), int(cfg.camera.height), float(cfg.camera.fov))
    image_size = (int(cfg.camera.height), int(cfg.camera.width))

    actors: list[carla.Actor] = []
    rng = random.Random(used_seed)

    with synchronous_mode(world, float(cfg.carla.fixed_delta_seconds)):
        tm = client.get_trafficmanager(int(cfg.carla.tm_port))
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(used_seed)

        video_writer: cv2.VideoWriter | None = None
        try:
            npcs, remaining = spawn_npcs(world, tm, int(cfg.scene.npc_count), rng)
            actors.extend(npcs)
            log.info("spawned %d/%d NPCs", len(npcs), int(cfg.scene.npc_count))

            suspect_params = SuspectParams(**dict(cfg.scene.suspect))
            suspect = spawn_reckless_suspect(world, tm, remaining, rng, suspect_params)
            actors.append(suspect)
            log.info("spawned suspect %s (id=%d)", suspect.type_id, suspect.id)

            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(100.0)

            rgb_cam, rgb_q = spawn_aerial_camera(
                world,
                width=int(cfg.camera.width),
                height=int(cfg.camera.height),
                fov=int(cfg.camera.fov),
            )
            actors.append(rgb_cam)

            adapter = ByteTrackAdapter(
                weights=str(weights_path),
                tracker_yaml=str(cfg.tracking.tracker_yaml),
                conf_threshold=float(cfg.detector.conf_threshold),
                iou_threshold=float(cfg.detector.iou_threshold),
                input_size=int(cfg.training.imgsz),
                device=str(cfg.training.device),
                fp16=bool(cfg.training.half),
            )

            log.info("altitude pinned to mission.altitude_m=%.1fm (adaptive controller dropped per D-12)", altitude_m)

            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_path), fourcc, video_fps,
                    (int(cfg.camera.width), int(cfg.camera.height)),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter at {video_path}")
            else:
                log.info("video saving disabled (mission.save_video=false); live view only")

            seed_fp: Fingerprint | None = None
            locked_track_id: int | None = None
            initial_lock_frame: int | None = None
            initial_lock_track_id: int | None = None

            frames_total = 0
            frames_suspect_visible = 0
            frames_locked = 0
            frames_iou_correct = 0

            target_ticks = int(duration_s / float(cfg.carla.fixed_delta_seconds))
            log.info("mission v0: %.1fs / %d ticks / altitude=%.1fm / video → %s",
                     duration_s, target_ticks, altitude_m, video_path)

            t0 = time.perf_counter()
            for tick in range(target_ticks):
                suspect_tf = suspect.get_transform()
                loc = suspect_tf.location
                pose = carla.Transform(
                    carla.Location(loc.x, loc.y, loc.z + altitude_m),
                    carla.Rotation(
                        pitch=float(cfg.camera.pitch),
                        yaw=float(suspect_tf.rotation.yaw),
                        roll=0,
                    ),
                )
                rgb_cam.set_transform(pose)
                world.tick()

                try:
                    image = rgb_q.get(timeout=2.0)
                except queue.Empty:
                    log.warning("tick %d: no frame", tick)
                    continue
                frame_bgr = carla_image_to_bgr(image)

                # GT projection (done AFTER tick so the camera's world transform is current)
                w2c = np.linalg.inv(_camera_world_matrix(rgb_cam))
                gt_bbox = world_bbox_to_image(
                    _suspect_world_vertices(suspect), w2c, K, image_size
                )
                if gt_bbox is not None:
                    frames_suspect_visible += 1

                # Detect + track
                tracks = adapter.update(frame_bgr)

                # Fingerprint bootstrap: first frame with a tracker box overlapping the GT
                if seed_fp is None and gt_bbox is not None:
                    for t in tracks:
                        if t.track_id is None:
                            continue
                        if iou_xyxy(t.bbox, gt_bbox) >= iou_gate:
                            seed_fp = extract(frame_bgr, t.bbox, bins=hsv_bins)
                            if seed_fp.is_valid():
                                locked_track_id = t.track_id
                                initial_lock_frame = frames_total
                                initial_lock_track_id = t.track_id
                                log.info("initial lock: frame=%d track_id=%d",
                                         frames_total, t.track_id)
                                break

                # Fingerprint matching + sticky rebind
                track_scores: dict[int, float] = {}
                if seed_fp is not None:
                    for t in tracks:
                        if t.track_id is None:
                            continue
                        cand_fp = extract(frame_bgr, t.bbox, bins=hsv_bins)
                        track_scores[t.track_id] = score(seed_fp, cand_fp)
                    locked_track_id = _choose_locked_track(
                        track_scores, locked_track_id, rebind_threshold, stickiness
                    )

                # Correctness accounting
                if locked_track_id is not None:
                    frames_locked += 1
                    locked_track = next(
                        (t for t in tracks if t.track_id == locked_track_id), None
                    )
                    if locked_track is not None and gt_bbox is not None:
                        if iou_xyxy(locked_track.bbox, gt_bbox) >= iou_gate:
                            frames_iou_correct += 1

                running_correctness = (
                    frames_iou_correct / frames_suspect_visible
                    if frames_suspect_visible > 0 else None
                )
                overlay = _render_mission_overlay(
                    frame_bgr, gt_bbox, tracks, locked_track_id, track_scores,
                    frames_total, running_correctness,
                )

                if video_writer is not None:
                    video_writer.write(overlay)
                if mjpeg_server is not None:
                    mjpeg_server.push(overlay)
                frames_total += 1

            elapsed = time.perf_counter() - t0
            correctness = (
                frames_iou_correct / frames_suspect_visible
                if frames_suspect_visible > 0 else 0.0
            )
            log.info(
                "mission done: %d frames in %.1fs · suspect-visible=%d · locked=%d · correct=%d · correctness=%.3f",
                frames_total, elapsed, frames_suspect_visible, frames_locked,
                frames_iou_correct, correctness,
            )

            result = MissionResult(
                run_id=run_id,
                duration_s=elapsed,
                frames_total=frames_total,
                frames_suspect_visible=frames_suspect_visible,
                frames_locked=frames_locked,
                frames_iou_correct=frames_iou_correct,
                correctness=round(correctness, 4),
                initial_lock_frame=initial_lock_frame,
                initial_lock_track_id=initial_lock_track_id,
                suspect_actor_id=int(suspect.id),
                suspect_type_id=str(suspect.type_id),
                seed=used_seed,
                weather=weather_name,
                video_path=str(video_path) if save_video else None,
                summary_path=str(summary_path),
            )

            with open(summary_path, "w") as f:
                payload = asdict(result)
                payload["seed_mode"] = seed_mode
                payload["fingerprint"] = {
                    "hsv_bins": hsv_bins,
                    "score_threshold_rebind": rebind_threshold,
                    "score_stickiness": stickiness,
                }
                payload["iou_correctness_threshold"] = iou_gate
                json.dump(payload, f, indent=2)
            log.info("summary → %s", summary_path)
            return result

        finally:
            if video_writer is not None:
                video_writer.release()
            teardown_pursuit(client, world, tm, actors)
