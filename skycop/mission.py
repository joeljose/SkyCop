"""Mission v1a — tracker-driven closed-loop pursuit with HSV fingerprint rebind.

Vertical slice of the SkyCop application. Spawns an NPC scene and a
reckless suspect; drone starts at the suspect's initial position (cheat —
dispatch bootstrap is FR-03 work) and from there follows the tracker's
locked-suspect bbox via a per-axis flight PID with target-velocity
feedforward. Altitude stays pinned per D-12. Camera yaw still follows
suspect yaw by GT (gimbal PID deferred to PR 2b).

Three primary metrics, per HOTA's "split orthogonal quality dimensions"
philosophy:

- ``id_accuracy`` — fraction of GT-visible frames where ``locked_track_id``
  corresponds to the GT-suspect-matched track (perception / CV quality)
- ``track_distance_m`` — world-space drone-to-suspect distance, mean + p95
  (control quality)
- ``in_frame_rate`` — fraction of ticks where GT bbox is fully inside the
  image (framing consequence)

Artifacts per run, under ``output/mission/<timestamp>/``:

- ``trace.jsonl`` — per-tick diagnostics (center offset, drone→suspect
  distance, commanded velocity magnitude)
- ``mission.mp4`` — overlay video (optional; default off, live view via MJPEG)
- ``summary.json`` — the three primaries + run metadata + fingerprint config
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

from skycop.control import PursuitPID, TargetStateTracker
from skycop.cv.capture import weather_preset
from skycop.cv.fingerprint import Fingerprint, extract, score
from skycop.cv.gt_projection import (
    build_camera_matrix,
    iou_xyxy,
    pixel_to_world_on_ground,
    project_points,
    world_bbox_to_image,
)
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
    # Three primaries (PR 2a metric panel — see D-13).
    id_accuracy: float               # locked == GT on visible frames
    track_distance_mean_m: float     # mean drone→suspect world distance
    track_distance_p95_m: float      # p95 drone→suspect world distance
    in_frame_rate: float             # GT bbox fully inside image
    # Legacy diagnostic kept for continuity with Mission v0 summaries.
    legacy_iou_correctness: float
    initial_lock_frame: int | None
    initial_lock_track_id: int | None
    suspect_actor_id: int
    suspect_type_id: str
    seed: int
    weather: str
    video_path: str | None
    trace_path: str
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
    running_id_accuracy: float | None,
    drone_to_suspect_m: float | None = None,
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
    if running_id_accuracy is not None:
        cv2.putText(out, f"id_acc {running_id_accuracy:.3f}", (8, h - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2, cv2.LINE_AA)
    if drone_to_suspect_m is not None:
        cv2.putText(out, f"dist {drone_to_suspect_m:5.1f}m", (8, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 80), 2, cv2.LINE_AA)
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

    # Flight controller config (PR 2a — D-13).
    ctrl_cfg = cfg.control
    flight_cfg = ctrl_cfg.flight
    ff_cfg = ctrl_cfg.feedforward
    hlk_cfg = ctrl_cfg.hold_last_known
    ground_z = float(ctrl_cfg.ground_plane.z)
    flight_Kp = float(flight_cfg.Kp)
    flight_Ki = float(flight_cfg.Ki)
    flight_Kd = float(flight_cfg.Kd)
    flight_clamp = float(flight_cfg.output_clamp_mps)
    flight_int_clamp = float(flight_cfg.integral_clamp)
    ff_enabled = bool(ff_cfg.enabled)
    ff_scale = float(ff_cfg.scale)
    ff_window = int(ff_cfg.window_size)
    hlk_trigger = int(hlk_cfg.trigger_ticks)

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
    trace_path = run_dir / "trace.jsonl"

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

            log.info("altitude pinned to mission.altitude_m=%.1fm (per D-12)", altitude_m)

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

            # Warm-up tick so the freshly-spawned suspect actually has its
            # spawn transform reflected (fresh CARLA actors report (0, 0, 0)
            # until the first world.tick() after spawn). Also drain the
            # queued frame so iter-0 reads a frame that matches iter-0's pose.
            world.tick()
            try:
                rgb_q.get(timeout=2.0)
            except queue.Empty:
                pass

            # Control loop state (PR 2a).
            # Drone cold-starts at suspect's initial XY — dispatch bootstrap (FR-03) deferred.
            suspect_tf0 = suspect.get_transform()
            drone_pos_world = np.array(
                [suspect_tf0.location.x, suspect_tf0.location.y, suspect_tf0.location.z + altitude_m],
                dtype=np.float64,
            )
            log.info(
                "drone cold-start at suspect spawn: (%.1f, %.1f, %.1f)",
                drone_pos_world[0], drone_pos_world[1], drone_pos_world[2],
            )

            pid_x = PursuitPID(Kp=flight_Kp, Ki=flight_Ki, Kd=flight_Kd,
                               output_clamp=flight_clamp, integral_clamp=flight_int_clamp)
            pid_y = PursuitPID(Kp=flight_Kp, Ki=flight_Ki, Kd=flight_Kd,
                               output_clamp=flight_clamp, integral_clamp=flight_int_clamp)
            target_tracker = TargetStateTracker(window_size=ff_window)
            ticks_since_lock = 0   # counter for hold-last-known

            dt = float(cfg.carla.fixed_delta_seconds)

            seed_fp: Fingerprint | None = None
            locked_track_id: int | None = None
            initial_lock_frame: int | None = None
            initial_lock_track_id: int | None = None

            frames_total = 0
            frames_suspect_visible = 0
            frames_locked = 0
            frames_iou_correct = 0          # legacy metric
            frames_id_correct = 0           # new — id_accuracy numerator
            frames_in_frame = 0             # new — in_frame_rate numerator
            distance_samples: list[float] = []

            target_ticks = int(duration_s / dt)
            log.info("mission v1a: %.1fs / %d ticks / flight Kp=%.2f Kd=%.2f / feedforward=%s",
                     duration_s, target_ticks, flight_Kp, flight_Kd, ff_enabled)

            trace_fh = open(trace_path, "w", buffering=1)
            t0 = time.perf_counter()
            try:
                for tick in range(target_ticks):
                    # Camera yaw follows suspect body yaw via GT (gimbal PID deferred to PR 2b).
                    suspect_yaw = float(suspect.get_transform().rotation.yaw)
                    pose = carla.Transform(
                        carla.Location(
                            float(drone_pos_world[0]),
                            float(drone_pos_world[1]),
                            float(drone_pos_world[2]),
                        ),
                        carla.Rotation(
                            pitch=float(cfg.camera.pitch),
                            yaw=suspect_yaw,
                            roll=0.0,
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

                    # GT bookkeeping (metrics only — not fed into the controller).
                    cam_to_world = _camera_world_matrix(rgb_cam)
                    w2c = np.linalg.inv(cam_to_world)
                    gt_bbox = world_bbox_to_image(
                        _suspect_world_vertices(suspect), w2c, K, image_size
                    )
                    suspect_loc = suspect.get_transform().location
                    suspect_xyz = np.array([suspect_loc.x, suspect_loc.y, suspect_loc.z])
                    drone_to_suspect = float(np.linalg.norm(
                        drone_pos_world[:2] - suspect_xyz[:2]
                    ))
                    distance_samples.append(drone_to_suspect)

                    suspect_visible = gt_bbox is not None
                    if suspect_visible:
                        frames_suspect_visible += 1
                        # Fully-in-frame check: project the 8 GT vertices un-clipped
                        # via project_points (world_bbox_to_image clips to bounds, so
                        # its output can't distinguish "touches edge" from "centered").
                        uv_verts = project_points(
                            _suspect_world_vertices(suspect), w2c, K,
                        )
                        finite = ~np.isnan(uv_verts).any(axis=1)
                        if int(finite.sum()) == 8:  # all 8 vertices project finitely
                            us, vs = uv_verts[:, 0], uv_verts[:, 1]
                            if (
                                us.min() >= 0 and us.max() <= image_size[1] - 1
                                and vs.min() >= 0 and vs.max() <= image_size[0] - 1
                            ):
                                frames_in_frame += 1

                    # Detect + track.
                    tracks = adapter.update(frame_bgr)

                    # Fingerprint bootstrap (first tick where a tracker box overlaps GT).
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

                    # Fingerprint matching + sticky rebind.
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

                    # Identify the "correct" tracker box for metric bookkeeping: the one
                    # whose bbox has IoU ≥ gate with the GT projection.
                    gt_matched_track_id: int | None = None
                    if gt_bbox is not None:
                        best_gt_iou = 0.0
                        for t in tracks:
                            if t.track_id is None:
                                continue
                            gt_iou = iou_xyxy(t.bbox, gt_bbox)
                            if gt_iou > best_gt_iou and gt_iou >= iou_gate:
                                best_gt_iou = gt_iou
                                gt_matched_track_id = t.track_id

                    locked_track = None
                    if locked_track_id is not None:
                        frames_locked += 1
                        locked_track = next(
                            (t for t in tracks if t.track_id == locked_track_id), None
                        )

                    # id_accuracy: on GT-visible frames, did we lock the right track_id?
                    if suspect_visible and gt_matched_track_id is not None:
                        if locked_track_id == gt_matched_track_id:
                            frames_id_correct += 1

                    # Legacy IoU correctness (for continuity with Mission v0 summary).
                    if (
                        locked_track is not None
                        and gt_bbox is not None
                        and iou_xyxy(locked_track.bbox, gt_bbox) >= iou_gate
                    ):
                        frames_iou_correct += 1

                    # ── Flight control ───────────────────────────────────
                    # Estimate target world position from locked track's bbox center
                    # (pixel → ground-plane inverse projection).
                    target_xy_world: tuple[float, float] | None = None
                    if locked_track is not None:
                        u = 0.5 * (locked_track.x1 + locked_track.x2)
                        v = 0.5 * (locked_track.y1 + locked_track.y2)
                        target_xy_world = pixel_to_world_on_ground(
                            u, v, K, cam_to_world, ground_z=ground_z,
                        )

                    center_offset_px = 0.0
                    velocity_cmd_magnitude = 0.0
                    ff_vx = 0.0
                    ff_vy = 0.0

                    if target_xy_world is not None:
                        ticks_since_lock = 0
                        tx, ty = target_xy_world
                        target_tracker.update(float(tick) * dt, tx, ty)

                        # Velocity feedforward.
                        if ff_enabled:
                            v_est = target_tracker.velocity
                            if v_est is not None:
                                ff_vx = ff_scale * v_est[0]
                                ff_vy = ff_scale * v_est[1]

                        err_x = tx - drone_pos_world[0]
                        err_y = ty - drone_pos_world[1]
                        vx_cmd = pid_x.step(err_x, dt, feedforward=ff_vx)
                        vy_cmd = pid_y.step(err_y, dt, feedforward=ff_vy)

                        drone_pos_world[0] += vx_cmd * dt
                        drone_pos_world[1] += vy_cmd * dt
                        velocity_cmd_magnitude = float(np.hypot(vx_cmd, vy_cmd))

                        # Centre-offset for the trace (distance of locked bbox centre from image centre).
                        img_cx = 0.5 * (image_size[1] - 1)
                        img_cy = 0.5 * (image_size[0] - 1)
                        center_offset_px = float(np.hypot(u - img_cx, v - img_cy))
                    else:
                        # Hold-last-known (SIM-17). Freeze drone pose; decay integrator state.
                        ticks_since_lock += 1
                        if ticks_since_lock >= hlk_trigger:
                            pid_x.reset()
                            pid_y.reset()
                            target_tracker.reset()

                    # ── Overlay + live push ──────────────────────────────
                    running_id_accuracy = (
                        frames_id_correct / frames_suspect_visible
                        if frames_suspect_visible > 0 else None
                    )
                    overlay = _render_mission_overlay(
                        frame_bgr, gt_bbox, tracks, locked_track_id, track_scores,
                        frames_total, running_id_accuracy, drone_to_suspect,
                    )
                    if video_writer is not None:
                        video_writer.write(overlay)
                    if mjpeg_server is not None:
                        mjpeg_server.push(overlay)

                    # ── Per-tick trace ───────────────────────────────────
                    trace_fh.write(json.dumps({
                        "tick": tick,
                        "drone_to_suspect_m": round(drone_to_suspect, 3),
                        "center_offset_px": round(center_offset_px, 1),
                        "velocity_cmd_m_s": round(velocity_cmd_magnitude, 3),
                        "locked_track_id": locked_track_id,
                        "gt_matched_track_id": gt_matched_track_id,
                        "suspect_visible": bool(suspect_visible),
                    }) + "\n")

                    frames_total += 1
            finally:
                trace_fh.close()

            elapsed = time.perf_counter() - t0

            # Three primaries + legacy metric.
            id_accuracy = (
                frames_id_correct / frames_suspect_visible
                if frames_suspect_visible > 0 else 0.0
            )
            in_frame_rate = (
                frames_in_frame / frames_total if frames_total > 0 else 0.0
            )
            legacy_iou_correctness = (
                frames_iou_correct / frames_suspect_visible
                if frames_suspect_visible > 0 else 0.0
            )
            if distance_samples:
                track_distance_mean_m = float(np.mean(distance_samples))
                track_distance_p95_m = float(np.percentile(distance_samples, 95.0))
            else:
                track_distance_mean_m = 0.0
                track_distance_p95_m = 0.0

            log.info(
                "mission done: %d frames in %.1fs · visible=%d · locked=%d · "
                "id_acc=%.3f · track_dist mean=%.2fm p95=%.2fm · in_frame=%.3f",
                frames_total, elapsed, frames_suspect_visible, frames_locked,
                id_accuracy, track_distance_mean_m, track_distance_p95_m,
                in_frame_rate,
            )

            result = MissionResult(
                run_id=run_id,
                duration_s=elapsed,
                frames_total=frames_total,
                frames_suspect_visible=frames_suspect_visible,
                frames_locked=frames_locked,
                id_accuracy=round(id_accuracy, 4),
                track_distance_mean_m=round(track_distance_mean_m, 3),
                track_distance_p95_m=round(track_distance_p95_m, 3),
                in_frame_rate=round(in_frame_rate, 4),
                legacy_iou_correctness=round(legacy_iou_correctness, 4),
                initial_lock_frame=initial_lock_frame,
                initial_lock_track_id=initial_lock_track_id,
                suspect_actor_id=int(suspect.id),
                suspect_type_id=str(suspect.type_id),
                seed=used_seed,
                weather=weather_name,
                video_path=str(video_path) if save_video else None,
                trace_path=str(trace_path),
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
                payload["control"] = {
                    "flight_Kp": flight_Kp,
                    "flight_Ki": flight_Ki,
                    "flight_Kd": flight_Kd,
                    "output_clamp_mps": flight_clamp,
                    "feedforward_enabled": ff_enabled,
                    "feedforward_scale": ff_scale,
                    "ff_window_size": ff_window,
                    "hold_last_known_trigger_ticks": hlk_trigger,
                    "ground_plane_z": ground_z,
                }
                payload["iou_correctness_threshold"] = iou_gate
                json.dump(payload, f, indent=2)
            log.info("summary → %s", summary_path)
            log.info("trace → %s", trace_path)
            return result

        finally:
            if video_writer is not None:
                video_writer.release()
            teardown_pursuit(client, world, tm, actors)
