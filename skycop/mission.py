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
import math
import queue
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np

from skycop.control import (
    Pose,
    PursuitPID,
    TargetStateTracker,
    UserDroneConfig,
    UserDroneController,
)
from skycop.control.collision import apply_slide_along
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
from skycop.sim.suspect_fsm import (
    FSMConfig,
    ParkingSubstate,
    SuspectFSM,
    SuspectState,
    TMKnobs,
)
from skycop.vendor.carla_agents import GlobalRoutePlanner, LocalPlanner

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
    # v0a (issue #44): suspect FSM outcome + per-state metrics.
    fsm_submission: str | None       # "ai_pass" / "timeout_lose" / None
    fsm_terminal_state: str          # final FSM state name
    per_state: dict = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────

def _suspect_world_vertices(vehicle: carla.Actor) -> np.ndarray:
    """Return (8, 3) world-space coords of the vehicle's bounding box."""
    verts = vehicle.bounding_box.get_world_vertices(vehicle.get_transform())
    return np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)


def _camera_world_matrix(camera: carla.Actor) -> np.ndarray:
    return np.array(camera.get_transform().get_matrix(), dtype=np.float64)


# ── FSM plumbing ──────────────────────────────────────────────────────

def _build_fsm(suspect_cfg) -> SuspectFSM:
    """Assemble a SuspectFSM from the OmegaConf `suspect` subtree."""
    fsm_sec = suspect_cfg.fsm
    knobs = suspect_cfg.knobs

    def _knobs(block) -> TMKnobs:
        return TMKnobs(
            speed_over_pct=float(block.speed_over_pct),
            ignore_lights_pct=float(block.ignore_lights_pct),
            ignore_signs_pct=float(block.ignore_signs_pct),
            lane_change_pct=float(block.lane_change_pct),
            follow_dist_m=float(block.follow_dist_m),
        )

    def _xyz(p) -> tuple[float, float, float]:
        return (float(p.x), float(p.y), float(p.z))

    cfg = FSMConfig(
        fleeing_duration_s=float(fsm_sec.fleeing_duration_s),
        roaming_duration_s=float(fsm_sec.roaming_duration_s),
        parking_lot_timeout_s=float(fsm_sec.parking_lot_timeout_s),
        parking_roadside_timeout_s=float(fsm_sec.parking_roadside_timeout_s),
        parked_confirm_timeout_s=float(fsm_sec.parked_confirm_timeout_s),
        reach_distance_m=float(fsm_sec.reach_distance_m),
        reach_speed_mps=float(fsm_sec.reach_speed_mps),
        ai_consecutive_lock_ticks=int(fsm_sec.ai_consecutive_lock_ticks),
        fleeing_knobs=_knobs(knobs.fleeing),
        roaming_knobs=_knobs(knobs.roaming),
        parking_lots=tuple(_xyz(p) for p in suspect_cfg.parking_lots),
        roadside_spots=tuple(_xyz(p) for p in suspect_cfg.roadside_spots),
    )
    return SuspectFSM(cfg=cfg)


def _apply_tm_knobs(
    tm: carla.TrafficManager,
    vehicle: carla.Actor,
    knobs: TMKnobs,
) -> None:
    """Write the six TM knobs the FSM emits on state entry."""
    tm.vehicle_percentage_speed_difference(vehicle, knobs.speed_over_pct)
    tm.ignore_lights_percentage(vehicle, knobs.ignore_lights_pct)
    tm.ignore_signs_percentage(vehicle, knobs.ignore_signs_pct)
    tm.random_left_lanechange_percentage(vehicle, knobs.lane_change_pct)
    tm.random_right_lanechange_percentage(vehicle, knobs.lane_change_pct)
    tm.distance_to_leading_vehicle(vehicle, knobs.follow_dist_m)


def _build_local_planner(
    vehicle: carla.Actor,
    grp: GlobalRoutePlanner,
    dest_xyz: tuple[float, float, float],
    target_speed_kmh: float,
    sampling_resolution_m: float,
    dt: float,
) -> LocalPlanner:
    """Route vehicle→destination and hand off to a fresh LocalPlanner.

    The caller must have already called ``vehicle.set_autopilot(False)``
    so the TM doesn't fight the planner.
    """
    dest_loc = carla.Location(*dest_xyz)
    route = grp.trace_route(vehicle.get_location(), dest_loc)
    planner = LocalPlanner(
        vehicle,
        opt_dict={
            "target_speed": target_speed_kmh,
            "dt": dt,
            "sampling_radius": sampling_resolution_m,
        },
    )
    planner.set_global_plan(route, stop_waypoint_creation=True)
    return planner


@dataclass
class _PerStateAccum:
    """Rolling accumulators so we can report id_accuracy etc. per FSM state."""
    frames_total: int = 0
    frames_visible: int = 0
    frames_id_correct: int = 0
    frames_in_frame: int = 0
    distance_samples: list[float] = field(default_factory=list)

    def finalise(self) -> dict:
        id_acc = (
            self.frames_id_correct / self.frames_visible
            if self.frames_visible > 0 else None
        )
        in_frame_rate = (
            self.frames_in_frame / self.frames_total
            if self.frames_total > 0 else None
        )
        if self.distance_samples:
            d_mean = float(np.mean(self.distance_samples))
            d_p95 = float(np.percentile(self.distance_samples, 95.0))
        else:
            d_mean = d_p95 = None
        return {
            "frames_total": self.frames_total,
            "frames_visible": self.frames_visible,
            "id_accuracy": round(id_acc, 4) if id_acc is not None else None,
            "in_frame_rate": round(in_frame_rate, 4) if in_frame_rate is not None else None,
            "track_distance_mean_m": round(d_mean, 3) if d_mean is not None else None,
            "track_distance_p95_m": round(d_p95, 3) if d_p95 is not None else None,
        }


def _fmt_score(s: float) -> str:
    return f"{s:.2f}"


def _fsm_hud_line(fsm_tick) -> str:
    """Short HUD string: FSM state + (substate or countdown)."""
    state = fsm_tick.state.value
    if fsm_tick.parking_substate is not None:
        return f"{state}·{fsm_tick.parking_substate.value}"
    if fsm_tick.countdown_s is not None:
        return f"{state} {fsm_tick.countdown_s:4.1f}s"
    return state


def _render_mission_overlay(
    frame_bgr: np.ndarray,
    gt_bbox: tuple[float, float, float, float] | None,
    tracks: list[Track],
    locked_track_id: int | None,
    track_scores: dict[int, float],
    frame_idx: int,
    running_id_accuracy: float | None,
    drone_to_suspect_m: float | None = None,
    hud_extra: str | None = None,
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
    if hud_extra:
        cv2.putText(out, hud_extra, (8, h - 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 255), 2, cv2.LINE_AA)
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
    mode = str(mission_cfg.get("mode", "ai")).lower()
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

    client = connect(host=cfg.carla.host, port=cfg.carla.port)
    world = client.get_world()
    if cfg.carla.map not in world.get_map().name:
        world = client.load_world(cfg.carla.map)
    world.set_weather(weather_preset(weather_name))

    # Start trigger (v0a): mission blocks here until the user clicks
    # "Start pursuit" on the MJPEG page (or instantly if the server wasn't
    # constructed with use_start_trigger=True). The server also tells us
    # which mode the user picked on the menu — we honour that over cfg.
    if mjpeg_server is not None and not mjpeg_server.started:
        log.info("mission paused — waiting for /start click on the MJPEG page…")
        mjpeg_server.wait_for_start()
        mode = mjpeg_server.started_mode or mode
        log.info("start received — mode=%s", mode)

    # YOLO weights only required in AI mode (pipeline is disabled in user mode).
    if mode == "ai" and not weights_path.exists():
        raise FileNotFoundError(f"Fine-tuned weights missing: {weights_path}. Run exp 08.")

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

            # YOLO + ByteTrack only in AI mode (skipped in user mode).
            adapter: ByteTrackAdapter | None = None
            if mode == "ai":
                adapter = ByteTrackAdapter(
                    weights=str(weights_path),
                    tracker_yaml=str(cfg.tracking.tracker_yaml),
                    conf_threshold=float(cfg.detector.conf_threshold),
                    iou_threshold=float(cfg.detector.iou_threshold),
                    input_size=int(cfg.training.imgsz),
                    device=str(cfg.training.device),
                    fp16=bool(cfg.training.half),
                )

            log.info("altitude pinned to mission.altitude_m=%.1fm (v0a bump — adaptive altitude is issue #45)", altitude_m)

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

            # Flight PID + feedforward + HLK — AI mode only.
            pid_x = pid_y = None
            target_tracker = None
            ticks_since_lock = 0
            if mode == "ai":
                pid_x = PursuitPID(Kp=flight_Kp, Ki=flight_Ki, Kd=flight_Kd,
                                   output_clamp=flight_clamp, integral_clamp=flight_int_clamp)
                pid_y = PursuitPID(Kp=flight_Kp, Ki=flight_Ki, Kd=flight_Kd,
                                   output_clamp=flight_clamp, integral_clamp=flight_int_clamp)
                target_tracker = TargetStateTracker(window_size=ff_window)

            # User-mode controller — body-frame snappy WASD-QE (v0c).
            user_ctrl: UserDroneController | None = None
            drone_yaw_rad = math.radians(float(suspect_tf0.rotation.yaw))
            if mode == "user":
                user_cfg = UserDroneConfig(
                    max_speed_mps=float(flight_clamp),
                    altitude_rate_mps=5.0,
                    yaw_rate_deg_s=60.0,
                    min_altitude_m=5.0,
                )
                user_ctrl = UserDroneController(cfg=user_cfg)
                log.info("user mode — WASD + QE + Shift/Ctrl controls active")

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

            # ── Suspect FSM (v0a, issue #44) ──────────────────────
            fsm = _build_fsm(cfg.suspect)
            suspect_controller_cfg = cfg.suspect.controller
            grp = GlobalRoutePlanner(
                world.get_map(),
                float(suspect_controller_cfg.sampling_resolution_m),
            )
            local_planner: LocalPlanner | None = None
            per_state: dict[SuspectState, _PerStateAccum] = {
                s: _PerStateAccum() for s in SuspectState
            }
            fsm_submission: str | None = None
            fsm_terminal_state: SuspectState = SuspectState.FLEEING

            target_ticks = int(duration_s / dt)   # safety cap; FSM terminal usually exits earlier
            log.info("mission v0a: fsm + flight PID / safety cap %.1fs / Kp=%.2f Kd=%.2f",
                     duration_s, flight_Kp, flight_Kd)

            trace_fh = open(trace_path, "w", buffering=1)
            t0 = time.perf_counter()
            try:
                for tick in range(target_ticks):
                    # ── Camera / drone pose for this tick ──────────────
                    # AI mode: camera yaw follows suspect body yaw via GT.
                    # User mode: camera yaw = user-controlled drone yaw.
                    if mode == "user":
                        camera_yaw_deg = math.degrees(drone_yaw_rad)
                    else:
                        camera_yaw_deg = float(suspect.get_transform().rotation.yaw)
                    pose = carla.Transform(
                        carla.Location(
                            float(drone_pos_world[0]),
                            float(drone_pos_world[1]),
                            float(drone_pos_world[2]),
                        ),
                        carla.Rotation(
                            pitch=float(cfg.camera.pitch),
                            yaw=camera_yaw_deg,
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
                    suspect_fully_in_frame = False
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
                                suspect_fully_in_frame = True

                    # ── AI mode: detect + track + fingerprint + flight PID ──
                    # User mode: skip all of this — user flies via WASDQE.
                    tracks = []
                    track_scores: dict[int, float] = {}
                    gt_matched_track_id: int | None = None
                    locked_track = None
                    center_offset_px = 0.0
                    velocity_cmd_magnitude = 0.0

                    if mode == "ai":
                        assert adapter is not None
                        assert pid_x is not None and pid_y is not None
                        assert target_tracker is not None

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
                        if gt_bbox is not None:
                            best_gt_iou = 0.0
                            for t in tracks:
                                if t.track_id is None:
                                    continue
                                gt_iou = iou_xyxy(t.bbox, gt_bbox)
                                if gt_iou > best_gt_iou and gt_iou >= iou_gate:
                                    best_gt_iou = gt_iou
                                    gt_matched_track_id = t.track_id

                        if locked_track_id is not None:
                            frames_locked += 1
                            locked_track = next(
                                (t for t in tracks if t.track_id == locked_track_id), None
                            )

                        # id_accuracy: on GT-visible frames, did we lock the right track_id?
                        if (
                            suspect_visible
                            and gt_matched_track_id is not None
                            and locked_track_id == gt_matched_track_id
                        ):
                            frames_id_correct += 1

                        # Legacy IoU correctness (for continuity with Mission v0 summary).
                        if (
                            locked_track is not None
                            and gt_bbox is not None
                            and iou_xyxy(locked_track.bbox, gt_bbox) >= iou_gate
                        ):
                            frames_iou_correct += 1

                        # Flight control: pixel → ground-plane inverse projection.
                        target_xy_world: tuple[float, float] | None = None
                        if locked_track is not None:
                            u = 0.5 * (locked_track.x1 + locked_track.x2)
                            v = 0.5 * (locked_track.y1 + locked_track.y2)
                            target_xy_world = pixel_to_world_on_ground(
                                u, v, K, cam_to_world, ground_z=ground_z,
                            )

                        ff_vx = 0.0
                        ff_vy = 0.0
                        if target_xy_world is not None:
                            ticks_since_lock = 0
                            tx, ty = target_xy_world
                            target_tracker.update(float(tick) * dt, tx, ty)
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

                            # Centre-offset for the trace (locked bbox centre vs image centre).
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

                    elif mode == "user":
                        assert user_ctrl is not None
                        assert mjpeg_server is not None
                        pressed = mjpeg_server.get_pressed_keys()
                        current_pose = Pose(
                            x=float(drone_pos_world[0]),
                            y=float(drone_pos_world[1]),
                            z=float(drone_pos_world[2]),
                            yaw_rad=drone_yaw_rad,
                        )
                        intended = user_ctrl.step(current_pose, pressed, dt)
                        safe = apply_slide_along(world, current_pose, intended)
                        drone_pos_world[0] = safe.x
                        drone_pos_world[1] = safe.y
                        drone_pos_world[2] = safe.z
                        drone_yaw_rad = safe.yaw_rad
                        velocity_cmd_magnitude = float(
                            math.hypot(safe.x - current_pose.x, safe.y - current_pose.y) / dt
                        )

                    # ── Suspect FSM step ─────────────────────────────────
                    suspect_tf_now = suspect.get_transform()
                    suspect_vel = suspect.get_velocity()
                    suspect_speed = float(math.hypot(suspect_vel.x, suspect_vel.y))
                    ai_lock_on_suspect = bool(
                        suspect_visible
                        and locked_track_id is not None
                        and locked_track_id == gt_matched_track_id
                    )
                    fsm_tick = fsm.tick(
                        t=float(tick) * dt,
                        suspect_xy=(suspect_tf_now.location.x, suspect_tf_now.location.y),
                        suspect_speed=suspect_speed,
                        ai_lock_on_suspect=ai_lock_on_suspect,
                    )

                    # Side-effects requested by the FSM on state entries.
                    act = fsm_tick.action
                    if act.apply_tm_knobs is not None:
                        _apply_tm_knobs(tm, suspect, act.apply_tm_knobs)
                    if act.set_path_to is not None:
                        # PARKING entry (trying_lot) OR roadside escalation — both
                        # rebuild the LocalPlanner. We disable autopilot the first
                        # time only.
                        if local_planner is None:
                            suspect.set_autopilot(False)
                        local_planner = _build_local_planner(
                            vehicle=suspect, grp=grp,
                            dest_xyz=act.set_path_to,
                            target_speed_kmh=float(suspect_controller_cfg.target_speed_kmh),
                            sampling_resolution_m=float(suspect_controller_cfg.sampling_resolution_m),
                            dt=dt,
                        )
                        log.info("FSM → PARKING: routing to %s", act.set_path_to)
                    if act.freeze_physics:
                        suspect.set_simulate_physics(False)
                        log.info("FSM → PARKED: physics frozen, confirmation window open")

                    # Drive the suspect during PARKING (off-autopilot); brake during
                    # park_in_place. In PARKED we've frozen physics; apply a zero-
                    # control so no residual command lingers.
                    if fsm_tick.state == SuspectState.PARKING:
                        if fsm_tick.parking_substate == ParkingSubstate.PARK_IN_PLACE:
                            suspect.apply_control(carla.VehicleControl(brake=1.0))
                        elif local_planner is not None:
                            suspect.apply_control(local_planner.run_step())
                    elif fsm_tick.state == SuspectState.PARKED:
                        suspect.apply_control(carla.VehicleControl(brake=1.0))

                    # Per-state metric accumulation — mirrors the aggregate counters.
                    acc = per_state[fsm_tick.state]
                    acc.frames_total += 1
                    if suspect_visible:
                        acc.frames_visible += 1
                        if locked_track_id is not None and locked_track_id == gt_matched_track_id:
                            acc.frames_id_correct += 1
                        if suspect_fully_in_frame:
                            acc.frames_in_frame += 1
                    acc.distance_samples.append(drone_to_suspect)

                    # Push FSM state to MJPEG server (used by v0c user page for
                    # the colour panel, submit enable, end modal).
                    if mjpeg_server is not None:
                        mjpeg_server.set_fsm_state(
                            state=fsm_tick.state.value,
                            countdown_s=fsm_tick.countdown_s,
                            terminal=bool(fsm_tick.terminal),
                            result=fsm_tick.submission,
                        )

                    # User-mode submission grading — only while in PARKED.
                    if (
                        mode == "user"
                        and mjpeg_server is not None
                        and fsm_tick.state == SuspectState.PARKED
                        and not fsm_tick.terminal
                    ):
                        sub = mjpeg_server.get_submission()
                        if sub is not None:
                            mjpeg_server.clear_submission()
                            passed = False
                            if gt_bbox is not None:
                                bx1, by1, bx2, by2 = sub["bbox"]
                                cx = 0.5 * (bx1 + bx2)
                                cy = 0.5 * (by1 + by2)
                                gx1, gy1, gx2, gy2 = gt_bbox
                                passed = (gx1 <= cx <= gx2) and (gy1 <= cy <= gy2)
                            fsm_submission = "user_pass" if passed else "user_fail"
                            fsm_terminal_state = SuspectState.PARKED
                            log.info(
                                "user submission graded: %s (bbox=%s, gt=%s)",
                                fsm_submission, sub["bbox"], gt_bbox,
                            )
                            mjpeg_server.set_fsm_state(
                                state=fsm_tick.state.value,
                                countdown_s=fsm_tick.countdown_s,
                                terminal=True,
                                result=fsm_submission,
                            )
                            # Write a final overlay frame + break the loop.
                            # (overlay + trace happen below before break)

                    # ── Overlay + live push ──────────────────────────────
                    # User mode: no tracker overlay, no GT bbox (user finds the
                    # suspect themselves), no running accuracy (n/a).
                    running_id_accuracy = (
                        frames_id_correct / frames_suspect_visible
                        if frames_suspect_visible > 0 else None
                    )
                    hud_extra = _fsm_hud_line(fsm_tick)
                    if mode == "user":
                        overlay = _render_mission_overlay(
                            frame_bgr, None, [], None, {},
                            frames_total, None, drone_to_suspect,
                            hud_extra=hud_extra,
                        )
                    else:
                        overlay = _render_mission_overlay(
                            frame_bgr, gt_bbox, tracks, locked_track_id, track_scores,
                            frames_total, running_id_accuracy, drone_to_suspect,
                            hud_extra=hud_extra,
                        )
                    if video_writer is not None:
                        video_writer.write(overlay)
                    if mjpeg_server is not None:
                        mjpeg_server.push(overlay)

                    # ── Per-tick trace ───────────────────────────────────
                    # GT poses logged so we can reconstruct the scene offline
                    # (e.g. to see where the suspect got stuck vs where the
                    # drone hovered when the tracker drifted).
                    trace_fh.write(json.dumps({
                        "tick": tick,
                        "drone_xy": [round(float(drone_pos_world[0]), 2),
                                     round(float(drone_pos_world[1]), 2)],
                        "suspect_xy": [round(suspect_tf_now.location.x, 2),
                                       round(suspect_tf_now.location.y, 2)],
                        "suspect_speed_mps": round(suspect_speed, 3),
                        "suspect_yaw_deg": round(float(suspect_tf_now.rotation.yaw), 1),
                        "drone_to_suspect_m": round(drone_to_suspect, 3),
                        "center_offset_px": round(center_offset_px, 1),
                        "velocity_cmd_m_s": round(velocity_cmd_magnitude, 3),
                        "locked_track_id": locked_track_id,
                        "gt_matched_track_id": gt_matched_track_id,
                        "suspect_visible": bool(suspect_visible),
                        "fsm_state": fsm_tick.state.value,
                        "fsm_sub": (fsm_tick.parking_substate.value
                                    if fsm_tick.parking_substate else None),
                        "fsm_countdown_s": (round(fsm_tick.countdown_s, 2)
                                            if fsm_tick.countdown_s is not None else None),
                    }) + "\n")

                    frames_total += 1

                    fsm_terminal_state = fsm_tick.state
                    if fsm_tick.terminal:
                        fsm_submission = fsm_tick.submission
                        log.info("FSM terminal at tick %d: submission=%s",
                                 tick, fsm_submission)
                        break
                    # User-mode bbox grading set fsm_submission out-of-band.
                    if fsm_submission in ("user_pass", "user_fail"):
                        log.info("User submission terminal at tick %d: %s",
                                 tick, fsm_submission)
                        break
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

            per_state_summary = {s.value: per_state[s].finalise() for s in SuspectState}

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
                fsm_submission=fsm_submission,
                fsm_terminal_state=fsm_terminal_state.value,
                per_state=per_state_summary,
            )
            log.info("FSM outcome: terminal_state=%s submission=%s",
                     result.fsm_terminal_state, result.fsm_submission)

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
