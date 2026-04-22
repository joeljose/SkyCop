"""Reusable single-run pursuit-capture logic.

Both exp 05 (eval-holdout capture) and exp 07 (training-data collection)
produce the same shape of artifact — RGB + YOLO-format labels + a manifest
from a seeded pursuit run. This module is that shared path.

Each call to `run_capture` runs one pursuit: sets weather, spawns the
scene, drives the drone through the adaptive-altitude loop, captures
paired RGB + instance-seg frames every Nth tick, extracts labels via
`skycop.cv.dataset.extract_yolo_labels_from_seg`, and writes a manifest.
"""

from __future__ import annotations

import logging
import queue
import random
import time
from dataclasses import dataclass
from pathlib import Path

import carla
import cv2

from skycop.cv.dataset import (
    DatasetManifest,
    extract_actor_boxes_from_seg,
    extract_yolo_labels_from_seg,
    write_yolo_label,
)
from skycop.cv.vehicle_classes import CLASS_NAMES, detector_class_for
from skycop.sim import (
    SuspectParams,
    carla_image_to_bgr,
    spawn_aerial_camera,
    spawn_npcs,
    spawn_reckless_suspect,
    synchronous_mode,
    teardown_pursuit,
)

log = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    """Summary of one captured run."""
    run_id: str
    seed: int
    weather: str
    frames_saved: int
    class_counts: dict[str, int]
    skip_counts: dict[str, int]
    duration_s: float
    output_dir: Path


def weather_preset(name: str) -> carla.WeatherParameters:
    """Look up a CARLA weather preset by name (e.g. ``'ClearNoon'``).

    Raises ``ValueError`` with a helpful message if unknown.
    """
    try:
        return getattr(carla.WeatherParameters, name)
    except AttributeError as e:
        valid = [
            attr for attr in dir(carla.WeatherParameters)
            if not attr.startswith("_")
            and isinstance(getattr(carla.WeatherParameters, attr, None), carla.WeatherParameters)
        ]
        raise ValueError(
            f"Unknown weather preset '{name}'. Valid: {sorted(valid)}"
        ) from e


def reset_world(client: carla.Client, tm_port: int) -> int:
    """Destroy every vehicle and sensor and put the world + TM back in async mode.

    Called between runs so one run's stale actors don't haunt the next.
    Returns the count of actors destroyed.
    """
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    try:
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(False)
        tm.set_hybrid_physics_mode(False)
    except Exception:
        pass

    actors = list(world.get_actors().filter("vehicle.*")) + \
             list(world.get_actors().filter("sensor.*"))
    for a in actors:
        try:
            if hasattr(a, "is_listening") and a.is_listening:
                a.stop()
            a.destroy()
        except Exception:
            pass
    return len(actors)


def ensure_map(client: carla.Client, map_name: str) -> carla.World:
    world = client.get_world()
    if map_name in world.get_map().name:
        return world
    log.info("loading map %s", map_name)
    return client.load_world(map_name)


def _make_actor_classifier(world: carla.World):
    """Cached actor_id → detector-class-index lookup for extraction callback."""
    cache: dict[int, int | None] = {}

    def classify(aid: int) -> int | None:
        if aid in cache:
            return cache[aid]
        actor = world.get_actor(aid)
        if actor is None or not actor.type_id.startswith("vehicle."):
            cache[aid] = None
            return None
        wheels = int(actor.attributes.get("number_of_wheels", "4"))
        idx = detector_class_for(actor.type_id, wheels)
        cache[aid] = idx
        return idx

    return classify


def _spawn_instance_seg_camera(world, width: int, height: int, fov: int):
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find("sensor.camera.instance_segmentation")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    transform = carla.Transform(carla.Location(0, 0, 100), carla.Rotation(pitch=-75))
    cam = world.spawn_actor(bp, transform)
    q: queue.Queue[carla.Image] = queue.Queue()
    cam.listen(q.put)
    return cam, q


def run_capture(
    cfg,
    output_dir: Path,
    *,
    client: carla.Client,
    run_id: str,
    seed: int,
    weather: str,
    target_frames: int,
    subsample_every: int,
    max_ticks: int,
    min_bbox_pixel: int,
    min_visibility: float,
    jpeg_quality: int,
) -> CaptureResult:
    """Run one seeded pursuit, dumping paired RGB + labels + manifest to `output_dir`.

    The world is assumed to be reset (no stale actors) when this is called;
    pair with `reset_world()` between invocations when looping.
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    actors: list[carla.Actor] = []

    world = ensure_map(client, cfg.carla.map)
    world.set_weather(weather_preset(weather))

    manifest = DatasetManifest(
        seed=seed,
        fixed_delta_seconds=cfg.carla.fixed_delta_seconds,
        map_name=cfg.carla.map,
        class_names=list(CLASS_NAMES),
        min_pixel=int(min_bbox_pixel),
        min_visibility=float(min_visibility),
    )

    t0 = time.perf_counter()
    saved = 0

    with synchronous_mode(world, cfg.carla.fixed_delta_seconds):
        tm = client.get_trafficmanager(cfg.carla.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)

        try:
            npcs, remaining = spawn_npcs(world, tm, cfg.scene.npc_count, rng)
            actors.extend(npcs)
            log.info("[%s] spawned %d/%d NPCs", run_id, len(npcs), cfg.scene.npc_count)

            suspect_params = SuspectParams(**dict(cfg.scene.suspect))
            suspect = spawn_reckless_suspect(world, tm, remaining, rng, suspect_params)
            actors.append(suspect)
            log.info("[%s] spawned suspect %s (id=%d)", run_id, suspect.type_id, suspect.id)

            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(100.0)

            rgb_cam, rgb_q = spawn_aerial_camera(
                world, width=cfg.camera.width, height=cfg.camera.height, fov=cfg.camera.fov,
            )
            actors.append(rgb_cam)

            seg_cam, seg_q = _spawn_instance_seg_camera(
                world, width=cfg.camera.width, height=cfg.camera.height, fov=cfg.camera.fov,
            )
            actors.append(seg_cam)

            # Altitude pinned per D-12 — adaptive altitude controller dropped
            # (empirically unnecessary on Town10HD per the REQUIREMENTS audit).
            pinned_altitude = float(cfg.camera.altitude)
            classify = _make_actor_classifier(world)

            log.info("[%s] capturing up to %d frames…", run_id, target_frames)

            for tick in range(max_ticks):
                loc = suspect.get_transform().location
                pose = carla.Transform(
                    carla.Location(loc.x, loc.y, loc.z + pinned_altitude),
                    carla.Rotation(pitch=cfg.camera.pitch, yaw=0, roll=0),
                )
                rgb_cam.set_transform(pose)
                seg_cam.set_transform(pose)

                world.tick()

                try:
                    rgb_img = rgb_q.get(timeout=2.0)
                    seg_img = seg_q.get(timeout=2.0)
                except queue.Empty:
                    continue

                if tick % subsample_every != 0:
                    continue

                seg_bgr = carla_image_to_bgr(seg_img)
                boxes, stats = extract_yolo_labels_from_seg(
                    seg_bgr, classify,
                    min_pixel=int(min_bbox_pixel),
                    min_visibility=float(min_visibility),
                )
                if not boxes:
                    continue

                rgb_bgr = carla_image_to_bgr(rgb_img)
                frame_id = f"frame_{saved:04d}"
                cv2.imwrite(
                    str(img_dir / f"{frame_id}.jpg"),
                    rgb_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                write_yolo_label(lbl_dir / f"{frame_id}.txt", boxes)

                manifest.record_frame(
                    index=saved,
                    tick=tick,
                    camera_pose={
                        "x": loc.x, "y": loc.y, "z": loc.z + pinned_altitude,
                        "pitch": float(cfg.camera.pitch), "yaw": 0.0,
                    },
                    suspect_pose={"x": loc.x, "y": loc.y, "z": loc.z},
                    boxes=boxes,
                    stats=stats,
                )

                saved += 1
                if saved >= target_frames:
                    break

            elapsed = time.perf_counter() - t0
            log.info(
                "[%s] captured %d frames in %.1fs (class_counts=%s)",
                run_id, saved, elapsed, manifest.class_counts,
            )

            manifest.save(output_dir / "manifest.json")

            return CaptureResult(
                run_id=run_id,
                seed=seed,
                weather=weather,
                frames_saved=saved,
                class_counts=dict(manifest.class_counts),
                skip_counts=dict(manifest.skip_counts),
                duration_s=elapsed,
                output_dir=output_dir,
            )

        finally:
            teardown_pursuit(client, world, tm, actors)


@dataclass
class TrackingCaptureResult:
    run_id: str
    seed: int
    weather: str
    frames_saved: int
    suspect_actor_id: int
    output_dir: Path
    duration_s: float


def run_tracking_capture(
    cfg,
    output_dir: Path,
    *,
    client: carla.Client,
    run_id: str,
    seed: int,
    weather: str,
    target_frames: int,
    min_suspect_visibility: float = 0.05,
    min_visibility_other: float = 0.3,
    min_bbox_pixel: int = 20,
    jpeg_quality: int = 92,
) -> TrackingCaptureResult:
    """Capture a continuous pursuit with per-frame ground-truth tracks.

    Unlike ``run_capture`` (which subsamples and emits YOLO labels for
    detector training), this writes every tick and records per-actor
    ``{actor_id, bbox_px, visibility}`` GT so a tracker can be evaluated
    against consistent actor identities across the sequence.

    The suspect's GT bbox is always emitted even at low visibility (using
    ``min_suspect_visibility``, default 0.05) so continuity can be scored
    through partial occlusions. Other actors use a stricter threshold.

    Output layout::

        <output_dir>/
          images/frame_NNNN.jpg
          tracks.json   # {suspect_actor_id, frames: [{frame, tick, objects: [...]}]}
          manifest.json
    """
    import json
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    actors: list[carla.Actor] = []
    world = ensure_map(client, cfg.carla.map)
    world.set_weather(weather_preset(weather))

    t0 = time.perf_counter()
    saved = 0
    tracks_record: list[dict] = []
    suspect_actor_id = -1

    with synchronous_mode(world, cfg.carla.fixed_delta_seconds):
        tm = client.get_trafficmanager(cfg.carla.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)

        try:
            npcs, remaining = spawn_npcs(world, tm, cfg.scene.npc_count, rng)
            actors.extend(npcs)
            log.info("[%s] spawned %d/%d NPCs", run_id, len(npcs), cfg.scene.npc_count)

            suspect_params = SuspectParams(**dict(cfg.scene.suspect))
            suspect = spawn_reckless_suspect(world, tm, remaining, rng, suspect_params)
            actors.append(suspect)
            suspect_actor_id = suspect.id
            log.info("[%s] spawned suspect %s (id=%d)", run_id, suspect.type_id, suspect_actor_id)

            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(100.0)

            rgb_cam, rgb_q = spawn_aerial_camera(
                world, width=cfg.camera.width, height=cfg.camera.height, fov=cfg.camera.fov,
            )
            actors.append(rgb_cam)
            seg_cam, seg_q = _spawn_instance_seg_camera(
                world, width=cfg.camera.width, height=cfg.camera.height, fov=cfg.camera.fov,
            )
            actors.append(seg_cam)

            # Altitude pinned per D-12.
            pinned_altitude = float(cfg.camera.altitude)

            # Lookup which actor ids are vehicles (cached)
            vehicle_ids: set[int] = set()

            def is_vehicle(aid: int) -> bool:
                if aid in vehicle_ids:
                    return True
                a = world.get_actor(aid)
                if a is not None and a.type_id.startswith("vehicle."):
                    vehicle_ids.add(aid)
                    return True
                return False

            log.info("[%s] capturing %d frames (every tick)…", run_id, target_frames)

            for tick in range(target_frames + 100):   # small buffer for dropped frames
                loc = suspect.get_transform().location
                pose = carla.Transform(
                    carla.Location(loc.x, loc.y, loc.z + pinned_altitude),
                    carla.Rotation(pitch=cfg.camera.pitch, yaw=0, roll=0),
                )
                rgb_cam.set_transform(pose)
                seg_cam.set_transform(pose)

                world.tick()

                try:
                    rgb_img = rgb_q.get(timeout=2.0)
                    seg_img = seg_q.get(timeout=2.0)
                except queue.Empty:
                    continue

                seg_bgr = carla_image_to_bgr(seg_img)
                all_detections = extract_actor_boxes_from_seg(seg_bgr)

                objects: list[dict] = []
                suspect_emitted = False
                for d in all_detections:
                    if not is_vehicle(d.actor_id):
                        continue
                    bw = d.x2 - d.x1 + 1
                    bh = d.y2 - d.y1 + 1
                    if d.actor_id == suspect_actor_id:
                        # Always emit the suspect, gated only on a very loose visibility
                        if d.visibility < min_suspect_visibility:
                            continue
                        suspect_emitted = True
                    else:
                        if bw < min_bbox_pixel or bh < min_bbox_pixel:
                            continue
                        if d.visibility < min_visibility_other:
                            continue
                    objects.append({
                        "actor_id": d.actor_id,
                        "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                        "visibility": round(d.visibility, 4),
                        "pixel_count": d.pixel_count,
                        "is_suspect": d.actor_id == suspect_actor_id,
                    })

                # Save RGB + per-frame tracks
                rgb_bgr = carla_image_to_bgr(rgb_img)
                frame_id = f"frame_{saved:04d}"
                cv2.imwrite(
                    str(img_dir / f"{frame_id}.jpg"),
                    rgb_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                tracks_record.append({
                    "frame": saved,
                    "tick": tick,
                    "suspect_present": suspect_emitted,
                    "objects": objects,
                })
                saved += 1
                if saved >= target_frames:
                    break

            elapsed = time.perf_counter() - t0
            log.info(
                "[%s] captured %d frames in %.1fs (%d frames with suspect visible)",
                run_id, saved, elapsed,
                sum(1 for r in tracks_record if r["suspect_present"]),
            )

            tracks_path = output_dir / "tracks.json"
            with open(tracks_path, "w") as f:
                json.dump({
                    "run_id": run_id,
                    "seed": seed,
                    "weather": weather,
                    "map_name": cfg.carla.map,
                    "suspect_actor_id": suspect_actor_id,
                    "suspect_type_id": suspect.type_id,
                    "n_frames": saved,
                    "frames": tracks_record,
                }, f)

            # Minimal manifest for human scanning
            manifest_path = output_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump({
                    "run_id": run_id,
                    "seed": seed,
                    "weather": weather,
                    "map_name": cfg.carla.map,
                    "suspect_actor_id": suspect_actor_id,
                    "n_frames": saved,
                    "n_suspect_visible_frames": sum(
                        1 for r in tracks_record if r["suspect_present"]
                    ),
                }, f, indent=2)

            return TrackingCaptureResult(
                run_id=run_id,
                seed=seed,
                weather=weather,
                frames_saved=saved,
                suspect_actor_id=suspect_actor_id,
                output_dir=output_dir,
                duration_s=elapsed,
            )

        finally:
            teardown_pursuit(client, world, tm, actors)
