"""
Experiment 05 — capture CARLA pursuit eval holdout.

Runs a single adaptive-altitude pursuit on experiment 04's scene and dumps
paired RGB + YOLO-format label files every Nth tick, plus a reproducibility
manifest. Frames with zero valid labels are skipped entirely.

Output:
  output/eval/carla_eval/
    images/frame_NNNN.jpg
    labels/frame_NNNN.txt
    manifest.json

This set is consumed by experiment 06 (VisDrone fine-tune) as a frozen eval
benchmark — pretrained vs fine-tuned vs later rounds are all scored against
these exact frames so progress is measurable. Do not regenerate casually.
"""

import queue
import random
import time
from pathlib import Path

import carla
import cv2

from skycop.config import load
from skycop.control import AdaptiveAltitudeController, AltitudeConfig
from skycop.cv import (
    CLASS_NAMES,
    DatasetManifest,
    detector_class_for,
    extract_yolo_labels_from_seg,
    write_yolo_label,
)
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


def ensure_map(client, map_name):
    world = client.get_world()
    if map_name in world.get_map().name:
        return world
    print(f"Loading {map_name}...")
    return client.load_world(map_name)


def make_actor_classifier(world):
    """Return a cached actor_id → detector-class-index callable.

    Every detected vehicle maps to the single detector class (index 0).
    The fine-grained fingerprint class (car/van/truck/bus) is still
    available via ``classify_blueprint`` and will be recorded into the
    manifest at fingerprint integration time, not here.
    """
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


def spawn_instance_seg_camera(world, width: int, height: int, fov: int):
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find("sensor.camera.instance_segmentation")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    # Initial spawn transform — overwritten every tick by the main loop.
    transform = carla.Transform(carla.Location(0, 0, 100), carla.Rotation(pitch=-75))
    cam = world.spawn_actor(bp, transform)
    q: queue.Queue[carla.Image] = queue.Queue()
    cam.listen(q.put)
    return cam, q


def run(cfg):
    rng = random.Random(cfg.seed)
    actors: list[carla.Actor] = []
    client = connect(host=cfg.carla.host, port=cfg.carla.port)
    world = ensure_map(client, cfg.carla.map)

    out_dir = Path(cfg.eval_capture.output_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest(
        seed=cfg.seed,
        fixed_delta_seconds=cfg.carla.fixed_delta_seconds,
        map_name=cfg.carla.map,
        class_names=list(CLASS_NAMES),
        min_pixel=int(cfg.eval_capture.min_bbox_pixel),
        min_visibility=float(cfg.eval_capture.min_visibility),
    )

    with synchronous_mode(world, cfg.carla.fixed_delta_seconds):
        tm = client.get_trafficmanager(cfg.carla.tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(cfg.seed)

        try:
            npcs, remaining = spawn_npcs(world, tm, cfg.scene.npc_count, rng)
            actors.extend(npcs)
            print(f"Spawned {len(npcs)}/{cfg.scene.npc_count} NPCs")

            suspect_params = SuspectParams(**dict(cfg.scene.suspect))
            suspect = spawn_reckless_suspect(world, tm, remaining, rng, suspect_params)
            actors.append(suspect)
            print(f"Spawned suspect {suspect.type_id} (id={suspect.id})")

            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(100.0)

            rgb_cam, rgb_q = spawn_aerial_camera(
                world,
                width=cfg.camera.width, height=cfg.camera.height, fov=cfg.camera.fov,
            )
            actors.append(rgb_cam)

            seg_cam, seg_q = spawn_instance_seg_camera(
                world,
                width=cfg.camera.width, height=cfg.camera.height, fov=cfg.camera.fov,
            )
            actors.append(seg_cam)

            altitude_ctrl = AdaptiveAltitudeController(world, AltitudeConfig(**dict(cfg.altitude)))
            classify = make_actor_classifier(world)

            target_frames = int(cfg.eval_capture.target_frames)
            max_ticks = int(cfg.eval_capture.max_ticks)
            subsample = int(cfg.eval_capture.subsample_every)
            jpeg_q = int(cfg.eval_capture.jpeg_quality)

            saved = 0
            print(f"Capturing up to {target_frames} frames (max {max_ticks} ticks)…")
            t0 = time.time()

            for tick in range(max_ticks):
                loc = suspect.get_transform().location
                target_z, _ = altitude_ctrl.step(loc.x, loc.y)
                pose = carla.Transform(
                    carla.Location(loc.x, loc.y, target_z),
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

                if tick % subsample != 0:
                    continue

                seg_bgr = carla_image_to_bgr(seg_img)
                boxes, stats = extract_yolo_labels_from_seg(
                    seg_bgr,
                    classify,
                    min_pixel=int(cfg.eval_capture.min_bbox_pixel),
                    min_visibility=float(cfg.eval_capture.min_visibility),
                )
                if not boxes:
                    continue

                rgb_bgr = carla_image_to_bgr(rgb_img)
                frame_id = f"frame_{saved:04d}"
                cv2.imwrite(
                    str(img_dir / f"{frame_id}.jpg"),
                    rgb_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_q],
                )
                write_yolo_label(lbl_dir / f"{frame_id}.txt", boxes)

                manifest.record_frame(
                    index=saved,
                    tick=tick,
                    camera_pose={
                        "x": loc.x, "y": loc.y, "z": target_z,
                        "pitch": float(cfg.camera.pitch), "yaw": 0.0,
                    },
                    suspect_pose={"x": loc.x, "y": loc.y, "z": loc.z},
                    boxes=boxes,
                    stats=stats,
                )

                saved += 1
                if saved % 25 == 0:
                    print(f"  {saved}/{target_frames} …")
                if saved >= target_frames:
                    break

            elapsed = time.time() - t0
            print(f"Captured {saved} frames in {elapsed:.1f}s ({saved / elapsed:.2f} fps)")

            manifest.save(out_dir / "manifest.json")
            print(f"Manifest  → {out_dir / 'manifest.json'}")
            print(f"Class counts: {manifest.class_counts}")
            print(f"Skip counts:  {manifest.skip_counts}")

        finally:
            destroy_all(actors)
            try:
                tm.set_synchronous_mode(False)
                tm.set_hybrid_physics_mode(False)
            except Exception:
                pass


def main():
    cfg = load("default", "altitude", "eval_capture")
    run(cfg)


if __name__ == "__main__":
    main()
