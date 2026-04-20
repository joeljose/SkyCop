"""
Experiment 04 — Adaptive Altitude

Builds on experiment 03 and layers in SIM-11..14:
  - Aerial camera height adapts each tick using world.cast_ray()
  - Lateral ring of rays detects buildings within 20m → climb
  - Downward ray measures rooftop height → ensure 12m clearance
  - Hard-clamped to [10m, 60m]
  - TM hybrid-physics mode anchored on the suspect (role_name='hero')
    to keep FPS sane on a 6GB VRAM GPU (carla_caveats §10)

Config via OmegaConf:
  configs/default.yaml + configs/altitude.yaml

Open http://localhost:5000 for the top-down feed with altitude HUD.
"""

import queue
import random
import time

import carla
import cv2

from skycop.config import load
from skycop.control import AdaptiveAltitudeController, AltitudeConfig
from skycop.dashboard import MJPEGServer
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


def overlay_hud(frame, altitude_m: float, urban: bool, rooftop_z: float | None) -> None:
    """Draw altitude HUD on the frame in-place."""
    txt_color = (0, 255, 255)  # BGR yellow
    cv2.putText(frame, f"ALT {altitude_m:5.1f}m", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, txt_color, 2, cv2.LINE_AA)
    state = "URBAN (climb)" if urban else "OPEN (low)"
    cv2.putText(frame, state, (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 1, cv2.LINE_AA)
    if rooftop_z is not None:
        cv2.putText(frame, f"rooftop {rooftop_z:.1f}m", (20, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 1, cv2.LINE_AA)


def run(server: MJPEGServer, cfg):
    rng = random.Random(cfg.seed)
    actors: list[carla.Actor] = []
    client = connect(host=cfg.carla.host, port=cfg.carla.port)
    world = ensure_map(client, cfg.carla.map)

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
            print(f"Spawned suspect {suspect.type_id} (id={suspect.id}) role={suspect_params.role_name}")

            # Hybrid physics anchored on the hero (the suspect) — carla_caveats §10.
            # Distant NPCs get simplified physics, freeing GPU for rendering.
            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(100.0)

            camera, img_queue = spawn_aerial_camera(
                world,
                width=cfg.camera.width,
                height=cfg.camera.height,
                fov=cfg.camera.fov,
            )
            actors.append(camera)

            alt_cfg = AltitudeConfig(**dict(cfg.altitude))
            altitude_ctrl = AdaptiveAltitudeController(world, alt_cfg)

            print("Running — http://localhost:5000")

            while True:
                loc = suspect.get_transform().location
                target_z, obs = altitude_ctrl.step(loc.x, loc.y)
                camera.set_transform(carla.Transform(
                    carla.Location(loc.x, loc.y, target_z),
                    carla.Rotation(pitch=cfg.camera.pitch, yaw=0, roll=0),
                ))

                world.tick()

                try:
                    image = img_queue.get(timeout=1.0)
                    frame = carla_image_to_bgr(image)
                    overlay_hud(frame, target_z, obs.building_near, obs.rooftop_z)
                    server.push(frame)
                except queue.Empty:
                    pass

        except KeyboardInterrupt:
            pass
        finally:
            destroy_all(actors)
            try:
                tm.set_synchronous_mode(False)
                tm.set_hybrid_physics_mode(False)
            except Exception:
                pass
            print("Stopped.")


def main():
    cfg = load("default", "altitude")
    server = MJPEGServer(
        title="SkyCop — Experiment 04",
        hud=f"Experiment 04 · adaptive altitude [{cfg.altitude.min_m:.0f}–{cfg.altitude.max_m:.0f}m]",
    )
    server.start(port=5000)
    time.sleep(0.3)
    run(server, cfg)


if __name__ == "__main__":
    main()
