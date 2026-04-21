"""
Experiment 03 — Suspect + Traffic

Closes the Environment milestone:
  - Town10HD_Opt loaded
  - 50 NPC vehicles on autopilot via Traffic Manager
  - 1 suspect vehicle driving recklessly (speeding, running lights, weaving)
  - Aerial top-down camera following the suspect at fixed altitude
  - Synchronous mode at 20 FPS, seeded for reproducibility (NFR-07)

Open http://localhost:5000 to watch the live top-down feed.
"""

import logging
import queue
import random
import time

import carla

from skycop.config import load
from skycop.dashboard import MJPEGServer
from skycop.logs import setup_logging
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

log = logging.getLogger("exp03")


def ensure_map(client, map_name):
    world = client.get_world()
    if map_name in world.get_map().name:
        return world
    log.info("loading map %s", map_name)
    return client.load_world(map_name)


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
            log.info("spawned %d/%d NPCs", len(npcs), cfg.scene.npc_count)

            suspect_params = SuspectParams(**dict(cfg.scene.suspect))
            suspect = spawn_reckless_suspect(world, tm, remaining, rng, suspect_params)
            actors.append(suspect)
            log.info("spawned suspect %s (id=%d)", suspect.type_id, suspect.id)

            camera, img_queue = spawn_aerial_camera(
                world,
                width=cfg.camera.width,
                height=cfg.camera.height,
                fov=cfg.camera.fov,
            )
            actors.append(camera)

            log.info("running — http://localhost:5000")

            while True:
                loc = suspect.get_transform().location
                camera.set_transform(carla.Transform(
                    carla.Location(loc.x, loc.y, loc.z + cfg.camera.altitude),
                    carla.Rotation(pitch=cfg.camera.pitch, yaw=0, roll=0),
                ))

                world.tick()

                try:
                    image = img_queue.get(timeout=1.0)
                    server.push(carla_image_to_bgr(image))
                except queue.Empty:
                    pass

        except KeyboardInterrupt:
            pass
        finally:
            teardown_pursuit(client, world, tm, actors)
            log.info("stopped")


def main():
    setup_logging()
    cfg = load("default")
    server = MJPEGServer(
        title="SkyCop — Experiment 03",
        hud=f"Experiment 03 · fixed-altitude follow @ {cfg.camera.altitude:.0f}m",
    )
    server.start(port=5000)
    time.sleep(0.3)
    run(server, cfg)


if __name__ == "__main__":
    main()
