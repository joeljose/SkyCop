"""
Experiment 01 — Hello CARLA World

Connects to the CARLA server, logs world info, spawns a vehicle with
an RGB camera, captures one frame using synchronous mode, and saves it.

Key concepts:
  - Client/World connection (via skycop.sim.connect)
  - Blueprint library
  - Spawning actors (vehicle + sensor)
  - Synchronous mode with world.tick()
  - Proper actor cleanup (important: CARLA leaks GPU memory on sensors)
"""

import logging
import os
import queue

import carla

from skycop.logs import setup_logging
from skycop.sim import connect, synchronous_mode

OUTPUT_DIR = "/app/output"

log = logging.getLogger("exp01")


def main():
    setup_logging()
    actors = []

    try:
        client = connect()
        world = client.get_world()

        log.info("connected to CARLA")
        log.info("map:  %s", world.get_map().name)
        log.info("maps: %s", client.get_available_maps())

        with synchronous_mode(world, fixed_delta_seconds=0.05):
            bp_lib = world.get_blueprint_library()
            vehicle_bp = bp_lib.find("vehicle.tesla.model3")
            spawn_points = world.get_map().get_spawn_points()

            vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
            if vehicle is None:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[1])
            if vehicle is None:
                raise RuntimeError("Could not spawn vehicle at any point")

            actors.append(vehicle)
            log.info("spawned: %s (id=%d)", vehicle.type_id, vehicle.id)

            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "1280")
            camera_bp.set_attribute("image_size_y", "720")
            camera_bp.set_attribute("fov", "90")

            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            actors.append(camera)

            image_queue: queue.Queue[carla.Image] = queue.Queue()
            camera.listen(image_queue.put)

            for _ in range(5):
                world.tick()

            image = image_queue.get(timeout=5.0)

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = f"{OUTPUT_DIR}/01_hello_world.png"
            image.save_to_disk(out_path)
            log.info("saved: %s (%dx%d)", out_path, image.width, image.height)

    finally:
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception:
                pass

    log.info("done — CARLA setup is working")


if __name__ == "__main__":
    main()
