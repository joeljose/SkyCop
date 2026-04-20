"""Aerial RGB camera helpers — spawn, grab frames, convert to BGR."""

from __future__ import annotations

import queue

import carla
import numpy as np


def spawn_aerial_camera(
    world: carla.World,
    width: int = 1280,
    height: int = 720,
    fov: int = 90,
    attach_to: carla.Actor | None = None,
    transform: carla.Transform | None = None,
) -> tuple[carla.Sensor, queue.Queue[carla.Image]]:
    """Spawn an RGB camera and return (sensor, frame_queue).

    If `attach_to` is None, the camera is free-floating — the caller updates
    its transform each tick. If given, the camera rides on the attached actor.
    """
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))

    if transform is None:
        transform = carla.Transform(
            carla.Location(0, 0, 100),
            carla.Rotation(pitch=-90),
        )

    camera = world.spawn_actor(cam_bp, transform, attach_to=attach_to)
    q: queue.Queue[carla.Image] = queue.Queue()
    camera.listen(q.put)
    return camera, q


def carla_image_to_bgr(image: carla.Image) -> np.ndarray:
    """Convert a CARLA BGRA image to a contiguous BGR numpy array."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
    return np.ascontiguousarray(arr)
