"""CARLA simulation wrappers — world connection, cameras, actor spawning."""

from skycop.sim.actors import (
    SuspectParams,
    destroy_all,
    four_wheel_blueprints,
    spawn_npcs,
    spawn_reckless_suspect,
    teardown_pursuit,
)
from skycop.sim.aerial_camera import carla_image_to_bgr, spawn_aerial_camera
from skycop.sim.carla_env import connect, synchronous_mode

__all__ = [
    "connect",
    "synchronous_mode",
    "spawn_aerial_camera",
    "carla_image_to_bgr",
    "spawn_npcs",
    "spawn_reckless_suspect",
    "SuspectParams",
    "destroy_all",
    "four_wheel_blueprints",
    "teardown_pursuit",
]
