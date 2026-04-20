"""CARLA world connection helpers.

- `connect()` — one-liner that reads CARLA_HOST/CARLA_PORT from the environment.
- `synchronous_mode()` — context manager that flips the world to synchronous
  mode at the given fixed timestep and restores the original settings on exit.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import carla


def connect(
    host: str | None = None,
    port: int | None = None,
    timeout: float = 60.0,
) -> carla.Client:
    """Return a connected CARLA client. Host/port default to env vars."""
    host = host or os.environ.get("CARLA_HOST", "carla-server")
    port = port or int(os.environ.get("CARLA_PORT", 2000))
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


@contextmanager
def synchronous_mode(
    world: carla.World,
    fixed_delta_seconds: float = 0.05,
) -> Iterator[carla.WorldSettings]:
    """Enable synchronous CARLA mode for the duration of the block.

    Restores the original settings on exit, even if the block raises.
    """
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    try:
        yield settings
    finally:
        try:
            world.apply_settings(original)
        except Exception:
            pass
