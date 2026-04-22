"""Adaptive altitude control for the aerial drone camera — SIM-11..14.

Two layers:

- `compute_target()` is a pure function (no CARLA) that takes the current
  environment observation and returns the next target Z. Directly unit-testable.
- `AdaptiveAltitudeController` wraps it with CARLA raycasting for obstacle
  detection and rooftop clearance.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import carla

log = logging.getLogger(__name__)


@dataclass
class AltitudeConfig:
    min_m: float = 10.0                  # SIM-13 floor
    max_m: float = 60.0                  # SIM-13 ceiling
    open_target_m: float = 15.0          # SIM-11
    urban_target_m: float = 40.0         # SIM-12
    lateral_scan_radius_m: float = 20.0  # SIM-12 trigger
    rooftop_clearance_m: float = 12.0    # SIM-14
    lateral_scan_rays: int = 8
    smoothing: float = 0.2               # IIR factor on the output Z (0 = snap, 1 = frozen)


@dataclass
class Observation:
    """Single-tick environment scan result."""
    building_near: bool
    rooftop_z: float | None  # top of any building directly below, else None


def compute_target(
    cfg: AltitudeConfig,
    obs: Observation,
    previous_z: float | None,
) -> float:
    """Pure target-Z calculator. No CARLA calls — directly unit-testable."""
    target = cfg.urban_target_m if obs.building_near else cfg.open_target_m
    if obs.rooftop_z is not None:
        target = max(target, obs.rooftop_z + cfg.rooftop_clearance_m)
    target = max(cfg.min_m, min(cfg.max_m, target))
    if previous_z is None:
        return target
    s = cfg.smoothing
    return s * previous_z + (1.0 - s) * target


class AdaptiveAltitudeController:
    """CARLA-integrated altitude controller using world raycasting.

    Optional ``trace_path`` makes every ``step()`` append a JSONL record
    (``{tick, target, current_z, building_near, rooftop_z, raycast_z,
    lateral_rays_hit}``). Pure instrumentation — no behaviour change. Intended
    to let us see what altitude actually does under live-mission conditions
    before we change ``compute_target``.
    """

    def __init__(
        self,
        world: carla.World,
        config: AltitudeConfig | None = None,
        trace_path: Path | None = None,
    ) -> None:
        self.world = world
        self.cfg = config or AltitudeConfig()
        self._current_z: float | None = None
        self._tick = 0
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._trace_fh = None
        if self._trace_path is not None:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_fh = open(self._trace_path, "w", buffering=1)  # line-buffered
            log.info("altitude trace → %s", self._trace_path)

    def step(self, x: float, y: float) -> tuple[float, Observation]:
        """Return (next_target_z, observation) for the drone above (x, y).

        The raycast origin is the controller's own previous Z — the caller is
        assumed to set the drone transform to the returned Z each tick.
        """
        raycast_z = self._current_z if self._current_z is not None else self.cfg.open_target_m
        obs, lateral_rays_hit = self._scan(x, y, raycast_z)
        target = _compute_target_raw(self.cfg, obs)
        self._current_z = compute_target(self.cfg, obs, self._current_z)
        if self._trace_fh is not None:
            rec = {
                "tick": self._tick,
                "target": round(target, 4),
                "current_z": round(float(self._current_z), 4),
                "building_near": bool(obs.building_near),
                "rooftop_z": round(obs.rooftop_z, 4) if obs.rooftop_z is not None else None,
                "raycast_z": round(raycast_z, 4),
                "lateral_rays_hit": int(lateral_rays_hit),
            }
            self._trace_fh.write(json.dumps(rec) + "\n")
        self._tick += 1
        return self._current_z, obs

    def close(self) -> None:
        """Close any open trace file. Idempotent."""
        if self._trace_fh is not None:
            self._trace_fh.close()
            self._trace_fh = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _scan(self, x: float, y: float, z: float) -> tuple[Observation, int]:
        building_near = False
        rays_hit = 0
        r = self.cfg.lateral_scan_radius_m
        for i in range(self.cfg.lateral_scan_rays):
            angle = 2.0 * math.pi * i / self.cfg.lateral_scan_rays
            start = carla.Location(x, y, z)
            end = carla.Location(x + r * math.cos(angle), y + r * math.sin(angle), z)
            for hit in self.world.cast_ray(start, end):
                if hit.label == carla.CityObjectLabel.Buildings:
                    rays_hit += 1
                    building_near = True
                    break  # inner loop — one hit per ray is enough

        rooftop_z: float | None = None
        down_start = carla.Location(x, y, z)
        down_end = carla.Location(x, y, 0.0)
        for hit in self.world.cast_ray(down_start, down_end):
            if hit.label == carla.CityObjectLabel.Buildings:
                zh = hit.location.z
                if rooftop_z is None or zh > rooftop_z:
                    rooftop_z = zh

        return Observation(building_near=building_near, rooftop_z=rooftop_z), rays_hit


def _compute_target_raw(cfg: AltitudeConfig, obs: Observation) -> float:
    """Target before smoothing — exposed for the trace so we can see what the
    controller is chasing tick-to-tick, not just where it ended up."""
    target = cfg.urban_target_m if obs.building_near else cfg.open_target_m
    if obs.rooftop_z is not None:
        target = max(target, obs.rooftop_z + cfg.rooftop_clearance_m)
    return max(cfg.min_m, min(cfg.max_m, target))
