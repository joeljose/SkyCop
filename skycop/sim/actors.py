"""Helpers for spawning NPCs and the reckless suspect vehicle.

All spawn helpers filter to 4-wheeled vehicles — motorcycles/bicycles don't
support autopilot (see docs/carla_caveats.md §7).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import carla


@dataclass
class SuspectParams:
    """Reckless driving knobs — FR-08.

    Speed convention is CARLA's: negative = faster than limit (caveat §9).
    """
    speed_over_pct: float = -80.0
    ignore_lights_pct: float = 100.0
    ignore_signs_pct: float = 100.0
    lane_change_pct: float = 50.0
    follow_dist_m: float = 1.0
    role_name: str = "hero"  # "hero" lets TM hybrid physics anchor on the suspect


def four_wheel_blueprints(bp_lib: carla.BlueprintLibrary) -> list[carla.ActorBlueprint]:
    out = []
    for bp in bp_lib.filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4:
            out.append(bp)
    return out


def spawn_npcs(
    world: carla.World,
    tm: carla.TrafficManager,
    count: int,
    rng: random.Random,
) -> tuple[list[carla.Actor], list[carla.Transform]]:
    """Spawn up to `count` NPC vehicles on autopilot. Returns (npcs, unused_spawn_points)."""
    bp_lib = world.get_blueprint_library()
    vehicle_bps = four_wheel_blueprints(bp_lib)
    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    npcs: list[carla.Actor] = []
    idx = 0
    while len(npcs) < count and idx < len(spawn_points):
        sp = spawn_points[idx]
        idx += 1
        bp = rng.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "npc")
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
    return npcs, spawn_points[idx:]


def spawn_reckless_suspect(
    world: carla.World,
    tm: carla.TrafficManager,
    spawn_points: list[carla.Transform],
    rng: random.Random,
    params: SuspectParams | None = None,
) -> carla.Actor:
    """Spawn a single suspect with reckless TM knobs applied."""
    params = params or SuspectParams()
    bp_lib = world.get_blueprint_library()
    bp = rng.choice(four_wheel_blueprints(bp_lib))
    if bp.has_attribute("color"):
        bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
    bp.set_attribute("role_name", params.role_name)

    suspect = None
    for sp in spawn_points:
        suspect = world.try_spawn_actor(bp, sp)
        if suspect:
            break
    if suspect is None:
        raise RuntimeError("No free spawn point left for suspect")

    suspect.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(suspect, params.speed_over_pct)
    tm.ignore_lights_percentage(suspect, params.ignore_lights_pct)
    tm.ignore_signs_percentage(suspect, params.ignore_signs_pct)
    tm.random_left_lanechange_percentage(suspect, params.lane_change_pct)
    tm.random_right_lanechange_percentage(suspect, params.lane_change_pct)
    tm.distance_to_leading_vehicle(suspect, params.follow_dist_m)
    return suspect


def destroy_all(actors: list[carla.Actor]) -> None:
    """Destroy actors in reverse order (sensors before vehicles) — caveat §6."""
    for a in reversed(actors):
        try:
            if hasattr(a, "is_listening") and a.is_listening:
                a.stop()
            a.destroy()
        except Exception:
            pass
