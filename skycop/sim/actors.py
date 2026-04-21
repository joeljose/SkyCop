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


def destroy_all(
    actors: list[carla.Actor],
    client: carla.Client | None = None,
) -> None:
    """Destroy actors in reverse-spawn order (sensors before vehicles).

    If a client is provided, destroys are submitted via ``apply_batch_sync``
    — a single atomic RPC rather than N per-actor calls. This avoids the
    registry-miss race where the Traffic Manager's async pass touches an
    actor being destroyed on another thread (carla_caveats §6).

    Sensor ``listen()`` callbacks are stopped first in either path so no
    callback fires mid-destroy.
    """
    # Stop sensor listeners regardless of destroy path — no callbacks mid-teardown.
    for a in actors:
        try:
            if hasattr(a, "is_listening") and a.is_listening:
                a.stop()
        except Exception:
            pass

    if client is not None:
        try:
            client.apply_batch_sync(
                [carla.command.DestroyActor(a.id) for a in reversed(actors)],
                True,
            )
            return
        except Exception:
            # Fall through to per-actor destroy if batch fails.
            pass

    for a in reversed(actors):
        try:
            a.destroy()
        except Exception:
            pass


def teardown_pursuit(
    client: carla.Client,
    world: carla.World,
    tm: carla.TrafficManager,
    actors: list[carla.Actor],
) -> None:
    """Tear down a pursuit scene cleanly, without CARLA SIGABRT on exit.

    Sequence matches CARLA's own ``generate_traffic.py`` recommendation plus
    the empirical finding that TM hybrid-physics teardown must run *after*
    vehicles leave the TM's management, not before.

    Steps:

    1. Stop all sensor listeners so no callback fires during destroy.
    2. Disable TM hybrid physics mode *while* vehicles are still attached —
       the TM iterates *its currently-registered* vehicles, not the whole
       world. This avoids the registry-miss race against CARLA-auto-destroyed
       actors.
    3. ``SetAutopilot(False)`` on every vehicle via ``apply_batch_sync`` —
       detaches vehicles from the TM so subsequent teardown has no
       vehicles to race against.
    4. Switch the world and TM out of synchronous mode — batch destroys
       need async to complete without requiring a tick.
    5. ``DestroyActor`` for all actors via ``apply_batch_sync`` — one
       atomic RPC, no per-actor race.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    # 1. Stop sensor listens.
    _log.debug("teardown: stopping %d sensor listeners", sum(
        1 for a in actors if hasattr(a, "is_listening") and a.is_listening
    ))
    for a in actors:
        try:
            if hasattr(a, "is_listening") and a.is_listening:
                a.stop()
        except Exception:
            pass

    # 2. Disable hybrid physics *while* vehicles are still registered to TM.
    _log.debug("teardown: disabling TM hybrid physics")
    try:
        tm.set_hybrid_physics_mode(False)
    except Exception as e:
        _log.debug("teardown: hybrid_physics(False) raised: %s", e)

    # 3. Detach vehicles from TM oversight (still in sync mode so this is deterministic).
    vehicles = [a for a in actors if a.type_id.startswith("vehicle.")]
    if vehicles:
        _log.debug("teardown: detaching %d vehicles from TM (SetAutopilot False)", len(vehicles))
        try:
            client.apply_batch_sync(
                [carla.command.SetAutopilot(v.id, False) for v in vehicles],
                True,
            )
        except Exception as e:
            _log.debug("teardown: SetAutopilot batch raised: %s", e)

    # 4. Switch world + TM to async so batch destroy can complete without a tick.
    _log.debug("teardown: switching to async mode")
    try:
        tm.set_synchronous_mode(False)
    except Exception as e:
        _log.debug("teardown: tm.set_synchronous_mode(False) raised: %s", e)
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    except Exception as e:
        _log.debug("teardown: world.apply_settings(async) raised: %s", e)

    # 5. Atomic batch destroy.
    _log.debug("teardown: batch-destroying %d actors", len(actors))
    try:
        client.apply_batch_sync(
            [carla.command.DestroyActor(a.id) for a in reversed(actors)],
            True,
        )
    except Exception as e:
        _log.debug("teardown: destroy batch raised (%s); falling back to per-actor", e)
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass

    _log.debug("teardown: done")
