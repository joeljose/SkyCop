"""Scout Town10HD spawn points for FSM parking destinations + validate TM.set_path().

Two things this script answers before we commit to the destination-bound PARKING
state in Mission v0a (issue #44):

1. **Scout** — enumerate Town10HD spawn points, with their (x, y, z, yaw)
   values, so we can hand-pick 2-3 parking-lot-ish and 2-3 roadside-ish
   candidates for ``configs/suspect.yaml``.

2. **set_path feasibility** — spawn a vehicle, call
   ``TrafficManager.set_path(actor, [target_location])``, tick in sync mode,
   and report whether the vehicle actually reaches the target within a
   timeout. If no, escalate to a custom waypoint-PID controller (tracked
   separately).

Run inside the client container::

    make exp N=11
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time

import carla

from skycop.logs import setup_logging
from skycop.sim import (
    connect,
    four_wheel_blueprints,
    synchronous_mode,
    teardown_pursuit,
)
from skycop.vendor.carla_agents import GlobalRoutePlanner, LocalPlanner

log = logging.getLogger(__name__)


def _dist2d(loc: carla.Location, target: carla.Location) -> float:
    return math.hypot(loc.x - target.x, loc.y - target.y)


def _scout_spawn_points(world: carla.World, n: int) -> None:
    """Print the first ``n`` spawn points with their XY — human picks candidates."""
    sps = world.get_map().get_spawn_points()
    log.info("Town10HD has %d spawn points; dumping first %d", len(sps), n)
    print(f"{'idx':>4}  {'x':>9}  {'y':>9}  {'z':>7}  {'yaw':>7}")
    for i, sp in enumerate(sps[:n]):
        loc = sp.location
        rot = sp.rotation
        print(f"{i:>4d}  {loc.x:>9.2f}  {loc.y:>9.2f}  {loc.z:>7.2f}  {rot.yaw:>7.1f}")


def _dense_route(world: carla.World, source: carla.Location, dest: carla.Location,
                 sampling_resolution_m: float = 2.0) -> list[carla.Location]:
    """Dense waypoint sequence from ``source`` to ``dest`` via vendored
    ``GlobalRoutePlanner`` (CARLA's own BFS/A* router).

    Greedy ``waypoint.next()``-walking was tried first and produced
    multi-kilometre U-shaped loops for 100 m targets (empirically observed
    — see issue #44). The proper graph search builds a road-network
    topology once and searches over it.
    """
    from skycop.vendor.carla_agents import GlobalRoutePlanner
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution_m)
    route = grp.trace_route(source, dest)
    return [wp.transform.location for wp, _opt in route]


def _test_custom_controller(
    client: carla.Client,
    world: carla.World,
    source_idx: int,
    target_idx: int,
    timeout_s: float,
    reach_distance_m: float,
    reach_speed_mps: float,
    fixed_dt: float,
    tm_port: int,
    sampling_resolution_m: float = 2.0,
) -> None:
    """Drive a vehicle with vendored CARLA LocalPlanner (autopilot OFF).

    Validates option C4 of issue #44: does CARLA's own LocalPlanner +
    VehiclePIDController drive the suspect to a destination via
    GlobalRoutePlanner's route? This is the battle-tested CARLA solution;
    our hand-rolled pure-pursuit ``WaypointFollower`` oscillated at tight
    turns, so we vendor the proven controller instead.
    """
    sps = world.get_map().get_spawn_points()
    source = sps[source_idx]
    target_loc = sps[target_idx].location
    log.info("[C4 LocalPlanner] source #%d → target #%d", source_idx, target_idx)

    actors: list[carla.Actor] = []
    with synchronous_mode(world, fixed_dt):
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(True)

        bp_lib = world.get_blueprint_library()
        bp = random.Random(0).choice(four_wheel_blueprints(bp_lib))
        bp.set_attribute("role_name", "hero")
        vehicle = world.try_spawn_actor(bp, source)
        if vehicle is None:
            raise RuntimeError("Failed to spawn source vehicle")
        actors.append(vehicle)
        # Autopilot OFF — LocalPlanner issues apply_control() each tick.
        vehicle.set_autopilot(False)

        wmap = world.get_map()
        grp = GlobalRoutePlanner(wmap, sampling_resolution_m)
        route = grp.trace_route(source.location, target_loc)
        log.info("[C4] GlobalRoutePlanner produced %d waypoints", len(route))

        planner = LocalPlanner(
            vehicle,
            opt_dict={
                "target_speed": 22.0,          # km/h cruise
                "dt": fixed_dt,
                "sampling_radius": sampling_resolution_m,
            },
        )
        planner.set_global_plan(route, stop_waypoint_creation=True)

        t0 = time.perf_counter()
        ticks = 0
        reached = False
        dist = float("inf")
        speed = 0.0
        while ticks * fixed_dt < timeout_s:
            world.tick()
            ticks += 1
            ctrl = planner.run_step()
            vehicle.apply_control(ctrl)

            loc = vehicle.get_location()
            v = vehicle.get_velocity()
            speed = math.hypot(v.x, v.y)
            dist = math.hypot(loc.x - target_loc.x, loc.y - target_loc.y)
            if ticks % 20 == 0:
                log.info(
                    "[C4] tick %4d  t=%5.1fs  pos=(%.1f,%.1f)  dist=%.2fm  speed=%.2fm/s "
                    "throt=%.2f steer=%+.2f brake=%.2f done=%s",
                    ticks, ticks * fixed_dt, loc.x, loc.y, dist, speed,
                    ctrl.throttle, ctrl.steer, ctrl.brake,
                    planner.done(),
                )
            if dist <= reach_distance_m and speed <= reach_speed_mps:
                reached = True
                break
            if planner.done() and dist <= reach_distance_m + 3.0:
                reached = True
                break

        wall = time.perf_counter() - t0
        if reached:
            log.info("[C4] ✅ REACHED in %.1fs sim-time (%.1fs wall) — %d ticks · final dist=%.2fm",
                     ticks * fixed_dt, wall, ticks, dist)
        else:
            log.warning("[C4] ❌ DID NOT REACH in %.1fs — final dist=%.2fm speed=%.2fm/s",
                        timeout_s, dist, speed)

    teardown_pursuit(client, world, tm, actors)


def _test_set_path(
    client: carla.Client,
    world: carla.World,
    source_idx: int,
    target_idx: int,
    timeout_s: float,
    reach_distance_m: float,
    reach_speed_mps: float,
    fixed_dt: float,
    tm_port: int,
    dense_route: bool = True,
    sampling_resolution_m: float = 2.0,
) -> None:
    """Spawn a vehicle, set_path to target, tick until reached or timeout."""
    sps = world.get_map().get_spawn_points()
    if source_idx >= len(sps) or target_idx >= len(sps):
        raise IndexError(f"spawn indices out of range (have {len(sps)})")
    source = sps[source_idx]
    target_tf = sps[target_idx]
    target_loc = target_tf.location
    log.info("source spawn #%d (%.1f, %.1f) → target #%d (%.1f, %.1f)",
             source_idx, source.location.x, source.location.y,
             target_idx, target_loc.x, target_loc.y)

    actors: list[carla.Actor] = []
    with synchronous_mode(world, fixed_dt):
        tm = client.get_trafficmanager(tm_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(42)

        bp_lib = world.get_blueprint_library()
        bp = random.Random(0).choice(four_wheel_blueprints(bp_lib))
        bp.set_attribute("role_name", "hero")
        vehicle = world.try_spawn_actor(bp, source)
        if vehicle is None:
            raise RuntimeError("Failed to spawn source vehicle")
        actors.append(vehicle)

        vehicle.set_autopilot(True, tm.get_port())
        # Prevent the "stopped at a red light forever in an empty world" trap —
        # an empty Town has no NPC flow to clear intersections.
        tm.ignore_lights_percentage(vehicle, 100.0)
        tm.ignore_signs_percentage(vehicle, 100.0)
        tm.vehicle_percentage_speed_difference(vehicle, 0.0)
        tm.set_hybrid_physics_mode(True)
        tm.set_hybrid_physics_radius(100.0)

        if dense_route:
            path = _dense_route(world, source.location, target_loc, sampling_resolution_m)
            log.info("GlobalRoutePlanner produced %d waypoints over %.1fm-spaced route",
                     len(path), sampling_resolution_m)
        else:
            path = [target_loc]
            log.info("using single-target path (one waypoint)")
        tm.set_path(vehicle, path)
        log.info("set_path called; ticking for up to %.1fs", timeout_s)

        t0 = time.perf_counter()
        ticks = 0
        reached = False
        elapsed = 0.0
        dist = float("inf")
        speed = 0.0
        while elapsed < timeout_s:
            world.tick()
            ticks += 1
            elapsed = ticks * fixed_dt
            loc = vehicle.get_location()
            v = vehicle.get_velocity()
            speed = math.hypot(v.x, v.y)
            dist = _dist2d(loc, target_loc)
            if ticks % 20 == 0:
                log.info(
                    "tick %4d  t=%5.1fs  pos=(%.1f,%.1f)  dist=%.2fm  speed=%.2fm/s",
                    ticks, elapsed, loc.x, loc.y, dist, speed,
                )
            if dist <= reach_distance_m and speed <= reach_speed_mps:
                reached = True
                break

        wall_elapsed = time.perf_counter() - t0
        if reached:
            log.info("✅ REACHED target in %.1fs sim-time (%.1fs wall) — %d ticks · final dist=%.2fm",
                     elapsed, wall_elapsed, ticks, dist)
        else:
            log.warning("❌ DID NOT REACH target in %.1fs — final dist=%.2fm speed=%.2fm/s",
                        elapsed, dist, speed)
            log.warning("    TM may have deviated; check `make exp N=11` output above")

    teardown_pursuit(client, world, tm, actors)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Parking scout + set_path feasibility")
    parser.add_argument("--scout", type=int, default=40,
                        help="Print first N spawn points for manual selection (default 40)")
    parser.add_argument("--source", type=int, default=0, help="Source spawn-point index")
    parser.add_argument("--target", type=int, default=10, help="Target spawn-point index")
    parser.add_argument("--timeout", type=float, default=90.0, help="Set_path sim-time timeout (s)")
    parser.add_argument("--reach-distance", type=float, default=3.0)
    parser.add_argument("--reach-speed", type=float, default=0.5)
    parser.add_argument("--skip-scout", action="store_true")
    parser.add_argument("--skip-feasibility", action="store_true")
    parser.add_argument("--sparse-path", action="store_true",
                        help="Test naive single-waypoint set_path instead of dense GlobalRoutePlanner")
    parser.add_argument("--sampling-resolution", type=float, default=2.0,
                        help="Metres between waypoints in the dense path (default 2.0)")
    parser.add_argument("--custom-controller", action="store_true",
                        help="Option C3: drive via WaypointFollower (autopilot off) instead of TM")
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tm-port", type=int, default=8000)
    args = parser.parse_args()

    client = connect()
    world = client.get_world()
    if args.map not in world.get_map().name:
        log.info("loading %s", args.map)
        world = client.load_world(args.map)

    if not args.skip_scout:
        _scout_spawn_points(world, args.scout)
    if not args.skip_feasibility:
        if args.custom_controller:
            _test_custom_controller(
                client, world,
                source_idx=args.source, target_idx=args.target,
                timeout_s=args.timeout,
                reach_distance_m=args.reach_distance,
                reach_speed_mps=args.reach_speed,
                fixed_dt=args.dt,
                tm_port=args.tm_port,
                sampling_resolution_m=args.sampling_resolution,
            )
        else:
            _test_set_path(
                client, world,
                source_idx=args.source, target_idx=args.target,
                timeout_s=args.timeout,
                reach_distance_m=args.reach_distance,
                reach_speed_mps=args.reach_speed,
                fixed_dt=args.dt,
                tm_port=args.tm_port,
                dense_route=not args.sparse_path,
                sampling_resolution_m=args.sampling_resolution,
            )


if __name__ == "__main__":
    main()
