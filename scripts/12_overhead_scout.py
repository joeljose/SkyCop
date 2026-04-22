"""Town10HD overhead-obstacle scout — prerequisite for issue #45.

Queries every ``CityObjectLabel`` that could host an over-road structure
(Buildings, RailTrack, Bridge, Static, Other), filters to objects whose
bbox top is above typical drone pursuit altitude minus clearance, and
keeps only those whose XY footprint overlaps a Driving lane. The output
is the honest map D-12's audit was missing.

Why: D-12 concluded "0 Bridge objects in Town10HD" and pinned altitude
at 15 m. Mission v0a (issue #44) empirically found Town10HD's monorail
crossings live under ``Buildings@12.3m`` + ``RailTrack@7.2m``, not Bridge
— the audit methodology was incomplete.

Run inside the client container::

    docker compose exec client python3 scripts/12_overhead_scout.py
    docker compose exec client python3 scripts/12_overhead_scout.py --min-top 5 --emit-json /app/output/scout/town10hd_overhead.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import carla

from skycop.logs import setup_logging
from skycop.sim import connect

log = logging.getLogger(__name__)


# Labels that can host overhead structures in CARLA's semantic taxonomy.
_OVERHEAD_LABELS = [
    carla.CityObjectLabel.Buildings,
    carla.CityObjectLabel.RailTrack,
    carla.CityObjectLabel.Bridge,
    carla.CityObjectLabel.Static,
    carla.CityObjectLabel.Other,
]


def _is_over_driving_lane(
    wmap: carla.Map,
    bx: float,
    by: float,
    ex: float,
    ey: float,
) -> bool:
    """True if the bbox XY footprint intersects any Driving-lane waypoint.

    We probe the bbox centre plus four corners — cheap and sufficient for
    the AABB approximation we're already doing.
    """
    probes = [
        (bx, by),
        (bx + ex, by + ey), (bx - ex, by + ey),
        (bx + ex, by - ey), (bx - ex, by - ey),
    ]
    for px, py in probes:
        wp = wmap.get_waypoint(
            carla.Location(px, py, 0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is not None and wp.transform.location.distance(
            carla.Location(px, py, 0.0)
        ) < max(ex, ey) + 1.0:
            return True
    return False


def scout(
    world: carla.World,
    min_top_m: float,
    pursuit_altitude_m: float,
    clearance_margin_m: float,
) -> list[dict]:
    """Return a list of overhead-structure records ready for JSON export."""
    wmap = world.get_map()
    records: list[dict] = []
    for label in _OVERHEAD_LABELS:
        bbs = world.get_level_bbs(label)
        for b in bbs:
            z_top = b.location.z + b.extent.z
            z_bot = b.location.z - b.extent.z
            if z_top < min_top_m:
                continue
            # An overhead structure that matters is one that's above the
            # drone (would-be occluder) or within the clearance window.
            if z_top < pursuit_altitude_m - clearance_margin_m:
                # Doesn't reach the drone; skip (still report if over road).
                would_occlude = False
            else:
                would_occlude = True
            if not _is_over_driving_lane(
                wmap, b.location.x, b.location.y, b.extent.x, b.extent.y
            ):
                continue
            records.append({
                "label": str(label).split(".")[-1],
                "xy": [round(b.location.x, 2), round(b.location.y, 2)],
                "extent_xy": [round(b.extent.x, 2), round(b.extent.y, 2)],
                "z_top": round(z_top, 2),
                "z_bot": round(z_bot, 2),
                "would_occlude_at_pursuit": would_occlude,
            })
    return records


def summarise(records: list[dict], pursuit_altitude_m: float) -> None:
    log.info("Found %d overhead-over-road records", len(records))
    occluders = [r for r in records if r["would_occlude_at_pursuit"]]
    log.info("  %d would occlude at pursuit altitude %.1fm", len(occluders), pursuit_altitude_m)

    # Label histogram
    from collections import Counter
    by_label = Counter(r["label"] for r in records)
    for label, n in by_label.most_common():
        log.info("  %-15s n=%d", label, n)

    # Top-10 tallest occluders
    tallest = sorted(occluders, key=lambda r: -r["z_top"])[:10]
    log.info("  top-10 tallest occluders:")
    for r in tallest:
        log.info(
            "    %-10s  xy=(%7.1f, %7.1f)  extent=(%5.1f x %5.1f)  z_top=%5.2f  z_bot=%5.2f",
            r["label"], r["xy"][0], r["xy"][1], r["extent_xy"][0], r["extent_xy"][1],
            r["z_top"], r["z_bot"],
        )

    # Altitude needed to clear ALL overhead structures + margin
    if occluders:
        max_top = max(r["z_top"] for r in occluders)
        log.info(
            "  absolute highest over-road structure: z_top=%.2fm  "
            "→ pursuit altitude should be ≥ %.1fm (with 5m clearance)",
            max_top, max_top + 5.0,
        )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Town10HD overhead scout")
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--min-top", type=float, default=3.0,
                        help="Ignore structures whose top is below this z (m)")
    parser.add_argument("--pursuit-altitude", type=float, default=25.0,
                        help="Current pursuit altitude (m) — for occlusion flag")
    parser.add_argument("--clearance", type=float, default=5.0,
                        help="Clearance margin (m) — structures within this of pursuit altitude flagged")
    parser.add_argument("--emit-json", type=Path,
                        help="If set, write the full record list as JSON here")
    args = parser.parse_args()

    client = connect()
    world = client.get_world()
    if args.map not in world.get_map().name:
        log.info("loading %s", args.map)
        world = client.load_world(args.map)

    records = scout(world, args.min_top, args.pursuit_altitude, args.clearance)
    summarise(records, args.pursuit_altitude)

    if args.emit_json:
        args.emit_json.parent.mkdir(parents=True, exist_ok=True)
        args.emit_json.write_text(json.dumps(records, indent=2))
        log.info("wrote %d records → %s", len(records), args.emit_json)


if __name__ == "__main__":
    main()
