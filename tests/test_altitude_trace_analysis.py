"""Unit tests for skycop.analysis.altitude_trace — pure stdlib, no CARLA."""

import json
from pathlib import Path

from skycop.analysis.altitude_trace import analyse


def _write_trace(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _record(
    tick: int, target: float, current_z: float, building_near: bool,
    rooftop_z: float | None = None, raycast_z: float = 15.0, rays_hit: int = 0,
) -> dict:
    return {
        "tick": tick,
        "target": target,
        "current_z": current_z,
        "building_near": building_near,
        "rooftop_z": rooftop_z,
        "raycast_z": raycast_z,
        "lateral_rays_hit": rays_hit,
    }


def test_analyse_empty_file(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    summary = analyse(p)
    assert summary["n"] == 0


def test_analyse_steady_altitude(tmp_path: Path):
    p = tmp_path / "steady.jsonl"
    _write_trace(p, [_record(i, target=15.0, current_z=15.0, building_near=False) for i in range(20)])
    summary = analyse(p)

    assert summary["n_ticks"] == 20
    assert summary["current_z"]["min"] == 15.0
    assert summary["current_z"]["max"] == 15.0
    assert summary["current_z"]["std"] == 0.0
    assert summary["abs_tick_delta_mean"] == 0.0
    assert summary["abs_tick_delta_p95"] == 0.0
    assert summary["building_near_fraction"] == 0.0
    # Zero-crossings around a mean equal to the constant value → none
    assert summary["oscillation"]["n_crossings"] == 0


def test_analyse_detects_oscillation_period(tmp_path: Path):
    # Square-wave altitude alternating every tick between 10 and 40 → mean 25,
    # every tick crosses the mean → ~20 crossings for 20 ticks, period of 2 ticks
    p = tmp_path / "osc.jsonl"
    recs = []
    for i in range(20):
        z = 10.0 if i % 2 == 0 else 40.0
        recs.append(_record(i, target=z, current_z=z, building_near=(i % 2 == 1)))
    _write_trace(p, recs)
    summary = analyse(p)

    assert summary["n_ticks"] == 20
    # Half-period in ticks ≈ 1 (transition each tick) → full period ≈ 2
    osc = summary["oscillation"]
    assert osc["n_crossings"] > 0
    assert osc["estimated_full_period_ticks"] == 2
    assert summary["building_near_fraction"] == 0.5


def test_analyse_counts_building_near(tmp_path: Path):
    p = tmp_path / "bn.jsonl"
    recs = [_record(i, target=15.0, current_z=15.0, building_near=(i < 7)) for i in range(20)]
    _write_trace(p, recs)
    summary = analyse(p)
    assert summary["building_near_fraction"] == 0.35  # 7/20
