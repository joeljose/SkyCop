"""Unit tests for the adaptive altitude controller's pure logic (no CARLA)."""

import math

from skycop.control.altitude import AltitudeConfig, Observation, compute_target


def _cfg(**overrides):
    return AltitudeConfig(**{**dict(AltitudeConfig().__dict__), **overrides})


def test_open_road_targets_low_altitude():
    cfg = _cfg(smoothing=0.0)  # snap so we see the exact target
    z = compute_target(cfg, Observation(building_near=False, rooftop_z=None), previous_z=None)
    assert z == cfg.open_target_m


def test_building_near_targets_urban_altitude():
    cfg = _cfg(smoothing=0.0)
    z = compute_target(cfg, Observation(building_near=True, rooftop_z=None), previous_z=None)
    assert z == cfg.urban_target_m


def test_rooftop_forces_clearance():
    # Rooftop at 35m with 12m clearance → target ≥ 47m
    cfg = _cfg(smoothing=0.0, urban_target_m=40.0, rooftop_clearance_m=12.0)
    z = compute_target(cfg, Observation(building_near=True, rooftop_z=35.0), previous_z=None)
    assert z == 47.0


def test_target_clamped_to_ceiling():
    # Rooftop at 55m + 12m = 67m → clamps to max_m
    cfg = _cfg(smoothing=0.0, max_m=60.0)
    z = compute_target(cfg, Observation(building_near=True, rooftop_z=55.0), previous_z=None)
    assert z == cfg.max_m


def test_target_clamped_to_floor():
    # Configs that resolve below floor still clamp up
    cfg = _cfg(smoothing=0.0, min_m=10.0, open_target_m=5.0)
    z = compute_target(cfg, Observation(building_near=False, rooftop_z=None), previous_z=None)
    assert z == cfg.min_m


def test_smoothing_blends_previous_and_target():
    cfg = _cfg(smoothing=0.5, open_target_m=10.0)
    prev = 40.0
    z = compute_target(cfg, Observation(building_near=False, rooftop_z=None), previous_z=prev)
    # 0.5*40 + 0.5*10 = 25
    assert math.isclose(z, 25.0)


def test_smoothing_converges_with_repeated_target():
    cfg = _cfg(smoothing=0.5, open_target_m=10.0)
    z = None
    for _ in range(20):
        z = compute_target(cfg, Observation(building_near=False, rooftop_z=None), previous_z=z)
    assert math.isclose(z, 10.0, abs_tol=1e-3)
