"""Unit tests for skycop.sim.suspect_fsm — pure state machine, no CARLA."""

from __future__ import annotations

import pytest

from skycop.sim.suspect_fsm import (
    FSMConfig,
    ParkingSubstate,
    SuspectFSM,
    SuspectState,
    TMKnobs,
)

# ── Fixtures ──────────────────────────────────────────────────────────

_FLEEING_KNOBS = TMKnobs(
    speed_over_pct=-80.0, ignore_lights_pct=100.0, ignore_signs_pct=100.0,
    lane_change_pct=50.0, follow_dist_m=1.0,
)
_ROAMING_KNOBS = TMKnobs(
    speed_over_pct=0.0, ignore_lights_pct=0.0, ignore_signs_pct=0.0,
    lane_change_pct=0.0, follow_dist_m=3.0,
)
def _cfg(**overrides) -> FSMConfig:
    base = dict(
        fleeing_duration_s=10.0,
        roaming_duration_s=5.0,
        parking_lot_timeout_s=20.0,
        parking_roadside_timeout_s=10.0,
        parked_confirm_timeout_s=60.0,
        reach_distance_m=3.0,
        reach_speed_mps=0.5,
        ai_consecutive_lock_ticks=3,
        fleeing_knobs=_FLEEING_KNOBS,
        roaming_knobs=_ROAMING_KNOBS,
        parking_lots=((100.0, 0.0, 0.0),),
        roadside_spots=((200.0, 0.0, 0.0),),
    )
    base.update(overrides)
    return FSMConfig(**base)


def _tick(fsm, t, xy=(0.0, 0.0), speed=10.0, lock=False):
    return fsm.tick(t, xy, speed, lock)


# ── Initial tick ──────────────────────────────────────────────────────

def test_first_tick_enters_fleeing_with_knobs():
    fsm = SuspectFSM(cfg=_cfg())
    r = _tick(fsm, 0.0)
    assert r.state == SuspectState.FLEEING
    assert r.action.apply_tm_knobs == _FLEEING_KNOBS
    assert r.action.set_path_to is None
    assert r.action.freeze_physics is False
    assert r.terminal is False


def test_fleeing_idle_ticks_emit_no_action():
    fsm = SuspectFSM(cfg=_cfg())
    _tick(fsm, 0.0)   # entry
    r = _tick(fsm, 1.0)
    assert r.state == SuspectState.FLEEING
    assert r.action.apply_tm_knobs is None
    assert r.action.set_path_to is None


# ── FLEEING → ROAMING ─────────────────────────────────────────────────

def test_fleeing_transitions_to_roaming_after_duration():
    fsm = SuspectFSM(cfg=_cfg(fleeing_duration_s=10.0))
    _tick(fsm, 0.0)
    _tick(fsm, 9.9)                       # still FLEEING (< 10s)
    r = _tick(fsm, 10.0)                  # boundary — transitions
    assert r.state == SuspectState.ROAMING
    assert r.action.apply_tm_knobs == _ROAMING_KNOBS


# ── ROAMING → PARKING.TRYING_LOT ──────────────────────────────────────

def test_roaming_transitions_to_parking_trying_lot():
    fsm = SuspectFSM(cfg=_cfg(fleeing_duration_s=1.0, roaming_duration_s=1.0))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)                       # → ROAMING
    r = _tick(fsm, 2.0)                   # → PARKING trying_lot
    assert r.state == SuspectState.PARKING
    assert r.parking_substate == ParkingSubstate.TRYING_LOT
    assert r.action.apply_tm_knobs is None         # PARKING runs off-autopilot
    assert r.action.set_path_to == (100.0, 0.0, 0.0)   # only candidate


def test_parking_picks_nearest_lot():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        parking_lots=((100.0, 0.0, 0.0), (50.0, 0.0, 0.0), (200.0, 0.0, 0.0)),
    ))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)
    r = _tick(fsm, 2.0, xy=(48.0, 0.0))   # closest to 50
    assert r.action.set_path_to == (50.0, 0.0, 0.0)


# ── PARKING → PARKED (happy path) ─────────────────────────────────────

def test_reaching_parking_lot_enters_parked():
    fsm = SuspectFSM(cfg=_cfg(fleeing_duration_s=1.0, roaming_duration_s=1.0))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)                                            # → ROAMING
    _tick(fsm, 2.0, xy=(0.0, 0.0))                             # → PARKING
    r = _tick(fsm, 3.0, xy=(100.5, 0.0), speed=0.1)            # arrived
    assert r.state == SuspectState.PARKED
    assert r.action.freeze_physics is True
    assert r.countdown_s == 60.0


def test_parking_not_reached_if_still_moving_fast():
    fsm = SuspectFSM(cfg=_cfg(fleeing_duration_s=1.0, roaming_duration_s=1.0))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)
    _tick(fsm, 2.0)
    r = _tick(fsm, 3.0, xy=(100.0, 0.0), speed=5.0)  # close but not stopped
    assert r.state == SuspectState.PARKING
    assert r.parking_substate == ParkingSubstate.TRYING_LOT


def test_parking_not_reached_if_too_far():
    fsm = SuspectFSM(cfg=_cfg(fleeing_duration_s=1.0, roaming_duration_s=1.0))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)
    _tick(fsm, 2.0)
    r = _tick(fsm, 3.0, xy=(90.0, 0.0), speed=0.1)   # stopped but 10m away
    assert r.state == SuspectState.PARKING


# ── PARKING timeouts ──────────────────────────────────────────────────

def test_lot_timeout_escalates_to_roadside():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        parking_lot_timeout_s=5.0,
    ))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)
    _tick(fsm, 2.0, xy=(0.0, 0.0))                      # enter trying_lot
    r = _tick(fsm, 7.0, xy=(10.0, 10.0), speed=10.0)    # 5s elapsed
    assert r.state == SuspectState.PARKING
    assert r.parking_substate == ParkingSubstate.TRYING_ROADSIDE
    assert r.action.set_path_to == (200.0, 0.0, 0.0)


def test_roadside_timeout_falls_through_to_park_in_place():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        parking_lot_timeout_s=2.0, parking_roadside_timeout_s=2.0,
    ))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)
    _tick(fsm, 2.0)                                     # trying_lot
    _tick(fsm, 4.0, xy=(10.0, 10.0), speed=10.0)        # → trying_roadside
    r = _tick(fsm, 6.0, xy=(10.0, 10.0), speed=10.0)    # → park_in_place
    assert r.state == SuspectState.PARKING
    assert r.parking_substate == ParkingSubstate.PARK_IN_PLACE
    # park-in-place has no knobs / no path — mission loop just brakes while
    # the suspect is already off-autopilot from PARKING entry.
    assert r.action.apply_tm_knobs is None
    assert r.action.set_path_to is None


def test_lot_timeout_skips_roadside_when_empty():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        parking_lot_timeout_s=2.0, roadside_spots=(),
    ))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)
    _tick(fsm, 2.0)
    r = _tick(fsm, 4.0, xy=(10.0, 10.0), speed=10.0)
    assert r.state == SuspectState.PARKING
    assert r.parking_substate == ParkingSubstate.PARK_IN_PLACE


def test_park_in_place_transitions_to_parked_when_speed_drops():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        parking_lot_timeout_s=1.0, roadside_spots=(),
    ))
    _tick(fsm, 0.0)
    _tick(fsm, 1.0)
    _tick(fsm, 2.0)
    _tick(fsm, 3.0, xy=(10.0, 10.0), speed=10.0)     # → park_in_place
    r = _tick(fsm, 4.0, xy=(10.0, 10.0), speed=0.1)  # stopped
    assert r.state == SuspectState.PARKED
    assert r.action.freeze_physics is True


# ── PARKED submission logic ───────────────────────────────────────────

def _drive_to_parked(fsm, t0=0.0):
    """Fast-forward the FSM to PARKED, returning the t the mission is at."""
    _tick(fsm, t0)
    _tick(fsm, t0 + 1.0)                                # → ROAMING
    _tick(fsm, t0 + 2.0)                                # → PARKING trying_lot
    r = _tick(fsm, t0 + 3.0, xy=(100.0, 0.0), speed=0.1)
    assert r.state == SuspectState.PARKED
    return t0 + 3.0


def test_parked_countdown_decrements_per_tick():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        parked_confirm_timeout_s=60.0,
    ))
    t_park = _drive_to_parked(fsm)
    r = _tick(fsm, t_park + 10.0, xy=(100.0, 0.0), speed=0.0)
    assert r.countdown_s == pytest.approx(50.0)
    assert r.submission is None
    assert r.terminal is False


def test_parked_ai_pass_after_k_consecutive_locks():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        ai_consecutive_lock_ticks=3,
    ))
    t_park = _drive_to_parked(fsm)
    r1 = _tick(fsm, t_park + 0.05, lock=True)
    r2 = _tick(fsm, t_park + 0.10, lock=True)
    r3 = _tick(fsm, t_park + 0.15, lock=True)
    assert r1.submission is None
    assert r2.submission is None
    assert r3.submission == "ai_pass"
    assert r3.terminal is True


def test_parked_lock_streak_resets_on_break():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        ai_consecutive_lock_ticks=3,
    ))
    t_park = _drive_to_parked(fsm)
    _tick(fsm, t_park + 0.05, lock=True)
    _tick(fsm, t_park + 0.10, lock=True)
    _tick(fsm, t_park + 0.15, lock=False)    # streak broken
    r = _tick(fsm, t_park + 0.20, lock=True)
    # Would have fired on tick #3 without the break; now streak is back to 1.
    assert r.submission is None


def test_parked_timeout_emits_lose():
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        parked_confirm_timeout_s=5.0,
    ))
    t_park = _drive_to_parked(fsm)
    # Tick inside the window — still running.
    r_mid = _tick(fsm, t_park + 4.0, lock=False)
    assert r_mid.submission is None
    # Tick at/past the window — lose.
    r_out = _tick(fsm, t_park + 5.0, lock=False)
    assert r_out.submission == "timeout_lose"
    assert r_out.terminal is True
    assert r_out.countdown_s == 0.0


def test_parked_ai_pass_wins_over_countdown_in_same_tick():
    """If the lock-streak completes on the same tick the countdown hits 0, ai_pass wins."""
    fsm = SuspectFSM(cfg=_cfg(
        fleeing_duration_s=1.0, roaming_duration_s=1.0,
        parked_confirm_timeout_s=0.2, ai_consecutive_lock_ticks=3,
    ))
    t_park = _drive_to_parked(fsm)
    _tick(fsm, t_park + 0.05, lock=True)
    _tick(fsm, t_park + 0.10, lock=True)
    r = _tick(fsm, t_park + 0.20, lock=True)    # K=3 satisfied; window also expired
    assert r.submission == "ai_pass"
    assert r.terminal is True


# ── Properties ────────────────────────────────────────────────────────

def test_state_property_tracks_current_state():
    fsm = SuspectFSM(cfg=_cfg(fleeing_duration_s=1.0))
    _tick(fsm, 0.0)
    assert fsm.state == SuspectState.FLEEING
    _tick(fsm, 1.0)
    assert fsm.state == SuspectState.ROAMING
