"""Suspect FSM — drives the target vehicle through realistic pursuit states.

Pure-logic module; no CARLA imports. Observations (``t``, ``suspect_xy``,
``suspect_speed``, ``ai_lock_on_suspect``) go in; a ``FSMTick`` with the
current state, any side-effects the mission loop should perform
(TM-knob re-apply, ``set_path``, freeze physics), a PARKED-countdown value
and a submission-event label comes out.

State machine::

    FLEEING ──(after fleeing_duration_s)──► ROAMING
       │
    ROAMING ──(after roaming_duration_s)──► PARKING (trying_lot)
       │
    PARKING.trying_lot ──(reached)──► PARKED
            └─(lot_timeout_s)──► PARKING.trying_roadside ──(reached)──► PARKED
                                     └─(roadside_timeout_s)──► PARKING.park_in_place ──► PARKED
       │
    PARKED starts a ``confirm_timeout_s`` countdown
       │
    submission("ai_pass") on first K consecutive ticks with ai_lock_on_suspect
    submission("timeout_lose") if countdown reaches 0 without a pass
    either submission ⇒ terminal=True

Transition rationale recorded in docs/design.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class SuspectState(str, Enum):
    FLEEING = "fleeing"
    ROAMING = "roaming"
    PARKING = "parking"
    PARKED = "parked"


class ParkingSubstate(str, Enum):
    TRYING_LOT = "trying_lot"
    TRYING_ROADSIDE = "trying_roadside"
    PARK_IN_PLACE = "park_in_place"


# ── Configuration ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class TMKnobs:
    """TrafficManager knob values for one FSM state.

    Sign convention matches CARLA's ``vehicle_percentage_speed_difference``:
    negative = faster than the speed limit, positive = slower. See
    docs/carla_caveats.md §9.
    """
    speed_over_pct: float
    ignore_lights_pct: float
    ignore_signs_pct: float
    lane_change_pct: float
    follow_dist_m: float


@dataclass(frozen=True)
class FSMConfig:
    fleeing_duration_s: float
    roaming_duration_s: float
    parking_lot_timeout_s: float
    parking_roadside_timeout_s: float
    parked_confirm_timeout_s: float
    reach_distance_m: float
    reach_speed_mps: float
    ai_consecutive_lock_ticks: int
    fleeing_knobs: TMKnobs
    roaming_knobs: TMKnobs
    # PARKING runs off-autopilot (custom WaypointFollower controller); no
    # TM knobs apply. The mission loop handles the disable-autopilot / brake
    # side-effects when it sees a PARKING state transition.
    parking_lots: tuple[tuple[float, float, float], ...]      # (x, y, z) waypoints
    roadside_spots: tuple[tuple[float, float, float], ...]


# ── Per-tick I/O ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class FSMAction:
    """Side-effects the mission loop should perform in response to a tick.

    All fields default to "do nothing" so ticks that change no state keep
    the mission loop quiet. State-entry ticks populate the relevant fields.
    """
    apply_tm_knobs: TMKnobs | None = None
    set_path_to: tuple[float, float, float] | None = None
    freeze_physics: bool = False


SubmissionEvent = Literal["ai_pass", "timeout_lose"]


@dataclass(frozen=True)
class FSMTick:
    state: SuspectState
    parking_substate: ParkingSubstate | None
    action: FSMAction
    countdown_s: float | None         # seconds remaining in PARKED window
    submission: SubmissionEvent | None
    terminal: bool                    # mission ends this tick


# ── FSM ────────────────────────────────────────────────────────────────

@dataclass
class SuspectFSM:
    cfg: FSMConfig
    rng: random.Random = field(default_factory=random.Random)

    _state: SuspectState = field(init=False, default=SuspectState.FLEEING)
    _parking_substate: ParkingSubstate | None = field(init=False, default=None)
    _state_entered_at: float = field(init=False, default=0.0)
    _parked_entered_at: float | None = field(init=False, default=None)
    _ai_lock_streak: int = field(init=False, default=0)
    _chosen_destination: tuple[float, float, float] | None = field(init=False, default=None)
    _initial_tick_emitted: bool = field(init=False, default=False)

    # ── Public API ─────────────────────────────────────────────────

    def tick(
        self,
        t: float,
        suspect_xy: tuple[float, float],
        suspect_speed: float,
        ai_lock_on_suspect: bool,
    ) -> FSMTick:
        """Advance the FSM by one tick. Returns the tick result."""
        # The first tick emitted also counts as an entry into FLEEING, so the
        # mission loop applies the initial knobs.
        entry_action: FSMAction = FSMAction()
        if not self._initial_tick_emitted:
            self._state_entered_at = t
            entry_action = FSMAction(apply_tm_knobs=self.cfg.fleeing_knobs)
            self._initial_tick_emitted = True
            return FSMTick(
                state=SuspectState.FLEEING,
                parking_substate=None,
                action=entry_action,
                countdown_s=None,
                submission=None,
                terminal=False,
            )

        # ── FLEEING ──
        if self._state == SuspectState.FLEEING:
            if t - self._state_entered_at >= self.cfg.fleeing_duration_s:
                return self._enter_roaming(t)
            return self._plain(SuspectState.FLEEING)

        # ── ROAMING ──
        if self._state == SuspectState.ROAMING:
            if t - self._state_entered_at >= self.cfg.roaming_duration_s:
                return self._enter_parking_trying_lot(t, suspect_xy)
            return self._plain(SuspectState.ROAMING)

        # ── PARKING ──
        if self._state == SuspectState.PARKING:
            return self._tick_parking(t, suspect_xy, suspect_speed)

        # ── PARKED ──
        if self._state == SuspectState.PARKED:
            return self._tick_parked(t, ai_lock_on_suspect)

        raise AssertionError(f"unknown state {self._state!r}")

    # ── Internal transitions ───────────────────────────────────────

    def _plain(
        self,
        state: SuspectState,
        substate: ParkingSubstate | None = None,
        countdown_s: float | None = None,
    ) -> FSMTick:
        return FSMTick(
            state=state,
            parking_substate=substate,
            action=FSMAction(),
            countdown_s=countdown_s,
            submission=None,
            terminal=False,
        )

    def _enter_roaming(self, t: float) -> FSMTick:
        self._state = SuspectState.ROAMING
        self._state_entered_at = t
        return FSMTick(
            state=SuspectState.ROAMING,
            parking_substate=None,
            action=FSMAction(apply_tm_knobs=self.cfg.roaming_knobs),
            countdown_s=None,
            submission=None,
            terminal=False,
        )

    def _enter_parking_trying_lot(
        self,
        t: float,
        suspect_xy: tuple[float, float],
    ) -> FSMTick:
        self._state = SuspectState.PARKING
        self._parking_substate = ParkingSubstate.TRYING_LOT
        self._state_entered_at = t
        dest = self._pick_nearest(suspect_xy, self.cfg.parking_lots)
        self._chosen_destination = dest
        return FSMTick(
            state=SuspectState.PARKING,
            parking_substate=ParkingSubstate.TRYING_LOT,
            action=FSMAction(set_path_to=dest),
            countdown_s=None,
            submission=None,
            terminal=False,
        )

    def _enter_parking_trying_roadside(
        self,
        t: float,
        suspect_xy: tuple[float, float],
    ) -> FSMTick:
        self._parking_substate = ParkingSubstate.TRYING_ROADSIDE
        self._state_entered_at = t
        dest = self._pick_nearest(suspect_xy, self.cfg.roadside_spots)
        self._chosen_destination = dest
        return FSMTick(
            state=SuspectState.PARKING,
            parking_substate=ParkingSubstate.TRYING_ROADSIDE,
            action=FSMAction(set_path_to=dest),
            countdown_s=None,
            submission=None,
            terminal=False,
        )

    def _enter_park_in_place(self, t: float) -> FSMTick:
        self._parking_substate = ParkingSubstate.PARK_IN_PLACE
        self._state_entered_at = t
        # No action needed — mission loop reads substate == PARK_IN_PLACE
        # and just applies a brake-only control each tick.
        return FSMTick(
            state=SuspectState.PARKING,
            parking_substate=ParkingSubstate.PARK_IN_PLACE,
            action=FSMAction(),
            countdown_s=None,
            submission=None,
            terminal=False,
        )

    def _enter_parked(self, t: float) -> FSMTick:
        self._state = SuspectState.PARKED
        self._parking_substate = None
        self._state_entered_at = t
        self._parked_entered_at = t
        self._ai_lock_streak = 0
        return FSMTick(
            state=SuspectState.PARKED,
            parking_substate=None,
            action=FSMAction(freeze_physics=True),
            countdown_s=self.cfg.parked_confirm_timeout_s,
            submission=None,
            terminal=False,
        )

    # ── Per-state tick handlers ────────────────────────────────────

    def _tick_parking(
        self,
        t: float,
        suspect_xy: tuple[float, float],
        suspect_speed: float,
    ) -> FSMTick:
        sub = self._parking_substate
        assert sub is not None  # invariant: PARKING always has a substate

        if sub == ParkingSubstate.PARK_IN_PLACE:
            # Gentle-brake knobs have been applied; wait for the car to stop.
            if suspect_speed < self.cfg.reach_speed_mps:
                return self._enter_parked(t)
            return self._plain(SuspectState.PARKING, substate=sub)

        # trying_lot or trying_roadside: check for arrival, else timeout-escalate.
        assert self._chosen_destination is not None
        dx = suspect_xy[0] - self._chosen_destination[0]
        dy = suspect_xy[1] - self._chosen_destination[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= self.cfg.reach_distance_m and suspect_speed < self.cfg.reach_speed_mps:
            return self._enter_parked(t)

        elapsed = t - self._state_entered_at
        timeout_s = (
            self.cfg.parking_lot_timeout_s
            if sub == ParkingSubstate.TRYING_LOT
            else self.cfg.parking_roadside_timeout_s
        )
        if elapsed >= timeout_s:
            if sub == ParkingSubstate.TRYING_LOT and self.cfg.roadside_spots:
                return self._enter_parking_trying_roadside(t, suspect_xy)
            return self._enter_park_in_place(t)

        return self._plain(SuspectState.PARKING, substate=sub)

    def _tick_parked(self, t: float, ai_lock_on_suspect: bool) -> FSMTick:
        assert self._parked_entered_at is not None
        elapsed = t - self._parked_entered_at
        countdown = max(0.0, self.cfg.parked_confirm_timeout_s - elapsed)

        if ai_lock_on_suspect:
            self._ai_lock_streak += 1
        else:
            self._ai_lock_streak = 0

        if self._ai_lock_streak >= self.cfg.ai_consecutive_lock_ticks:
            return FSMTick(
                state=SuspectState.PARKED,
                parking_substate=None,
                action=FSMAction(),
                countdown_s=countdown,
                submission="ai_pass",
                terminal=True,
            )

        if countdown <= 0.0:
            return FSMTick(
                state=SuspectState.PARKED,
                parking_substate=None,
                action=FSMAction(),
                countdown_s=0.0,
                submission="timeout_lose",
                terminal=True,
            )

        return FSMTick(
            state=SuspectState.PARKED,
            parking_substate=None,
            action=FSMAction(),
            countdown_s=countdown,
            submission=None,
            terminal=False,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _pick_nearest(
        self,
        xy: tuple[float, float],
        candidates: tuple[tuple[float, float, float], ...],
    ) -> tuple[float, float, float] | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda p: (p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2,
        )

    # ── Observability ──────────────────────────────────────────────

    @property
    def state(self) -> SuspectState:
        return self._state

    @property
    def parking_substate(self) -> ParkingSubstate | None:
        return self._parking_substate
