"""Unit tests for skycop.control.target_state — pure, no CARLA."""

import pytest

from skycop.control.target_state import TargetStateTracker


def test_empty_tracker_returns_none():
    t = TargetStateTracker(window_size=3)
    assert t.position is None
    assert t.velocity is None
    assert len(t) == 0


def test_single_sample_has_position_but_no_velocity():
    t = TargetStateTracker(window_size=3)
    t.update(0.0, 10.0, 20.0)
    assert t.position == (10.0, 20.0)
    assert t.velocity is None
    assert len(t) == 1


def test_two_samples_yields_finite_difference_velocity():
    t = TargetStateTracker(window_size=3)
    t.update(0.0, 0.0, 0.0)
    t.update(0.1, 1.0, 2.0)
    vx, vy = t.velocity
    assert vx == pytest.approx(10.0)    # 1 m / 0.1 s
    assert vy == pytest.approx(20.0)    # 2 m / 0.1 s


def test_velocity_averages_over_full_window():
    """Three samples at constant velocity → velocity equals that velocity."""
    t = TargetStateTracker(window_size=3)
    t.update(0.0, 0.0, 0.0)
    t.update(0.1, 1.0, 0.5)
    t.update(0.2, 2.0, 1.0)
    vx, vy = t.velocity
    assert vx == pytest.approx(10.0)
    assert vy == pytest.approx(5.0)


def test_velocity_smooths_noise():
    """A single noisy sample shouldn't fully dominate the velocity estimate."""
    t = TargetStateTracker(window_size=4)
    # Steady motion at v=5 on +x, with one noisy sample.
    t.update(0.0, 0.0, 0.0)
    t.update(0.1, 0.5, 0.0)
    t.update(0.2, 1.5, 0.0)   # noisy — big jump
    t.update(0.3, 1.5, 0.0)   # noisy — no motion
    vx, vy = t.velocity
    # 3 pairwise velocities: (0.5-0)/0.1=5, (1.5-0.5)/0.1=10, (1.5-1.5)/0.1=0
    # Average: 5.0
    assert vx == pytest.approx(5.0)
    assert vy == pytest.approx(0.0)


def test_window_drops_oldest_sample():
    t = TargetStateTracker(window_size=2)
    t.update(0.0, 0.0, 0.0)
    t.update(0.1, 1.0, 0.0)
    t.update(0.2, 3.0, 0.0)  # oldest sample evicted
    vx, vy = t.velocity
    # Only the last two samples: (3.0 - 1.0)/0.1 = 20
    assert vx == pytest.approx(20.0)
    assert len(t) == 2


def test_non_increasing_timestamp_raises():
    t = TargetStateTracker(window_size=3)
    t.update(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        t.update(0.0, 1.0, 1.0)     # same timestamp
    with pytest.raises(ValueError):
        t.update(-0.1, 1.0, 1.0)    # earlier timestamp


def test_reset_clears_history():
    t = TargetStateTracker(window_size=3)
    t.update(0.0, 0.0, 0.0)
    t.update(0.1, 1.0, 1.0)
    t.reset()
    assert t.position is None
    assert t.velocity is None
    assert len(t) == 0


def test_window_size_must_be_at_least_two():
    with pytest.raises(ValueError):
        TargetStateTracker(window_size=1)
    with pytest.raises(ValueError):
        TargetStateTracker(window_size=0)
