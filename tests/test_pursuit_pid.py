"""Unit tests for skycop.control.pursuit_pid — scalar PID, pure."""

import math

import pytest

from skycop.control.pursuit_pid import PursuitPID


def test_proportional_only_zero_error_zero_output():
    pid = PursuitPID(Kp=1.0)
    assert pid.step(error=0.0, dt=0.05) == 0.0


def test_proportional_scales_with_error():
    pid = PursuitPID(Kp=2.0)
    assert pid.step(error=3.0, dt=0.05) == 6.0


def test_output_clamp_saturates():
    pid = PursuitPID(Kp=10.0, output_clamp=5.0)
    assert pid.step(error=10.0, dt=0.05) == 5.0
    # Symmetric clamp.
    assert pid.step(error=-10.0, dt=0.05) == -5.0


def test_integral_accumulates_over_steps():
    pid = PursuitPID(Kp=0.0, Ki=1.0)
    # Three ticks of error=2 for dt=0.1 → integral = 0.6 → output = 0.6
    pid.step(error=2.0, dt=0.1)
    pid.step(error=2.0, dt=0.1)
    out = pid.step(error=2.0, dt=0.1)
    assert abs(out - 0.6) < 1e-9


def test_integral_clamp_prevents_windup():
    pid = PursuitPID(Kp=0.0, Ki=1.0, integral_clamp=0.3)
    # Integrator would accumulate to 3.0 without the clamp.
    for _ in range(30):
        pid.step(error=1.0, dt=0.1)
    # Integrator capped at 0.3 → output = Ki * 0.3 = 0.3.
    out = pid.step(error=1.0, dt=0.1)
    assert abs(out - 0.3) < 1e-9


def test_derivative_responds_to_error_change():
    pid = PursuitPID(Kp=0.0, Kd=1.0)
    # First step: no prior — derivative contribution is 0.
    assert pid.step(error=5.0, dt=0.1) == 0.0
    # Second step: error dropped to 0 → d_err / dt = (0 - 5)/0.1 = -50
    out = pid.step(error=0.0, dt=0.1)
    assert abs(out - (-50.0)) < 1e-9


def test_derivative_skipped_after_reset():
    pid = PursuitPID(Kp=0.0, Kd=1.0)
    pid.step(error=5.0, dt=0.1)
    pid.step(error=0.0, dt=0.1)           # builds derivative state
    pid.reset()
    # After reset, no derivative kick on the first new call.
    assert pid.step(error=10.0, dt=0.1) == 0.0


def test_reset_clears_integrator():
    pid = PursuitPID(Kp=0.0, Ki=1.0)
    pid.step(error=2.0, dt=0.1)
    pid.step(error=2.0, dt=0.1)
    pid.reset()
    # Fresh integrator → one step of error=2, dt=0.1 gives 0.2
    assert abs(pid.step(error=2.0, dt=0.1) - 0.2) < 1e-9


def test_feedforward_adds_after_pid_mix_and_is_clamped_together():
    pid = PursuitPID(Kp=1.0, output_clamp=10.0)
    out = pid.step(error=3.0, dt=0.1, feedforward=5.0)
    # Kp*error + feedforward = 3 + 5 = 8 → within clamp
    assert abs(out - 8.0) < 1e-9
    # If the sum exceeds the clamp, the clamp takes effect.
    pid2 = PursuitPID(Kp=1.0, output_clamp=10.0)
    out2 = pid2.step(error=8.0, dt=0.1, feedforward=5.0)
    assert out2 == 10.0   # 8+5=13, clamped to 10


def test_dt_must_be_positive():
    pid = PursuitPID(Kp=1.0)
    with pytest.raises(ValueError):
        pid.step(error=1.0, dt=0.0)
    with pytest.raises(ValueError):
        pid.step(error=1.0, dt=-0.1)


def test_step_converges_toward_step_input_target():
    """With Kp alone, given a constant position error, PID drives toward it.

    Simulating a 1-D plant where position updates by velocity * dt and PID
    output is the commanded velocity. Error = target - position.
    """
    pid = PursuitPID(Kp=2.0, output_clamp=100.0)
    target, position, dt = 10.0, 0.0, 0.05
    for _ in range(200):  # 10 s of simulation
        error = target - position
        velocity_cmd = pid.step(error=error, dt=dt)
        position += velocity_cmd * dt
    # Converged to within 1 cm of target.
    assert abs(position - target) < 0.01


def test_feedforward_closes_lag_on_ramp_target():
    """Ramp target (constant velocity). Both target and plant advance in the
    same tick (standard parallel-update control sim). Without feedforward,
    position-only PID has steady-state lag v/Kp. With feedforward = target
    velocity, steady-state error → 0 (modulo discretisation).
    """
    v = 5.0   # target moves at 5 m/s
    dt = 0.05
    Kp = 1.0

    # Without feedforward: steady-state lag ≈ v/Kp (= 5 m here).
    pid_no_ff = PursuitPID(Kp=Kp, output_clamp=100.0)
    target, position = 0.0, 0.0
    for _ in range(400):   # 20 s
        error = target - position
        cmd = pid_no_ff.step(error=error, dt=dt)
        position += cmd * dt
        target += v * dt
    lag_without_ff = target - position
    # Continuous-time prediction: v/Kp. Discrete skew shifts this slightly.
    assert 4.5 < lag_without_ff < 5.5

    # With feedforward: lag collapses by at least an order of magnitude.
    pid_with_ff = PursuitPID(Kp=Kp, output_clamp=100.0)
    target, position = 0.0, 0.0
    for _ in range(400):
        error = target - position
        cmd = pid_with_ff.step(error=error, dt=dt, feedforward=v)
        position += cmd * dt
        target += v * dt
    lag_with_ff = target - position
    assert abs(lag_with_ff) < 0.1
    assert abs(lag_with_ff) < 0.05 * lag_without_ff


def test_damped_response_to_sinusoidal_target_is_bounded():
    """Sinusoidal reference — PID should track with bounded error."""
    dt = 0.05
    pid = PursuitPID(Kp=4.0, Kd=0.5, output_clamp=50.0)
    position = 0.0
    max_error = 0.0
    for n in range(400):   # 20 s
        t = n * dt
        target = 2.0 * math.sin(0.5 * t)
        error = target - position
        max_error = max(max_error, abs(error))
        position += pid.step(error=error, dt=dt) * dt
    # Arbitrary-but-reasonable: max error during a sinusoid ≤ target amplitude.
    assert max_error < 2.5
