"""Scalar PID controller with output clamp, integral anti-windup, and a
feedforward input.

Used by Mission v1a's flight loop to convert drone-to-target world-space
position error into a commanded velocity. One instance per axis (x, y);
altitude is pinned per D-12 so no altitude PID.

Pure Python + no state outside the class — unit-testable in isolation with
synthetic step / ramp / sinusoidal target sequences.

The feedforward term lets the caller add the target's estimated velocity on
top of the PID correction, which is required for high-speed pursuit:
position-only control has steady-state distance = target_speed / Kp, which
blows up at 22 m/s even for modest Kp.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PursuitPID:
    """Scalar PID with velocity-feedforward support.

    Parameters
    ----------
    Kp, Ki, Kd
        Standard PID gains.
    output_clamp
        Absolute value of the output saturation; ``abs(output) ≤ output_clamp``.
    integral_clamp
        Absolute-value cap on the accumulated integral state; anti-windup.

    Each ``step`` call receives the current error and ``dt``, plus an
    optional feedforward term added to the output after the PID mix.
    """

    Kp: float
    Ki: float = 0.0
    Kd: float = 0.0
    output_clamp: float = float("inf")
    integral_clamp: float = float("inf")

    _integral: float = field(default=0.0, init=False)
    _prev_error: float | None = field(default=None, init=False)

    def step(self, error: float, dt: float, feedforward: float = 0.0) -> float:
        """Advance one tick. Returns the commanded output (clamped)."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Integral term with pre-output anti-windup.
        self._integral = _clamp(self._integral + error * dt, self.integral_clamp)

        # Derivative term — skip on the very first call to avoid a transient kick.
        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = (
            self.Kp * error
            + self.Ki * self._integral
            + self.Kd * derivative
            + feedforward
        )
        return _clamp(output, self.output_clamp)

    def reset(self) -> None:
        """Drop integrator state + derivative memory. Use on lock loss."""
        self._integral = 0.0
        self._prev_error = None


def _clamp(value: float, bound: float) -> float:
    if value > bound:
        return bound
    if value < -bound:
        return -bound
    return value
