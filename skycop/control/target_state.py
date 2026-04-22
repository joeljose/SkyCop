"""Rolling-window target state estimator.

Takes per-tick target world positions (typically the output of
``gt_projection.pixel_to_world_on_ground`` applied to the locked track's
bbox centre) and yields a smoothed velocity estimate via finite difference
over the last ``window_size`` samples.

The velocity estimate is the flight PID's feedforward term. Without it the
drone would have steady-state distance equal to ``target_speed / Kp`` — an
unusable lag at pursuit speeds.

Pure: no CARLA, no numpy required (but accepts numpy arrays implicitly via
duck typing; only tuple math is used).
"""

from __future__ import annotations

from collections import deque


class TargetStateTracker:
    """Ring buffer of recent (t, x, y) samples → smoothed velocity."""

    def __init__(self, window_size: int = 3) -> None:
        if window_size < 2:
            raise ValueError(f"window_size must be ≥2, got {window_size}")
        self.window_size = int(window_size)
        self._samples: deque[tuple[float, float, float]] = deque(maxlen=self.window_size)

    def update(self, t: float, x: float, y: float) -> None:
        """Record a new target observation at time ``t``."""
        if self._samples and t <= self._samples[-1][0]:
            raise ValueError(
                f"timestamps must be strictly increasing; "
                f"got {t} after {self._samples[-1][0]}"
            )
        self._samples.append((float(t), float(x), float(y)))

    @property
    def position(self) -> tuple[float, float] | None:
        """Most recent observed position, or None if no observations yet."""
        if not self._samples:
            return None
        _, x, y = self._samples[-1]
        return (x, y)

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Mean finite-difference velocity across the window.

        Returns ``None`` until at least 2 samples have been recorded. For a
        full window of size N, averages the N-1 pairwise velocities to
        smooth out per-tick detection noise.
        """
        n = len(self._samples)
        if n < 2:
            return None

        vx_sum = 0.0
        vy_sum = 0.0
        pairs = 0
        samples = list(self._samples)
        for (t0, x0, y0), (t1, x1, y1) in zip(samples[:-1], samples[1:], strict=True):
            dt = t1 - t0
            if dt <= 0.0:
                continue
            vx_sum += (x1 - x0) / dt
            vy_sum += (y1 - y0) / dt
            pairs += 1

        if pairs == 0:
            return None
        return (vx_sum / pairs, vy_sum / pairs)

    def reset(self) -> None:
        """Drop all history. Use on track loss."""
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)
