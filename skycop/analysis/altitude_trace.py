"""Stats helper for altitude_trace.jsonl files produced by missions run
with ``control_mode: adaptive``.

Usage from the client container::

    python -m skycop.analysis.altitude_trace output/mission/<run>/altitude_trace.jsonl

Prints:

- altitude range (min / max / mean / std)
- per-tick altitude delta stats (mean abs delta, 95th percentile)
- oscillation cadence (most common "zero-crossing" period around the mean
  — naïve but fast; we're looking for order-of-magnitude, not audio)
- fraction of ticks with ``building_near=True``
- target vs current_z agreement (does smoothing lag/overshoot?)

Pure stdlib + json. No numpy dependency on purpose so this stays trivial
and runs without any container setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {
        "n": n,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(mean, 3),
        "std": round(var ** 0.5, 3),
    }


def _abs_deltas(values: list[float]) -> list[float]:
    return [abs(values[i] - values[i - 1]) for i in range(1, len(values))]


def _zero_crossings_around_mean(values: list[float]) -> list[int]:
    if len(values) < 2:
        return []
    mean = sum(values) / len(values)
    shifted = [v - mean for v in values]
    crossings: list[int] = []
    for i in range(1, len(shifted)):
        if shifted[i - 1] * shifted[i] < 0:
            crossings.append(i)
    return crossings


def _period_estimate(crossings: list[int]) -> dict:
    if len(crossings) < 2:
        return {"n_crossings": len(crossings)}
    gaps = [crossings[i] - crossings[i - 1] for i in range(1, len(crossings))]
    # 2 crossings per full cycle → period ≈ 2 × median gap
    gap_counter = Counter(gaps)
    most_common_gap, most_common_n = gap_counter.most_common(1)[0]
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    return {
        "n_crossings": len(crossings),
        "median_half_period_ticks": median_gap,
        "estimated_full_period_ticks": 2 * median_gap,
        "modal_half_period_ticks": most_common_gap,
        "modal_half_period_count": most_common_n,
    }


def analyse(trace_path: Path) -> dict:
    records: list[dict] = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        return {"trace_path": str(trace_path), "n": 0}

    target_zs = [float(r["target"]) for r in records]
    current_zs = [float(r["current_z"]) for r in records]
    raycast_zs = [float(r["raycast_z"]) for r in records]
    building_flags = [bool(r["building_near"]) for r in records]
    rays_hit = [int(r["lateral_rays_hit"]) for r in records]

    abs_deltas = _abs_deltas(current_zs)
    abs_deltas.sort()
    p95 = abs_deltas[int(0.95 * (len(abs_deltas) - 1))] if abs_deltas else 0.0

    return {
        "trace_path": str(trace_path),
        "n_ticks": len(records),
        "target": _summary_stats(target_zs),
        "current_z": _summary_stats(current_zs),
        "raycast_z": _summary_stats(raycast_zs),
        "abs_tick_delta_mean": round(sum(abs_deltas) / len(abs_deltas), 3) if abs_deltas else 0.0,
        "abs_tick_delta_p95": round(p95, 3),
        "building_near_fraction": round(sum(building_flags) / len(building_flags), 3),
        "lateral_rays_hit": _summary_stats([float(x) for x in rays_hit]),
        "oscillation": _period_estimate(_zero_crossings_around_mean(current_zs)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_path", type=Path, help="altitude_trace.jsonl produced by the mission")
    args = ap.parse_args()

    if not args.trace_path.exists():
        print(f"not found: {args.trace_path}", file=sys.stderr)
        return 1

    summary = analyse(args.trace_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
