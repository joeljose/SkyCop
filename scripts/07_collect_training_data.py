"""
Experiment 07 — collect CARLA pursuit training dataset.

Orchestrates N pursuit runs; **each run is isolated in its own Python
subprocess** because CARLA's C++ SIGABRT on actor-registry teardown can
kill the whole process on exit (bypasses Python ``try/except``). The
parent script monitors subprocess exit codes and continues regardless.

Each subprocess writes its own per-run manifest before cleanup fires;
the parent compiles them into a combined ``dataset_manifest.json``
after each run so a mid-loop failure preserves the work on disk.

Also resumable: existing per-run manifests are reused, so a re-run
skips already-captured pursuits.

Output:
  output/dataset/carla_pursuit/
    run_NN_seed###_<Weather>/
      images/, labels/, manifest.json
    dataset_manifest.json       ← combined summary + split assignments

Usage:
  make exp N=07                 # orchestrator mode (default)

Invoked internally per-run:
  python3 scripts/07_collect_training_data.py --single-run \\
      --run-id <id> --seed N --weather W --output-dir <path>
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from skycop.config import load
from skycop.cv.capture import reset_world, run_capture
from skycop.logs import setup_logging
from skycop.sim import connect

log = logging.getLogger("exp07")


# ── worker mode: one pursuit, one python process, exit when done ─────────

def _single_run(args, cfg) -> None:
    client = connect(host=cfg.carla.host, port=cfg.carla.port)
    cleared = reset_world(client, cfg.carla.tm_port)
    if cleared:
        log.info("reset: destroyed %d orphan actors/sensors", cleared)
    time.sleep(0.5)

    result = run_capture(
        cfg,
        output_dir=Path(args.output_dir),
        client=client,
        run_id=args.run_id,
        seed=args.seed,
        weather=args.weather,
        target_frames=int(cfg.training_dataset.target_frames_per_run),
        subsample_every=int(cfg.training_dataset.subsample_every),
        max_ticks=int(cfg.training_dataset.max_ticks_per_run),
        min_bbox_pixel=int(cfg.training_dataset.min_bbox_pixel),
        min_visibility=float(cfg.training_dataset.min_visibility),
        jpeg_quality=int(cfg.training_dataset.jpeg_quality),
    )
    log.info("worker done: %d frames in %.1fs", result.frames_saved, result.duration_s)


# ── orchestrator mode: spawn one subprocess per run, tolerate crashes ───

def _build_dataset_manifest(results: list[dict], val_seeds: set[int]) -> dict:
    return {
        "runs": results,
        "total_frames": sum(r["frames_saved"] for r in results),
        "train_frames": sum(r["frames_saved"] for r in results if r["split"] == "train"),
        "val_frames": sum(r["frames_saved"] for r in results if r["split"] == "val"),
        "train_runs": [r["run_id"] for r in results if r["split"] == "train"],
        "val_runs": [r["run_id"] for r in results if r["split"] == "val"],
        "val_seeds": sorted(val_seeds),
    }


def _save_manifest_atomic(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def _read_run_result(run_dir: Path, run_id: str, seed: int, weather: str, split: str) -> dict | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        m = json.load(f)
    return {
        "run_id": run_id,
        "seed": seed,
        "weather": weather,
        "split": split,
        "frames_saved": len(m.get("frames", [])),
        "class_counts": m.get("class_counts", {}),
        "skip_counts": m.get("skip_counts", {}),
        "output_dir": str(run_dir),
    }


def _orchestrate(cfg) -> None:
    root_out = Path(cfg.training_dataset.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    runs_cfg = list(cfg.training_dataset.runs)
    val_seeds = set(int(s) for s in cfg.training_dataset.val_seeds)

    total_t0 = time.perf_counter()
    results: list[dict] = []

    for i, spec in enumerate(runs_cfg):
        seed = int(spec.seed)
        weather = str(spec.weather)
        run_id = f"run_{i:02d}_seed{seed}_{weather}"
        run_dir = root_out / run_id
        split = "val" if seed in val_seeds else "train"

        existing = _read_run_result(run_dir, run_id, seed, weather, split)
        if existing is not None and existing["frames_saved"] > 0:
            log.info("══ [%d/%d] %s  (cached: %d frames)  ══",
                     i + 1, len(runs_cfg), run_id, existing["frames_saved"])
            results.append(existing)
            _save_manifest_atomic(_build_dataset_manifest(results, val_seeds),
                                  root_out / "dataset_manifest.json")
            continue

        log.info("══ [%d/%d] %s  split=%s  ══", i + 1, len(runs_cfg), run_id, split)

        cmd = [
            sys.executable, __file__,
            "--single-run",
            "--run-id", run_id,
            "--seed", str(seed),
            "--weather", weather,
            "--output-dir", str(run_dir),
        ]
        # Don't raise on non-zero exit: CARLA's SIGABRT during cleanup is
        # expected and the manifest was already atomically written before
        # the crash. We check for the manifest file afterwards.
        proc = subprocess.run(cmd, check=False)
        log.info("[%s] subprocess exit=%d", run_id, proc.returncode)

        result = _read_run_result(run_dir, run_id, seed, weather, split)
        if result is None:
            log.warning("[%s] no manifest on disk — run yielded zero frames or crashed early",
                        run_id)
            continue

        results.append(result)
        _save_manifest_atomic(_build_dataset_manifest(results, val_seeds),
                              root_out / "dataset_manifest.json")

    total_elapsed = time.perf_counter() - total_t0
    log.info("═══ done: %d/%d runs, %d total frames in %.1fs ═══",
             len(results), len(runs_cfg),
             sum(r["frames_saved"] for r in results),
             total_elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run", action="store_true",
                        help="Worker mode: run one pursuit and exit (called internally).")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weather")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    setup_logging()
    cfg = load("default", "training_dataset")

    if args.single_run:
        _single_run(args, cfg)
    else:
        _orchestrate(cfg)


if __name__ == "__main__":
    main()
