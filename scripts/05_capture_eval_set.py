"""
Experiment 05 — capture CARLA pursuit eval holdout.

Runs a single adaptive-altitude pursuit on experiment 04's scene and dumps
paired RGB + YOLO-format label files every Nth tick, plus a reproducibility
manifest. Frames with zero valid labels are skipped entirely.

Output:
  output/eval/carla_eval/
    images/frame_NNNN.jpg
    labels/frame_NNNN.txt
    manifest.json

This set is consumed by experiment 06 (pretrained baseline) and experiment
08 (fine-tuned scoring) as a frozen eval benchmark — same exact frames scored
against every round of training. Do not regenerate casually.
"""

import logging
import time
from pathlib import Path

from skycop.config import load
from skycop.cv.capture import reset_world, run_capture
from skycop.logs import setup_logging
from skycop.sim import connect

log = logging.getLogger("exp05")


def main():
    setup_logging()
    cfg = load("default", "altitude", "eval_capture")
    client = connect(host=cfg.carla.host, port=cfg.carla.port)

    # Ensure a clean world state before we start — prior runs leave orphan
    # sensors on CARLA's registry (carla_caveats §6).
    cleared = reset_world(client, cfg.carla.tm_port)
    if cleared:
        log.info("reset world: destroyed %d orphan actors/sensors", cleared)
    time.sleep(0.5)

    result = run_capture(
        cfg,
        output_dir=Path(cfg.eval_capture.output_dir),
        client=client,
        run_id="eval_holdout",
        seed=cfg.seed,
        weather="ClearNoon",
        target_frames=int(cfg.eval_capture.target_frames),
        subsample_every=int(cfg.eval_capture.subsample_every),
        max_ticks=int(cfg.eval_capture.max_ticks),
        min_bbox_pixel=int(cfg.eval_capture.min_bbox_pixel),
        min_visibility=float(cfg.eval_capture.min_visibility),
        jpeg_quality=int(cfg.eval_capture.jpeg_quality),
    )

    log.info("done: %d frames, skip_counts=%s", result.frames_saved, result.skip_counts)


if __name__ == "__main__":
    main()
