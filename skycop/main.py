"""SkyCop mission entrypoint — ``python -m skycop.main`` / ``make app``."""

import sys

from skycop.config import load
from skycop.logs import setup_logging
from skycop.mission import run_mission

sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    setup_logging()
    cfg = load("default", "detector", "training", "tracking", "mission")
    run_mission(cfg)


if __name__ == "__main__":
    main()
