"""SkyCop mission entrypoint — ``python -m skycop.main`` / ``make app``.

Starts the MJPEG live view, then runs the mission with the live view
wired in as the primary inspection path. Browse to
``http://localhost:<mjpeg_port>`` while the mission is running to watch
the overlay feed in real time.
"""

import logging
import sys

from skycop.config import load
from skycop.dashboard import MJPEGServer
from skycop.logs import setup_logging
from skycop.mission import run_mission

sys.stdout.reconfigure(line_buffering=True)

log = logging.getLogger("skycop.main")


def main() -> None:
    setup_logging()
    cfg = load("default", "detector", "training", "tracking", "control", "mission", "suspect")

    port = int(cfg.mission.get("mjpeg_port", 5000))
    server = MJPEGServer(
        title="SkyCop — Mission v0a",
        hud="Open this page, then press Start below. The pursuit begins on click — no ticks happen before you do.",
        use_start_trigger=True,
    )
    server.start(port=port)
    log.info("live view + start page: http://localhost:%d", port)
    log.info("waiting for start click…")

    run_mission(cfg, mjpeg_server=server)


if __name__ == "__main__":
    main()
