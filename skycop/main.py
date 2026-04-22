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
        title="SkyCop Pursuit",
        hud="AI drone pursuit · live MJPEG feed",
        use_start_trigger=True,
        image_size=(int(cfg.camera.height), int(cfg.camera.width)),
    )
    server.start(port=port)
    log.info("live view + start page: http://localhost:%d", port)

    # Replay loop: each mission ends with a "Back to menu" link in the end
    # modal. We rearm the server so the menu shows again, then block on the
    # next start click. Ctrl-C in the terminal to quit the process.
    round_num = 1
    while True:
        log.info("round %d — waiting for start click…", round_num)
        try:
            run_mission(cfg, mjpeg_server=server)
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — exiting")
            return
        except Exception:
            log.exception("mission crashed — resetting for next round")
        # Between-round rearm: clears the start event so the next round
        # blocks on wait_for_start, but keeps the prior round's end modal
        # state visible to the page until the user clicks "Back to menu".
        server.rearm_start_event()
        round_num += 1


if __name__ == "__main__":
    main()
