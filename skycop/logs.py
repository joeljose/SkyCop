"""Logging configuration for SkyCop entrypoints.

Library modules just do `logger = logging.getLogger(__name__)` and emit —
they never configure handlers. Scripts call `setup_logging()` once at the
top of their `main()` to install a sensible console handler on the root
logger.

Output goes to stderr line-buffered (stdout is reserved for CARLA/Flask
framework chatter and any machine-readable output). Level defaults to
INFO and can be overridden via the `SKYCOP_LOG_LEVEL` env var.
"""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str | None = None) -> None:
    """Install a console handler on the root logger. Idempotent."""
    if level is None:
        level = os.environ.get("SKYCOP_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    # Remove any previously-installed handlers so repeated calls don't duplicate.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet known noisy loggers.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
