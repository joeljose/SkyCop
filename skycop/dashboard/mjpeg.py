"""Flask-based MJPEG streamer for live CARLA camera frames.

Usage:

    server = MJPEGServer(title="Lesson 03 — suspect follow")
    server.start(port=5000)
    # ... in a CARLA loop:
    server.push(frame_bgr)

Scripts that need extra routes (e.g. keyboard capture) can attach them to
`server.app` before calling `start()`, or pass a custom `html` to override
the index page.

The server runs Flask on a daemon thread so the caller's main loop keeps
ticking CARLA. Frames are held behind a lock and served on `/stream`.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from flask import Flask, Response

_DEFAULT_HTML = """<!DOCTYPE html>
<html><head><title>{title}</title>
<style>
  body {{ margin: 0; background: #111; color: #eee; font-family: monospace; }}
  img  {{ width: 100vw; height: 100vh; object-fit: contain; display: block; }}
  #hud {{ position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.7);
          padding: 8px 12px; border-radius: 6px; font-size: 13px; }}
</style>
</head><body>
<img src="/stream"/>
<div id="hud">{hud}</div>
</body></html>"""


class MJPEGServer:
    """Minimal MJPEG server. One stream, one frame buffer."""

    def __init__(
        self,
        title: str = "SkyCop",
        hud: str = "",
        html: str | None = None,
    ) -> None:
        self._title = title
        self._hud = hud or title
        self._html = html
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self.app = Flask(__name__)
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.route("/")
        def _index() -> str:
            if self._html is not None:
                return self._html
            return _DEFAULT_HTML.format(title=self._title, hud=self._hud)

        @self.app.route("/stream")
        def _stream() -> Response:
            return Response(
                self._generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def _generate(self):
        while True:
            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )

    def push(self, frame: np.ndarray) -> None:
        """Set the latest frame. Expected BGR uint8."""
        with self._lock:
            self._frame = frame

    def start(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        """Start Flask on a daemon thread."""
        def _run() -> None:
            self.app.run(host=host, port=port, threaded=True, use_reloader=False)
        threading.Thread(target=_run, daemon=True).start()
