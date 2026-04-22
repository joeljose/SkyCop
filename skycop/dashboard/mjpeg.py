"""Flask-based MJPEG streamer for live CARLA camera frames.

Usage:

    server = MJPEGServer(title="Experiment 03 — suspect follow")
    server.start(port=5000)
    # ... in a CARLA loop:
    server.push(frame_bgr)

Scripts that need extra routes (e.g. keyboard capture) can attach them to
`server.app` before calling `start()`, or pass a custom `html` to override
the index page.

The server runs Flask on a daemon thread so the caller's main loop keeps
ticking CARLA. Frames are held behind a lock and served on `/stream`.

Optional start trigger (v0a, issue #44): construct with ``use_start_trigger=True``.
The index page renders a *splash* with a "Start pursuit" button until the
user clicks it (``POST /start``). ``wait_for_start(timeout)`` blocks until
fired — the mission loop calls this before spawning NPCs so you can open
the page before any world ticks happen.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, redirect, url_for

_LIVE_HTML = """<!DOCTYPE html>
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

_SPLASH_HTML = """<!DOCTYPE html>
<html><head><title>{title}</title>
<style>
  body {{ margin: 0; min-height: 100vh; background: #111; color: #eee;
          font-family: monospace; display: flex; align-items: center;
          justify-content: center; flex-direction: column; gap: 1.5rem; }}
  h1 {{ font-size: 2rem; margin: 0; color: #7fd; }}
  p  {{ font-size: 1rem; color: #aaa; max-width: 28rem; text-align: center; }}
  form button {{ background: #7fd; color: #111; border: 0; padding: 1rem 2rem;
                 font-size: 1.2rem; border-radius: 0.5rem; cursor: pointer;
                 font-family: inherit; letter-spacing: 0.05em; }}
  form button:hover {{ background: #4be; }}
</style>
</head><body>
<h1>{title}</h1>
<p>{hud}</p>
<form method="POST" action="/start"><button type="submit">Start pursuit</button></form>
</body></html>"""


class MJPEGServer:
    """Minimal MJPEG server. One stream, one frame buffer, optional start trigger."""

    def __init__(
        self,
        title: str = "SkyCop",
        hud: str = "",
        html: str | None = None,
        use_start_trigger: bool = False,
    ) -> None:
        self._title = title
        self._hud = hud or title
        self._html = html
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._use_start_trigger = use_start_trigger
        self._start_event = threading.Event()
        if not use_start_trigger:
            # Backwards-compatible: servers without a splash just start already-ready.
            self._start_event.set()
        self.app = Flask(__name__)
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.route("/")
        def _index() -> str:
            if self._use_start_trigger and not self._start_event.is_set():
                return _SPLASH_HTML.format(title=self._title, hud=self._hud)
            if self._html is not None:
                return self._html
            return _LIVE_HTML.format(title=self._title, hud=self._hud)

        @self.app.route("/start", methods=["POST"])
        def _start() -> Response:
            self._start_event.set()
            return redirect(url_for("_index"))

        @self.app.route("/stream")
        def _stream() -> Response:
            return Response(
                self._generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def wait_for_start(self, timeout: float | None = None) -> bool:
        """Block until the Start button is pressed (or ``timeout`` expires).

        Returns ``True`` if started, ``False`` on timeout. When the server
        was constructed without ``use_start_trigger=True`` the event is
        pre-set so this returns immediately.
        """
        return self._start_event.wait(timeout=timeout)

    @property
    def started(self) -> bool:
        return self._start_event.is_set()

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
