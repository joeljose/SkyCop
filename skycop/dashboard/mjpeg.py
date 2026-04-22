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
          font-family: ui-monospace, Menlo, Consolas, monospace;
          display: flex; align-items: center; justify-content: center;
          flex-direction: column; gap: 2rem; padding: 2rem; box-sizing: border-box; }}
  h1 {{ font-size: 2.4rem; margin: 0; color: #7fd; letter-spacing: 0.05em; }}
  .tagline {{ font-size: 0.95rem; color: #999; max-width: 30rem;
              text-align: center; line-height: 1.5; margin: 0; }}
  .modes {{ display: flex; gap: 1.5rem; flex-wrap: wrap; justify-content: center; }}
  .mode {{ background: #1b1b1b; border: 1px solid #333; border-radius: 0.75rem;
           padding: 1.5rem 1.75rem; width: 18rem; display: flex;
           flex-direction: column; gap: 0.75rem; }}
  .mode h2 {{ margin: 0; font-size: 1.15rem; color: #eee; }}
  .mode p  {{ margin: 0; font-size: 0.85rem; color: #aaa; line-height: 1.45; flex: 1; }}
  .mode button {{ background: #7fd; color: #111; border: 0; padding: 0.8rem 1rem;
                  font-size: 1rem; border-radius: 0.5rem; cursor: pointer;
                  font-family: inherit; letter-spacing: 0.04em; }}
  .mode button:hover {{ background: #4be; }}
  .mode.disabled {{ opacity: 0.55; }}
  .mode.disabled button {{ background: #333; color: #888; cursor: not-allowed; }}
  .mode.disabled button:hover {{ background: #333; }}
  .version {{ margin: 0; font-size: 0.75rem; color: #555; }}
</style>
</head><body>
<h1>{title}</h1>
<p class="tagline">Drone pursuit simulation. Pick a mode to begin. Mission ends when the suspect parks and the pursuit is confirmed — or when you lose them.</p>
<div class="modes">
  <div class="mode">
    <h2>AI Pursuit</h2>
    <p>The drone follows a reckless suspect autonomously using YOLO + ByteTrack + HSV re-ID. Watch the live camera feed while the flight PID keeps the suspect centred. Mission auto-ends on parked-car confirmation.</p>
    <form method="POST" action="/start"><button type="submit">Start AI Pursuit</button></form>
  </div>
  <div class="mode disabled">
    <h2>Manual Pursuit</h2>
    <p>Fly the drone yourself. Chase the suspect, draw a bounding box on the parked vehicle, submit. Coming in v0c — manual drone controls and the draw-bbox submission UI.</p>
    <button type="button" disabled>Coming in v0c</button>
  </div>
</div>
<p class="version">v0b · Mission SkyCop</p>
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
