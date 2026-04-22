"""Flask MJPEG streamer + menu + user-mode game server.

Serves three surfaces in v0c:

- ``/`` — splash menu with two modes (AI / Manual) until ``POST /start``
  fires, then the per-mode live view. Selected mode is captured in the
  form field ``mode`` (``ai`` or ``user``).
- ``/stream`` — MJPEG multipart stream of the latest camera frame.
- ``/ws/input`` — WebSocket for keydown/keyup events in manual mode.
  The mission loop reads the current pressed-key set each tick.
- ``POST /submit`` — user-mode bbox submission (JSON body
  ``{"bbox": [x1, y1, x2, y2]}``). Mission loop polls ``get_submission()``.
- ``GET /state`` — JSON snapshot of FSM state + countdown + terminal result,
  used by the manual page to colour its state panel and drive the submit
  button / end modal.

Threading: Flask runs on a daemon thread. Frames, pressed-key set, game
state, and submission slot are all held behind the same ``_lock`` so the
mission loop can snapshot them without racing WebSocket / HTTP handlers.
"""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, redirect, request, url_for
from flask_sock import Sock

_LIVE_AI_HTML = """<!DOCTYPE html>
<html><head><title>{title} — AI Pursuit</title>
<style>
  body {{ margin: 0; background: #111; color: #eee;
          font-family: ui-monospace, Menlo, Consolas, monospace; overflow: hidden; }}
  img  {{ width: 100vw; height: 100vh; object-fit: contain; display: block; }}
  #border {{ position: fixed; inset: 0; pointer-events: none; z-index: 30;
             box-sizing: border-box; border: 8px solid transparent;
             transition: border-color 0.3s ease; }}
  #border.fleeing {{ border-color: #e14; }}
  #border.roaming {{ border-color: #f90; }}
  #border.parking {{ border-color: #fc3; }}
  #border.parked  {{ border-color: #4e8; }}
  #panel {{ position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
            background: rgba(0,0,0,0.85); padding: 14px 28px; border-radius: 10px;
            font-size: 16px; display: flex; flex-direction: column; gap: 6px;
            align-items: center; min-width: 320px; border: 3px solid #333;
            text-align: center; }}
  #panel .state {{ font-size: 28px; font-weight: bold; letter-spacing: 0.12em; }}
  #panel .hint  {{ font-size: 14px; color: #ccc; }}
  #panel.fleeing {{ border-color: #e14; background: rgba(225, 17, 68, 0.15); }}
    #panel.fleeing .state {{ color: #f66; }}
  #panel.roaming {{ border-color: #f90; background: rgba(255, 153, 0, 0.15); }}
    #panel.roaming .state {{ color: #fa4; }}
  #panel.parking {{ border-color: #fc3; background: rgba(255, 204, 51, 0.15); }}
    #panel.parking .state {{ color: #fd6; }}
  #panel.parked  {{ border-color: #4e8; background: rgba(68, 238, 136, 0.2); }}
    #panel.parked  .state {{ color: #6f8; }}
  #flash {{ position: fixed; top: 38%; left: 50%;
            transform: translate(-50%, -50%) scale(0.9);
            padding: 28px 68px; border-radius: 14px; font-size: 48px;
            font-weight: bold; letter-spacing: 0.15em;
            background: rgba(0,0,0,0.7); border: 4px solid;
            opacity: 0; pointer-events: none; z-index: 40;
            transition: opacity 0.25s ease, transform 0.25s ease; }}
  #flash.show {{ opacity: 1; transform: translate(-50%, -50%) scale(1.0); }}
  #flash.fleeing {{ border-color: #e14; color: #f66; }}
  #flash.roaming {{ border-color: #f90; color: #fa4; }}
  #flash.parking {{ border-color: #fc3; color: #fd6; }}
  #flash.parked  {{ border-color: #4e8; color: #6f8; }}
  #end {{ position: fixed; inset: 0; background: rgba(0,0,0,0.85);
          display: none; align-items: center; justify-content: center;
          flex-direction: column; gap: 1.5rem; z-index: 50; }}
  #end.show {{ display: flex; }}
  #end h1 {{ margin: 0; font-size: 2.2rem; }}
  #end h1.win {{ color: #7fd; }} #end h1.lose {{ color: #f66; }}
  #end p  {{ margin: 0; color: #ccc; max-width: 30rem; text-align: center; }}
  #end a  {{ background: #7fd; color: #111; padding: 0.75rem 1.75rem;
             text-decoration: none; border-radius: 0.5rem; font-family: inherit;
             font-weight: bold; letter-spacing: 0.05em; }}
</style>
</head><body>
<img src="/stream"/>
<div id="border" class="fleeing"></div>
<div id="panel" class="fleeing">
  <div class="state">FLEEING</div>
  <div class="hint">Autonomous pursuit · {hud}</div>
</div>
<div id="flash" class="fleeing">FLEEING</div>
<div id="end">
  <h1 id="end-title">—</h1>
  <p id="end-detail">—</p>
  <a href="/menu">Back to menu</a>
</div>
<script>
const panel = document.getElementById('panel');
const border = document.getElementById('border');
const flash = document.getElementById('flash');
const stateLabel = panel.querySelector('.state');
const hintLabel = panel.querySelector('.hint');
const endPanel = document.getElementById('end');
const endTitle = document.getElementById('end-title');
const endDetail = document.getElementById('end-detail');
const hints = {{
  fleeing: 'Suspect fleeing — drone pursuing',
  roaming: 'Suspect roaming — drone pursuing',
  parking: 'Suspect parking — drone holding',
  parked:  'Suspect parked — AI confirming',
}};
const flashLabels = {{
  fleeing: 'FLEEING', roaming: 'ROAMING', parking: 'PARKING', parked: 'PARKED',
}};
let lastState = null;
let flashTimer = null;
function showFlash(newState) {{
  flash.textContent = flashLabels[newState] || newState.toUpperCase();
  flash.className = newState;
  void flash.offsetWidth;
  flash.classList.add('show');
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {{ flash.classList.remove('show'); }}, 1200);
}}
async function pollState() {{
  try {{
    const r = await fetch('/state'); const s = await r.json();
    stateLabel.textContent = s.state.toUpperCase();
    hintLabel.textContent = hints[s.state] || '';
    if (s.state === 'parked' && s.countdown_s != null) {{
      hintLabel.textContent = `AI confirming · ${{s.countdown_s.toFixed(1)}}s`;
    }}
    panel.className = s.state;
    border.className = s.state;
    if (lastState !== null && s.state !== lastState && !s.terminal) {{
      showFlash(s.state);
    }}
    lastState = s.state;
    if (s.terminal) {{
      endPanel.classList.add('show');
      const win = s.result === 'ai_pass';
      endTitle.textContent = win ? 'Pursuit confirmed' : 'Suspect lost';
      endTitle.className = win ? 'win' : 'lose';
      const detail = {{
        ai_pass: 'AI locked the parked suspect. Autonomous pursuit succeeded.',
        timeout_lose: 'The 60-second confirmation window expired with no lock. The suspect is gone.',
      }};
      endDetail.textContent = detail[s.result] || 'Round ended.';
    }} else {{
      endPanel.classList.remove('show');
    }}
  }} catch (e) {{}}
}}
setInterval(pollState, 250);
pollState();
</script>
</body></html>"""

_LIVE_USER_HTML = """<!DOCTYPE html>
<html><head><title>{title} — Manual Pursuit</title>
<style>
  body {{ margin: 0; background: #111; color: #eee;
          font-family: ui-monospace, Menlo, Consolas, monospace; overflow: hidden; }}
  #stage {{ position: relative; width: 100vw; height: 100vh; }}
  img  {{ width: 100%; height: 100%; object-fit: contain; display: block;
          position: absolute; inset: 0; }}
  canvas {{ position: absolute; inset: 0; width: 100%; height: 100%;
            cursor: crosshair; pointer-events: none; }}
  canvas.active {{ pointer-events: auto; }}
  /* Full-viewport coloured border — always visible, matches suspect state. */
  #border {{ position: fixed; inset: 0; pointer-events: none; z-index: 30;
             box-sizing: border-box; border: 8px solid transparent;
             transition: border-color 0.3s ease; }}
  #border.fleeing {{ border-color: #e14; }}
  #border.roaming {{ border-color: #f90; }}
  #border.parking {{ border-color: #fc3; }}
  #border.parked  {{ border-color: #4e8; }}
  /* Top-centre state banner — readable from the middle of the screen. */
  #panel {{ position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
            background: rgba(0,0,0,0.85); padding: 14px 28px; border-radius: 10px;
            font-size: 16px; display: flex; flex-direction: column; gap: 6px;
            align-items: center; min-width: 320px; border: 3px solid #333;
            text-align: center; }}
  #panel .state {{ font-size: 28px; font-weight: bold; letter-spacing: 0.12em; }}
  #panel .hint  {{ font-size: 14px; color: #ccc; }}
  #panel.fleeing {{ border-color: #e14; background: rgba(225, 17, 68, 0.15); }}
    #panel.fleeing .state {{ color: #f66; }}
  #panel.roaming {{ border-color: #f90; background: rgba(255, 153, 0, 0.15); }}
    #panel.roaming .state {{ color: #fa4; }}
  #panel.parking {{ border-color: #fc3; background: rgba(255, 204, 51, 0.15); }}
    #panel.parking .state {{ color: #fd6; }}
  #panel.parked  {{ border-color: #4e8; background: rgba(68, 238, 136, 0.2); }}
    #panel.parked  .state {{ color: #6f8; }}
  /* Transition flash — big centred banner for ~1 s on state change. */
  #flash {{ position: fixed; top: 38%; left: 50%;
            transform: translate(-50%, -50%) scale(0.9);
            padding: 28px 68px; border-radius: 14px; font-size: 48px;
            font-weight: bold; letter-spacing: 0.15em;
            background: rgba(0,0,0,0.7); border: 4px solid;
            opacity: 0; pointer-events: none; z-index: 40;
            transition: opacity 0.25s ease, transform 0.25s ease; }}
  #flash.show {{ opacity: 1; transform: translate(-50%, -50%) scale(1.0); }}
  #flash.fleeing {{ border-color: #e14; color: #f66; }}
  #flash.roaming {{ border-color: #f90; color: #fa4; }}
  #flash.parking {{ border-color: #fc3; color: #fd6; }}
  #flash.parked  {{ border-color: #4e8; color: #6f8; }}
  #controls {{ position: fixed; top: 12px; right: 12px;
               background: rgba(0,0,0,0.75); padding: 10px 14px;
               border-radius: 8px; font-size: 12px; color: #bbb; line-height: 1.7; }}
  #submit {{ position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
             padding: 12px 32px; font-size: 16px; font-family: inherit;
             background: #333; color: #888; border: 0; border-radius: 6px;
             cursor: not-allowed; letter-spacing: 0.05em; }}
  #submit.ready {{ background: #7fd; color: #111; cursor: pointer; }}
  #submit.ready:hover {{ background: #4be; }}
  #end {{ position: fixed; inset: 0; background: rgba(0,0,0,0.85);
          display: none; align-items: center; justify-content: center;
          flex-direction: column; gap: 1.5rem; z-index: 50; }}
  #end.show {{ display: flex; }}
  #end h1 {{ margin: 0; font-size: 2.2rem; }}
  #end h1.win {{ color: #7fd; }} #end h1.lose {{ color: #f66; }}
  #end p  {{ margin: 0; color: #ccc; max-width: 30rem; text-align: center; }}
  #end a  {{ background: #7fd; color: #111; padding: 0.75rem 1.75rem;
             text-decoration: none; border-radius: 0.5rem; font-family: inherit;
             font-weight: bold; letter-spacing: 0.05em; }}
</style>
</head><body>
<div id="stage">
  <img src="/stream"/>
  <canvas id="canvas"></canvas>
</div>
<div id="border" class="fleeing"></div>
<div id="panel" class="fleeing">
  <div class="state">FLEEING</div>
  <div class="hint">connecting…</div>
</div>
<div id="flash" class="fleeing">FLEEING</div>
<div id="controls">
  <b>W/S</b> forward · back<br>
  <b>A/D</b> strafe left · right<br>
  <b>Q/E</b> yaw left · right<br>
  <b>Shift/Ctrl</b> ascend · descend<br>
  <br><b>PARKED</b>: draw bbox · click Submit
</div>
<button id="submit" disabled>Submit (wait for PARKED)</button>
<div id="end">
  <h1 id="end-title">—</h1>
  <p id="end-detail">—</p>
  <a href="/menu">Back to menu</a>
</div>

<script>
// ── WebSocket input ───────────────────────────────────────────────
const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/input');
const sendKey = (type, key) => {{
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({{type, key}}));
}};
const trackedKeys = new Set(['w', 'a', 's', 'd', 'q', 'e', 'shift', 'control']);
const normKey = (e) => {{
  let k = e.key.toLowerCase();
  if (k === 'w' || k === 'a' || k === 's' || k === 'd' || k === 'q' || k === 'e') return k;
  if (k === 'shift') return 'shift';
  if (k === 'control') return 'control';
  return null;
}};
window.addEventListener('keydown', e => {{
  const k = normKey(e); if (k && trackedKeys.has(k)) {{ sendKey('keydown', k); e.preventDefault(); }}
}});
window.addEventListener('keyup', e => {{
  const k = normKey(e); if (k && trackedKeys.has(k)) {{ sendKey('keyup', k); e.preventDefault(); }}
}});
// Release all keys on blur so focus loss doesn't strand a held key
window.addEventListener('blur', () => trackedKeys.forEach(k => sendKey('keyup', k)));

// ── Bbox canvas ───────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let bbox = null, dragging = false, dragStart = null;
const fitCanvas = () => {{
  canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
  redraw();
}};
window.addEventListener('resize', fitCanvas);
const redraw = () => {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (bbox) {{
    ctx.strokeStyle = '#4e8'; ctx.lineWidth = 3;
    ctx.strokeRect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]);
    ctx.fillStyle = 'rgba(78, 232, 136, 0.12)';
    ctx.fillRect(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]);
  }}
}};
canvas.addEventListener('mousedown', e => {{
  if (!canvas.classList.contains('active')) return;
  const r = canvas.getBoundingClientRect();
  dragStart = [e.clientX - r.left, e.clientY - r.top];
  dragging = true; bbox = null;
}});
canvas.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const r = canvas.getBoundingClientRect();
  const now = [e.clientX - r.left, e.clientY - r.top];
  bbox = [Math.min(dragStart[0], now[0]), Math.min(dragStart[1], now[1]),
          Math.max(dragStart[0], now[0]), Math.max(dragStart[1], now[1])];
  redraw();
}});
canvas.addEventListener('mouseup', () => {{ dragging = false; }});
fitCanvas();

// ── Submit ───────────────────────────────────────────────────────
const submit = document.getElementById('submit');
submit.addEventListener('click', async () => {{
  if (!submit.classList.contains('ready') || !bbox) return;
  // Convert canvas pixel coords → image pixel coords ({width}×{height}).
  const sx = {width} / canvas.clientWidth;
  const sy = {height} / canvas.clientHeight;
  const imgBbox = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy];
  submit.disabled = true;
  await fetch('/submit', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{bbox: imgBbox}}),
  }});
}});

// ── Game state polling ───────────────────────────────────────────
const panel = document.getElementById('panel');
const border = document.getElementById('border');
const flash = document.getElementById('flash');
const stateLabel = panel.querySelector('.state');
const hintLabel = panel.querySelector('.hint');
const endPanel = document.getElementById('end');
const endTitle = document.getElementById('end-title');
const endDetail = document.getElementById('end-detail');
const hints = {{
  fleeing: 'Suspect fleeing — keep visual',
  roaming: 'Suspect roaming — maintain pursuit',
  parking: 'Suspect parking — hold position',
  parked:  'Draw bbox · submit',
}};
const flashLabels = {{
  fleeing: 'FLEEING', roaming: 'ROAMING', parking: 'PARKING', parked: 'PARKED',
}};
let lastState = null;
let flashTimer = null;
function showFlash(newState) {{
  flash.textContent = flashLabels[newState] || newState.toUpperCase();
  flash.className = newState;
  // trigger reflow so the 'show' transition always plays
  void flash.offsetWidth;
  flash.classList.add('show');
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {{ flash.classList.remove('show'); }}, 1200);
}}
async function pollState() {{
  try {{
    const r = await fetch('/state'); const s = await r.json();
    stateLabel.textContent = s.state.toUpperCase();
    hintLabel.textContent = hints[s.state] || '';
    if (s.state === 'parked' && s.countdown_s != null) {{
      hintLabel.textContent = `Draw bbox · submit · ${{s.countdown_s.toFixed(1)}}s`;
    }}
    panel.className = s.state;
    border.className = s.state;
    if (lastState !== null && s.state !== lastState && !s.terminal) {{
      showFlash(s.state);
    }}
    lastState = s.state;
    // Canvas + submit enable only when PARKED
    if (s.state === 'parked' && !s.terminal) {{
      canvas.classList.add('active');
      submit.classList.add('ready'); submit.disabled = false;
      submit.textContent = 'Submit bbox';
    }} else {{
      canvas.classList.remove('active');
      submit.classList.remove('ready'); submit.disabled = true;
      submit.textContent = s.terminal ? 'Round over' : 'Submit (wait for PARKED)';
    }}
    if (s.terminal) {{
      endPanel.classList.add('show');
      const win = s.result === 'user_pass';
      endTitle.textContent = win ? 'Pursuit confirmed' : 'Suspect lost';
      endTitle.className = win ? 'win' : 'lose';
      const detail = {{
        user_pass: 'Your bbox covered the parked suspect. Good pursuit.',
        user_fail: 'Bbox was drawn but did not cover the suspect. Try again.',
        timeout_lose: 'The 60-second confirmation window expired with no submission. The suspect is gone.',
      }};
      endDetail.textContent = detail[s.result] || 'Round ended.';
    }} else {{
      // A fresh round was started (terminal cleared by the server) — hide
      // the end modal so the new round is visible.
      endPanel.classList.remove('show');
    }}
  }} catch (e) {{ /* server restarting or paused — keep polling */ }}
}}
setInterval(pollState, 250);
pollState();
</script>
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
    <form method="POST" action="/start"><input type="hidden" name="mode" value="ai"/>
      <button type="submit">Start AI Pursuit</button>
    </form>
  </div>
  <div class="mode">
    <h2>Manual Pursuit</h2>
    <p>Fly the drone yourself with WASD + QE + Shift/Ctrl. Chase the suspect, then when it parks draw a bounding box and submit within 60 seconds.</p>
    <form method="POST" action="/start"><input type="hidden" name="mode" value="user"/>
      <button type="submit">Start Manual Pursuit</button>
    </form>
  </div>
</div>
<p class="version">v0c · Mission SkyCop</p>
</body></html>"""


class MJPEGServer:
    """MJPEG stream + menu + v0c user-mode game server."""

    def __init__(
        self,
        title: str = "SkyCop",
        hud: str = "",
        html: str | None = None,
        use_start_trigger: bool = False,
        image_size: tuple[int, int] = (720, 1280),  # (h, w) — matches camera
    ) -> None:
        self._title = title
        self._hud = hud or title
        self._html = html
        self._image_size = image_size
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()

        # Start-trigger gate (v0a).
        self._use_start_trigger = use_start_trigger
        self._start_event = threading.Event()
        if not use_start_trigger:
            self._start_event.set()
        self._started_mode: str = "ai"  # set on /start

        # Per-tick FSM snapshot the page polls via GET /state (v0c).
        self._fsm_state: str = "fleeing"
        self._fsm_countdown_s: float | None = None
        self._fsm_terminal: bool = False
        self._fsm_result: str | None = None

        # User-mode input + submission slots (v0c).
        self._pressed: set[str] = set()
        self._submission: dict | None = None

        self.app = Flask(__name__)
        self._sock = Sock(self.app)
        self._register_routes()

    # ── Flask wiring ────────────────────────────────────────────────

    def _register_routes(self) -> None:
        @self.app.route("/")
        def _index() -> str:
            if self._use_start_trigger and not self._start_event.is_set():
                return _SPLASH_HTML.format(title=self._title)
            if self._html is not None:
                return self._html
            if self._started_mode == "user":
                return _LIVE_USER_HTML.format(
                    title=self._title,
                    width=self._image_size[1], height=self._image_size[0],
                )
            return _LIVE_AI_HTML.format(title=self._title, hud=self._hud)

        @self.app.route("/start", methods=["POST"])
        def _start() -> Response:
            mode = request.form.get("mode", "ai").strip().lower()
            if mode not in ("ai", "user"):
                mode = "ai"
            with self._lock:
                self._started_mode = mode
            self._start_event.set()
            return redirect(url_for("_index"))

        @self.app.route("/menu")
        def _menu() -> Response:
            # "Back to menu" link in the end modal — force-rearm the server
            # so the splash shows even if the previous round hasn't fully
            # torn down yet, then redirect. Eliminates a race where the
            # browser would land on the stale live page.
            self.reset_for_menu()
            return redirect(url_for("_index"))

        @self.app.route("/stream")
        def _stream() -> Response:
            return Response(
                self._generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/state")
        def _state() -> Response:
            with self._lock:
                payload = {
                    "state": self._fsm_state,
                    "countdown_s": self._fsm_countdown_s,
                    "terminal": self._fsm_terminal,
                    "result": self._fsm_result,
                }
            return Response(json.dumps(payload), mimetype="application/json")

        @self.app.route("/submit", methods=["POST"])
        def _submit() -> Response:
            body = request.get_json(silent=True) or {}
            bbox = body.get("bbox")
            if not (
                isinstance(bbox, list) and len(bbox) == 4
                and all(isinstance(v, int | float) for v in bbox)
            ):
                return Response('{"ok":false,"error":"malformed bbox"}',
                                status=400, mimetype="application/json")
            with self._lock:
                self._submission = {
                    "bbox": [float(v) for v in bbox],
                    "ts": time.time(),
                }
            return Response('{"ok":true}', mimetype="application/json")

        @self._sock.route("/ws/input")
        def _ws_input(ws) -> None:  # noqa: ANN001 - flask_sock's ws type is dynamic
            try:
                while True:
                    msg = ws.receive()
                    if msg is None:
                        break
                    try:
                        ev = json.loads(msg)
                    except Exception:
                        continue
                    key = str(ev.get("key", "")).lower()
                    evtype = ev.get("type")
                    if not key:
                        continue
                    with self._lock:
                        if evtype == "keydown":
                            self._pressed.add(key)
                        elif evtype == "keyup":
                            self._pressed.discard(key)
            except Exception:
                return

    # ── Public API (mission loop) ───────────────────────────────────

    def wait_for_start(self, timeout: float | None = None) -> bool:
        return self._start_event.wait(timeout=timeout)

    @property
    def started(self) -> bool:
        return self._start_event.is_set()

    @property
    def started_mode(self) -> str:
        with self._lock:
            return self._started_mode

    def set_fsm_state(
        self,
        state: str,
        countdown_s: float | None,
        terminal: bool = False,
        result: str | None = None,
    ) -> None:
        """Mission loop reports FSM state each tick; the page polls /state."""
        with self._lock:
            self._fsm_state = state
            self._fsm_countdown_s = countdown_s
            self._fsm_terminal = terminal
            self._fsm_result = result

    def get_pressed_keys(self) -> frozenset[str]:
        """Snapshot of keys currently held by the user."""
        with self._lock:
            return frozenset(self._pressed)

    def get_submission(self) -> dict | None:
        """Most recent bbox submission, if any. Mission loop clears it."""
        with self._lock:
            return self._submission

    def clear_submission(self) -> None:
        with self._lock:
            self._submission = None

    def rearm_start_event(self) -> None:
        """Between-round rearm: clear the start event, drop any held keys,
        and clear any pending submission — but **keep** the FSM snapshot
        (state / terminal / result) so the prior round's end modal stays
        visible until the user explicitly clicks "Back to menu".
        """
        with self._lock:
            self._start_event.clear()
            self._pressed.clear()
            self._submission = None
            self._started_mode = "ai"

    def reset_for_menu(self) -> None:
        """Full reset — event + input + FSM snapshot + mode. Called from
        the /menu route when the user explicitly chooses to return to the
        splash. Causes any page on the live view to see fresh state."""
        with self._lock:
            self._start_event.clear()
            self._pressed.clear()
            self._submission = None
            self._fsm_state = "fleeing"
            self._fsm_countdown_s = None
            self._fsm_terminal = False
            self._fsm_result = None
            self._started_mode = "ai"

    # ── MJPEG plumbing ──────────────────────────────────────────────

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
        with self._lock:
            self._frame = frame

    def start(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        def _run() -> None:
            self.app.run(host=host, port=port, threaded=True, use_reloader=False)
        threading.Thread(target=_run, daemon=True).start()
