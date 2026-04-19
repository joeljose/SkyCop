"""
Lesson 02 — Drone View

Fly around the CARLA world as a drone controlled from your browser.
Open http://localhost:5000 to see the live camera feed and use keyboard to move.

Controls:
  W/S     — move forward / backward
  A/D     — strafe left / right
  Q/E     — descend / ascend
  ↑/↓     — pitch camera up / down
  ←/→     — yaw camera left / right
  Shift   — hold for boost (3x speed)
  T       — spawn NPC traffic (toggle)
  1-5     — switch maps (Town01-05)

Architecture:
  - CARLA spectator camera = the "drone" (no physics, free movement)
  - Flask serves MJPEG stream + HTML page with keyboard capture
  - Browser sends key state via POST, Python moves the spectator
"""

import carla
import cv2
import math
import numpy as np
import os
import queue
import random
import threading
import time
from flask import Flask, Response, request, jsonify

CARLA_HOST = os.environ.get("CARLA_HOST", "carla-server")
CARLA_PORT = int(os.environ.get("CARLA_PORT", 2000))

# Movement settings
MOVE_SPEED = 0.8       # meters per tick
BOOST_MULTIPLIER = 3.0
ROTATE_SPEED = 2.0     # degrees per tick
ALTITUDE_SPEED = 0.5   # meters per tick

# Camera settings
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FOV = 90

# ── Global state ──────────────────────────────────────
keys_pressed = set()
latest_frame = None
frame_lock = threading.Lock()
traffic_spawned = False
npc_actors = []

app = Flask(__name__)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>CARLA Drone View</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; color: #eee; font-family: monospace; overflow: hidden; }
  #stream { width: 100vw; height: 100vh; object-fit: contain; display: block; }
  #hud {
    position: fixed; top: 10px; left: 10px;
    background: rgba(0,0,0,0.7); padding: 10px 14px; border-radius: 6px;
    font-size: 13px; line-height: 1.6; pointer-events: none;
  }
  #hud .key { color: #0ff; }
  #status {
    position: fixed; bottom: 10px; left: 10px;
    background: rgba(0,0,0,0.7); padding: 6px 12px; border-radius: 6px;
    font-size: 12px;
  }
  #status.active { color: #0f0; }
  #status.inactive { color: #f55; }
</style>
</head>
<body>
<img id="stream" src="/stream" />
<div id="hud">
  <b>CARLA Drone View</b><br>
  <span class="key">W/S</span> forward/back &nbsp;
  <span class="key">A/D</span> strafe<br>
  <span class="key">Q/E</span> down/up &nbsp;
  <span class="key">Shift</span> boost<br>
  <span class="key">Arrows</span> look around<br>
  <span class="key">T</span> toggle traffic &nbsp;
  <span class="key">1-5</span> switch map
</div>
<div id="status" class="inactive">Click here to capture keyboard</div>
<script>
const active = new Set();
let focused = false;

document.addEventListener('click', () => {
  focused = true;
  document.getElementById('status').textContent = 'Keyboard active';
  document.getElementById('status').className = 'active';
});

document.addEventListener('keydown', (e) => {
  if (!focused) return;
  e.preventDefault();
  const k = e.key.toLowerCase();
  if (!active.has(k)) {
    active.add(k);
    send();
  }
});

document.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  active.delete(k);
  send();
});

window.addEventListener('blur', () => {
  active.clear();
  send();
  focused = false;
  document.getElementById('status').textContent = 'Click to capture keyboard';
  document.getElementById('status').className = 'inactive';
});

function send() {
  fetch('/keys', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({keys: [...active]})
  }).catch(() => {});
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/keys", methods=["POST"])
def update_keys():
    global keys_pressed
    data = request.get_json()
    keys_pressed = set(data.get("keys", []))
    return jsonify(ok=True)


def generate_mjpeg():
    """Yield MJPEG frames from the latest CARLA camera capture."""
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        else:
            time.sleep(0.05)


@app.route("/stream")
def stream():
    return Response(
        generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def spawn_traffic(client, world, count=40):
    """Spawn NPC vehicles with autopilot."""
    global npc_actors
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    for sp in spawn_points[:count]:
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True, 8000)
            npc_actors.append(npc)

    print(f"  Spawned {len(npc_actors)} NPC vehicles")


def destroy_traffic():
    """Destroy all NPC actors."""
    global npc_actors
    for a in npc_actors:
        try:
            a.destroy()
        except Exception:
            pass
    count = len(npc_actors)
    npc_actors = []
    print(f"  Destroyed {count} NPC vehicles")


def carla_loop():
    """Main CARLA loop: move spectator based on key state, capture frames."""
    global latest_frame, traffic_spawned

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()

    print(f"Connected to CARLA — map: {world.get_map().name}")

    # Enable synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set up spectator (the "drone")
    spectator = world.get_spectator()

    # Start at a spawn point, elevated
    spawn_points = world.get_map().get_spawn_points()
    start = spawn_points[0]
    spectator.set_transform(carla.Transform(
        carla.Location(x=start.location.x, y=start.location.y, z=30.0),
        carla.Rotation(pitch=-30, yaw=start.rotation.yaw)
    ))

    # Attach a camera to the spectator for capturing frames
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
    cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
    cam_bp.set_attribute("fov", str(CAM_FOV))

    # Camera at spectator origin, looking same direction
    camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)
    img_queue = queue.Queue()
    camera.listen(img_queue.put)

    map_keys = {"1": "Town01", "2": "Town02", "3": "Town03", "4": "Town04", "5": "Town05"}

    print("Drone ready — open http://localhost:5000")

    try:
        while True:
            # ── Handle map switching ──
            for key, map_name in map_keys.items():
                if key in keys_pressed:
                    print(f"  Loading {map_name}...")
                    # Clean up before map change
                    camera.stop()
                    camera.destroy()
                    destroy_traffic()
                    traffic_spawned = False

                    world = client.load_world(map_name)
                    settings = world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                    world.apply_settings(settings)

                    spectator = world.get_spectator()
                    sp = world.get_map().get_spawn_points()[0]
                    spectator.set_transform(carla.Transform(
                        carla.Location(x=sp.location.x, y=sp.location.y, z=30.0),
                        carla.Rotation(pitch=-30, yaw=sp.rotation.yaw)
                    ))

                    bp_lib = world.get_blueprint_library()
                    cam_bp = bp_lib.find("sensor.camera.rgb")
                    cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
                    cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
                    cam_bp.set_attribute("fov", str(CAM_FOV))
                    camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)
                    img_queue = queue.Queue()
                    camera.listen(img_queue.put)
                    print(f"  Loaded {map_name}")
                    keys_pressed.discard(key)
                    break

            # ── Handle traffic toggle ──
            if "t" in keys_pressed:
                keys_pressed.discard("t")
                if traffic_spawned:
                    destroy_traffic()
                    traffic_spawned = False
                else:
                    spawn_traffic(client, world)
                    traffic_spawned = True

            # ── Movement ──
            t = spectator.get_transform()
            loc = t.location
            rot = t.rotation

            speed = MOVE_SPEED
            if "shift" in keys_pressed:
                speed *= BOOST_MULTIPLIER

            # Forward vector from yaw
            yaw_rad = math.radians(rot.yaw)
            fwd_x = math.cos(yaw_rad)
            fwd_y = math.sin(yaw_rad)
            # Right vector (perpendicular)
            right_x = -fwd_y
            right_y = fwd_x

            # WASD movement (horizontal plane)
            if "w" in keys_pressed:
                loc.x += fwd_x * speed
                loc.y += fwd_y * speed
            if "s" in keys_pressed:
                loc.x -= fwd_x * speed
                loc.y -= fwd_y * speed
            if "a" in keys_pressed:
                loc.x -= right_x * speed
                loc.y -= right_y * speed
            if "d" in keys_pressed:
                loc.x += right_x * speed
                loc.y += right_y * speed

            # Q/E altitude
            if "e" in keys_pressed:
                loc.z += ALTITUDE_SPEED * (BOOST_MULTIPLIER if "shift" in keys_pressed else 1)
            if "q" in keys_pressed:
                loc.z -= ALTITUDE_SPEED * (BOOST_MULTIPLIER if "shift" in keys_pressed else 1)
            loc.z = max(2.0, loc.z)  # floor

            # Arrow keys for looking
            if "arrowleft" in keys_pressed:
                rot.yaw -= ROTATE_SPEED
            if "arrowright" in keys_pressed:
                rot.yaw += ROTATE_SPEED
            if "arrowup" in keys_pressed:
                rot.pitch = max(-89, rot.pitch - ROTATE_SPEED)
            if "arrowdown" in keys_pressed:
                rot.pitch = min(89, rot.pitch + ROTATE_SPEED)

            spectator.set_transform(carla.Transform(loc, rot))

            # ── Tick and capture ──
            world.tick()

            try:
                image = img_queue.get(timeout=1.0)
                # Convert CARLA image (BGRA) to BGR for OpenCV
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
                with frame_lock:
                    latest_frame = arr
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        camera.destroy()
        destroy_traffic()
        world.apply_settings(original_settings)
        print("Drone stopped.")


def main():
    # Start CARLA loop in background thread
    t = threading.Thread(target=carla_loop, daemon=True)
    t.start()

    # Start Flask (blocks)
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
