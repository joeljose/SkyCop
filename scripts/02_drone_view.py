"""
Experiment 02 — Drone View

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
"""

import logging
import math
import queue
import random
import threading
import time

import carla
from flask import jsonify, request

from skycop.dashboard import MJPEGServer
from skycop.logs import setup_logging
from skycop.sim import carla_image_to_bgr, connect, spawn_aerial_camera, synchronous_mode

log = logging.getLogger("exp02")

# Movement settings
MOVE_SPEED = 0.8       # metres per tick
BOOST_MULTIPLIER = 3.0
ROTATE_SPEED = 2.0     # degrees per tick
ALTITUDE_SPEED = 0.5   # metres per tick

# Camera settings
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FOV = 90
FIXED_DT = 0.05

keys_pressed: set[str] = set()
traffic_spawned = False
npc_actors: list[carla.Actor] = []


def update_keys():
    global keys_pressed
    data = request.get_json()
    keys_pressed = set(data.get("keys", []))
    return jsonify(ok=True)


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>SkyCop — Experiment 02 · Drone View</title>
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
  if (!active.has(k)) { active.add(k); send(); }
});

document.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  active.delete(k); send();
});

window.addEventListener('blur', () => {
  active.clear(); send();
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
</html>"""


def spawn_traffic(client, world, count=40):
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

    log.info("spawned %d NPC vehicles", len(npc_actors))


def destroy_traffic():
    global npc_actors
    for a in npc_actors:
        try:
            a.destroy()
        except Exception:
            pass
    count = len(npc_actors)
    npc_actors = []
    log.info("destroyed %d NPC vehicles", count)


def reset_camera(world, spectator):
    camera, q = spawn_aerial_camera(
        world,
        width=CAM_WIDTH, height=CAM_HEIGHT, fov=CAM_FOV,
        attach_to=spectator,
        transform=carla.Transform(),
    )
    return camera, q


def carla_loop(server: MJPEGServer):
    global traffic_spawned

    client = connect()
    world = client.get_world()
    log.info("connected to CARLA — map: %s", world.get_map().name)

    with synchronous_mode(world, FIXED_DT):
        spectator = world.get_spectator()
        spawn_points = world.get_map().get_spawn_points()
        start = spawn_points[0]
        spectator.set_transform(carla.Transform(
            carla.Location(x=start.location.x, y=start.location.y, z=30.0),
            carla.Rotation(pitch=-30, yaw=start.rotation.yaw),
        ))

        camera, img_queue = reset_camera(world, spectator)
        map_keys = {"1": "Town01", "2": "Town02", "3": "Town03", "4": "Town04", "5": "Town05"}
        log.info("drone ready — open http://localhost:5000")

        try:
            while True:
                # Map switch
                for key, map_name in map_keys.items():
                    if key in keys_pressed:
                        log.info("loading %s…", map_name)
                        camera.stop()
                        camera.destroy()
                        destroy_traffic()
                        traffic_spawned = False
                        world = client.load_world(map_name)
                        s = world.get_settings()
                        s.synchronous_mode = True
                        s.fixed_delta_seconds = FIXED_DT
                        world.apply_settings(s)
                        spectator = world.get_spectator()
                        sp = world.get_map().get_spawn_points()[0]
                        spectator.set_transform(carla.Transform(
                            carla.Location(x=sp.location.x, y=sp.location.y, z=30.0),
                            carla.Rotation(pitch=-30, yaw=sp.rotation.yaw),
                        ))
                        camera, img_queue = reset_camera(world, spectator)
                        log.info("loaded %s", map_name)
                        keys_pressed.discard(key)
                        break

                # Traffic toggle
                if "t" in keys_pressed:
                    keys_pressed.discard("t")
                    if traffic_spawned:
                        destroy_traffic()
                        traffic_spawned = False
                    else:
                        spawn_traffic(client, world)
                        traffic_spawned = True

                # Movement
                t = spectator.get_transform()
                loc = t.location
                rot = t.rotation

                speed = MOVE_SPEED * (BOOST_MULTIPLIER if "shift" in keys_pressed else 1)
                yaw_rad = math.radians(rot.yaw)
                fwd_x, fwd_y = math.cos(yaw_rad), math.sin(yaw_rad)
                right_x, right_y = -fwd_y, fwd_x

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

                alt = ALTITUDE_SPEED * (BOOST_MULTIPLIER if "shift" in keys_pressed else 1)
                if "e" in keys_pressed:
                    loc.z += alt
                if "q" in keys_pressed:
                    loc.z -= alt
                loc.z = max(2.0, loc.z)

                if "arrowleft" in keys_pressed:
                    rot.yaw -= ROTATE_SPEED
                if "arrowright" in keys_pressed:
                    rot.yaw += ROTATE_SPEED
                if "arrowup" in keys_pressed:
                    rot.pitch = max(-89, rot.pitch - ROTATE_SPEED)
                if "arrowdown" in keys_pressed:
                    rot.pitch = min(89, rot.pitch + ROTATE_SPEED)

                spectator.set_transform(carla.Transform(loc, rot))

                world.tick()

                try:
                    image = img_queue.get(timeout=1.0)
                    server.push(carla_image_to_bgr(image))
                except queue.Empty:
                    pass

        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()
            camera.destroy()
            destroy_traffic()
            log.info("drone stopped")


def main():
    setup_logging()
    server = MJPEGServer(title="SkyCop — Experiment 02", html=HTML_PAGE)
    # Add keyboard input route on the same Flask app.
    server.app.add_url_rule("/keys", "update_keys", update_keys, methods=["POST"])
    server.start(port=5000)
    time.sleep(0.3)  # let Flask boot before CARLA loop prints "Drone ready"

    t = threading.Thread(target=carla_loop, args=(server,), daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(timeout=1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
