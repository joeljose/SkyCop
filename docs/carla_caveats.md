# CARLA 0.9.16 Python API — Development Caveats

A practical reference for SkyCop (synchronous mode, 20 FPS, top-down RGB aerial-drone computer vision). Each caveat is sourced inline so claims can be verified against upstream docs or the issue tracker.

---

## 1. Synchronous mode is a world-level property, not a client-level one

- Setting `settings.synchronous_mode = True` flips the *server's* state. If any client disconnects without reverting it, the server stays in sync mode waiting for a `tick()` that will never come, and every subsequent `get_world()`, `wait_for_tick()`, or `tick()` call blocks indefinitely.
- Recovery: run a tiny "unstick" client that does `settings = world.get_settings(); settings.synchronous_mode = False; settings.fixed_delta_seconds = None; world.apply_settings(settings)`. Also reset the Traffic Manager: `tm.set_synchronous_mode(False)`.
- Always wrap sync-mode scripts in `try/finally` and ideally install a `signal.signal(SIGTERM, handler)` that applies the revert before exit — a plain `atexit` hook won't run on SIGTERM/SIGKILL.
- **Source:** https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/ · issue https://github.com/carla-simulator/carla/issues/3663
- **Observed in SkyCop:** confirmed during script 03 smoke test — a SIGTERM left `Sync: True` in the world, breaking subsequent clients until we manually reset.

## 2. Both the world AND the Traffic Manager must be switched to sync mode

- `tm.set_synchronous_mode(True)` is mandatory if *any* vehicle uses autopilot; otherwise the TM runs in its own async loop and produces non-deterministic, jittery behaviour that drifts relative to world ticks.
- The TM runs on port `8000` by default. If another stale CARLA process holds that port, a silent fallback to a different port can make your `tm.set_random_device_seed(...)` apply to the wrong TM instance.
- **Source:** https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#synchronous-mode · https://github.com/carla-simulator/carla/issues/4479

## 3. `fixed_delta_seconds` must stay ≤ 0.1 s, and sub-stepping matters

- Physics becomes unstable above 0.1 s per step (wheels pop, vehicles teleport through colliders). For 20 FPS you want `fixed_delta_seconds = 0.05`.
- CARLA 0.9.11+ exposes physics sub-stepping: `settings.substepping = True`, `max_substep_delta_time = 0.01`, `max_substeps = 10`. The constraint is `fixed_delta_seconds ≤ max_substep_delta_time * max_substeps`. At 20 FPS the defaults are fine; at 10 FPS you must raise `max_substeps`.
- **Source:** https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#physics-substepping

## 4. Only one client should call `world.tick()`

- If two clients both tick, they each advance the world by one step — causing "double-speed" simulation. Designate a single ticker; other clients call `world.wait_for_tick()` or subscribe via `world.on_tick(callback)`.
- `tick()` returns the `frame` integer of the new snapshot. Use it — not Python wall-clock — to correlate sensor data.
- **Source:** https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/#client-server-synchrony

## 5. Sensor `listen()` callbacks run on a C++ worker thread

- The callback is not invoked on your main Python thread. Do not touch non-thread-safe state (matplotlib, OpenCV windows, unprotected dicts) directly; push data into a `queue.Queue` and consume from the main loop.
- For synchronous capture, a bounded `queue.Queue(maxsize=N)` combined with `queue.get(timeout=2.0)` per tick gives natural back-pressure: if a frame is missed the `get()` raises and you know a drop occurred, rather than silently desyncing.
- Carla sensor images are **BGRA**, not RGB — four channels, alpha present. `np.frombuffer(image.raw_data, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]` gets BGR; append `[:, :, ::-1]` for RGB.
- **Source:** https://carla.readthedocs.io/en/0.9.16/core_sensors/ · https://carla.readthedocs.io/en/0.9.16/ref_sensors/#rgb-camera

## 6. Sensors leak GPU memory if not destroyed in reverse-spawn order

- Underlying issue: sensors hold CUDA-backed render targets; destroying the parent vehicle first orphans them and the server holds the VRAM until process exit.
- Rule: destroy sensors *before* their attached actors. Prefer `client.apply_batch_sync([carla.command.DestroyActor(x) for x in reversed(actor_list)])` over per-actor `.destroy()` — it is atomic and doesn't require a tick between calls in sync mode.
- Orphaned sensors also survive script crashes. On startup, iterate `world.get_actors().filter('sensor.*')` and destroy anything whose parent is gone.
- **Source:** https://github.com/carla-simulator/carla/issues/1221 · https://carla.readthedocs.io/en/0.9.16/python_api/#carla.Client.apply_batch_sync

### 6a. Pursuit-scene teardown order that avoids SIGABRT

Empirically verified in SkyCop: running ad-hoc cleanup (`tm.set_hybrid_physics_mode(False)` → `tm.set_synchronous_mode(False)` → per-actor `.destroy()`) reliably produces `terminate called after throwing an instance of 'std::runtime_error' what(): Responding error from function set_actor_simulate_physics: Actor could not be found in the registry` on process exit (exit code `-6` / SIGABRT). Every CARLA script that spawns NPCs + hero + hybrid physics hit this at some point.

Root cause: CARLA auto-destroys actors asynchronously (collisions, map-boundary, stale from earlier runs). The Traffic Manager keeps its own list of "my vehicles" that can include IDs CARLA has already removed from the registry. When `set_hybrid_physics_mode(False)` (or certain teardown paths) iterate the TM's list and call `set_actor_simulate_physics` on those stale IDs, CARLA throws a C++ `std::runtime_error` that Python can't catch.

Working sequence (from SkyCop `skycop.sim.teardown_pursuit`):

1. **Stop sensor listeners** — `sensor.stop()` before destroy, so in-flight callbacks don't race the teardown.
2. **`tm.set_hybrid_physics_mode(False)` while vehicles are still attached** — the TM iterates *its* registered list at that moment, which is still coherent mid-run.
3. **`apply_batch_sync([SetAutopilot(v.id, False) for v in vehicles], True)`** — detach vehicles from TM oversight in one atomic RPC.
4. **Switch world + TM to async** — batch destroys in sync mode queue on the next tick and can deadlock; async destroys complete immediately.
5. **`apply_batch_sync([DestroyActor(a.id) for a in reversed(actors)], True)`** — one atomic destroy for everything. No per-actor race.

Always use this sequence when you've enabled hybrid physics or when you have > ~5 actors + sensors. For a single-vehicle-single-camera script, per-actor `.destroy()` is usually fine but costs you nothing to use the helper.

- **Source:** [CARLA generate_traffic.py reference](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/generate_traffic.py) · [apply_batch_sync docs](https://carla.readthedocs.io/en/0.9.16/python_api/#carla.Client.apply_batch_sync)

### 6b. Dual-sensor long captures segfault CARLA on destroy

Empirically verified across SkyCop exps 07, 08, 10: when a pursuit scene spawns **both** an RGB camera and an instance-segmentation camera, runs them in sync mode with every-tick capture for **≥ 250 frames**, and then invokes the §6a teardown, the CARLA server SIGSEGVs during the destroy phase:

```
carla-server-1 | Signal 11 caught.
carla-server-1 | CommonUnixCrashHandler: Signal=11
carla-server-1 | CarlaUE4.sh: line 5: 57 Segmentation fault (core dumped) CarlaUE4-Linux-Shipping
```

The server container exits with code 139. The Python client then blocks 60 s on the dead RPC socket and raises a C++ `carla::client::TimeoutException`, which `std::terminate`s the process (SIGABRT, exit 134) — Python's `except Exception` cannot catch it because the exception is thrown from the C++ binding layer.

**Does not help:**

- Moving the sensor destroy from `apply_batch_sync([DestroyActor, ...])` to per-actor `sensor.destroy()` before the batch. The server still segfaults on the first sensor.
- Wrapping destroys in `try/except Exception`. The C++ `terminate` bypasses Python exception handling.

**Root cause (inferred):** UE4 render-target cleanup for the instance-seg camera after a long run trips a use-after-free or similar memory corruption inside `CarlaUE4-Linux-Shipping`. Single-RGB captures with the same frame count and the same teardown sequence do not hit this.

**Scope for SkyCop:** confined to the **offline training/eval pipeline** (`run_capture`, `run_tracking_capture` in `skycop/cv/capture.py`). The production mission uses a single RGB camera per **SIM-07**, so this bug does not affect `skycop.main` or the live-pursuit helper in `skycop/cv/inloop.py`.

**Recommended workaround when a dual-sensor capture is actually needed:** do not rely on teardown at all. Run each capture in a fresh `carla-server` container and `docker compose stop carla-server` between runs — the destroy RPC is skipped entirely, the server exits cleanly when the container stops, and VRAM is reclaimed by process teardown. Costs ~15 s per run for the restart; deterministic and avoids the crash.

- **Source:** empirical, SkyCop issue #25 (closed with this finding). Upstream reports of dual-sensor / instance-seg cleanup crashes: https://github.com/carla-simulator/carla/issues/5790 · https://github.com/carla-simulator/carla/issues/6073

## 7. Blueprint attribute footguns

- `role_name` has no effect on simulation — it is a free-form tag you set to find your own actors later. Hero-specific behaviour only triggers when `role_name == 'hero'` for some TM heuristics.
- `color` is only settable on blueprints where it exists *and* has `recommended_values`. Setting it on a blueprint that lacks the attribute silently no-ops in some versions, raises in others. Always guard with `if bp.has_attribute('color')`.
- Vehicles with `number_of_wheels == 2` (motorcycles, bicycles) **do not support autopilot** and the TM will either ignore them or log a warning. Filter them out: `[bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]`.
- **Source:** https://carla.readthedocs.io/en/0.9.16/core_actors/#blueprints · https://github.com/carla-simulator/carla/issues/3860

## 8. Traffic Manager determinism requires three things simultaneously

- `tm.set_random_device_seed(seed)` — TM's own RNG.
- `world.set_pedestrians_seed(seed)` — walker AI.
- Synchronous mode on both world and TM (see caveat 2).
- Missing any one of these and runs will diverge even with identical vehicle spawns. The seed must be set *before* any vehicle registers with the TM.
- **Source:** https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#deterministic-mode

## 9. Traffic Manager speed convention is inverted

- `tm.vehicle_percentage_speed_difference(vehicle, pct)`: **positive pct = slower than the speed limit, negative = faster**. `-30.0` means "30% above limit." Easily the single most-reported TM bug report that turns out to be a doc-read.
- `tm.global_percentage_speed_difference(pct)` has the same sign convention.
- `tm.distance_to_leading_vehicle(vehicle, m)` is metres, not a percentage. Default ~1.0 m is aggressive; raise to 3–5 m for stable top-down tracking scenes.
- `tm.auto_lane_change(vehicle, False)` makes re-ID evaluation cleaner (vehicles don't weave), at the cost of less-realistic traffic flow.
- **Source:** https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#general-considerations

## 10. Hybrid physics mode — essential on a 6 GB GPU

- `tm.set_hybrid_physics_mode(True)` with `tm.set_hybrid_physics_radius(70.0)` disables wheel/suspension physics for vehicles further than 70 m from any `role_name == 'hero'` actor; they get a simple linear integrator instead. Large FPS win.
- Caveat: for a top-down aerial view with *no* hero vehicle, hybrid mode has nothing to anchor on. Either tag a tracked vehicle `hero`, or set the radius large enough to cover the drone FOV, or leave hybrid off.
- **SkyCop suggestion:** anchor the drone camera to a `role_name='hero'` invisible dummy vehicle (spawned, then `set_simulate_physics(False)` and teleported each tick), so hybrid-physics mode stays useful.
- **Source:** https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/#hybrid-physics-mode

## 11. Camera FOV → focal length conversion

- CARLA cameras take horizontal `fov` in degrees. Focal length in pixels: `fx = W / (2 * tan(fov_rad / 2))`, and `fy = fx` (pixels are square). Principal point is `(W/2, H/2)` exactly — no lens distortion by default on the RGB camera.
- The default `fov` is **90°**. For a drone at 80 m altitude wanting ~80×80 m ground coverage, you want fov ≈ 53° — not the default.
- `sensor_tick` on the blueprint skips frames server-side (e.g. `sensor_tick=0.1` with `fixed_delta_seconds=0.05` yields every other frame). Preferable to dropping frames client-side because the server doesn't render the skipped ones — direct VRAM savings.
- **Source:** https://carla.readthedocs.io/en/0.9.16/ref_sensors/#rgb-camera · https://github.com/carla-simulator/carla/issues/56

## 12. Spectator has no physics, no collision — ideal for a drone

- `world.get_spectator()` returns a pseudo-actor that you can `set_transform()` anywhere each tick. No gravity, no clipping, no collision. This is the cheapest "drone" — just place a camera and parent it to the spectator, or set both transforms in lockstep.
- Attaching a camera *to* the spectator with `attach_to=spectator` is supported but couples the camera to the spectator's frame. A free-floating sensor whose transform you set manually each tick gives finer control (useful for smooth drone paths with spline interpolation).
- The spectator is also the viewpoint in the rendered server window — moving it moves what the server GUI shows, which matters if you run with the display on.
- **Source:** https://carla.readthedocs.io/en/0.9.16/core_actors/#spectator

## 13. `Town10HD_Opt` vs `Town10HD`, and `load_world()` semantics

- The `_Opt` suffix means "layered": you can toggle categories (`world.unload_map_layer(carla.MapLayer.Foliage)`) to save VRAM. The naming looks like "optimised = lower quality", but `Town10HD_Opt` has the same base assets as `Town10HD` — the layers just give you control.
- `client.load_world('Town10HD_Opt')` **destroys all actors**, resets sync mode to default (async), and respawns the spectator. Re-apply your sync settings and re-seed the TM after every `load_world()`.
- `reload_world()` resets actors but keeps the map loaded — faster, and recommended between episodes.
- **Source:** https://carla.readthedocs.io/en/0.9.16/core_map/#layered-maps · https://carla.readthedocs.io/en/0.9.16/python_api/#carla.Client.load_world

## 14. `map.transform_to_geolocation()` uses the map's OpenDRIVE georeference

- Returns a `carla.GeoLocation` (lat/lon/alt). The reference point is stored in the OpenDRIVE header (`<georeference>` tag); maps without it return meaningless values (often lat/lon near 0,0 with CARLA metre-coordinates in degrees).
- Town10HD has a georeference; some community maps do not. Verify with `world.get_map().to_opendrive()` and grep for `<georeference>`.
- CARLA's world frame is **left-handed, Z-up, X-forward, Y-right** (Unreal convention). Most CV/ROS code is right-handed — a sign flip on Y is required when exporting poses.
- **Source:** https://carla.readthedocs.io/en/0.9.16/python_api/#carla.Map.transform_to_geolocation · https://github.com/carla-simulator/carla/issues/3589

## 15. Running headless on a 6 GB VRAM GPU

- `-RenderOffScreen`: no X11 window, uses Vulkan off-screen rendering. Required inside Docker without an X server. Replaces the older `-opengl` + Xvfb hack.
- `-quality-level=Low`: drops shadow resolution, post-processing, reflection captures. Can halve VRAM. Changes the look of the RGB image — if you train a detector, keep this constant between train and eval.
- `-benchmark -fps=20`: `-benchmark` forces a fixed time step regardless of wall-clock ("render as fast as possible, step as if it were 20 FPS"); `-fps=N` sets that step. This is the server-side equivalent of `fixed_delta_seconds = 1/N` and must match it or you get ticks from the wrong clock.
- On 6 GB VRAM, also unload unused map layers (`unload_map_layer(MapLayer.Foliage | MapLayer.Props)`) and cap sensor resolution — a single 1920×1080 RGB camera at 20 FPS uses ~400 MB of render target memory alone.
- **Source:** https://carla.readthedocs.io/en/0.9.16/adv_rendering_options/ · https://carla.readthedocs.io/en/0.9.16/build_docker/

## 16. Docker / headless gotchas

- The official `carlasim/carla:0.9.16` image ships as a non-root user `carla` (UID 1000). Volume-mounting from a host UID other than 1000 produces permission errors on output files. In SkyCop we override with `user: $UID:$GID` in compose.
- GPU access requires `--gpus all` and the NVIDIA container toolkit; Vulkan additionally needs `NVIDIA_DRIVER_CAPABILITIES=all` (not just `compute,utility`). Missing `graphics` and the server silently falls back to CPU rasterisation at ~2 FPS.
- Server needs port `2000/tcp` (and `2001/tcp` used internally); TM uses `8000/tcp` by default. If connecting from another container, expose all three, and pass `--tm-port` on the client to avoid port collisions when multiple simulations share a host.
- The server takes ~10–20 s to accept connections after launch. A client that connects immediately gets `RuntimeError: time-out of 5000ms`. Use `client.set_timeout(20.0)` and a retry loop.
- **Host-side pitfall observed in SkyCop:** the `./.client-cache` bind-mount directory, if created by Docker's mount code instead of `make setup`, is root-owned on the host and the container user (UID 1000) cannot write to it — CARLA then fails at `get_world()` with `boost::filesystem::create_directories: Permission denied`. Always create it via `make setup` or chown after a `make clean`.
- **Source:** https://carla.readthedocs.io/en/0.9.16/build_docker/ · https://github.com/carla-simulator/carla/issues/4676

## 17. Determinism: same seed still isn't bit-identical frames

- Even with world seed + pedestrian seed + TM seed + sync mode, RGB camera output differs across runs because Unreal's renderer uses temporal anti-aliasing and screen-space effects that depend on prior frames' GPU state. Disable TAA via `-quality-level=Low` or a custom rendering config for reproducible pixels.
- Vehicle dynamics *are* deterministic under the triple-seed regime. Sensor extrinsics, tracked trajectories, and bounding-box labels match across runs — only the rendered texture differs subtly.
- **Source:** https://github.com/carla-simulator/carla/issues/4443 · https://carla.readthedocs.io/en/0.9.16/adv_rendering_options/#rendering-quality

## 18. Bounding boxes are in actor-local coordinates

- `vehicle.bounding_box` is in the actor's local frame; for world-space you must apply the actor's transform. `bounding_box.get_world_vertices(actor.get_transform())` does this in one call (0.9.12+). Forgetting the transform gives boxes clustered at the origin — a classic early-project bug for labelling pipelines.
- For static scene boxes (buildings, props), use `world.get_level_bbs(carla.CityObjectLabel.Buildings)` — these are already in world space.
- **Source:** https://carla.readthedocs.io/en/0.9.16/tuto_G_bounding_boxes/

---

*Maintained in `docs/carla_caveats.md`. Add new caveats as SkyCop development uncovers them — prefer linking to upstream sources so claims can be verified.*
