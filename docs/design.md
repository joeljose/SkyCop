# SkyCop — Design Document

Companion to `docs/REQUIREMENTS.md`. Requirements answer *what* the system does; this document captures *how* it is built and *why* each non-obvious choice was made. It is intended to evolve — every design decision with lasting consequences lands here, with a short rationale so a future reviewer (or future me) can tell load-bearing choices from incidental ones.

**Status:** Draft · active · **Last updated:** April 2026

---

## Table of Contents

1. [Goals and Non-Goals of the Design](#1-goals-and-non-goals-of-the-design)
2. [Repository Layout](#2-repository-layout)
3. [Container and Dependency Model](#3-container-and-dependency-model)
4. [Configuration Strategy](#4-configuration-strategy)
5. [Package Module Responsibilities](#5-package-module-responsibilities)
6. [Tick Model and Runtime Loop](#6-tick-model-and-runtime-loop)
7. [Adaptive Altitude Controller](#7-adaptive-altitude-controller)
8. [Scene Composition](#8-scene-composition)
9. [Dashboard Surface](#9-dashboard-surface)
10. [Testing Strategy](#10-testing-strategy)
11. [Build, Run, Release](#11-build-run-release)
12. [Design Log — Decisions and Revisions](#12-design-log--decisions-and-revisions)
13. [Open Questions](#13-open-questions)

---

## 1. Goals and Non-Goals of the Design

**Design goals**
- **Reproducibility.** Anyone with Docker + an NVIDIA GPU should run `make setup && make up && make exp N=04` and get the same behaviour. Seeded RNGs, pinned deps, bind-mounted source, no host installs.
- **Teachability.** The repo reads top-to-bottom: `docs/REQUIREMENTS.md` explains the mission, `docs/design.md` (this file) explains the architecture, `scripts/NN_*.py` are progressively richer experiments, `skycop/` is the library those experiments compose. A senior reviewer should be able to skim and understand the whole in ~15 minutes.
- **Hot iteration.** Edits on the host must be live inside the container without a rebuild. CARLA loop times are 30–60 s per cold start; adding a rebuild step on top is unacceptable for research-y work.
- **Scale-ready, not scaled.** The architecture should accommodate the full requirement set (CV pipeline, dashboard, scoring) without demanding it up front. No YAGNI abstractions.

**Explicit non-goals**
- Not a distributable Python library. No wheel, no PyPI, no semver on the package. `pyproject.toml` exists for dep/tool config only.
- Not a general-purpose CARLA framework. The helpers in `skycop.sim` are sized for this mission, not for arbitrary CARLA projects.
- Not production-grade. No auth, no observability beyond print statements, no failure-mode design. It is a simulation portfolio demo.

---

## 2. Repository Layout

```
SkyCop/
├── pyproject.toml          # deps + ruff + pytest config; NOT a distributable
├── Makefile                # sole operator entry point (make setup/up/exp/test/lint/app)
├── docker-compose.yml      # carla-server + client services
├── client/
│   └── Dockerfile          # system libs + pip install of pyproject deps
├── skycop/                 # THE package, bind-mounted at /app; PYTHONPATH=/app
│   ├── config.py           # OmegaConf loader
│   ├── main.py             # `python -m skycop.main` — mission entrypoint (stub)
│   ├── sim/                # CARLA wrappers (connect, sync, cameras, actors)
│   ├── cv/                 # detection, tracking, fingerprint, re-ID (not yet wired)
│   ├── control/            # PID, adaptive altitude, collision safety
│   └── dashboard/          # MJPEG streamer, Streamlit (later), WebSocket (later)
├── configs/                # yaml config files loaded by skycop.config.load(...)
│   ├── default.yaml        # scene + carla + camera defaults
│   └── altitude.yaml       # altitude controller thresholds
├── scripts/                # experiments (01_hello_world, 02_drone_view, ...)
├── tests/                  # pytest — pure logic only; no CARLA in CI-style paths
├── docs/
│   ├── REQUIREMENTS.md     # what the system does
│   ├── design.md           # how it's built (this file)
│   └── carla_caveats.md    # 18 sourced CARLA footguns
└── output/                 # run artifacts (images, recordings) — gitignored
```

**Why flat `skycop/` at repo root and not `src/skycop/`.** The `src/` layout is PyPA's recommendation for distributable libraries because it prevents accidental cwd imports. In this project cwd *is* the source — nothing is `pip install`-ed. `src/` would just add a directory to navigate without buying any safety.

---

## 3. Container and Dependency Model

**One build context, two services.**

- `carla-server` runs the stock `carlasim/carla:0.9.16` image with `-RenderOffScreen -quality-level=Low -benchmark -fps=20`. Headless Vulkan rendering; no X server required.
- `client` is built from `client/Dockerfile`. Its only build-time job is installing Python deps; it carries no project source. At runtime the repo is bind-mounted to `/app` and `PYTHONPATH=/app` makes `import skycop` resolve to `/app/skycop`.

**Why no `pip install -e .`.** Editable installs work, but they need the source present at build time, which contradicts bind-mount-only. PYTHONPATH is the cheapest way to make `import skycop` hot-reload across host edits.

**User model.** The Makefile exports `CARLA_UID=$(id -u)` and `CARLA_GID=$(id -g)`. The client service runs as that UID, so bind-mounted files stay owned by the host user and there's no root in the container runtime. No `USER` directive in the Dockerfile — it's unnecessary in this setup and would add a UID mismatch.

**Dep authority.** `pyproject.toml` is the single source of truth for Python deps (runtime + dev). The Dockerfile parses it at build time with `tomli` and installs. `client/requirements.txt` was retired; keeping two sources of truth is an anti-pattern.

---

## 4. Configuration Strategy

**OmegaConf, not full Hydra.** The choice was made after weighing both (details in Design Log D-02). OmegaConf gives yaml + dot-list overrides + type coercion with no cwd-change magic and no config-group DSL. The project's "composition" needs are small (~6 canonical runs: 3 difficulties × human/autonomous).

**Compose pattern.**

```python
from skycop.config import load
cfg = load("default", "altitude", overrides=["seed=7", "camera.fov=53"])
```

Later files override earlier ones; `overrides=[...]` wins over everything. `CONFIGS_ROOT = /app/configs` is module-level but monkey-patchable in tests.

**Files.**
- `configs/default.yaml` — scene, CARLA connection, camera defaults.
- `configs/altitude.yaml` — altitude controller thresholds (SIM-11..14).
- Future: `configs/difficulty/{rookie,officer,detective}.yaml`, `configs/weather/{clear,night,wet}.yaml`, `configs/model/yolo.yaml`.

**Values get promoted to configs when they cross the "something I'll want to tweak per-run" line.** Until then they stay as constants inside their module. We resist the urge to put *every* number in yaml.

---

## 5. Package Module Responsibilities

### `skycop.sim`

CARLA integration layer. Public surface:
- `connect(host, port)` — reads env vars by default, returns a `carla.Client`.
- `synchronous_mode(world, dt)` — context manager. **Restores original settings on exit.** This is the single place that handles caveat §1.
- `spawn_aerial_camera(world, ...)` — spawns an RGB sensor + returns a `(sensor, queue.Queue)` pair.
- `carla_image_to_bgr(image)` — BGRA → BGR conversion, contiguous output.
- `spawn_npcs`, `spawn_reckless_suspect`, `SuspectParams`, `destroy_all`, `four_wheel_blueprints` — scene assembly.

### `skycop.control`

- `AltitudeConfig` — all thresholds as a dataclass.
- `compute_target(cfg, observation, previous_z)` — pure function; the testable core of SIM-11..14.
- `AdaptiveAltitudeController(world, cfg)` — CARLA-integrated wrapper. `.step(x, y)` returns `(target_z, observation)`.

Future additions: `control.pid` (SIM-15..18), `control.safety` (depth raycast + safe-altitude override, phase 1 of the tick model).

### `skycop.dashboard`

- `MJPEGServer(title, hud, html=None)` — Flask streamer, daemon-threaded. `.push(frame)` from the CARLA loop, `.app` exposed for scripts that want to attach extra routes (e.g. experiment 02's keyboard capture).

Future: `dashboard.streamlit`, `dashboard.events` (WebSocket event bus per §7 of requirements).

### `skycop.cv`

Empty. Populated in the Detection milestone with `cv.detect` (YOLOv8 wrapper), `cv.track` (ByteTrack), `cv.fingerprint` (HSV + geometry), `cv.reid`.

### `skycop.config`

- `load(*names, overrides=None)` — merges `configs/<name>.yaml` files and dot-list overrides. Raises `FileNotFoundError` on missing configs rather than silently returning defaults.

### `skycop.main`

`python -m skycop.main`. Currently a stub that prints package version and usage. Will grow into the mission orchestrator once all milestones land.

---

## 6. Tick Model and Runtime Loop

Per requirements §2.2, every tick runs three phases in order:

1. **Collision safety** (depth raycast; overrides tracking output).
2. **CV tracking** (YOLOv8 + ByteTrack + fingerprint).
3. **Control** (PID + camera transform apply).

Currently experiments 03/04 run only phases 1-like (raycast) and 3 (camera placement). CV lands with the Detection milestone.

**Key invariants already enforced by the package:**

- **One client ticks.** All experiments connect as the sole ticker. Multi-client scenarios are out of scope.
- **Synchronous mode is scoped.** The `synchronous_mode(world, dt)` context manager is the only place sync mode is toggled. SIGTERM bypasses it (a known caveat) so we additionally provide documentation and a manual reset idiom for ops.
- **Sensors destroyed before vehicles.** `destroy_all(actors)` iterates in reverse-spawn order; sensors with `.is_listening` are `.stop()`-ed first. This closes the leak documented in caveat §6.
- **Fixed 20 FPS.** `fixed_delta_seconds = 0.05`, CARLA server launched with `-benchmark -fps=20` to match. Both settings must agree (caveat §15).

---

## 7. Adaptive Altitude Controller

Implements SIM-11..14. Two-layer design:

**Pure layer (`compute_target`):**

```
target = urban_target if building_near else open_target
if rooftop_z is not None:
    target = max(target, rooftop_z + rooftop_clearance)
target = clamp(target, min, max)
if previous_z is None: return target
return smoothing * previous_z + (1 - smoothing) * target
```

All parameters come from `AltitudeConfig`. No CARLA, no globals — directly unit-tested.

**CARLA layer (`AdaptiveAltitudeController`):**

Each tick, casts `N=8` horizontal rays from the drone's previous-Z position outward at `lateral_scan_radius_m` (default 20m). If any hit has label `CityObjectLabel.Buildings`, `building_near=True`. Casts one downward ray to find the tallest `Buildings` hit directly below — that's `rooftop_z`. Feeds both into `compute_target`.

**Trade-offs worth knowing:**

- **8 rays per tick is arbitrary.** More is more precise, slower. At 20 FPS with the current render budget it's negligible; revisit if the detection pipeline gets heavy.
- **Raycast origin is the previous tick's Z, not the target Z.** Means the controller can lag by one frame on sharp altitude changes — acceptable given the IIR smoother already damps fast transitions.
- **`CityObjectLabel.Buildings` only.** Trees, poles, power lines are not matched. Deliberate for 15–40m operational altitude; adjust if the ceiling drops below 15m (future work, out of scope per §10 of requirements).

---

## 8. Scene Composition

**Suspect spawns with `role_name='hero'`** so the Traffic Manager's hybrid-physics mode (`tm.set_hybrid_physics_mode(True)`, radius 100m) anchors on it. NPCs outside 100m of the suspect get simplified physics — FPS win that matters on 6GB VRAM (caveat §10).

**Suspect reckless parameters** live in `SuspectParams` dataclass with CARLA's inverted speed convention (negative = faster) called out in docstring. `-80.0` = 80% over speed limit, matching FR-08's 60–100% range.

**Seeded determinism.** `cfg.seed` seeds both the Python `random.Random` used for NPC/suspect blueprint selection and `tm.set_random_device_seed()`. Missing the TM seed is a common determinism bug (caveat §8). Walker/pedestrian seeding will be added with the Game milestone.

---

## 9. Dashboard Surface

Experiments run their own lightweight surface: `MJPEGServer` serves `/` (HTML with embedded stream) and `/stream` (multipart JPEG). Experiment 02 adds a `/keys` POST route on the same Flask app for keyboard capture — demonstrates the "attach extra routes" extension pattern.

The real §7 dashboard (Streamlit + Leaflet + WebSocket event bus) will sit alongside, reusing the MJPEG feed. Design for that lands with the Dashboard milestone.

---

## 10. Testing Strategy

**Unit tests only** — no CARLA server required. `make test` runs in <1 second and is part of every PR gate.

Current coverage:
- `test_imports.py` — package + submodule import smoke.
- `test_altitude.py` — `compute_target` maths: clamping, smoothing convergence, rooftop clearance overriding urban target.
- `test_config.py` — OmegaConf merger: file overlays, dot-list overrides, missing-config errors.

**What we explicitly do not test:**
- CARLA integration behaviours (spawns, raycast hits). Mocking CARLA deeply is expensive and brittle; we rely on end-to-end runs of the experiment scripts instead.
- Flask routes. MJPEG is exercised whenever an experiment runs; separate route tests would be over-engineering for a demo.

**E2E verification** is manual: run each experiment against a live CARLA server, check Flask serves HTTP 200, verify the frame stream is live. Documented in the PR descriptions, not automated. Added to the test strategy if a CV regression workflow demands it.

---

## 11. Build, Run, Release

**Entrypoint contract: everything through `make`.** Operators never invoke `docker compose`, `pip`, or `python3` directly. The Makefile is organised into six sections — Lifecycle, Observability, Experiments, Application, Dev, World Control — and `make help` prints them.

**Experiments are discovery-based.** `make exp N=NN` fuzzy-matches `scripts/NN_*.py`, so adding a new experiment needs no Makefile edit. `make exp-list` shows what's available.

**Releases are commit-grained, not tagged.** For a demo we don't need semver. Each meaningful feature lands as a focused commit; the git history is the changelog. (This document's Design Log complements it with *why* notes.)

---

## 12. Design Log — Decisions and Revisions

### D-01 · Package layout: flat `skycop/`, not `src/skycop/`

**Status:** Decided · **Date:** 2026-04-20

Considered: (a) `src/skycop/` per PyPA recommendation; (b) flat `skycop/`; (c) everything under `client/`. Chose (b). The `src/` layout's purpose is protecting against cwd imports during library distribution — not relevant here since we never install the package. Flat is one fewer directory to navigate.

### D-02 · Configs: OmegaConf, not Hydra; not pydantic-settings

**Status:** Decided · **Date:** 2026-04-20

Considered: (a) Hydra — industry-standard for ML configs; (b) OmegaConf alone — subset of (a); (c) pydantic-settings + PyYAML; (d) stay on module-level constants. Chose (b). The demo's end-state is ~6 canonical runs, not an experiment sweep factory. Hydra's auto-output-dir and cwd-change would fight bind-mounts. Pydantic gives validation but no CLI composition. OmegaConf hits the sweet spot: yaml + overlays + dot-list overrides, no magic. Upgrade to Hydra is easy if ablation sweeps later justify it.

### D-03 · Suspect `role_name='hero'` for hybrid physics

**Status:** Decided · **Date:** 2026-04-20

The Traffic Manager's hybrid-physics mode anchors its radius on `role_name='hero'` actors. Our drone is visually the hero but has no physics, so can't be tagged. Tagging the suspect keeps hybrid physics useful at the cost of a mild semantic mismatch. Alternative — spawn a separate invisible hero dummy — adds a second actor to manage. Revisit if we ever need the suspect tagged differently for scoring.

### D-04 · Tick model enforcement spread across the package

**Status:** Decided · **Date:** 2026-04-20

Sync mode is owned by `skycop.sim.synchronous_mode()` (context manager). Destroy order is owned by `skycop.sim.destroy_all()`. Seed setting is per-script since the RNGs are created there. No single "engine" class wraps the loop — each experiment owns its tick loop explicitly so the control flow is readable top-to-bottom. When the application's `main.py` lands, it will introduce a thin `Mission` orchestrator, but not an engine abstraction.

### D-05 · Dashboard: MJPEG now, Streamlit later

**Status:** Decided · **Date:** 2026-04-20

Streamlit is the target per DB-10. For experiments 01–04, just a live camera feed is enough; MJPEG keeps the first few experiments dependency-light and avoids Streamlit boilerplate. Streamlit lands with the Dashboard milestone and will consume the same MJPEG feed plus the WebSocket event bus.

### D-06 · Commit grain: focused commits, no tags

**Status:** Decided · **Date:** 2026-04-20

Each logical unit (scaffold, feature slice, docs update) is its own commit. No release tags until we either cut the demo or hit a reviewer-facing milestone worth naming.

### D-07 · Aerial camera pitch: −75° operational, −90° reserved for parking re-ID

**Status:** Decided · **Date:** 2026-04-20

Considered: (a) pure nadir (−90°) for the whole mission; (b) chase-cam (−30° to −45°); (c) oblique aerial (−60° to −75°); (d) dual-camera fixed-nadir + oblique.

Chose (c) at −75° as the operational default, with an explicit mode transition to −90° for the parking re-ID phase (implemented with the mission FSM in exp 10, not now).

**Why not pure nadir.** CV-12 wants vehicle class (car / van / truck / bus). From straight overhead, all four look like rectangles of similar aspect ratio; side features that actually distinguish them (cab height, cargo length, window layout) are invisible. We'd be asking the detector to do something the physics of the view rules out.

**Why not chase-cam.** Too much sky and too little per-target pixel density — poor fingerprint signal, useless for the top-down-ish mental model the mission is built around.

**Why not dual-camera.** Doubles render VRAM (~400 MB at 1280×720), doubles CV throughput, and adds a fusion layer for decisions that only weakly couple. Single camera with a mode-aware pitch is simpler and still services both needs because the modes are non-overlapping in time (pursuit vs parking).

**Trade-offs accepted.** Colour-HSV fingerprint (CV-11) takes a ~5% noise increase at −75° vs −90° because side shadows leak into the roof histogram bin. Ground-plane pixel-to-world projection (CM-01..04) gets marginally more complex because the ray hits ground further than directly below. Both are measurable and not deal-breakers.

**How to reinstate −90°.** The value is in `configs/default.yaml` under `camera.pitch`. Parking-phase snap will be a single reassignment when the suspect velocity drops below 2 km/h for 3 s (CV-26), implemented in the mission FSM — altitude also drops to ~10m at that time so the nadir view still gives useful re-ID resolution.

---

## 13. Open Questions

- **Suspect FSM owner.** Scripts 03/04 use TM autopilot with reckless knobs. The proper Fleeing→Roaming→Parking→Parked FSM (FR-07..11) is deferred. Lives in `skycop.sim.suspect_fsm` (module to be created) or bespoke per-script? Leaning on the module.
- **Frame-level ground truth export.** For CV evaluation (CM-07..09), we'll need per-frame CARLA ground-truth dumps. Format — JSONL alongside frames, or a single run-level file? Format affects downstream tooling.
- **Human-vs-autonomous replay.** §6.4 demands both modes play the same scenario for comparison. Do we record inputs and replay, or re-seed and re-run? Replay is robust to non-determinism in renderer (caveat §17) but requires an input-logging layer.
- **Evaluation artifact shape.** §12 lists metrics to log; shape is undecided. A single `eval.json` per run, or time-series parquet? Small to start; parquet only if dashboards demand it.

---

*This document is the authoritative design record. Before introducing a new cross-cutting mechanism (new config layer, new process boundary, new coordinate frame), add an entry to the Design Log with the rationale. Revisit open questions whenever a milestone forces them.*
