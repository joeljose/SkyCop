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

### D-11 · Cross-map validity probe for detector evaluation

**Status:** Decided · **Date:** 2026-04-21

Every training, val, and test holdout in exps 05–07 sits on Town10HD_Opt. A detector that memorises Town10HD-specific features (road textures, lighting, building-shadow patterns, vehicle asset set) would score high on the same-map test and collapse on any other map — and the same-map number alone can't distinguish memorisation from genuine generalisation.

Exp 08 introduces a small **cross-map validity probe**: 100 frames captured on a different CARLA map (Town01_Opt), with a disjoint seed (900), never touched during training. The fine-tuned detector is evaluated on both the same-map holdout (exp 05, Town10HD_Opt, seed 42) and the cross-map probe. The gap between the two numbers is the memorisation signal.

Interpretation rules (encoded in `scripts/08_finetune_yolo.py`'s `_interpret`):

| Same-map | Cross-map | Verdict |
|---|---|---|
| < 30% | any | **BROKEN** — debug pipeline before reporting |
| ≥ 70% | < 30% | **MEMORISATION** — caveat loudly, plan multi-map training |
| ≥ 70% | ≥ 65% | **GOOD** — real generalisation |
| anything else | — | **PARTIAL** — report both, flag the gap |

**Town choice.** First tried Town03 (the obvious urban counterpart to Town10HD_Opt), but CARLA 0.9.16 segfaults on loading Town03 with a fresh world + 50 NPCs queued — renderer crash inside UE4, not a client-side bug. Town01_Opt loads cleanly and has enough road variety (intersections, curves, different architectural style) to function as a meaningful generalisation probe. The config comment in `configs/training.yaml` records the reason so future authors don't try Town03 and get confused.

**Why not test across all 6+ maps?** Wall-clock cost. Every probe is ~30 s of CARLA runtime. For a portfolio demo we want one representative number, not a matrix. If the one-map probe shows catastrophic overfitting, we'd expand; if the gap is defensible, one probe is enough.

**Follow-ups flagged by the first fine-tune result.** Exp 08 landed at same-map 96.2% / cross-map 58.1% — **38-point gap**. Genuine aerial features were learned (cross-map >> the 5% pretrained baseline), but substantial Town10HD overfitting is present. Future fine-tune rounds (exp 11-v2 if needed) should include training data from at least one additional map; VisDrone warm-start is still a fallback option if multi-map CARLA data proves insufficient.

### D-10 · Literature and industry survey retrofit

**Status:** Decided · **Date:** 2026-04-20

A formal literature + industry survey now lives at `docs/literature_survey.md` and serves as the citation index for REQUIREMENTS.md and subsequent design decisions. Retrofitted — it should have preceded the requirements doc on day one; the project went through several empirical reversals (D-07 pitch, D-08 detector size, D-09 taxonomy) that earlier survey work would have shortcut.

Key conclusions from the survey that affect design:

- **SkyCop's novelty axis is autonomous target-acquisition-from-description.** Skydio × Axon DFR is the closest shipping analogue and deliberately keeps a human-in-the-loop on target selection. Defense autonomy stacks (Anduril/Shield AI) are opaque on whether they do this. Academic CARLA/AirSim work covers components but not end-to-end pursuit. Positioning: research demo exploring a capability commercial vendors avoid on policy grounds, deliberately simpler than any shipping DFR stack on every other axis.
- **The "dispatch description → visual match" flow is supported by the vision-language ReID lineage** (CLIP-ReID, OWL-ViT, GroundingDINO). D-09's decision to move class off the detector and onto the fingerprint module has direct published backing. Future exp 11+ should consider CLIP-ReID embedding similarity as the primary match score, with HSV + geometry as a fallback.
- **CV-06 tracker choice deserves revisiting.** ByteTrack is defensible, but BoT-SORT / StrongSORT are stronger fits for a moving-camera drone scenario (they explicitly add camera-motion compensation). Will revisit after the tracking experiment measures ByteTrack's ID-switch rate against the moving-camera baseline.
- **CV-03's 85% mAP target is aspirational.** VisDrone SOTA sits at 45–55%; 85% assumes in-domain CARLA training exceeds that. Will be revalidated against the exp 08 fine-tune result, not taken as a fixed target.

Going forward, every design log entry that proposes a cross-cutting mechanism should either cite the survey or add a new entry to it. Requirements reviews now include a "cited or flagged?" check per item.

### D-09 · Detector taxonomy: single class "vehicle"

**Status:** Decided · **Date:** 2026-04-20

The detector emits one class (`vehicle`). Fine-grained class (`car / van / truck / bus`) moves to the fingerprint module as an attribute sourced from CARLA blueprint metadata at training-data capture time.

**Rationale.** Three independent arguments all point the same way.

1. **CARLA structurally confounds fine-grained classes.** The Traffic Manager doesn't autopilot 2-wheelers (dropped earlier in D-07's predecessor). Bus class in Town10HD_Opt is backed by a single vehicle model (`vehicle.mitsubishi.fusorosa`) which we also always pick as suspect — so every bus instance in training data is the same make/model at similar framing. Fine-tuning learns "Fusorosa ≈ bus", not a general bus detector. The van class is ~3% of NPC spawns, the truck class ~5%, and cars ~75% — a 25:1 imbalance that biases any unweighted fine-tune.
2. **Dispatch provides the suspect's class before CV runs.** FR-03 specifies the dispatch alert carries a vehicle description including class. The detector does not need to classify to fulfil the mission — it needs to localize, so the fingerprint can score candidates against a known description.
3. **The detector's class output, if used as a hard filter, would remove the suspect.** If the fine-tuned detector's recall on van is (say) 70%, filtering by detector-class when dispatch says "van" drops 30% of real van vehicles — potentially including the suspect. The fingerprint's soft-scoring across colour + geometry + class is robust to this; hard class filtering is not.

**How the fingerprint gets class.** At training-data capture (exp 05 onward), each extracted bbox already knows its CARLA actor id, so `skycop.cv.classify_blueprint()` supplies a `car / van / truck / bus` label into the per-frame manifest (free, from ground truth). At inference time in a CARLA-hosted demo, the same metadata is available via `world.get_actor(id)`. For a real-world deployment (out of scope) the class would come from bbox-geometry heuristics or a small secondary classifier — future work.

**Trade-offs accepted.** (a) We no longer claim "detector knows what class each vehicle is"; (b) CV-03's 85% mAP target now applies to single-class localization, where it's easier; (c) requirements.md CV-01 and CV-12 were updated to reflect the reframe. The class attribute still exists and still feeds into the suspect-match score — just not through the detector.

**Baseline numbers (exp 06 re-run at single-class).** Pretrained YOLOv8s on COCO: mAP@0.5 = 4.95% (up from 1.89% at 4-class, because class-confusion penalty is gone, but recall is still ~4.6% — pretrained genuinely does not recognise aerial vehicles). That's the number exp 07 aims to beat.

### D-08 · Detection milestone: measure pretrained in-loop before any training

**Status:** Decided · **Date:** 2026-04-20

The original plan was "fine-tune on VisDrone first, then test in-loop." Flipped after grilling: we didn't know if speed, accuracy, or both were the constraint, and training a wrong-size model is a 2–3 hour mistake.

New sequence:

1. **Exp 06 — pretrained baseline (no training).** Run COCO YOLOv8s in-loop, measure sustained FPS, VRAM, and mAP on the CARLA holdout. Gives us the reference point every downstream step compares against. Total cost: weights download + ~90 s of runtime.
2. **Exp 07 — fine-tune** — only once we know what's broken. If FPS was the bottleneck, consider YOLOv8n. If recall is the issue, target dataset composition accordingly. If class confusion dominates, weight the loss toward confusable pairs.

Exp 06 resolved: FPS is fine (26), mAP is catastrophic (~2%). Gap is recall (6%) + class confusion (vans → trucks, buses missed). Fine-tuning is the right next step, and the in-app recording hook — rather than VisDrone — is now the favoured data source because it captures the *actual* operational distribution (−75° pitch, 15–40m altitude, CARLA textures).

**Why this matters:** the original "VisDrone warm-start then CARLA" plan assumed we needed a two-stage dataset strategy. Exp 06's numbers suggest the pretrained model sees almost nothing useful in our frames (n_predictions = 44 across 200 frames containing 763 ground-truth boxes) — so the warm-start may offer diminishing returns vs just training from the pretrained initialisation on in-domain CARLA data directly. Exp 07 will test this empirically by starting without VisDrone and only adding it if the CARLA-only fine-tune underperforms.

### D-07 · Aerial camera pitch: −75° operational, −90° reserved for parking re-ID

**Status:** Decided · **Date:** 2026-04-20

Considered: (a) pure nadir (−90°) for the whole mission; (b) chase-cam (−30° to −45°); (c) oblique aerial (−60° to −75°); (d) dual-camera fixed-nadir + oblique.

Chose (c) at −75° as the operational default, with an explicit mode transition to −90° for the parking re-ID phase (implemented with the mission FSM in exp 10, not now).

**Why not pure nadir.** CV-12 wants vehicle class (car / van / truck / bus). From straight overhead, all four look like rectangles of similar aspect ratio; side features that actually distinguish them (cab height, cargo length, window layout) are invisible. We'd be asking the detector to do something the physics of the view rules out.

**Why not chase-cam.** Too much sky and too little per-target pixel density — poor fingerprint signal, useless for the top-down-ish mental model the mission is built around.

**Why not dual-camera.** Doubles render VRAM (~400 MB at 1280×720), doubles CV throughput, and adds a fusion layer for decisions that only weakly couple. Single camera with a mode-aware pitch is simpler and still services both needs because the modes are non-overlapping in time (pursuit vs parking).

**Trade-offs accepted.** Colour-HSV fingerprint (CV-11) takes a ~5% noise increase at −75° vs −90° because side shadows leak into the roof histogram bin. Ground-plane pixel-to-world projection (CM-01..04) gets marginally more complex because the ray hits ground further than directly below. Both are measurable and not deal-breakers.

**How to reinstate −90°.** The value is in `configs/default.yaml` under `camera.pitch`. Parking-phase snap will be a single reassignment when the suspect velocity drops below 2 km/h for 3 s (CV-26), implemented in the mission FSM — altitude also drops to ~10m at that time so the nadir view still gives useful re-ID resolution.

### D-12 · Adaptive altitude dropped — altitude pinned for road-following pursuit

**Status:** Decided · **Date:** 2026-04-22

**Context.** SIM-11/12/14 specified an adaptive altitude scheme: 15 m over open roads, climb to 40 m when 8-ray lateral raycasts detect a building within 20 m, with a 12 m rooftop-clearance overlay. PR #35 added a measurement harness that traced actual altitude behaviour under this controller and found:

- Period-2 oscillation on every run (altitude swinging 15→35→15 m every 100 ms).
- Per-tick altitude jumps up to 19.4 m (p95) — the `0.8 × (40 − 15)` artefact of the formula direction.
- `building_near` fraction 50–65 % almost everywhere (Town10 is dense — lateral rays almost always hit something).
- Correctness dropped from 0.999 (pinned) to 0.64–0.95 as altitude variance broke ByteTrack IoU association.

**Investigation.** Queried Town10HD geometry directly via `world.get_level_bbs(carla.CityObjectLabel.Buildings)` + `.Bridge` + `.Static`:

| Label | Count |
|---|---|
| `Buildings` | 781 |
| `Bridge` | **0** |
| `Static` | 667 |
| `Poles` | 880 |

Only **7 obstacles** cross the drone's 10–20 m flight band AND sit within 8 m of a drivable road — all of them are regular building *walls* adjacent to roads, not decks spanning over them. The skyscrapers (up to 200 m) are set back from the road network.

**Rationale.** Two stacked reasons for dropping the scheme:

1. **No over-road obstacles.** A drone whose XY tracks a road-bound vehicle cannot collide with any Town10HD structure at 15 m altitude, because the only risks are walls *beside* roads and the drone is never over them.
2. **Forward-ray would false-positive at every turn.** Even the narrower "forward raycast along suspect heading" proposal fires spuriously at every T-junction, 4-way intersection, and curve — the road ends at a building across the intersection, the ray hits that building, but the suspect turns before reaching it. The climb would be wasted and the altitude swings would be just as jarring.

**Decision.** Drop adaptive altitude entirely. Pin the drone at `mission.altitude_m` (15 m default). If we ever load a map with actual over-road structures (skyways, elevated highways with low clearance), revisit based on that map's geometry — not on a speculative requirement.

**Deletions.** `skycop/control/`, `skycop/analysis/altitude_trace.py`, `configs/altitude.yaml`, `tests/test_altitude*.py`, `scripts/04_adaptive_altitude.py`, and the `control_mode` config flag. Legacy CV helpers (`skycop/cv/capture.py`, `skycop/cv/inloop.py`) switched to `cfg.camera.altitude` as a fixed pin.

**Requirement fate.** SIM-11 / SIM-12 / SIM-14 marked superseded in REQUIREMENTS.md §5.3. SIM-13 (hard clamp 10–60 m) kept as a documented envelope even though no logic enforces it with altitude pinned.

**Restoration path.** If adaptive altitude becomes necessary (new map, new mission shape), start from data: query the target map's geometry, classify which structures actually sit over drivable roads, and design a trigger against *those* specific classes — not a generic "building within 20 m" proxy.

### D-13 · Flight control: per-axis PID with target-velocity feedforward, no gimbal PID in v1a

**Status:** Decided · **Date:** 2026-04-22

**Context.** Mission v0 drove the camera transform from `suspect.get_transform()` directly — omniscient positioning that sidestepped the entire control-loop question. PR 2a (Mission v1a) replaces that with a tracker-driven closed loop so fingerprint re-identification becomes load-bearing instead of decorative.

**Decisions.**

1. **Two decoupled per-axis PIDs (world X, world Y).** One coupled multi-axis PID would add no expressive power and hides per-axis tuning. Scalar PIDs per axis are simpler, unit-testable in isolation, and match the physical DOFs of a drone moving in a horizontal plane. Altitude is pinned per D-12 — no Z PID.

2. **Velocity feedforward is mandatory, not optional.** Position-only PID has steady-state distance ≈ `target_speed / Kp`. At pursuit speed (22 m/s) and reasonable Kp (~1.0), that's a 22 m lag — unusable. The flight PID adds the target's estimated world-velocity to the PID output: `v_cmd = Kp·error + Kd·d(error)/dt + Ki·∫error + v_target_est`. Target velocity comes from `TargetStateTracker`, a 3-sample rolling finite-difference over target world positions. Configurable multiplier (`control.feedforward.scale`) lets us disable FF for debugging.

3. **Pixel → world via ray/ground-plane intersection.** Each tick, the locked track's bbox centre pixel is projected to a world point via `pixel_to_world_on_ground` (new function in `skycop/cv/gt_projection.py`). Closed-form, handles any camera pitch/yaw, no small-angle approximations. The ground plane is at `control.ground_plane.z = 0.0` for v1a — Town10HD roads are approximately flat; a per-run road-height estimate is future work.

4. **No gimbal PID in v1a (PR 2b deferred).** Camera yaw still follows suspect yaw via GT (the choice from PR #33). A proper gimbal PID would require a second control loop that yaws the camera to centre the bbox horizontally in image, decoupled from drone body motion. That interacts with the flight PID — if both fight over the same image error, classic two-loop instability. The scope-split keeps PR 2a's behaviour observable: if the video shows the body pursues but the suspect weaves side-to-side in frame on aggressive turns, we add the gimbal PID in PR 2b. If the video looks fine, we don't build what we don't need.

5. **Hold-last-known on track loss (SIM-17).** When `locked_track_id` is absent for `control.hold_last_known.trigger_ticks` consecutive ticks, freeze the drone pose and reset the PID integrators and target-state buffer. This prevents integrator windup during blackout and eliminates drift. The drone stays hovering until the tracker re-locks. A search pattern (CV-22..25) is a separate future unit.

6. **Cold-start cheats for v1a.** The drone is spawned at the suspect's initial position, so the first tick has zero pursuit error and the flight PID warms up gracefully. This deliberately skips the dispatch bootstrap problem (FR-03) — turning "last-known location + vehicle description" into an initial drone position is a standalone unit.

**Metric panel (PR 2a).** Three primaries reported in `summary.json`, informed by HOTA's orthogonal-axis philosophy and visual-servoing's dual-metric convention:

- `id_accuracy` — fraction of GT-visible frames where `locked_track_id` matches the GT-suspect-corresponding track (pure CV quality).
- `track_distance_mean_m` / `track_distance_p95_m` — world-space drone→suspect distance, mean + 95th percentile (pure control quality).
- `in_frame_rate` — fraction of ticks where the GT bbox falls fully inside the image (framing consequence of the other two).

Plus a per-tick `trace.jsonl` (`drone_to_suspect_m`, `center_offset_px`, `velocity_cmd_m_s`, `locked_track_id`, `gt_matched_track_id`) for post-hoc diagnosis.

**Legacy metric** (`legacy_iou_correctness` — Mission v0's IoU-vs-GT) is kept in the summary for continuity but is **not** the pass bar; it conflated CV and control quality into one number.

**Gains starting point.** Initial guess: `Kp = 0.8`, `Ki = 0.0`, `Kd = 0.15`, output clamp 20 m/s per axis. Tuned by watching the live mission; any iteration lands as a config-only follow-up, not a code change.

**Known cheats documented here:**

- Drone cold-starts at suspect's initial XY (FR-03 dispatch bootstrap separate).
- Ground plane assumed flat at `z = 0.0` (adequate for Town10HD roads; per-run estimation is future work).
- Camera yaw still follows suspect yaw via GT (gimbal PID is PR 2b).

---

### D-14 · Suspect FSM v0a — mixed transitions, custom PARKING controller, pass-fail confirmation

**Status:** Decided · **Date:** 2026-04-22 · **Issue:** #44

**Context.** Mission v1a (D-13) proved pursuit survives normal-autopilot driving (id_accuracy 0.975, track_distance p95 0.98 m). Before bootstrapping the dispatch flow we needed a gate: *if the drone can keep up through realistic suspect behaviour — aggressive, normal, parking — only then proceed*. That gate is the Suspect FSM (FR-07..11). v0a is the first slice; v0b/v0c add the menu and user-drone mode.

**Decisions.**

1. **Four states with mixed transition semantics.** FLEEING (time ~20 s) → ROAMING (time ~15 s) → PARKING (destination-bound) → PARKED (state-bound). Time-based FLEEING/ROAMING are sufficient for v0a metrics; PARKING is destination-bound because v0c will need a plausible parked pose for the user to click "Parked" on. Pure-time PARKING would park the suspect in the middle of a traffic lane.

2. **PARKING escalation chain.** Primary target = nearest hand-picked parking-lot waypoint; on timeout, fallback to nearest roadside spot; on *that* timeout, full-brake "park in place." Either of the first two reaching destination (`dist ≤ reach_distance_m ∧ speed < reach_speed_mps`) enters PARKED. Park-in-place likewise enters PARKED once the vehicle has actually stopped. This keeps the "parks at a plausible spot" invariant without infinite mission loops when TM routing deviates.

3. **Confirmation is pass-fail via bbox-centre-in-GT, not IoU.** On PARKED entry a 60 s countdown starts. AI mode submits automatically on the **first K=3 consecutive ticks of a valid lock** on the suspect track (K-voting ignores single-tick detector blips); v0c user mode will submit the drawn bbox on button click. Grading is the *same* rule in both modes: submission bbox centre must fall inside the suspect's GT-projected bbox — pass or fail, no threshold to invent. Failing to reach a parking lot is *not* a lose; only failing to confirm within 60 s post-PARKED is.

4. **Custom PARKING controller via vendored CARLA LocalPlanner.** `TrafficManager.set_path()` is empirically broken in 0.9.13–0.9.16 (verified in `scripts/11_parking_scout.py` — sparse and dense waypoint lists both ignored by the TM). Route-to-destination requires bypassing the TM: at PARKING entry we `set_autopilot(False)` and drive the suspect with CARLA's own `LocalPlanner + VehiclePIDController`, fed by a `GlobalRoutePlanner` route. These live in CARLA's source tree but *not* in the pip-installed wheel, so we vendored them under `skycop/vendor/carla_agents/` (+ a three-line `_misc.py` subset + `road_option.py` enum extraction). License is MIT; `LICENSE_CARLA` records provenance. A hand-rolled pure-pursuit controller (`skycop.control.WaypointFollower`) was tried first and oscillated at tight corners — CARLA's PID is tuned for CARLA vehicle dynamics, ours was not.

5. **Pure-logic FSM module, CARLA-aware mission loop.** `skycop.sim.suspect_fsm.SuspectFSM` is a pure state machine: observations in (`t`, `suspect_xy`, `suspect_speed`, `ai_lock_on_suspect`), side-effect *requests* out (`apply_tm_knobs`, `set_path_to`, `freeze_physics`). The mission loop translates those requests to CARLA calls (TM setters, LocalPlanner build, `set_simulate_physics(False)`). This keeps the FSM unit-testable with a fake clock (18 tests, no CARLA) while the mission side-effects stay at the boundary.

6. **Start trigger at `POST /start`.** Mission waits on a `threading.Event` before spawning NPCs. The MJPEG server serves a splash page at `/` until the event fires, then the live feed. This is required so the operator can open the page *before* any ticks happen — useful for observation and screen capture, not for control.

7. **Per-state metric breakdown in `summary.json`.** Each FSM state collects its own `{frames_total, frames_visible, id_accuracy, in_frame_rate, track_distance_mean/p95}`. Aggregate metrics still reported for continuity with v1a. Per-state numbers tell us *where* the pipeline struggles (e.g., FLEEING might drop in_frame_rate on sharp turns); aggregate numbers alone can't.

8. **Exploratory metrics — no hard numeric gate.** v0a doesn't auto-pass/fail on per-state thresholds. User is final judge watching the MJPEG feed and the numbers side-by-side. Rationale: thresholds picked before data are guesses; thresholds picked from the first run are post-hoc. For v0a we just collect and look.

**Deferred from v0a (explicit non-goals).**

- **v0b** — menu page with mode selector, "Start" as the only control; currently the start trigger is the only UI.
- **v0c** — user-controlled drone mode + draw-bbox submission UI for the pursuit game.
- **FR-11 parking-lot geometry** — v0a uses hand-picked Town10HD spawn points rather than inferred lot polygons.
- **CV-26 nadir pitch snap** — PARKED continues at pitch −75°.
- **CV-27..32 parked-suspect re-ID** — once PARKED, tracker is still regular ByteTrack + HSV; no re-ID pipeline.
- **Dispatch bootstrap (FR-03)** — drone still cold-starts at suspect's XY in v0a.

---

## 13. Open Questions

- **Suspect FSM owner.** ~~Scripts 03/04 use TM autopilot with reckless knobs. The proper Fleeing→Roaming→Parking→Parked FSM (FR-07..11) is deferred.~~ Resolved by D-14 — FSM lives in `skycop.sim.suspect_fsm`, CARLA side-effects in the mission loop.
- **Frame-level ground truth export.** For CV evaluation (CM-07..09), we'll need per-frame CARLA ground-truth dumps. Format — JSONL alongside frames, or a single run-level file? Format affects downstream tooling.
- **Human-vs-autonomous replay.** §6.4 demands both modes play the same scenario for comparison. Do we record inputs and replay, or re-seed and re-run? Replay is robust to non-determinism in renderer (caveat §17) but requires an input-logging layer.
- **Evaluation artifact shape.** §12 lists metrics to log; shape is undecided. A single `eval.json` per run, or time-series parquet? Small to start; parquet only if dashboards demand it.

---

*This document is the authoritative design record. Before introducing a new cross-cutting mechanism (new config layer, new process boundary, new coordinate frame), add an entry to the Design Log with the rationale. Revisit open questions whenever a milestone forces them.*
