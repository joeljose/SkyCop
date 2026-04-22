# SkyCop

Simulation-based autonomous drone pursuit assistance. A drone above a
reckless-driving suspect vehicle in CARLA's Town10HD keeps the suspect
visually locked through detector gaps, track-ID switches, and visually
similar NPCs — using YOLOv8s detection, ByteTrack multi-object tracking,
and an HSV-histogram fingerprint for appearance-based re-identification.

The project is a senior-CV-engineer portfolio piece focused on the
perception and decision-intelligence layer. Flight dynamics, rotor
physics, and wind are intentionally abstracted — the aerial camera
represents the drone's sensor output under nominal flight.

**Current status:** Mission v0 — fixed-altitude pursuit with CARLA-GT
seeded fingerprint. Correctness 0.999 (1198 / 1199 IoU-correct frames
over a 60 s mission). See [docs/progress.md](docs/progress.md) for the
running history and [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for
per-requirement status.

## Quick start

Requires:

- NVIDIA GPU with recent drivers (tested on RTX 4050 / 6 GB VRAM)
- Docker with the NVIDIA container toolkit (`NVIDIA_DRIVER_CAPABILITIES=all`
  so Vulkan is available for CARLA's off-screen renderer — see
  [docs/carla_caveats.md §16](docs/carla_caveats.md))

```
make setup     # first time: build client image, create cache dirs
make up        # start CARLA server + client (CARLA takes ~60 s to boot)
make app       # run the mission (60 s by default)
```

Open <http://localhost:5000> while `make app` is running to watch the
overlay stream (GT bbox + tracker boxes + fingerprint scores + locked
track + running correctness). `make help` lists every available target.

## What it does each tick

1. Pin the camera 15 m directly above the suspect, pitch −75°, yaw matching
   the suspect's body heading ([D-07](docs/design.md#d-07) · [D-12](docs/design.md#d-12)).
2. Capture an RGB frame (1280 × 720).
3. Run YOLOv8s, fine-tuned on CARLA-generated data, single-class `vehicle`
   ([D-08](docs/design.md#d-08) · [D-09](docs/design.md#d-09)). Same-map
   mAP@0.5 = 0.962; cross-map 0.581 ([D-11](docs/design.md#d-11)).
4. Feed detections into ByteTrack for persistent track IDs.
5. On the first tick where a tracker bbox overlaps the CARLA-projected
   suspect bbox, extract an HSV histogram as the seed fingerprint.
6. Every subsequent tick, score all tracker bboxes against the seed.
   Sticky-rebind the "suspect track_id" when a candidate beats the
   current lock by more than `score_stickiness`.
7. Project CARLA's world-space suspect bbox into image coords, compute
   IoU with the locked tracker box, accumulate the correctness metric.
8. Overlay everything and push one frame to the MJPEG server + optional
   `mission.mp4`.

The drone's XY position is currently taken from `suspect.get_transform()`
directly — the PID controller that closes the loop on tracker output
is the next unit of work (see [Roadmap](#roadmap)).

## Repository layout

```
skycop/
  main.py              entrypoint — starts MJPEG + runs mission
  mission.py           Mission v0 orchestrator (spawn → pursue → teardown)
  config.py            OmegaConf loader — layered YAML merge
  logs.py              console logger setup
  cv/
    fingerprint.py     HSV histogram extract + intersection score
    gt_projection.py   CARLA world-bbox → image pixels
    tracker_viz.py     overlay rendering helpers
    track.py           ByteTrack adapter over Ultralytics
    inference.py       YOLO wrapper
    capture.py         training-data + tracking-holdout capture (legacy)
    inloop.py          live-pursuit measurement loop (legacy)
    dataset.py         instance-seg bbox extraction
    tracking_eval.py   offline suspect-continuity scoring
    eval.py, training.py, vehicle_classes.py
  sim/
    actors.py          spawn helpers + teardown_pursuit
    aerial_camera.py   RGB sensor spawn + BGRA→BGR
    carla_env.py       connect + synchronous_mode context manager
  dashboard/
    mjpeg.py           Flask MJPEG streamer
configs/               OmegaConf YAML (layered via skycop.config.load)
scripts/               historical experiment record (frozen at 10b)
tests/                 pytest unit tests (~80, all pure — no CARLA)
docs/                  requirements, design log, progress, caveats, survey
```

## Architecture at a glance

```
         CARLA Town10HD_Opt
         (50 NPC vehicles + reckless suspect, sync mode, 20 FPS)
                  │
                  ▼
         RGB camera (SkyCop's eye)
                  │
                  ▼                               ┌────────────────┐
         YOLOv8s detector  ──► ByteTrack MOT  ──► │ HSV fingerprint│
                  │                               │ + sticky rebind│
                  ▼                               └──────┬─────────┘
         CARLA world-bbox projection (GT)                ▼
                  │                               "locked track_id"
                  │                                      │
                  └────── IoU correctness ◄──────────────┘
                                  │
                                  ▼
                          MJPEG overlay stream
                          + summary.json + optional mission.mp4
```

## What's not built yet

Honest list. Each is queued in [docs/progress.md](docs/progress.md) with
the reasoning for deferral.

- **PID drone controller.** Mission v0 "cheats" by positioning the
  drone from CARLA's ground-truth suspect transform. SIM-15..18 calls
  for a tracker-driven PID so the drone can actually lose the suspect
  and fingerprint re-binding is load-bearing.
- **Suspect FSM.** Fleeing → Roaming → Parking → Parked state machine
  from FR-07..11. Mission currently ends on a timer, not on arrest.
- **Dispatch bootstrap (FR-03).** The initial "which tracker box is
  the suspect" resolution currently uses CARLA's actor_id. Production
  would use a dispatch alert (last-known location + vehicle description).
- **Multi-attribute fingerprint.** Only HSV today. Roof shape, apparent
  size, speed/heading (CV-11..16) land when HSV alone hits a failure case.
- **Occlusion recovery + parking-lot re-ID** (CV-17..32).
- **Dashboard.** Streamlit + Leaflet.js dashboard per §7 is planned.
  MJPEG is the live surface today.
- **Full adaptive altitude.** Deliberately dropped ([D-12](docs/design.md#d-12))
  after auditing Town10HD geometry. To be re-added only if we ever
  load a map with actual over-road structures.

## Tech stack

Python 3.10 · CARLA 0.9.16 · YOLOv8s (Ultralytics) · ByteTrack · OpenCV ·
NumPy · PyTorch fp16 · OmegaConf · Flask (MJPEG) · Docker Compose.

Authoritative versions in [`pyproject.toml`](pyproject.toml).

## Hardware budget

Everything fits on an RTX 4050 / 6 GB VRAM laptop:

- CARLA server (Low quality, Town10HD): ~3 GB
- YOLOv8s fp16 inference: ~0.04 GB
- RGB camera + tracker: ~0.2 GB

Total steady state ~4.2 GB. Room for a heavier detector or a second
sensor inside the 5.5 GB NFR-03 ceiling.

## Documentation

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — functional + non-functional
  requirements with per-item status (✅ / 🔨 / ⬜ / ⛔ / ⚠)
- [docs/design.md](docs/design.md) — architecture + design log
  (D-01..D-12) capturing the decisions that shaped the codebase
- [docs/progress.md](docs/progress.md) — live experiment + PR log; the
  fastest way to see what happened and what's next
- [docs/carla_caveats.md](docs/carla_caveats.md) — 18 CARLA 0.9.16 pitfalls
  surfaced during development (sync mode, TM determinism, Docker/GPU,
  destroy order, dual-sensor segfault, …)
- [docs/literature_survey.md](docs/literature_survey.md) — academic
  references grounding the detection, tracking, re-ID, and search
  components

## Common commands

```
make help        # list every target with a description
make status      # container + CARLA health + GPU utilisation
make test        # pytest
make lint        # ruff
make app         # run Mission v0 → http://localhost:5000
make down        # stop all containers
```

## Notes

- Commit style: focused, one logical unit per commit, no AI-authored footers.
- Experiments (`scripts/NN_*.py`) are frozen at `10b`; further development
  happens inside `skycop/` per the pivot recorded in progress.md.
- CARLA server uses ~3 GB VRAM when running. `docker compose stop carla-server`
  between sessions to save thermals.
