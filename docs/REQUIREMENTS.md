# SkyCop — Autonomous Drone Pursuit Assistance System
## Requirements Document

**Version:** 1.1  
**Status:** Draft  
**Project type:** Senior Computer Vision Engineer Portfolio Project  
**Simulation platform:** CARLA 0.9.16 (Town10HD)  
**Target hardware:** NVIDIA RTX 4050 6GB VRAM, 16GB RAM  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Functional Requirements](#3-functional-requirements)
4. [Computer Vision Requirements](#4-computer-vision-requirements)
5. [Simulation Requirements](#5-simulation-requirements)
6. [Game / Interactive Layer Requirements](#6-game--interactive-layer-requirements)
7. [Dashboard Requirements](#7-dashboard-requirements)
8. [Coordinate Mapping Requirements](#8-coordinate-mapping-requirements)
9. [Non-Functional Requirements](#9-non-functional-requirements)
10. [Out of Scope](#10-out-of-scope)
11. [Known Limitations](#11-known-limitations)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Tech Stack Summary](#13-tech-stack-summary)
14. [Milestones](#14-milestones)

---

## 1. Project Overview

### 1.1 Purpose

SkyCop is a simulation-based autonomous drone pursuit assistance system. It demonstrates that a CV-powered aerial drone can outperform a human operator in tracking a fleeing suspect vehicle through a photorealistic urban environment.

The system runs entirely in the CARLA autonomous driving simulator and is designed as an interactive game — a human player competes against the autonomous CV pipeline on the same scenario, with a scored debrief comparing both performances.

### 1.2 Core Claim

An autonomous drone equipped with computer vision — vehicle detection, multi-object tracking, appearance fingerprinting, and route prediction — achieves higher tracking continuity, faster target acquisition, and more reliable re-identification after occlusion than a human operator flying the same drone manually.

### 1.3 Simulation Scope

This project focuses on the **computer vision and decision intelligence layer** of an aerial pursuit system. Flight dynamics, rotor physics, and wind modelling are intentionally abstracted — the aerial camera represents the drone's sensor output under nominal flight conditions. This is a deliberate architectural decision: the CV contribution is independent of flight controller implementation, and the same pipeline could be deployed on a real drone by replacing `spectator.set_transform()` with MAVLink velocity commands.

---

## 2. System Architecture

### 2.1 High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                    CARLA Simulator                       │
│  City (Town10) · Traffic · Suspect FSM · Physics        │
└──────────────────────────┬──────────────────────────────┘
                           │ camera frames + world state
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  CV Pipeline                             │
│  YOLOv8 Detection · ByteTrack · Fingerprinting          │
│  Occlusion Recovery · Route Prediction                  │
└──────────┬──────────────────────────────┬───────────────┘
           │ PID commands                 │ world coords + events
           ▼                             ▼
┌──────────────────┐         ┌───────────────────────────┐
│  Drone Controller│         │  WebSocket Event Bus       │
│  Adaptive altitude│        │  Alerts · Positions · Score│
│  Collision safety │        └────────────┬──────────────┘
└──────────────────┘                      │
                                          ▼
                           ┌─────────────────────────────┐
                           │  Streamlit Dashboard         │
                           │  Map · Camera feed · Alerts  │
                           │  Scoring · Debrief           │
                           └─────────────────────────────┘
```

### 2.2 Tick Model

All subsystems run within a single synchronous CARLA tick at **20 FPS** (50ms fixed timestep). Each tick executes the following phases in order:

| Phase | Responsibility |
|-------|----------------|
| 1. Collision safety | Depth raycasting, safe altitude enforcement |
| 2. CV tracking | YOLO detection, ByteTrack, fingerprint update |
| 3. Control | PID controller, camera position update |

Collision safety output always takes precedence over tracking commands — if raycasting detects an obstacle, the drone climbs regardless of the tracking target.

---

## 3. Functional Requirements

### 3.1 Mission Story

A suspect vehicle begins driving recklessly through the city — speeding, running red lights, and making illegal lane changes. The suspect **has no knowledge of the drone** and does not attempt to evade it. On the **first traffic violation**, a dispatch alert is sent to the drone operator (human player or autonomous agent) with the last known location and a vehicle description. The drone receives a **configurable number of violation reports** as initial intel (location + description each time), then receives no further violation alerts. The number of reports varies by difficulty level — more reports at easier levels, fewer at harder levels.

The drone's mission: locate the suspect using the initial violation reports, follow it through the city, and ultimately pinpoint its parked location when it stops.

The suspect roams for a configurable duration, then drives to a surface parking lot and parks among other stationary vehicles. The mission ends when the drone either confirms or fails to identify the suspect's parking spot.

### 3.2 Mission Flow

| ID | Requirement |
|----|-------------|
| FR-01 | System shall generate a suspect vehicle scenario on mission start with randomised vehicle model, colour, and route |
| FR-02 | Suspect shall begin driving and committing traffic violations immediately on mission start |
| FR-03 | On the first traffic violation, a dispatch alert shall be sent to the player within 1 second, including last known location and vehicle description. The drone shall receive a number of violation reports determined by difficulty level (with updated location each time), then no further violation alerts. **Acquisition algorithm** — turning dispatch text + location into a visual lock — draws from road-constrained target tracking (RBPF on road graphs, HMM map matching) and description-conditioned detection (OWL-ViT, GroundingDINO, CLIP-ReID). Full algorithm to be specified in §4.8 "Target Acquisition" (follow-up PR). See `docs/literature_survey.md` §4. |
| FR-04 | Drone shall be deployable by the player within 10 seconds of receiving a dispatch alert |
| FR-05 | The mission shall end when the suspect parks and the system either confirms or fails to identify the suspect vehicle |
| FR-06 | The system shall produce a scored debrief at mission end comparing human player vs autonomous CV performance |

### 3.3 Suspect Vehicle Behaviour

The suspect is unaware of the drone. Its behaviour is driven by a scripted finite state machine, not reactive to drone position.

| ID | Requirement |
|----|-------------|
| FR-07 | Suspect shall follow a 4-state FSM: **Fleeing → Roaming → Parking → Parked** |
| FR-08 | **Fleeing:** suspect exceeds speed limit by 60–100%, runs red lights, changes lanes aggressively. The first violation triggers dispatch; after all violation reports for the current difficulty are sent, the roam timer starts. |
| FR-09 | **Roaming:** suspect continues driving at high speed along a randomised route through the city. This is the primary tracking window. Duration: configurable (default 60–120 seconds). |
| FR-10 | **Parking:** suspect drives to a pre-designated surface parking lot and pulls into a bay. Speed gradually decreases to normal during approach. |
| FR-11 | **Parked:** suspect vehicle stops and remains stationary for the rest of the mission. The parking lot shall contain 20–40 other stationary vehicles with `set_simulate_physics(False)` to prevent drift. |

### 3.4 Alert Events

| ID | Requirement |
|----|-------------|
| FR-12 | System shall generate simulated CCTV ping events when suspect passes designated junction cameras, providing updated location |
| FR-13 | System shall generate witness report events with approximate (±50m) location at random intervals during Fleeing and Roaming states |
| FR-14 | System shall generate intercept suggestion events based on route prediction output |
| FR-15 | All events shall be delivered to the dashboard via WebSocket with a structured JSON payload |

---

## 4. Computer Vision Requirements

### 4.1 Vehicle Detection

| ID | Requirement |
|----|-------------|
| CV-01 | System shall use YOLOv8s (single-class `vehicle`; see design D-08 for size and D-09 for taxonomy) fine-tuned on CARLA-generated synthetic data. VisDrone warm-start optional — used only if the CARLA-only fine-tune underperforms against the eval holdout. |
| CV-02 | Detection shall run at minimum 20 FPS on RTX 4050 using fp16 inference |
| CV-03 | System shall achieve minimum 85% mAP@0.5 on aerial vehicle detection benchmark. **⚠ Aspirational**: published VisDrone-DET SOTA sits around 45–55% mAP@0.5 for aerial fine-tuned YOLO variants; 85% assumes in-domain CARLA training will substantially exceed VisDrone-like distribution — to be revalidated after exp 08 (see `docs/literature_survey.md` §1, §7). |
| CV-04 | Training data shall be generated using CARLA's instance segmentation camera to produce pixel-accurate bounding box labels automatically — no manual annotation |
| CV-05 | Detection input resolution shall be 640×640 pixels (standard YOLO input) |

### 4.2 Multi-Object Tracking

| ID | Requirement |
|----|-------------|
| CV-06 | System shall use a tracking-by-detection MOT with Kalman motion prediction — ByteTrack as the v1 baseline ([Zhang et al. 2022](https://arxiv.org/abs/2110.06864)); BoT-SORT ([arXiv:2206.14651](https://arxiv.org/abs/2206.14651)) or StrongSORT ([arXiv:2202.13514](https://arxiv.org/abs/2202.13514)) are preferred for the moving-camera scenario and may replace ByteTrack after exp 10 measures the camera-motion compensation benefit. See `docs/literature_survey.md` §2. |
| CV-07 | Tracker shall maintain track continuity through partial occlusion (e.g. other vehicles passing in front) |
| CV-08 | Tracker shall assign persistent track IDs that survive across frames |
| CV-09 | Suspect track ID shall be assigned at first confirmed detection and maintained throughout the mission |
| CV-10 | System shall handle high-speed target motion (up to 130 km/h) without track loss on open roads |

### 4.3 Vehicle Fingerprinting

The system shall build and maintain a multi-attribute fingerprint of the suspect vehicle from the top-down aerial view. License plate recognition is explicitly out of scope — plates face front/rear and are not visible from above at operational altitude. The fingerprint replaces ALPR and provides richer, more actionable information for ground units.

| ID | Attribute | Method |
|----|-----------|--------|
| CV-11 | Colour | HSV histogram on roof bounding box region |
| CV-12 | Vehicle class | One of `{car, van, truck, bus}`; sourced from the dispatch description at mission start and from CARLA blueprint metadata at training-data-capture time. **Not** produced by the detector — the detector is single-class (see design D-09). The fingerprint compares the dispatch class against candidates' blueprint-derived class, with soft-score weighting rather than hard filtering. |
| CV-13 | Roof shape | Bounding box aspect ratio + contour descriptor |
| CV-14 | Apparent size | Normalised bounding box area relative to altitude |
| CV-15 | Speed | Kalman filter velocity estimate from track positions |
| CV-16 | Heading | Track direction vector over last 10 frames |

The fingerprint shall be stored as a structured object and updated every 10 frames during active tracking.

### 4.4 Occlusion Recovery

| ID | Requirement |
|----|-------------|
| CV-17 | When tracker loses the suspect track (e.g. vehicle passes under bridge), system shall enter occlusion recovery mode |
| CV-18 | Recovery shall search the predicted re-emergence zone derived from road graph and last known velocity. Grounded in road-constrained tracking literature ([VS-IMM](https://ieeexplore.ieee.org/document/869492), [RBPF](https://arxiv.org/abs/1301.3853), [HMM map matching](https://dl.acm.org/doi/10.1145/1653771.1653818)) — see `docs/literature_survey.md` §4. |
| CV-19 | Each candidate vehicle in the search zone shall be scored against the stored fingerprint using weighted multi-attribute matching. Matching approach draws from vehicle Re-ID literature ([VRAI](https://arxiv.org/abs/1904.01400), [CLIP-ReID](https://arxiv.org/abs/2211.13977) for text-description-to-visual matching) — see `docs/literature_survey.md` §3. |
| CV-20 | System shall re-acquire the suspect track if a candidate scores above 0.75 confidence threshold. **⚠ Speculative**: threshold is un-calibrated; to be tuned empirically. |
| CV-21 | System shall log each occlusion event and recovery outcome for evaluation |

### 4.5 Lost Track Fallback

When occlusion recovery fails (no candidate above threshold within the predicted zone):

| ID | Requirement |
|----|-------------|
| CV-22 | System shall hold position at last known location for 5 seconds, scanning nearby roads |
| CV-23 | If not re-acquired, system shall execute an expanding spiral search pattern centred on last known position, covering a radius up to 200m. Informed by [POMDP UAV search with negative-information updates (Chung & Burdick 2012)](https://ieeexplore.ieee.org/document/6051437) — negative observations reduce posterior belief in the searched cell — `docs/literature_survey.md` §4. |
| CV-24 | During search, all CCTV pings and witness reports (if available at current difficulty) shall override the search pattern and redirect the drone |
| CV-25 | If the suspect is not re-acquired within 60 seconds of track loss, system shall flag the track as lost and continue searching until mission timeout. **⚠ 60s target is un-cited**: 5–30s occlusion recovery is an open empirical question in the literature (see `docs/literature_survey.md` §3 occlusion recovery note). To be validated during exp 10+. |

### 4.6 Parking Lot Re-Identification

| ID | Requirement |
|----|-------------|
| CV-26 | When suspect vehicle velocity drops below 2 km/h for 3 consecutive seconds, system shall enter parking identification mode |
| CV-27 | System shall use **approach trajectory tracking** as the primary identification method — the drone observes which parking bay the suspect pulls into |
| CV-28 | As secondary confirmation, system shall score all visible stationary vehicles in the parking lot against the stored fingerprint |
| CV-29 | System shall output a ranked candidate list with confidence scores |
| CV-30 | If top candidate scores above 0.85, system shall declare confirmed identification |
| CV-31 | If top candidate scores between 0.60 and 0.85, system shall flag as uncertain and highlight top 3 candidates on dashboard |
| CV-32 | If top candidate scores below 0.60, system shall declare mission failed and log the tracking degradation point in debrief |

### 4.7 Speed Estimation

| ID | Requirement |
|----|-------------|
| CV-33 | System shall estimate suspect vehicle speed in km/h from pixel displacement between frames, corrected for drone altitude and camera FOV |
| CV-34 | Speed estimate shall be displayed on dashboard and included in fingerprint profile |
| CV-35 | Speed estimation error shall not exceed ±15 km/h at operational altitude |

---

## 5. Simulation Requirements

### 5.1 Environment

| ID | Requirement |
|----|-------------|
| SIM-01 | Simulation shall use CARLA Town10HD_Opt map — dense urban environment with junctions, bridges, and surface parking lots |
| SIM-02 | System shall spawn 40–60 NPC vehicles via CARLA Traffic Manager on mission start |
| SIM-03 | Traffic shall be configured with realistic urban density — slight speed variation, occasional lane changes |
| SIM-04 | Simulation shall run in synchronous mode at 20 FPS fixed timestep for deterministic behaviour |
| SIM-05 | CARLA shall run at **Low quality level** to stay within the 6GB VRAM budget |
| SIM-06 | Weather shall be configurable — minimum three scenarios: ClearNoon, CloudyNight, WetCloudy |

### 5.2 Aerial Camera

| ID | Requirement |
|----|-------------|
| SIM-07 | Aerial camera shall be an RGB sensor: 1280×720 resolution, 90° FOV |
| SIM-08 | An instance segmentation camera co-located with the RGB camera shall run in parallel for training data generation |
| SIM-09 | Camera shall implement simulated noise: motion blur proportional to speed, vibration jitter (±2px), atmospheric haze at altitude |
| SIM-10 | System shall support both spectator camera (visual preview) and programmatic RGB sensor (CV pipeline input) simultaneously |

### 5.3 Adaptive Altitude Control

| ID | Requirement |
|----|-------------|
| SIM-11 | Drone camera shall target 15m altitude on open roads with no nearby buildings |
| SIM-12 | Drone camera shall climb to 40m when lateral clearance raycasts detect building geometry within 20m |
| SIM-13 | Altitude shall be hard-clamped between 10m (floor) and 60m (ceiling) at all times |
| SIM-14 | System shall perform downward raycasting to detect rooftop height below camera and maintain minimum 12m clearance above tallest structure |

### 5.4 PID Drone Controller

| ID | Requirement |
|----|-------------|
| SIM-15 | PID controller shall maintain suspect vehicle bounding box centred in frame using proportional-integral-derivative control on X and Y axes |
| SIM-16 | Altitude PID axis shall use bounding box pixel height as proxy for distance — targeting 80px apparent vehicle height |
| SIM-17 | Controller shall hold last known position when tracker loses the suspect, rather than drifting |
| SIM-18 | PID parameters shall be tunable via configuration file without code changes |

---

## 6. Game / Interactive Layer Requirements

### 6.1 Game Modes

| ID | Requirement |
|----|-------------|
| GM-01 | System shall support Manual Mode — player controls drone camera using keyboard via browser interface |
| GM-02 | System shall support Autonomous Mode — CV pipeline controls drone automatically |
| GM-03 | Player shall be able to toggle between modes at any point during a mission using Spacebar |
| GM-04 | Both modes shall play identical scenarios (same seed) for direct performance comparison |

### 6.2 Player Controls (Manual Mode)

Controls are captured via the browser (no local display required — runs headless in Docker).

| Key | Action |
|-----|--------|
| W / S | Move drone forward / backward |
| A / D | Strafe drone left / right |
| Q / E | Altitude down / up |
| Arrow keys | Look around (pitch / yaw) |
| Shift | Hold for speed boost |
| Space | Toggle autonomous mode on/off |
| R | Snap camera to last alert location |
| Tab | Toggle fullscreen drone view / dashboard |
| 1 / 2 / 3 | Jump to CCTV junction camera positions |

### 6.3 Difficulty Levels

| Level | Max speed | Violation reports | Roam duration | CCTV pings | Witness reports | Parking lot |
|-------|-----------|-------------------|---------------|------------|-----------------|-------------|
| Rookie | 60 km/h | 5 | 60s | Every 10s | Accurate ±10m | Open visible bay |
| Officer | 100 km/h | 3 | 90s | Every 30s | Noisy ±50m | Side street lot |
| Detective | 130 km/h | 1 | 120s | None | None | Covered corner bay |

### 6.4 Scoring

| Component | Weight | Calculation |
|-----------|--------|-------------|
| Tracking continuity | 40 pts | % of frames with active suspect track |
| Time to first acquisition | 20 pts | Inverse of seconds from dispatch to first lock |
| Occlusion recoveries | 20 pts | Number of successful re-acquisitions |
| Final identification | 20 pts | Correct bay identified in parking lot |

---

## 7. Dashboard Requirements

| ID | Requirement |
|----|-------------|
| DB-01 | Dashboard shall display live Leaflet.js city map with real-time drone position marker |
| DB-02 | Dashboard shall draw suspect vehicle trail as a polyline updated at 5 Hz |
| DB-03 | Dashboard shall display top 3 predicted escape route corridors as coloured overlays with probability labels |
| DB-04 | Dashboard shall display live drone camera feed with detection overlay (bounding box, track ID, speed, confidence) |
| DB-05 | Dashboard shall show a timestamped alert queue on the right panel — dispatch alerts, CCTV pings, witness reports |
| DB-06 | Dashboard shall display current suspect fingerprint profile: colour swatch, vehicle class, estimated speed, heading |
| DB-07 | Dashboard shall show live score and mission timer in the top bar |
| DB-08 | Dashboard shall display a mission debrief screen on completion showing: tracking continuity %, occlusion events, re-acquisitions, final confidence score, and human vs autonomous comparison if both played |
| DB-09 | All dashboard data shall be received via WebSocket from the CARLA simulation process |
| DB-10 | Dashboard shall be built with Streamlit and shall be launchable with a single command |

---

## 8. Coordinate Mapping Requirements

### 8.1 Pixel-to-World Projection

| ID | Requirement |
|----|-------------|
| CM-01 | System shall implement a `PixelToWorldProjector` class converting YOLO bounding box pixel centres to CARLA world coordinates using camera intrinsics, drone position, and ground plane intersection |
| CM-02 | Projector shall use camera FOV to compute focal length: `fx = (W/2) / tan(FOV/2)` |
| CM-03 | Projector shall apply rotation matrix from drone pitch and yaw to transform camera ray to world frame |
| CM-04 | Projector shall assume ground plane at `z = road_height` for ray-plane intersection |

### 8.2 World-to-GPS Conversion

| ID | Requirement |
|----|-------------|
| CM-05 | System shall use CARLA's built-in `map.transform_to_geolocation()` to convert world coordinates to latitude/longitude |
| CM-06 | GPS coordinates shall be broadcast via WebSocket to the Leaflet.js dashboard map |

### 8.3 Localisation Evaluation

| ID | Requirement |
|----|-------------|
| CM-07 | System shall log the Euclidean error between CV-projected world position and CARLA API ground truth position every frame |
| CM-08 | Mean localisation error shall be reported per altitude band (15m, 25m, 40m) in the evaluation output |
| CM-09 | Target localisation accuracy: ≤ 1.5m error at 15m altitude, ≤ 3.5m error at 40m altitude |

---

## 9. Non-Functional Requirements

### 9.1 Performance

| ID | Requirement |
|----|-------------|
| NFR-01 | Full CV pipeline (detection + tracking + fingerprint) shall run within a single 50ms tick on RTX 4050 6GB |
| NFR-02 | YOLOv8 inference shall use fp16 (half precision) to minimise VRAM consumption |
| NFR-03 | Total VRAM usage (CARLA rendering + CV inference + sensors) shall not exceed 5.5GB |
| NFR-04 | CARLA shall run at Low quality level; Epic quality exceeds the 6GB VRAM budget |
| NFR-05 | Dashboard WebSocket latency shall not exceed 100ms |
| NFR-06 | Depth raycasting for collision safety shall complete within 5ms per tick |

### 9.2 Reproducibility

| ID | Requirement |
|----|-------------|
| NFR-07 | All CARLA scenarios shall be seeded with a configurable random seed for reproducible evaluation runs |
| NFR-08 | Project shall include a `requirements.txt` for Python dependency management |
| NFR-09 | Project shall include a Docker Compose file enabling single-command environment setup |
| NFR-10 | All trained model weights shall be versioned and downloadable via a release tag |

### 9.3 Code Quality

| ID | Requirement |
|----|-------------|
| NFR-11 | All modules shall have unit tests for CV components (detection, fingerprinting, projection) |
| NFR-12 | Codebase shall follow PEP 8 with type hints on all public methods |
| NFR-13 | Configuration (PID gains, altitude thresholds, scoring weights) shall be externalised to YAML — no magic numbers in code |

---

## 10. Out of Scope

The following are explicitly out of scope for this project. Each is a deliberate, reasoned decision — not an omission.

| Item | Reason |
|------|--------|
| License plate recognition (ALPR) | Plates are vertical surfaces facing front/rear. From a top-down aerial view they are geometrically invisible. This is a physical constraint, not a model limitation. |
| Full drone physics (rotors, IMU, wind) | CV pipeline is independent of flight controller. Contribution is in the perception and intelligence layer. Real deployment would replace `set_transform()` with MAVLink commands. |
| Multi-storey / underground garage | Interior geometry not available in CARLA. Surface parking lot provides equivalent CV challenge (static re-identification) with full visual access. |
| Power line / cable avoidance | Power lines are strung at 6–12m height. System operates at 15–40m. Not a realistic collision scenario at operational altitude. Noted as future work for sub-15m deployment. |
| Real-world drone hardware integration | Simulation-only project. Architecture is designed for hardware portability. |
| Multi-drone coordination | Single drone system. Multi-drone is noted as a natural extension in future work. |
| Person / pedestrian detection | Suspect is always a vehicle. Pedestrian CV is a separate problem domain. |

---

## 11. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| CARLA has no indoor garage geometry | Cannot simulate underground parking | Use surface parking lot — equivalent re-ID challenge |
| Power lines not modelled in CARLA | No wire collision simulation | State explicitly in README; add custom cylinder actors if needed |
| CARLA spectator camera has no native gimbal model | Camera orientation tied to drone body | Decouple via transform offset in controller |
| YOLOv8 accuracy degrades on very small objects at 40m+ | Detection confidence drops on distant vehicles | Use adaptive altitude — fly lower on open roads |
| Sim-to-real domain gap | Models trained in CARLA may not transfer directly to real footage | Domain randomisation during training (weather, lighting, textures) |
| RTX 4050 6GB VRAM limits quality | Must run CARLA at Low quality | Acceptable for CV training; visual fidelity is sufficient for detection tasks |

---

## 12. Evaluation Metrics

The following metrics shall be computed automatically at mission end and logged to a JSON evaluation file.

| Metric | Definition | Target |
|--------|------------|--------|
| Tracking continuity | % of frames with active suspect track | ≥ 90% (autonomous) |
| Mean time to first acquisition | Seconds from dispatch to first confirmed lock | ≤ 8s |
| Occlusion recovery rate | % of occlusion events where track was re-acquired | ≥ 85% |
| Parking identification accuracy | % of missions where correct bay identified | ≥ 80% at Officer difficulty |
| Mean localisation error (15m) | Euclidean error: projected vs ground truth | ≤ 1.5m |
| Mean localisation error (40m) | Euclidean error: projected vs ground truth | ≤ 3.5m |
| Detection mAP@0.5 | Standard COCO metric on VisDrone test set | ≥ 85% |
| Human vs autonomous delta | Tracking continuity: autonomous − human | ≥ +15% |

---

## 13. Tech Stack Summary

| Layer | Component | Tool / Library |
|-------|-----------|----------------|
| Simulation | City, traffic, physics | CARLA 0.9.16 (Town10HD) |
| Suspect AI | Finite state machine | Python — custom FSM |
| Alert system | Dispatch, CCTV pings, witness | Python event queue → WebSocket |
| Detection | Aerial vehicle detection | YOLOv8m (Ultralytics) |
| Tracking | Multi-object tracking | ByteTrack + Kalman filter |
| Fingerprinting | Appearance matching | OpenCV HSV + custom scorer |
| Depth / safety | Obstacle raycasting | CARLA `world.cast_ray()` |
| Control | Drone PID controller | Python — custom PID class |
| Player input | Manual drone control | Browser keyboard capture (Flask) |
| Coordinate mapping | Pixel → world → GPS | NumPy + CARLA geolocation API |
| Route prediction | Escape route graph | NetworkX + CARLA waypoint API |
| Dashboard | Map, alerts, camera feed | Streamlit + Leaflet.js |
| Communication | CARLA → dashboard | asyncio WebSocket |
| Training | Model training | PyTorch fp16 + Ultralytics |
| Experiment tracking | Metrics, loss curves | Weights & Biases (wandb) |
| Environment | Containerised setup | Docker + Docker Compose |

---

## 14. Milestones

| Phase | Deliverable | Definition of Done |
|-------|-------------|-------------------|
| Environment | CARLA setup | Town10, 50 NPC vehicles, suspect spawned, aerial camera streaming frames, adaptive altitude working |
| Detection | CV detection + tracking | YOLOv8m fine-tuned, ByteTrack integrated, suspect tracked at 80 km/h, mAP ≥ 85% |
| Suspect AI | FSM + alerts | All 4 states running, dispatch/CCTV/witness events firing, parking lot pre-populated |
| Re-ID | Fingerprinting + re-ID | Fingerprint built on first detection, occlusion recovery working, parking lot identification outputting confidence scores |
| Dashboard | Dashboard + player controls | Streamlit dashboard live with map/feed/alerts, manual mode playable, scoring computed, debrief rendered |
| Polish | Evaluation + packaging | Demo video recorded, README complete, evaluation metrics logged, Docker setup tested, architecture diagram included |

---

## Appendix A — Suspect FSM State Transitions

```
                    ┌─────────────┐
                    │   Fleeing   │ ← mission start, reckless driving
                    └──────┬──────┘
                           │ 1st violation → dispatch alert
                           │ last violation report sent → roam timer starts
                           ▼
                    ┌─────────────┐
                    │   Roaming   │ ← still driving fast, random route
                    └──────┬──────┘
                           │ roam timer expires (60–120s)
                           ▼
                    ┌─────────────┐
                    │   Parking   │ ← drives to lot, slows down
                    └──────┬──────┘
                           │ vehicle stops in bay
                           ▼
                    ┌─────────────┐
                    │   Parked    │ ← stationary, mission finale
                    └─────────────┘
```

The suspect is **unaware of the drone** throughout the mission. Behaviour is entirely scripted — no reactive evasion based on drone proximity.

## Appendix B — Coordinate Projection Pipeline

```
Camera pixel (cx, cy)
        │
        │  pixel_to_world()
        │  using: focal length, drone position, drone pitch/yaw
        ▼
CARLA world coords (x, y)   ← compared to ground truth for error metric
        │
        │  map.transform_to_geolocation()
        ▼
GPS (latitude, longitude)
        │
        │  WebSocket → dashboard
        ▼
Leaflet.js map marker
```

## Appendix C — Fingerprint Schema

```json
{
  "track_id": 7,
  "class": "SUV",
  "colour_hsv": [0.61, 0.08, 0.22],
  "colour_label": "dark grey",
  "bbox_aspect_ratio": 1.42,
  "normalised_area": 0.0031,
  "last_speed_kmh": 74.2,
  "last_heading_deg": 22.0,
  "confidence": 0.91,
  "occlusion_count": 1,
  "frames_tracked": 847,
  "last_world_pos": [230.4, 118.7]
}
```

## Appendix D — VRAM Budget (RTX 4050 6GB)

| Component | Estimated VRAM |
|-----------|---------------|
| CARLA server (Low quality, Town10HD) | 2.0–3.5 GB |
| YOLOv8m fp16 inference | 400–600 MB |
| ByteTrack + fingerprinting | ~100 MB |
| RGB + instance segmentation cameras | ~200 MB |
| **Total** | **2.7–4.4 GB** |
| **Headroom** | **1.6–3.3 GB** |

Epic quality is not feasible on this hardware — CARLA rendering alone would consume 4–5 GB.

---

*Document maintained in `REQUIREMENTS.md` in the project repository.*  
*Last updated: April 2026*
