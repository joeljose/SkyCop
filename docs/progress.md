# SkyCop — Progress Log

Live state of the project: which milestones are closed, which experiments have landed, what's next. Updated inside the branch that closes each unit of work — never as a stale afterthought.

**Last updated:** 2026-04-20

**Legend:** ✅ complete · 🔨 in progress · ⬜ planned

---

## Milestones

Mirrors `REQUIREMENTS.md §14`. Each milestone is closed when its Definition of Done is met and the experiments listed under "evidence" have landed.

| Phase | Status | DoD (abbrev.) | Evidence |
|---|---|---|---|
| Environment | ✅ | Town10, 50 NPCs, suspect, aerial camera streaming, adaptive altitude | exp 01–04 |
| Detection | 🔨 | YOLOv8 fine-tuned (VisDrone + CARLA synthetic), ByteTrack integrated, suspect tracked at 80 km/h, mAP ≥ 85% | exp 05–08 planned |
| Suspect AI | ⬜ | 4-state FSM, dispatch / CCTV / witness events, parking lot pre-populated | — |
| Re-ID | ⬜ | Fingerprint on first detection, occlusion recovery, parking-lot identification with confidence | — |
| Dashboard | ⬜ | Streamlit live, manual mode playable, scoring, debrief | — |
| Polish | ⬜ | Demo video, README, eval JSON, Docker tested, architecture diagram | — |

## Experiments

One row per `scripts/NN_*.py`. "Produces" names the durable artifact the experiment hands to the next step (training data, weights, metrics) — not transient state.

| # | Script | Purpose | Status | Produces |
|---|---|---|---|---|
| 01 | `hello_world` | Connect, spawn vehicle, capture one frame | ✅ | `output/01_hello_world.png` |
| 02 | `drone_view` | Free-fly drone via Flask keyboard (smoke test for the spectator + MJPEG pipeline) | ✅ | — (interactive) |
| 03 | `suspect_and_traffic` | 50 NPCs + reckless suspect, fixed-altitude aerial follow | ✅ | — |
| 04 | `adaptive_altitude` | SIM-11..14: adaptive altitude via `world.cast_ray()`, hybrid physics anchored on hero | ✅ | — |
| 05 | `collect_dataset` | Sweep drone pitch/alt/yaw/weather; RGB + instance-seg → YOLO-format labels | ⬜ | `output/dataset/` + `dataset_manifest.json` |
| 06 | `yolo_baseline` | Pretrained YOLOv8s on our frames → measure mAP; demonstrates the domain gap | ⬜ | `output/baseline/metrics.json` |
| 07 | `finetune_yolo` | Fine-tune YOLOv8s on VisDrone warm-start + CARLA synthetic | ⬜ | `output/weights/best.pt` |
| 08 | `yolo_inloop` | Drop fine-tuned weights into the live pipeline; boxes + IDs overlaid on MJPEG | ⬜ | — |
| 09 | `mission_tracer` | First end-to-end mission slice: dispatch → detect → track → mission-end (stub tracking / fingerprint) | ⬜ | — |

## Currently

- [x] Terminology rename (`lesson` → `experiment`) and this progress log (#1)
- [ ] **Exp 05 — synthetic dataset collection** (next)
- [ ] Exp 06 — baseline pretrained inference
- [ ] Exp 07 — fine-tune on VisDrone + CARLA synthetic
- [ ] Exp 08 — in-loop inference with fine-tuned weights
- [ ] Exp 09 — first end-to-end mission (tracer bullet)

## Log

Reverse chronological. One line per landed PR. Commit SHA linked where relevant.

- **2026-04-20** · `4e027f5` Add `docs/design.md` — living application design record
- **2026-04-20** · `03b028c` Add adaptive altitude controller + OmegaConf configs (exp 04)
- **2026-04-20** · `634c660` Restructure around `skycop/` package with pyproject-driven deps (exps 01–03 refactored)
- **2026-04-19** · `5815c09` Initial project setup
