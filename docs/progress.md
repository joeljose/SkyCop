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
| Detection | 🔨 | YOLOv8s fine-tuned on 4 classes (car/van/truck/bus), ByteTrack integrated, mAP ≥ 85% on CARLA holdout | exp 05 ✅ · exp 06–08 planned |
| Suspect AI | ⬜ | 4-state FSM, dispatch / CCTV / witness events, parking lot pre-populated | — |
| Re-ID | ⬜ | Fingerprint on first detection, occlusion recovery, parking-lot identification with confidence | — |
| Dashboard | ⬜ | Streamlit live, manual mode playable, scoring, debrief | — |
| Polish | ⬜ | Demo video, README, eval JSON, Docker tested, architecture diagram | — |

## Experiments

One row per `scripts/NN_*.py`. "Produces" names the durable artifact the experiment hands to the next step.

| # | Script | Purpose | Status | Produces |
|---|---|---|---|---|
| 01 | `hello_world` | Connect, spawn vehicle, capture one frame | ✅ | `output/01_hello_world.png` |
| 02 | `drone_view` | Free-fly drone via Flask keyboard (smoke test for the spectator + MJPEG pipeline) | ✅ | — (interactive) |
| 03 | `suspect_and_traffic` | 50 NPCs + reckless suspect, fixed-altitude aerial follow | ✅ | — |
| 04 | `adaptive_altitude` | SIM-11..14: adaptive altitude via `world.cast_ray()`, hybrid physics anchored on hero | ✅ | — |
| 05 | `capture_eval_set` | CARLA pursuit eval holdout — 200 frames + YOLO labels + manifest | ✅ | `output/eval/carla_eval/` (200 jpg + 200 txt + manifest.json) |
| 06 | `finetune_yolo` | YOLOv8s fine-tune on VisDrone warm-start; scores pretrained + fine-tuned on exp 05 holdout | ⬜ | `output/weights/best.pt` + `output/eval/metrics.json` |
| 07 | `yolo_inloop` | Fine-tuned weights in live pipeline, boxes + conf overlaid on MJPEG | ⬜ | — |
| 08 | `bytetrack` | Multi-object tracker on top of detections — persistent track IDs | ⬜ | — |
| 09 | `fingerprint` | HSV colour + geometry attributes per track (CV-11..16) | ⬜ | — |
| 10 | `mission_tracer` | First end-to-end mission slice: dispatch → detect → track → mission-end | ⬜ | — |
| 11 | `dashboard_scoring` | Streamlit + event bus + scoring + debrief | ⬜ | — |
| 12 | `finetune_round2` | *If* exps 07–10 exposed detector gaps, fine-tune again on in-app-captured pursuit frames | ⬜ | `output/weights/best_v2.pt` |

## Currently

- [x] Exp 05 — CARLA pursuit eval holdout (200 frames, all 4 classes represented)
- [ ] **Exp 06 — VisDrone warm-start fine-tune + mAP on holdout** (next)
- [ ] Exp 07 — fine-tuned weights in live pipeline
- [ ] Exp 08 — ByteTrack integration
- [ ] Exp 09 — fingerprint
- [ ] Exp 10 — first end-to-end mission

### Known gaps / debt

- **`dropped_unknown_class: 9423`** in the manifest is not error count — it's a per-frame sum of non-vehicle Unreal mesh component IDs the extractor correctly filtered out (roads, buildings, signs). Could be optimised by pre-filtering on the semantic-label R channel; cheap to do but not worth doing until profiling shows capture is a bottleneck.

### Deliberate scope choices

- **4-class taxonomy — motorcycles excluded.** CARLA's Traffic Manager cannot autopilot 2-wheelers, so our pursuit scenes produce zero motorcycle training or eval data. Carrying a class we can neither train nor evaluate on is worse than dropping it; `skycop.cv.vehicle_classes` documents how to reinstate "motorcycle" if that constraint goes away.

## Log

Reverse chronological. One line per landed PR.

- **2026-04-20** · #3 — Exp 05: CARLA pursuit eval holdout capture + `skycop.cv.dataset` / `vehicle_classes`
- **2026-04-20** · #2 `be7ba5c` — chore: rename lesson→experiment + add progress log
- **2026-04-20** · `4e027f5` Add `docs/design.md` — living application design record
- **2026-04-20** · `03b028c` Add adaptive altitude controller + OmegaConf configs (exp 04)
- **2026-04-20** · `634c660` Restructure around `skycop/` package with pyproject-driven deps (exps 01–03 refactored)
- **2026-04-19** · `5815c09` Initial project setup
