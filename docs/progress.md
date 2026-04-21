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
| 06 | `yolo_baseline` | Pretrained YOLOv8s in-loop baseline — FPS/VRAM/mAP on exp 05 holdout | ✅ | `output/eval/baseline_metrics.json` |
| 07 | `collect_training_data` | CARLA pursuit dataset — 10 runs × 500 frames across 3 weather presets | ✅ | `output/dataset/carla_pursuit/` (5000 jpg + 5000 txt + dataset_manifest.json) |
| 08 | `finetune_yolo` | Fine-tune YOLOv8s on exp 07 dataset, score against exp 05 holdout | ⬜ | `output/weights/best.pt` + `output/eval/fine_tuned_metrics.json` |
| 09 | `yolo_inloop` | Fine-tuned weights live, boxes + conf overlaid on MJPEG (compare vs baseline visually and in FPS) | ⬜ | — |
| 10 | `bytetrack` | Multi-object tracker on top of detections — persistent track IDs | ⬜ | — |
| 11 | `fingerprint` | HSV colour + geometry attributes per track (CV-11..16) | ⬜ | — |
| 12 | `mission_tracer` | First end-to-end mission slice: dispatch → detect → track → mission-end | ⬜ | — |
| 13 | `dashboard_scoring` | Streamlit + event bus + scoring + debrief | ⬜ | — |

## Currently

- [x] Exp 05 — CARLA pursuit eval holdout (200 frames, all 4 classes represented)
- [x] Exp 06 — Pretrained baseline: 26 FPS sustained, mAP@0.5 = 4.95% single-class (pretrained does not recognise aerial vehicles — ~5% recall is the real problem, class confusion is now off the board per D-09)
- [x] Exp 07 — Training-data collection: 5000 frames across 10 runs, 3 weather presets, ~22,000 labels all class-0
- [ ] **Literature survey PR** — grounds requirements in actual citations before more experiments; revisits REQUIREMENTS.md in light of findings
- [ ] **SIGABRT teardown fix** — proper fix for the `set_actor_simulate_physics` registry race on run cleanup (tracked for after survey)
- [ ] Exp 08 — Fine-tune YOLOv8s on the exp 07 dataset
- [ ] Exp 09 — fine-tuned weights in live pipeline (visual + FPS comparison vs baseline)
- [ ] Exp 10 — ByteTrack integration
- [ ] Exp 11 — fingerprint
- [ ] Exp 12 — first end-to-end mission
- [ ] Follow-up chore — convert scripts 01–05 to `logging` (script 06 already on it)

### Detection baseline — exp 06 numbers (single-class taxonomy)

Re-run after the taxonomy collapse to single-class `vehicle` (design D-09). Pretrained COCO YOLOv8s on `output/eval/carla_eval/` (200 frames, 804 vehicle GT boxes):

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| Sustained FPS (live pursuit, 60s) | 25.63 | ≥ 18 | ✅ plenty of headroom |
| Detection mean / p95 | 12.00 / 15.07 ms | — | ✅ fits the 50 ms tick budget |
| Peak VRAM (torch process) | 0.03 GB | ≤ 5.5 GB | ✅ (CARLA itself dominates GPU, not the model) |
| mAP@0.5 (single class `vehicle`) | **0.050** | — | ❌ pretrained unfit |
| mAP@0.5:0.95 | 0.041 | — | ❌ |
| Predictions vs ground truths | 37 / 804 | — | Recall ≈ 4.6% — pretrained genuinely does not recognise aerial vehicles |

**Interpretation:** the pipeline works end-to-end at speed but the pretrained model sees only a tiny fraction of vehicles. The 4-class → 1-class collapse removed class-confusion as a confounding factor; the remaining 95% gap is pure recall. Exp 08 targets this directly via in-domain CARLA fine-tuning.

### Training dataset — exp 07 numbers

Collected over 10 independent pursuits on Town10HD_Opt (seeds 100–109, distinct from the exp 05 eval seed = 42):

| Split | Runs | Frames | Vehicle labels | Weather coverage |
|---|---|---|---|---|
| Train | 7 | 3,500 | ~16,000 | ClearNoon ×3 · CloudyNight ×2 · WetCloudyNoon ×2 |
| Val | 3 | 1,500 | ~6,400 | ClearNoon ×1 · CloudyNight ×1 · WetCloudyNoon ×1 |
| **Total** | **10** | **5,000** | **~22,400** | — |

Diverse suspect vehicles drawn across runs (Ford Crown, Dodge Charger, Carlacola, VW T2, Mercedes Coupe, Microlino, etc.) so the detector isn't structurally biased toward any one make/model being "the target." Run-level split (not frame-level) enforced — train and val share zero frames. Each run per-run manifest records the seed, weather, per-frame camera pose, and class-count stats. Capture wall-clock: ~28 minutes for the full sweep.

### Known gaps / debt

- **`dropped_unknown_class: 9423`** in the manifest is not error count — it's a per-frame sum of non-vehicle Unreal mesh component IDs the extractor correctly filtered out (roads, buildings, signs). Could be optimised by pre-filtering on the semantic-label R channel; cheap to do but not worth doing until profiling shows capture is a bottleneck.

### Deliberate scope choices

- **4-class taxonomy — motorcycles excluded.** CARLA's Traffic Manager cannot autopilot 2-wheelers, so our pursuit scenes produce zero motorcycle training or eval data. Carrying a class we can neither train nor evaluate on is worse than dropping it; `skycop.cv.vehicle_classes` documents how to reinstate "motorcycle" if that constraint goes away.

## Log

Reverse chronological. One line per landed PR.

- **2026-04-20** · #11 — Exp 07: training-data collection (5000 frames, 10 runs, 3 weather presets); `skycop.cv.capture` helper + subprocess-per-run orchestrator; SIGABRT on CARLA teardown worked around but flagged for proper fix
- **2026-04-20** · #9 — Detector taxonomy collapsed to single-class `vehicle`; fingerprint classes preserved for CV-12; eval holdout + baseline re-run under new taxonomy; design log D-09
- **2026-04-20** · #7 — Exp 06: pretrained YOLOv8s baseline (FPS/VRAM + mAP on holdout); design log D-08; `skycop.logs` + `skycop.cv.inference` + `skycop.cv.eval`
- **2026-04-20** · #5 — Aerial camera pitch −90° → −75° (design log D-07); eval holdout regenerated under the new operational distribution
- **2026-04-20** · #3 — Exp 05: CARLA pursuit eval holdout capture + `skycop.cv.dataset` / `vehicle_classes`
- **2026-04-20** · #2 `be7ba5c` — chore: rename lesson→experiment + add progress log
- **2026-04-20** · `4e027f5` Add `docs/design.md` — living application design record
- **2026-04-20** · `03b028c` Add adaptive altitude controller + OmegaConf configs (exp 04)
- **2026-04-20** · `634c660` Restructure around `skycop/` package with pyproject-driven deps (exps 01–03 refactored)
- **2026-04-19** · `5815c09` Initial project setup
