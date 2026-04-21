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
| 08 | `finetune_yolo` | Fine-tune YOLOv8s on exp 07 dataset, score same-map + cross-map | ✅ | `output/weights/run_v1/weights/best.pt` + `output/eval/fine_tuned_metrics.json` + `output/eval/carla_eval_town01/` |
| 09 | `yolo_inloop` | Fine-tuned weights live, boxes + conf overlaid on MJPEG (compare vs baseline visually and in FPS) | ✅ | `output/eval/inloop_metrics.json` |
| 10 | `bytetrack` | ByteTrack on detections + suspect-continuity eval | ✅ | `output/eval/tracking_metrics.json` + `output/eval/tracking/run_A,B/` |
| 11 | `fingerprint` | HSV colour + geometry attributes per track (CV-11..16) | ⬜ | — |
| 12 | `mission_tracer` | First end-to-end mission slice: dispatch → detect → track → mission-end | ⬜ | — |
| 13 | `dashboard_scoring` | Streamlit + event bus + scoring + debrief | ⬜ | — |

## Currently

- [x] Exp 05 — CARLA pursuit eval holdout (200 frames, all 4 classes represented)
- [x] Exp 06 — Pretrained baseline: 26 FPS sustained, mAP@0.5 = 4.95% single-class (pretrained does not recognise aerial vehicles — ~5% recall is the real problem, class confusion is now off the board per D-09)
- [x] Exp 07 — Training-data collection: 5000 frames across 10 runs, 3 weather presets, ~22,000 labels all class-0
- [x] Literature + industry survey (`docs/literature_survey.md`); REQUIREMENTS.md annotated with citations/flags; design log D-10
- [x] **SIGABRT teardown fix** — `skycop.sim.teardown_pursuit` replaces ad-hoc cleanup in all pursuit callers. Exp 05 now exits clean (code 0). Sequence documented in `docs/carla_caveats.md` §6a.
- [x] Exp 08 — Fine-tune YOLOv8s: same-map mAP@0.5 = 0.962, cross-map 0.581 (38-pt gap — real features learned + Town10HD overfit component). Design log D-11.
- [x] Exp 09 — Fine-tuned in-loop inference: 25.0 FPS (baseline 26.2, delta −1.17), 13.9 ms mean detection, 0.042 GB peak VRAM — all within NFR-01/NFR-03. Live pursuit verified end-to-end.
- [x] Exp 10 — ByteTrack + suspect-continuity eval: max continuity **0.996** (run_B), min **0.724** (run_A), mean 0.86 across 2 runs. Verdict **GOOD** (CV-07 ≥ 0.80 on at least one run).
- [ ] **teardown_pursuit dual-sensor hang fix** — separate small PR; `apply_batch_sync` times out at 60 s after dual-RGB-seg every-tick captures; proposed fix: destroy heavy sensors individually before the batch to free render targets first.
- [ ] Exp 11 — fingerprint (HSV colour + geometry, CV-11..16)
- [ ] Exp 11 — fingerprint
- [ ] Exp 12 — first end-to-end mission
- [ ] Exp 11-v2 *(conditional)* — re-fine-tune with multi-map training data if exp 09 visual inspection shows the 38-pt cross-map gap is operationally problematic

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

### Tracking — exp 10 numbers

ByteTrack (Ultralytics' bundled) on fine-tuned YOLOv8s detections. Primary metric is **suspect track continuity** per D-11 — scene-wide HOTA/MOTA not computed because the mission cares about one track, not the whole scene. Two 250-frame captures across different seeds + weathers for hard-moment diversity.

| Run | Seed | Weather | Suspect | Continuity | ID switches | Suspect present / detected / total |
|---|---|---|---|---|---|---|
| run_A | 300 | ClearNoon | dodge.charger_police | 0.724 | 14 | 237 / 221 / 250 |
| run_B | 301 | WetCloudyNoon | jeep.wrangler_rubicon | **0.996** | 0 | 247 / 247 / 250 |
| Mean | — | — | — | **0.860** | — | — |

**Verdict: GOOD** (CV-07 compliance: continuity ≥ 0.80 on at least one run). Interesting delta between runs — run_B was essentially perfect, run_A had 14 ID switches caused by 16 frames of briefly-missed detections that ByteTrack's Kalman prediction couldn't hold through. Fingerprint module (exp 11) should close this gap via appearance-based re-ID across brief loss events.

Known infra issue documented alongside: **`teardown_pursuit` hangs** when capturing with dual sensors (RGB + instance-seg) at every-tick rate for 250+ frames. `apply_batch_sync` times out at 60 s and SIGABRTs the process. Data survives (tracks.json is written before teardown), but the eval runs in a separate process after captures land on disk. Proper fix (destroy heavy sensors first before the batch) queued as a separate PR.

### Detection in-loop — exp 09 numbers

Fine-tuned YOLOv8s weights from exp 08 dropped into the live pursuit pipeline. 60 s run, MJPEG overlay on `http://localhost:5000` during the run for visual check (can't be automated in metrics).

| Metric | Pretrained baseline (exp 06) | Fine-tuned (exp 09) | Δ |
|---|---|---|---|
| Sustained FPS (60 s run) | 26.16 | **24.99** | −1.17 |
| Detection mean | 11.80 ms | **13.93 ms** | +2.12 ms |
| Detection p95 | 14.99 ms | 15.05 ms | +0.06 ms |
| Peak VRAM (torch) | 0.032 GB | **0.042 GB** | +0.010 GB |
| Frame total | 38.22 ms | 40.01 ms | +1.79 ms |

Both comfortably inside the 18 FPS threshold and 5.5 GB VRAM budget. The small detection-time increase is most likely because the fine-tuned model emits higher-confidence detections that pass the 0.25 threshold (more boxes → more NMS work) rather than any architectural difference — same YOLOv8s, same fp16 path.

Infrastructure refactor: `skycop/cv/inloop.py` now owns the live-pursuit measurement loop; both exp 06 and exp 09 call it. Previously exp 06 embedded ~100 lines of loop code inline.

### Detection fine-tune — exp 08 numbers

YOLOv8s fine-tuned from pretrained COCO on the exp 07 dataset (3500 train frames / 1500 val / 7 runs / 3 weather presets on Town10HD_Opt). Training: 30 epochs, batch-size auto-picked at 4 (AutoBatch at 60% VRAM target — **underutilised**, see AutoBatch note below), fp16 via AMP, ~29 min wall-clock.

| Eval set | Source | Frames | mAP@0.5 | mAP@0.5:0.95 | Predictions / GT |
|---|---|---|---|---|---|
| **Same-map** | exp 05 holdout, Town10HD_Opt, seed 42 | 200 | **0.962** | 0.812 | 831 / 809 |
| **Cross-map** | Town01_Opt probe, seed 900 | 100 | **0.581** | 0.561 | 116 / 157 |
| Pretrained baseline (exp 06, same-map) | same as above | 200 | 0.050 | 0.041 | 37 / 804 |
| **Gap (same − cross)** | — | — | **0.381** | 0.251 | — |

**Interpretation (encoded in D-11):** PARTIAL. Same-map 96.2% far exceeds CV-03's 85% aspirational target, confirming the detector learned something real — cross-map 58.1% is still 12× the pretrained baseline, so it's not pure memorisation. But a 38-point gap is a substantial Town10HD-specific overfitting component. Multi-map training (exp 11-v2 candidate) is the right intervention if downstream experiments show the cross-map gap hurts the mission.

**AutoBatch headroom (documented for future fine-tunes).** Ultralytics' AutoBatch picked `batch=4` targeting 60% CUDA memory at its probe. Actual training ran at **25% VRAM usage** (1.5 GB / 6 GB) at 88% compute utilisation. Force `batch=16` on subsequent fine-tunes to halve epoch time — memory headroom verified via nvidia-smi sampling.

**Cross-map choice — Town01_Opt, not Town03.** Town03 was first pick but CARLA 0.9.16 segfaults on loading Town03 with 50 NPCs queued (renderer crash in UE4, not our bug). Town01_Opt loads cleanly and has enough road variety to function as a meaningful generalisation probe. Documented inline in `configs/training.yaml` so future authors don't re-try Town03.

**Shared-memory fix.** `docker-compose.yml` adds `shm_size: 2gb` to the client service. PyTorch dataloader workers exchange batches via `/dev/shm`; Docker's default 64 MB is too small and workers die with "unable to allocate shared memory" — training then stalls at 0% GPU utilisation. Noted in the compose file itself.

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

- **2026-04-21** · #23 — Exp 10: ByteTrack + suspect-continuity eval. Max 0.996 / mean 0.86. `skycop.cv.track` + `skycop.cv.tracking_eval` + `scripts/10_bytetrack.py`. Teardown-hang on dual-sensor captures documented for follow-up.
- **2026-04-21** · #21 — Exp 09: fine-tuned YOLOv8s in-loop inference. 25.0 FPS / 13.9 ms mean / 0.042 GB VRAM — all within NFR-01/NFR-03. Refactored live-pursuit measurement loop into `skycop.cv.inloop` shared by exp 06 and exp 09.
- **2026-04-21** · #19 — Exp 08: YOLOv8s fine-tune. Same-map 0.962 / cross-map 0.581 / 38-pt gap (PARTIAL — genuine aerial features + Town10HD overfit). Design log D-11 documents cross-map probe methodology. `docker-compose.yml` gains `shm_size: 2gb` (PyTorch dataloader workers need ≥1 GB, 64 MB default stalls training at 0% GPU util).
- **2026-04-20** · #15 — Proper CARLA pursuit teardown sequence (`skycop.sim.teardown_pursuit`); SIGABRT on cleanup resolved across all pursuit scripts; docs/carla_caveats.md §6a documents root cause + fix
- **2026-04-20** · #13 — Literature + industry survey retrofit: `docs/literature_survey.md` + REQUIREMENTS.md citation audit + design log D-10
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
