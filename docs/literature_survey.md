# SkyCop — Literature and Industry Survey

Retrofit survey grounding SkyCop's requirements and design in published work. Compiled 2026-04-20, after exp 06/07 empirically reversed several day-one speculative choices in `REQUIREMENTS.md`. Intended as the citation index that requirements and design entries reference — not as an exhaustive review.

**Scope of this document:** academic state of the art and commercial landscape for each sub-problem SkyCop touches, followed by a positioning summary of where the project sits, followed by a per-requirement citation-or-flag audit.

**Honesty note.** Where I could not verify a citation (author, year, arXiv ID, or exact claim) I've flagged it. Unverified citations are included as pointers, not authority. Downstream writers on this project should confirm specifics before quoting numbers.

---

## 1. Aerial vehicle detection

### Benchmarks

| Benchmark | Viewpoint | Vehicle scale | Relevance to SkyCop |
|---|---|---|---|
| **VisDrone-DET** ([Zhu et al., arXiv:2001.06303](https://arxiv.org/abs/2001.06303)) | 10–150 m altitude, Chinese cities, 10 classes including car/van/truck/bus | Median vehicle ~40×40 px at 2000×1500 | Weak fit — higher altitude than our 15–40 m, smaller vehicles than our 30–170 px |
| **UAVDT** ([Du et al., arXiv:1804.00518](https://arxiv.org/abs/1804.00518)) | UAV vehicle detection + MOT, altitude/weather/angle attributes | Varied, includes lower altitudes | Closer to pursuit scenario than VisDrone |
| **DOTA** ([Xia et al., arXiv:1711.10398](https://arxiv.org/abs/1711.10398)) | Satellite/aerial, oriented boxes | Vehicles 10–50 px | Mostly irrelevant to our altitude but widely used for oriented-box methods |
| **AI-TOD** (tiny-object benchmark, arXiv ID uncertain) | Sub-20 px objects | Mean ~12.8 px | Not our regime; referenced for small-object limits |

**Best-reported mAP@0.5 on VisDrone-DET** for strong detectors sits in the 40–55% range, well below COCO performance for the same architectures — quantifying the aerial domain gap. This is consistent with exp 06's finding that COCO-pretrained YOLOv8s scored 4.95% on our holdout.

### Detectors

- **YOLOv8** (Ultralytics, 2023) — the workhorse for aerial fine-tuning. Typical out-of-box VisDrone-DET mAP@0.5 sits around 35–45%, rising to 45–55% with aerial fine-tuning.
- **YOLOv10** ([Wang et al., NeurIPS 2024, arXiv:2405.14458](https://arxiv.org/abs/2405.14458)) — NMS-free training, improved small-object recall.
- **RT-DETR** ([Zhao et al., CVPR 2024, arXiv:2304.08069](https://arxiv.org/abs/2304.08069)) — real-time DETR competitive with YOLO on COCO, less tested on aerial benchmarks.
- **TPH-YOLOv5** ([Zhu et al., arXiv:2108.11539](https://arxiv.org/abs/2108.11539)) — adds transformer prediction heads for drone imagery.

### Small-object techniques relevant to aerial

- **P2 detection head** (stride-4 output) — standard addition for aerial YOLO forks. Not yet added in SkyCop; worth measuring if recall is an issue.
- **SPD-Conv** ([Sunkara & Luo, arXiv:2208.03641](https://arxiv.org/abs/2208.03641)) — space-to-depth replacement for strided convolution to preserve fine detail.
- **SAHI** ([Akyon et al., ICIP 2022, arXiv:2202.06934](https://arxiv.org/abs/2202.06934)) — slicing-aided hyper inference at test time. Standard for aerial small objects.
- **Higher input resolution** (1280 or 1536 vs 640) — roughly linear cost increase; SkyCop defaults to 640 today.

### Synthetic-to-real transfer

- **AirSim** ([Shah et al., arXiv:1705.05065](https://arxiv.org/abs/1705.05065)) is more common than CARLA for aerial simulation in the published literature.
- **Domain randomization** ([Tobin et al., arXiv:1703.06907](https://arxiv.org/abs/1703.06907)) — randomize texture/light/weather to force invariance. SkyCop exp 07 applies a light version (3 weather presets, random NPC spawns).
- Consensus finding across the sim-to-real literature: synthetic-pretrain + small real-set fine-tune beats either alone; pure synthetic training typically loses 10–20 mAP points vs real-trained baselines. **Implication for SkyCop:** CARLA-only fine-tune is defensible for the demo, but a public-deployment claim would need either real-world fine-tuning or sim-specific evaluation caveats.

---

## 2. Multi-object tracking

### Lineage

The tracking-by-detection family is mature and dominant. Key ancestors-in-use in 2025:

| Tracker | Year | Core idea | Citation |
|---|---|---|---|
| **SORT** | 2016 | Kalman filter + Hungarian IoU | [arXiv:1602.00763](https://arxiv.org/abs/1602.00763) |
| **DeepSORT** | 2017 | SORT + CNN appearance embedding | [arXiv:1703.07402](https://arxiv.org/abs/1703.07402) |
| **ByteTrack** | 2022 | Second-pass association on low-confidence boxes | [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) |
| **OC-SORT** | 2023 | Observation-centric re-update during occlusion | [arXiv:2203.14360](https://arxiv.org/abs/2203.14360) |
| **StrongSORT** | 2023 | Modernised DeepSORT (BoT backbone, EMA, AFLink, GSI) | [arXiv:2202.13514](https://arxiv.org/abs/2202.13514) |
| **BoT-SORT** | 2022 | ByteTrack + camera-motion compensation + ReID | [arXiv:2206.14651](https://arxiv.org/abs/2206.14651) |
| **UCMCTrack** | 2024 | Motion-only tracker with camera-motion compensation | [arXiv:2312.08952](https://arxiv.org/abs/2312.08952) |
| **Hybrid-SORT** | 2024 | Weak cues (height, confidence) fused with strong cues | [arXiv:2308.00783](https://arxiv.org/abs/2308.00783) |
| **MOTIP** | 2024 | Transformer ID-as-token tracking | [arXiv:2403.16848](https://arxiv.org/abs/2403.16848) |
| **GHOST** | 2023 | Simple ReID + on-the-fly domain adaptation | [arXiv:2206.04656](https://arxiv.org/abs/2206.04656) |

**Implication for SkyCop (REQUIREMENTS.md CV-06 currently specifies ByteTrack):** ByteTrack is a defensible baseline but **BoT-SORT** or **StrongSORT** are stronger fits — both add camera-motion compensation, which matters for a drone that is actively moving to keep the suspect in frame. UCMCTrack is a lighter alternative if appearance features hurt more than help at ~30 px target sizes. Worth considering a CV-06 revision after exp 10 lands.

---

## 3. Vehicle re-identification and fingerprinting

### Benchmarks

- **VeRi-776** ([Liu et al., ECCV 2016](https://github.com/JDAI-CV/VeRidataset)) — 776 vehicles, 20 ground-level cameras, urban.
- **VehicleID** ([Liu et al., CVPR 2016](https://pkuml.org/resources/pku-vehicleid.html))
- **VERI-Wild** ([Lou et al., CVPR 2019, arXiv:1902.02792](https://arxiv.org/abs/1902.02792)) — 40k IDs, unconstrained.
- **VRAI** ([Wang et al., ACM MM 2019, arXiv:1904.01400](https://arxiv.org/abs/1904.01400)) — aerial-specific vehicle re-ID.

Aerial ReID is notably harder than ground-level: loss of side-profile cues, heavy emphasis on roof/hood, viewpoint variation as the drone follows. **Implication for SkyCop:** the HSV + geometry fingerprint originally specified in CV-11..16 is defensible as a *starting point* (cheap, interpretable, no training data needed), but state-of-the-art on VRAI uses learned embeddings — worth adding a learned-embedding branch in exp 11+.

### Metric learning

Standard losses:
- **Triplet / FaceNet** ([Schroff et al., arXiv:1503.03832](https://arxiv.org/abs/1503.03832))
- **ArcFace** ([Deng et al., arXiv:1801.07698](https://arxiv.org/abs/1801.07698))
- **Circle loss** ([Sun et al., CVPR 2020, arXiv:2002.10857](https://arxiv.org/abs/2002.10857))

### Lightweight ReID backbones

- **OSNet** ([Zhou et al., ICCV 2019, arXiv:1905.00953](https://arxiv.org/abs/1905.00953)) — omni-scale network, often bundled with BoT-SORT/StrongSORT as the appearance embedding.

### Vision-language ReID — notable for SkyCop

**CLIP-ReID** ([Li et al., AAAI 2023, arXiv:2211.13977](https://arxiv.org/abs/2211.13977)) uses vision-language pretraining so that a text description can be matched against visual features for re-identification. This is a direct fit for SkyCop's "dispatch alert contains vehicle description → match against detected candidates" flow (design D-09). Noted as a candidate for the fingerprint module in exp 11+ — if it works end-to-end, the project could skip hand-crafted HSV histograms entirely and use CLIP-ReID embedding similarity as the suspect-match score.

### Occlusion recovery

- **Kalman / constant-velocity motion prediction** — reliable for ~1–2 s gaps, degrades sharply beyond.
- **Appearance-memory banks** (StrongSORT EMA, [arXiv:2202.13514](https://arxiv.org/abs/2202.13514)).
- **Global batch association** — AFLink in StrongSORT; offline graph methods like **GHOST** ([arXiv:2206.04656](https://arxiv.org/abs/2206.04656)).
- **Trajectory inpainting** (GSI in StrongSORT).

No published number I trust for 5–30 s occlusion recovery on road networks — SkyCop's occlusion targets (CV-18..25) need empirical measurement, not a citation-backed claim. Requirements target of 85% re-acquisition within 60 s (CV-25) is plausible but un-validated.

---

## 4. Target acquisition and active search

This is the sub-problem REQUIREMENTS.md hand-waves (FR-03..05) — the one where the drone turns a dispatch alert into a lock. The literature is substantial.

### Road-constrained tracking

The foundational observation: ground vehicles are confined to a road graph. Classical treatment:

- **Variable Structure IMM on roads** ([Kirubarajan et al., IEEE TAES 2000](https://ieeexplore.ieee.org/document/869492)) — reference formulation, still widely cited.
- **Rao-Blackwellised Particle Filter** ([Doucet et al., arXiv:1301.3853](https://arxiv.org/abs/1301.3853)) — factors discrete road-segment state (sampled) and continuous along-road position/velocity (Kalman), dramatic particle-count reduction.
- **Road-map assisted tracking** ([Ulmke & Koch, IEEE TAES 2006](https://ieeexplore.ieee.org/document/4285403)) — applies RBPF on digital road maps.
- **HMM map matching** ([Newson & Krumm, SIGSPATIAL 2009](https://dl.acm.org/doi/10.1145/1653771.1653818)) — road segments as HMM states; often repurposed for tracking under uncertainty.

**Implication for SkyCop:** the acquisition algorithm I sketched in the design.md D-10 stub (belief map over waypoint edges with forward propagation along the graph) is a direct application of HMM map matching / RBPF without rolling our own theory. Cite these when writing the formal algorithm.

### Active / occlusion-aware search

- **POMDP UAV search with negative information** ([Chung & Burdick, IEEE T-RO 2012](https://ieeexplore.ieee.org/document/6051437)) — foundational treatment. Null observations reduce posterior belief in searched cells proportional to sensor TPR — exactly the idea SkyCop needs.
- **Mutual-information sensor planning** ([Hoffmann & Tomlin, IEEE TAC 2010](https://ieeexplore.ieee.org/document/5406129)) — information-gain drone motion planning.
- **Active target localization** ([Charrow et al., RSS 2014](http://www.roboticsproceedings.org/rss10/p55.html)).
- **Next-best-view planning** ([Bircher et al., ICRA 2016](https://ieeexplore.ieee.org/document/7487281)) — receding-horizon visibility-aware coverage, extended in follow-ups for occlusion likelihoods.
- **Occlusion-aware ground target search** ([arXiv:2511.07822](https://arxiv.org/abs/2511.07822), Nov 2025) — recent work specifically on UAV search under occlusions in urban environments. *I have not verified this paper's specific claims* — listed as a pointer for authors to read.

### Description-conditioned detection

The academic lineage that supports "dispatch gives a textual description → find matching vehicle":

- **OWL-ViT** ([Minderer et al., ECCV 2022, arXiv:2205.06230](https://arxiv.org/abs/2205.06230)) — open-vocabulary detection, can detect "red sedan" without retraining.
- **GroundingDINO** ([Liu et al., arXiv:2303.05499](https://arxiv.org/abs/2303.05499)) — text-conditioned object detection.
- **CLIP-ReID** ([arXiv:2211.13977](https://arxiv.org/abs/2211.13977)) — see §3.

This connects directly to D-09's decision to move class from the detector to the fingerprint/matcher: text-to-visual matching is the state-of-the-art approach and there's published infrastructure to borrow from.

### Visual servoing / gimbal tracking

- **Image-Based Visual Servoing** ([Chaumette & Hutchinson, IEEE RAM 2006/2007](https://ieeexplore.ieee.org/document/4015566)) — canonical reference for bbox-center-error → gimbal rate control via interaction matrix. Supports CV-15..16 in requirements.
- **Proportional navigation / pure pursuit / lead pursuit** — missile guidance lineage (Shneydor 1998, out-of-print); proportional navigation dominates UAV tracking applications.

bbox-size-based altitude control (loose outer loop on apparent size, fast inner loop on pixel center error) is effectively folklore — I could not find a canonical academic citation. SkyCop's AdaptiveAltitudeController (exp 04) uses a different scheme entirely (raycast-based urban detection), which is arguably more defensible than the bbox-size method anyway.

---

## 5. Commercial and government landscape

### Drone-as-First-Responder (DFR): Skydio × Axon

The June 20, 2024 partnership is the closest shipping analogue to SkyCop and the most important reference for positioning.

End-to-end flow per [Axon's press release](https://www.prnewswire.com/news-releases/axon-and-skydio-partner-to-deliver-scalable-drone-offering-for-public-safety-including-drone-as-first-responder-solution-302177624.html), [Skydio's DFR Command product page](https://www.skydio.com/software/dfr-command), and [Skydio's how-DFR-works page](https://www.skydio.com/solutions/dfr/how-dfr-works):

1. CAD / 911 / ALPR / ShotSpotter event triggers auto-launch of the nearest docked drone within ~20 s.
2. Drone flies autonomously to GPS coordinates.
3. **A remote human pilot** (RTCC dispatcher or operator) watches the live feed via browser-based DFR Command.
4. "Skydio Shadow" mode does autonomous visual follow of a subject **once the operator clicks on it in the video**.

**Target acquisition and identification is never autonomous** — the human-in-the-loop selects what to follow. No shipping product ingests "red sedan, plate ABC" and finds the matching vehicle from the air. This is SkyCop's novelty axis: not the pieces (detection, tracking, gimbal) but the closed-loop automation of the target-ID step.

The Orlando PD program ([DroneXL, Feb 2026](https://dronexl.co/2026/02/25/orlando-skydio-drone-program/)) reported trial drones beating ground officers to the scene one-third of the time — validates the DFR value proposition but doesn't address autonomous pursuit.

### Tactical and military

- **BRINC LEMUR 2** ([product page](https://brincdrones.com/lemur-2/)) — indoor SWAT/tactical drone (LiDAR mapping, glass breaker, two-way audio for negotiation). Explicitly not a pursuit platform.
- **Teal Drones Black Widow** (Red Cat, NDAA-compliant, [press release](https://ir.redcatholdings.com/news-events/press-releases/detail/191/red-cats-teal-drones-black-widow-system-approved-for-nato-nspa-catalogue)) — short-range ISR quad, human-piloted EO/IR reconnaissance.
- **Anduril Lattice + Shield AI Hivemind** ([Anduril news Feb 2026](https://www.anduril.com/news/yfq-44a-flies-with-mission-autonomy-software-from-anduril-and-shield-ai), [Air & Space Forces Magazine](https://www.airandspaceforces.com/anduril-cca-switches-ai-pilots-midflight/)) — flew together on the YFQ-44A Fury CCA with mid-flight autonomy-stack handoff. Public scope covers GPS/comms-denied flight, wingman/swarm ops, sensor fusion. **Whether target-from-description workflows exist is not publicly detailed and I make no claims either way.**

### Traditional police aviation

Manned helicopter + gyro-stabilized gimbal (Wescam MX, FLIR Star SAFIRE, Trakka TASE) with a Tactical Flight Officer manually slewing the gimbal during pursuit ([FLIR](https://www.flir.com/discover/rd-science/thermal-imaging-identification-protects-police-and-the-public-during-high-speed-pursuits/), [AirMed&Rescue](https://www.airmedandrescue.com/latest/long-read/getting-clear-picture-procuring-right-surveillance-equipment-police-aviation)). No autonomy in target selection or camera tracking. SkyCop's "AI-powered vs human operator" framing (requirements §6.1–6.4) maps directly to "autonomous vs TFO-operated" as the point of comparison.

### Sim-based analogues

Most CARLA research targets ground autonomy. A "CARLA Drone" monocular 3D detection benchmark exists ([Springer 2025](https://link.springer.com/chapter/10.1007/978-3-031-85187-2_9)) but is detection-only, not pursuit. [AirSim](https://microsoft.github.io/AirSim/) and [Air Learning](https://link.springer.com/article/10.1007/s10994-021-06006-6) support UAV RL navigation experiments but published scenarios are avoidance / inspection / racing — not suspect pursuit with fingerprint re-ID. I did not find an open-source analogue to SkyCop's scope during the survey.

---

## 6. Positioning of SkyCop

| Axis | SkyCop | Closest commercial | Gap |
|---|---|---|---|
| Hardware / flight control | **Simulated** — out of scope | Skydio X10 onboard autonomy | N/A — we don't compete on flight |
| BVLOS regulation / fleet / dispatch | **Not modelled** | Skydio × Axon full DFR stack | N/A — deliberately out of scope |
| Autonomous target acquisition from description | **Core focus** — closed loop from dispatch text to visual lock | Not shipped; human-in-the-loop everywhere in DFR | **Novel axis** |
| Soft-score fingerprint matching | **Core focus** — multi-attribute, class from dispatch not detector | ReID within tracker is internal engineering, not a product surface | Reimplementation of a research component, but applied in a deliberate mission-level loop |
| Human-vs-autonomous head-to-head | **Core focus** — scored debrief | Not shipped | **Novel presentation axis** |

**Fair framing for the portfolio:** SkyCop is a research demonstration exploring a capability that commercial vendors have publicly avoided (deliberate human-in-the-loop policies). It is not a reimplementation of an existing product. It is *deliberately simpler* than any shipping DFR or ISR stack on every axis except target-acquisition autonomy, where the published commercial scope is empty.

This framing is defensible with the citations above. Do not overclaim ("SkyCop does what Skydio can't") — Skydio's design deliberately keeps humans in the loop on target selection. SkyCop explores what *would* an autonomous version look like.

---

## 7. Requirements audit — citation or flag per item

Items grounded in cited work are marked ✓. Items flagged as project-specific speculation (no cited backing) are marked ⚠ and should be either cited, revised, or explicitly acknowledged as speculative.

| ID | Status | Citation / Note |
|---|---|---|
| CV-01 detector choice | ✓ | YOLOv8s: ultralytics docs; single-class: D-09 + this survey §1 |
| CV-02 FPS @ fp16 | ✓ | Matches published YOLOv8s throughput on consumer GPUs |
| CV-03 85% mAP target | ⚠ | No cited basis — 85% is aspirational. VisDrone SOTA is ~55%; achievable via in-domain CARLA training, still unverified for SkyCop |
| CV-04 instance-seg auto-labels | ✓ | Standard CARLA synthetic-data technique (AirSim/domain randomization lineage, §1) |
| CV-05 640×640 input | ✓ | YOLO default; higher res is a known-good alternative (SAHI, P2 head) |
| CV-06 ByteTrack | ⚠ | BoT-SORT / StrongSORT are stronger for moving-camera scenarios; revise after exp 10 |
| CV-07..10 tracking properties | ✓ | Standard MOT claims, grounded in §2 |
| CV-11..16 fingerprint attributes | Mixed | HSV histograms, Kalman velocity: standard. Learned-embedding alternative (CLIP-ReID, OSNet) not yet in requirements; should be added |
| CV-17..25 occlusion recovery | ⚠ | 85% recovery within 60s target has no cited basis; needs empirical measurement |
| CV-26..32 parking re-ID | Mixed | Approach-trajectory tracking is defensible engineering; confidence thresholds (0.75, 0.85, 0.60) are un-cited speculation |
| CV-33..35 speed estimation | ✓ | Standard computer-vision approach; ±15 km/h target is a reasonable demo-level claim |
| FR-03..05 dispatch flow | ⚠ | Acquisition algorithm not specified; survey §4 + design D-10 stub supply the reference. Requirements need to either cite or defer to a §4.8 "Target Acquisition" section |
| FR-07..11 suspect FSM | ✓ | Scripted FSM, not a research claim; explicitly non-reactive per requirements §3.3 |
| FR-12..15 alerts | ✓ | Not research claims; implementation contract |
| SIM-01..10 CARLA setup | ✓ | Grounded in docs/carla_caveats.md |
| SIM-11..14 adaptive altitude | ⚠ | No single cited source for the scheme; combination of raycast + clamp is defensible engineering |
| SIM-15..18 PID | ✓ | IBVS lineage (§4) |
| DB-01..10 dashboard | ✓ | Implementation contract, no research claims |
| NFR-01..13 non-functional | ✓ | Standard engineering targets |

**Summary of citation gaps:** five flags, none fatal. The biggest are CV-03 (mAP target number, to be revalidated after exp 08) and CV-06 (tracker choice, to be revalidated after the MOT experiment). Acquisition (FR-03..05) needs a formal §4.8 section referencing the literature in §4 above, but that's a design expansion rather than a contradiction.

---

## 8. Follow-up actions (for future PRs, not this one)

1. **REQUIREMENTS.md inline citations** — add `[^cv01]` style footnote markers to items marked ✓ in §7; add `[^speculative]` markers to items marked ⚠. *Handled in this PR for the load-bearing items.*
2. **Add §4.8 Target Acquisition** to REQUIREMENTS.md citing §4 above. *Separate PR after design D-10 lands.*
3. **Revisit CV-06 tracker choice** after exp 10 (ByteTrack integration) — if camera motion is the dominant failure mode, swap to BoT-SORT. Document the empirical finding.
4. **Revisit CV-03 mAP target** after exp 08 (fine-tune) — ground the 85% number in the measured CARLA-in-domain ceiling rather than aspiration.
5. **Consider CLIP-ReID or OSNet for fingerprint** once exp 10 is stable. If the embedding-based match score outperforms hand-crafted HSV + geometry, update CV-11..16.

---

## References (aggregated)

Alphabetised list of all citations used above. Verify before quoting specific numbers.

**Aerial detection:**
- VisDrone: [arXiv:2001.06303](https://arxiv.org/abs/2001.06303)
- UAVDT: [arXiv:1804.00518](https://arxiv.org/abs/1804.00518)
- DOTA: [arXiv:1711.10398](https://arxiv.org/abs/1711.10398)
- YOLOv10: [arXiv:2405.14458](https://arxiv.org/abs/2405.14458)
- RT-DETR: [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- TPH-YOLOv5: [arXiv:2108.11539](https://arxiv.org/abs/2108.11539)
- SPD-Conv: [arXiv:2208.03641](https://arxiv.org/abs/2208.03641)
- SAHI: [arXiv:2202.06934](https://arxiv.org/abs/2202.06934)
- AirSim: [arXiv:1705.05065](https://arxiv.org/abs/1705.05065)
- Domain randomization: [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)

**MOT and ReID:**
- SORT: [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)
- DeepSORT: [arXiv:1703.07402](https://arxiv.org/abs/1703.07402)
- ByteTrack: [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- OC-SORT: [arXiv:2203.14360](https://arxiv.org/abs/2203.14360)
- StrongSORT: [arXiv:2202.13514](https://arxiv.org/abs/2202.13514)
- BoT-SORT: [arXiv:2206.14651](https://arxiv.org/abs/2206.14651)
- UCMCTrack: [arXiv:2312.08952](https://arxiv.org/abs/2312.08952)
- Hybrid-SORT: [arXiv:2308.00783](https://arxiv.org/abs/2308.00783)
- MOTIP: [arXiv:2403.16848](https://arxiv.org/abs/2403.16848)
- GHOST: [arXiv:2206.04656](https://arxiv.org/abs/2206.04656)
- VERI-Wild: [arXiv:1902.02792](https://arxiv.org/abs/1902.02792)
- VRAI: [arXiv:1904.01400](https://arxiv.org/abs/1904.01400)
- OSNet: [arXiv:1905.00953](https://arxiv.org/abs/1905.00953)
- FaceNet/Triplet: [arXiv:1503.03832](https://arxiv.org/abs/1503.03832)
- ArcFace: [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
- Circle Loss: [arXiv:2002.10857](https://arxiv.org/abs/2002.10857)
- CLIP-ReID: [arXiv:2211.13977](https://arxiv.org/abs/2211.13977)

**Acquisition and active search:**
- VS-IMM on roads: [Kirubarajan 2000](https://ieeexplore.ieee.org/document/869492)
- RBPF: [arXiv:1301.3853](https://arxiv.org/abs/1301.3853)
- Road-map tracking: [Ulmke & Koch 2006](https://ieeexplore.ieee.org/document/4285403)
- HMM map matching: [Newson & Krumm 2009](https://dl.acm.org/doi/10.1145/1653771.1653818)
- POMDP search: [Chung & Burdick 2012](https://ieeexplore.ieee.org/document/6051437)
- Mutual-information sensor planning: [Hoffmann & Tomlin 2010](https://ieeexplore.ieee.org/document/5406129)
- Active target localisation: [Charrow et al. RSS 2014](http://www.roboticsproceedings.org/rss10/p55.html)
- Next-best-view: [Bircher et al. ICRA 2016](https://ieeexplore.ieee.org/document/7487281)
- Occlusion-aware search: [arXiv:2511.07822](https://arxiv.org/abs/2511.07822) (unverified claims)
- OWL-ViT: [arXiv:2205.06230](https://arxiv.org/abs/2205.06230)
- GroundingDINO: [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- IBVS: [Chaumette & Hutchinson 2006](https://ieeexplore.ieee.org/document/4015566)

**Commercial and government:**
- [Axon × Skydio partnership, June 2024](https://www.prnewswire.com/news-releases/axon-and-skydio-partner-to-deliver-scalable-drone-offering-for-public-safety-including-drone-as-first-responder-solution-302177624.html)
- [Skydio DFR Command](https://www.skydio.com/software/dfr-command)
- [Skydio: how DFR works](https://www.skydio.com/solutions/dfr/how-dfr-works)
- [Skydio Autonomy overview](https://www.skydio.com/skydio-autonomy)
- [Orlando DFR adoption, DroneXL Feb 2026](https://dronexl.co/2026/02/25/orlando-skydio-drone-program/)
- [BRINC LEMUR 2](https://brincdrones.com/lemur-2/)
- [Teal Drones Black Widow, Red Cat Sep 2025](https://ir.redcatholdings.com/news-events/press-releases/detail/191/red-cats-teal-drones-black-widow-system-approved-for-nato-nspa-catalogue)
- [Anduril × Shield AI, Feb 2026](https://www.anduril.com/news/yfq-44a-flies-with-mission-autonomy-software-from-anduril-and-shield-ai)
- [Air & Space Forces Magazine on CCA handoff](https://www.airandspaceforces.com/anduril-cca-switches-ai-pilots-midflight/)
- [FLIR on high-speed pursuits](https://www.flir.com/discover/rd-science/thermal-imaging-identification-protects-police-and-the-public-during-high-speed-pursuits/)
- [AirMed&Rescue on surveillance equipment procurement](https://www.airmedandrescue.com/latest/long-read/getting-clear-picture-procuring-right-surveillance-equipment-police-aviation)
- [CARLA Drone 3D detection, Springer 2025](https://link.springer.com/chapter/10.1007/978-3-031-85187-2_9)
- [AirSim](https://microsoft.github.io/AirSim/)
- [Air Learning](https://link.springer.com/article/10.1007/s10994-021-06006-6)

---

*This survey is maintained in `docs/literature_survey.md`. When adding new experiments or design decisions, add the relevant citation to §7 and update §8 follow-ups. Do not delete unverified entries — mark them explicitly so future authors know to check.*
