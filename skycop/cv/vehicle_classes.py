"""CARLA vehicle blueprint → SkyCop class mapping.

Two taxonomies live here, serving different layers of the pipeline:

**Detector taxonomy** — a single class, ``vehicle``. Used for YOLOv8 training
and inference. Rationale in ``docs/design.md`` D-09: dispatch already supplies
the suspect's class to the mission; the detector's job is localization, not
classification. Fine-grained classification from 15–40m aerial at −75° pitch
is unreliable (structural imbalance + single-model-per-rare-class confound in
CARLA) and adds noise without informational value for the mission.

**Fingerprint taxonomy** — ``car / van / truck / bus``, preserved for the
CV-11..16 fingerprint module. Sourced from CARLA blueprint metadata at
data-collection time (simulation-only shortcut) or from bbox geometry at
inference time (future work), not from the detector's output.

Two-wheelers (motorcycles + bicycles) are excluded from both taxonomies:
CARLA's Traffic Manager cannot autopilot them, so the training distribution
contains none, and at 40m altitude bicycles would be below YOLO's practical
detection floor.
"""

from __future__ import annotations

# ── Detector taxonomy ─────────────────────────────────────────────────────

CLASS_NAMES: list[str] = ["vehicle"]
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── Fingerprint taxonomy (attribute for CV-11..16, NOT detector output) ──

FINGERPRINT_CLASSES: list[str] = ["car", "van", "truck", "bus"]
FINGERPRINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(FINGERPRINT_CLASSES)}


# Substrings in blueprint type_id that mark a bicycle — we drop these from the dataset.
_BICYCLE_SUBSTRINGS: tuple[str, ...] = ("crossbike", "century", "omafiets")

# Substrings that identify specific 4-wheel categories — used only by the
# fingerprint taxonomy. Checked in order — first hit wins.
_FOUR_WHEEL_PATTERNS: list[tuple[str, str]] = [
    ("firetruck", "truck"),
    ("carlacola", "truck"),
    ("cybertruck", "truck"),
    ("ambulance", "van"),
    ("sprinter", "van"),
    (".t2", "van"),
    ("fusorosa", "bus"),
]


def classify_blueprint(type_id: str, number_of_wheels: int) -> str | None:
    """Return fine-grained fingerprint class for a CARLA vehicle blueprint.

    Returns one of ``FINGERPRINT_CLASSES`` on success, or ``None`` if the
    blueprint is excluded from the pipeline entirely (all 2-wheelers).

    **Not** used by the detector — it's for the fingerprint module. The
    detector uses ``detector_class_for`` instead.
    """
    if number_of_wheels == 2:
        return None

    tid = type_id.lower()
    for pattern, cls in _FOUR_WHEEL_PATTERNS:
        if pattern in tid:
            return cls

    return "car"


def detector_class_for(type_id: str, number_of_wheels: int) -> int | None:
    """Return the SkyCop detector class index for a CARLA vehicle blueprint.

    All vehicles the detector cares about collapse to a single class (index 0,
    ``vehicle``). 2-wheelers return ``None`` so they're dropped at label-
    extraction time.
    """
    if number_of_wheels == 2:
        return None
    # Any other vehicle is a detection target.
    return CLASS_INDEX["vehicle"]


def class_index(name: str) -> int:
    """Return the detector class index for a class name. Raises KeyError if unknown."""
    return CLASS_INDEX[name]
