"""CARLA vehicle blueprint → SkyCop class mapping.

Four target classes:
  car, van, truck, bus

Two-wheelers (motorcycles, bicycles) are explicitly dropped from the
taxonomy. Reasons:

- CARLA's Traffic Manager does not autopilot 2-wheelers, so our pursuit
  scenes never spawn any — no training data, no evaluation signal.
- At 40m operational altitude, bicycles are ~8×24 px anyway, below YOLO's
  practical detection floor (~20×20 px).

If we later gain a way to populate scenes with motorcycles (manual
driving, parked-only props, or a CARLA TM update), this module is the
one place the class would be reinstated: add "motorcycle" to CLASS_NAMES
and return it from `classify_blueprint` for 2-wheelers that aren't
bicycle-branded.

The mapping is pragmatic and substring-based. When in doubt, falls back
to "car" for 4-wheeled vehicles. CARLA does not expose a class taxonomy
on blueprints themselves, so this lookup is maintained manually.
"""

from __future__ import annotations

CLASS_NAMES: list[str] = ["car", "van", "truck", "bus"]
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Substrings in blueprint type_id that mark a bicycle — we drop these from the dataset.
_BICYCLE_SUBSTRINGS: tuple[str, ...] = ("crossbike", "century", "omafiets")

# Substrings that identify specific 4-wheel classes. Checked in order — first hit wins.
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
    """Map a CARLA vehicle blueprint to one of the 4 SkyCop classes.

    Returns the class name on success, or None if the blueprint should be
    excluded from labels (all 2-wheelers — bicycles and motorcycles).
    """
    if number_of_wheels == 2:
        return None

    tid = type_id.lower()
    for pattern, cls in _FOUR_WHEEL_PATTERNS:
        if pattern in tid:
            return cls

    return "car"


def class_index(name: str) -> int:
    """Return the YOLO class index for a class name. Raises KeyError if unknown."""
    return CLASS_INDEX[name]
