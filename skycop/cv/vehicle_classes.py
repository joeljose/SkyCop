"""CARLA vehicle blueprint → SkyCop class mapping.

Five target classes (see grill Q2 for rationale):
  car, van, truck, bus, motorcycle

Bicycles are dropped — at 40m altitude they're ~8×24 px, below YOLO's
practical detection floor (~20×20 px).

The mapping is pragmatic and substring-based. When in doubt, falls back
to "car" for 4-wheeled vehicles. CARLA does not expose a class taxonomy
on blueprints themselves, so this lookup is maintained manually.
"""

from __future__ import annotations

CLASS_NAMES: list[str] = ["car", "van", "truck", "bus", "motorcycle"]
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
    """Map a CARLA vehicle blueprint to one of the 5 SkyCop classes.

    Returns the class name on success, or None if the blueprint should be
    excluded from labels (bicycles, non-vehicles).
    """
    tid = type_id.lower()

    if number_of_wheels == 2:
        if any(b in tid for b in _BICYCLE_SUBSTRINGS):
            return None
        return "motorcycle"

    for pattern, cls in _FOUR_WHEEL_PATTERNS:
        if pattern in tid:
            return cls

    return "car"


def class_index(name: str) -> int:
    """Return the YOLO class index for a class name. Raises KeyError if unknown."""
    return CLASS_INDEX[name]
