"""OmegaConf-based config loader.

Scripts compose a config by merging files from `configs/` plus dot-list overrides:

    from skycop.config import load
    cfg = load("default", "altitude", overrides=["seed=7", "camera.fov=53"])
    print(cfg.altitude.open_target_m)

This is a deliberate downgrade from full Hydra — no CLI magic, no cwd changes,
no config groups. We just merge a handful of yaml files in order. Good enough
for a demo with ~6 canonical runs.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIGS_ROOT = Path("/app/configs")


def load(*names: str, overrides: list[str] | None = None) -> DictConfig:
    """Merge `configs/<name>.yaml` for each name, then apply dot-list overrides.

    Later configs win on key collision; overrides win over everything.
    """
    parts = []
    for name in names:
        path = CONFIGS_ROOT / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        parts.append(OmegaConf.load(path))
    if overrides:
        parts.append(OmegaConf.from_dotlist(list(overrides)))
    merged = OmegaConf.merge(*parts) if parts else OmegaConf.create({})
    assert isinstance(merged, DictConfig)
    return merged
