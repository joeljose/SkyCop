"""Tests for skycop.config — OmegaConf-based merger."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from skycop import config as skycop_config


@pytest.fixture
def tmp_configs(tmp_path, monkeypatch):
    """Point CONFIGS_ROOT at an isolated tmp dir with two small yaml files."""
    (tmp_path / "base.yaml").write_text(
        "seed: 1\nscene:\n  npc_count: 10\ncamera:\n  fov: 90\n"
    )
    (tmp_path / "overlay.yaml").write_text(
        "scene:\n  npc_count: 50\ndetector:\n  conf_threshold: 0.25\n"
    )
    monkeypatch.setattr(skycop_config, "CONFIGS_ROOT", tmp_path)
    return tmp_path


def test_single_config_loads(tmp_configs):
    cfg = skycop_config.load("base")
    assert cfg.seed == 1
    assert cfg.scene.npc_count == 10


def test_later_file_overrides_earlier(tmp_configs):
    cfg = skycop_config.load("base", "overlay")
    assert cfg.scene.npc_count == 50             # overlay wins
    assert cfg.camera.fov == 90                  # base survives
    assert cfg.detector.conf_threshold == 0.25   # overlay-only key present


def test_dotlist_overrides_win_over_files(tmp_configs):
    cfg = skycop_config.load("base", "overlay", overrides=["seed=99", "camera.fov=53"])
    assert cfg.seed == 99
    assert cfg.camera.fov == 53
    assert cfg.scene.npc_count == 50


def test_missing_config_raises(tmp_configs):
    with pytest.raises(FileNotFoundError):
        skycop_config.load("nonexistent")


def test_empty_load_returns_empty_dict(tmp_configs):
    cfg = skycop_config.load()
    assert OmegaConf.to_container(cfg) == {}


def test_real_default_config_loads():
    """Integration — the real configs/default.yaml at /app/configs loads cleanly."""
    real_root = Path("/app/configs")
    if not real_root.exists():
        pytest.skip("not inside container or configs missing")
    from skycop.config import load as real_load
    cfg = real_load("default", "detector")
    assert cfg.seed == 42
    assert cfg.detector.conf_threshold == 0.25
