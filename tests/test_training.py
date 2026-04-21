"""Tests for skycop.cv.training — dataset YAML construction (no GPU/ultralytics)."""

from pathlib import Path

from skycop.cv.training import TrainingConfig, best_weights_path, build_dataset_yaml


def _seed_run_dir(root: Path, name: str, n_images: int) -> None:
    (root / name / "images").mkdir(parents=True, exist_ok=True)
    (root / name / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        (root / name / "images" / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (root / name / "labels" / f"frame_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_build_dataset_yaml_writes_absolute_paths_by_split(tmp_path: Path):
    # Three runs; two in train, one in val.
    _seed_run_dir(tmp_path, "run_00_seed100_ClearNoon", 3)
    _seed_run_dir(tmp_path, "run_01_seed101_ClearNoon", 2)
    _seed_run_dir(tmp_path, "run_03_seed103_ClearNoon", 2)

    yaml_path = tmp_path / "dataset.yaml"
    build_dataset_yaml(
        dataset_root=tmp_path,
        train_runs=["run_00_seed100_ClearNoon", "run_01_seed101_ClearNoon"],
        val_runs=["run_03_seed103_ClearNoon"],
        out_path=yaml_path,
    )

    assert yaml_path.exists()
    contents = yaml_path.read_text()
    assert "nc: 1" in contents
    assert "names: ['vehicle']" in contents
    assert "train:" in contents
    assert "val:" in contents

    train_txt = tmp_path / "train.txt"
    val_txt = tmp_path / "val.txt"
    assert train_txt.exists() and val_txt.exists()

    train_lines = [line for line in train_txt.read_text().splitlines() if line]
    val_lines = [line for line in val_txt.read_text().splitlines() if line]
    assert len(train_lines) == 5   # 3 + 2 images across two train runs
    assert len(val_lines) == 2     # one val run with 2 images

    for p in train_lines + val_lines:
        assert Path(p).is_absolute()
        assert Path(p).exists()

    # Sanity: no cross-split contamination.
    val_run_image = next(
        iter((tmp_path / "run_03_seed103_ClearNoon" / "images").glob("*.jpg"))
    )
    assert str(val_run_image.resolve()) not in train_lines


def test_best_weights_path_follows_ultralytics_convention():
    cfg = TrainingConfig(project="output/weights", name="run_v1")
    assert best_weights_path(cfg) == Path("output/weights/run_v1/weights/best.pt")


def test_training_config_defaults_match_yaml():
    cfg = TrainingConfig()
    # Guardrails: aerial-specific augmentation defaults must stay off.
    assert cfg.mosaic == 0.0
    assert cfg.degrees == 0.0
    # Determinism on by default.
    assert cfg.deterministic is True
    assert cfg.seed == 42
    # fp16 training by default.
    assert cfg.half is True
    # Early stopping configured.
    assert cfg.patience > 0
