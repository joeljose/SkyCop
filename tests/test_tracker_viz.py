"""Unit tests for skycop.cv.tracker_viz — overlay + video writer (pure, no CARLA, no YOLO)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from skycop.cv.tracker_viz import render_overlay, write_video
from skycop.cv.tracking_eval import GTBox, TrackerBox


def _blank(h: int = 64, w: int = 128) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_render_overlay_preserves_shape_and_dtype():
    img = _blank()
    out = render_overlay(
        image=img,
        gt_boxes=[GTBox(actor_id=7, x1=10, y1=10, x2=40, y2=30)],
        tracker_boxes=[TrackerBox(track_id=3, x1=12, y1=12, x2=38, y2=28)],
        suspect_actor_id=7,
        locked_track_id=3,
        frame_idx=0,
    )
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_render_overlay_does_not_mutate_input():
    img = _blank()
    original = img.copy()
    render_overlay(
        image=img,
        gt_boxes=[GTBox(actor_id=1, x1=5, y1=5, x2=20, y2=20)],
        tracker_boxes=[],
        suspect_actor_id=1,
        locked_track_id=None,
        frame_idx=0,
    )
    assert np.array_equal(img, original), "render_overlay must not mutate the input image"


def test_render_overlay_draws_suspect_and_tracker():
    # An empty frame should have pixels drawn after overlay runs — proves *some* drawing happened.
    img = _blank()
    out = render_overlay(
        image=img,
        gt_boxes=[GTBox(actor_id=7, x1=10, y1=10, x2=40, y2=30)],
        tracker_boxes=[TrackerBox(track_id=3, x1=12, y1=12, x2=38, y2=28)],
        suspect_actor_id=7,
        locked_track_id=3,
        frame_idx=5,
    )
    assert out.sum() > 0, "expected non-zero pixels after drawing"


def test_render_overlay_switch_banner_adds_pixels():
    img = _blank()
    base = render_overlay(
        image=img, gt_boxes=[], tracker_boxes=[],
        suspect_actor_id=1, locked_track_id=1, frame_idx=0,
    )
    with_banner = render_overlay(
        image=img, gt_boxes=[], tracker_boxes=[],
        suspect_actor_id=1, locked_track_id=1, frame_idx=0,
        switch_event="ID SWITCH — locked=t1 matched=t9",
    )
    # Banner spans the top rows; pixel sum strictly grows when banner is rendered.
    assert with_banner.sum() > base.sum()


def test_render_overlay_handles_none_track_id():
    # unconfirmed tracker box (track_id=None) should render without crashing
    img = _blank()
    out = render_overlay(
        image=img,
        gt_boxes=[GTBox(actor_id=1, x1=5, y1=5, x2=20, y2=20)],
        tracker_boxes=[TrackerBox(track_id=None, x1=6, y1=6, x2=18, y2=18)],
        suspect_actor_id=1,
        locked_track_id=1,
        frame_idx=0,
    )
    assert out.shape == img.shape


def test_write_video_rejects_empty():
    with pytest.raises(ValueError):
        write_video([], fps=10, out_path=Path("/tmp/skycop_viz_empty.mp4"))


def test_write_video_rejects_shape_mismatch(tmp_path: Path):
    frames = [_blank(64, 128), _blank(32, 64)]
    with pytest.raises(ValueError):
        write_video(frames, fps=10, out_path=tmp_path / "bad.mp4")


def test_write_video_produces_readable_file(tmp_path: Path):
    # 30 frames of solid colour; write + read back via cv2.VideoCapture.
    frames = [np.full((48, 96, 3), 128, dtype=np.uint8) for _ in range(30)]
    out_path = tmp_path / "ok.mp4"
    result = write_video(frames, fps=10, out_path=out_path)

    assert result == out_path
    assert out_path.exists() and out_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(out_path))
    try:
        assert cap.isOpened()
        n_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert n_read == 30, f"expected 30 frames, got {n_read}"
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 96
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    finally:
        cap.release()
