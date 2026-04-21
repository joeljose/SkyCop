"""Unit tests for skycop.cv.fingerprint — pure, no CARLA, no YOLO."""

import numpy as np
import pytest

from skycop.cv.fingerprint import Fingerprint, extract, score


def _solid(bgr: tuple[int, int, int], h: int = 48, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = bgr[0]
    img[:, :, 1] = bgr[1]
    img[:, :, 2] = bgr[2]
    return img


def test_extract_produces_valid_fingerprint_with_default_bins():
    fp = extract(_solid((20, 50, 200)), bbox=(0, 0, 64, 48))
    assert fp.is_valid()
    assert fp.hsv_hist.shape == (8 ** 3,)
    assert abs(float(fp.hsv_hist.sum()) - 1.0) < 1e-5


def test_extract_empty_bbox_returns_invalid_fingerprint():
    fp = extract(_solid((0, 0, 0)), bbox=(10, 10, 10, 10))  # zero-area
    assert not fp.is_valid()
    assert fp.hsv_hist.shape == (8 ** 3,)


def test_extract_offscreen_bbox_returns_invalid_fingerprint():
    fp = extract(_solid((0, 0, 0)), bbox=(9999, 9999, 10000, 10000))
    assert not fp.is_valid()


def test_extract_clamps_partially_offscreen_bbox():
    # bbox extends past the frame but has some overlap → must still produce a valid fp
    fp = extract(_solid((100, 100, 100)), bbox=(-20, -20, 20, 20))
    assert fp.is_valid()


def test_score_identical_solid_returns_one():
    img = _solid((30, 60, 180))
    fp = extract(img, bbox=(0, 0, 64, 48))
    assert abs(score(fp, fp) - 1.0) < 1e-5


def test_score_symmetric():
    a = extract(_solid((30, 60, 180)), bbox=(0, 0, 64, 48))
    b = extract(_solid((100, 100, 100)), bbox=(0, 0, 64, 48))
    assert abs(score(a, b) - score(b, a)) < 1e-6


def test_score_different_colours_less_than_one():
    red = extract(_solid((0, 0, 220)), bbox=(0, 0, 64, 48))
    blue = extract(_solid((220, 0, 0)), bbox=(0, 0, 64, 48))
    assert score(red, blue) < 1.0
    # distinct pure hues shouldn't share a single HSV bin — expect near-zero overlap
    assert score(red, blue) < 0.1


def test_score_bounded_0_to_1():
    a = extract(_solid((30, 60, 180)), bbox=(0, 0, 64, 48))
    b = extract(_solid((100, 100, 100)), bbox=(0, 0, 64, 48))
    s = score(a, b)
    assert 0.0 <= s <= 1.0


def test_score_invalid_fingerprints_return_zero():
    empty = Fingerprint(np.zeros(8 ** 3, dtype=np.float32))
    ok = extract(_solid((50, 50, 50)), bbox=(0, 0, 64, 48))
    assert score(empty, ok) == 0.0
    assert score(ok, empty) == 0.0
    assert score(empty, empty) == 0.0


def test_score_shape_mismatch_raises():
    a = Fingerprint(np.ones(10, dtype=np.float32) / 10.0)
    b = Fingerprint(np.ones(20, dtype=np.float32) / 20.0)
    with pytest.raises(ValueError):
        score(a, b)
