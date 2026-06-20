"""Tests for the vectorized demosaicer."""

import numpy as np
import pytest

from dippipe.config import VALID_PATTERNS
from dippipe.stages.demosaic import demosaic


@pytest.mark.parametrize("pattern", VALID_PATTERNS)
def test_constant_image_is_preserved(pattern):
    img = np.full((8, 8), 100, dtype=np.uint16)
    out = demosaic(img, pattern)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint16
    # A flat field must stay flat in every channel, including the borders.
    assert np.all(out == 100)


@pytest.mark.parametrize("shape", [(7, 5), (10, 4), (3, 9), (16, 16)])
def test_arbitrary_sizes(shape):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4096, size=shape, dtype=np.uint16)
    out = demosaic(img, "RGGB")
    assert out.shape == (shape[0], shape[1], 3)
    assert out.dtype == np.uint16


def test_measured_samples_are_kept_rggb():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 4096, size=(8, 8), dtype=np.uint16)
    out = demosaic(img, "RGGB")
    # RGGB: R at (even,even), B at (odd,odd), G at the other two cells.
    assert np.array_equal(out[0::2, 0::2, 0], img[0::2, 0::2])  # red samples
    assert np.array_equal(out[1::2, 1::2, 2], img[1::2, 1::2])  # blue samples
    assert np.array_equal(out[0::2, 1::2, 1], img[0::2, 1::2])  # green row 0
    assert np.array_equal(out[1::2, 0::2, 1], img[1::2, 0::2])  # green row 1


def test_linear_gradient_interpolates_midpoints():
    # A horizontal gradient should be reconstructed (near) exactly by bilinear
    # interpolation away from the borders.
    w = 8
    row = np.arange(w, dtype=np.float64) * 10.0
    img = np.tile(row, (8, 1))
    out = demosaic(img.astype(np.float64), "RGGB")
    interior = out[2:-2, 2:-2, :]
    expected = np.tile(row, (8, 1))[2:-2, 2:-2]
    for ch in range(3):
        assert np.allclose(interior[:, :, ch], expected, atol=1e-6)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        demosaic(np.zeros((4, 4, 3), dtype=np.uint16), "RGGB")


def test_rejects_unknown_pattern():
    with pytest.raises(ValueError):
        demosaic(np.zeros((4, 4), dtype=np.uint16), "XYZW")
