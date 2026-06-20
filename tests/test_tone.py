"""Tests for the vectorized histogram equalisation."""

import numpy as np

from dippipe.stages.tone import equalize


def test_equalize_shape_and_dtype():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4096, size=(32, 32), dtype=np.uint16)
    out = equalize(y, bits=12)
    assert out.shape == y.shape
    assert out.dtype == y.dtype


def test_equalize_is_monotonic_mapping():
    # Equalisation maps levels through a non-decreasing CDF, so the ordering of
    # distinct input levels must be preserved.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 256, size=(64, 64), dtype=np.uint16)
    out = equalize(y, bits=8)
    levels = np.unique(y)
    mapped = np.array([out[y == lvl][0] for lvl in levels])
    assert np.all(np.diff(mapped) >= 0)


def test_equalize_spreads_range():
    # A low-contrast image should use a wider range after equalisation.
    y = np.full((16, 16), 100, dtype=np.uint16)
    y[:8] = 110
    out = equalize(y, bits=8)
    assert out.max() >= y.max() - 100
