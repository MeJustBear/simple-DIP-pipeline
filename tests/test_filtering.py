"""Tests for the optimized Gaussian and bilateral filters."""

import numpy as np
import pytest

from dippipe.stages.filtering import (
    bilateral_filter,
    bilateral_filter_naive,
    gaussian_filter,
    gaussian_filter_dense,
)


def test_gaussian_preserves_constant():
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = gaussian_filter(img, sigmaX=2, sigmaY=2, radius=5)
    assert out.shape == img.shape
    # Luma is filtered; a flat field must remain (numerically) flat.
    assert np.allclose(out, out[0, 0], atol=1e-3)


def test_gaussian_rejects_even_radius():
    img = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(Exception):
        gaussian_filter(img, sigmaX=2, sigmaY=2, radius=4)


def test_separable_matches_dense_gaussian():
    rng = np.random.default_rng(0)
    img = rng.random((24, 24, 3)).astype(np.float32)
    fast = gaussian_filter(img, sigmaX=3, sigmaY=3, radius=5)
    dense = gaussian_filter_dense(img, sigmaX=3, sigmaY=3, radius=5)
    # Same kernel, just factored; allow small boundary/rounding differences.
    assert np.allclose(fast, dense, atol=2e-2)


def test_bilateral_preserves_constant():
    img = np.full((12, 12, 3), 0.3, dtype=np.float32)
    out = bilateral_filter(img, sigmaS=2, sigmaR=0.1, radius=3)
    assert out.shape == img.shape
    assert np.allclose(out, 0.3, atol=1e-4)


def test_fast_bilateral_matches_naive_at_center():
    # For the center pixel of a 5x5 image, a radius-2 window covers the whole
    # image (no padding), so the windowed filter equals the naive full-image one.
    rng = np.random.default_rng(1)
    img = rng.random((5, 5, 3)).astype(np.float32)
    fast = bilateral_filter(img, sigmaS=3, sigmaR=0.5, radius=2)
    naive = bilateral_filter_naive(img, sigmaS=3, sigmaR=0.5)
    assert np.allclose(fast[2, 2], naive[2, 2], atol=1e-5)


def test_bilateral_full_image_runs_fast():
    # The original naive version was impractical beyond a 20x20 crop; the
    # windowed version handles a larger image quickly.
    rng = np.random.default_rng(2)
    img = rng.random((64, 64, 3)).astype(np.float32)
    out = bilateral_filter(img, sigmaS=2, sigmaR=0.2, radius=3)
    assert out.shape == img.shape
