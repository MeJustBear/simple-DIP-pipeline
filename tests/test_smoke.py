"""Sanity tests for the migrated algorithms.

These check shapes/dtypes and that the public API imports cleanly. They do
not assert numerical correctness of the original lab logic, which is preserved
verbatim during this stage.
"""

import numpy as np
import pytest

import dippipe
from dippipe import (
    rgb2ycbcr,
    ycbcr2rgb,
    demosaicing,
    white_patch,
    grayworld,
    awb_comb,
    gaussian_filter,
    bilateral_filter,
)


def test_version_exposed():
    assert isinstance(dippipe.__version__, str)


def test_color_roundtrip_shapes():
    rng = np.random.default_rng(0)
    img = rng.random((4, 4, 3)).astype(np.float32)
    ycbcr = rgb2ycbcr(img)
    assert ycbcr.shape == (4, 4, 3)
    rgb = ycbcr2rgb(ycbcr)
    assert rgb.shape == (4, 4, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_demosaicing_shape_and_dtype():
    rng = np.random.default_rng(1)
    bayer = rng.integers(0, 4096, size=(6, 6), dtype=np.uint16)
    out = demosaicing(bayer, 6, 6)
    assert out.shape == (6, 6, 3)
    assert out.dtype == np.uint16


@pytest.mark.parametrize("algo", [white_patch, grayworld, awb_comb])
def test_awb_shape(algo):
    rng = np.random.default_rng(2)
    rgb = rng.integers(1, 4096, size=(4, 4, 3)).astype(np.float64)
    out = algo(rgb)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint16


def test_gaussian_filter_shape():
    rng = np.random.default_rng(3)
    rgb = rng.random((8, 8, 3)).astype(np.float32)
    out = gaussian_filter(rgb, sigmaX=2, sigmaY=2, radius=3)
    assert out.shape == (8, 8, 3)


def test_gaussian_filter_rejects_even_radius():
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(Exception):
        gaussian_filter(rgb, sigmaX=2, sigmaY=2, radius=4)


def test_bilateral_filter_shape_small():
    rng = np.random.default_rng(4)
    rgb = rng.random((4, 4, 3)).astype(np.float32)
    out = bilateral_filter(rgb, sigmaS=2, sigmaR=2)
    assert out.shape == (4, 4, 3)
