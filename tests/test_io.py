"""Tests for universal RAW reading and geometry inference."""

import numpy as np
import pytest

from dippipe.config import RawSpec
from dippipe.io import read_raw, read_raw_normalized, resolve_geometry


def _write_frames(path, frames):
    frames.astype(np.uint16).tofile(path)


def test_read_explicit_geometry_selects_frame(tmp_path):
    n, h, w = 3, 4, 6
    frames = np.arange(n * h * w, dtype=np.uint16).reshape(n, h, w)
    path = tmp_path / "cap.raw"
    _write_frames(path, frames)

    for idx in range(n):
        frame = read_raw(path, width=w, height=h, frame_index=idx)
        assert frame.shape == (h, w)
        assert np.array_equal(frame, frames[idx])


def test_infer_height_from_width_and_frames(tmp_path):
    n, h, w = 4, 8, 5
    frames = np.arange(n * h * w, dtype=np.uint16).reshape(n, h, w)
    path = tmp_path / "cap.raw"
    _write_frames(path, frames)

    frame = read_raw(path, width=w, num_frames=n, frame_index=2)
    assert frame.shape == (h, w)
    assert np.array_equal(frame, frames[2])


def test_infer_width_from_height_and_frames(tmp_path):
    n, h, w = 2, 7, 9
    frames = np.arange(n * h * w, dtype=np.uint16).reshape(n, h, w)
    path = tmp_path / "cap.raw"
    _write_frames(path, frames)

    frame = read_raw(path, height=h, num_frames=n, frame_index=1)
    assert frame.shape == (h, w)
    assert np.array_equal(frame, frames[1])


def test_num_frames_derived_when_both_dims_known(tmp_path):
    n, h, w = 5, 3, 3
    frames = np.arange(n * h * w, dtype=np.uint16).reshape(n, h, w)
    spec = RawSpec(width=w, height=h)
    total_pixels = n * h * w
    rw, rh, rn = resolve_geometry(spec, total_pixels)
    assert (rw, rh, rn) == (w, h, n)


def test_frame_index_out_of_range_raises(tmp_path):
    n, h, w = 2, 4, 4
    frames = np.arange(n * h * w, dtype=np.uint16).reshape(n, h, w)
    path = tmp_path / "cap.raw"
    _write_frames(path, frames)
    with pytest.raises(ValueError):
        read_raw(path, width=w, height=h, frame_index=5)


def test_ambiguous_geometry_raises():
    spec = RawSpec(width=10)  # height and num_frames unknown
    with pytest.raises(ValueError):
        resolve_geometry(spec, 1000)


def test_inconsistent_size_raises(tmp_path):
    h, w = 4, 4
    frames = np.arange(h * w + 3, dtype=np.uint16)  # not a multiple of frame
    path = tmp_path / "cap.raw"
    frames.tofile(path)
    with pytest.raises(ValueError):
        read_raw(path, width=w, height=h)


def test_normalized_read_in_unit_range(tmp_path):
    h, w = 4, 4
    frame = np.full((h, w), 4095, dtype=np.uint16)
    path = tmp_path / "cap.raw"
    frame.tofile(path)
    out = read_raw_normalized(path, width=w, height=h, bit_depth=12)
    assert out.dtype == np.float32
    assert pytest.approx(out.max(), rel=1e-6) == 1.0
    assert out.min() >= 0.0
