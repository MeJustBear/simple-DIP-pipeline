"""End-to-end tests for the pipeline runner."""

import numpy as np

from dippipe.config import PipelineConfig, RawSpec
from dippipe.pipeline import build_default_pipeline


def _make_raw(tmp_path, width=16, height=12):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 4096, size=(height, width), dtype=np.uint16)
    path = tmp_path / "cap.raw"
    frame.tofile(path)
    return path, frame


def test_pipeline_writes_all_artifacts(tmp_path):
    path, frame = _make_raw(tmp_path)
    spec = RawSpec(width=16, height=12, bit_depth=12)
    config = PipelineConfig(raw=spec, gaussian_radius=3, bilateral_sigma=(2.0, 0.3))
    pipeline = build_default_pipeline(config)

    out_dir = tmp_path / "out"
    result = pipeline.run(frame, out_dir, save_previews=True)

    assert result.shape == (12, 16, 3)
    names = ["00_demosaic", "01_awb", "02_tone", "03_filter"]
    for name in names:
        assert (out_dir / f"{name}.npy").exists()
        assert (out_dir / f"{name}.png").exists()


def test_pipeline_resume_reuses_artifacts(tmp_path):
    path, frame = _make_raw(tmp_path)
    spec = RawSpec(width=16, height=12, bit_depth=12)
    config = PipelineConfig(raw=spec, gaussian_radius=3, bilateral_sigma=(2.0, 0.3))
    pipeline = build_default_pipeline(config)
    out_dir = tmp_path / "out"

    first = pipeline.run(frame, out_dir, save_previews=False)
    # Second run with resume should load stored artifacts and match.
    second = pipeline.run(frame, out_dir, resume=True, save_previews=False)
    assert np.allclose(first, second)
