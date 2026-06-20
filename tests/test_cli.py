"""Tests for the CLI subcommands."""

import numpy as np

from dippipe.cli import main
from dippipe.io.artifacts import load_array


def _make_raw(tmp_path, width=16, height=12):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 4096, size=(height, width), dtype=np.uint16)
    path = tmp_path / "cap.raw"
    frame.tofile(path)
    return path


def test_cli_single_stages_chain(tmp_path):
    raw = _make_raw(tmp_path)
    rgb = tmp_path / "01_rgb.npy"
    awb = tmp_path / "02_awb.npy"
    tone = tmp_path / "03_tone.npy"
    out = tmp_path / "04_out.npy"

    main(["demosaic", str(raw), "-o", str(rgb),
          "--width", "16", "--height", "12", "--no-preview"])
    assert load_array(rgb).shape == (12, 16, 3)

    main(["awb", str(rgb), "-o", str(awb), "--method", "combine", "--no-preview"])
    assert load_array(awb).shape == (12, 16, 3)

    main(["tone", str(awb), "-o", str(tone), "--gamma", "0.4545", "--no-preview"])
    assert load_array(tone).shape == (12, 16, 3)

    main(["filter", str(tone), "-o", str(out),
          "--radius", "3", "--sigma-spatial", "2", "--no-preview"])
    assert load_array(out).shape == (12, 16, 3)


def test_cli_run_all(tmp_path):
    raw = _make_raw(tmp_path)
    out_dir = tmp_path / "pipeline_out"
    main(["run-all", str(raw), "-o", str(out_dir),
          "--width", "16", "--height", "12", "--radius", "3",
          "--sigma-spatial", "2", "--no-preview"])
    for name in ["00_demosaic", "01_awb", "02_tone", "03_filter"]:
        assert (out_dir / f"{name}.npy").exists()
