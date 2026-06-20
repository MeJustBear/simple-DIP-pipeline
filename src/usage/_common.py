"""Shared helpers for the usage examples.

These scripts demonstrate the :mod:`dippipe` package on the sample captures in
the repository's ``examples/`` directory. Run them from anywhere, e.g.::

    python src/usage/01_full_pipeline.py
"""

from pathlib import Path

# Repository root is two levels up from src/usage/.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
OUTPUT_DIR = REPO_ROOT / "usage_output"

# A 1280x1024, 12-bit, 10-frame RGGB capture shipped with the repo.
SAMPLE_RAW = EXAMPLES_DIR / "dump_white_color_10_frames.raw"
SAMPLE_WIDTH = 1280
SAMPLE_HEIGHT = 1024
SAMPLE_BIT_DEPTH = 12
SAMPLE_PATTERN = "RGGB"
