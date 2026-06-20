"""Typed configuration objects replacing the global variables from the
original lab scripts.

``RawSpec`` describes how to interpret a headerless ``.raw`` capture (geometry,
bit depth, frame layout, Bayer pattern). ``PipelineConfig`` aggregates the
tunable parameters of the processing stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

VALID_PATTERNS: Tuple[str, ...] = ("RGGB", "BGGR", "GRBG", "GBRG")


@dataclass
class RawSpec:
    """Description of a headerless RAW Bayer capture.

    Any of ``width`` / ``height`` / ``num_frames`` may be left as ``None`` and
    inferred from the file size at read time, as long as the remaining
    information is sufficient to resolve the geometry unambiguously.
    """

    width: Optional[int] = None
    height: Optional[int] = None
    bit_depth: int = 12
    dtype: np.dtype = np.uint16
    frame_index: int = 0
    num_frames: Optional[int] = None
    bayer_pattern: str = "RGGB"

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        self.bayer_pattern = self.bayer_pattern.upper()
        if self.bayer_pattern not in VALID_PATTERNS:
            raise ValueError(
                f"Unknown Bayer pattern {self.bayer_pattern!r}; "
                f"expected one of {VALID_PATTERNS}"
            )
        if self.bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        for name in ("width", "height", "num_frames"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be a positive integer when set")

    @property
    def max_value(self) -> int:
        """Maximum representable sample value for the given bit depth."""
        return (1 << self.bit_depth) - 1

    @property
    def itemsize(self) -> int:
        """Size of a single sample in bytes."""
        return self.dtype.itemsize


@dataclass
class PipelineConfig:
    """Tunable parameters for the processing stages (replaces module globals)."""

    raw: RawSpec = field(default_factory=RawSpec)
    gamma: float = 1.0 / 2.2
    coef_a: float = 1.0
    gaussian_sigma: Tuple[float, float] = (10.0, 10.0)
    gaussian_radius: int = 5
    bilateral_sigma: Tuple[float, float] = (2.0, 2.0)
