"""Pipeline runner and stage registry.

Replaces the ``tasks = [0, 0, 1, 0]`` flag soup and hardcoded ``1Task/`` ...
``4Task/`` paths from the original ``main.py``. The pipeline runs an ordered
list of stages, persisting each result as ``NN_<name>.npy`` (+ a ``.png``
preview) so any step can be resumed or inspected independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from dippipe.config import PipelineConfig
from dippipe.io.artifacts import load_array, save_array, save_preview
from dippipe.stages.base import Stage
from dippipe.stages.steps import AWBStage, DemosaicStage, FilterStage, ToneStage


@dataclass
class Pipeline:
    """An ordered sequence of stages with artifact persistence."""

    stages: List[Stage]
    preview_max_value: float = 4095.0

    def run(
        self,
        data: np.ndarray,
        out_dir: os.PathLike,
        resume: bool = False,
        save_previews: bool = True,
    ) -> np.ndarray:
        """Run all stages, writing artifacts to ``out_dir``.

        When ``resume`` is true, a stage whose ``.npy`` artifact already exists
        is skipped and its stored output is loaded instead.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for index, stage in enumerate(self.stages):
            stem = out_dir / f"{index:02d}_{stage.name}"
            npy_path = stem.with_suffix(".npy")

            if resume and npy_path.exists():
                data = load_array(npy_path)
                continue

            data = stage.run(data)
            save_array(npy_path, data)
            if save_previews:
                save_preview(stem.with_suffix(".png"), data, self.preview_max_value)

        return data


def build_default_pipeline(
    config: Optional[PipelineConfig] = None,
    awb_method: str = "combine",
) -> Pipeline:
    """Construct the standard ISP pipeline: demosaic -> AWB -> tone -> filter."""
    config = config or PipelineConfig()
    stages = [
        DemosaicStage(pattern=config.raw.bayer_pattern),
        AWBStage(method=awb_method),
        ToneStage(gamma=config.gamma, coef_a=config.coef_a,
                  bit_depth=config.raw.bit_depth),
        FilterStage(gaussian_sigma=config.gaussian_sigma,
                    gaussian_radius=config.gaussian_radius,
                    bilateral_sigma=config.bilateral_sigma),
    ]
    return Pipeline(stages=stages, preview_max_value=float(config.raw.max_value))


#: Factories for running a single stage from the CLI.
STAGE_REGISTRY = {
    "demosaic": DemosaicStage,
    "awb": AWBStage,
    "tone": ToneStage,
    "filter": FilterStage,
}
