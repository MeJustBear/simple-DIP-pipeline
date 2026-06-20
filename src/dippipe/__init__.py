"""Simple digital image processing (ISP) pipeline.

Public API re-exports the algorithm functions migrated from the original
lab scripts. IO, the stage protocol, the pipeline runner and the CLI are
introduced in later refactoring stages.
"""

from dippipe.config import PipelineConfig, RawSpec
from dippipe.io import (
    load_array,
    read_raw,
    read_raw_normalized,
    save_array,
    save_preview,
    to_float,
)
from dippipe.color import rgb2ycbcr, ycbcr2rgb
from dippipe.stages.demosaic import demosaic, demosaicing
from dippipe.stages.awb import white_patch, grayworld, awb_comb
from dippipe.stages.tone import gamma_correction, equalize, hist_equalise
from dippipe.stages.filtering import (
    gaussian_filter,
    gaussian_filter_dense,
    bilateral_filter,
    bilateral_filter_naive,
)
from dippipe.stages.base import Stage
from dippipe.stages.steps import AWBStage, DemosaicStage, FilterStage, ToneStage
from dippipe.pipeline import Pipeline, build_default_pipeline

__version__ = "0.3.0"

__all__ = [
    "PipelineConfig",
    "RawSpec",
    "read_raw",
    "read_raw_normalized",
    "to_float",
    "load_array",
    "save_array",
    "save_preview",
    "rgb2ycbcr",
    "ycbcr2rgb",
    "demosaic",
    "demosaicing",
    "white_patch",
    "grayworld",
    "awb_comb",
    "gamma_correction",
    "equalize",
    "hist_equalise",
    "gaussian_filter",
    "gaussian_filter_dense",
    "bilateral_filter",
    "bilateral_filter_naive",
    "Stage",
    "DemosaicStage",
    "AWBStage",
    "ToneStage",
    "FilterStage",
    "Pipeline",
    "build_default_pipeline",
    "__version__",
]
