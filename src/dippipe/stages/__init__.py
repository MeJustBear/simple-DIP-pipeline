"""Pipeline stage implementations.

Each module corresponds to one ISP step. The shared stage protocol and the
pipeline runner are introduced in a later refactoring stage.
"""

from dippipe.stages.base import Stage
from dippipe.stages.demosaic import demosaic, demosaicing
from dippipe.stages.awb import white_patch, grayworld, awb_comb
from dippipe.stages.tone import gamma_correction, equalize, hist_equalise
from dippipe.stages.filtering import (
    gaussian_filter,
    gaussian_filter_dense,
    bilateral_filter,
    bilateral_filter_naive,
)
from dippipe.stages.steps import (
    AWBStage,
    DemosaicStage,
    FilterStage,
    ToneStage,
)

__all__ = [
    "Stage",
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
    "DemosaicStage",
    "AWBStage",
    "ToneStage",
    "FilterStage",
]
