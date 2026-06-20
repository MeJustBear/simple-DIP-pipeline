"""Concrete pipeline stages wrapping the algorithm functions.

Each stage adapts a pure algorithm to the :class:`~dippipe.stages.base.Stage`
protocol. Array conventions between stages:

* :class:`DemosaicStage`  -- in: 2D Bayer (int)        -> out: RGB (int, sensor scale)
* :class:`AWBStage`       -- in: RGB (int)             -> out: RGB (int, sensor scale)
* :class:`ToneStage`      -- in: RGB (int)             -> out: RGB (float, [0, 1])
* :class:`FilterStage`    -- in: RGB (float, [0, 1])   -> out: RGB (float, [0, 1])
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from dippipe.color import rgb2ycbcr, ycbcr2rgb
from dippipe.stages.base import Stage
from dippipe.stages.demosaic import demosaic
from dippipe.stages.awb import awb_comb, grayworld, white_patch
from dippipe.stages.tone import equalize, gamma_correction
from dippipe.stages.filtering import bilateral_filter, gaussian_filter

AWB_METHODS = {
    "white_patch": white_patch,
    "grayworld": grayworld,
    "combine": awb_comb,
}


class DemosaicStage(Stage):
    name = "demosaic"

    def __init__(self, pattern: str = "RGGB"):
        self.pattern = pattern

    def run(self, data: np.ndarray) -> np.ndarray:
        return demosaic(data, self.pattern)


class AWBStage(Stage):
    name = "awb"

    def __init__(self, method: str = "combine"):
        if method not in AWB_METHODS:
            raise ValueError(
                f"Unknown AWB method {method!r}; choose from {sorted(AWB_METHODS)}"
            )
        self.method = method

    def run(self, data: np.ndarray) -> np.ndarray:
        return AWB_METHODS[self.method](data.astype(np.float64))


class ToneStage(Stage):
    name = "tone"

    def __init__(self, gamma: float = 1.0 / 2.2, coef_a: float = 1.0,
                 bit_depth: int = 12):
        self.gamma = gamma
        self.coef_a = coef_a
        self.bit_depth = bit_depth

    def run(self, data: np.ndarray) -> np.ndarray:
        max_value = (1 << self.bit_depth) - 1
        rgb = data.astype(np.float32) / max_value
        ycbcr = rgb2ycbcr(rgb)

        y_gamma = gamma_correction(ycbcr, self.gamma, self.coef_a)
        y_int = np.round(y_gamma * max_value).astype(np.uint16)
        y_eq = equalize(y_int, self.bit_depth).astype(np.float32) / max_value

        ycbcr[:, :, 0] = y_eq
        return ycbcr2rgb(ycbcr)


class FilterStage(Stage):
    name = "filter"

    def __init__(self, gaussian_sigma: Tuple[float, float] = (10.0, 10.0),
                 gaussian_radius: int = 5,
                 bilateral_sigma: Tuple[float, float] = (2.0, 2.0),
                 bilateral_radius=None):
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_radius = gaussian_radius
        self.bilateral_sigma = bilateral_sigma
        self.bilateral_radius = bilateral_radius

    def run(self, data: np.ndarray) -> np.ndarray:
        blurred = gaussian_filter(
            data,
            sigmaX=self.gaussian_sigma[0],
            sigmaY=self.gaussian_sigma[1],
            radius=self.gaussian_radius,
        )
        return bilateral_filter(
            blurred,
            sigmaS=self.bilateral_sigma[0],
            sigmaR=self.bilateral_sigma[1],
            radius=self.bilateral_radius,
        )
