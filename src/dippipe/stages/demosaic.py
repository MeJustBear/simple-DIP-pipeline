"""Bayer bilinear demosaicing.

Two implementations are provided:

* :func:`demosaic` -- a vectorized, size-agnostic bilinear demosaicer that
  supports all four Bayer patterns (goal 1 of the refactoring plan);
* :func:`demosaicing` -- the original loop-based RGGB implementation kept
  verbatim for reference and regression comparison.
"""

import numpy as np
from scipy.ndimage import convolve

from dippipe.config import VALID_PATTERNS

# Bilinear interpolation kernels for normalized convolution. Division by the
# kernel applied to the sample mask makes border handling automatic and keeps
# the result independent of the absolute weights.
_KERNEL_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
_KERNEL_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float64)

# Row/col parity of the four cells of the 2x2 Bayer tile, in pattern order.
_TILE_POSITIONS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _channel_masks(height, width, pattern):
    masks = {ch: np.zeros((height, width), dtype=np.float64) for ch in "RGB"}
    for channel, (row0, col0) in zip(pattern, _TILE_POSITIONS):
        masks[channel][row0::2, col0::2] = 1.0
    return masks


def demosaic(data, pattern="RGGB"):
    """Vectorized bilinear demosaicing for an arbitrarily sized Bayer frame.

    Parameters
    ----------
    data:
        2D Bayer mosaic array of any height/width.
    pattern:
        One of ``RGGB`` / ``BGGR`` / ``GRBG`` / ``GBRG``.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` RGB array; the dtype matches the input (integer inputs
        are rounded and clipped to their representable range).
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2D Bayer mosaic")
    pattern = pattern.upper()
    if pattern not in VALID_PATTERNS:
        raise ValueError(
            f"Unknown Bayer pattern {pattern!r}; expected one of {VALID_PATTERNS}"
        )

    in_dtype = data.dtype
    work = data.astype(np.float64)
    height, width = work.shape
    masks = _channel_masks(height, width, pattern)

    out = np.zeros((height, width, 3), dtype=np.float64)
    for index, channel in enumerate("RGB"):
        kernel = _KERNEL_G if channel == "G" else _KERNEL_RB
        mask = masks[channel]
        numerator = convolve(work * mask, kernel, mode="constant", cval=0.0)
        denominator = convolve(mask, kernel, mode="constant", cval=0.0)
        denominator = np.where(denominator == 0.0, 1.0, denominator)
        interpolated = numerator / denominator
        # Keep measured samples exactly; interpolate only the missing ones.
        out[:, :, index] = np.where(mask > 0.0, work, interpolated)

    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        out = np.clip(np.round(out), info.min, info.max).astype(in_dtype)
    else:
        out = out.astype(in_dtype)
    return out


def demosaicing(data, height, width):
    """ 3d array of demosaicing result 0-Red 1- Green 2- Blue """
    result = np.zeros([height, width, 3], dtype=np.uint16)

    for i in range(0, height, 2):
        coefficients_blue = [1, 1, 1]
        coefficients_red = [1, 1, 1]
        coefficients_green = [1, 1, 1, 1]
        for j in range(0, width, 2):

            if j + 2 == width:
                coefficients_red[0] = 0
                coefficients_red[1] = 0
                coefficients_green[0] = 0
            if i + 2 == height:
                coefficients_red[2] = 0
                coefficients_red[0] = 0
                coefficients_green[1] = 0

            if i == 0:
                coefficients_blue[0] = 0
                coefficients_blue[2] = 0
                coefficients_green[2] = 0
            if j == 0:
                coefficients_blue[1] = 0
                coefficients_blue[0] = 0
                coefficients_green[3] = 0

            # Red block
            result[i][j][0] = data[i][j]
            tmp = (data[i][j] + coefficients_red[0] * data[coefficients_red[0] * (i + 2)][coefficients_red[0] * (j + 2)]
                   + coefficients_red[2] * data[coefficients_red[2] * (i + 2)][j] + coefficients_red[1] * data[i][
                       coefficients_red[1] * (j + 2)]) / (1 + sum(coefficients_red))
            result[i + 1][j + 1][0] = tmp
            tmp = (data[i][j] + tmp + coefficients_red[1] * data[i][coefficients_red[1] * (j + 2)]) / (
                    2 + coefficients_red[1])
            result[i][j + 1][0] = tmp
            tmp = (data[i][j] + coefficients_red[2] * data[coefficients_red[2] * (i + 2)][j]) / (
                    1 + coefficients_red[2])
            result[i + 1][j][0] = tmp

            # Blue block
            result[i + 1][j + 1][2] = data[i + 1][j + 1]
            tmp1 = (data[i + 1][j + 1] +
                    coefficients_blue[0] * data[coefficients_blue[0] * (i - 1)][coefficients_blue[0] * (j - 1)] +
                    coefficients_blue[1] * data[coefficients_blue[1] * (i + 1)][coefficients_blue[1] * (j - 1)] +
                    coefficients_blue[2] * data[coefficients_blue[2] * (i - 1)][coefficients_blue[2] * (j + 1)]) / \
                   (sum(coefficients_blue) + 1)
            result[i][j][2] = tmp1
            tmp = (tmp1 + data[i + 1][j + 1] +
                   coefficients_blue[1] * data[coefficients_blue[1] * (i + 1)][coefficients_blue[1] * (j - 1)]) / \
                  (2 + coefficients_blue[1])
            result[i + 1][j][2] = tmp
            tmp = (tmp1 + data[i + 1][j + 1] + coefficients_blue[2] *
                   data[coefficients_blue[2] * (i - 1)][coefficients_blue[2] * (j + 1)]) / (2 + coefficients_blue[2])
            result[i][j + 1][2] = tmp

            # Green block
            result[i][j + 1][1] = data[i][j + 1]
            result[i + 1][j][1] = data[i + 1][j]
            tmp = (data[i][j + 1] + data[i + 1][j] +
                   coefficients_green[0] * data[coefficients_green[0] * (i + 1)][coefficients_green[0] * (j + 2)] +
                   coefficients_green[1] * data[coefficients_green[1] * (i + 2)][coefficients_green[1] * (j + 1)]) / \
                  (2 + coefficients_green[0] + coefficients_green[1])
            result[i + 1][j + 1][1] = tmp
            tmp = (data[i + 1][j] + data[i][j + 1] +
                   coefficients_green[2] * int(data[coefficients_green[2] * i][coefficients_green[2] * (j - 1)]) +
                   coefficients_green[3] * int(data[coefficients_green[3] * (i - 1)][coefficients_green[3] * j])) / \
                  (2 + coefficients_green[2] + coefficients_green[3])
            result[i][j][1] = tmp

    return result
