"""Spatial filters: Gaussian and bilateral.

Optimized implementations (goal 3 of the refactoring plan):

* :func:`gaussian_filter` -- separable Gaussian using two 1D convolutions and a
  vectorized kernel, instead of the dense 2D kernel built with nested loops;
* :func:`bilateral_filter` -- windowed, fully vectorized bilateral filter
  (``O(H*W*(2r+1)^2)``) that can run on a full image instead of a 20x20 crop.

The original naive versions are kept as :func:`gaussian_filter_dense` and
:func:`bilateral_filter_naive` for reference and regression comparison.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d

from dippipe.color import rgb2ycbcr, ycbcr2rgb


def _calc(x, y, sigmaX, sigmaY):
    return 1/(2 * np.pi*sigmaX * sigmaY)*np.exp(-(x**2 + y**2)/(2*sigmaX * sigmaY))


def _calc_gauss(x, sigma):
    return 1/(2 * np.pi*sigma**2)*np.exp(-(x**2)/(2*sigma**2))


def _calc_euclid(x, y):
    return np.sqrt(((x[0] - x[1])**2 + (y[0] - y[1])**2))


def _gaussian_kernel_1d(variance, radius):
    """Normalized 1D Gaussian kernel of length ``radius``.

    The original code used an isotropic exponent ``(x^2 + y^2) / (2*sigmaX*sigmaY)``,
    so both axes share the same variance ``sigmaX*sigmaY``; the 2D kernel thus
    factorizes into two identical 1D kernels.
    """
    axis = np.arange(radius, dtype=np.float64) - (radius - 1) / 2.0
    kernel = np.exp(-(axis ** 2) / (2.0 * variance))
    return kernel / kernel.sum()


def gaussian_filter(rgbTriple, sigmaX, sigmaY, radius, on_luma=True):
    """Separable Gaussian blur.

    By default operates on the luma (Y) channel only, matching the original
    behavior. Set ``on_luma=False`` to blur all three RGB channels.
    """
    if not radius % 2:
        raise Exception("incorrect raduis value")

    kernel = _gaussian_kernel_1d(sigmaX * sigmaY, radius)

    if on_luma:
        ycbcr = rgb2ycbcr(rgbTriple)
        y = ycbcr[:, :, 0].astype(np.float64)
        y = convolve1d(y, kernel, axis=0, mode="reflect")
        y = convolve1d(y, kernel, axis=1, mode="reflect")
        ycbcr[:, :, 0] = y
        return ycbcr2rgb(ycbcr)

    out = np.empty(rgbTriple.shape, dtype=np.float64)
    for channel in range(3):
        tmp = convolve1d(rgbTriple[:, :, channel].astype(np.float64), kernel,
                         axis=0, mode="reflect")
        out[:, :, channel] = convolve1d(tmp, kernel, axis=1, mode="reflect")
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32)


def _bilateral_2d(image, sigmaS, sigmaR, radius):
    image = image.astype(np.float64)
    height, width = image.shape

    axis = np.arange(-radius, radius + 1)
    offset_y, offset_x = np.meshgrid(axis, axis, indexing="ij")
    spatial = np.exp(-(offset_x ** 2 + offset_y ** 2) / (2.0 * sigmaS ** 2))

    padded = np.pad(image, radius, mode="edge")
    accum = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            shifted = padded[radius + di:radius + di + height,
                             radius + dj:radius + dj + width]
            w = spatial[di + radius, dj + radius] * np.exp(
                -((image - shifted) ** 2) / (2.0 * sigmaR ** 2)
            )
            accum += w * shifted
            weight += w

    return accum / weight


def bilateral_filter(rgbTriple, sigmaS, sigmaR, radius=None, on_luma=True):
    """Windowed, vectorized bilateral filter.

    ``radius`` defaults to ``round(3*sigmaS)`` (the spatial Gaussian's effective
    support). By default filters the luma channel; ``on_luma=False`` filters all
    RGB channels.
    """
    if radius is None:
        radius = max(1, int(round(3 * sigmaS)))

    if on_luma:
        ycbcr = rgb2ycbcr(rgbTriple)
        ycbcr[:, :, 0] = _bilateral_2d(ycbcr[:, :, 0], sigmaS, sigmaR, radius)
        return ycbcr2rgb(ycbcr)

    out = np.empty(rgbTriple.shape, dtype=np.float32)
    for channel in range(3):
        out[:, :, channel] = _bilateral_2d(rgbTriple[:, :, channel],
                                           sigmaS, sigmaR, radius)
    return out


# --------------------------------------------------------------------------- #
# Reference (naive) implementations, kept for regression comparison.
# --------------------------------------------------------------------------- #

def gaussian_filter_dense(rgbTriple, sigmaX, sigmaY, radius):
    """Original dense-kernel Gaussian (nested-loop kernel + 2D convolution)."""
    if not radius % 2:
        raise Exception("incorrect raduis value")

    kernel = np.zeros([radius, radius], dtype=np.float32)
    ycbcr = rgb2ycbcr(rgbTriple)

    for i in range(radius):
        for j in range(radius):
            kernel[i][j] = _calc(i - (radius - 1) / 2, j - (radius - 1) / 2, sigmaX, sigmaY)

    kernel = kernel * (1 / np.sum(kernel))

    ycbcr[:, :, 0] = signal.convolve2d(ycbcr[:, :, 0], kernel, mode="same", boundary="symm", fillvalue=0)

    return ycbcr2rgb(ycbcr)


def bilateral_filter_naive(rgbTriple, sigmaS, sigmaR):
    """Original O(H^2 * W^2) bilateral filter over the whole image."""
    ycbcr = rgb2ycbcr(rgbTriple)
    copyLuma = np.copy(ycbcr[:, :, 0])

    for i in range(len(copyLuma)):
        for j in range(len(copyLuma[0])):
            currentSum = 0
            Wp = 0
            for k in range(len(copyLuma)):
                for l in range(len(copyLuma[0])):
                    currGaus = _calc_gauss(_calc_euclid([i, k], [j, l]), sigmaS) * _calc_gauss(ycbcr[i, j, 0] - ycbcr[k, l, 0], sigmaR)
                    Wp += currGaus
                    currentSum += currGaus * ycbcr[k, l, 0]
            copyLuma[i, j] = (1 / Wp) * currentSum
    ycbcr[:, :, 0] = copyLuma

    return ycbcr2rgb(ycbcr)
