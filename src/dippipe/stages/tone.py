"""Tone mapping: gamma correction and histogram equalisation.

Migrated verbatim from the original ``funcs.py`` lab code; logic unchanged.
The matplotlib-based histogram helper keeps its side effects for now; it will
be decoupled from the algorithm when visualisation becomes an explicit option
in a later stage.
"""

import numpy as np
from matplotlib import pyplot as plt

nfig = 1


def gamma_correction(colorDif, gamma, coefA=1.0):
    colorDif = colorDif[:, :, 0] ** gamma
    result = colorDif * coefA
    np.clip(result, 0.0, 1.0, out=result)
    return result


def equalize(y, bits):
    """Vectorized histogram equalisation of an integer luma channel.

    Pure replacement for :func:`hist_equalise`: no matplotlib side effects and
    no per-pixel Python loop. ``y`` holds integer levels in ``[0, 2**bits)``;
    returns the equalised levels as the same integer dtype.

    Parameters
    ----------
    y:
        2D integer array of luma levels.
    bits:
        Bit depth; the number of levels is ``2**bits``.
    """
    levels = 1 << bits
    histogram, _ = np.histogram(y, bins=np.arange(0, levels + 2), density=True)
    cdf = np.cumsum(histogram)
    mapping = np.round(cdf * (levels - 1)).astype(y.dtype)
    return mapping[y]


def _histogram(array, title, maxBin):
    global nfig
    f = plt.figure(nfig)
    nfig += 1

    plt.title(title)
    histogram = np.histogram(array, bins=np.arange(0, maxBin), density=True)
    plt.bar(histogram[1][:-1], histogram[0][:], color="r")
    checksum = histogram[0].sum()
    if checksum > 1:
        raise Exception("Incorrect checksum")
    print("Sum hist = ", checksum)
    plt.grid("on")
    #plt.legend()
    return histogram[0][:]


def hist_equalise(y, l):
    """
    histogram equalisation algorithm
    rgbTriple: M x N x 3 np array
    l: bits for value
    mn: area of image
    """
    l = pow(2, l)
    res = np.ndarray.copy(y)

    print("Before equalise")
    hist = _histogram(y, "histogram before", (l + 2))
    print("Start equalise")

    np.cumsum(hist, out=hist)
    hist = np.uint16(np.round(hist * (l - 1)))

    for i in range(len(res)):
        for j in range(len(res[0])):
            try:
                res[i][j] = hist[res[i][j]]
            except Exception as e:
                print("err")
    _histogram(res, "histogram after", (l + 2))

    return res
