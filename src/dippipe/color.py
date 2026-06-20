"""RGB <-> YCbCr color space conversions (BT.601 coefficients).

Migrated verbatim from the original ``funcs.py`` lab code; logic unchanged
except replacing the removed ``np.float`` alias with the builtin ``float``.
"""

import numpy as np


def rgb2ycbcr(image):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = image.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 0.5
    np.clip(ycbcr, 0, 1, out=ycbcr)
    return np.float32(ycbcr)


def ycbcr2rgb(image):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = image.astype(float)
    rgb[:, :, [1, 2]] -= 0.5
    rgb = rgb.dot(xform.T)
    np.clip(rgb, 0, 1, out=rgb)
    return np.float32(rgb)
