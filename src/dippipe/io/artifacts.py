"""Persistence of pipeline artifacts.

Each stage result is stored as a ``.npy`` array (lossless, exact) plus an
optional ``.png`` preview for quick visual inspection. Previews are written
with :func:`matplotlib.image.imsave`, which is headless and does not touch the
global pyplot state.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
from matplotlib import image as mpimg

PathLike = Union[str, os.PathLike]


def save_array(path: PathLike, array: np.ndarray) -> None:
    """Save a stage result as a ``.npy`` file."""
    np.save(path, array)


def load_array(path: PathLike) -> np.ndarray:
    """Load a stage result from a ``.npy`` file."""
    return np.load(path)


def _to_display(array: np.ndarray, max_value: Optional[float]) -> np.ndarray:
    """Map an arbitrary stage array to ``float`` RGB in ``[0, 1]`` for preview."""
    out = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        scale = float(max_value) if max_value else float(max(array.max(), 1))
        out = out / scale
    return np.clip(out, 0.0, 1.0)


def save_preview(
    path: PathLike,
    array: np.ndarray,
    max_value: Optional[float] = None,
) -> None:
    """Write a PNG preview of a stage result.

    Integer arrays are normalized by ``max_value`` (or their own maximum if
    not given); float arrays are assumed to already lie in ``[0, 1]`` and are
    only clipped.
    """
    mpimg.imsave(os.fspath(path), _to_display(array, max_value))
