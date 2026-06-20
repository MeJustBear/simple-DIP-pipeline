"""Universal reader for headerless ``.raw`` Bayer captures.

The original lab code hardcoded a 1280x1024 frame and a fixed file path. This
module reads a single frame from a multi-frame ``.raw`` file of *any* size:
the geometry can be given explicitly via :class:`~dippipe.config.RawSpec`, or
partially inferred from the file size.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional, Tuple, Union

import numpy as np

from dippipe.config import RawSpec

PathLike = Union[str, os.PathLike]


def resolve_geometry(spec: RawSpec, total_pixels: int) -> Tuple[int, int, int]:
    """Resolve ``(width, height, num_frames)`` from ``spec`` and file pixel count.

    Missing values are inferred when the remaining information is sufficient:

    * both ``width`` and ``height`` known -> ``num_frames`` derived;
    * one spatial dimension plus ``num_frames`` known -> the other derived.

    Raises ``ValueError`` when the geometry is ambiguous or inconsistent with
    the actual file size.
    """
    width, height, num_frames = spec.width, spec.height, spec.num_frames

    if width and height:
        frame_pixels = width * height
        if total_pixels % frame_pixels:
            raise ValueError(
                f"File holds {total_pixels} samples, not a multiple of the "
                f"frame size {width}x{height} ({frame_pixels})."
            )
        available = total_pixels // frame_pixels
        if num_frames is None:
            num_frames = available
        elif num_frames > available:
            raise ValueError(
                f"Requested num_frames={num_frames} but file only holds "
                f"{available} frame(s) of {width}x{height}."
            )
    elif width and num_frames:
        denom = width * num_frames
        if total_pixels % denom:
            raise ValueError(
                f"Cannot infer height: {total_pixels} samples is not divisible "
                f"by width*num_frames ({denom})."
            )
        height = total_pixels // denom
    elif height and num_frames:
        denom = height * num_frames
        if total_pixels % denom:
            raise ValueError(
                f"Cannot infer width: {total_pixels} samples is not divisible "
                f"by height*num_frames ({denom})."
            )
        width = total_pixels // denom
    else:
        raise ValueError(
            "Insufficient geometry: provide both width and height, or one "
            "spatial dimension together with num_frames."
        )

    if spec.frame_index >= num_frames:
        raise ValueError(
            f"frame_index={spec.frame_index} out of range; file holds "
            f"{num_frames} frame(s)."
        )
    return width, height, num_frames


def read_raw(
    path: PathLike,
    spec: Optional[RawSpec] = None,
    **overrides,
) -> np.ndarray:
    """Read a single Bayer frame from a ``.raw`` file as a 2D array.

    Parameters
    ----------
    path:
        Path to the ``.raw`` file.
    spec:
        A :class:`~dippipe.config.RawSpec`. If ``None`` one is built from the
        keyword ``overrides`` (e.g. ``width=1280, height=1024``).
    **overrides:
        Field overrides applied on top of ``spec`` (or used to build a fresh
        spec when ``spec`` is ``None``).
    """
    if spec is None:
        spec = RawSpec(**overrides)
    elif overrides:
        spec = replace(spec, **overrides)
        spec.__post_init__()

    filesize = os.path.getsize(path)
    if filesize % spec.itemsize:
        raise ValueError(
            f"File size {filesize} is not a multiple of the sample size "
            f"{spec.itemsize} bytes for dtype {spec.dtype}."
        )
    total_pixels = filesize // spec.itemsize

    width, height, _ = resolve_geometry(spec, total_pixels)
    frame_pixels = width * height
    offset_bytes = frame_pixels * spec.frame_index * spec.itemsize

    data = np.fromfile(
        path, dtype=spec.dtype, count=frame_pixels, offset=offset_bytes
    )
    if data.size != frame_pixels:
        raise ValueError(
            f"Expected {frame_pixels} samples for frame {spec.frame_index}, "
            f"read {data.size}."
        )
    return data.reshape(height, width)


def to_float(data: np.ndarray, spec: RawSpec) -> np.ndarray:
    """Normalize raw integer samples to ``float32`` in ``[0, 1]``."""
    out = data.astype(np.float32) / float(spec.max_value)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def read_raw_normalized(
    path: PathLike,
    spec: Optional[RawSpec] = None,
    **overrides,
) -> np.ndarray:
    """Read a frame and normalize it to ``float32`` in ``[0, 1]``."""
    if spec is None:
        spec = RawSpec(**overrides)
    elif overrides:
        spec = replace(spec, **overrides)
        spec.__post_init__()
    data = read_raw(path, spec)
    return to_float(data, spec)
