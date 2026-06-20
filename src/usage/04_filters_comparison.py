"""Example 4: optimized filters vs. the reference naive implementations.

Compares runtimes of the fast (separable Gaussian, windowed bilateral) and the
original naive versions on a small crop, and confirms they agree numerically.
"""

import time

import numpy as np

from _common import (
    SAMPLE_BIT_DEPTH,
    SAMPLE_HEIGHT,
    SAMPLE_PATTERN,
    SAMPLE_RAW,
    SAMPLE_WIDTH,
)

from dippipe import (
    RawSpec,
    bilateral_filter,
    bilateral_filter_naive,
    demosaic,
    gaussian_filter,
    gaussian_filter_dense,
    read_raw,
)


def _timed(label, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    print(f"{label}: {time.perf_counter() - start:.3f}s")
    return result


def main() -> None:
    spec = RawSpec(
        width=SAMPLE_WIDTH,
        height=SAMPLE_HEIGHT,
        bit_depth=SAMPLE_BIT_DEPTH,
        bayer_pattern=SAMPLE_PATTERN,
        frame_index=1,
    )
    frame = read_raw(SAMPLE_RAW, spec)
    rgb = demosaic(frame, spec.bayer_pattern).astype(np.float32) / spec.max_value

    # Gaussian: separable vs. dense kernel, on the full image.
    fast_g = _timed("gaussian (separable)", gaussian_filter, rgb, 10, 10, 5)
    dense_g = _timed("gaussian (dense kernel)", gaussian_filter_dense, rgb, 10, 10, 5)
    print(f"  max abs diff: {np.abs(fast_g - dense_g).max():.2e}\n")

    # Bilateral: the naive version is only practical on a tiny crop.
    crop = rgb[560:580, 890:910, :]
    fast_b = _timed("bilateral (windowed, crop)", bilateral_filter, crop, 2, 0.1)
    naive_b = _timed("bilateral (naive, crop)", bilateral_filter_naive, crop, 2, 0.1)
    center = (crop.shape[0] // 2, crop.shape[1] // 2)
    print(f"  center pixel diff: {np.abs(fast_b[center] - naive_b[center]).max():.2e}")

    # The optimized bilateral can run on the full image too.
    _timed("bilateral (windowed, full image)", bilateral_filter, rgb, 2, 0.1)


if __name__ == "__main__":
    main()
