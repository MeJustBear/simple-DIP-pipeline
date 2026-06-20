"""Example 3: universal RAW reading and demosaicing.

Shows that frame geometry need not be hardcoded: it can be given explicitly,
or partially inferred from the file size, and any Bayer pattern is supported.
"""

from _common import SAMPLE_HEIGHT, SAMPLE_RAW, SAMPLE_WIDTH

from dippipe import RawSpec, demosaic, read_raw
from dippipe.io.raw import resolve_geometry


def main() -> None:
    # 1) Fully explicit geometry, selecting the 3rd frame (index 2).
    explicit = read_raw(
        SAMPLE_RAW, width=SAMPLE_WIDTH, height=SAMPLE_HEIGHT, frame_index=2
    )
    print(f"Explicit read -> {explicit.shape}, dtype {explicit.dtype}")

    # 2) Infer the number of frames from the file size (both dims known).
    import os

    spec = RawSpec(width=SAMPLE_WIDTH, height=SAMPLE_HEIGHT)
    total_pixels = os.path.getsize(SAMPLE_RAW) // spec.itemsize
    w, h, n = resolve_geometry(spec, total_pixels)
    print(f"Inferred geometry: width={w}, height={h}, num_frames={n}")

    # 3) Infer the height from width + num_frames (one dimension unknown).
    inferred = read_raw(
        SAMPLE_RAW, width=SAMPLE_WIDTH, num_frames=n, frame_index=0
    )
    print(f"Inferred-height read -> {inferred.shape}")

    # 4) Demosaic with an explicit pattern; the function is size-agnostic.
    rgb = demosaic(explicit, pattern="RGGB")
    print(f"Demosaiced -> {rgb.shape}, dtype {rgb.dtype}")


if __name__ == "__main__":
    main()
