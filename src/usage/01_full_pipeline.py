"""Example 1: run the whole ISP pipeline with the high-level API.

Reads a RAW frame and runs demosaicing -> AWB -> tone mapping -> filtering,
writing every intermediate result (``.npy`` + ``.png`` preview) to disk.
"""

from _common import (
    OUTPUT_DIR,
    SAMPLE_BIT_DEPTH,
    SAMPLE_HEIGHT,
    SAMPLE_PATTERN,
    SAMPLE_RAW,
    SAMPLE_WIDTH,
)

from dippipe import PipelineConfig, RawSpec, build_default_pipeline, read_raw


def main() -> None:
    spec = RawSpec(
        width=SAMPLE_WIDTH,
        height=SAMPLE_HEIGHT,
        bit_depth=SAMPLE_BIT_DEPTH,
        bayer_pattern=SAMPLE_PATTERN,
        frame_index=1,
    )
    frame = read_raw(SAMPLE_RAW, spec)

    # PipelineConfig holds the tunable parameters for every stage.
    config = PipelineConfig(raw=spec, gamma=1 / 2.2)
    pipeline = build_default_pipeline(config, awb_method="combine")

    out_dir = OUTPUT_DIR / "full_pipeline"
    final = pipeline.run(frame, out_dir)

    print(f"Final RGB shape: {final.shape}, dtype: {final.dtype}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
