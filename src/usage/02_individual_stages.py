"""Example 2: run each stage independently and persist its result.

Demonstrates that any step can be executed on its own, taking the previous
step's saved ``.npy`` artifact as input -- the same idea the CLI exposes via
``dippipe demosaic`` / ``awb`` / ``tone`` / ``filter``.
"""

from _common import (
    OUTPUT_DIR,
    SAMPLE_BIT_DEPTH,
    SAMPLE_HEIGHT,
    SAMPLE_PATTERN,
    SAMPLE_RAW,
    SAMPLE_WIDTH,
)

from dippipe import (
    AWBStage,
    DemosaicStage,
    FilterStage,
    RawSpec,
    ToneStage,
    load_array,
    read_raw,
    save_array,
    save_preview,
)


def main() -> None:
    out_dir = OUTPUT_DIR / "individual_stages"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = RawSpec(
        width=SAMPLE_WIDTH,
        height=SAMPLE_HEIGHT,
        bit_depth=SAMPLE_BIT_DEPTH,
        bayer_pattern=SAMPLE_PATTERN,
        frame_index=1,
    )

    # --- Step 1: demosaic ---------------------------------------------------
    frame = read_raw(SAMPLE_RAW, spec)
    rgb = DemosaicStage(pattern=spec.bayer_pattern).run(frame)
    save_array(out_dir / "rgb.npy", rgb)
    save_preview(out_dir / "rgb.png", rgb, max_value=spec.max_value)

    # --- Step 2: white balance (loaded from the saved artifact) -------------
    rgb = load_array(out_dir / "rgb.npy")
    awb = AWBStage(method="grayworld").run(rgb)
    save_array(out_dir / "awb.npy", awb)
    save_preview(out_dir / "awb.png", awb, max_value=spec.max_value)

    # --- Step 3: tone mapping ----------------------------------------------
    tone = ToneStage(gamma=1 / 2.2, bit_depth=spec.bit_depth).run(awb)
    save_array(out_dir / "tone.npy", tone)
    save_preview(out_dir / "tone.png", tone)  # already in [0, 1]

    # --- Step 4: filtering --------------------------------------------------
    filtered = FilterStage(
        gaussian_sigma=(10.0, 10.0),
        gaussian_radius=5,
        bilateral_sigma=(2.0, 0.1),
    ).run(tone)
    save_array(out_dir / "filtered.npy", filtered)
    save_preview(out_dir / "filtered.png", filtered)

    print(f"Per-stage artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
