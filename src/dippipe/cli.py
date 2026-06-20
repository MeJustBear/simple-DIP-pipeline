"""Command line interface for the ISP pipeline.

Subcommands let every stage run independently (reading/writing ``.npy``
artifacts), or run the whole pipeline at once::

    dippipe demosaic capture.raw -o 01_rgb.npy --width 1280 --height 1024
    dippipe awb 01_rgb.npy -o 02_awb.npy --method combine
    dippipe tone 02_awb.npy -o 03_tone.npy --gamma 0.4545
    dippipe filter 03_tone.npy -o 04_out.npy
    dippipe run-all capture.raw -o out/ --width 1280 --height 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dippipe import __version__
from dippipe.config import VALID_PATTERNS, PipelineConfig, RawSpec
from dippipe.io.artifacts import load_array, save_array, save_preview
from dippipe.io.raw import read_raw
from dippipe.pipeline import build_default_pipeline
from dippipe.stages.steps import AWB_METHODS, AWBStage, FilterStage, ToneStage


def _add_raw_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame", type=int, default=0, help="frame index to read")
    parser.add_argument("--bit-depth", type=int, default=12)
    parser.add_argument("--pattern", choices=VALID_PATTERNS, default="RGGB")


def _raw_spec(args) -> RawSpec:
    return RawSpec(
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        frame_index=args.frame,
        bit_depth=args.bit_depth,
        bayer_pattern=args.pattern,
    )


def _save(output: Path, array: np.ndarray, preview: bool, max_value: float) -> None:
    output = Path(output)
    save_array(output, array)
    if preview:
        save_preview(output.with_suffix(".png"), array, max_value)


def _cmd_demosaic(args) -> None:
    spec = _raw_spec(args)
    frame = read_raw(args.input, spec)
    from dippipe.stages.demosaic import demosaic

    result = demosaic(frame, spec.bayer_pattern)
    _save(args.output, result, not args.no_preview, spec.max_value)


def _cmd_awb(args) -> None:
    data = load_array(args.input)
    result = AWBStage(method=args.method).run(data)
    _save(args.output, result, not args.no_preview, (1 << args.bit_depth) - 1)


def _cmd_tone(args) -> None:
    data = load_array(args.input)
    result = ToneStage(gamma=args.gamma, coef_a=args.coef_a,
                       bit_depth=args.bit_depth).run(data)
    _save(args.output, result, not args.no_preview, 1.0)


def _cmd_filter(args) -> None:
    data = load_array(args.input)
    result = FilterStage(
        gaussian_sigma=(args.gaussian_sigma, args.gaussian_sigma),
        gaussian_radius=args.radius,
        bilateral_sigma=(args.sigma_spatial, args.sigma_range),
        bilateral_radius=args.bilateral_radius,
    ).run(data)
    _save(args.output, result, not args.no_preview, 1.0)


def _cmd_run_all(args) -> None:
    spec = _raw_spec(args)
    config = PipelineConfig(
        raw=spec,
        gamma=args.gamma,
        coef_a=args.coef_a,
        gaussian_sigma=(args.gaussian_sigma, args.gaussian_sigma),
        gaussian_radius=args.radius,
        bilateral_sigma=(args.sigma_spatial, args.sigma_range),
    )
    frame = read_raw(args.input, spec)
    pipeline = build_default_pipeline(config, awb_method=args.method)
    pipeline.run(frame, args.output, resume=args.resume,
                 save_previews=not args.no_preview)
    print(f"Pipeline finished; artifacts written to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dippipe",
        description="Simple digital image processing pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"dippipe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # demosaic
    p = sub.add_parser("demosaic", help="read a RAW frame and demosaic it")
    p.add_argument("input", help="path to the .raw file")
    p.add_argument("-o", "--output", required=True, help="output .npy path")
    p.add_argument("--no-preview", action="store_true")
    _add_raw_args(p)
    p.set_defaults(func=_cmd_demosaic)

    # awb
    p = sub.add_parser("awb", help="apply automatic white balance")
    p.add_argument("input", help="input .npy (demosaiced RGB)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--method", choices=sorted(AWB_METHODS), default="combine")
    p.add_argument("--bit-depth", type=int, default=12)
    p.add_argument("--no-preview", action="store_true")
    p.set_defaults(func=_cmd_awb)

    # tone
    p = sub.add_parser("tone", help="gamma correction + histogram equalisation")
    p.add_argument("input", help="input .npy (RGB)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--gamma", type=float, default=1.0 / 2.2)
    p.add_argument("--coef-a", type=float, default=1.0)
    p.add_argument("--bit-depth", type=int, default=12)
    p.add_argument("--no-preview", action="store_true")
    p.set_defaults(func=_cmd_tone)

    # filter
    p = sub.add_parser("filter", help="gaussian + bilateral filtering")
    p.add_argument("input", help="input .npy (RGB, [0, 1])")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--gaussian-sigma", type=float, default=10.0)
    p.add_argument("--radius", type=int, default=5)
    p.add_argument("--sigma-spatial", type=float, default=2.0)
    p.add_argument("--sigma-range", type=float, default=2.0)
    p.add_argument("--bilateral-radius", type=int, default=None)
    p.add_argument("--no-preview", action="store_true")
    p.set_defaults(func=_cmd_filter)

    # run-all
    p = sub.add_parser("run-all", help="run the full pipeline")
    p.add_argument("input", help="path to the .raw file")
    p.add_argument("-o", "--output", required=True, help="output directory")
    p.add_argument("--method", choices=sorted(AWB_METHODS), default="combine")
    p.add_argument("--gamma", type=float, default=1.0 / 2.2)
    p.add_argument("--coef-a", type=float, default=1.0)
    p.add_argument("--gaussian-sigma", type=float, default=10.0)
    p.add_argument("--radius", type=int, default=5)
    p.add_argument("--sigma-spatial", type=float, default=2.0)
    p.add_argument("--sigma-range", type=float, default=2.0)
    p.add_argument("--resume", action="store_true",
                   help="skip stages whose artifact already exists")
    p.add_argument("--no-preview", action="store_true")
    _add_raw_args(p)
    p.set_defaults(func=_cmd_run_all)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
