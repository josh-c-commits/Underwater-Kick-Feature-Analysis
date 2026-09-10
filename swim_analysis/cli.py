"""Command-line entry point for swim_analysis.

    python -m swim_analysis normalize-video <video> <output_video>
    python -m swim_analysis extract <video> <csv> --model PATH
    python -m swim_analysis annotate <video> <csv> <output_video>
    python -m swim_analysis rank-visibility <csv> [--start-frame N --end-frame N | --start-time S --end-time S --fps F]
"""

from __future__ import annotations

import argparse

from .landmarks import DEFAULT_VISIBILITY_THRESHOLD


def _add_duration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--start-time", type=float, default=None, help="seconds")
    parser.add_argument("--end-time", type=float, default=None, help="seconds")
    parser.add_argument("--fps", type=float, default=None,
                         help="required if using --start-time/--end-time")


def _add_mapping_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mapping", type=str, default=None,
                         help="index,name file for CSVs with numbered columns")


def _validate_duration_args(args: argparse.Namespace) -> None:
    frame_given = args.start_frame is not None or args.end_frame is not None
    time_given = args.start_time is not None or args.end_time is not None
    if frame_given and time_given:
        raise SystemExit(
            "Use either --start-frame/--end-frame OR --start-time/--end-time, not both."
        )
    if time_given and args.fps is None:
        raise SystemExit("--fps is required when using --start-time/--end-time.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swim_analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="video -> csv (run pose detection)")
    p_extract.add_argument("input_video")
    p_extract.add_argument("csv_path")
    p_extract.add_argument("--model", required=True, help="path to pose_landmarker .task file")
    p_extract.add_argument("--no-normalize", action="store_true",
                            help="skip the ffmpeg re-encode step")
    p_extract.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=False,
                            dest="use_cpu",
                            help="use the CPU delegate instead of GPU (default is GPU, "
                                 "for speed). Pass --cpu if you hit the macOS Metal "
                                 "crash (DrishtiMetalHelper / 'Service is unavailable')")

    p_annotate = sub.add_parser("annotate", help="video + csv -> annotated video")
    p_annotate.add_argument("input_video")
    p_annotate.add_argument("csv_path")
    p_annotate.add_argument("output_video")
    p_annotate.add_argument("--visibility-threshold", type=float,
                             default=DEFAULT_VISIBILITY_THRESHOLD)

    p_normalize = sub.add_parser(
        "normalize-video",
        help="re-encode a video once so every later stage decodes identical frames",
    )
    p_normalize.add_argument("input_video")
    p_normalize.add_argument("output_video")

    p_rank_v = sub.add_parser("rank-visibility", help="rank landmarks by average visibility")
    p_rank_v.add_argument("csv_path")
    p_rank_v.add_argument("--out", default=None, help="optional path to save the table as csv")
    _add_duration_args(p_rank_v)
    _add_mapping_arg(p_rank_v)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "normalize-video":
        from .pose_extraction import normalize_video
        normalize_video(args.input_video, args.output_video)
        print(f"Normalized video saved to: {args.output_video}")
        print("Use this exact file with 'extract --no-normalize' so frame numbering "
              "stays consistent across stages.")

    elif args.command == "extract":
        from .pose_extraction import extract
        extract(
            args.input_video,
            args.csv_path,
            args.model,
            normalize=not args.no_normalize,
            use_cpu=args.use_cpu,
        )

    elif args.command == "annotate":
        from .annotate import annotate
        annotate(
            args.input_video,
            args.csv_path,
            args.output_video,
            visibility_threshold=args.visibility_threshold,
        )

    elif args.command == "rank-visibility":
        _validate_duration_args(args)
        from .ranking import rank_by_visibility
        table = rank_by_visibility(
            args.csv_path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_time=args.start_time,
            end_time=args.end_time,
            fps=args.fps,
            mapping_path=args.mapping,
        )
        print(table.to_string(index=False))
        if args.out:
            table.to_csv(args.out, index=False)
            print(f"\nSaved table to: {args.out}")


if __name__ == "__main__":
    main()
