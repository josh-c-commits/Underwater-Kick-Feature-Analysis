"""Command-line entry point for swim_analysis.

    python -m swim_analysis normalize-video <video> <output_video>
    python -m swim_analysis extract <video> <csv> --model PATH
    python -m swim_analysis annotate <video> <csv> <output_video>
    python -m swim_analysis generate-outline <video> <masks_dir> [--point X Y | --bbox X1 Y1 X2 Y2]
        [--reprompt FRAME X Y ...] [--reprompt-bbox FRAME X1 Y1 X2 Y2 ...]
    python -m swim_analysis outline-extent <masks_dir> [--start-frame N --end-frame N | --start-time S --end-time S --fps F]
    python -m swim_analysis rank-visibility <csv> [--start-frame N --end-frame N | --start-time S --end-time S --fps F]
    python -m swim_analysis tune-pose <video> <masks_dir> --model PATH --ranking-csv PATH
        [--select-top-n N | --select-threshold T] [--n-calls N]
"""

from __future__ import annotations

import argparse

from .landmarks import DEFAULT_VISIBILITY_THRESHOLD, LANDMARK_NAMES


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
        help="re-encode a video once so extract and generate-outline decode identical frames",
    )
    p_normalize.add_argument("input_video")
    p_normalize.add_argument("output_video")

    p_outline = sub.add_parser(
        "generate-outline",
        help="run SAM2 to produce per-frame outline masks from a video clip",
    )
    p_outline.add_argument("input_video")
    p_outline.add_argument("masks_dir")
    p_outline.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"),
                            help="pixel coords to prompt SAM2 with on frame 1 (skips interactive click)")
    p_outline.add_argument("--bbox", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                            help="bounding box prompt instead of a point, on frame 1")
    p_outline.add_argument("--reprompt", action="append", type=int, nargs=3,
                            metavar=("FRAME", "X", "Y"), default=None,
                            help="add a corrective point prompt at FRAME (1-indexed) to re-lock "
                                 "onto the swimmer if tracking has drifted by then -- repeatable")
    p_outline.add_argument("--reprompt-bbox", action="append", type=int, nargs=5,
                            metavar=("FRAME", "X1", "Y1", "X2", "Y2"), default=None,
                            help="add a corrective bbox prompt at FRAME -- repeatable")
    p_outline.add_argument("--model-variant", default="sam2_t.pt",
                            help="sam2_t.pt (default, fastest) / sam2_s.pt / sam2_b.pt / sam2_l.pt")
    p_outline.add_argument("--device", default=None,
                            help="'cpu', 'mps' (Apple Silicon), 'cuda', or a CUDA index like '0'. "
                                 "Omit to let ultralytics auto-select (use this to test explicitly "
                                 "instead of guessing which device actually got used).")
    p_outline.add_argument("--imgsz", type=int, default=None,
                            help="internal processing resolution (default is likely 1024 "
                                 "regardless of source video size -- lower this, e.g. 512, to "
                                 "actually reduce compute/memory per frame. May be unstable; "
                                 "test on a short clip first.")
    p_outline.add_argument("--preview-first-frame", metavar="OUT_PATH", default=None,
                            help="just save a frame to OUT_PATH and exit (no segmentation) -- use "
                                 "this to find pixel coords for --point/--reprompt on a headless "
                                 "machine. Saves frame 1 unless --preview-frame-index is given.")
    p_outline.add_argument("--preview-frame-index", type=int, default=1,
                            help="which frame --preview-first-frame saves (1-indexed, default 1)")

    p_extent = sub.add_parser(
        "outline-extent",
        help="rightmost x-pixel of the SAM2 outline mask, per frame (no MediaPipe involved)",
    )
    p_extent.add_argument("masks_dir", help="directory of frame_000001.png-style outline masks")
    p_extent.add_argument("--out", default=None, help="optional path to save the table as csv")
    _add_duration_args(p_extent)

    p_rank_v = sub.add_parser("rank-visibility", help="rank landmarks by average visibility")
    p_rank_v.add_argument("csv_path")
    p_rank_v.add_argument("--out", default=None, help="optional path to save the table as csv")
    _add_duration_args(p_rank_v)
    _add_mapping_arg(p_rank_v)

    p_tune = sub.add_parser(
        "tune-pose",
        help="Bayesian-optimize pose-landmarker confidence thresholds against outline masks",
    )
    p_tune.add_argument("input_video", help="the SAME normalized video the masks_dir was generated from")
    p_tune.add_argument("masks_dir", help="SAM2 outline masks directory (from generate-outline)")
    p_tune.add_argument("--model", required=True, help="path to pose_landmarker .task file")
    p_tune.add_argument("--ranking-csv", required=True,
                         help="rank-visibility --out table; used to choose which landmarks are judged")
    select_group = p_tune.add_mutually_exclusive_group(required=True)
    select_group.add_argument("--select-top-n", type=int, default=None,
                               help="judge only the N most-visible landmarks from --ranking-csv")
    select_group.add_argument("--select-threshold", type=float, default=None,
                               help="judge every landmark with avg_visibility >= this, from --ranking-csv")
    p_tune.add_argument("--n-calls", type=int, default=20, help="number of Bayesian-optimization trials")
    p_tune.add_argument("--n-initial-points", type=int, default=5,
                         help="random trials before the optimizer starts modeling the objective")
    p_tune.add_argument("--missing-penalty", type=float, default=1.0,
                         help="points added per judged landmark that isn't detected in a frame")
    p_tune.add_argument("--distance-penalty-scale", type=float, default=0.05,
                         help="growth rate of exp(scale * pixels_outside_outline) - 1")
    p_tune.add_argument("--visibility-threshold", type=float, default=DEFAULT_VISIBILITY_THRESHOLD,
                         help="per-frame presence cutoff used while scoring (distinct from "
                              "--select-threshold, which only picks *which* landmarks are judged)")
    p_tune.add_argument("--num-poses", type=int, default=1)
    p_tune.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True,
                         dest="use_cpu", help="use the CPU delegate for each trial's extraction (default True)")
    p_tune.add_argument("--out", default=None, help="optional path to save best params + trial history as JSON")
    _add_duration_args(p_tune)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "normalize-video":
        from .pose_extraction import normalize_video
        normalize_video(args.input_video, args.output_video)
        print(f"Normalized video saved to: {args.output_video}")
        print("Use this exact file with both 'extract --no-normalize' and "
              "'generate-outline' to guarantee identical frame numbering.")

    elif args.command == "generate-outline":
        from .outline_capture import (
            generate_outline_masks,
            pick_point_interactively,
            save_frame_preview,
        )

        if args.preview_first_frame:
            save_frame_preview(args.input_video, args.preview_frame_index, args.preview_first_frame)
            print(f"Frame {args.preview_frame_index} saved to: {args.preview_first_frame}")
            return

        point, bbox = None, None
        if args.bbox is not None:
            bbox = tuple(args.bbox)
        elif args.point is not None:
            point = tuple(args.point)
        else:
            point = pick_point_interactively(args.input_video)
            print(f"Using clicked point: {point}")

        reprompts = [(f, (x, y)) for f, x, y in (args.reprompt or [])]
        reprompt_bboxes = [
            (f, (x1, y1, x2, y2)) for f, x1, y1, x2, y2 in (args.reprompt_bbox or [])
        ]

        generate_outline_masks(
            args.input_video,
            args.masks_dir,
            point=point,
            bbox=bbox,
            reprompts=reprompts,
            reprompt_bboxes=reprompt_bboxes,
            model_variant=args.model_variant,
            device=args.device,
            imgsz=args.imgsz,
        )

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

    elif args.command == "outline-extent":
        _validate_duration_args(args)
        from .outline import outline_rightmost_x_series
        table = outline_rightmost_x_series(
            args.masks_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_time=args.start_time,
            end_time=args.end_time,
            fps=args.fps,
        )
        print(table.to_string(index=False))
        if args.out:
            table.to_csv(args.out, index=False)
            print(f"\nSaved table to: {args.out}")

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

    elif args.command == "tune-pose":
        _validate_duration_args(args)
        import json

        import pandas as pd

        from .tuning import optimize_pose_hyperparameters, select_judged_landmarks

        ranking_table = pd.read_csv(args.ranking_csv)
        judged = select_judged_landmarks(
            ranking_table, threshold=args.select_threshold, top_n=args.select_top_n
        )
        print(f"Judging {len(judged)} landmark(s): {[LANDMARK_NAMES[i] for i in judged]}")

        result = optimize_pose_hyperparameters(
            args.input_video,
            args.masks_dir,
            args.model,
            judged,
            n_calls=args.n_calls,
            n_initial_points=args.n_initial_points,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            start_time=args.start_time,
            end_time=args.end_time,
            fps=args.fps,
            visibility_threshold=args.visibility_threshold,
            missing_penalty=args.missing_penalty,
            distance_penalty_scale=args.distance_penalty_scale,
            num_poses=args.num_poses,
            use_cpu=args.use_cpu,
        )

        print("\nBest hyperparameters found:")
        for name, value in result.best_params.items():
            print(f"  {name} = {value:.4f}")
        print(f"Best score: {result.best_score:.2f} (lower is better)")

        if args.out:
            with open(args.out, "w") as f:
                json.dump(
                    {
                        "best_params": result.best_params,
                        "best_score": result.best_score,
                        "judged_landmarks": [LANDMARK_NAMES[i] for i in judged],
                        "trials": [
                            {"params": dict(zip(result.best_params.keys(), p)), "score": s}
                            for p, s in zip(result.all_params, result.all_scores)
                        ],
                    },
                    f,
                    indent=2,
                )
            print(f"Saved tuning results to: {args.out}")


if __name__ == "__main__":
    main()
