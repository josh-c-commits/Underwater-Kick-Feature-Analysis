"""Command-line entry point for analysis.

Typical pipeline:

    python -m analysis normalize-video raw.mov clip.mp4
    python -m analysis track clip.mp4 boxes.csv --seed X Y --sheet sheet.png
    python -m analysis calibrate clip.mp4 calib.json --line near_rope --line far_rope \
        --marks 7.86 15.0
    python -m analysis analyze boxes.csv kinematics.csv --calibration calib.json
    python -m analysis extract clip.mp4 poses.csv --model PATH --boxes boxes.csv
"""

from __future__ import annotations

import argparse
import os

from .landmarks import DEFAULT_VISIBILITY_THRESHOLD, LANDMARK_NAMES


def _ensure_parent(path: str) -> str:
    """Create the directory an output file is about to be written into.

    Called before the work starts, not after: tracking a long clip and then
    failing on a missing directory throws away minutes of computation for a
    reason that was knowable up front.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return path


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
    parser = argparse.ArgumentParser(prog="analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_normalize = sub.add_parser(
        "normalize-video",
        help="re-encode a video once so every later stage decodes identical frames",
    )
    p_normalize.add_argument("input_video")
    p_normalize.add_argument("output_video")

    p_track = sub.add_parser(
        "track", help="video -> per-frame swimmer boxes (median background subtraction)"
    )
    p_track.add_argument("input_video")
    p_track.add_argument("out_csv")
    p_track.add_argument("--seed", type=int, nargs=2, metavar=("X", "Y"), default=None,
                          help="pixel coords on frame 1 identifying WHICH swimmer to follow. "
                               "Without it the largest blob wins, which is usually whoever is "
                               "nearest the camera rather than your subject.")
    p_track.add_argument("--pick-seed", action="store_true",
                          help="click the swimmer on frame 1 in a zoomable window instead of "
                               "passing --seed (needs a display)")
    p_track.add_argument("--roi", type=int, nargs=2, metavar=("Y0", "Y1"), default=None,
                          help="restrict the search to this row band; omit to auto-detect "
                               "(which excludes persistently moving rows like lane ropes)")
    p_track.add_argument("--no-auto-roi", action="store_true",
                          help="search the whole frame instead of auto-detecting a band")
    p_track.add_argument("--sigma", type=float, default=6.0,
                          help="detection threshold in robust deviations above the ROI median")
    p_track.add_argument("--max-samples", type=int, default=120,
                          help="frames median-composited into the background plate. All are "
                               "held in RAM at once, so ~120 is fine for small clips but 1080p "
                               "wants 40-60 (120 frames of 1920x1080 is ~750MB).")
    p_track.add_argument("--stabilize", action="store_true",
                          help="compensate camera drift by phase-correlating each frame "
                               "against the background plate. Use when the camera isn't "
                               "rigidly mounted. Only corrects translation, not rotation "
                               "or zoom, and costs ~36ms/frame at 1080p.")
    p_track.add_argument("--min-area", type=int, default=80)
    p_track.add_argument("--max-jump", type=float, default=60.0)
    p_track.add_argument("--preview", default=None, metavar="OUT_VIDEO",
                          help="also write the video with the tracked box drawn on it")
    p_track.add_argument("--sheet", default=None, metavar="OUT_PNG",
                          help="also write a contact sheet of crops for quick verification")
    p_track.add_argument("--sheet-stride", type=int, default=30)

    p_calib = sub.add_parser(
        "calibrate", help="click known distance marks to build an image->world map"
    )
    p_calib.add_argument("input_video")
    p_calib.add_argument("out_json")
    p_calib.add_argument("--line", action="append", required=True, metavar="NAME",
                          help="name of a reference line to click along, e.g. near_rope. "
                               "Repeat for each; two (the ropes either side of the swimmer) "
                               "is what makes depth ambiguity measurable.")
    p_calib.add_argument("--marks", type=float, nargs="+", required=True, metavar="METRES",
                          help="world distances of the marks you'll click, e.g. 7.86 15.0")
    p_calib.add_argument("--max-samples", type=int, default=120,
                          help="frames to median-composite into the reference image")

    p_label = sub.add_parser(
        "label", help="hand-label keypoints on sampled frames to create ground truth"
    )
    p_label.add_argument("input_video")
    p_label.add_argument("out_csv")
    p_label.add_argument("--frames", type=int, nargs="+", default=None,
                          help="explicit 1-indexed frames to label")
    p_label.add_argument("--sample", type=int, default=None,
                          help="instead, label this many frames spread across the clip")
    p_label.add_argument("--landmarks", nargs="+", default=None,
                          help="landmark names to place (default: a compact whole-body set)")
    p_label.add_argument("--boxes", default=None,
                          help="tracking csv; crops each frame around the swimmer for labelling")

    p_analyze = sub.add_parser("analyze", help="boxes csv -> kinematics csv + summary")
    p_analyze.add_argument("boxes_csv")
    p_analyze.add_argument("out_csv")
    p_analyze.add_argument("--video", default=None,
                            help="source video, to read fps from (else pass --fps)")
    p_analyze.add_argument("--fps", type=float, default=None)
    p_analyze.add_argument("--calibration", default=None,
                            help="calibration json; without it speeds stay in px/s")
    p_analyze.add_argument("--smooth-window", type=int, default=11)

    p_bodylen = sub.add_parser(
        "body-length",
        help="plot MediaPipe's world-landmark body length against image position",
    )
    p_bodylen.add_argument("csv_path", help="landmark csv containing world columns")
    p_bodylen.add_argument("out_png")
    p_bodylen.add_argument("--chain", nargs="+", default=None,
                            help="landmark chain to measure along")

    p_extract = sub.add_parser("extract", help="video -> csv (run pose detection)")
    p_extract.add_argument("input_video")
    p_extract.add_argument("csv_path")
    p_extract.add_argument("--model", required=True, help="path to pose_landmarker .task file")
    p_extract.add_argument("--no-normalize", action="store_true",
                            help="skip the ffmpeg re-encode step")
    p_extract.add_argument("--boxes", default=None,
                            help="tracking csv; crops around the swimmer before detection, "
                                 "which is what makes a ~0.3%%-of-frame subject detectable")
    p_extract.add_argument("--crop-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                            help="fixed crop size (default: 3x the median tracked box)")
    p_extract.add_argument("--no-world", action="store_true",
                            help="skip pose_world_landmarks columns")
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

    p_rank_v = sub.add_parser("rank-visibility", help="rank landmarks by average visibility")
    p_rank_v.add_argument("csv_path")
    p_rank_v.add_argument("--out", default=None, help="optional path to save the table as csv")
    _add_duration_args(p_rank_v)
    _add_mapping_arg(p_rank_v)

    return parser


DEFAULT_LABEL_SET = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle", "left_foot_index", "right_foot_index",
]


def _run_track(args) -> None:
    from .frames import median_background
    from .tracking import contact_sheet, detect_boxes, draw_preview, suggest_roi

    for path in (args.out_csv, args.preview, args.sheet):
        if path:
            _ensure_parent(path)

    roi = tuple(args.roi) if args.roi else None
    background = median_background(args.input_video, max_samples=args.max_samples)

    if roi is None and not args.no_auto_roi:
        roi = suggest_roi(args.input_video, max_samples=min(args.max_samples, 60))
        print(f"Auto-detected search band: rows {roi[0]}-{roi[1]}")

    seed = tuple(args.seed) if args.seed else None
    if seed is None and args.pick_seed:
        from .frames import frame_at
        from .pointpicker import pick_point

        seed = pick_point(frame_at(args.input_video, 1),
                          title="Click the swimmer to track (frame 1)")
        if seed is None:
            raise SystemExit("No seed point selected.")
        print(f"Using seed point: {seed}")
    if seed is None:
        print("No --seed given: following the largest moving object, which on footage "
              "with more than one lane occupied is often the wrong swimmer.")

    boxes = detect_boxes(
        args.input_video, background=background, roi=roi, seed_point=seed,
        min_area=args.min_area, sigma=args.sigma, max_jump=args.max_jump,
        stabilize=args.stabilize,
    )
    boxes.to_csv(args.out_csv, index=False)
    print(f"Boxes saved to: {args.out_csv}")

    if args.preview:
        draw_preview(args.input_video, boxes, args.preview)
    if args.sheet:
        contact_sheet(args.input_video, boxes, args.sheet, stride=args.sheet_stride)


def _run_calibrate(args) -> None:
    from .calibration import Calibration, ReferenceLine
    from .frames import frame_size, median_background
    from .pointpicker import label_keypoints

    _ensure_parent(args.out_json)
    print(f"Building a median reference image from {args.max_samples} frames...")
    reference = median_background(args.input_video, max_samples=args.max_samples)
    names = [f"{mark:g}m" for mark in args.marks]

    lines = []
    for line_name in args.line:
        placed = label_keypoints(
            reference, names,
            title=f"{line_name}: click each distance mark along this line.",
        )
        if placed is None:
            raise SystemExit(f"Cancelled while labelling {line_name}.")
        knots = [
            (float(placed[name][0]), float(placed[name][1]), float(mark))
            for name, mark in zip(names, args.marks)
            if placed.get(name) is not None
        ]
        if len(knots) < 2:
            raise SystemExit(f"{line_name}: need at least 2 marks, got {len(knots)}.")
        lines.append(ReferenceLine(name=line_name, knots=knots))
        print(f"{line_name}: {len(knots)} marks recorded.")

    calibration = Calibration(
        lines=lines, frame_size=frame_size(args.input_video), video=args.input_video
    )
    calibration.save(args.out_json)
    print(f"Calibration saved to: {args.out_json}")

    low, high = calibration.covered_image_x()
    print(f"Calibrated image-x range: {low:.0f}-{high:.0f}px (outside this is extrapolation).")
    if len(lines) >= 2:
        middle = (low + high) / 2
        print(f"Depth ambiguity at x={middle:.0f}px: "
              f"{calibration.depth_ambiguity(middle):.2f}m between reference lines.")


def _run_label(args) -> None:
    import csv as csv_module

    import pandas as pd

    from .frames import frame_at, frame_count, frame_size
    from .landmarks import NUM_LANDMARKS, named_header
    from .pointpicker import label_keypoints
    from .tracking import fixed_box

    _ensure_parent(args.out_csv)
    if (args.frames is None) == (args.sample is None):
        raise SystemExit("Provide exactly one of --frames or --sample.")

    total = frame_count(args.input_video)
    if args.frames:
        targets = list(args.frames)
    else:
        step = max(1, total // args.sample)
        targets = list(range(1, total + 1, step))[: args.sample]

    names = args.landmarks or DEFAULT_LABEL_SET
    width, height = frame_size(args.input_video)
    boxes = pd.read_csv(args.boxes).set_index("frame") if args.boxes else None

    with open(args.out_csv, "w", newline="") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(named_header())

        for number in targets:
            frame = frame_at(args.input_video, number)
            box = (0, 0, width, height)
            if boxes is not None and number in boxes.index and bool(boxes.loc[number, "found"]):
                row = boxes.loc[number]
                box = fixed_box(float(row["centroid_x"]), float(row["centroid_y"]),
                                int(row["box_w"]) * 3, int(row["box_h"]) * 3, width, height)
                frame = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

            placed = label_keypoints(
                frame, names, title=f"Frame {number} of {targets[-1]} -- label the swimmer."
            )
            if placed is None:
                print(f"Cancelled at frame {number}; wrote {targets.index(number)} rows.")
                break

            values = {}
            for name, point in placed.items():
                if point is None:
                    continue
                values[name] = (
                    (box[0] + point[0]) / width,
                    (box[1] + point[1]) / height,
                )

            out_row = [number]
            for name in LANDMARK_NAMES:
                if name in values:
                    x, y = values[name]
                    out_row += [f"{x:.8f}", f"{y:.8f}", "0.0", "1.0"]
                else:
                    out_row += ["", "", "", ""]
            writer.writerow(out_row)
            print(f"frame {number}: {len(values)}/{len(names)} landmarks placed")

    print(f"Ground truth saved to: {args.out_csv}")


def _run_analyze(args) -> None:
    import json

    import pandas as pd

    from .analysis import kinematics, summarize
    from .frames import fps as video_fps

    _ensure_parent(args.out_csv)
    if args.fps is None and args.video is None:
        raise SystemExit("Provide --fps or --video so the frame rate is known.")
    fps = args.fps if args.fps is not None else video_fps(args.video)

    calibration = None
    if args.calibration:
        from .calibration import Calibration

        calibration = Calibration.load(args.calibration)

    boxes = pd.read_csv(args.boxes_csv)
    table = kinematics(boxes, fps, calibration=calibration,
                       smooth_window=args.smooth_window)
    table.to_csv(args.out_csv, index=False)
    print(f"Kinematics saved to: {args.out_csv}\n")
    print(json.dumps(summarize(table, fps), indent=2))
    if calibration is None:
        print("\nSpeeds are in px/s. Pass --calibration for metres, or compare the "
              "velocity fluctuation index, which is dimensionless.")


def _run_body_length(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from .analysis import body_length_series

    _ensure_parent(args.out_png)
    table = pd.read_csv(args.csv_path)
    lengths = body_length_series(table, chain=args.chain)
    finite = np.isfinite(lengths)
    if not finite.any():
        raise SystemExit("No frames had usable world landmarks.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(table["frame"], lengths, lw=1)
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("body length (m)")
    axes[0].set_title("Body length over time")

    x_col = "nose_x" if "nose_x" in table.columns else None
    if x_col:
        axes[1].scatter(table[x_col][finite], lengths[finite], s=4)
        axes[1].set_xlabel("normalized image x")
        axes[1].set_ylabel("body length (m)")
        axes[1].set_title("Body length vs image position")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=120)

    values = lengths[finite]
    print(f"body length: mean {values.mean():.3f}m  sd {values.std():.3f}m  "
          f"spread {values.max() - values.min():.3f}m over {finite.sum()} frames")
    print("A roughly flat line means the world landmarks are metrically consistent; "
          "drift with image position points at lens distortion or model failure.")
    print(f"Plot saved to: {args.out_png}")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "normalize-video":
        from .pose_extraction import normalize_video
        _ensure_parent(args.output_video)
        normalize_video(args.input_video, args.output_video)
        print(f"Normalized video saved to: {args.output_video}")
        print("Use this exact file for every later stage so frame numbering stays consistent.")

    elif args.command == "track":
        _run_track(args)

    elif args.command == "calibrate":
        _run_calibrate(args)

    elif args.command == "label":
        _run_label(args)

    elif args.command == "analyze":
        _run_analyze(args)

    elif args.command == "body-length":
        _run_body_length(args)

    elif args.command == "extract":
        from .pose_extraction import extract
        _ensure_parent(args.csv_path)
        extract(
            args.input_video,
            args.csv_path,
            args.model,
            normalize=not args.no_normalize,
            use_cpu=args.use_cpu,
            boxes_csv=args.boxes,
            crop_size=tuple(args.crop_size) if args.crop_size else None,
            include_world=not args.no_world,
        )

    elif args.command == "annotate":
        from .annotate import annotate
        _ensure_parent(args.output_video)
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
