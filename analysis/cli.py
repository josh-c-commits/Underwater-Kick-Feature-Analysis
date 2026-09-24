"""Command-line entry point for analysis.

Typical pipeline:

    python -m analysis normalize-video raw.mov clip.mp4
    python -m analysis track clip.mp4 boxes.csv --seed X Y --sheet sheet.png
    python -m analysis calibrate clip.mp4 calib.json --line floor --mark-range 0 15 1
    python -m analysis analyze boxes.csv kinematics.csv --calibration calib.json
    python -m analysis extract clip.mp4 poses.csv --model PATH --boxes boxes.csv
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Optional, Tuple

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


class _MarkRange(argparse.Action):
    """--mark-range START STOP STEP: evenly spaced marks, STOP included when a
    step lands on it. Appends to the same list as --marks, so the two can be
    mixed and still pair up with --line in command-line order."""

    def __call__(self, parser, namespace, values, option_string=None):
        start, stop, step = values
        if step <= 0 or stop <= start:
            parser.error(f"{option_string} needs START below STOP and a positive STEP, "
                         f"got {start:g} {stop:g} {step:g}")
        count = int(math.floor((stop - start) / step + 1e-9)) + 1
        # rounded so 0.1 steps give 0.3, not 0.30000000000000004
        marks = [round(start + i * step, 6) for i in range(count)]
        setattr(namespace, self.dest, (getattr(namespace, self.dest) or []) + [marks])


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
                          help="measure and compensate camera drift, for footage where the "
                               "camera may not have stayed put. Positions are corrected into "
                               "fixed pool coordinates. Switches itself off (identical output) "
                               "when it finds no camera motion beyond water noise. Corrects "
                               "sliding only, not rotation or zoom; reads the video ~3x, so "
                               "it's noticeably slower at 1080p. Use the same flag with "
                               "`calibrate` so both share one reference.")
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
                          help="name of a line of marks to click along, e.g. floor for "
                               "markers placed along the swimmer's lane line. Repeat for "
                               "each line, e.g. a second row of markers at swimmer depth.")
    p_calib.add_argument("--marks", type=float, nargs="+", action="append", dest="mark_lists",
                          metavar="METRES",
                          help="distances from the wall of the marks you'll click, e.g. "
                               "0 2.5 5 10 15. Give it once for every line to share, or once "
                               "per --line, in the same order, when lines differ.")
    p_calib.add_argument("--mark-range", type=float, nargs=3, action=_MarkRange,
                          dest="mark_lists", metavar=("START", "STOP", "STEP"),
                          help="evenly spaced marks, e.g. 0 15 1 for a marker every metre "
                               "out to 15m. Counts as one --marks list.")
    when = p_calib.add_mutually_exclusive_group()
    when.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"),
                      help="build the reference image from only these frames (1-indexed, "
                           "inclusive): when the markers were down, if they came out before "
                           "the swim. Otherwise the whole clip is used, and markers that "
                           "were only there briefly vanish from it.")
    when.add_argument("--seconds", type=float, nargs=2, metavar=("FROM", "TO"),
                      help="the same as --frames, in seconds from the start of the video")
    p_calib.add_argument("--edit", action="store_true",
                          help="reopen the clicks already saved in OUT_JSON, to fix one mark "
                               "without redoing the rest")
    p_calib.add_argument("--max-samples", type=int, default=120,
                          help="frames to median-composite into the reference image")
    p_calib.add_argument("--stabilize", action="store_true",
                          help="build the reference image the same way `track --stabilize` "
                               "builds its plate. Use it whenever you track with --stabilize: "
                               "on a moving camera a plain median is smeared and sits in a "
                               "different position, so clicked marks wouldn't line up with "
                               "measured positions. With --frames it also corrects for the "
                               "camera moving between the markers going down and the swim.")

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

    p_overlay = sub.add_parser(
        "overlay",
        help="draw distance lines (and optionally tracking) onto the video to check them",
    )
    p_overlay.add_argument("input_video")
    p_overlay.add_argument("out_video")
    p_overlay.add_argument("--calibration", default=None,
                            help="calibration json: draws lines of constant distance along "
                                 "the pool. Solid where it interpolates between reference "
                                 "lines, dashed where it's extrapolating -- trust those less.")
    p_overlay.add_argument("--boxes", default=None,
                            help="tracking csv: draws box, centroid, leading edge, and the "
                                 "swimmer's distance when a calibration is also given")
    p_overlay.add_argument("--every", type=float, default=1.0, metavar="METRES",
                            help="spacing of the distance lines (every 5m drawn heavier)")
    p_overlay.add_argument("--still", action="store_true",
                            help="draw onto the pool's median image (no swimmers, no ripple) "
                                 "and save a picture instead of a video, with your clicked "
                                 "marks shown. Rebuilds exactly the image the marks were "
                                 "clicked on: stabilized or not, from the same --frames. "
                                 "OUT should be a .png.")
    p_overlay.add_argument("--max-samples", type=int, default=120,
                            help="frames median-composited into the still's reference image")

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
    from .overlay import draw_overlay
    from .tracking import contact_sheet, detect_boxes, suggest_roi

    for path in (args.out_csv, args.preview, args.sheet):
        if path:
            _ensure_parent(path)

    roi = tuple(args.roi) if args.roi else None
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
        args.input_video, roi=roi, seed_point=seed, max_samples=args.max_samples,
        min_area=args.min_area, sigma=args.sigma, max_jump=args.max_jump,
        stabilize=args.stabilize,
    )
    boxes.to_csv(args.out_csv, index=False)
    print(f"Boxes saved to: {args.out_csv}")

    if args.preview:
        draw_overlay(args.input_video, args.preview, boxes=boxes)
    if args.sheet:
        contact_sheet(args.input_video, boxes, args.sheet, stride=args.sheet_stride)


def _mark_lists(args) -> List[List[float]]:
    """One list of mark distances per --line: a single list is shared by
    every line, otherwise lists pair with lines in command-line order."""
    lists = args.mark_lists
    if not lists:
        raise SystemExit("Give the distances of the marks you'll click, with --marks "
                         "(e.g. --marks 0 5 10 15) or --mark-range (e.g. --mark-range 0 15 1).")
    if len(lists) == 1:
        lists = lists * len(args.line)
    elif len(lists) != len(args.line):
        raise SystemExit(f"{len(lists)} lists of marks for {len(args.line)} --line(s). Give "
                         "one list for every line to share, or one per line, in order.")
    for name, marks in zip(args.line, lists):
        if len(marks) < 2:
            raise SystemExit(f"{name} needs at least two distances: a single known point "
                             "can't define a scale.")
        if len(set(marks)) != len(marks):
            raise SystemExit(f"{name}'s marks repeat a distance: {marks}")
    return lists


def _frame_range(args) -> Optional[Tuple[int, int]]:
    """--frames or --seconds as a checked (first, last) frame range, or None."""
    from .frames import fps, frame_count

    if args.frames is None and args.seconds is None:
        return None
    if args.seconds is not None:
        begin, finish = args.seconds
        if begin < 0 or finish <= begin:
            raise SystemExit(f"--seconds needs FROM before TO, got {begin:g} {finish:g}")
        rate = fps(args.input_video)
        first = int(math.floor(begin * rate)) + 1
        last = max(first, int(math.ceil(finish * rate)))
    else:
        first, last = args.frames
        if first < 1 or last < first:
            raise SystemExit(f"--frames needs 1 <= FIRST <= LAST, got {first} {last} "
                             "(frames count from 1)")
    total = frame_count(args.input_video)
    if total > 0 and first > total:
        raise SystemExit(f"{args.input_video} has only {total} frames.")
    if total > 0 and last > total:
        print(f"The video ends at frame {total}, so using frames {first}-{total}.")
        last = total
    return first, last


def _check_calibration_frames(args, reference, plate, frames) -> None:
    """Warn if the reference from `frames` doesn't sit where tracking's plate does."""
    from .frames import median_background
    from .tracking import plate_offset

    if plate is None:
        print("Building the plate tracking will use, to check the camera didn't move...")
        plate = median_background(args.input_video, max_samples=args.max_samples)
    shift = plate_offset(reference, plate)
    span = f"frames {frames[0]}-{frames[1]}"
    if shift is None:
        if args.stabilize:
            print(f"Warning: couldn't line {span} up with the stabilized plate. Stabilization "
                  "may have gone wrong on this clip: compare `overlay --still` images made "
                  "with and without --stabilize before trusting either.")
        else:
            print(f"Warning: couldn't compare {span} with the rest of the clip. The whole-clip "
                  "image is too blurred or featureless to measure against, which is what a "
                  "camera drifting throughout looks like. If it did, use --stabilize here "
                  "and on `track`.")
        return
    distance = math.hypot(*shift)
    if distance <= 2.0:
        print(f"Camera check: {span} line up with the rest of the clip "
              f"(within {distance:.1f}px).")
    elif args.stabilize:
        print(f"Warning: even after stabilizing, {span} sit {distance:.1f}px off the plate "
              "tracking uses, so readings may be off by that much. Stabilization treats "
              "shifts under ~4px as water noise, or couldn't measure these frames.")
    else:
        print(f"Warning: during {span} the camera sits {distance:.1f}px from where it is for "
              "the rest of the clip. It moved, perhaps knocked while the markers went in or "
              "out. Every mark would be off by that much against tracked positions. Re-run "
              "this and `track` with --stabilize, which corrects for it.")


def _run_calibrate(args) -> None:
    from .calibration import (
        Calibration, bend_report, coincident_lines, reference_line_from_clicks,
    )
    from .frames import frame_size
    from .pointpicker import label_keypoints
    from .tracking import calibration_reference

    # Fail before building the reference image or opening any window.
    mark_lists = _mark_lists(args)
    frames = _frame_range(args)
    reference_kind = "stabilized" if args.stabilize else "median"
    previous = {}
    if args.edit:
        if not os.path.exists(args.out_json):
            raise SystemExit(f"--edit reopens the clicks saved in {args.out_json}, "
                             "which doesn't exist yet.")
        saved = Calibration.load(args.out_json)
        previous = {line.name: {f"{w:g}m": (x, y) for x, y, w in line.knots}
                    for line in saved.lines}
        if (saved.reference, saved.frames) != (reference_kind, frames):
            print("Note: those clicks were made on a different reference image "
                  f"({saved.reference}, frames {saved.frames or 'all'}); check each still "
                  "sits on its mark.")

    _ensure_parent(args.out_json)
    reference, plate = calibration_reference(args.input_video, frames, args.stabilize,
                                             args.max_samples, progress=True)
    if frames is not None:
        _check_calibration_frames(args, reference, plate, frames)

    lines = []
    for line_name, marks in zip(args.line, mark_lists):
        names = [f"{mark:g}m" for mark in marks]
        existing = {k: v for k, v in previous.get(line_name, {}).items() if k in names}
        placed = label_keypoints(
            reference, names,
            title=f"{line_name}: click each mark along this line (s = not visible here).",
            existing=existing or None,
        )
        if placed is None:
            raise SystemExit(f"Cancelled while labelling {line_name}.")
        line, reason = reference_line_from_clicks(line_name, placed, names, marks)
        if line is None:
            print(f"Skipping {line_name}: {reason}.")
            continue
        lines.append(line)
        print(f"{line_name}: {len(line.knots)} marks recorded.")

    if not lines:
        raise SystemExit(
            "No usable reference lines, so nothing was saved. Each line needs at least "
            "two visible marks whose real distances you know -- try different --marks."
        )
    for first, second in coincident_lines(lines):
        print(f"Warning: '{first}' and '{second}' were clicked in the same places, so they "
              "are one line counted twice. The second line should run somewhere else in "
              "the frame, e.g. a row of markers at swimmer depth above a floor row.")
    if len(lines) == 1:
        print("One reference line: every reading comes from it whatever the swimmer's "
              "height in the frame, which is right for a level camera square-on to the "
              "lane. A second row of markers at swimmer depth would measure any tilt.")

    calibration = Calibration(
        lines=lines, frame_size=frame_size(args.input_video), video=args.input_video,
        reference=reference_kind, frames=frames,
    )
    calibration.save(args.out_json)
    print(f"Calibration saved to: {args.out_json}")

    low, high = calibration.covered_image_x()
    print(f"Calibrated image-x range: {low:.0f}-{high:.0f}px (outside this is extrapolation).")
    if len(lines) >= 2:
        middle = (low + high) / 2
        print(f"Depth ambiguity at x={middle:.0f}px: "
              f"{calibration.depth_ambiguity(middle):.2f}m between reference lines.")

    print("\nHow the marks check out against each other:")
    for line in lines:
        print("\n".join(bend_report(line)))
    print("\nTo see the lines and your clicked marks on the reference image:\n"
          f"  uv run python -m analysis overlay {args.input_video} <out.png> "
          f"--calibration {args.out_json} --still")


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

    elif args.command == "overlay":
        if args.calibration is None and args.boxes is None:
            raise SystemExit("Give --calibration, --boxes, or both.")
        import pandas as pd

        from .calibration import Calibration
        from .overlay import draw_calibration_still, draw_overlay

        _ensure_parent(args.out_video)
        if args.still:
            if args.calibration is None:
                raise SystemExit("--still draws a calibration's lines; pass --calibration.")
            if not args.out_video.lower().endswith((".png", ".jpg", ".jpeg")):
                raise SystemExit("--still writes a picture: give an output path ending .png")
            from .tracking import calibration_reference

            calibration = Calibration.load(args.calibration)
            # Rebuilt exactly as calibrate built it, so the marks show under
            # their rings -- including markers only down for calibration.frames.
            reference, _ = calibration_reference(
                args.input_video, calibration.frames, calibration.reference == "stabilized",
                args.max_samples, progress=True,
            )
            draw_calibration_still(reference, calibration, args.out_video, every=args.every)
            return
        draw_overlay(
            args.input_video,
            args.out_video,
            boxes=pd.read_csv(args.boxes) if args.boxes else None,
            calibration=Calibration.load(args.calibration) if args.calibration else None,
            every=args.every,
        )

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
