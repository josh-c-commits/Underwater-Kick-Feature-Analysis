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


def _ignored_rows(video_path: str) -> Optional[int]:
    """Rows above this are to be ignored, per the clip's normalize-video sidecar."""
    from .normalize import ignore_above

    top = ignore_above(video_path)
    if top:
        print(f"Ignoring rows above {top}, as set when the clip was normalized.")
    return top


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
        help="re-encode a video once (HDR to SDR, optional levelling) so every later "
             "stage decodes identical frames",
    )
    p_normalize.add_argument("input_video")
    p_normalize.add_argument("output_video")
    tilt = p_normalize.add_mutually_exclusive_group()
    tilt.add_argument("--level", action="store_true",
                      help="click two points on a line that's horizontal in the pool (e.g. the "
                           "lane rope) and rotate the frame to make it level")
    tilt.add_argument("--rotate", type=float, default=None, metavar="DEGREES",
                      help="rotate counter-clockwise by this many degrees instead of clicking a "
                           "level line")
    p_normalize.add_argument("--ignore-top", type=float, default=None, metavar="FRACTION",
                              help="have later stages ignore this fraction of the frame from the "
                                   "top, e.g. 0.33 for the water surface. The pixels are kept; "
                                   "only measurement skips them.")
    p_normalize.add_argument("--width", type=int, default=None, metavar="PX",
                              help="downscale to this width (default: full resolution)")
    p_normalize.add_argument("--no-tone-map", action="store_true",
                              help="keep HDR (HLG) values as they are instead of converting "
                                   "to standard range")

    p_track = sub.add_parser(
        "track", help="video -> per-frame swimmer boxes (median background subtraction)"
    )
    p_track.add_argument("input_video")
    p_track.add_argument("out_csv")
    p_track.add_argument("--seed", type=float, nargs=2, metavar=("X", "Y"), default=None,
                          help="the swimmer's position on frame 1 (a keyframe there). Without "
                               "any keyframe the largest blob wins, which is usually whoever "
                               "is nearest the camera rather than your subject.")
    p_track.add_argument("--pick-seed", type=int, nargs="?", const=1, default=None,
                          metavar="FRAME",
                          help="click the swimmer on this frame (default 1) in a zoomable window")
    p_track.add_argument("--seed-at", type=float, nargs=3, action="append", default=[],
                          metavar=("FRAME", "X", "Y"),
                          help="the swimmer's position on any frame; repeat for more keyframes")
    p_track.add_argument("--absent", type=int, action="append", default=[], metavar="FRAME",
                          help="a frame where the swimmer isn't visible; tracking won't place "
                               "them there")
    p_track.add_argument("--keyframes", default=None, metavar="JSON",
                          help="keyframes file (default data/keyframes/<clip>.json). Keyframes "
                               "given with --seed/--pick-seed/--seed-at/--absent are added to it "
                               "and saved, so they accumulate.")
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
    p_track.add_argument("--min-area", type=int, default=None,
                          help="smallest blob, in pixels (default: 80, scaled up above 1920 wide)")
    p_track.add_argument("--preview", default=None, metavar="OUT_VIDEO",
                          help="also write the video with the tracked box drawn on it")
    p_track.add_argument("--sheet", default=None, metavar="OUT_PNG",
                          help="also write a contact sheet of crops for quick verification")
    p_track.add_argument("--sheet-stride", type=int, default=30)

    p_review = sub.add_parser(
        "review", help="step through a tracked clip frame by frame and fix it with keyframes"
    )
    p_review.add_argument("input_video")
    p_review.add_argument("boxes_csv", help="the output of `track` (updated in place on save)")
    p_review.add_argument("--keyframes", default=None, metavar="JSON",
                          help="keyframes file (default data/keyframes/<clip>.json)")
    p_review.add_argument("--calibration", default=None,
                          help="calibration json: shows distance lines and readings in metres")
    p_review.add_argument("--display-width", type=int, default=1920, metavar="PX",
                          help="frames are shown at most this wide (clicks map back to full size)")
    p_review.add_argument("--every", type=float, default=1.0, metavar="METRES",
                          help="spacing of the calibration's distance lines")

    p_calib = sub.add_parser(
        "calibrate", help="click known distance marks to build an image->world map"
    )
    p_calib.add_argument("input_video")
    p_calib.add_argument("out_json", nargs="?", default=None,
                         help="where to save it (default: data/calibrations/<clip>.json, which "
                              "analyze, overlay and export then find on their own)")
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
    p_calib.add_argument("--feet", action="store_true",
                         help="the --marks / --mark-range numbers are in feet (converted to "
                              "metres), e.g. --mark-range 6 69 9 --feet for floor lines 6 ft from "
                              "the wall and every 9 ft after")
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
    lens = p_calib.add_mutually_exclusive_group()
    lens.add_argument("--no-lens", action="store_true",
                      help="interpolate between marks instead of undoing the lens, even when the "
                           "clip's lens is known (the 0.5x lens is detected automatically)")
    lens.add_argument("--lens-focal", type=float, default=None, metavar="PX",
                      help="undo a flat underwater window with this focal length (px at this "
                           "video's width) instead of the detected lens")
    p_calib.add_argument("--surface", action="store_true",
                         help="also click points along the lane rope, which floats on the surface: "
                              "the reference for depth below the surface")
    p_calib.add_argument("--under-rope", action="store_true",
                         help="you swam directly beneath the rope: give two --line names, the "
                              "floor lines either side of it, and click the rope (implies "
                              "--surface). The most accurate depth")
    p_calib.add_argument("--rope-offset", type=float, default=0.0, metavar="METRES",
                         help="how much nearer the camera the rope is than your lane (negative "
                              "if farther), to correct depth for it; 0 assumes the same distance")
    p_calib.add_argument("--camera-depth", type=float, default=0.5, metavar="METRES",
                         help="how deep the camera was, used with --rope-offset (default 0.5; "
                              "within 25 cm is plenty)")

    p_label = sub.add_parser(
        "label", help="hand-label keypoints on sampled frames to create ground truth"
    )
    p_label.add_argument("input_video")
    p_label.add_argument("out_csv", nargs="?", default=None,
                         help="where the labels go (default: data/labels/<clip>.csv). If it "
                              "exists, labelling resumes: frames already done are skipped")
    p_label.add_argument("--frames", type=int, nargs="+", default=None,
                         help="explicit 1-indexed frames to label")
    p_label.add_argument("--sample", type=int, default=None,
                         help="instead, label this many frames spread across the clip (default 40)")
    p_label.add_argument("--side", choices=["left", "right"], default=None,
                         help="the side of the body facing the camera; selects a --preset. For a "
                              "swimmer crossing right to left: left when swimming on the front, "
                              "right when on the back")
    p_label.add_argument("--preset", choices=["track", "body"], default="track",
                         help="with --side: 'track' is nose, hip and toes (enough to score the "
                              "tracker); 'body' is the whole near-side chain, for scoring pose "
                              "estimation (default: track)")
    p_label.add_argument("--landmarks", nargs="+", default=None,
                         help="exact landmark names to place instead of a preset")
    p_label.add_argument("--redo", action="store_true",
                         help="show frames that are already labelled too, to correct them")
    p_label.add_argument("--boxes", default=None,
                         help="tracking csv; start each frame zoomed on the tracker's box instead "
                              "of following your own previous clicks")

    p_evaluate = sub.add_parser(
        "evaluate-track", help="score a tracking csv against hand labels from 'label'"
    )
    p_evaluate.add_argument("input_video")
    p_evaluate.add_argument("boxes_csv")
    p_evaluate.add_argument("labels_csv", nargs="?", default=None,
                            help="default: data/labels/<clip>.csv")
    p_evaluate.add_argument("--reach", type=float, default=0.5,
                            help="how near, in body lengths, the tracker's centre must be to the "
                                 "labelled hip to count as the right swimmer (default 0.5)")
    p_evaluate.add_argument("--out", default=None,
                            help="also save the frame-by-frame comparison as csv")

    p_analyze = sub.add_parser("analyze", help="boxes csv -> kinematics csv + summary")
    p_analyze.add_argument("boxes_csv")
    p_analyze.add_argument("out_csv")
    p_analyze.add_argument("--video", default=None,
                            help="source video, to read fps from (else pass --fps)")
    p_analyze.add_argument("--fps", type=float, default=None)
    p_analyze.add_argument("--calibration", default=None,
                            help="calibration json (default: the clip's own at "
                                 "data/calibrations/<clip>.json, found via --video); without "
                                 "one, speeds stay in px/s")
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

    p_export = sub.add_parser(
        "export", help="video of the tracked swimmer: spotlit in the full frame, or followed"
    )
    p_export.add_argument("input_video")
    p_export.add_argument("boxes_csv")
    p_export.add_argument("out_video")
    p_export.add_argument("--mode", choices=["spotlight", "follow"], default="spotlight",
                          help="spotlight: the whole frame greyed out except the swimmer; "
                               "follow: a camera that glides along with the swimmer, showing only "
                               "them (default: spotlight)")
    p_export.add_argument("--width", type=int, default=None,
                          help="output width in px (default: 1920 for spotlight, 1280 for follow)")
    p_export.add_argument("--zoom", type=float, default=3.0,
                          help="follow: how many of the swimmer's lengths the view spans (default 3)")
    p_export.add_argument("--calibration", default=None,
                          help="calibration json, to show speed in m/s rather than px/s")
    p_export.add_argument("--no-hud", action="store_true",
                          help="leave out the time / status / speed line")

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

    top = _ignored_rows(args.input_video)
    roi = tuple(args.roi) if args.roi else None
    if roi is None and not args.no_auto_roi:
        roi = suggest_roi(args.input_video, max_samples=min(args.max_samples, 60),
                          ignore_above=top)
        print(f"Auto-detected search band: rows {roi[0]}-{roi[1]}")

    from .association import (
        Keyframe, keyframes_path, load_keyframes, merge_keyframes, save_keyframes,
    )

    key_file = args.keyframes or keyframes_path(args.input_video)
    stored = load_keyframes(key_file)
    added = [Keyframe(1, *args.seed)] if args.seed else []
    added += [Keyframe(int(frame), x, y) for frame, x, y in args.seed_at]
    added += [Keyframe(int(frame)) for frame in args.absent]
    if args.pick_seed is not None:
        from .frames import frame_at
        from .pointpicker import pick_point

        point = pick_point(frame_at(args.input_video, args.pick_seed),
                           title=f"Click the swimmer to track (frame {args.pick_seed})")
        if point is None:
            raise SystemExit("No point selected.")
        added.append(Keyframe(args.pick_seed, float(point[0]), float(point[1])))
    keyframes = merge_keyframes(stored, added)
    if added:
        save_keyframes(key_file, keyframes, video=args.input_video)
        print(f"Keyframes saved to: {key_file}")
    if keyframes:
        marks = ", ".join(str(k.frame) + (" (absent)" if k.absent else "") for k in keyframes)
        print(f"Keyframes on frames: {marks}")
    else:
        print("No keyframes: following the largest moving object, which on footage with "
              "more than one lane occupied is often the wrong swimmer. Add one with "
              "--pick-seed FRAME.")

    from .tracking import candidates_path, detect_candidates

    candidates = detect_candidates(
        args.input_video, roi=roi, min_area=args.min_area, sigma=args.sigma,
        stabilize=args.stabilize, max_samples=args.max_samples, ignore_above=top,
    )
    boxes = detect_boxes(args.input_video, keyframes=keyframes, candidates=candidates)
    boxes.to_csv(args.out_csv, index=False)
    candidates[0].to_csv(candidates_path(args.out_csv), index=False)
    print(f"Boxes saved to: {args.out_csv}")
    print(f"To review frame by frame and fix it with keyframes:\n"
          f"  uv run python -m analysis review {args.input_video} {args.out_csv}")

    if args.preview:
        draw_overlay(args.input_video, args.preview, boxes=boxes)
    if args.sheet:
        contact_sheet(args.input_video, boxes, args.sheet, stride=args.sheet_stride)


def _mark_lists(args) -> List[List[float]]:
    """One list of mark distances per --line: a single list is shared by
    every line, otherwise lists pair with lines in command-line order."""
    lists = args.mark_lists
    if lists and getattr(args, "feet", False):
        lists = [[round(value * 0.3048, 4) for value in marks] for marks in lists]
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


def _check_calibration_frames(args, reference, plate, frames, top=None) -> None:
    """Warn if the reference from `frames` doesn't sit where tracking's plate does."""
    from .frames import median_background
    from .tracking import plate_offset

    if plate is None:
        print("Building the plate tracking will use, to check the camera didn't move...")
        plate = median_background(args.input_video, max_samples=args.max_samples,
                                  ignore_above=top)
    shift = plate_offset(reference, plate, ignore_above=top)
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
        Calibration, bend_report, calibration_path, coincident_lines, reference_line_from_clicks,
        ruler_report,
    )
    from .frames import frame_size
    from .pointpicker import label_keypoints
    from .tracking import calibration_reference

    # Fail before building the reference image or opening any window.
    mark_lists = _mark_lists(args)
    frames = _frame_range(args)
    if args.out_json is None:
        args.out_json = calibration_path(args.input_video)
    if args.under_rope:
        if len(args.line) != 2:
            raise SystemExit("--under-rope needs two --line names: the floor lines on either side "
                             "of the rope you swam beneath.")
        args.surface = True
    lens = None
    if args.lens_focal:
        from .lens import FlatPort

        width, height = frame_size(args.input_video)
        lens = FlatPort(args.lens_focal, width / 2.0, height / 2.0)
    elif not args.no_lens:
        from .lens import lens_for_video

        lens = lens_for_video(args.input_video)
    print(f"Lens: undoing the underwater window's refraction (focal length {lens.focal:.0f} px)."
          if lens else "Lens: none known, so distances are interpolated between marks.")
    reference_kind = "stabilized" if args.stabilize else "median"
    previous = {}
    if args.edit:
        if not os.path.exists(args.out_json):
            raise SystemExit(f"--edit reopens the clicks saved in {args.out_json}, "
                             "which doesn't exist yet.")
        saved = Calibration.load(args.out_json)
        previous = {line.name: {f"{w:g}m": (x, y) for x, y, w in line.knots}
                    for line in saved.lines}
        previous["surface"] = {f"rope {i + 1}": point for i, point in enumerate(saved.surface)}
        if (saved.reference, saved.frames) != (reference_kind, frames):
            print("Note: those clicks were made on a different reference image "
                  f"({saved.reference}, frames {saved.frames or 'all'}); check each still "
                  "sits on its mark.")

    _ensure_parent(args.out_json)
    top = _ignored_rows(args.input_video)
    reference, plate = calibration_reference(args.input_video, frames, args.stabilize,
                                             args.max_samples, progress=True, ignore_above=top)
    if frames is not None:
        _check_calibration_frames(args, reference, plate, frames, top)

    lines = []
    for line_name, marks in zip(args.line, mark_lists):
        names = [f"{mark:g}m" for mark in marks]
        existing = {k: v for k, v in previous.get(line_name, {}).items() if k in names}
        title = (f"{line_name}: click where each floor line crosses the line you swam along, "
                 "starting from the wall you pushed off from (s = not visible)." if lens else
                 f"{line_name}: click each mark along this line (s = not visible here).")
        placed = label_keypoints(reference, names, title=title, existing=existing or None)
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
    surface = []
    if args.surface:
        names = [f"rope {i}" for i in range(1, 6)]
        placed = label_keypoints(
            reference, names,
            title="Click 3-5 points along the lane rope, spread across the frame "
                  "(s = skip one; enter when done).",
            existing=previous.get("surface") or None,
        )
        if placed is None:
            raise SystemExit("Cancelled while clicking the rope.")
        surface = [placed[name] for name in names if placed.get(name) is not None]
        if len(surface) < 2:
            raise SystemExit("The rope needs at least two points; nothing was saved.")
    for first, second in coincident_lines(lines):
        print(f"Warning: '{first}' and '{second}' were clicked in the same places, so they "
              "are one line counted twice. The second line should run somewhere else in "
              "the frame, e.g. a row of markers at swimmer depth above a floor row.")
    if len(lines) == 1 and lens is None:
        print("One reference line: every reading comes from it whatever the swimmer's "
              "height in the frame, which is right for a level camera square-on to the "
              "lane. A second row of markers at swimmer depth would measure any tilt.")

    calibration = Calibration(
        lines=lines, frame_size=frame_size(args.input_video), video=args.input_video,
        reference=reference_kind, frames=frames, lens=lens, surface=surface,
        under_rope=args.under_rope, rope_offset=args.rope_offset, camera_depth=args.camera_depth,
    )
    if args.under_rope and len(lines) < 2:
        raise SystemExit("--under-rope needs both floor lines usable; nothing was saved.")
    calibration.save(args.out_json)
    print(f"Calibration saved to: {args.out_json}")

    if lens is not None:
        print("\nHow the marks fit the lens model:")
        for line in lines:
            print("\n".join(ruler_report(calibration, line)))
        where = "halfway between the two lines" if args.under_rope else f"along '{lines[0].name}'"
        print(f"\nYour lane ({where}) is {calibration.swimmer_distance():.1f} m from the camera.")
        if surface:
            print(f"Rope: the camera's leftover roll is {math.degrees(calibration.roll()):+.2f} degrees, "
                  "taken out of every reading.")
            if not args.under_rope and not args.rope_offset:
                print("Depth assumes the rope is at your lane's distance. If it borders your lane, "
                      "depth can be off by ~camera depth x half a lane / that distance (a few cm); "
                      "--rope-offset corrects it, and swimming beneath the rope (--under-rope) "
                      "removes it.")
        else:
            print("No rope clicked, so no depth below the surface: add --surface for it.")
    else:
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
    import cv2
    import pandas as pd

    from . import pointpicker
    from .frames import frame_count, frame_size
    from .groundtruth import (box_view, follow_view, labels_path, preset_names, read_labels,
                              sample_frames, write_labels)
    from .review import FrameSource

    if args.frames is not None and args.sample is not None:
        raise SystemExit("Give --frames or --sample, not both.")
    if args.landmarks:
        unknown = [name for name in args.landmarks if name not in LANDMARK_NAMES]
        if unknown:
            raise SystemExit(f"Unknown landmark name(s): {', '.join(unknown)}.")
        names = list(args.landmarks)
    elif args.side:
        names = preset_names(args.preset, args.side)
    else:
        names = DEFAULT_LABEL_SET

    out = args.out_csv or labels_path(args.input_video)
    width, height = frame_size(args.input_video)
    targets = (sorted(set(args.frames)) if args.frames
               else sample_frames(frame_count(args.input_video), args.sample or 40))
    labels = read_labels(out, width, height)
    todo = [f for f in targets
            if args.redo or any(name not in labels.get(f, {}) for name in names)]
    if not todo:
        print(f"All {len(targets)} frames already have {', '.join(names)} in {out}. "
              "Use --redo to go through them again.")
        return
    if len(todo) < len(targets):
        print(f"Resuming: {len(targets) - len(todo)} of {len(targets)} frames are already labelled.")
    boxes = pd.read_csv(args.boxes).set_index("frame") if args.boxes else None

    source = FrameSource(args.input_video, display_width=width)  # full resolution
    done = 0
    try:
        for number in todo:
            image = cv2.cvtColor(source.get(number), cv2.COLOR_RGB2BGR)
            if boxes is not None:
                view = box_view(boxes.loc[number] if number in boxes.index else None, width, height)
            else:
                view = follow_view(labels, number, width, height)
            existing = {name: point for name, point in labels.get(number, {}).items()
                        if name in names}
            placed = pointpicker.label_keypoints(
                image, names, existing=existing, view=view,
                title=f"Frame {number}  ({done + 1} of {len(todo)} to go).  Label the swimmer "
                      "you're analysing; 'a' if they're not in view.",
            )
            if placed is None:
                break
            points = dict(labels.get(number, {}))
            for name in names:  # a point cleared with 'u' is removed, not kept from before
                if name in placed:
                    points[name] = placed[name]
                else:
                    points.pop(name, None)
            if points:
                labels[number] = points
            else:
                labels.pop(number, None)
            write_labels(out, labels, width, height)  # after every frame, so stopping loses nothing
            done += 1
            seen = sum(1 for name in names if points.get(name) is not None)
            print(f"frame {number}: {seen}/{len(names)} placed")
    finally:
        source.close()
    left = len(todo) - done
    print(f"Labelled {done} frame(s); saved to: {out}"
          + (f"\n{left} left. Run the same command again to carry on." if left else ""))


def _run_evaluate(args) -> None:
    import pandas as pd

    from .analysis import kinematics
    from .frames import fps as video_fps
    from .frames import frame_size
    from .groundtruth import format_summary, labels_path, read_labels, score_tracking

    labels_csv = args.labels_csv or labels_path(args.input_video)
    if not os.path.exists(labels_csv):
        raise SystemExit(f"No labels at {labels_csv}. Make them with: "
                         f"python -m analysis label {args.input_video} --side left|right")
    if os.path.getmtime(args.input_video) > os.path.getmtime(labels_csv):
        print(f"Warning: {args.input_video} changed after the labels were made. If it was "
              "re-normalized with a different level, the labels no longer line up.")
    width, height = frame_size(args.input_video)
    rate = video_fps(args.input_video)
    labels = read_labels(labels_csv, width, height)
    boxes = pd.read_csv(args.boxes_csv)
    table, summary = score_tracking(boxes, labels, rate, kinematics(boxes, rate), reach=args.reach)
    print(format_summary(summary))
    if args.out:
        _ensure_parent(args.out)
        table.to_csv(args.out, index=False)
        print(f"Frame-by-frame comparison saved to: {args.out}")


def _load_calibration(path: Optional[str], video: Optional[str]):
    """The calibration given, or else the clip's own at data/calibrations/<clip>.json."""
    from .calibration import Calibration, find_calibration

    if path is None:
        path = find_calibration(video)
        if path is not None:
            print(f"Using the clip's calibration: {path}")
    return Calibration.load(path) if path else None


def _run_analyze(args) -> None:
    import json

    import pandas as pd

    from .analysis import kinematics, summarize
    from .frames import fps as video_fps

    _ensure_parent(args.out_csv)
    if args.fps is None and args.video is None:
        raise SystemExit("Provide --fps or --video so the frame rate is known.")
    fps = args.fps if args.fps is not None else video_fps(args.video)

    calibration = _load_calibration(args.calibration, args.video)

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
        from .normalize import info_path, median_frame, normalize_video, roll_from_points

        if args.ignore_top is not None and not 0 < args.ignore_top < 1:
            raise SystemExit("--ignore-top is a fraction of the frame height, between 0 and 1.")
        _ensure_parent(args.output_video)
        level_line = None
        if args.level:
            from .pointpicker import label_keypoints

            names = ["a point near the left edge", "a point near the right edge"]
            print("Building a still to click on...")
            placed = label_keypoints(
                median_frame(args.input_video), names,
                title="Click a line that's horizontal in the pool, e.g. the lane rope, near the "
                      "left and right edges, about equally far from the middle\n(the lens bends "
                      "lines away from the centre; points either side cancel it). Scroll to zoom.",
            )
            if not placed or any(placed.get(name) is None for name in names):
                raise SystemExit("Levelling needs both points; nothing was written.")
            level_line = [list(placed[name]) for name in names]
            print(f"Camera roll from your line: {roll_from_points(*level_line):+.2f} degrees.")
        print("Normalizing (4K takes a few minutes)...")
        info = normalize_video(args.input_video, args.output_video, rotate=args.rotate,
                               level_line=level_line, tone_map=not args.no_tone_map,
                               ignore_top=args.ignore_top, width=args.width)
        print(f"Normalized video saved to: {args.output_video} ({info.width}x{info.height})")
        if info.tone_mapped:
            print("Converted from HLG HDR (BT.2020) to standard dynamic range (BT.709).")
        if info.rotation_deg:
            print(f"Rotated {info.rotation_deg:+.2f} degrees counter-clockwise to level it.")
        if info.ignore_above:
            print(f"Later stages will ignore rows above {info.ignore_above}.")
        print(f"Details saved to: {info_path(args.output_video)}")
        print("Use this exact file for every later stage so frame numbering stays consistent.")

    elif args.command == "track":
        _run_track(args)

    elif args.command == "calibrate":
        _run_calibrate(args)

    elif args.command == "review":
        from .association import keyframes_path
        from .review import run_review

        calibration = _load_calibration(args.calibration, args.input_video)
        key_file = args.keyframes or keyframes_path(args.input_video)
        saved = run_review(args.input_video, args.boxes_csv, key_file, calibration,
                           args.display_width, args.every)
        print(f"Saved keyframes to {key_file} and re-tracked boxes to {args.boxes_csv}."
              if saved else "Closed without saving.")

    elif args.command == "label":
        _run_label(args)

    elif args.command == "evaluate-track":
        _run_evaluate(args)

    elif args.command == "analyze":
        _run_analyze(args)

    elif args.command == "overlay":
        import pandas as pd

        from .overlay import draw_calibration_still, draw_overlay

        calibration = _load_calibration(args.calibration, args.input_video)
        if calibration is None and args.boxes is None:
            raise SystemExit("Give --calibration, --boxes, or both (the clip has no calibration "
                             "at data/calibrations/ yet).")
        _ensure_parent(args.out_video)
        if args.still:
            if calibration is None:
                raise SystemExit("--still draws a calibration's lines; pass --calibration.")
            if not args.out_video.lower().endswith((".png", ".jpg", ".jpeg")):
                raise SystemExit("--still writes a picture: give an output path ending .png")
            from .tracking import calibration_reference

            # Rebuilt exactly as calibrate built it, so the marks show under
            # their rings -- including markers only down for calibration.frames.
            reference, _ = calibration_reference(
                args.input_video, calibration.frames, calibration.reference == "stabilized",
                args.max_samples, progress=True, ignore_above=_ignored_rows(args.input_video),
            )
            draw_calibration_still(reference, calibration, args.out_video, every=args.every)
            return
        draw_overlay(
            args.input_video,
            args.out_video,
            boxes=pd.read_csv(args.boxes) if args.boxes else None,
            calibration=calibration,
            every=args.every,
        )

    elif args.command == "export":
        import pandas as pd

        from .export import export_video

        _ensure_parent(args.out_video)
        export_video(args.input_video, pd.read_csv(args.boxes_csv), args.out_video, mode=args.mode,
                     width=args.width, zoom=args.zoom, hud=not args.no_hud,
                     calibration=_load_calibration(args.calibration, args.input_video))

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
