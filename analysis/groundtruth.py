"""
Hand-labelled ground truth: reading and writing label files, and scoring the
tracker against them.

Labels use the same CSV layout as MediaPipe's output -- one row per frame,
x/y/z/visibility per landmark, x and y as fractions of the frame -- so the same
file can score pose estimation later. A landmark marked "not visible" is kept
as visibility 0 with no position: that the swimmer was out of sight is a fact
the tracker can be scored on (it shouldn't report a position there), which is
different from a frame nobody looked at.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .landmarks import LANDMARK_NAMES, named_header

Point = Tuple[float, float]
Labels = Dict[int, Dict[str, Optional[Point]]]  # frame -> name -> (x, y) px, or None = not visible

LABELS_FOLDER = os.path.join("data", "labels")

PRESETS = {
    # enough to score the tracker: which swimmer, where, and how far the body reaches
    "track": ["nose", "hip", "foot_index"],
    # the near-side chain, for scoring pose estimation and joint angles later
    "body": ["nose", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "foot_index"],
}


def labels_path(video: str, folder: str = LABELS_FOLDER) -> str:
    """Where a clip's labels live by default: data/labels/<clip>.csv."""
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(folder, stem + ".csv")


def preset_names(preset: str, side: str) -> List[str]:
    """A preset's landmark names for the side of the body facing the camera."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    return [name if name == "nose" else f"{side}_{name}" for name in PRESETS[preset]]


def sample_frames(total: int, count: int) -> List[int]:
    """`count` frames spread evenly from frame 1. The same arguments always give
    the same frames, which is what lets a labelling session resume."""
    step = max(1, total // max(1, count))
    return list(range(1, total + 1, step))[:count]


# ---------- files ----------

def read_labels(path: str, width: int, height: int) -> Labels:
    """{frame: {name: (x, y) in pixels, or None if marked not visible}}.
    Landmarks never labelled on a frame are left out; a missing file is no labels."""
    if not os.path.exists(path):
        return {}
    table = pd.read_csv(path)
    labels: Labels = {}
    for row in table.to_dict("records"):
        points: Dict[str, Optional[Point]] = {}
        for name in LANDMARK_NAMES:
            x, y = row.get(f"{name}_x"), row.get(f"{name}_y")
            visibility = row.get(f"{name}_visibility")
            if pd.notna(x) and pd.notna(y):
                points[name] = (float(x) * width, float(y) * height)
            elif pd.notna(visibility) and float(visibility) == 0.0:
                points[name] = None
        labels[int(row["frame"])] = points
    return labels


def write_labels(path: str, labels: Labels, width: int, height: int) -> None:
    """Write every labelled frame in order. The file is replaced in one step, so
    quitting mid-write can't leave it half-written."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(named_header())
        for frame in sorted(labels):
            row: list = [frame]
            for name in LANDMARK_NAMES:
                if name not in labels[frame]:
                    row += ["", "", "", ""]
                elif labels[frame][name] is None:
                    row += ["", "", "", "0.0"]
                else:
                    x, y = labels[frame][name]
                    row += [f"{x / width:.8f}", f"{y / height:.8f}", "0.0", "1.0"]
            writer.writerow(row)
    os.replace(temporary, path)


# ---------- helping the labeller ----------

def _centre(points: Dict[str, Optional[Point]]) -> Optional[Point]:
    placed = [p for p in points.values() if p is not None]
    if not placed:
        return None
    return (float(np.mean([p[0] for p in placed])), float(np.mean([p[1] for p in placed])))


def follow_view(labels: Labels, frame: int, width: int, height: int,
                span: float = 1 / 3, max_gap: int = 120) -> Optional[Tuple[float, float, float, float]]:
    """
    Where to start the view when labelling `frame`: a window `span` of the frame
    wide around where your own most recent labels put the swimmer, moved on at
    the speed between your two latest. None (the whole frame) when there's
    nothing within `max_gap` frames to go on.

    It follows your clicks rather than the tracker's output on purpose: labels
    that start from the tracker's guess would tend to agree with it.
    """
    earlier = [f for f in sorted(labels) if f < frame and _centre(labels[f]) is not None]
    if not earlier or frame - earlier[-1] > max_gap:
        return None
    last = earlier[-1]
    x, y = _centre(labels[last])
    if len(earlier) >= 2 and last - earlier[-2] <= max_gap:
        px, py = _centre(labels[earlier[-2]])
        steps = (frame - last) / float(last - earlier[-2])
        x, y = x + (x - px) * steps, y + (y - py) * steps
    return _window(x, y, width, height, span)


def box_view(row: Optional[pd.Series], width: int, height: int,
             span: float = 1 / 3) -> Optional[Tuple[float, float, float, float]]:
    """The same window, around the tracker's centre on this frame instead."""
    if row is None or not bool(row.get("found", False)):
        return None
    x, y = float(row["centroid_x"]), float(row["centroid_y"])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return _window(x, y, width, height, span)


def _window(x: float, y: float, width: int, height: int, span: float):
    """(x0, x1, y0, y1) of a frame-shaped window centred on (x, y), kept inside the frame."""
    w = width * span
    h = w * height / width
    x0 = min(max(x - w / 2, 0.0), width - w)
    y0 = min(max(y - h / 2, 0.0), height - h)
    return (x0, x0 + w, y0, y0 + h)


# ---------- scoring the tracker ----------

def _hip_name(labels: Labels) -> Optional[str]:
    counts = {name: sum(name in points for points in labels.values())
              for name in ("left_hip", "right_hip")}
    best = max(counts, key=counts.get)
    return best if counts[best] else None


def body_length(labels: Labels) -> Optional[float]:
    """Median nose-to-toes distance in pixels, over frames where both were placed."""
    lengths = []
    for points in labels.values():
        nose = points.get("nose")
        toes = points.get("left_foot_index") or points.get("right_foot_index")
        if nose is not None and toes is not None:
            lengths.append(math.dist(nose, toes))
    return float(np.median(lengths)) if lengths else None


def _inside(point: Point, row, margin: float) -> bool:
    x0, y0, w, h = (float(row[k]) for k in ("box_x", "box_y", "box_w", "box_h"))
    return (x0 - margin * w <= point[0] <= x0 + (1 + margin) * w
            and y0 - margin * h <= point[1] <= y0 + (1 + margin) * h)


def score_tracking(
    boxes: pd.DataFrame,
    labels: Labels,
    fps: float,
    kinematics_table: Optional[pd.DataFrame] = None,
    reach: float = 0.5,
    box_margin: float = 0.1,
) -> Tuple[pd.DataFrame, dict]:
    """
    Compare a tracking table with labelled hip positions, frame by frame.

    Each labelled frame gets a status: "on target" (the tracker's centre is
    within `reach` body lengths of the hip), "wrong swimmer", "lost", "false
    alarm" (a position reported where you marked the swimmer out of view),
    "correctly empty", or "keyframe" -- frames you clicked while tracking, which
    are left out because they'd only be scoring your own click.

    `kinematics_table` (analysis.kinematics on the same boxes) adds the position
    the speed calculation actually uses, gap-filled and smoothed, so the speed
    error measured here is the one the analysis would make.
    """
    hip = _hip_name(labels)
    if hip is None:
        raise ValueError("The labels have no hip positions to score against.")
    length = body_length(labels)
    if length is None:
        found = boxes["found"].fillna(False).astype(bool)
        widths = boxes.loc[found, "box_w"].dropna()
        length = float(widths.median()) if len(widths) else float("nan")
    by_frame = boxes.set_index("frame")
    used = None
    if kinematics_table is not None:
        kin = kinematics_table.set_index("frame")
        cam_x = kin["cam_dx"] if "cam_dx" in kin else 0.0
        cam_y = kin["cam_dy"] if "cam_dy" in kin else 0.0
        used = pd.DataFrame({"x": kin["x_smooth"] + cam_x, "y": kin["y_smooth"] + cam_y,
                             "filled": kin["filled"]})

    rows = []
    for frame in sorted(labels):
        if hip not in labels[frame]:
            continue
        truth = labels[frame][hip]
        box = by_frame.loc[frame] if frame in by_frame.index else None
        found = box is not None and bool(box["found"])
        cx = float(box["centroid_x"]) if found else float("nan")
        cy = float(box["centroid_y"]) if found else float("nan")
        source = box.get("source") if box is not None else None
        if source == "keyframe":
            status = "keyframe"
        elif truth is None:
            status = "false alarm" if found else "correctly empty"
        elif not found:
            status = "lost"
        else:
            near = math.hypot(cx - truth[0], cy - truth[1]) <= reach * length
            status = "on target" if near else "wrong swimmer"
        merged = box.get("merged") if found else False
        row = {"frame": frame, "status": status,
               "hip_x": truth[0] if truth else np.nan, "hip_y": truth[1] if truth else np.nan,
               "centroid_x": cx, "centroid_y": cy, "merged": bool(merged) if pd.notna(merged) else False}
        if used is not None and frame in used.index:
            row.update(used_x=float(used.at[frame, "x"]), used_y=float(used.at[frame, "y"]),
                       filled=bool(used.at[frame, "filled"]))
        nose = labels[frame].get("nose")
        toes = labels[frame].get("left_foot_index") or labels[frame].get("right_foot_index")
        if (status == "on target" and nose is not None and toes is not None
                and pd.notna(box.get("box_x"))):
            row["box_covers_body"] = _inside(nose, box, box_margin) and _inside(toes, box, box_margin)
        rows.append(row)
    table = pd.DataFrame(rows)
    return table, _summary(table, fps, length, labels)


def _summary(table: pd.DataFrame, fps: float, length: float, labels: Labels) -> dict:
    counts = table["status"].value_counts()
    seen = table[table["status"].isin(["on target", "wrong swimmer", "lost"])]
    unseen = table[table["status"].isin(["false alarm", "correctly empty"])]

    def frames(status):
        return [int(f) for f in table.loc[table["status"] == status, "frame"]]

    summary = {
        "labelled": len(table), "body_length_px": length,
        "visible": len(seen), "not_visible": len(unseen),
        "on_target": int(counts.get("on target", 0)),
        "wrong_swimmer": frames("wrong swimmer"), "lost": frames("lost"),
        "false_alarms": frames("false alarm"), "keyframes": int(counts.get("keyframe", 0)),
    }

    # direction of travel, from your own hip labels
    visible = table.dropna(subset=["hip_x"]).sort_values("frame")
    direction = float(np.sign(visible["hip_x"].iloc[-1] - visible["hip_x"].iloc[0])) if len(visible) > 1 else 1.0
    direction = direction or 1.0
    summary["direction"] = direction

    if "used_x" in table:
        compared = table[table["status"].isin(["on target", "lost", "wrong swimmer"])]
        compared = compared.dropna(subset=["hip_x", "used_x"]).sort_values("frame")
        # don't score positions taken from the wrong swimmer
        compared = compared[compared["status"] != "wrong swimmer"]
    else:
        compared = table[table["status"] == "on target"].sort_values("frame")
        compared = compared.assign(used_x=compared["centroid_x"], used_y=compared["centroid_y"])
        compared = compared[~compared["merged"]]
    if len(compared):
        along = (compared["used_x"] - compared["hip_x"]) * direction
        across = compared["used_y"] - compared["hip_y"]
        summary.update(
            position_frames=len(compared),
            offset_along_px=float(along.median()), offset_down_px=float(across.median()),
            spread_along_px=float(1.4826 * (along - along.median()).abs().median()),
        )
        if "filled" in compared:
            gaps = compared[compared["filled"].fillna(False).astype(bool)]
            if len(gaps):
                summary["filled_frames"] = len(gaps)
                summary["filled_error_px"] = float(np.hypot(gaps["used_x"] - gaps["hip_x"],
                                                             gaps["used_y"] - gaps["hip_y"]).median())
    if len(compared) >= 3:
        t = compared["frame"].to_numpy(dtype=float) / fps
        hip_speed = direction * np.polyfit(t, compared["hip_x"].to_numpy(dtype=float), 1)[0]
        track_speed = direction * np.polyfit(t, compared["used_x"].to_numpy(dtype=float), 1)[0]
        summary.update(hip_speed_px_s=float(hip_speed), track_speed_px_s=float(track_speed))
        dt = np.diff(t)
        v_hip = direction * np.diff(compared["hip_x"].to_numpy(dtype=float)) / dt
        v_track = direction * np.diff(compared["used_x"].to_numpy(dtype=float)) / dt
        close = dt <= 1.0  # pairs more than a second apart say little about speed in between
        if close.any():
            difference = v_track[close] - v_hip[close]
            summary.update(interval_pairs=int(close.sum()),
                           interval_gap_s=float(np.median(dt[close])),
                           interval_rms_px_s=float(np.sqrt(np.mean(difference ** 2))))
    if "box_covers_body" in table:
        covered = table["box_covers_body"].dropna()
        summary.update(box_checked=len(covered), box_covers=int(covered.astype(bool).sum()))
    return summary


def format_summary(summary: dict) -> str:
    """The scoring as a short readable report."""
    lines = []
    length = summary["body_length_px"]
    lines.append(f"{summary['labelled']} labelled frames; body length {length:.0f} px.")
    if summary["visible"]:
        lines.append(f"Right swimmer: {summary['on_target']} of {summary['visible']} frames where you saw "
                     f"the swimmer ({100 * summary['on_target'] / summary['visible']:.0f}%).")
    for key, words in (("wrong_swimmer", "wrong swimmer"), ("lost", "lost"),
                       ("false_alarms", "a position where you marked the swimmer out of view")):
        if summary[key]:
            lines.append(f"  {words}: frame{'s' if len(summary[key]) > 1 else ''} "
                         + ", ".join(str(f) for f in summary[key]))
    if summary["not_visible"] and not summary["false_alarms"]:
        lines.append(f"  no false alarms on the {summary['not_visible']} frames with the swimmer out of view")
    if summary["keyframes"]:
        lines.append(f"  ({summary['keyframes']} frame(s) you clicked as keyframes left out)")
    if "position_frames" in summary:
        ahead = summary["offset_along_px"]
        lines.append(
            f"Position used for speed vs your hip labels ({summary['position_frames']} frames): typically "
            f"{abs(ahead):.0f} px {'ahead of' if ahead >= 0 else 'behind'} the hip and "
            f"{abs(summary['offset_down_px']):.0f} px {'below' if summary['offset_down_px'] >= 0 else 'above'} it; "
            f"varies by +-{summary['spread_along_px']:.0f} px along the swim "
            f"({summary['spread_along_px'] / length:.2f} body lengths).")
    if "filled_frames" in summary:
        lines.append(f"  {summary['filled_frames']} of those fell in filled gaps; filled positions were "
                     f"{summary['filled_error_px']:.0f} px from the hip.")
    if "hip_speed_px_s" in summary:
        hip, track = summary["hip_speed_px_s"], summary["track_speed_px_s"]
        lines.append(f"Mean speed: hip {hip:.0f} px/s, tracker {track:.0f} px/s "
                     f"({100 * (track - hip) / hip:+.1f}%).")
    if "interval_rms_px_s" in summary:
        lines.append(f"Speed between labelled frames ({summary['interval_gap_s']:.2f} s apart): differs by "
                     f"{summary['interval_rms_px_s']:.0f} px/s RMS "
                     f"({100 * summary['interval_rms_px_s'] / abs(summary['hip_speed_px_s']):.0f}% of the mean).")
    if summary.get("box_checked"):
        lines.append(f"Box contains the nose and toes on {summary['box_covers']} of "
                     f"{summary['box_checked']} frames.")
    return "\n".join(lines)
