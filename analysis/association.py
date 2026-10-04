"""Which detected blob is the swimmer, frame by frame.

Detection (tracking.detect_candidates) finds every blob that differs from the
background, in every frame. This module decides which of them, if any, is the
swimmer. Splitting the two means the slow part, reading the video, happens
once, while this part can be rerun instantly, e.g. after a keyframe is added.

Keyframes
---------
The user marks the swimmer on any frames: where they first appear, and wherever
tracking went wrong. Tracking runs forwards and backwards from every keyframe,
so one click at frame 150 covers frames 1-149 as well as what follows. A
keyframe can also mark the swimmer absent (out of view), which stops tracking
from inventing them there.

Motion model
------------
A Kalman filter keeps a best estimate of the swimmer's position and velocity,
with its uncertainty. Each frame, candidates are judged by how surprising they
would be under that estimate (squared Mahalanobis distance) and by how well
their size matches the swimmer's. While no candidate is plausible the estimate
coasts, its uncertainty grows, and so does the region searched -- but never
past what the swimmer could reach without suddenly going much faster than they
were. Nothing plausible means the frame is reported lost rather than guessed:
a wrong blob corrupts every measurement downstream, a missing one doesn't.

All motion settings are in body lengths (the swimmer's blob length at the
keyframe), so the same settings work at 1080p or 4K, near the camera or far.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GATE_CHI2 = 9.21  # 99% of a 2-D Gaussian lies within this squared Mahalanobis distance
# A blob this much bigger and taller than the swimmer usually alone is two people
MERGED_AREA, MERGED_HEIGHT = 1.8, 1.5
# Pieces of one swimmer that detection split apart are rejoined (see _with_fragments):
FRAGMENT_GAP = 0.2     # the largest gap between pieces, in full body lengths
FRAGMENT_REACH = 1.15  # the rejoined swimmer may be at most this long, in full body lengths
ANCHOR_DETECTIONS = 15  # detections after a keyframe that set the swimmer's expected size


@dataclass
class Keyframe:
    """The swimmer's position on one frame, or None for x/y if they're not visible."""

    frame: int
    x: Optional[float] = None
    y: Optional[float] = None

    @property
    def absent(self) -> bool:
        return self.x is None or self.y is None


def keyframes_path(video_path: str, folder: str = os.path.join("data", "keyframes")) -> str:
    return os.path.join(folder, os.path.splitext(os.path.basename(video_path))[0] + ".json")


def load_keyframes(path: str) -> List[Keyframe]:
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        data = json.load(handle)
    return sorted((Keyframe(int(k["frame"]), k.get("x"), k.get("y")) for k in data["keyframes"]),
                  key=lambda k: k.frame)


def save_keyframes(path: str, keyframes: Sequence[Keyframe], video: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    entries = [{"frame": k.frame} if k.absent else {"frame": k.frame, "x": k.x, "y": k.y}
               for k in sorted(keyframes, key=lambda k: k.frame)]
    with open(path, "w") as handle:
        json.dump({"video": video, "keyframes": entries}, handle, indent=2)


def merge_keyframes(existing: Sequence[Keyframe], new: Sequence[Keyframe]) -> List[Keyframe]:
    """New keyframes replace existing ones on the same frame."""
    by_frame = {k.frame: k for k in existing}
    by_frame.update({k.frame: k for k in new})
    return sorted(by_frame.values(), key=lambda k: k.frame)


@dataclass
class MotionModel:
    """Settings in body lengths (L) and seconds; converted per frame for a clip."""

    accel: Tuple[float, float] = (3.0, 3.0)  # white-noise acceleration, L/s^2 (along, across)
    measurement: float = 0.05                # centroid jitter as the body undulates, L
    initial_speed: float = 3.0               # velocity uncertainty at a keyframe, L/s per axis
    speed_change: float = 1.0                # how much faster than its last speed it may get, L/s
    max_coast: float = 1.5                   # seconds lost before a pass gives up
    area_spread: float = 0.5                 # std of log(area / expected area)


class SwimmerFilter:
    """Constant-velocity Kalman filter over (x, y, vx, vy), in pixels and frames."""

    def __init__(self, x: float, y: float, length: float, fps: float, model: MotionModel):
        self.model = model
        self.per_frame = length / fps        # one L/s, expressed in px per frame
        self.length = length
        v0 = model.initial_speed * self.per_frame
        p0 = model.measurement * length
        self.state = np.array([x, y, 0.0, 0.0])
        self.cov = np.diag([p0 ** 2, p0 ** 2, v0 ** 2, v0 ** 2])
        ax, ay = (a * length / fps ** 2 for a in model.accel)
        g = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]])
        self.process = g @ np.diag([ax ** 2, ay ** 2]) @ g.T
        self.noise = np.eye(2) * (model.measurement * length) ** 2
        self.transition = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.anchor = np.array([x, y])       # last confirmed position
        self.anchor_speed = 0.0              # speed there, px per frame
        self.coasting = 0
        self.seen = False                    # confirmed at least once since the keyframe

    def predict(self) -> None:
        self.state = self.transition @ self.state
        self.cov = self.transition @ self.cov @ self.transition.T + self.process

    def distance2(self, point: Tuple[float, float]) -> float:
        residual = np.asarray(point, float) - self.state[:2]
        innovation = self.cov[:2, :2] + self.noise
        return float(residual @ np.linalg.solve(innovation, residual))

    def reachable(self, point: Tuple[float, float]) -> bool:
        """Within what the swimmer could cover since last seen, going at most a
        little faster than they were."""
        steps = self.coasting + 1
        # Before the first confirmation the speed is unknown, not zero: allow what
        # the filter's own initial velocity uncertainty does.
        speed = self.anchor_speed if self.seen else self.model.initial_speed * self.per_frame
        limit = (speed + self.model.speed_change * self.per_frame) * steps
        limit += 0.25 * self.length
        return float(np.hypot(*(np.asarray(point, float) - self.anchor))) <= limit

    def update(self, point: Tuple[float, float]) -> None:
        innovation = self.cov[:2, :2] + self.noise
        gain = self.cov[:, :2] @ np.linalg.inv(innovation)
        self.state = self.state + gain @ (np.asarray(point, float) - self.state[:2])
        self.cov = (np.eye(4) - gain[:, :2] @ np.eye(2, 4)) @ self.cov
        self.anchor = self.state[:2].copy()
        self.anchor_speed = float(np.hypot(*self.state[2:]))
        self.coasting = 0
        self.seen = True


def _score(filter_: SwimmerFilter, candidate: dict, expected_area: Optional[float],
           model: MotionModel) -> float:
    score = -0.5 * filter_.distance2((candidate["centroid_x"], candidate["centroid_y"]))
    if expected_area:
        score -= 0.5 * (np.log(candidate["area"] / expected_area) / model.area_spread) ** 2
    return score


def _choose(filter_: SwimmerFilter, candidates: List[dict], expected_area: Optional[float],
            area_bounds: Tuple[float, float], model: MotionModel) -> Optional[int]:
    best, best_score = None, -np.inf
    for index, candidate in enumerate(candidates):
        point = (candidate["centroid_x"], candidate["centroid_y"])
        if filter_.distance2(point) > GATE_CHI2 or not filter_.reachable(point):
            continue
        if expected_area:
            ratio = candidate["area"] / expected_area
            if not area_bounds[0] <= ratio <= area_bounds[1]:
                continue
        score = _score(filter_, candidate, expected_area, model)
        if score > best_score:
            best, best_score = index, score
    return best


def _snap(candidates: List[dict], point: Tuple[float, float], radius: float) -> Optional[int]:
    """The candidate a click landed on: the one whose box contains it, else the
    nearest centroid within `radius`."""
    x, y = point
    inside = [i for i, c in enumerate(candidates)
              if c["box_x"] <= x <= c["box_x"] + c["box_w"] and c["box_y"] <= y <= c["box_y"] + c["box_h"]]
    pool = inside or list(range(len(candidates)))
    if not pool:
        return None
    nearest = min(pool, key=lambda i: np.hypot(candidates[i]["centroid_x"] - x,
                                                candidates[i]["centroid_y"] - y))
    if inside:
        return nearest
    gap = np.hypot(candidates[nearest]["centroid_x"] - x, candidates[nearest]["centroid_y"] - y)
    return nearest if gap <= radius else None


def _run_pass(frames: Sequence[int], by_frame: Dict[int, List[dict]], start: Tuple[float, float],
              length: float, expected_area: Optional[float], fps: float, model: MotionModel,
              area_bounds: Tuple[float, float]) -> Dict[int, Tuple[int, Optional[float]]]:
    """Follow the swimmer from `start` through `frames` in the order given.
    Returns {frame: (chosen candidate index, expected area then)} for the frames
    where one was chosen."""
    filter_ = SwimmerFilter(start[0], start[1], length, fps, model)
    chosen: Dict[int, Tuple[int, Optional[float]]] = {}
    early_areas: List[float] = [] if expected_area is None else [expected_area]
    anchored = False
    for frame in frames:
        filter_.predict()
        candidates = by_frame.get(frame, [])
        pick = _choose(filter_, candidates, expected_area, area_bounds, model)
        if pick is None:
            filter_.coasting += 1
            if filter_.coasting > model.max_coast * fps:
                break
            continue
        candidate = candidates[pick]
        filter_.update((candidate["centroid_x"], candidate["centroid_y"]))
        chosen[frame] = (pick, expected_area)
        if not anchored:
            # The expected size is anchored to the keyframe's blob together with the
            # first detections after it: that blob alone can be far off (on one real
            # clip it was twice the usual size, swollen by bubbles, and the swimmer
            # then failed the size check). Never a rolling window, though, which
            # would drift a little with each larger blob until it admitted a
            # different swimmer.
            early_areas.append(candidate["area"])
            if len(early_areas) >= ANCHOR_DETECTIONS:
                expected_area = float(np.median(early_areas))
                anchored = True
    return chosen


def associate(
    candidates: pd.DataFrame,
    frame_count: int,
    fps: float,
    keyframes: Sequence[Keyframe] = (),
    model: Optional[MotionModel] = None,
    max_area_ratio: float = 3.0,
    min_area_ratio: float = 1.0 / 3.0,
    default_length: float = 50.0,
) -> pd.DataFrame:
    """
    One row per frame: the chosen candidate's columns, plus `source` (detected,
    keyframe or lost), `conflict` (the forward and backward passes chose
    different blobs; the pass from the nearer keyframe was kept) and
    `size_ratio` and `merged` (see _flag_merges: another swimmer overlapping
    the subject, pulling the centroid toward them).

    Without keyframes the largest blob on the first frame that has any is taken
    as the swimmer, and followed forwards.
    """
    model = model or MotionModel()
    area_bounds = (min_area_ratio, max_area_ratio)
    by_frame: Dict[int, List[dict]] = {
        int(frame): group.to_dict("records") for frame, group in candidates.groupby("frame")
    } if len(candidates) else {}
    all_frames = list(range(1, frame_count + 1))

    keys = sorted(keyframes, key=lambda k: k.frame)
    if not keys:
        first = next((f for f in all_frames if by_frame.get(f)), None)
        if first is None:
            return _flag_merges(_assemble(all_frames, by_frame, {}, {}, {}, set()), fps)
        start = max(by_frame[first], key=lambda c: c["area"])
        keys = [Keyframe(first, start["centroid_x"], start["centroid_y"])]

    # frame -> (candidate index, keyframe frame, expected area)
    forward: Dict[int, Tuple[int, int, Optional[float]]] = {}
    backward: Dict[int, Tuple[int, int, Optional[float]]] = {}
    manual: Dict[int, Tuple[float, float]] = {}
    absent = {k.frame for k in keys if k.absent}
    for i, key in enumerate(keys):
        if key.absent:
            continue
        here = by_frame.get(key.frame, [])
        length = default_length
        snapped = _snap(here, (key.x, key.y), radius=default_length)
        expected_area = None
        start = (key.x, key.y)
        if snapped is not None:
            blob = here[snapped]
            length = float(max(blob["box_w"], blob["box_h"]))
            expected_area = float(blob["area"])
            start = (blob["centroid_x"], blob["centroid_y"])
            forward[key.frame] = (snapped, key.frame, expected_area)
        else:
            manual[key.frame] = (key.x, key.y)
        nxt = keys[i + 1].frame if i + 1 < len(keys) else frame_count + 1
        prev = keys[i - 1].frame if i > 0 else 0
        for store, frames in ((forward, range(key.frame + 1, nxt)),
                              (backward, range(key.frame - 1, prev, -1))):
            picks = _run_pass(list(frames), by_frame, start, length, expected_area, fps, model,
                              area_bounds)
            for frame, (index, area) in picks.items():
                store[frame] = (index, key.frame, area)
    full_length = _full_length(by_frame, forward, backward)
    return _flag_merges(_assemble(all_frames, by_frame, forward, backward, manual, absent,
                                  full_length), fps)


def _full_length(by_frame, *stores) -> Optional[float]:
    """The swimmer's whole length in px: the upper quartile of the chosen blobs'
    widths, since a body split in pieces measures short and the odd merge long."""
    widths = [by_frame[frame][pick[0]]["box_w"] for store in stores for frame, pick in store.items()]
    return float(np.percentile(widths, 75)) if len(widths) >= 10 else None


def _gap(candidate: dict, x0: float, x1: float) -> float:
    return max(candidate["box_x"] - x1, x0 - (candidate["box_x"] + candidate["box_w"]), 0.0)


def _with_fragments(candidates: List[dict], pick: int, full_length: float) -> List[int]:
    """
    The chosen blob plus any pieces of the same swimmer that detection split off.

    Near the surface the head and arms often come away from the body as a
    separate blob, and following only the bigger piece puts the centre on the
    hips and legs -- measured against hand labels, up to half a body length
    behind the hip. A piece joins when it sits level with the body, within
    FRAGMENT_GAP of it, is no bigger than the chosen blob, and leaves the whole
    no longer than FRAGMENT_REACH full lengths and no taller than half of one,
    which keeps out a neighbouring swimmer.
    """
    main = candidates[pick]
    x0, x1 = main["box_x"], main["box_x"] + main["box_w"]
    y0, y1 = main["box_y"], main["box_y"] + main["box_h"]
    slack = 0.1 * full_length
    group, remaining = [pick], [i for i in range(len(candidates)) if i != pick]
    joined = True
    while joined:
        joined = False
        for i in sorted(remaining, key=lambda i: _gap(candidates[i], x0, x1)):
            c = candidates[i]
            cx0, cx1 = c["box_x"], c["box_x"] + c["box_w"]
            cy0, cy1 = c["box_y"], c["box_y"] + c["box_h"]
            if (_gap(c, x0, x1) <= FRAGMENT_GAP * full_length
                    and min(cy1, y1 + slack) > max(cy0, y0 - slack)
                    and c["area"] <= main["area"]
                    and max(x1, cx1) - min(x0, cx0) <= FRAGMENT_REACH * full_length
                    and max(y1, cy1) - min(y0, cy0) <= 0.5 * full_length):
                group.append(i)
                remaining.remove(i)
                x0, x1, y0, y1 = min(x0, cx0), max(x1, cx1), min(y0, cy0), max(y1, cy1)
                joined = True
                break  # the union grew: look again from the nearest piece
    return group


def _combined(candidates: List[dict], group: List[int]) -> dict:
    """One blob from several: the union box, the area-weighted centroid."""
    members = [candidates[i] for i in group]
    if len(members) == 1:
        return members[0]
    area = float(sum(m["area"] for m in members))
    x0 = min(m["box_x"] for m in members)
    y0 = min(m["box_y"] for m in members)
    x1 = max(m["box_x"] + m["box_w"] for m in members)
    y1 = max(m["box_y"] + m["box_h"] for m in members)
    return {"box_x": x0, "box_y": y0, "box_w": x1 - x0, "box_h": y1 - y0,
            "centroid_x": sum(m["centroid_x"] * m["area"] for m in members) / area,
            "centroid_y": sum(m["centroid_y"] * m["area"] for m in members) / area,
            "area": area,
            "edge_left": min(m["edge_left"] for m in members),
            "edge_right": max(m["edge_right"] for m in members)}


def _assemble(all_frames, by_frame, forward, backward, manual, absent,
              full_length: Optional[float] = None) -> pd.DataFrame:
    rows = []
    for frame in all_frames:
        candidates = by_frame.get(frame, [])
        f, b = forward.get(frame), backward.get(frame)
        conflict = False
        if frame in absent:
            pick = None
        elif f and b and f[0] != b[0]:
            conflict = True
            pick = f if abs(frame - f[1]) <= abs(frame - b[1]) else b
        else:
            pick = f or b
        row = {"frame": frame, "found": False, "source": "lost", "conflict": conflict, "area": 0}
        if pick is not None:
            group = _with_fragments(candidates, pick[0], full_length) if full_length else [pick[0]]
            candidate = _combined(candidates, group)
            row.update({key: candidate[key] for key in (
                "box_x", "box_y", "box_w", "box_h", "centroid_x", "centroid_y",
                "area", "edge_left", "edge_right")})
            row.update({"found": True, "source": "keyframe" if pick[1] == frame else "detected"})
        elif frame in manual:
            x, y = manual[frame]
            row.update({"found": True, "source": "keyframe", "centroid_x": x, "centroid_y": y})
        rows.append(row)
    return pd.DataFrame(rows)


def _flag_merges(table: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    size_ratio and merged: when another swimmer overlaps the subject in the
    image, the two become one blob and the centroid is pulled toward the other.

    The comparison is with the swimmer's lower-quartile size within 3 seconds
    either side: kicking and changing distance alone swing a lone swimmer's
    area by half, and a merge can last long enough to drag a median up with it,
    but it can only push the lower quartile so far. A merge needs the blob to
    be both bigger and taller, which is what two swimmers stacked one above
    the other look like; that separates them far better than size alone.
    """
    if "area" not in table:
        table["size_ratio"], table["merged"] = np.nan, False
        return table
    found = table["found"] & (table["area"] > 0)
    window = int(6 * fps) | 1
    ratios = {}
    for column in ("area", "box_h"):
        series = (pd.to_numeric(table[column], errors="coerce") if column in table
                  else pd.Series(np.nan, index=table.index)).where(found)
        typical = series.rolling(window, center=True, min_periods=15).quantile(0.25)
        ratios[column] = series / typical.fillna(series.quantile(0.25))
    table["size_ratio"] = ratios["area"]
    table["merged"] = ((ratios["area"] >= MERGED_AREA) & (ratios["box_h"] >= MERGED_HEIGHT)).fillna(False)
    return table


def smooth_track(
    x: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    fps: float,
    length: float,
    model: Optional[MotionModel] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (x, y, x_std, y_std) on every frame from the first observation to the last,
    given positions only on `observed` frames: a Kalman filter forwards, then a
    Rauch-Tung-Striebel pass backwards.

    Across a gap the result matches the swimmer's position *and* velocity at both
    ends, and the standard deviations say how far to trust it -- widest in the
    middle of the gap. NaN outside the observed span.
    """
    model = model or MotionModel()
    count = len(x)
    nan = np.full(count, np.nan)
    seen = np.flatnonzero(np.asarray(observed, bool) & np.isfinite(x) & np.isfinite(y))
    if len(seen) < 2:
        return nan.copy(), nan.copy(), nan.copy(), nan.copy()
    start, end = int(seen[0]), int(seen[-1])
    f = SwimmerFilter(float(x[start]), float(y[start]), length, fps, model)
    transition, process, noise = f.transition, f.process, f.noise
    pick = np.eye(2, 4)

    m = np.zeros((count, 4))
    p = np.zeros((count, 4, 4))
    m_pred = np.zeros((count, 4))
    p_pred = np.zeros((count, 4, 4))
    m[start], p[start] = f.state, f.cov
    is_seen = np.zeros(count, bool)
    is_seen[seen] = True
    for t in range(start + 1, end + 1):
        m_pred[t] = transition @ m[t - 1]
        p_pred[t] = transition @ p[t - 1] @ transition.T + process
        if is_seen[t]:
            innovation = pick @ p_pred[t] @ pick.T + noise
            gain = p_pred[t] @ pick.T @ np.linalg.inv(innovation)
            m[t] = m_pred[t] + gain @ (np.array([x[t], y[t]]) - pick @ m_pred[t])
            p[t] = (np.eye(4) - gain @ pick) @ p_pred[t]
        else:
            m[t], p[t] = m_pred[t], p_pred[t]

    smoothed, smoothed_cov = m.copy(), p.copy()
    for t in range(end - 1, start - 1, -1):
        gain = p[t] @ transition.T @ np.linalg.inv(p_pred[t + 1])
        smoothed[t] = m[t] + gain @ (smoothed[t + 1] - m_pred[t + 1])
        smoothed_cov[t] = p[t] + gain @ (smoothed_cov[t + 1] - p_pred[t + 1]) @ gain.T

    xs, ys, sx, sy = nan.copy(), nan.copy(), nan.copy(), nan.copy()
    span = slice(start, end + 1)
    xs[span], ys[span] = smoothed[span, 0], smoothed[span, 1]
    sx[span] = np.sqrt(np.maximum(smoothed_cov[span, 0, 0], 0.0))
    sy[span] = np.sqrt(np.maximum(smoothed_cov[span, 1, 1], 0.0))
    return xs, ys, sx, sy
