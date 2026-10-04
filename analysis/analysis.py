"""Kinematics from tracked boxes and landmarks.

Deliberately split by what each metric needs, because a lot of the interesting
biomechanics turns out not to need pool calibration at all:

  * kick frequency (Hz)                  -- needs only fps
  * velocity fluctuation index           -- dimensionless, cancels any constant
                                            scale error entirely
  * amplitude and velocity in body lengths -- needs a body-length estimate, not
                                            a metre calibration
  * distance per kick, absolute velocity -- needs calibration.py

That matters practically: a swimmer's lateral position in the lane biases the
pixels-per-metre scale by 10-25% depending on camera standoff, but if that
position is roughly constant through a swim the bias is a constant multiplier,
which divides straight out of any ratio. Prefer the dimensionless and
body-length-normalized forms wherever the question allows it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .landmarks import LANDMARK_NAMES, world_columns


def smooth(values: Sequence[float], window: int = 11, polyorder: int = 2) -> np.ndarray:
    """
    Savitzky-Golay smoothing that tolerates gaps.

    A polynomial-fit filter is the right family here: it preserves the peaks
    and troughs of an oscillation, where a moving average would flatten exactly
    the kick extremes the amplitude measurements depend on. NaNs are bridged
    before filtering and restored afterwards, so lost tracking frames don't
    poison their neighbours.
    """
    series = pd.Series(np.asarray(values, dtype=float))
    missing = series.isna()
    if missing.all():
        return series.to_numpy()

    filled = series.interpolate(limit_direction="both").to_numpy()
    usable = min(len(filled), window if window % 2 == 1 else window + 1)
    if usable < 3 or usable <= polyorder:
        return filled
    smoothed = savgol_filter(filled, usable, polyorder)
    smoothed[missing.to_numpy()] = np.nan
    return smoothed


def derivative(values: Sequence[float], fps: float) -> np.ndarray:
    """Per-second rate of change, by central differences."""
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return np.full(len(array), np.nan)
    return np.gradient(array) * fps


def velocity_fluctuation_index(velocity: Sequence[float]) -> float:
    """
    (max - min) / mean over the series: how much speed is lost and regained
    within the stroke cycle.

    Dimensionless on purpose, so it is immune to every scale error in the
    pipeline. Fast underwaters are characterised as much by *not decelerating*
    between kicks as by peak speed, which makes this one of the few numbers
    worth comparing across swimmers filmed on different days and setups.
    """
    array = np.asarray(velocity, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    mean = array.mean()
    if np.isclose(mean, 0.0):
        return float("nan")
    return float((array.max() - array.min()) / abs(mean))


def dominant_frequency(
    values: Sequence[float],
    fps: float,
    min_hz: float = 0.5,
    max_hz: float = 8.0,
) -> float:
    """
    Strongest oscillation frequency (Hz) in a detrended signal, via FFT.

    Band-limited because the interesting content sits between roughly 1 and 4Hz
    for dolphin kick: below that is the swimmer's overall progress down the
    pool, above it is tracking jitter.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 8:
        return float("nan")

    # Remove a fitted line, not just the mean. A swimmer travelling down the
    # pool contributes a large linear ramp, and a ramp leaks across the whole
    # spectrum -- enough to outrank the kick even inside the band limit.
    index = np.arange(array.size, dtype=float)
    slope, intercept = np.polyfit(index, array, 1)
    array = array - (slope * index + intercept)
    windowed = array * np.hanning(array.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(array.size, d=1.0 / fps)

    band = (freqs >= min_hz) & (freqs <= max_hz)
    if not band.any():
        return float("nan")
    return float(freqs[band][int(np.argmax(spectrum[band]))])


def kick_frequency(
    times: Sequence[float],
    values: Sequence[float],
    min_hz: float = 0.5,
    max_hz: float = 8.0,
) -> float:
    """
    Strongest oscillation frequency (Hz) in samples taken at the given times,
    by Lomb-Scargle periodogram.

    Unlike an FFT, this needs no even spacing, so missing frames are simply left
    out. Dropping them and running an FFT on what remains splices the stretches
    either side of each gap together with a jump in phase, and the FFT's bins
    sit ~0.1 Hz apart for a 10-second clip anyway. Simulated at 2 Hz with up to
    three gaps, that combination missed by a median 0.04 Hz (worst 0.25 Hz);
    this method by 0.001 Hz (worst 0.004 Hz). A fitted line is removed first,
    for the same reason dominant_frequency does.
    """
    from scipy.signal import lombscargle

    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    if t.size < 8 or np.ptp(t) <= 0:
        return float("nan")
    v = v - np.polyval(np.polyfit(t, v, 1), t)
    freqs = np.linspace(min_hz, max_hz, 1500)
    power = lombscargle(t, v, 2 * np.pi * freqs)
    return float(freqs[int(np.argmax(power))])


def _short_gaps(observed: np.ndarray, limit: int) -> np.ndarray:
    """Frames in runs of unobserved frames no longer than `limit`, lying between
    two observed frames."""
    mask = np.zeros(len(observed), bool)
    seen = np.flatnonzero(observed)
    if len(seen) < 2:
        return mask
    i = seen[0]
    while i <= seen[-1]:
        if observed[i]:
            i += 1
            continue
        j = i
        while j <= seen[-1] and not observed[j]:
            j += 1
        if j - i <= limit:
            mask[i:j] = True
        i = j
    return mask


def camera_path(boxes: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-frame camera offset (dx, dy) from the background plate, as arrays to
    subtract from frame coordinates to get pool-fixed (plate) coordinates.

    Zero when the table has no camera columns (older runs, or tracking without
    --stabilize, where the camera is taken to be fixed). Frames whose estimate
    was too weak to trust are NaN in the table and are interpolated from their
    neighbours here: the camera was somewhere on those frames, and its path is
    far smoother than any single noisy estimate of it.
    """
    count = len(boxes)
    if "cam_dx" not in boxes.columns or "cam_dy" not in boxes.columns:
        return np.zeros(count), np.zeros(count)

    def fill(column: str) -> np.ndarray:
        series = pd.Series(boxes[column].to_numpy(dtype=float))
        return series.interpolate(limit_direction="both").fillna(0.0).to_numpy()

    return fill("cam_dx"), fill("cam_dy")


def direction_of_travel(x_positions: Sequence[float]) -> int:
    """
    +1 if the swimmer moves toward increasing image x, -1 if decreasing,
    0 if there's too little data to say. Uses a fitted slope over every
    tracked frame rather than first-vs-last position, so a noisy frame at
    either end can't flip the answer.
    """
    array = np.asarray(x_positions, dtype=float)
    index = np.arange(array.size, dtype=float)
    mask = np.isfinite(array)
    if mask.sum() < 2:
        return 0
    slope = np.polyfit(index[mask], array[mask], 1)[0]
    return int(np.sign(slope))


def body_length_series(landmarks: pd.DataFrame, chain: Optional[Sequence[str]] = None) -> np.ndarray:
    """
    Per-frame body length in metres, summed along a chain of world landmarks.

    Uses pose_world_landmarks, which are metric and hip-centred, so this is
    independent of where the swimmer is in frame -- which is exactly what makes
    it a validation signal. A correct model gives a roughly constant length as
    the swimmer crosses the frame; systematic drift with image position is
    evidence of lens distortion or of the model failing out of distribution.
    """
    chain = list(chain or ["left_shoulder", "left_hip", "left_knee", "left_ankle"])
    indices = [LANDMARK_NAMES.index(name) for name in chain]

    columns = []
    for index in indices:
        cols = world_columns(index)
        if not all(col in landmarks.columns for col in cols):
            raise ValueError(
                "Landmark CSV has no world-landmark columns; re-run extract with "
                "world landmarks enabled."
            )
        columns.append(cols)

    total = np.zeros(len(landmarks), dtype=float)
    for first, second in zip(columns, columns[1:]):
        a = landmarks[list(first)].to_numpy(dtype=float)
        b = landmarks[list(second)].to_numpy(dtype=float)
        total += np.linalg.norm(b - a, axis=1)
    total[total == 0] = np.nan
    return total


def kinematics(
    boxes: pd.DataFrame,
    fps: float,
    calibration=None,
    smooth_window: int = 11,
    fill_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Per-frame position and speed from a tracking table.

    centroid_x/centroid_y are copied through in raw *frame* coordinates -- where
    the swimmer appeared in the image. Everything derived (the smoothed
    positions, speeds, world distances) is in *plate* coordinates instead:
    frame coordinates minus the camera's offset. On a fixed camera the two are
    identical. On a drifting one, frame coordinates contain the camera's motion
    as well as the swimmer's, so measuring speed or mapping to metres from them
    would silently add the drift to the result -- and the calibration was built
    on the plate, so plate coordinates are the only ones it maps correctly.

    Position comes from the blob centroid rather than any single landmark: it's
    a far more stable centre-of-mass proxy than a wrist or ankle. When the table
    has edge columns, the leading edge is added too -- whichever edge faces the
    direction of travel.

    Gaps of up to `fill_seconds` -- lost frames, and frames where another
    swimmer merged into the blob and biased its centroid -- are filled by a
    Kalman smoother (association.smooth_track) and flagged `filled`. They count
    toward speed, which needs a continuous position, but not toward kick
    frequency, which they could only fake. 0 turns filling off.
    """
    result = pd.DataFrame({"frame": boxes["frame"].to_numpy()})
    x = boxes["centroid_x"].to_numpy(dtype=float)
    y = boxes["centroid_y"].to_numpy(dtype=float)
    cam_dx, cam_dy = camera_path(boxes)

    result["centroid_x"] = x
    result["centroid_y"] = y
    if "cam_dx" in boxes.columns:
        result["cam_dx"] = cam_dx
        result["cam_dy"] = cam_dy
    plate_x, plate_y = x - cam_dx, y - cam_dy
    filled = np.zeros(len(boxes), bool)
    if fill_seconds > 0:
        from .association import smooth_track

        found = (boxes["found"].fillna(False).astype(bool).to_numpy() if "found" in boxes
                 else np.isfinite(x))
        merged = (boxes["merged"].fillna(False).astype(bool).to_numpy() if "merged" in boxes
                  else np.zeros(len(boxes), bool))
        observed = found & ~merged & np.isfinite(plate_x) & np.isfinite(plate_y)
        if observed.sum() >= 2:
            widths = boxes["box_w"].to_numpy(dtype=float)[observed] if "box_w" in boxes else []
            length = float(np.nanmedian(widths)) if np.isfinite(widths).any() else 50.0
            xs, ys, _, _ = smooth_track(plate_x, plate_y, observed, fps, length)
            filled = _short_gaps(observed, int(round(fill_seconds * fps))) & np.isfinite(xs)
            plate_x = np.where(filled, xs, np.where(observed, plate_x, np.nan))
            plate_y = np.where(filled, ys, np.where(observed, plate_y, np.nan))
    result["filled"] = filled
    result["x_smooth"] = smooth(plate_x, smooth_window)
    result["y_smooth"] = smooth(plate_y, smooth_window)
    result["speed_px_s"] = derivative(result["x_smooth"], fps)
    result["accel_px_s2"] = derivative(smooth(result["speed_px_s"], smooth_window), fps)

    direction = direction_of_travel(result["x_smooth"])
    has_edges = {"edge_left", "edge_right"} <= set(boxes.columns)
    if has_edges and direction != 0:
        edge = boxes["edge_right" if direction > 0 else "edge_left"].to_numpy(dtype=float)
        result["lead_x_smooth"] = smooth(edge - cam_dx, smooth_window)

    if calibration is not None:
        from .calibration import series_world_x

        world = series_world_x(calibration, result["x_smooth"], result["y_smooth"])
        result["world_x_m"] = world
        result["speed_m_s"] = derivative(smooth(world, smooth_window), fps)
        result["accel_m_s2"] = derivative(smooth(result["speed_m_s"], smooth_window), fps)
        if getattr(calibration, "lens", None) is not None:
            if calibration.surface:
                # depth below the surface of the tracked centre, on the plate
                result["depth_m"] = calibration.depth(result["x_smooth"].to_numpy(float),
                                                      result["y_smooth"].to_numpy(float))
        else:
            result["depth_ambiguity_m"] = [
                calibration.depth_ambiguity(float(v)) if np.isfinite(v) else np.nan
                for v in result["x_smooth"]
            ]
        if "lead_x_smooth" in result.columns:
            result["lead_world_x_m"] = series_world_x(
                calibration, result["lead_x_smooth"], result["y_smooth"]
            )
    return result


def summarize(kinematics_table: pd.DataFrame, fps: float) -> dict:
    """Headline numbers for one swim."""
    speed_column = "speed_m_s" if "speed_m_s" in kinematics_table else "speed_px_s"
    # The direction of whatever the speed was measured along: image x for px/s,
    # distance from the wall for m/s, which can run the other way across the frame.
    direction = direction_of_travel(kinematics_table["world_x_m" if speed_column == "speed_m_s"
                                                      else "x_smooth"])
    # Speed is signed by image direction, so a swimmer going right-to-left has
    # negative speeds -- and max() of those is their *slowest* moment, not
    # their fastest. Flipping to "speed in the direction of travel" first keeps
    # mean and peak meaning the same thing whichever way the swimmer goes.
    speed = kinematics_table[speed_column].to_numpy(dtype=float) * (direction or 1)
    vertical = kinematics_table["y_smooth"].to_numpy(dtype=float)
    frames = kinematics_table["frame"].to_numpy(dtype=float)
    # filled frames are interpolated, so they can't count toward the kick's rhythm
    observed = ~kinematics_table["filled"].to_numpy(bool) if "filled" in kinematics_table \
        else np.ones(len(vertical), bool)

    finite = speed[np.isfinite(speed)]
    kick_hz = kick_frequency(frames[observed] / fps, vertical[observed])
    mean_speed = float(np.mean(finite)) if finite.size else float("nan")

    summary = {
        "frames": int(len(kinematics_table)),
        "tracked_frames": int(np.isfinite(kinematics_table["centroid_x"]).sum()),
        "filled_frames": int(kinematics_table["filled"].sum()) if "filled" in kinematics_table else 0,
        "speed_units": "m/s" if speed_column == "speed_m_s" else "px/s",
        "direction": {1: "left-to-right", -1: "right-to-left"}.get(
            direction_of_travel(kinematics_table["x_smooth"]), "unknown"),
        "mean_speed": mean_speed,
        "peak_speed": float(np.max(finite)) if finite.size else float("nan"),
        "velocity_fluctuation_index": velocity_fluctuation_index(speed),
        "kick_frequency_hz": kick_hz,
    }
    if np.isfinite(kick_hz) and kick_hz > 0 and np.isfinite(mean_speed):
        summary["distance_per_kick"] = abs(mean_speed) / kick_hz
    if "depth_m" in kinematics_table and np.isfinite(kinematics_table["depth_m"]).any():
        depth = kinematics_table.loc[observed, "depth_m"].to_numpy(float)
        depth = depth[np.isfinite(depth)]
        if depth.size:
            summary["mean_depth_m"] = float(np.mean(depth))
    if "world_x_m" in kinematics_table and np.isfinite(kinematics_table["world_x_m"]).any():
        world = kinematics_table["world_x_m"].to_numpy(float)
        summary["distance_from_wall_m"] = [float(np.nanmin(world)), float(np.nanmax(world))]
    return summary
