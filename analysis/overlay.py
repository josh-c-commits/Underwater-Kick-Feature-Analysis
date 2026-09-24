"""Draw tracking and calibration results back onto the video.

The point is verification, not decoration. Two questions that are hard to
answer from a CSV become obvious in a few seconds of video:

  * Did tracking follow the right swimmer? -- the box, centroid and leading
    edge are drawn on every frame, with LOST where the tracker let go.
  * Is the calibration right? -- lines of constant distance along the pool
    are drawn where the calibration puts them. If the 15m line lands on the
    pool's actual 15m marking, the calibration agrees with reality; if it
    doesn't, you can see by how much and where.

Everything is drawn in *frame* coordinates. Calibration lines are defined on
the (possibly stabilized) background plate, so on a moving camera they are
shifted by that frame's camera offset -- which makes them stay put on the
pool as the camera drifts, and is itself a visible check on stabilization.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .analysis import camera_path, direction_of_travel
from .frames import fps as video_fps
from .frames import frame_size, iter_frames

_LINE = (0, 220, 255)       # distance lines (BGR: amber)
_MAJOR = (0, 140, 255)      # every 5m (orange)
_BOX = (0, 255, 0)
_CENTROID = (0, 0, 255)
_LEAD = (255, 0, 255)


def _dashed(image, start, end, colour, thickness, dash=12, gap=8) -> None:
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b - a))
    if length < 1.0:
        return
    step = (b - a) / length
    position = 0.0
    while position < length:
        p = a + step * position
        q = a + step * min(position + dash, length)
        cv2.line(image, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)),
                 colour, thickness, cv2.LINE_AA)
        position += dash + gap


def _distance_lines(calibration, height: int, every: float):
    """Precompute isolines once: they're fixed in plate coordinates."""
    low, high = calibration.world_range()
    count = int(np.floor(high / every + 1e-9) - np.ceil(low / every - 1e-9)) + 1
    print(f"Calibrated range is {low:g}m to {high:g}m, and lines are only drawn inside it "
          f"(every {every:g}m -> {max(count, 0)} lines). Click marks further out to extend it.")
    start = np.ceil(low / every - 1e-9) * every
    lines = []
    for distance in np.arange(start, high + 1e-9, every):
        segments = calibration.isoline(float(distance), height)
        if segments:
            lines.append((float(distance), segments))
    return lines


def _draw_distance_lines(image, lines, shift=(0.0, 0.0)) -> None:
    sx, sy = shift
    for distance, segments in lines:
        major = abs(distance / 5.0 - round(distance / 5.0)) < 1e-6
        colour = _MAJOR if major else _LINE
        thickness = 2 if major else 1
        for (x0, y0), (x1, y1), solid in segments:
            start, end = (x0 + sx, y0 + sy), (x1 + sx, y1 + sy)
            if solid:
                cv2.line(image, tuple(np.round(start).astype(int)),
                         tuple(np.round(end).astype(int)), colour, thickness, cv2.LINE_AA)
            else:
                _dashed(image, start, end, colour, thickness)
        label_x = int(round(segments[0][0][0] + sx)) + 4
        cv2.putText(image, f"{distance:g}m", (label_x, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def _draw_knots(image, calibration) -> None:
    """Mark every clicked reference point with its distance. A correct line
    passes through each mark on its reference line, so this doubles as a check
    on the clicks themselves."""
    for line in calibration.lines:
        for x, y, world in line.sorted_knots():
            centre = (int(round(x)), int(round(y)))
            cv2.circle(image, centre, 7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(image, centre, 2, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(image, f"{world:g}", (centre[0] + 9, centre[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_calibration_still(reference_bgr, calibration, out_path: str, every: float = 1.0) -> None:
    """
    The calibration's distance lines and clicked marks, drawn on a still
    reference image -- the median (or stabilized) plate the marks were clicked
    on, free of swimmers and ripple. The quickest way to see whether the lines
    land on the pool's real markings.
    """
    image = reference_bgr.copy()
    _draw_distance_lines(image, _distance_lines(calibration, image.shape[0], every))
    _draw_knots(image, calibration)
    cv2.imwrite(out_path, image)
    print(f"Calibration still saved to: {os.path.abspath(out_path)}")


def draw_overlay(
    video_path: str,
    out_video: str,
    boxes: Optional[pd.DataFrame] = None,
    calibration=None,
    every: float = 1.0,
) -> None:
    """
    Write a copy of the video with tracking and/or calibration drawn on it.

    Distance lines are solid where the calibration interpolates between
    reference lines and dashed where it's holding the nearest line's value
    (see Calibration.isoline) -- so the dashed stretches are exactly where
    readings deserve less trust.
    """
    if boxes is None and calibration is None:
        raise ValueError("Nothing to draw: pass boxes, a calibration, or both.")

    width, height = frame_size(video_path)
    writer = cv2.VideoWriter(
        out_video, cv2.VideoWriter_fourcc(*"mp4v"), video_fps(video_path), (width, height)
    )

    lines = _distance_lines(calibration, height, every) if calibration is not None else []

    rows: Dict[int, pd.Series] = {}
    shifts: Dict[int, Tuple[float, float]] = {}
    lead_column = None
    if boxes is not None:
        dx, dy = camera_path(boxes)
        for (_, row), sx, sy in zip(boxes.iterrows(), dx, dy):
            rows[int(row["frame"])] = row
            shifts[int(row["frame"])] = (float(sx), float(sy))
        if {"edge_left", "edge_right"} <= set(boxes.columns):
            direction = direction_of_travel(boxes["centroid_x"].to_numpy(float) - dx)
            lead_column = {1: "edge_right", -1: "edge_left"}.get(direction)

    try:
        for number, frame in iter_frames(video_path):
            sx, sy = shifts.get(number, (0.0, 0.0))

            _draw_distance_lines(frame, lines, (sx, sy))

            row = rows.get(number)
            if row is not None:
                if bool(row["found"]):
                    x, y = int(row["box_x"]), int(row["box_y"])
                    w, h = int(row["box_w"]), int(row["box_h"])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), _BOX, 2)
                    cx, cy = float(row["centroid_x"]), float(row["centroid_y"])
                    cv2.circle(frame, (int(round(cx)), int(round(cy))), 4, _CENTROID, -1)
                    if lead_column is not None and pd.notna(row.get(lead_column)):
                        lx = int(round(float(row[lead_column])))
                        cv2.line(frame, (lx, y), (lx, y + h), _LEAD, 2)
                    if calibration is not None:
                        world = calibration.world_x(cx - sx, cy - sy)
                        text = f"{world:.2f} m" if np.isfinite(world) else "outside calibration"
                        cv2.putText(frame, text, (x, max(14, y - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX, 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "LOST", (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Overlay video saved to: {os.path.abspath(out_video)}")
