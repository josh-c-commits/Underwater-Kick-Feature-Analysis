"""video + csv -> annotated video.

No MediaPipe needed here -- this just draws landmark positions that
are already sitting in a CSV onto the matching frames of an unedited
video. Frames are matched to CSV rows by the 'frame' column.
"""

from __future__ import annotations

import os

import cv2
import pandas as pd

from .landmarks import (
    NUM_LANDMARKS,
    POSE_CONNECTIONS,
    DEFAULT_VISIBILITY_THRESHOLD,
    detect_column_style,
    landmark_columns,
)


def _row_to_points(row, style, width, height, threshold):
    """dict[landmark_index] -> (x_px, y_px, visibility) for visible landmarks in this row."""
    points = {}
    for i in range(NUM_LANDMARKS):
        x_col, y_col, _, vis_col = landmark_columns(i, style)
        x, y, vis = row.get(x_col), row.get(y_col), row.get(vis_col)
        if pd.isna(x) or pd.isna(y) or pd.isna(vis):
            continue
        vis = float(vis)
        if vis < threshold:
            continue
        points[i] = (int(float(x) * width), int(float(y) * height), vis)
    return points


def draw_landmarks(frame, points):
    annotated = frame.copy()
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx not in points or end_idx not in points:
            continue
        x1, y1, _ = points[start_idx]
        x2, y2, _ = points[end_idx]
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x, y, _ in points.values():
        cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(annotated, (x, y), 4, (255, 255, 255), 1)
    return annotated


def annotate(
    input_video: str,
    csv_path: str,
    output_video: str,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
) -> None:
    """
    Draw the landmark positions from csv_path onto input_video (an
    unedited video) and write the result to output_video.
    """
    df = pd.read_csv(csv_path)
    style = detect_column_style(list(df.columns))
    df = df.set_index("frame")

    vid_capture = cv2.VideoCapture(input_video)
    if not vid_capture.isOpened():
        raise RuntimeError(f"Could not open {input_video} with OpenCV.")

    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0

    success, frame = vid_capture.read()
    if not success:
        raise RuntimeError("Could not read first frame from input video.")
    height, width = frame.shape[:2]

    out = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open. Check codec availability or output path.")

    frame_count = 0
    try:
        while True:
            frame_count += 1
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

            points = {}
            if frame_count in df.index:
                row = df.loc[frame_count]
                points = _row_to_points(row, style, width, height, visibility_threshold)

            out.write(draw_landmarks(frame, points))

            success, frame = vid_capture.read()
            if not success:
                break
    finally:
        vid_capture.release()
        out.release()

    print(f"Annotated video saved to: {os.path.abspath(output_video)}")
