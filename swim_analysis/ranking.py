"""csv -> ranked table of landmarks by visibility.

rank_by_visibility: average each landmark's visibility over a
duration (frame range or time range) and return landmarks sorted by
that average, most visible first.

(The old rank_by_max_x -- ranking landmarks by raw x-position -- was
replaced by outline.outline_rightmost_x_series, which measures the
SAM2 outline's rightmost pixel directly instead of relying on
MediaPipe landmark positions.)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .landmarks import (
    NUM_LANDMARKS,
    DEFAULT_VISIBILITY_THRESHOLD,
    detect_column_style,
    landmark_columns,
    load_index_to_name_map,
    display_name,
)
from .duration import resolve_frame_bounds


def _select_duration(
    df: pd.DataFrame,
    start_frame: Optional[int],
    end_frame: Optional[int],
    start_time: Optional[float],
    end_time: Optional[float],
    fps: Optional[float],
) -> pd.DataFrame:
    df = df.sort_values("frame")
    start_frame, end_frame = resolve_frame_bounds(start_frame, end_frame, start_time, end_time, fps)
    if start_frame is not None:
        df = df[df["frame"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame"] <= end_frame]
    return df


def rank_by_visibility(
    csv_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    mapping_path: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    style = detect_column_style(list(df.columns))
    index_to_name = load_index_to_name_map(mapping_path)
    df = _select_duration(df, start_frame, end_frame, start_time, end_time, fps)

    rows = []
    for i in range(NUM_LANDMARKS):
        _, _, _, vis_col = landmark_columns(i, style)
        mean_vis = df[vis_col].mean(skipna=True)
        rows.append({"landmark": display_name(i, style, index_to_name), "avg_visibility": mean_vis})

    table = pd.DataFrame(rows)
    return table.sort_values("avg_visibility", ascending=False).reset_index(drop=True)
