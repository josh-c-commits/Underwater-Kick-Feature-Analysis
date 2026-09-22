import pandas as pd
import pytest

from analysis.annotate import _row_to_points
from analysis.landmarks import LANDMARK_NAMES, NUM_LANDMARKS, named_header
from analysis.ranking import rank_by_visibility


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=named_header()).to_csv(path, index=False)
    return str(path)


def _row(frame, **landmark_values):
    """landmark_values: name -> (x, y, visibility)"""
    row = {col: None for col in named_header()}
    row["frame"] = frame
    for name, (x, y, vis) in landmark_values.items():
        row[f"{name}_x"] = x
        row[f"{name}_y"] = y
        row[f"{name}_z"] = 0.0
        row[f"{name}_visibility"] = vis
    return row


# ---------- ranking ----------

def test_rank_orders_by_mean_visibility(tmp_path):
    rows = [
        _row(1, nose=(0.5, 0.5, 0.10), left_wrist=(0.5, 0.5, 0.90)),
        _row(2, nose=(0.5, 0.5, 0.30), left_wrist=(0.5, 0.5, 0.70)),
    ]
    table = rank_by_visibility(_write_csv(tmp_path / "a.csv", rows))

    assert list(table.columns) == ["landmark", "avg_visibility"]
    assert len(table) == NUM_LANDMARKS
    assert table.iloc[0]["landmark"] == "left_wrist"
    assert table.iloc[0]["avg_visibility"] == pytest.approx(0.80)

    nose = table[table["landmark"] == "nose"].iloc[0]
    assert nose["avg_visibility"] == pytest.approx(0.20)


def test_rank_respects_frame_window(tmp_path):
    rows = [
        _row(1, nose=(0.5, 0.5, 0.0)),
        _row(2, nose=(0.5, 0.5, 1.0)),
        _row(3, nose=(0.5, 0.5, 1.0)),
    ]
    path = _write_csv(tmp_path / "b.csv", rows)

    all_frames = rank_by_visibility(path)
    windowed = rank_by_visibility(path, start_frame=2)

    def nose_of(t):
        return t[t["landmark"] == "nose"].iloc[0]["avg_visibility"]

    assert nose_of(all_frames) == pytest.approx(2 / 3)
    assert nose_of(windowed) == pytest.approx(1.0), "frame 1 should be excluded"


def test_rank_ignores_blank_landmarks(tmp_path):
    # a landmark that never appears should average to NaN, not 0, so it sorts last
    rows = [_row(1, nose=(0.5, 0.5, 0.5))]
    table = rank_by_visibility(_write_csv(tmp_path / "c.csv", rows))
    left_pinky = table[table["landmark"] == "left_pinky"].iloc[0]
    assert pd.isna(left_pinky["avg_visibility"])


# ---------- annotate ----------

def test_row_to_points_scales_normalized_coords_to_pixels():
    row = _row(1, nose=(0.5, 0.25, 0.9))
    points = _row_to_points(row, "named", width=1624, height=320, threshold=0.5)
    assert points[LANDMARK_NAMES.index("nose")] == (812, 80, 0.9)


def test_row_to_points_drops_landmarks_below_threshold():
    row = _row(1, nose=(0.5, 0.5, 0.4), left_wrist=(0.5, 0.5, 0.6))
    points = _row_to_points(row, "named", 100, 100, threshold=0.5)
    assert LANDMARK_NAMES.index("nose") not in points
    assert LANDMARK_NAMES.index("left_wrist") in points


def test_row_to_points_skips_missing_landmarks(tmp_path):
    # go through a real CSV round-trip: pose_extraction writes blank fields for
    # undetected frames, and pd.read_csv turns those into NaN -- which is the
    # form _row_to_points actually has to cope with.
    path = _write_csv(tmp_path / "blank.csv", [_row(1)])
    row = pd.read_csv(path).iloc[0]

    assert _row_to_points(row, "named", 100, 100, threshold=0.0) == {}


def test_row_to_points_survives_a_partially_detected_row(tmp_path):
    path = _write_csv(tmp_path / "partial.csv", [_row(1, nose=(0.5, 0.5, 0.9))])
    row = pd.read_csv(path).iloc[0]

    points = _row_to_points(row, "named", 100, 100, threshold=0.5)
    assert list(points) == [LANDMARK_NAMES.index("nose")]
