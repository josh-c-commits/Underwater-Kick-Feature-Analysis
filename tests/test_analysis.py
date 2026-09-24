import math

import numpy as np
import pandas as pd
import pytest

from analysis.analysis import (
    body_length_series,
    camera_path,
    derivative,
    direction_of_travel,
    dominant_frequency,
    kinematics,
    smooth,
    summarize,
    velocity_fluctuation_index,
)
from analysis.calibration import Calibration, ReferenceLine
from analysis.landmarks import LANDMARK_NAMES, NUM_LANDMARKS, named_header, world_columns

FPS = 60.0


# ---------- smoothing ----------

def test_smooth_preserves_a_straight_line():
    ramp = np.arange(50, dtype=float)
    assert smooth(ramp) == pytest.approx(ramp, abs=1e-6)


def test_smooth_reduces_noise():
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 4 * np.pi, 200))
    noisy = clean + rng.normal(0, 0.2, 200)
    assert np.abs(smooth(noisy, 15) - clean).mean() < np.abs(noisy - clean).mean()


def test_smooth_keeps_gaps_as_gaps():
    values = np.arange(40, dtype=float)
    values[10:13] = np.nan
    out = smooth(values)
    assert np.isnan(out[10:13]).all(), "lost frames must not be invented"
    assert np.isfinite(out[0]) and np.isfinite(out[-1])


def test_smooth_handles_an_all_nan_series():
    assert np.isnan(smooth([np.nan] * 10)).all()


def test_smooth_handles_series_shorter_than_the_window():
    assert len(smooth([1.0, 2.0, 3.0], window=11)) == 3


# ---------- derivatives ----------

def test_derivative_recovers_a_known_rate():
    # 3 px per frame at 60fps is 180 px/s
    positions = np.arange(0, 300, 3, dtype=float)
    assert derivative(positions, FPS)[5:-5] == pytest.approx(180.0)


def test_derivative_of_a_single_sample_is_nan():
    assert np.isnan(derivative([1.0], FPS)).all()


# ---------- fluctuation ----------

def test_velocity_fluctuation_index_is_zero_for_constant_speed():
    assert velocity_fluctuation_index([2.0] * 20) == pytest.approx(0.0)


def test_velocity_fluctuation_index_is_scale_invariant():
    speeds = [1.0, 2.0, 3.0, 2.0, 1.5]
    plain = velocity_fluctuation_index(speeds)
    scaled = velocity_fluctuation_index([s * 7.3 for s in speeds])
    assert plain == pytest.approx(scaled), "must survive any constant scale error"


def test_velocity_fluctuation_index_ignores_nans():
    assert velocity_fluctuation_index([1.0, np.nan, 3.0]) == pytest.approx(1.0)


# ---------- frequency ----------

def test_dominant_frequency_recovers_a_known_kick_rate():
    t = np.arange(0, 4, 1 / FPS)
    signal = np.sin(2 * np.pi * 2.5 * t)
    assert dominant_frequency(signal, FPS) == pytest.approx(2.5, abs=0.15)


def test_dominant_frequency_ignores_slow_drift():
    # a swimmer travelling down the pool is a big low-frequency ramp that must
    # not be mistaken for the kick
    t = np.arange(0, 4, 1 / FPS)
    signal = np.sin(2 * np.pi * 3.0 * t) + 40 * t
    assert dominant_frequency(signal, FPS) == pytest.approx(3.0, abs=0.2)


def test_dominant_frequency_needs_enough_samples():
    assert math.isnan(dominant_frequency([1.0, 2.0], FPS))


# ---------- body length ----------

def _world_frame(points_by_name, rows=3):
    data = {col: [np.nan] * rows for col in named_header(world=True)}
    data["frame"] = list(range(1, rows + 1))
    for name, (x, y, z) in points_by_name.items():
        wx, wy, wz = world_columns(LANDMARK_NAMES.index(name))
        data[wx] = [x] * rows
        data[wy] = [y] * rows
        data[wz] = [z] * rows
    return pd.DataFrame(data)


def test_body_length_sums_a_chain_of_segments():
    frame = _world_frame({
        "left_shoulder": (0.0, 0.0, 0.0),
        "left_hip": (0.0, 0.5, 0.0),
        "left_knee": (0.0, 0.9, 0.0),
        "left_ankle": (0.0, 1.3, 0.0),
    })
    assert body_length_series(frame) == pytest.approx([1.3, 1.3, 1.3])


def test_body_length_requires_world_columns():
    plain = pd.DataFrame({col: [0.0] for col in named_header()})
    with pytest.raises(ValueError, match="world-landmark columns"):
        body_length_series(plain)


# ---------- kinematics ----------

def _boxes(x_positions, y=150.0):
    return pd.DataFrame({
        "frame": range(1, len(x_positions) + 1),
        "found": [np.isfinite(x) for x in x_positions],
        "centroid_x": x_positions,
        "centroid_y": [y] * len(x_positions),
    })


def test_kinematics_reports_pixel_speed_without_calibration():
    table = kinematics(_boxes(np.arange(0, 300, 3, dtype=float)), FPS)
    assert "speed_px_s" in table
    assert "speed_m_s" not in table
    assert table["speed_px_s"].iloc[10:-10].mean() == pytest.approx(180.0, rel=0.1)


def test_kinematics_adds_world_columns_with_a_calibration():
    calibration = Calibration(lines=[
        ReferenceLine("near", [(0.0, 150.0, 0.0), (1000.0, 150.0, 10.0)]),
        ReferenceLine("far", [(0.0, 120.0, 0.0), (1000.0, 120.0, 10.2)]),
    ])
    table = kinematics(_boxes(np.linspace(100, 900, 60)), FPS, calibration=calibration)

    assert "world_x_m" in table and "speed_m_s" in table
    assert "depth_ambiguity_m" in table
    assert np.isfinite(table["world_x_m"]).all()
    # 100px/m here, so 800px over 59 frames at 60fps is about 8.1 m/s
    assert table["speed_m_s"].iloc[10:-10].mean() == pytest.approx(8.1, rel=0.15)


def test_summarize_reports_units_and_core_metrics():
    table = kinematics(_boxes(np.arange(0, 300, 3, dtype=float)), FPS)
    summary = summarize(table, FPS)

    assert summary["speed_units"] == "px/s"
    assert summary["frames"] == 100
    assert summary["tracked_frames"] == 100
    assert np.isfinite(summary["velocity_fluctuation_index"])


def test_summarize_counts_lost_frames_separately():
    positions = np.arange(0, 150, 3, dtype=float)
    positions[5:10] = np.nan
    summary = summarize(kinematics(_boxes(positions), FPS), FPS)
    assert summary["frames"] == 50
    assert summary["tracked_frames"] == 45


# ---------- camera path ----------

def test_camera_path_is_zero_without_camera_columns():
    dx, dy = camera_path(_boxes(np.arange(10, dtype=float)))
    assert (dx == 0).all() and (dy == 0).all()


def test_camera_path_interpolates_unreliable_frames():
    boxes = _boxes(np.arange(5, dtype=float))
    boxes["cam_dx"] = [0.0, np.nan, np.nan, 3.0, 4.0]
    boxes["cam_dy"] = np.nan  # nothing usable at all
    dx, dy = camera_path(boxes)
    assert dx == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert (dy == 0).all()


def test_kinematics_measures_in_pool_coordinates_not_frame_coordinates():
    # the swimmer moves 3 px/frame through the pool while the camera drifts
    # 2 px/frame the same way, so the frame shows 5 px/frame
    n = 100
    boxes = _boxes(np.arange(n) * 5.0)
    boxes["cam_dx"] = np.arange(n) * 2.0
    boxes["cam_dy"] = 0.0
    table = kinematics(boxes, FPS)

    assert table["centroid_x"].iloc[50] == pytest.approx(250.0), "raw detection kept as-is"
    assert table["speed_px_s"].iloc[10:-10].mean() == pytest.approx(3 * FPS, rel=0.02)


# ---------- direction and leading edge ----------

def test_direction_of_travel():
    assert direction_of_travel(np.arange(50.0)) == 1
    assert direction_of_travel(np.arange(50.0)[::-1]) == -1
    assert direction_of_travel([np.nan, 3.0]) == 0


def test_leading_edge_is_the_edge_facing_the_direction_of_travel():
    right = _boxes(np.arange(0, 300, 3, dtype=float))
    right["edge_left"], right["edge_right"] = right["centroid_x"] - 10, right["centroid_x"] + 40
    left = _boxes(np.arange(300, 0, -3, dtype=float))
    left["edge_left"], left["edge_right"] = left["centroid_x"] - 40, left["centroid_x"] + 10

    r = kinematics(right, FPS)
    l = kinematics(left, FPS)
    assert (r["lead_x_smooth"] - r["x_smooth"]).iloc[10:-10].mean() == pytest.approx(40, abs=0.5)
    assert (l["x_smooth"] - l["lead_x_smooth"]).iloc[10:-10].mean() == pytest.approx(40, abs=0.5)


def test_leading_edge_is_absent_for_tables_without_edges():
    assert "lead_x_smooth" not in kinematics(_boxes(np.arange(0, 300, 3, dtype=float)), FPS)


def test_summary_peak_speed_is_the_fastest_moment_for_right_to_left_swimmers():
    # speed oscillates 2..4 px/frame; moving leftward, so image speeds are negative.
    # max() of the raw signed speeds would report the SLOWEST moment.
    t = np.arange(240)
    per_frame = 3.0 + np.sin(2 * np.pi * t / 60)
    positions = 2000.0 - np.cumsum(per_frame)
    summary = summarize(kinematics(_boxes(positions), FPS, smooth_window=5), FPS)

    assert summary["direction"] == "right-to-left"
    assert summary["mean_speed"] == pytest.approx(3.0 * FPS, rel=0.05)
    assert summary["peak_speed"] == pytest.approx(4.0 * FPS, rel=0.08)
