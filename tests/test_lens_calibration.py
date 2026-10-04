"""The lens model end to end, on exact synthetic geometry: a camera behind flat
glass underwater, the pool's floor lines 9 ft apart crossing the swimmer's lane,
a swimmer at a known distance and depth, and the lane rope on the surface."""

import json

import numpy as np
import pytest

from analysis.calibration import Calibration, ReferenceLine, ruler_report
from analysis.lens import FlatPort, for_frame, is_ultrawide

W, H = 3840, 2160
LENS = for_frame(W, H)
FW = LENS.ideal_focal
FLOOR_LINES = [1.829 + 2.743 * k for k in range(8)]   # 6 ft from the wall, then every 9 ft
CAMERA_X = 11.43        # the camera faces the middle of the 25-yard swim
LANE = 12.0             # the swimmer's lane is 12 m from the camera
CAMERA_HEIGHT = 1.2     # above the floor
CAMERA_DEPTH = 0.5      # below the surface


def project(X, Y, Z, yaw=0.0, roll=0.0, pitch=0.0):
    """World (X along the swim, Y up from the camera, Z away from it) -> filmed
    image. pitch > 0 tilts the camera down."""
    X, Y, Z = (np.asarray(v, dtype=float) for v in (X, Y, Z))
    x = X - CAMERA_X
    Y, Z = Y * np.cos(pitch) + Z * np.sin(pitch), -Y * np.sin(pitch) + Z * np.cos(pitch)
    xc = x * np.cos(yaw) - Z * np.sin(yaw)
    zc = x * np.sin(yaw) + Z * np.cos(yaw)
    u, v = FW * xc / zc, -FW * Y / zc
    u, v = u * np.cos(roll) - v * np.sin(roll), u * np.sin(roll) + v * np.cos(roll)
    return LENS.distort(LENS.cx + u, LENS.cy + v)


def calibration(lane=LANE, yaw=0.0, roll=0.0, rope_distance=LANE, under_rope=False, **options):
    lanes = [lane - 1.1, lane + 1.1] if under_rope else [lane]
    lines = []
    for i, Z in enumerate(lanes):
        xs, ys = project(FLOOR_LINES, -CAMERA_HEIGHT, Z, yaw, roll)
        lines.append(ReferenceLine(f"lane line {i}", [(x, y, w) for x, y, w in zip(xs, ys, FLOOR_LINES)]))
    rope_x, rope_y = project([2.0, 8.0, 14.0, 20.0], CAMERA_DEPTH, rope_distance, yaw, roll)
    return Calibration(lines=lines, frame_size=(W, H), lens=LENS, under_rope=under_rope,
                       surface=list(zip(rope_x, rope_y)), **options)


def swimmer(X, depth, Z=LANE, yaw=0.0, roll=0.0):
    return project(X, CAMERA_DEPTH - depth, Z, yaw, roll)


def test_distance_along_the_swim_is_read_exactly():
    cal = calibration()
    for X in (3.0, 10.0, 20.0, 22.5):  # 22.5 m is past the last floor line: extrapolated
        assert cal.world_x(*swimmer(X, 0.8)) == pytest.approx(X, abs=1e-3)


def test_depth_below_the_surface_is_read_exactly():
    cal = calibration()
    for depth in (0.2, 0.8, 1.5):
        assert float(cal.depth(*swimmer(9.0, depth))) == pytest.approx(depth, abs=1e-3)


def test_the_lanes_distance_comes_out_of_the_scale():
    assert calibration().swimmer_distance() == pytest.approx(LANE, rel=1e-3)


def test_a_rope_nearer_than_the_swimmer_is_corrected_for():
    nearer = calibration(rope_distance=LANE - 1.2)
    raw = float(nearer.depth(*swimmer(9.0, 0.8)))
    assert raw != pytest.approx(0.8, abs=0.02), "uncorrected, the nearer rope reads deeper"
    fixed = calibration(rope_distance=LANE - 1.2, rope_offset=1.2, camera_depth=CAMERA_DEPTH)
    assert float(fixed.depth(*swimmer(9.0, 0.8))) == pytest.approx(0.8, abs=1e-3)


def test_under_the_rope_the_swimmer_is_read_between_the_two_lane_lines():
    cal = calibration(under_rope=True)
    assert cal.world_x(*swimmer(12.0, 0.6)) == pytest.approx(12.0, abs=1e-3)
    assert float(cal.depth(*swimmer(12.0, 0.6))) == pytest.approx(0.6, abs=1e-3)


def test_leftover_camera_roll_is_taken_out_using_the_rope():
    cal = calibration(roll=np.radians(0.8))
    assert np.degrees(cal.roll()) == pytest.approx(0.8, abs=0.01)
    assert cal.world_x(*swimmer(6.0, 1.0, roll=np.radians(0.8))) == pytest.approx(6.0, abs=2e-3)
    assert float(cal.depth(*swimmer(6.0, 1.0, roll=np.radians(0.8)))) == pytest.approx(1.0, abs=2e-3)


def test_a_turned_camera_needs_the_perspective_term():
    yaw = np.radians(6)
    cal = calibration(yaw=yaw)
    assert cal.world_x(*swimmer(5.0, 0.8, yaw=yaw)) == pytest.approx(5.0, abs=5e-3)
    assert cal.swimmer_rulers()[0].params[2] != 0.0


def test_an_uneven_floor_barely_moves_the_ruler():
    """The floor 30 cm deeper under one end of the lane: the marks move up or
    down in the picture, hardly sideways, so readings hardly change."""
    xs, ys = project(FLOOR_LINES, [-CAMERA_HEIGHT - 0.3 * k / 7 for k in range(8)], LANE)
    cal = calibration()
    cal.lines = [ReferenceLine("lane line", [(x, y, w) for x, y, w in zip(xs, ys, FLOOR_LINES)])]
    assert cal.world_x(*swimmer(20.0, 0.8)) == pytest.approx(20.0, abs=1e-3)


def test_a_camera_tilted_3_degrees_with_an_uneven_floor_is_still_close():
    """Your camera is within ~3 degrees of level. Tilted that much, the ruler (on
    the floor) and the swimmer (0.9 m above it) sit at slightly different
    distances along the line of sight: a ~0.4% scale error, 4-5 cm at the frame
    edges, ~0 in the middle. The floor 30 cm deeper under one end adds ~0.1%.
    Depth stays within a centimetre."""
    pitch = np.radians(3)
    xs, ys = project(FLOOR_LINES, [-CAMERA_HEIGHT - 0.3 * k / 7 for k in range(8)], LANE, pitch=pitch)
    rope_x, rope_y = project([2.0, 8.0, 14.0, 20.0], CAMERA_DEPTH, LANE, pitch=pitch)
    cal = Calibration(lines=[ReferenceLine("lane line", list(zip(xs, ys, FLOOR_LINES)))],
                      frame_size=(W, H), lens=LENS, surface=list(zip(rope_x, rope_y)))
    for X in (2.0, 11.0, 21.0):
        sx, sy = project(X, CAMERA_DEPTH - 0.8, LANE, pitch=pitch)
        assert cal.world_x(sx, sy) == pytest.approx(X, abs=0.06)
        assert float(cal.depth(sx, sy)) == pytest.approx(0.8, abs=0.01)


def test_a_misclicked_mark_is_named():
    cal = calibration()
    x, y, w = cal.lines[0].knots[3]
    cal.lines[0].knots[3] = (x + 25, y, w)
    report = "\n".join(ruler_report(cal, cal.lines[0]))
    assert f"{w:g} m mark" in report and "mis-clicked" in report


def test_saving_and_loading_keeps_the_lens_model(tmp_path):
    path = str(tmp_path / "cal.json")
    cal = calibration(rope_offset=1.0, camera_depth=0.7)
    cal.save(path)
    back = Calibration.load(path)
    assert back.lens == cal.lens and back.rope_offset == 1.0 and back.camera_depth == 0.7
    assert back.world_x(*swimmer(9.0, 0.8)) == pytest.approx(9.0, abs=1e-3)
    assert json.load(open(path))["version"] == 2


def test_version_1_files_still_load(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "lines": [{"name": "floor", "knots": [
        {"image_x": 100, "image_y": 500, "world_x": 0}, {"image_x": 300, "image_y": 500, "world_x": 2}]}]}))
    old = Calibration.load(str(path))
    assert old.lens is None and old.world_x(200, 500) == pytest.approx(1.0)


def test_the_lens_round_trips_and_knows_the_ultrawide():
    lens = FlatPort(1670.0, 1920.0, 1080.0)
    ux, uy = lens.undistort([10.0, 3800.0], [20.0, 2100.0])
    bx, by = lens.distort(ux, uy)
    assert np.allclose(bx, [10, 3800]) and np.allclose(by, [20, 2100])
    assert is_ultrawide("iPhone 17 Pro back camera 2.22mm f/2.2")
    assert not is_ultrawide("iPhone 17 Pro back camera 6.86mm f/1.78")
