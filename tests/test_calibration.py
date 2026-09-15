import math

import numpy as np
import pytest

from swim_analysis.calibration import Calibration, ReferenceLine, series_world_x


def line(name, y, knots):
    return ReferenceLine(name=name, knots=[(x, y, w) for x, w in knots])


@pytest.fixture
def two_line_calibration():
    # same two world marks, projecting to different columns on near and far
    # ropes -- that offset is the depth parallax
    near = line("near_rope", 150.0, [(400.0, 7.86), (900.0, 15.0)])
    far = line("far_rope", 120.0, [(430.0, 7.86), (940.0, 15.0)])
    return Calibration(lines=[near, far], frame_size=(1624, 320))


def test_world_x_interpolates_along_a_line(two_line_calibration):
    # halfway along the near rope's span is halfway between the two marks
    assert two_line_calibration.world_x(650.0, 150.0) == pytest.approx((7.86 + 15.0) / 2)


def test_world_x_at_a_knot_returns_that_knot(two_line_calibration):
    assert two_line_calibration.world_x(400.0, 150.0) == pytest.approx(7.86)
    assert two_line_calibration.world_x(900.0, 150.0) == pytest.approx(15.0)


def test_world_x_blends_between_lines_by_height(two_line_calibration):
    on_near = two_line_calibration.world_x(700.0, 150.0)
    on_far = two_line_calibration.world_x(700.0, 120.0)
    between = two_line_calibration.world_x(700.0, 135.0)

    assert on_near != pytest.approx(on_far), "the two ropes should disagree"
    assert min(on_near, on_far) < between < max(on_near, on_far)


def test_outside_the_knot_range_is_nan_not_a_clamped_guess(two_line_calibration):
    # the breakout happens largely outside the calibrated band, so silently
    # clamping would hand back confident numbers for uncalibrated water
    assert math.isnan(two_line_calibration.world_x(50.0, 150.0))
    assert math.isnan(two_line_calibration.world_x(1600.0, 150.0))


def test_extrapolation_is_available_when_asked_for(two_line_calibration):
    value = two_line_calibration.world_x(200.0, 150.0, extrapolate=True)
    assert np.isfinite(value)
    assert value < 7.86, "extrapolating left of the first mark means less distance"


def test_depth_ambiguity_measures_disagreement_between_lines(two_line_calibration):
    spread = two_line_calibration.depth_ambiguity(700.0)
    assert spread > 0
    near = two_line_calibration.lines[0].world_x_at(700.0)
    far = two_line_calibration.lines[1].world_x_at(700.0)
    assert spread == pytest.approx(abs(near - far))


def test_depth_ambiguity_is_nan_with_a_single_line():
    single = Calibration(lines=[line("only", 150.0, [(400.0, 7.86), (900.0, 15.0)])])
    assert math.isnan(single.depth_ambiguity(700.0))


def test_single_line_still_maps_positions():
    single = Calibration(lines=[line("only", 150.0, [(400.0, 7.86), (900.0, 15.0)])])
    assert single.world_x(650.0, 999.0) == pytest.approx((7.86 + 15.0) / 2)


def test_a_line_needs_two_knots_to_be_usable():
    thin = Calibration(lines=[ReferenceLine("one", [(400.0, 150.0, 7.86)])])
    with pytest.raises(ValueError, match="at least 2 knots"):
        thin.world_x(500.0, 150.0)


def test_knots_may_be_given_out_of_order():
    scrambled = Calibration(lines=[line("l", 150.0, [(900.0, 15.0), (400.0, 7.86)])])
    assert scrambled.world_x(650.0, 150.0) == pytest.approx((7.86 + 15.0) / 2)


def test_covered_range_reports_where_readings_are_trustworthy(two_line_calibration):
    low, high = two_line_calibration.covered_image_x()
    assert low == pytest.approx(400.0)
    assert high == pytest.approx(940.0)


def test_round_trips_through_json(tmp_path, two_line_calibration):
    path = tmp_path / "calib.json"
    two_line_calibration.save(str(path))
    loaded = Calibration.load(str(path))

    assert [l.name for l in loaded.lines] == ["near_rope", "far_rope"]
    assert loaded.frame_size == (1624, 320)
    assert loaded.world_x(650.0, 150.0) == pytest.approx(
        two_line_calibration.world_x(650.0, 150.0)
    )


def test_unknown_version_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"version": 99, "lines": []}')
    with pytest.raises(ValueError, match="Unsupported calibration version"):
        Calibration.load(str(path))


def test_series_world_x_passes_nan_through(two_line_calibration):
    out = series_world_x(two_line_calibration, [500.0, float("nan"), 700.0],
                         [150.0, 150.0, 150.0])
    assert np.isfinite(out[0])
    assert math.isnan(out[1])
    assert np.isfinite(out[2])
