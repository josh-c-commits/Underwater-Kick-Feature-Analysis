import math

import numpy as np
import pytest

from analysis.calibration import Calibration, ReferenceLine, series_world_x


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


# ---------- building lines from labeller clicks ----------

from analysis.calibration import reference_line_from_clicks  # noqa: E402

NAMES = ["7.86m", "15m", "10m"]
MARKS = [7.86, 15.0, 10.0]


def test_clicks_become_a_reference_line():
    placed = {"7.86m": (400, 150), "15m": (900, 152), "10m": (560, 151)}
    line, reason = reference_line_from_clicks("near", placed, NAMES, MARKS)
    assert line is not None and reason == ""
    assert len(line.knots) == 3


def test_a_skipped_mark_is_left_out_not_fatal():
    # the exact situation that broke the first real calibration attempt:
    # 15m not visible, so it gets skipped (None)
    placed = {"7.86m": (400, 150), "15m": None, "10m": (560, 151)}
    line, _ = reference_line_from_clicks("near", placed, NAMES, MARKS)
    assert line is not None
    assert sorted(k[2] for k in line.knots) == [7.86, 10.0]


def test_one_visible_mark_cannot_define_a_line():
    placed = {"7.86m": (400, 150), "15m": None}
    line, reason = reference_line_from_clicks("near", placed, NAMES[:2], MARKS[:2])
    assert line is None
    assert "two known points" in reason


def test_marks_never_reached_count_as_missing():
    line, reason = reference_line_from_clicks("near", {}, NAMES, MARKS)
    assert line is None


def test_out_of_order_clicks_are_rejected():
    # 15m clicked to the LEFT of 7.86m while 10m sits between: not monotonic
    placed = {"7.86m": (400, 150), "15m": (300, 150), "10m": (560, 150)}
    line, reason = reference_line_from_clicks("near", placed, NAMES, MARKS)
    assert line is None
    assert "mis-click" in reason


def test_distances_decreasing_left_to_right_are_fine():
    # camera facing the other way down the pool: still one consistent direction
    placed = {"7.86m": (900, 150), "15m": (300, 150), "10m": (700, 150)}
    line, _ = reference_line_from_clicks("near", placed, NAMES, MARKS)
    assert line is not None


def test_two_marks_in_the_same_column_are_rejected():
    placed = {"7.86m": (400, 150), "15m": (400, 180)}
    line, reason = reference_line_from_clicks("near", placed, NAMES[:2], MARKS[:2])
    assert line is None


# ---------- isolines (what the overlay draws) ----------

def test_isoline_passes_through_each_lines_mark(two_line_calibration):
    segments = two_line_calibration.isoline(11.0, frame_height=320)
    solid = [seg for seg in segments if seg[2]]
    assert len(solid) == 1, "two reference lines -> one interpolated stretch between them"
    (x0, y0), (x1, y1), _ = solid[0]
    # each endpoint sits on a reference line, where the calibration reads 11m
    assert two_line_calibration.world_x(x0, y0) == pytest.approx(11.0, abs=1e-6)
    assert two_line_calibration.world_x(x1, y1) == pytest.approx(11.0, abs=1e-6)


def test_isoline_agrees_with_world_x_along_its_length(two_line_calibration):
    (x0, y0), (x1, y1), _ = [seg for seg in two_line_calibration.isoline(11.0, 320) if seg[2]][0]
    for t in np.linspace(0, 1, 9):
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        assert two_line_calibration.world_x(x, y) == pytest.approx(11.0, abs=0.05)


def test_isoline_is_held_vertical_beyond_the_reference_lines(two_line_calibration):
    segments = two_line_calibration.isoline(11.0, frame_height=320)
    held = [seg for seg in segments if not seg[2]]
    assert len(held) == 2
    for (x0, _), (x1, _), _ in held:
        assert x0 == pytest.approx(x1), "outside the lines world_x holds, so the isoline is vertical"
    tops = [min(seg[0][1], seg[1][1]) for seg in held]
    bottoms = [max(seg[0][1], seg[1][1]) for seg in held]
    assert min(tops) == 0.0 and max(bottoms) == 319.0, "held stretches reach the frame edges"


def test_single_line_isoline_has_no_interpolated_stretch():
    single = Calibration(lines=[line("only", 150.0, [(400.0, 7.86), (900.0, 15.0)])])
    segments = single.isoline(10.0, 320)
    assert segments and not any(seg[2] for seg in segments)


def test_isoline_outside_the_calibrated_range_is_empty(two_line_calibration):
    assert two_line_calibration.isoline(3.0, 320) == []
    assert two_line_calibration.isoline(20.0, 320) == []


def test_inverse_mapping_round_trips():
    near = line("near", 150.0, [(400.0, 7.86), (650.0, 10.0), (900.0, 15.0)])
    for world in (7.86, 9.0, 12.5, 15.0):
        assert near.world_x_at(near.image_x_for(world)) == pytest.approx(world)


def test_inverse_mapping_handles_distances_decreasing_left_to_right():
    reversed_line = line("rev", 150.0, [(400.0, 15.0), (900.0, 7.86)])
    assert reversed_line.world_x_at(reversed_line.image_x_for(10.0)) == pytest.approx(10.0)


def test_world_range(two_line_calibration):
    assert two_line_calibration.world_range() == (7.86, 15.0)


def test_reference_image_kind_round_trips(tmp_path, two_line_calibration):
    two_line_calibration.reference = "stabilized"
    path = tmp_path / "c.json"
    two_line_calibration.save(str(path))
    assert Calibration.load(str(path)).reference == "stabilized"


def test_older_calibrations_default_to_the_median_reference(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"version": 1, "lines": []}')  # written before the field existed
    assert Calibration.load(str(path)).reference == "median"


# ---------- a line clicked twice ----------

from analysis.calibration import coincident_lines  # noqa: E402


def test_the_same_line_clicked_twice_is_flagged():
    a = ReferenceLine("wall", [(238.0, 498.0, 0.0), (990.0, 455.0, 7.86)])
    b = ReferenceLine("near_rope", [(234.0, 498.0, 0.0), (993.0, 453.0, 7.86)])  # the real case
    assert coincident_lines([a, b]) == [("wall", "near_rope")]


def test_genuinely_different_lines_are_not_flagged(two_line_calibration):
    assert coincident_lines(two_line_calibration.lines) == []


# ---------- frame range the reference was built from ----------

def test_frame_range_round_trips(tmp_path, two_line_calibration):
    two_line_calibration.frames = (31, 180)
    path = tmp_path / "calib.json"
    two_line_calibration.save(str(path))
    assert Calibration.load(str(path)).frames == (31, 180)


def test_older_calibrations_have_no_frame_range(two_line_calibration):
    data = two_line_calibration.to_dict()
    del data["frames"]
    assert Calibration.from_dict(data).frames is None


# ---------- bend report ----------

import re  # noqa: E402

from analysis.calibration import (  # noqa: E402
    bend_report,
    neighbour_misses,
    straight_ruler_error,
    suspect_marks,
)

METRES = np.arange(0, 16.0)  # a marker every metre out to 15m


def wide_lens(distances, width=1624, standoff=8.0, centre=7.5):
    """Columns where floor marks at `distances` land through a wide lens whose
    scale changes by about 20% across the frame, like this footage's."""
    angle = np.arctan((np.asarray(distances, float) - centre) / standoff) - np.radians(2.0)
    return width / 2 + (width / 2) / np.radians(47.0) * angle * (1 + 0.35 * angle ** 2)


def marked_line(columns, distances, name="floor"):
    return ReferenceLine(name, [(float(x), 250.0, float(w)) for x, w in zip(columns, distances)])


def clicked(seed, noise=1.5):
    """Marks every metre, clicked with a realistic 1.5px of hand jitter."""
    return wide_lens(METRES) + np.random.default_rng(seed).normal(0, noise, len(METRES))


def test_a_straight_ruler_shows_no_bend():
    line = marked_line(100 + 80 * METRES, METRES)
    metres, pixels, _ = straight_ruler_error(line)
    assert metres == pytest.approx(0, abs=1e-9)
    assert pixels == pytest.approx(0, abs=1e-6)
    assert all(miss[1] == pytest.approx(0, abs=1e-9) for miss in neighbour_misses(line))


def test_bend_is_measured_at_the_inner_marks():
    # a straight ruler through 0m@100px and 10m@1100px puts 5m at 600px; it's at 650
    metres, pixels, at = straight_ruler_error(marked_line([100, 650, 1100], [0, 5, 10]))
    assert at == 5
    assert pixels == pytest.approx(50)
    assert metres == pytest.approx(0.5)


def test_two_marks_cannot_be_checked():
    line = marked_line([100, 900], [0, 10])
    assert straight_ruler_error(line) is None
    assert "third mark" in bend_report(line)[0]


def test_neighbour_miss_is_leave_one_out_interpolation():
    # 0m@100 and 10m@1100 put 5m at 600px; it was clicked at 620
    world, metres, pixels, gap = neighbour_misses(
        marked_line([100, 620, 1100, 1600], [0, 5, 10, 15])
    )[0]
    assert world == 5
    assert pixels == pytest.approx(20)
    assert metres == pytest.approx(0.2)
    assert gap == pytest.approx(500)


def test_honest_clicks_through_a_wide_lens_raise_no_suspects():
    for seed in range(50):
        assert suspect_marks(marked_line(clicked(seed), METRES)) == [], f"seed {seed}"


def test_a_mis_click_is_found():
    columns = clicked(1)
    columns[9] += 35  # about 30% of the gap between marks
    suspects = suspect_marks(marked_line(columns, METRES))
    assert [s[0] for s in suspects] == [9.0]
    assert suspects[0][2] == pytest.approx(35, abs=5)


@pytest.mark.parametrize("end", [0, -1])
def test_a_mis_clicked_end_mark_is_blamed_rather_than_its_neighbour(end):
    columns = clicked(2)
    columns[end] += 60
    assert [s[0] for s in suspect_marks(marked_line(columns, METRES))] == [METRES[end]]


def test_distances_decreasing_left_to_right_are_checked_the_same_way():
    columns = clicked(3)
    columns[9] += 35
    mirrored = marked_line(1624 - columns, METRES)
    assert [s[0] for s in suspect_marks(mirrored)] == [9.0]


def test_too_few_marks_to_tell_a_mis_click_from_bend():
    distances = METRES[:13:3]  # five marks
    columns = wide_lens(distances)
    columns[2] += 80
    assert suspect_marks(marked_line(columns, distances)) == []


def _bend_figure(report):
    return re.search(r"off by up to (.+?) \((\d+)px\), at ([\d.]+)m", report[1]).groups()


def test_report_measures_bend_with_the_mis_click_set_aside():
    columns = clicked(4)
    honest = bend_report(marked_line(np.delete(columns, 9), np.delete(METRES, 9)))
    columns[9] += 35
    report = bend_report(marked_line(columns, METRES))

    assert any(text.startswith("  possible mis-click: 9m") for text in report)
    assert "leaving out the possible mis-click" in report[1]
    assert _bend_figure(report) == _bend_figure(honest)


def test_report_on_clean_clicks():
    report = bend_report(marked_line(clicked(5), METRES))
    assert report[0] == "floor (16 marks, 0m to 15m):"
    assert "lens bend" in report[1] and "between marks" in report[2]
    assert report[-1] == "  clicks: no mark stands out from its neighbours."
