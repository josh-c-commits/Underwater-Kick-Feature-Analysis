"""Drives the picker's event handlers directly with synthetic events, so the
zoom/pan/click coordinate math is covered without needing a display."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from analysis.pointpicker import _PointPicker  # noqa: E402

W, H = 1624, 320


@pytest.fixture
def picker():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    p = _PointPicker(img, title="test")
    p.fig.canvas.draw()
    yield p
    matplotlib.pyplot.close(p.fig)


def event(picker, **kw):
    ev = SimpleNamespace(
        inaxes=picker.ax, button=1, x=0, y=0, xdata=None, ydata=None, key=None, step=0
    )
    for k, v in kw.items():
        setattr(ev, k, v)
    return ev


def view_size(picker):
    x0, x1 = picker.ax.get_xlim()
    y0, y1 = picker.ax.get_ylim()
    return abs(x1 - x0), abs(y1 - y0)


def test_image_axes_start_with_inverted_y(picker):
    (_, _), (y0, y1) = picker._home
    assert y0 > y1, "imshow axes must keep y growing downward"


def test_scroll_up_zooms_in(picker):
    before = view_size(picker)
    picker.on_scroll(event(picker, button="up", xdata=800.0, ydata=170.0))
    after = view_size(picker)
    assert after[0] < before[0]
    assert after[1] < before[1]


def test_zoom_keeps_the_cursor_point_pinned(picker):
    for _ in range(4):
        picker.on_scroll(event(picker, button="up", xdata=800.0, ydata=170.0))
    x0, x1 = picker.ax.get_xlim()
    y0, y1 = picker.ax.get_ylim()
    assert x0 < 800.0 < x1
    assert min(y0, y1) < 170.0 < max(y0, y1)


def test_zoom_out_is_clamped(picker):
    for _ in range(30):
        picker.on_scroll(event(picker, button="down", xdata=800.0, ydata=170.0))
    width, _ = view_size(picker)
    assert width <= W * 4.0 + 1


def test_zoom_in_is_clamped(picker):
    for _ in range(60):
        picker.on_scroll(event(picker, button="up", xdata=800.0, ydata=170.0))
    width, height = view_size(picker)
    assert width >= 8.0 - 1e-6
    assert height >= 8.0 - 1e-6


def test_clean_click_places_the_point_at_that_pixel(picker):
    picker.on_press(event(picker, x=500, y=200))
    picker.on_release(event(picker, x=502, y=201, xdata=812.4, ydata=168.7))
    assert picker.point == (812, 169)


def test_drag_pans_and_does_not_place_a_point(picker):
    before = picker.ax.get_xlim()
    picker.on_press(event(picker, x=500, y=200))
    picker.on_motion(event(picker, x=560, y=200))
    picker.on_release(event(picker, x=560, y=200, xdata=900.0, ydata=170.0))
    after = picker.ax.get_xlim()

    assert picker.point is None, "a drag must not be treated as a click"
    assert abs(after[0] - before[0]) > 1.0, "a drag should pan the view"


def test_tiny_movement_still_counts_as_a_click(picker):
    picker.on_press(event(picker, x=500, y=200))
    picker.on_motion(event(picker, x=502, y=201))
    picker.on_release(event(picker, x=502, y=201, xdata=100.0, ydata=100.0))
    assert picker.point == (100, 100)


def test_clicks_are_clamped_into_the_image(picker):
    picker.on_press(event(picker, x=10, y=10))
    picker.on_release(event(picker, x=10, y=10, xdata=-50.0, ydata=99999.0))
    assert picker.point == (0, H - 1)


def test_right_click_does_not_place_a_point(picker):
    picker.on_press(event(picker, button=3, x=10, y=10))
    picker.on_release(event(picker, button=3, x=10, y=10, xdata=100.0, ydata=100.0))
    assert picker.point is None


def test_reset_key_restores_the_home_view(picker):
    picker.on_scroll(event(picker, button="up", xdata=800.0, ydata=170.0))
    picker.on_key(event(picker, key="r"))
    assert picker.ax.get_xlim() == picker._home[0]
    assert picker.ax.get_ylim() == picker._home[1]


def test_clear_key_removes_the_point(picker):
    picker.point = (100, 100)
    picker._draw_marker()
    picker.on_key(event(picker, key="u"))
    assert picker.point is None


def test_escape_marks_the_pick_cancelled(picker):
    picker.point = (10, 10)
    picker.on_key(event(picker, key="escape"))
    assert picker._cancelled is True


def test_mask_overlay_does_not_change_reported_coordinates():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    mask[150:200, 800:840] = True
    p = _PointPicker(img, mask=mask, title="masked")
    p.fig.canvas.draw()
    p.on_press(event(p, x=10, y=10))
    p.on_release(event(p, x=10, y=10, xdata=820.0, ydata=175.0))
    assert p.point == (820, 175)
    matplotlib.pyplot.close(p.fig)
