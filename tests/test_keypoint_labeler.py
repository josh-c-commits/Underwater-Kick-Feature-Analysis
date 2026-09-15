from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from swim_analysis.pointpicker import _KeypointLabeler  # noqa: E402

NAMES = ["nose", "left_wrist", "right_wrist"]


@pytest.fixture
def labeler():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    lab = _KeypointLabeler(img, NAMES, title="test")
    lab.fig.canvas.draw()
    yield lab
    matplotlib.pyplot.close(lab.fig)


def key(labeler, k):
    return SimpleNamespace(inaxes=labeler.ax, button=1, x=0, y=0, xdata=None,
                           ydata=None, key=k, step=0)


def click(labeler, x, y):
    press = SimpleNamespace(inaxes=labeler.ax, button=1, x=10, y=10,
                            xdata=None, ydata=None, key=None, step=0)
    release = SimpleNamespace(inaxes=labeler.ax, button=1, x=10, y=10,
                              xdata=float(x), ydata=float(y), key=None, step=0)
    labeler.on_press(press)
    labeler.on_release(release)


def test_starts_on_the_first_name(labeler):
    assert labeler.current == "nose"


def test_empty_names_is_rejected():
    with pytest.raises(ValueError, match="names"):
        _KeypointLabeler(np.zeros((10, 10, 3), dtype=np.uint8), [])


def test_click_places_the_current_point_and_advances(labeler):
    click(labeler, 100, 50)
    assert labeler.points["nose"] == (100, 50)
    assert labeler.current == "left_wrist"


def test_clicking_the_last_name_does_not_run_off_the_end(labeler):
    labeler.index = len(NAMES) - 1
    click(labeler, 10, 20)
    assert labeler.points["right_wrist"] == (10, 20)
    assert labeler.current == "right_wrist"


def test_reclicking_moves_an_existing_point(labeler):
    click(labeler, 100, 50)
    labeler.on_key(key(labeler, "p"))
    click(labeler, 111, 55)
    assert labeler.points["nose"] == (111, 55)


def test_skip_records_none_rather_than_omitting(labeler):
    # "not visible" and "not yet labelled" must stay distinguishable
    labeler.on_key(key(labeler, "s"))
    assert labeler.points["nose"] is None
    assert "nose" in labeler.points
    assert labeler.current == "left_wrist"


def test_clear_removes_the_entry_entirely(labeler):
    click(labeler, 100, 50)
    labeler.on_key(key(labeler, "p"))
    labeler.on_key(key(labeler, "u"))
    assert "nose" not in labeler.points


def test_navigation_is_clamped_at_both_ends(labeler):
    for _ in range(5):
        labeler.on_key(key(labeler, "p"))
    assert labeler.current == NAMES[0]
    for _ in range(10):
        labeler.on_key(key(labeler, "n"))
    assert labeler.current == NAMES[-1]


def test_enter_returns_what_was_placed(labeler):
    click(labeler, 10, 10)
    labeler.on_key(key(labeler, "s"))
    labeler.on_key(key(labeler, "enter"))
    result = labeler._result()
    assert result == {"nose": (10, 10), "left_wrist": None}


def test_escape_discards_everything(labeler):
    click(labeler, 10, 10)
    labeler.on_key(key(labeler, "escape"))
    assert labeler._result() is None


def test_existing_points_are_preloaded():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    lab = _KeypointLabeler(img, NAMES, existing={"nose": (5, 5)})
    assert lab.points["nose"] == (5, 5)
    matplotlib.pyplot.close(lab.fig)


def test_zoom_and_pan_still_work_in_labelling_mode(labeler):
    before = labeler.ax.get_xlim()
    labeler.on_scroll(SimpleNamespace(inaxes=labeler.ax, button="up", x=0, y=0,
                                      xdata=200.0, ydata=100.0, key=None, step=1))
    after = labeler.ax.get_xlim()
    assert (after[1] - after[0]) < (before[1] - before[0])
