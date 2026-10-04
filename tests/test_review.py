"""The reviewer's logic and interactions, driven with synthetic events on the
Agg backend so no display is needed."""

from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from analysis.association import Keyframe, load_keyframes  # noqa: E402
from analysis.frames import frame_at  # noqa: E402
from analysis.review import FrameSource, Reviewer, frame_status, next_problem  # noqa: E402
from analysis.tracking import candidates_path, detect_boxes, detect_candidates  # noqa: E402

from conftest import moving_square_frames, write_video  # noqa: E402


def centre_x(frame):
    """The square's centre: it starts at x=40 and steps 3 px a frame (24 px wide)."""
    return 40 + 3 * (frame - 1) + 12


# ---------- pure logic ----------

def test_next_problem_skips_the_rest_of_the_current_stretch():
    status = np.array(["detected", "lost", "lost", "detected", "conflict", "detected", "merged"],
                      dtype=object)
    assert next_problem(status, 1) == 2
    assert next_problem(status, 2) == 5
    assert next_problem(status, 5) == 7
    assert next_problem(status, 7) is None
    assert next_problem(status, 7, -1) == 5
    assert next_problem(status, 4, -1) == 2, "land on the start of the previous stretch"


def test_frame_status_reads_the_table():
    table = pd.DataFrame({
        "frame": [1, 2, 3, 4, 5],
        "found": [True, False, True, True, True],
        "source": ["keyframe", "lost", "detected", "detected", "detected"],
        "conflict": [False, False, True, False, False],
        "merged": [False, False, False, True, False],
    })
    assert list(frame_status(table, [Keyframe(2)])) == [
        "keyframe", "absent", "conflict", "merged", "detected"]


def test_frame_source_matches_direct_reads(moving_square_video):
    source = FrameSource(moving_square_video)
    for number in (5, 2, 3, 30, 31):  # a jump back, sequential steps, a jump forward
        expected = cv2.cvtColor(frame_at(moving_square_video, number), cv2.COLOR_BGR2RGB)
        assert np.array_equal(source.get(number), expected)
    source.close()


# ---------- interactions ----------

@pytest.fixture
def review(tmp_path):
    video = write_video(tmp_path / "clip.mp4", moving_square_frames(count=40, step=3))
    blobs, camera = detect_candidates(video, progress=False)
    blobs = blobs[~blobs.frame.between(10, 14)]  # five frames where nothing was detected
    boxes_csv = str(tmp_path / "boxes.csv")
    start = [Keyframe(1, centre_x(1), 92)]
    detect_boxes(video, keyframes=start, candidates=(blobs, camera),
                 progress=False).to_csv(boxes_csv, index=False)
    blobs.to_csv(candidates_path(boxes_csv), index=False)
    reviewer = Reviewer(video, boxes_csv, blobs, start, str(tmp_path / "keys.json"), fps=30.0)
    yield reviewer, boxes_csv, tmp_path
    matplotlib.pyplot.close(reviewer.fig)


def press(reviewer, key):
    reviewer.on_key(SimpleNamespace(key=key))


def test_stepping_and_jumping(review):
    reviewer = review[0]
    press(reviewer, "right")
    assert reviewer.frame == 2
    press(reviewer, "up")
    assert reviewer.frame == 12
    press(reviewer, "end")
    press(reviewer, "right")
    assert reviewer.frame == 40, "stays on the last frame"
    press(reviewer, "home")
    assert reviewer.frame == 1


def test_next_problem_lands_on_the_gap(review):
    reviewer = review[0]
    press(reviewer, "n")
    assert reviewer.frame == 10


def test_clicking_the_swimmer_adds_a_keyframe_and_retracks(review):
    reviewer = review[0]
    reviewer.go(20)
    reviewer._on_click(int(centre_x(20) / reviewer.scale), 92)
    assert [k.frame for k in reviewer.keyframes] == [1, 20]
    assert reviewer.dirty
    assert reviewer.table.set_index("frame").loc[20, "source"] == "keyframe"


def test_absent_then_delete(review):
    reviewer = review[0]
    reviewer.go(30)
    press(reviewer, "a")
    assert not reviewer.table.set_index("frame").loc[30, "found"]
    assert reviewer.status[29] == "absent"
    press(reviewer, "d")
    assert reviewer.table.set_index("frame").loc[30, "found"]


def test_timeline_click_jumps_there(review):
    reviewer = review[0]
    reviewer.on_release(SimpleNamespace(inaxes=reviewer.ax_time, xdata=24.6, button=1, x=0, y=0))
    assert reviewer.frame == 25


def test_save_writes_keyframes_and_the_retracked_boxes(review):
    reviewer, boxes_csv, tmp = review
    reviewer.go(20)
    reviewer._on_click(int(centre_x(20)), 92)
    press(reviewer, "s")
    assert reviewer.saved and not reviewer.dirty
    assert [k.frame for k in load_keyframes(str(tmp / "keys.json"))] == [1, 20]
    assert pd.read_csv(boxes_csv).set_index("frame").loc[20, "source"] == "keyframe"
