import cv2
import numpy as np
import pandas as pd
import pytest

from analysis.association import Keyframe
from analysis.export import camera_track, export_video, follow_crop, spotlight, swimmer_path
from analysis.tracking import detect_boxes

FPS = 30.0


def table(frames=40, lost=(), merged=(), keyframe=None):
    rows = []
    for f in range(1, frames + 1):
        found = f not in lost
        x = 300.0 - 4 * (f - 1)
        rows.append({"frame": f, "found": found,
                     "centroid_x": x if found else np.nan, "centroid_y": 100.0 if found else np.nan,
                     "box_x": x - 30 if found else np.nan, "box_y": 90.0 if found else np.nan,
                     "box_w": 60.0 if found else np.nan, "box_h": 20.0 if found else np.nan,
                     "source": "keyframe" if f == keyframe else ("detected" if found else "lost"),
                     "conflict": False, "merged": f in merged})
    return pd.DataFrame(rows)


# ---------- the path ----------

def test_each_frame_says_how_its_position_was_obtained():
    path = swimmer_path(table(lost=range(10, 13), merged={20}, keyframe=5), FPS).set_index("frame")
    assert path.loc[1, "status"] == "detected"
    assert path.loc[5, "status"] == "keyframe"
    assert path.loc[11, "status"] == "filled", "a short gap is bridged by the smoother"
    assert path.loc[20, "status"] == "merged"
    assert np.isfinite(path.loc[11, "x"])
    assert path.loc[11, "x"] == pytest.approx(300 - 4 * 10, abs=2)


def test_a_long_loss_has_no_position():
    path = swimmer_path(table(frames=120, lost=range(20, 100)), FPS).set_index("frame")
    assert path.loc[60, "status"] == "lost" and not np.isfinite(path.loc[60, "x"])


def test_speed_is_in_the_direction_of_travel():
    path = swimmer_path(table(), FPS)
    assert path["speed"].iloc[20] == pytest.approx(4 * FPS, rel=0.05), "moving left, reported positive"


def test_the_camera_carries_on_across_gaps_and_holds_at_the_ends():
    path = swimmer_path(table(frames=120, lost=list(range(1, 11)) + list(range(50, 90))), FPS)
    x, _ = camera_track(path, FPS)
    assert np.isfinite(x).all()
    assert x[0] == pytest.approx(300 - 4 * 10, abs=6), "held at the first position (frame 11)"
    assert x[50] > x[70] > x[88], "keeps moving through the long gap"


# ---------- drawing ----------

def test_spotlight_keeps_the_swimmer_and_greys_the_rest():
    frame = np.zeros((200, 400, 3), np.uint8)
    frame[:] = (200, 120, 40)  # blue water
    out = spotlight(frame, (200, 100), (60, 20), (0, 255, 0))
    assert tuple(out[100, 200]) == (200, 120, 40), "the swimmer keeps its colour"
    corner = out[5, 5]
    assert corner[0] == corner[1] == corner[2] and corner[0] < 120, "elsewhere grey and dimmed"
    assert len(set(spotlight(frame, None, None, (0, 255, 0))[100, 200])) == 1, "lost: all grey"


def test_follow_crop_slides_inward_at_the_edges():
    frame = np.arange(100 * 200).reshape(100, 200).astype(np.uint8)
    crop = follow_crop(frame, (5, 50), (40, 20))
    assert crop.shape == (20, 40)
    assert np.array_equal(crop, frame[40:60, 0:40])


# ---------- end to end ----------

@pytest.fixture
def tracked(moving_square_video):
    boxes = detect_boxes(moving_square_video, keyframes=[Keyframe(1, 52, 92)], progress=False)
    return moving_square_video, boxes


def read_all(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def test_spotlight_video(tmp_path, tracked):
    video, boxes = tracked
    out = export_video(video, boxes, str(tmp_path / "spot.mp4"), width=320, progress=False)
    frames = read_all(out)
    assert len(frames) == 40 and frames[0].shape == (160, 320, 3)
    x = 52 + 6 * 19  # the square's centre on frame 20
    assert frames[19][92, x].mean() < 60, "the swimmer shows through"
    assert 80 < frames[19][20, 300].mean() < 140, "the background is dimmed (200 -> ~110)"


def test_follow_video_keeps_the_swimmer_in_the_middle(tmp_path, tracked):
    video, boxes = tracked
    out = export_video(video, boxes, str(tmp_path / "follow.mp4"), mode="follow", width=320,
                       progress=False)
    frames = read_all(out)
    assert len(frames) == 40 and frames[0].shape == (180, 320, 3)
    for frame in frames[10:30]:
        assert frame[70:110, 140:180].mean() < 80, "the square stays centred"
