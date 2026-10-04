import os

import numpy as np
import pandas as pd
import pytest

from analysis import pointpicker
from analysis.analysis import kinematics
from analysis.association import Keyframe
from analysis.cli import main
from analysis.groundtruth import (
    follow_view,
    format_summary,
    labels_path,
    preset_names,
    read_labels,
    sample_frames,
    score_tracking,
    write_labels,
)
from analysis.landmarks import named_header
from analysis.tracking import detect_boxes

W, H = 1000, 500
FPS = 60.0


# ---------- label files ----------

def test_round_trip_keeps_not_visible_apart_from_unlabelled(tmp_path):
    path = str(tmp_path / "labels" / "clip.csv")
    labels = {16: {"nose": (100.0, 50.0), "right_hip": (300.5, 60.25), "right_foot_index": None},
              1: {"right_hip": None}}
    write_labels(path, labels, W, H)
    back = read_labels(path, W, H)
    assert sorted(back) == [1, 16]
    assert back[1] == {"right_hip": None}
    assert back[16]["right_foot_index"] is None
    assert back[16]["right_hip"] == pytest.approx((300.5, 60.25))
    assert "left_hip" not in back[16], "never labelled is not the same as not visible"


def test_label_files_have_mediapipes_layout(tmp_path):
    path = str(tmp_path / "clip.csv")
    write_labels(path, {5: {"nose": (500.0, 250.0)}}, W, H)
    table = pd.read_csv(path)
    assert list(table.columns) == named_header()
    assert table.loc[0, "nose_x"] == pytest.approx(0.5), "x as a fraction of the width, like MediaPipe"


def test_presets_use_the_side_facing_the_camera():
    assert preset_names("track", "left") == ["nose", "left_hip", "left_foot_index"]
    assert "right_knee" in preset_names("body", "right")
    with pytest.raises(ValueError):
        preset_names("track", "front")


def test_sampled_frames_are_the_same_every_time():
    frames = sample_frames(632, 40)
    assert frames == sample_frames(632, 40)
    assert frames[:3] == [1, 16, 31] and len(frames) == 40


def test_default_labels_file_is_named_after_the_clip():
    assert labels_path("data/normalized/josh_back_01.mp4") == os.path.join("data", "labels", "josh_back_01.csv")


# ---------- the view follows your own clicks ----------

def centre(view):
    x0, x1, y0, y1 = view
    return (x0 + x1) / 2, (y0 + y1) / 2


def test_view_starts_where_you_last_clicked():
    assert follow_view({}, 10, W, H) is None, "nothing to go on: show the whole frame"
    view = follow_view({1: {"right_hip": (500.0, 250.0)}}, 16, W, H)
    assert centre(view) == pytest.approx((500, 250))
    assert view[1] - view[0] == pytest.approx(W / 3)


def test_view_moves_on_at_the_swimmers_speed():
    labels = {1: {"right_hip": (700.0, 250.0)}, 16: {"right_hip": (600.0, 250.0)}}
    assert centre(follow_view(labels, 31, W, H))[0] == pytest.approx(500)


def test_view_stays_inside_the_frame():
    x0, x1, y0, y1 = follow_view({1: {"nose": (5.0, 495.0)}}, 2, W, H)
    assert x0 == 0 and y1 == pytest.approx(H)


def test_frames_with_the_swimmer_out_of_view_are_passed_over():
    labels = {1: {"right_hip": (500.0, 250.0)}, 16: {"right_hip": None}}
    assert centre(follow_view(labels, 31, W, H))[0] == pytest.approx(500)


# ---------- scoring ----------

def hip_at(frame):
    """A swimmer moving left at 3 px a frame (180 px/s)."""
    return 900.0 - 3.0 * (frame - 1)


def scenario(frames=range(1, 241), every=15):
    """Tracker centre 20 px ahead of the hip, a box that covers nose to toes, and:
    frame 1 swimmer out of view but tracked (false alarm), 40-50 lost, 91 on
    someone else, 121 a keyframe, 211 out of view and not tracked."""
    rows = []
    for f in frames:
        found = not (40 <= f <= 50 or f == 211)
        x = hip_at(f) - 20 + (300 if f == 91 else 0)
        rows.append({"frame": f, "found": found,
                     "centroid_x": x if found else np.nan, "centroid_y": 250.0 if found else np.nan,
                     "box_x": x - 150 if found else np.nan, "box_y": 230.0 if found else np.nan,
                     "box_w": 300.0 if found else np.nan, "box_h": 40.0 if found else np.nan,
                     "source": "keyframe" if f == 121 else ("detected" if found else "lost"),
                     "conflict": False, "merged": False})
    labels = {}
    for f in range(1, max(frames) + 1, every):
        if f in (1, 211):
            labels[f] = {"nose": None, "right_hip": None, "right_foot_index": None}
        else:
            labels[f] = {"nose": (hip_at(f) - 120, 245.0), "right_hip": (hip_at(f), 250.0),
                         "right_foot_index": (hip_at(f) + 150, 255.0)}
    return pd.DataFrame(rows), labels


def test_each_labelled_frame_gets_a_status():
    boxes, labels = scenario()
    table, summary = score_tracking(boxes, labels, FPS)
    status = table.set_index("frame")["status"]
    assert status[1] == "false alarm"
    assert status[46] == "lost"
    assert status[91] == "wrong swimmer"
    assert status[121] == "keyframe"
    assert status[211] == "correctly empty"
    assert status[16] == "on target"
    assert summary["false_alarms"] == [1] and summary["lost"] == [46]
    assert summary["wrong_swimmer"] == [91] and summary["keyframes"] == 1
    assert summary["visible"] == 16 - 2 - 1  # labelled minus out of view minus the keyframe
    assert summary["on_target"] == summary["visible"] - 2


def test_offset_and_speed_against_the_hip():
    boxes, labels = scenario()
    _, summary = score_tracking(boxes, labels, FPS)
    assert summary["body_length_px"] == pytest.approx(270.0, abs=0.5)
    assert summary["offset_along_px"] == pytest.approx(20.0), "20 px ahead in the direction of travel"
    assert summary["spread_along_px"] == pytest.approx(0.0, abs=1e-6)
    assert summary["hip_speed_px_s"] == pytest.approx(180.0)
    assert summary["track_speed_px_s"] == pytest.approx(180.0)
    assert summary["interval_rms_px_s"] == pytest.approx(0.0, abs=1e-6)
    assert summary["box_covers"] == summary["box_checked"] > 0


def test_with_kinematics_the_filled_gap_is_scored_too():
    boxes, labels = scenario()
    _, summary = score_tracking(boxes, labels, FPS, kinematics(boxes, FPS))
    assert summary["filled_frames"] == 1  # frame 46, inside the 40-50 gap
    assert summary["filled_error_px"] == pytest.approx(20.0, abs=3.0), "the fill follows the track"
    assert summary["track_speed_px_s"] == pytest.approx(180.0, rel=0.02)
    text = format_summary(summary)
    assert "Right swimmer: 11 of 13" in text and "frame 91" in text


def test_scoring_needs_hip_labels():
    boxes, _ = scenario()
    with pytest.raises(ValueError):
        score_tracking(boxes, {1: {"nose": (1.0, 1.0)}}, FPS)


# ---------- the label command ----------

@pytest.fixture
def labeller(monkeypatch):
    """Stub labeller that clicks every landmark at the square's centre, or
    stops (escape) on the call numbers listed in `stop_on`."""
    calls, stop_on = [], set()

    def label(image, names, title="", existing=None, view=None):
        calls.append({"names": list(names), "existing": existing, "view": view, "title": title})
        if len(calls) in stop_on:
            return None
        frame = int(title.split()[1])
        return {name: (52 + 6 * (frame - 1), 92) for name in names}

    monkeypatch.setattr(pointpicker, "label_keypoints", label)
    return calls, stop_on


def test_labelling_saves_every_frame_and_resumes(tmp_path, moving_square_video, labeller):
    calls, stop_on = labeller
    out = str(tmp_path / "labels.csv")
    stop_on.add(3)
    main(["label", moving_square_video, out, "--side", "right", "--sample", "4"])
    assert len(pd.read_csv(out)) == 2, "the two frames done before stopping are kept"
    assert calls[0]["names"] == ["nose", "right_hip", "right_foot_index"]

    calls.clear()
    stop_on.clear()
    main(["label", moving_square_video, out, "--side", "right", "--sample", "4"])
    assert [int(c["title"].split()[1]) for c in calls] == [21, 31], "carries on where it stopped"
    assert calls[0]["view"] is not None, "starts zoomed on where the earlier clicks were"

    calls.clear()
    main(["label", moving_square_video, out, "--side", "right", "--sample", "4"])
    assert calls == [], "nothing left to do"
    labels = read_labels(out, 320, 160)
    assert labels[31]["right_hip"] == (52 + 6 * 30, 92)


def test_redo_shows_the_existing_clicks(tmp_path, moving_square_video, labeller):
    calls, _ = labeller
    out = str(tmp_path / "labels.csv")
    main(["label", moving_square_video, out, "--side", "left", "--frames", "5"])
    main(["label", moving_square_video, out, "--side", "left", "--frames", "5", "--redo"])
    assert calls[1]["existing"]["left_hip"] == (52 + 6 * 4, 92)


def test_unknown_landmark_names_are_refused(tmp_path, moving_square_video, labeller):
    with pytest.raises(SystemExit):
        main(["label", moving_square_video, str(tmp_path / "x.csv"), "--landmarks", "left_hipp"])


def test_evaluate_track_reports_on_a_tracked_clip(tmp_path, moving_square_video, capsys):
    boxes_csv = str(tmp_path / "boxes.csv")
    detect_boxes(moving_square_video, keyframes=[Keyframe(1, 52, 92)],
                 progress=False).to_csv(boxes_csv, index=False)
    labels_csv = str(tmp_path / "labels.csv")
    write_labels(labels_csv, {f: {"nose": (40.0 + 6 * (f - 1), 92.0),
                                  "right_hip": (52.0 + 6 * (f - 1), 92.0),
                                  "right_foot_index": (64.0 + 6 * (f - 1), 92.0)}
                              for f in range(1, 41, 5)}, 320, 160)
    main(["evaluate-track", moving_square_video, boxes_csv, labels_csv,
          "--out", str(tmp_path / "scored.csv")])
    text = capsys.readouterr().out
    assert "Right swimmer: 7 of 7" in text, text  # frame 1 is the keyframe, left out
    assert len(pd.read_csv(tmp_path / "scored.csv")) == 8
