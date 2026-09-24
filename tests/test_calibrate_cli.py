"""The calibrate command end to end, with the click window replaced by a stub
that 'clicks' wherever a test says each mark is."""
import json

import cv2
import numpy as np
import pytest

import analysis.pointpicker as pointpicker
from analysis.calibration import Calibration
from analysis.cli import main
from analysis.tracking import shift_image

from conftest import write_video
from test_stabilization import textured

MARKER_FRAMES = 10


def floor_video(tmp_path, bump=(0.0, 0.0), count=40):
    """A textured pool floor with a dark marker every 20px along row 80 for the
    first 10 frames only -- markers down, filmed, then taken out. `bump` shifts
    those first frames, as if the camera was knocked when the markers came out."""
    world = cv2.cvtColor(textured(160, 320), cv2.COLOR_GRAY2BGR)
    frames = []
    for i in range(count):
        scene = world.copy()
        if i < MARKER_FRAMES:
            for x in range(40, 300, 20):
                scene[76:84, x - 4:x + 4] = 0
            scene = shift_image(scene, *bump) if any(bump) else scene
        frames.append(scene)
    return write_video(tmp_path / "floor.mp4", frames)


@pytest.fixture
def labeller(monkeypatch):
    """Stub labeller: mark 'Nm' is clicked at x = 40 + 20*N on row 80."""
    calls = []

    def label(image, names, title="", existing=None):
        calls.append({"names": list(names), "existing": existing, "image": image})
        return {name: (40 + 20 * float(name[:-1]), 80) for name in names}

    monkeypatch.setattr(pointpicker, "label_keypoints", label)
    return calls


def calibrate(video, out, *extra):
    main(["calibrate", video, str(out), *extra])
    return Calibration.load(str(out))


def test_mark_range_names_every_marker(tmp_path, labeller):
    calib = calibrate(floor_video(tmp_path), tmp_path / "c.json",
                      "--line", "floor", "--mark-range", "0", "12", "1")
    assert labeller[0]["names"] == [f"{n}m" for n in range(13)]
    assert [k[2] for k in calib.lines[0].sorted_knots()] == list(range(13))


def test_fractional_mark_range_has_clean_names(tmp_path, labeller):
    calibrate(floor_video(tmp_path), tmp_path / "c.json",
              "--line", "floor", "--mark-range", "0", "1", "0.1")
    assert labeller[0]["names"][3] == "0.3m"
    assert len(labeller[0]["names"]) == 11


def test_each_line_can_have_its_own_marks(tmp_path, labeller):
    calib = calibrate(floor_video(tmp_path), tmp_path / "c.json",
                      "--line", "floor", "--line", "upper",
                      "--mark-range", "0", "12", "1", "--marks", "2", "5", "10")
    assert labeller[0]["names"] == [f"{n}m" for n in range(13)]
    assert labeller[1]["names"] == ["2m", "5m", "10m"]
    assert [len(line.knots) for line in calib.lines] == [13, 3]


def test_one_mark_list_is_shared_by_every_line(tmp_path, labeller):
    calibrate(floor_video(tmp_path), tmp_path / "c.json",
              "--line", "a", "--line", "b", "--marks", "0", "6")
    assert labeller[0]["names"] == labeller[1]["names"] == ["0m", "6m"]


@pytest.mark.parametrize("extra, message", [
    (["--line", "a", "--line", "b", "--line", "c", "--marks", "0", "1", "--marks", "0", "2"],
     "lists of marks"),
    (["--line", "a"], "--marks"),
    (["--line", "a", "--marks", "3"], "at least two"),
    (["--line", "a", "--marks", "3", "3"], "repeat"),
    (["--line", "a", "--marks", "0", "3", "--frames", "5", "2"], "FIRST <= LAST"),
    (["--line", "a", "--marks", "0", "3", "--frames", "500", "600"], "only 40 frames"),
])
def test_bad_arguments_fail_before_any_window_opens(tmp_path, labeller, extra, message):
    with pytest.raises(SystemExit, match=message):
        main(["calibrate", floor_video(tmp_path), str(tmp_path / "c.json"), *extra])
    assert labeller == []


def test_backwards_mark_range_is_rejected(tmp_path, labeller):
    with pytest.raises(SystemExit):
        main(["calibrate", floor_video(tmp_path), str(tmp_path / "c.json"),
              "--line", "a", "--mark-range", "10", "0", "1"])


def test_markers_only_show_in_a_reference_built_from_their_frames(tmp_path, labeller):
    video = floor_video(tmp_path)
    calibrate(video, tmp_path / "whole.json", "--line", "floor", "--marks", "0", "5")
    calib = calibrate(video, tmp_path / "part.json", "--line", "floor", "--marks", "0", "5",
                      "--frames", "1", str(MARKER_FRAMES))
    whole, part = labeller[0]["image"], labeller[1]["image"]
    assert whole[80, 60].mean() > 60, "markers down for 10 of 40 frames should vanish"
    assert part[80, 60].mean() < 40, "markers should show in their own frames"
    assert calib.frames == (1, MARKER_FRAMES)


def test_seconds_convert_to_frames(tmp_path, labeller):
    calib = calibrate(floor_video(tmp_path), tmp_path / "c.json",
                      "--line", "floor", "--marks", "0", "5", "--seconds", "0", "0.3")
    assert calib.frames == (1, 9)  # 30fps: 0-0.3s is frames 1-9


def test_a_range_past_the_end_is_trimmed(tmp_path, labeller, capsys):
    calib = calibrate(floor_video(tmp_path), tmp_path / "c.json",
                      "--line", "floor", "--marks", "0", "5", "--frames", "30", "99")
    assert calib.frames == (30, 40)
    assert "ends at frame 40" in capsys.readouterr().out


def test_steady_camera_passes_the_check(tmp_path, labeller, capsys):
    calibrate(floor_video(tmp_path), tmp_path / "c.json",
              "--line", "floor", "--marks", "0", "5", "--frames", "1", "10")
    assert "Camera check: frames 1-10 line up" in capsys.readouterr().out


def test_a_camera_knocked_after_the_markers_is_caught(tmp_path, labeller, capsys):
    calibrate(floor_video(tmp_path, bump=(6.0, 3.0)), tmp_path / "c.json",
              "--line", "floor", "--marks", "0", "5", "--frames", "1", "10")
    out = capsys.readouterr().out
    found = re.search(r"camera sits ([\d.]+)px", out)
    assert found, out
    assert float(found.group(1)) == pytest.approx(np.hypot(6, 3), abs=1.0)
    assert "--stabilize" in out


def test_report_is_printed(tmp_path, labeller, capsys):
    calibrate(floor_video(tmp_path), tmp_path / "c.json",
              "--line", "floor", "--mark-range", "0", "12", "1")
    out = capsys.readouterr().out
    assert "How the marks check out against each other:" in out
    assert "floor (13 marks, 0m to 12m):" in out
    assert "no mark stands out" in out


def test_edit_reopens_the_saved_clicks(tmp_path, labeller):
    video, out = floor_video(tmp_path), tmp_path / "c.json"
    calibrate(video, out, "--line", "floor", "--marks", "0", "5", "9")
    calibrate(video, out, "--line", "floor", "--marks", "0", "5", "9", "--edit")
    assert labeller[0]["existing"] is None
    assert labeller[1]["existing"] == {"0m": (40.0, 80.0), "5m": (140.0, 80.0),
                                       "9m": (220.0, 80.0)}


def test_edit_offers_only_marks_still_being_asked_for(tmp_path, labeller):
    video, out = floor_video(tmp_path), tmp_path / "c.json"
    calibrate(video, out, "--line", "floor", "--marks", "0", "5", "9")
    calibrate(video, out, "--line", "floor", "--marks", "0", "5", "--edit")
    assert set(labeller[1]["existing"]) == {"0m", "5m"}


def test_edit_needs_a_saved_calibration(tmp_path, labeller):
    with pytest.raises(SystemExit, match="doesn't exist"):
        main(["calibrate", floor_video(tmp_path), str(tmp_path / "none.json"),
              "--line", "floor", "--marks", "0", "5", "--edit"])


def test_still_is_rebuilt_from_the_marker_frames(tmp_path, labeller):
    video, out = floor_video(tmp_path), tmp_path / "c.json"
    calibrate(video, out, "--line", "floor", "--marks", "0", "5",
              "--frames", "1", str(MARKER_FRAMES))
    still = tmp_path / "still.png"
    main(["overlay", video, str(still), "--calibration", str(out), "--still"])
    image = cv2.imread(str(still))
    # 3px right of the 3m line, still on the marker, clear of rings and labels
    assert image[80, 103].mean() < 40, "the still should show the markers"


def test_saved_json_records_the_frames(tmp_path, labeller):
    out = tmp_path / "c.json"
    calibrate(floor_video(tmp_path), out, "--line", "floor", "--marks", "0", "5",
              "--frames", "2", "8")
    assert json.loads(out.read_text())["frames"] == [2, 8]


import re  # noqa: E402
