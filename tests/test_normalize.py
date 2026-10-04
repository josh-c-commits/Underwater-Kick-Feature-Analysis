import shutil

import cv2
import numpy as np
import pytest

from analysis.normalize import (
    ClipInfo,
    filter_chain,
    hlg_to_sdr,
    ignore_above,
    info_path,
    normalize_video,
    read_clip_info,
    roll_from_points,
    write_cube,
)

from conftest import write_video

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


# ---------- HDR to SDR ----------

def test_black_stays_black_and_greys_stay_ordered():
    grey = np.linspace(0.0, 1.0, 21)[:, None].repeat(3, axis=1)
    out = hlg_to_sdr(grey)[:, 0]
    assert out[0] == pytest.approx(0.0)
    assert np.all(np.diff(out) >= 0), "brighter HLG must never come out darker"
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_hlg_reference_white_lands_at_70_percent_of_sdr_white():
    # 75% HLG is the 203-nit reference white of ITU-R BT.2408; it's placed at 0.70
    # linear (0.70 ** (1/2.4) encoded) to leave headroom for the sunlit floor
    expected = 0.70 ** (1 / 2.4)
    assert hlg_to_sdr(np.array([0.75, 0.75, 0.75])) == pytest.approx([expected] * 3, abs=0.01)


def test_out_of_gamut_colour_keeps_its_brightness():
    """Pool cyan falls outside BT.709. Clipping it would change its brightness;
    pulling it toward grey at its own luminance must not."""
    cyan = np.array([0.10, 0.55, 0.60])
    out_linear = hlg_to_sdr(cyan) ** 2.4
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    scene = np.where(cyan <= 0.5, cyan ** 2 / 3, (np.exp((cyan - c) / a) + b) / 12)
    scene_y = scene @ [0.2627, 0.6780, 0.0593]
    expected_y = 1000 * scene_y ** 1.2 / 203 * 0.70  # display luminance, relative to SDR white
    assert expected_y < 0.8, "pick a colour below the highlight knee"
    assert out_linear @ [0.2126, 0.7152, 0.0722] == pytest.approx(expected_y, rel=1e-3)
    assert out_linear.min() >= 0.0


def test_lookup_table_matches_the_function(tmp_path):
    path = tmp_path / "t.cube"
    write_cube(str(path), size=5)
    lines = path.read_text().splitlines()
    assert lines[0] == "LUT_3D_SIZE 5"
    assert len(lines) == 1 + 5 ** 3
    # red varies fastest: entry 1 is (0.25, 0, 0), entry 5 is (0, 0.25, 0)
    assert [float(v) for v in lines[2].split()] == pytest.approx(hlg_to_sdr(np.array([0.25, 0, 0])), abs=1e-6)
    assert [float(v) for v in lines[6].split()] == pytest.approx(hlg_to_sdr(np.array([0, 0.25, 0])), abs=1e-6)


def test_filter_chain_order():
    chain = filter_chain(hlg=True, rotate=2.5, width=1920, cube="/tmp/x.cube")
    steps = [step.split("=")[0] for step in chain.split(",")]
    assert steps.index("lut3d") < steps.index("rotate") < steps.index("format", 2)
    assert "rotate=-(2.5)*PI/180" in chain
    assert "lut3d" not in filter_chain(hlg=False, rotate=None, width=None, cube="")


# ---------- levelling ----------

def test_roll_from_two_points():
    assert roll_from_points((0, 0), (100, 10)) == pytest.approx(np.degrees(np.arctan(0.1)))
    assert roll_from_points((100, 10), (0, 0)) == pytest.approx(np.degrees(np.arctan(0.1)))
    assert roll_from_points((0, 50), (1000, 50)) == 0.0
    with pytest.raises(ValueError):
        roll_from_points((5, 0), (5, 100))


def tilted_line_video(tmp_path, degrees=4.0, size=(320, 180)):
    width, height = size
    frames = []
    for _ in range(6):
        frame = np.full((height, width, 3), 200, np.uint8)
        y_left = height / 2 - np.tan(np.radians(degrees)) * width / 2
        y_right = height / 2 + np.tan(np.radians(degrees)) * width / 2
        cv2.line(frame, (0, int(round(y_left))), (width - 1, int(round(y_right))), (20, 20, 20), 3)
        frames.append(frame)
    return write_video(tmp_path / "tilted.mp4", frames), (0.0, y_left), (width - 1.0, y_right)


def line_height(gray, column):
    """Row of the dark line in this column, searching the middle rows only: the
    black fill rotation adds in the corners is darker than the line."""
    return 30.0 + float(np.argmin(gray[30:150, column].astype(float)))


@needs_ffmpeg
def test_levelling_by_two_points_makes_the_line_horizontal(tmp_path):
    source, left, right = tilted_line_video(tmp_path)
    out = str(tmp_path / "level.mp4")
    info = normalize_video(source, out, level_line=[list(left), list(right)])

    ok, frame = cv2.VideoCapture(out).read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert line_height(gray, 60) == pytest.approx(line_height(gray, 260), abs=1.5)
    assert info.rotation_deg == pytest.approx(4.0, abs=0.1)
    assert (info.width, info.height) == (320, 180)
    assert info.tone_mapped is None  # this test video isn't HDR


@needs_ffmpeg
def test_sidecar_records_what_was_done(tmp_path):
    source, left, right = tilted_line_video(tmp_path)
    out = str(tmp_path / "clip.mp4")
    normalize_video(source, out, rotate=1.5, ignore_top=0.25)
    info = read_clip_info(out)
    assert info.rotation_deg == 1.5
    assert info.ignore_above == 45
    assert ignore_above(out) == 45


def test_clip_info_round_trip_and_absence(tmp_path):
    video = str(tmp_path / "clip.mp4")
    assert read_clip_info(video) is None and ignore_above(video) is None
    ClipInfo(source="raw.mov", width=3840, height=2160, rotation_deg=2.0,
             level_line=[[0.0, 1.0], [3000.0, 105.0]], tone_mapped="hlg",
             ignore_above=720).save(info_path(video))
    info = read_clip_info(video)
    assert info.level_line == [[0.0, 1.0], [3000.0, 105.0]]
    assert info.ignore_above == 720 and info.tone_mapped == "hlg"
