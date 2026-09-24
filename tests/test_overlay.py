import cv2
import numpy as np
import pandas as pd
import pytest

from analysis.calibration import Calibration, ReferenceLine
from analysis.frames import frame_count, iter_frames
from analysis.overlay import draw_overlay
from analysis.tracking import detect_boxes


def calibration_for(width=320):
    # a vertical-ish ruler: 0m at x=40, 10m at x=280, on two lines
    return Calibration(lines=[
        ReferenceLine("upper", [(40.0, 40.0, 0.0), (280.0, 40.0, 10.0)]),
        ReferenceLine("lower", [(40.0, 120.0, 0.0), (280.0, 120.0, 10.0)]),
    ])


def test_overlay_writes_every_frame(moving_square_video, tmp_path):
    out = str(tmp_path / "overlay.mp4")
    boxes = detect_boxes(moving_square_video, progress=False)
    draw_overlay(moving_square_video, out, boxes=boxes, calibration=calibration_for())
    assert frame_count(out) == frame_count(moving_square_video)


def test_distance_lines_are_drawn_where_the_calibration_puts_them(moving_square_video, tmp_path):
    out = str(tmp_path / "lines.mp4")
    draw_overlay(moving_square_video, out, calibration=calibration_for(), every=5.0)
    frame = next(iter_frames(out))[1].astype(int)
    original = next(iter_frames(moving_square_video))[1].astype(int)
    changed = np.abs(frame - original).sum(axis=2) > 60

    # 5m sits at x=160 on both lines; the column there should be marked,
    # a column midway between distance lines should not be
    assert changed[50:110, 158:163].any(), "5m line missing"
    assert not changed[50:110, 95:105].any(), "something drawn between the lines"


def test_lines_follow_the_camera_offset(moving_square_video, tmp_path):
    boxes = detect_boxes(moving_square_video, progress=False)
    boxes["cam_dx"], boxes["cam_dy"] = 30.0, 0.0  # camera 30px right of the plate
    out = str(tmp_path / "shifted.mp4")
    draw_overlay(moving_square_video, out, boxes=boxes, calibration=calibration_for(), every=5.0)
    frame = next(iter_frames(out))[1].astype(int)
    original = next(iter_frames(moving_square_video))[1].astype(int)
    changed = np.abs(frame - original).sum(axis=2) > 60
    assert changed[50:110, 188:193].any(), "5m line should have moved with the camera to x=190"


def test_overlay_needs_something_to_draw(moving_square_video, tmp_path):
    with pytest.raises(ValueError, match="Nothing to draw"):
        draw_overlay(moving_square_video, str(tmp_path / "x.mp4"))


# ---------- still image on the reference plate ----------

from analysis.cli import main  # noqa: E402
from analysis.overlay import draw_calibration_still  # noqa: E402


def test_still_draws_lines_and_marks_on_the_reference(tmp_path):
    reference = np.full((160, 320, 3), 200, dtype=np.uint8)
    out = str(tmp_path / "still.png")
    draw_calibration_still(reference, calibration_for(), out, every=5.0)

    image = cv2.imread(out).astype(int)
    assert image.shape == reference.shape
    changed = np.abs(image - reference.astype(int)).sum(axis=2) > 60
    assert changed[50:110, 158:163].any(), "5m line missing"
    assert not changed[70:90, 95:105].any(), "nothing should be drawn between lines"
    # clicked marks sit at the knots, e.g. (40, 40) and (280, 120)
    assert changed[33:48, 33:48].any() and changed[113:128, 273:288].any(), "marks missing"


def test_still_refuses_a_video_extension(moving_square_video, tmp_path):
    calib = tmp_path / "c.json"
    calibration_for().save(str(calib))
    with pytest.raises(SystemExit, match=".png"):
        main(["overlay", moving_square_video, str(tmp_path / "x.mp4"),
              "--calibration", str(calib), "--still"])


def test_still_needs_a_calibration(moving_square_video, tmp_path):
    boxes = tmp_path / "b.csv"
    detect_boxes(moving_square_video, progress=False).to_csv(boxes, index=False)
    with pytest.raises(SystemExit, match="--calibration"):
        main(["overlay", moving_square_video, str(tmp_path / "x.png"),
              "--boxes", str(boxes), "--still"])


def test_still_end_to_end_through_the_cli(moving_square_video, tmp_path):
    calib = tmp_path / "c.json"
    calibration_for().save(str(calib))
    out = tmp_path / "still.png"
    main(["overlay", moving_square_video, str(out), "--calibration", str(calib), "--still"])
    assert cv2.imread(str(out)) is not None
