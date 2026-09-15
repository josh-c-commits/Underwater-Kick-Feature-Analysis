"""Covers the crop -> full-frame coordinate handling without running MediaPipe
(which is what build_row exists to make possible)."""

from types import SimpleNamespace

import pytest

from swim_analysis.landmarks import NUM_LANDMARKS, named_header, world_columns
from swim_analysis.pose_extraction import build_row
from swim_analysis.tracking import crop_to_frame_norm

FRAME_W, FRAME_H = 1624, 320


def landmark(x, y, z=0.0, visibility=0.9):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def full_pose(x=0.5, y=0.5):
    return [landmark(x, y) for _ in range(NUM_LANDMARKS)]


# ---------- coordinate conversion ----------

def test_crop_to_frame_norm_is_identity_for_a_full_frame_box():
    x, y = crop_to_frame_norm(0.25, 0.75, (0, 0, FRAME_W, FRAME_H), FRAME_W, FRAME_H)
    assert x == pytest.approx(0.25)
    assert y == pytest.approx(0.75)


def test_crop_to_frame_norm_maps_crop_centre_to_box_centre():
    box = (800, 100, 200, 160)  # centre pixel (900, 180)
    x, y = crop_to_frame_norm(0.5, 0.5, box, FRAME_W, FRAME_H)
    assert x * FRAME_W == pytest.approx(900)
    assert y * FRAME_H == pytest.approx(180)


def test_crop_to_frame_norm_maps_corners():
    box = (800, 100, 200, 160)
    x0, y0 = crop_to_frame_norm(0.0, 0.0, box, FRAME_W, FRAME_H)
    x1, y1 = crop_to_frame_norm(1.0, 1.0, box, FRAME_W, FRAME_H)
    assert (x0 * FRAME_W, y0 * FRAME_H) == pytest.approx((800, 100))
    assert (x1 * FRAME_W, y1 * FRAME_H) == pytest.approx((1000, 260))


# ---------- row assembly ----------

def test_row_without_a_box_matches_the_plain_header():
    row = build_row(7, full_pose(), None, None, FRAME_W, FRAME_H, include_world=False)
    assert len(row) == len(named_header())
    assert row[0] == 7


def test_row_with_box_and_world_matches_the_extended_header():
    row = build_row(
        7, full_pose(), full_pose(), (10, 20, 30, 40), FRAME_W, FRAME_H, include_world=True
    )
    assert len(row) == len(named_header(world=True, box=True))
    assert row[1:5] == [10, 20, 30, 40], "box origin must be recorded"


def test_landmarks_are_stored_in_full_frame_coordinates_not_crop_coordinates():
    box = (800, 100, 200, 160)
    # a landmark dead centre of the crop is at pixel (900, 180) of the frame
    row = build_row(1, [landmark(0.5, 0.5)], None, box, FRAME_W, FRAME_H, include_world=False)
    x, y = float(row[5]), float(row[6])  # after frame + 4 box columns

    assert x * FRAME_W == pytest.approx(900)
    assert y * FRAME_H == pytest.approx(180)
    assert x != pytest.approx(0.5), "crop-relative coordinates must not leak through"


def test_missing_pose_produces_blanks_but_keeps_the_row_width():
    plain = build_row(3, None, None, None, FRAME_W, FRAME_H, include_world=False)
    assert len(plain) == len(named_header())
    assert plain[1:] == [""] * (NUM_LANDMARKS * 4)

    with_world = build_row(3, None, None, (0, 0, 0, 0), FRAME_W, FRAME_H, include_world=True)
    assert len(with_world) == len(named_header(world=True, box=True))


def test_world_landmarks_are_stored_unconverted():
    # world landmarks are metres from the hip midpoint, not image coordinates,
    # so cropping must not rescale them
    world = [landmark(0.123, -0.456, 0.789) for _ in range(NUM_LANDMARKS)]
    row = build_row(1, full_pose(), world, (800, 100, 200, 160),
                    FRAME_W, FRAME_H, include_world=True)

    offset = 5 + NUM_LANDMARKS * 4
    assert float(row[offset]) == pytest.approx(0.123)
    assert float(row[offset + 1]) == pytest.approx(-0.456)
    assert float(row[offset + 2]) == pytest.approx(0.789)


def test_world_columns_line_up_with_the_header():
    header = named_header(world=True)
    for i in range(NUM_LANDMARKS):
        for col in world_columns(i):
            assert col in header


def test_pose_landmark_columns_are_unchanged_by_the_world_addition():
    # existing consumers (annotate, ranking) index by name and must not shift
    plain = named_header()
    extended = named_header(world=True)
    assert extended[: len(plain)] == plain
