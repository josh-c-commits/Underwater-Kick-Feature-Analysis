import numpy as np
import pytest

from analysis.tracking import BOX_COLUMNS, detect_boxes, fixed_box, suggest_roi

from conftest import moving_square_frames, write_video


# ---------- fixed_box ----------

def test_fixed_box_centres_on_the_point():
    assert fixed_box(500, 200, 100, 80, 1624, 320) == (450, 160, 100, 80)


def test_fixed_box_keeps_its_size_at_the_frame_edge():
    # shifted inward, NOT clipped -- a shrinking crop would make landmark
    # coordinates incomparable between frames, which is the whole point of
    # using a fixed size
    x, y, w, h = fixed_box(5, 5, 100, 80, 1624, 320)
    assert (w, h) == (100, 80)
    assert (x, y) == (0, 0)

    x, y, w, h = fixed_box(1620, 315, 100, 80, 1624, 320)
    assert (w, h) == (100, 80)
    assert (x, y) == (1524, 240)


def test_fixed_box_only_shrinks_when_bigger_than_the_frame():
    assert fixed_box(100, 100, 5000, 5000, 1624, 320) == (0, 0, 1624, 320)


def test_fixed_box_stays_inside_the_frame_everywhere():
    for cx in range(0, 1624, 97):
        for cy in range(0, 320, 41):
            x, y, w, h = fixed_box(cx, cy, 120, 90, 1624, 320)
            assert x >= 0 and y >= 0
            assert x + w <= 1624 and y + h <= 320


# ---------- detection ----------

def test_detect_boxes_follows_a_moving_object(moving_square_video):
    table = detect_boxes(moving_square_video, progress=False)

    assert list(table.columns) == BOX_COLUMNS
    assert len(table) == 40
    assert table["found"].sum() >= 35

    found = table[table["found"]]
    # the square starts at x=40 stepping +6/frame, so the centroid must climb
    assert found["centroid_x"].is_monotonic_increasing
    assert found["centroid_x"].iloc[0] == pytest.approx(52, abs=6)
    assert found["centroid_y"].mean() == pytest.approx(92, abs=6)


def test_detect_boxes_reports_every_frame_even_when_lost(moving_square_video):
    table = detect_boxes(moving_square_video, min_area=10**6, progress=False)
    assert len(table) == 40
    assert not table["found"].any(), "an impossible min_area should find nothing"
    assert table["area"].eq(0).all()


def test_roi_excludes_objects_outside_the_band(moving_square_video):
    # the square sits at y~80-104; a band well below it should see nothing
    table = detect_boxes(moving_square_video, roi=(130, 160), progress=False)
    assert not table["found"].any()


def test_seed_point_chooses_between_two_objects(tmp_path):
    """A small target and a much larger decoy: seeding must pick the small one
    and the area gate must keep it there."""
    frames = moving_square_frames(count=30, square=16, start_x=20, step=2, y=30)
    for i, frame in enumerate(frames):
        big = 60
        x = 200 - i * 2
        frame[90:90 + big, x:x + big] = 20  # large decoy moving the other way
    path = write_video(tmp_path / "two.mp4", frames)

    seeded = detect_boxes(path, seed_point=(28, 38), progress=False)
    found = seeded[seeded["found"]]

    assert len(found) >= 20
    assert found["centroid_y"].mean() < 70, "should track the small upper object"
    # the decoy is ~14x the area of the target; the gate must never admit it
    assert found["area"].max() < 16 * 16 * 4


def test_unseeded_detection_prefers_the_largest_object(tmp_path):
    frames = moving_square_frames(count=20, square=16, start_x=20, step=2, y=30)
    for i, frame in enumerate(frames):
        frame[90:150, 200 - i * 2:260 - i * 2] = 20
    path = write_video(tmp_path / "two2.mp4", frames)

    table = detect_boxes(path, progress=False)
    found = table[table["found"]]
    assert found["centroid_y"].mean() > 70, "without a seed the big object wins"


def test_suggest_roi_excludes_a_persistently_noisy_band(tmp_path):
    rng = np.random.default_rng(0)
    frames = moving_square_frames(count=40, y=100, square=20)
    for frame in frames:
        # a band that changes every frame, standing in for a swaying lane rope
        frame[0:30, :] = rng.integers(0, 255, size=(30, frame.shape[1], 3), dtype=np.uint8)
    path = write_video(tmp_path / "noisy.mp4", frames)

    y0, y1 = suggest_roi(path, max_samples=20)
    assert y0 >= 25, f"noisy band should be excluded, got roi=({y0},{y1})"
    assert y1 > y0
