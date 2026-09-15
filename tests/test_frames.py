import numpy as np
import pytest

from swim_analysis.frames import (
    frame_at,
    frame_count,
    frame_size,
    fps,
    iter_frames,
    median_background,
    sanitize_fps,
)

from conftest import moving_square_frames, write_video


def test_frame_size_and_count(moving_square_video):
    assert frame_size(moving_square_video) == (320, 160)
    assert frame_count(moving_square_video) == 40


def test_fps_reads_back(moving_square_video):
    assert fps(moving_square_video) == pytest.approx(30.0, abs=0.5)


@pytest.mark.parametrize("reported", [0.0, -1.0, float("nan")])
def test_fps_falls_back_when_container_reports_nothing_usable(reported):
    # containers really do report these; a 0 fps would make every timestamp
    # downstream a division by zero
    assert sanitize_fps(reported, default=59.94) == pytest.approx(59.94)


def test_sanitize_fps_passes_through_real_rates():
    assert sanitize_fps(59.94) == pytest.approx(59.94)
    assert sanitize_fps(30.0) == pytest.approx(30.0)


def test_frame_numbers_are_1_indexed(moving_square_video):
    numbers = [n for n, _ in iter_frames(moving_square_video)]
    assert numbers[0] == 1
    assert numbers == list(range(1, 41))


def test_frame_at_rejects_zero(moving_square_video):
    with pytest.raises(ValueError, match="1-indexed"):
        frame_at(moving_square_video, 0)


def test_frame_at_matches_sequential_read(moving_square_video):
    sequential = {n: f for n, f in iter_frames(moving_square_video)}
    for number in (1, 7, 40):
        seeked = frame_at(moving_square_video, number)
        # codec round-trip means near-, not bit-, equality
        assert np.abs(seeked.astype(int) - sequential[number].astype(int)).mean() < 6.0


def test_iter_frames_stride_and_window(moving_square_video):
    numbers = [n for n, _ in iter_frames(moving_square_video, stride=5)]
    assert numbers == [1, 6, 11, 16, 21, 26, 31, 36]

    windowed = [n for n, _ in iter_frames(moving_square_video, start_frame=10, end_frame=14)]
    assert windowed == [10, 11, 12, 13, 14]


def test_iter_frames_rejects_bad_stride(moving_square_video):
    with pytest.raises(ValueError, match="stride"):
        list(iter_frames(moving_square_video, stride=0))


def test_median_background_removes_the_moving_object(moving_square_video):
    background = median_background(moving_square_video, max_samples=40)

    assert background.shape == (160, 320, 3)
    # the square is dark and never dwells anywhere, so the plate should be
    # uniformly the light background rather than containing any of it
    assert background.min() > 150, "moving object leaked into the background plate"
    assert background.mean() == pytest.approx(200, abs=12)


def test_median_background_keeps_static_structure(tmp_path):
    frames = moving_square_frames(count=30)
    for frame in frames:
        frame[10:20, :] = 0  # a static dark stripe, like a lane line
    path = write_video(tmp_path / "static.mp4", frames)

    background = median_background(path, max_samples=30)
    assert background[10:20, :].mean() < 40, "static structure should survive"
    assert background[100:150, :].mean() > 150, "moving object should not"
