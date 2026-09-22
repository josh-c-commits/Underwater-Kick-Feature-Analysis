import pytest

from analysis.duration import resolve_frame_bounds


def test_frame_bounds_pass_through_unchanged():
    assert resolve_frame_bounds(5, 40, None, None, None) == (5, 40)


def test_no_bounds_at_all():
    assert resolve_frame_bounds(None, None, None, None, None) == (None, None)


def test_time_bounds_convert_to_1_indexed_frames():
    # frame numbering is 1-indexed everywhere in this project, so t=0 is frame 1
    assert resolve_frame_bounds(None, None, 0.0, 1.0, 30.0) == (1, 31)


def test_time_bounds_round_rather_than_truncate():
    # 1.02s * 30fps = 30.6 -> rounds to 31, +1 for 1-indexing
    assert resolve_frame_bounds(None, None, 1.02, None, 30.0) == (32, None)


def test_only_end_time_given():
    assert resolve_frame_bounds(None, None, None, 2.0, 25.0) == (None, 51)


def test_time_bounds_without_fps_is_an_error():
    with pytest.raises(ValueError, match="fps is required"):
        resolve_frame_bounds(None, None, 1.0, 2.0, None)


def test_fps_only_required_when_time_bounds_used():
    # frame bounds with no fps must not raise
    assert resolve_frame_bounds(2, 3, None, None, None) == (2, 3)
