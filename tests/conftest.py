import cv2
import numpy as np
import pytest


def write_video(path, frames, fps=30.0):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return str(path)


def moving_square_frames(
    count=40,
    size=(320, 160),
    square=24,
    start_x=40,
    step=6,
    y=80,
    background=200,
    foreground=20,
):
    """Static light background with one dark square tracking left to right.

    High contrast on purpose: these go through a real video codec, and a
    low-contrast target would make the tests fail for compression reasons
    rather than for anything the tracker did.
    """
    width, height = size
    frames = []
    for i in range(count):
        frame = np.full((height, width, 3), background, dtype=np.uint8)
        x = start_x + i * step
        frame[y:y + square, x:x + square] = foreground
        frames.append(frame)
    return frames


@pytest.fixture
def moving_square_video(tmp_path):
    return write_video(tmp_path / "square.mp4", moving_square_frames())
