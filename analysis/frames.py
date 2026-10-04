"""Shared video frame access and background compositing.

Frame numbers are 1-indexed everywhere in this project -- frame 1 is
the first frame of the video -- matching the 'frame' column that every
CSV stage keys on. OpenCV is 0-indexed internally, so the conversion
lives here and nowhere else.

median_background() is the load-bearing piece: taking the per-pixel
median across frames spread over a clip cancels anything that moves
(the swimmer, surface ripple, refraction shimmer) and keeps whatever
is static (lane ropes, wall markings, pool floor). That single image
is what both tracking.py (as the reference to subtract) and
calibration.py (as the crisp frame to click references on) are built
on.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_FPS = 30.0


def _open(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path} with OpenCV.")
    return cap


def frame_count(video_path: str) -> int:
    cap = _open(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def frame_size(video_path: str) -> Tuple[int, int]:
    """(width, height) in pixels."""
    cap = _open(video_path)
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def sanitize_fps(value: float, default: float = DEFAULT_FPS) -> float:
    """Fall back to `default` for the 0/NaN frame rates some containers
    report (phone-recorded .mov files are the usual offender)."""
    if not value or value != value or value <= 0:  # covers 0 and NaN
        return default
    return float(value)


def fps(video_path: str, default: float = DEFAULT_FPS) -> float:
    """Frames per second, sanitized -- see sanitize_fps."""
    cap = _open(video_path)
    try:
        value = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return sanitize_fps(value, default)


def frame_at(video_path: str, frame_number: int) -> np.ndarray:
    """Read a single 1-indexed frame."""
    if frame_number < 1:
        raise ValueError(f"Frame numbers are 1-indexed; got {frame_number}.")
    cap = _open(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, frame = cap.read()
        if not success:
            raise RuntimeError(f"Could not read frame {frame_number} of {video_path}.")
        return frame
    finally:
        cap.release()


def iter_frames(
    video_path: str,
    start_frame: int = 1,
    end_frame: Optional[int] = None,
    stride: int = 1,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Yield (frame_number, bgr_frame) sequentially. Reads straight through
    rather than seeking per frame, since seeking is both slower and
    unreliable on some codecs.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cap = _open(video_path)
    try:
        number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            number += 1
            if number < start_frame:
                continue
            if end_frame is not None and number > end_frame:
                break
            if (number - start_frame) % stride == 0:
                yield number, frame
    finally:
        cap.release()


def sample_frames(
    video_path: str,
    max_samples: int = 120,
    start_frame: int = 1,
    end_frame: Optional[int] = None,
) -> List[np.ndarray]:
    """Up to max_samples frames spread evenly across the clip, or across
    start_frame..end_frame (1-indexed, inclusive) when given."""
    total = frame_count(video_path)
    if total <= 0:
        # some containers don't report a frame count; fall back to reading all
        return [frame for _, frame in iter_frames(video_path, start_frame, end_frame)][:max_samples]
    last = total if end_frame is None else min(end_frame, total)
    stride = max(1, (last - start_frame + 1) // max_samples)
    return [frame for _, frame in iter_frames(video_path, start_frame, last, stride)][:max_samples]


def stack_median(stack: np.ndarray, rows_per_strip: int = 64) -> np.ndarray:
    """
    Per-pixel median over the first axis of an (n, h, w[, c]) uint8 stack,
    computed a strip of rows at a time.

    np.median over the whole stack makes a full sorted copy of it. Working in
    strips keeps that copy small, so the peak memory is about the stack itself:
    120 samples of 4K is ~3GB this way, against ~9GB building a list of frames,
    stacking it and taking the median in one go.
    """
    out = np.empty(stack.shape[1:], dtype=np.uint8)
    for row in range(0, stack.shape[1], rows_per_strip):
        out[row:row + rows_per_strip] = np.median(stack[:, row:row + rows_per_strip], axis=0)
    return out


def median_background(
    video_path: str,
    max_samples: int = 120,
    start_frame: int = 1,
    end_frame: Optional[int] = None,
    ignore_above: Optional[int] = None,
) -> np.ndarray:
    """
    Per-pixel median across frames spread over the clip: a static
    background plate with moving things (swimmer, ripple) removed.

    max_samples trades accuracy for memory and time. Samples are held in one
    preallocated array, so memory is roughly max_samples frames: ~25MB each
    at 4K, ~6MB at 1080p.

    start_frame/end_frame restrict it to part of the clip -- e.g. the few
    seconds when calibration markers were on the pool floor. Over the whole
    clip those markers would vanish from the median like anything else that
    isn't there most of the time.

    ignore_above: rows above this are left out of the median and copied from
    the first sample instead -- for a region nothing should be measured in,
    such as the water surface. It also cuts memory in proportion.
    """
    total = frame_count(video_path)
    if total > 0:
        last = total if end_frame is None else min(end_frame, total)
        stride = max(1, (last - start_frame + 1) // max_samples)
        expected = min(max_samples, len(range(start_frame, last + 1, stride)))
    else:  # some containers don't report a frame count; read from the start
        last, stride, expected = end_frame, 1, max_samples
    top = max(0, ignore_above or 0)

    plate, stack, count = None, None, 0
    for _, frame in iter_frames(video_path, start_frame, last, stride):
        if stack is None:
            plate = frame.copy()
            stack = np.empty((max(expected, 1),) + frame[top:].shape, dtype=np.uint8)
        if count == len(stack):
            break
        stack[count] = frame[top:]
        count += 1
    if not count:
        raise RuntimeError(f"No frames could be read from {video_path}.")
    plate[top:] = stack_median(stack[:count])
    return plate
