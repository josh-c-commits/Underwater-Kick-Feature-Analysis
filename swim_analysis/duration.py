"""Shared duration-window resolution.

Turns either a frame range or a time range (+ fps) into a concrete
(start_frame, end_frame) pair. Used by both ranking.py (landmark
tables) and outline.py (outline-derived tables) so both accept the
same --start-frame/--end-frame/--start-time/--end-time/--fps flags
with identical behavior.
"""

from __future__ import annotations

from typing import Optional, Tuple


def resolve_frame_bounds(
    start_frame: Optional[int],
    end_frame: Optional[int],
    start_time: Optional[float],
    end_time: Optional[float],
    fps: Optional[float],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (start_frame, end_frame); either may be None, meaning "no
    lower/upper bound". Raises ValueError if time bounds are given
    without fps.
    """
    if start_time is not None or end_time is not None:
        if fps is None:
            raise ValueError("fps is required when using start_time/end_time.")
        if start_time is not None:
            start_frame = int(round(start_time * fps)) + 1
        if end_time is not None:
            end_frame = int(round(end_time * fps)) + 1
    return start_frame, end_frame
