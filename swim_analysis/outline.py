"""Utilities for working with per-frame SAM2 outline masks.

A "mask" here is a single-channel image where nonzero pixels belong
to the swimmer's silhouette for that frame and zero pixels are
background. Masks are expected to live in a directory, one PNG file
per frame, named to match the same 1-indexed frame numbering used
everywhere else in this project (pose_extraction.py's frame_count,
the CSV's 'frame' column):

    frame_000001.png, frame_000002.png, ...

This module has no dependency on MediaPipe or the landmark CSV at
all -- it only ever looks at the outline itself. Two things are built
on top of it:

  * outline_rightmost_x_series() -- the rightmost outline pixel's
    x-position per frame (no landmark involved).
  * outside_distance_transform() -- a 0-inside / distance-outside
    field used by the hyperparameter-tuning penalty (tuning.py).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from .duration import resolve_frame_bounds

MASK_FILENAME_PATTERN = "frame_{frame:06d}.png"
_FRAME_FILE_RE = re.compile(r"frame_(\d+)\.png$")


def mask_path_for_frame(masks_dir: str, frame: int) -> str:
    return os.path.join(masks_dir, MASK_FILENAME_PATTERN.format(frame=frame))


def load_mask(masks_dir: str, frame: int) -> Optional[np.ndarray]:
    """
    Load the boolean mask (True = swimmer) for a frame. Returns None
    if no mask file exists for that frame at all (e.g. it's outside
    the range SAM2 was run over).
    """
    path = mask_path_for_frame(masks_dir, frame)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 127


def available_frames(masks_dir: str) -> list:
    """Sorted list of frame numbers that have a mask file on disk."""
    frames = []
    for path in glob.glob(os.path.join(masks_dir, "frame_*.png")):
        m = _FRAME_FILE_RE.search(os.path.basename(path))
        if m:
            frames.append(int(m.group(1)))
    return sorted(frames)


def rightmost_x(mask: np.ndarray) -> Optional[int]:
    """
    x (column) index of the rightmost True pixel in the mask, or None
    if the mask has no swimmer pixels at all (e.g. SAM2 lost the
    subject that frame).
    """
    cols_with_swimmer = np.any(mask, axis=0)
    nonzero_cols = np.nonzero(cols_with_swimmer)[0]
    if nonzero_cols.size == 0:
        return None
    return int(nonzero_cols[-1])


def outside_distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    0 everywhere inside the silhouette; Euclidean distance (in pixels)
    to the nearest silhouette pixel everywhere outside it. Distances
    *inside* the body are deliberately clamped to 0 rather than
    computed, since only "how far outside the outline" matters for
    the tuning penalty -- a landmark placed anywhere inside a
    correctly-detected body shouldn't be penalized at all.
    """
    outside = ~mask
    dist = distance_transform_edt(outside)
    dist[mask] = 0.0
    return dist


def outline_rightmost_x_series(
    masks_dir: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
) -> pd.DataFrame:
    """
    For each frame in the given range, the x-position of the
    rightmost pixel of the SAM2 outline mask -- purely a property of
    the outline, with no MediaPipe landmark involved.

    Frames with a missing mask file, or an empty mask, get a blank
    (None/NaN) value rather than being skipped, so output frame
    numbers stay contiguous and alignable with other per-frame data.
    """
    start_frame, end_frame = resolve_frame_bounds(
        start_frame, end_frame, start_time, end_time, fps
    )

    frames_on_disk = available_frames(masks_dir)
    if not frames_on_disk:
        raise RuntimeError(
            f"No mask files found in {masks_dir!r} "
            f"(expected files like frame_000001.png)."
        )

    lo = start_frame if start_frame is not None else frames_on_disk[0]
    hi = end_frame if end_frame is not None else frames_on_disk[-1]

    rows = []
    for frame in range(lo, hi + 1):
        mask = load_mask(masks_dir, frame)
        x = rightmost_x(mask) if mask is not None else None
        rows.append({"frame": frame, "rightmost_x": x})

    return pd.DataFrame(rows)
