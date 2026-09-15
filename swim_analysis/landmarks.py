"""Shared landmark constants and column-name helpers.

MediaPipe Pose always produces the same 33 landmarks in the same
order. This module is the single source of truth for that order, and
for figuring out how those 33 landmarks show up as CSV columns --
either by name (e.g. ``left_shoulder_x``) or by index
(e.g. ``11_x``), plus an optional index -> friendly-name mapping file
for the numbered case.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional, Tuple

LANDMARK_NAMES: List[str] = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

NUM_LANDMARKS = len(LANDMARK_NAMES)  # 33

POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),   # Face
    (9, 10),                                                          # Mouth
    (11, 12),                                                         # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),       # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),       # Right arm
    (11, 23), (12, 24), (23, 24),                                     # Torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),                 # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),                 # Right leg
]

DEFAULT_VISIBILITY_THRESHOLD = 0.5

FIELDS = ("x", "y", "z", "visibility")

# MediaPipe also returns pose_world_landmarks: the same 33 points in metres
# with the origin at the midpoint of the hips. Unlike the image-space columns
# these carry no absolute position (so they can't give velocity), but they are
# metric and independent of how the frame was cropped, which makes them the
# right source for joint angles -- a 2D angle is corrupted by foreshortening
# whenever a limb points toward the camera.
WORLD_FIELDS = ("wx", "wy", "wz")

# Written alongside the landmarks when extraction runs on a crop, so full-frame
# position stays recoverable and the crop remains auditable after the fact.
BOX_FIELDS = ("box_x", "box_y", "box_w", "box_h")


def named_header(world: bool = False, box: bool = False) -> List[str]:
    """Header row using landmark names: left_shoulder_x, left_shoulder_y, ...

    Landmark x/y are always stored in FULL-FRAME normalized coordinates, even
    when detection ran on a crop -- the crop is an implementation detail of
    getting MediaPipe to see a big enough subject, and letting it leak into the
    stored coordinates would silently break every consumer that assumes
    x * frame_width gives a pixel position.
    """
    header = ["frame"]
    if box:
        header += list(BOX_FIELDS)
    for name in LANDMARK_NAMES:
        header += [f"{name}_{f}" for f in FIELDS]
    if world:
        for name in LANDMARK_NAMES:
            header += [f"{name}_{f}" for f in WORLD_FIELDS]
    return header


def world_columns(index: int, style: str = "named") -> Tuple[str, str, str]:
    """(wx_col, wy_col, wz_col) for a landmark index."""
    if style == "named":
        key = LANDMARK_NAMES[index]
    elif style == "numbered":
        key = str(index)
    else:
        raise ValueError(f"Unknown column style: {style!r}")
    return tuple(f"{key}_{f}" for f in WORLD_FIELDS)  # type: ignore[return-value]


def numbered_header() -> List[str]:
    """Header row using landmark indices: 11_x, 11_y, 11_z, 11_visibility, ..."""
    header = ["frame"]
    for i in range(NUM_LANDMARKS):
        header += [f"{i}_{f}" for f in FIELDS]
    return header


def load_index_to_name_map(mapping_path: Optional[str]) -> Dict[int, str]:
    """
    Load an index -> name mapping file for CSVs that only have numbered
    columns. Expected format, no header required:

        0,nose
        1,left_eye_inner
        ...

    If mapping_path is None, falls back to the built-in LANDMARK_NAMES
    order (index i -> LANDMARK_NAMES[i]).
    """
    if mapping_path is None:
        return {i: name for i, name in enumerate(LANDMARK_NAMES)}

    mapping: Dict[int, str] = {}
    with open(mapping_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            idx_raw, name = row[0].strip(), row[1].strip()
            try:
                idx = int(idx_raw)
            except ValueError:
                continue  # skip a header row like "index,name"
            mapping[idx] = name
    return mapping


def detect_column_style(columns: List[str]) -> str:
    """
    Inspect a DataFrame's columns and return 'named' or 'numbered'
    depending on whether landmark columns look like 'left_shoulder_x'
    or '11_x'.
    """
    for name in LANDMARK_NAMES:
        if f"{name}_x" in columns:
            return "named"
    for i in range(NUM_LANDMARKS):
        if f"{i}_x" in columns:
            return "numbered"
    raise ValueError(
        "Could not detect column style: no recognizable "
        "'<name>_x' or '<index>_x' columns found in the CSV header."
    )


def landmark_columns(index: int, style: str) -> Tuple[str, str, str, str]:
    """Return (x_col, y_col, z_col, visibility_col) for a landmark index."""
    if style == "named":
        key = LANDMARK_NAMES[index]
    elif style == "numbered":
        key = str(index)
    else:
        raise ValueError(f"Unknown column style: {style!r}")
    return tuple(f"{key}_{f}" for f in FIELDS)  # type: ignore[return-value]


def display_name(index: int, style: str, index_to_name: Dict[int, str]) -> str:
    """Friendly display name for a landmark, regardless of column style."""
    if style == "named":
        return LANDMARK_NAMES[index]
    return index_to_name.get(index, str(index))
