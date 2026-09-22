"""Per-frame swimmer bounding boxes via median background subtraction.

The camera is fixed and the swimmer is the only large thing moving, so
subtracting a median background plate (frames.median_background) is
enough to isolate them -- no detector, no model weights, no GPU. That
also sidesteps the scale problem that breaks COCO-trained person
detectors here: the swimmer is ~0.3% of the frame, far too small for a
detector that downscales the whole image, but perfectly visible to a
difference image at native resolution.

Two things stop this from being naive frame differencing:

  * Surface ripple and swaying lane ropes also move. `roi` restricts
    the search to a horizontal band, and `min_area` drops speckle.
  * The largest blob isn't always the swimmer. Once a box is
    established, candidates are scored by distance from where the
    swimmer was predicted to be, so tracking stays on the same object
    instead of jumping to whatever is briefly biggest.

Output is deliberately raw -- the detected box and blob centroid per
frame, unsmoothed. Smoothing belongs downstream in analysis.py, where
the window is a tunable choice; smoothing here would destroy
information that later stages can't recover.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .frames import fps as video_fps
from .frames import frame_size, iter_frames, median_background

BOX_COLUMNS = [
    "frame",
    "found",
    "box_x",
    "box_y",
    "box_w",
    "box_h",
    "centroid_x",
    "centroid_y",
    "area",
]


def fixed_box(
    centroid_x: float,
    centroid_y: float,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """
    A box of exactly (width, height) centred on the centroid, shifted
    inward so it stays inside the frame.

    Shifted rather than clipped on purpose: a fixed crop size is what
    makes landmark coordinates comparable between frames, so the size
    must not shrink near the edges of the frame. Size is only reduced
    if the requested box is larger than the frame itself.
    """
    width = min(width, frame_width)
    height = min(height, frame_height)

    x = int(round(centroid_x - width / 2))
    y = int(round(centroid_y - height / 2))
    x = max(0, min(x, frame_width - width))
    y = max(0, min(y, frame_height - height))
    return x, y, width, height


def crop_to_frame_norm(
    crop_x: float,
    crop_y: float,
    box: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float]:
    """
    Convert normalized [0,1] coordinates inside a crop back to normalized
    [0,1] coordinates of the full frame.

    This is what keeps the crop from leaking into stored data: detection runs
    on the crop, but everything written to disk is full-frame, so downstream
    consumers never need to know a crop happened.
    """
    box_x, box_y, box_w, box_h = box
    return (
        (box_x + crop_x * box_w) / frame_width,
        (box_y + crop_y * box_h) / frame_height,
    )


def estimate_translation(
    frame_gray: np.ndarray, reference_gray: np.ndarray
) -> Tuple[float, float, float]:
    """
    (dx, dy, response): the sub-pixel shift mapping `reference_gray` onto
    `frame_gray`, by phase correlation.

    Phase correlation compares the two images in the frequency domain, where a
    spatial shift is a phase ramp, so the answer falls out as the location of a
    single correlation peak. That makes it both fast and largely indifferent to
    brightness changes -- useful underwater, where light flickers constantly.

    It only recovers translation. Rotation or zoom will not be corrected and
    will instead show up as a weak `response`, which is the caller's cue that
    the estimate should not be trusted.
    """
    shift, response = cv2.phaseCorrelate(
        np.float32(reference_gray), np.float32(frame_gray)
    )
    return float(shift[0]), float(shift[1]), float(response)


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate an image by a (sub-pixel) offset, leaving vacated edges black."""
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


def _blank_border(diff: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Zero the edge strip that a shift pulled in from outside the image.

    Those pixels have no counterpart in the other image, so their difference is
    meaningless -- and large. Left alone they form a bright frame-wide border
    blob that outweighs the swimmer.
    """
    height, width = diff.shape[:2]
    left = int(np.ceil(abs(dx))) if dx > 0 else 0
    right = int(np.ceil(abs(dx))) if dx < 0 else 0
    top = int(np.ceil(abs(dy))) if dy > 0 else 0
    bottom = int(np.ceil(abs(dy))) if dy < 0 else 0

    if left:
        diff[:, :min(left, width)] = 0
    if right:
        diff[:, max(0, width - right):] = 0
    if top:
        diff[:min(top, height), :] = 0
    if bottom:
        diff[max(0, height - bottom):, :] = 0
    return diff


def _binary_diff(
    frame: np.ndarray,
    background_gray: np.ndarray,
    threshold: Optional[int],
    blur: int,
    roi: Optional[Tuple[int, int]],
    sigma: float,
    stabilize: bool = False,
    min_response: float = 0.02,
    min_shift: float = 2.5,
) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 0)

    shift = (0.0, 0.0)
    if stabilize:
        dx, dy, response = estimate_translation(gray, background_gray)
        # Below min_shift, correcting costs more than it fixes: warping
        # resamples (and so slightly blurs) the plate, and blanking the border
        # discards usable frame. Surface ripple alone produces a confident but
        # meaningless sub-pixel estimate, and acting on it measurably worsened
        # detection on steady footage (94% -> 90% tracked).
        #
        # The default separates the two regimes by measurement rather than
        # taste: on a fixed camera the estimate never exceeded ~2.2px, while
        # genuinely drifting footage reached 8px at p90 and 62px at worst.
        # Anything under a couple of pixels is ripple; real movement is larger.
        if response >= min_response and np.hypot(dx, dy) >= min_shift:
            # The *plate* is warped onto the frame, never the other way round.
            # Warping the frame would leave every detection in shifted
            # coordinates that then have to be undone, and any mistake there
            # silently biases every position downstream. Moving the reference
            # instead keeps detections in true frame coordinates throughout.
            background_gray = shift_image(background_gray, dx, dy)
            shift = (dx, dy)

    diff = cv2.absdiff(gray, background_gray)
    if shift != (0.0, 0.0):
        diff = _blank_border(diff, shift[0], shift[1])

    # Everything outside the ROI is zeroed *before* thresholding, not filtered
    # out afterwards. On this footage the lane rope sways hard enough to average
    # ~4x the difference signal of the swimmer's own band, so a threshold
    # computed over the whole frame is set by the rope and buries the swimmer.
    if roi is not None:
        masked = np.zeros_like(diff)
        y0, y1 = max(0, roi[0]), min(diff.shape[0], roi[1])
        masked[y0:y1] = diff[y0:y1]
        diff = masked
        sample = diff[y0:y1]
    else:
        sample = diff

    if threshold is None:
        # Median + k*MAD, not Otsu: Otsu assumes a bimodal histogram, but the
        # swimmer is well under 1% of the pixels, so Otsu is badly biased here.
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        threshold = int(median + sigma * max(mad, 1.0))

    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def suggest_roi(video_path: str, max_samples: int = 60, quantile: float = 0.75) -> Tuple[int, int]:
    """
    Report the rows worth searching, by measuring how much each image row
    moves across the clip.

    Static structure (pool floor, walls) barely changes; a swaying lane rope
    changes constantly in every frame; the swimmer only disturbs a few columns
    at a time, so it contributes little to a row's average. Rows above
    `quantile` of the row-energy distribution are therefore rope-like and get
    excluded, and the largest surviving contiguous band is returned.
    """
    background = median_background(video_path, max_samples=max_samples)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    energies = []
    for _, frame in iter_frames(video_path, stride=max(1, 300 // max_samples)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        energies.append(cv2.absdiff(gray, background_gray).mean(axis=1))
        if len(energies) >= max_samples:
            break

    row_energy = np.mean(np.stack(energies), axis=0)
    cutoff = float(np.quantile(row_energy, quantile))

    best, current = (0, 0), None
    for y, value in enumerate(row_energy):
        if value <= cutoff:
            current = (current[0], y + 1) if current else (y, y + 1)
            if current[1] - current[0] > best[1] - best[0]:
                best = current
        else:
            current = None
    return best


def _pick_component(
    binary: np.ndarray,
    min_area: int,
    roi: Optional[Tuple[int, int]],
    predicted: Optional[Tuple[float, float]],
    max_jump: float,
    area_reference: Optional[float],
    area_ratio: float,
):
    """
    Return (box, centroid, area) for the chosen blob, or None if nothing
    passes the gates.

    Returning None matters more than it looks. The obvious fallback --
    "take the biggest blob instead" -- is precisely how a tracker hops
    onto a swimmer in the next lane when the two cross, and it does so
    silently. Reporting the frame as lost keeps that failure visible
    instead of quietly corrupting every downstream measurement.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    for i in range(1, count):  # 0 is the background label
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        if roi is not None and not (roi[0] <= cy <= roi[1]):
            continue
        # A blob several times the size of the one we've been following is a
        # different object, however close it happens to be right now.
        if area_reference is not None:
            ratio = area / area_reference
            if ratio > area_ratio or ratio < 1.0 / area_ratio:
                continue
        box = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        candidates.append((box, (cx, cy), area))

    if not candidates:
        return None
    if predicted is None:
        return max(candidates, key=lambda c: c[2])

    px, py = predicted
    nearest = min(candidates, key=lambda c: np.hypot(c[1][0] - px, c[1][1] - py))
    if np.hypot(nearest[1][0] - px, nearest[1][1] - py) > max_jump:
        return None
    return nearest


def detect_boxes(
    video_path: str,
    background: Optional[np.ndarray] = None,
    roi: Optional[Tuple[int, int]] = None,
    seed_point: Optional[Tuple[int, int]] = None,
    min_area: int = 80,
    threshold: Optional[int] = None,
    sigma: float = 6.0,
    max_jump: float = 60.0,
    area_ratio: float = 3.0,
    area_anchor_samples: int = 15,
    max_radius_factor: float = 4.0,
    stabilize: bool = False,
    min_response: float = 0.02,
    min_shift: float = 2.5,
    blur: int = 5,
    max_samples: int = 120,
    progress: bool = True,
) -> pd.DataFrame:
    """
    One row per video frame: the detected box, blob centroid and area,
    with found=False on frames where nothing plausible was seen.

    roi: (y_min, y_max) band to search, which is the main defence against
    locking onto a swaying lane rope -- see suggest_roi(). It gates the
    threshold calculation too, not just candidate selection.
    sigma: threshold in robust deviations (MAD) above the ROI's median
    difference. Lower it if the swimmer is being missed, raise it if
    ripple is being picked up.
    max_jump: how far (px) the centroid may move between frames before
    the match is treated as implausible.
    stabilize: compensate camera drift by phase-correlating each frame
    against the background plate before differencing. Background
    subtraction assumes a fixed camera; without this, a camera that
    wanders even 25px turns every high-contrast edge in the scene --
    lane ropes, floor lines, tile borders -- into false motion.
    seed_point: (x, y) on frame 1 identifying *which* swimmer to follow.
    Without it the largest blob wins, which on any footage with more
    than one lane occupied is usually the wrong person -- whoever is
    nearest the camera looks biggest. Strongly recommended.
    """
    if background is None:
        if progress:
            print(f"Building median background from up to {max_samples} frames...")
        background = median_background(video_path, max_samples=max_samples)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        background_gray = cv2.GaussianBlur(background_gray, (blur | 1, blur | 1), 0)

    rows = []
    predicted: Optional[Tuple[float, float]] = (
        (float(seed_point[0]), float(seed_point[1])) if seed_point else None
    )
    previous: Optional[Tuple[float, float]] = None
    velocity = (0.0, 0.0)
    early_areas: List[float] = []
    coasting = 0

    for number, frame in iter_frames(video_path):
        binary = _binary_diff(frame, background_gray, threshold, blur, roi, sigma,
                              stabilize=stabilize, min_response=min_response,
                              min_shift=min_shift)
        # Anchored to the first few accepted areas, never a rolling window. A
        # rolling reference drifts: each slightly-larger blob shifts the median,
        # which admits a larger one still, and the gate walks itself onto a
        # different swimmer a few frames at a time.
        area_reference = (
            float(np.median(early_areas)) if len(early_areas) >= area_anchor_samples else None
        )
        # The search radius widens the longer the swimmer has been missing, so
        # re-acquisition stays local. Falling back to "largest blob anywhere"
        # lets the box teleport across the frame onto another lane entirely.
        radius = max_jump * min(1.0 + coasting, max_radius_factor)
        hit = _pick_component(
            binary, min_area, roi, predicted, radius, area_reference, area_ratio
        )

        if hit is None:
            rows.append({"frame": number, "found": False, "box_x": None, "box_y": None,
                         "box_w": None, "box_h": None, "centroid_x": None,
                         "centroid_y": None, "area": 0})
            # Coast on the last known velocity for a few frames: a swimmer
            # briefly occluded (bubbles, a crossing swimmer) is still where
            # physics says they are, so re-acquisition should find them.
            coasting += 1
            previous = None  # velocity estimate across a gap is meaningless
            if predicted is not None:
                # Coast forward, but decay the velocity: a stale motion estimate
                # shouldn't keep flinging the search window down the pool.
                velocity = (velocity[0] * 0.8, velocity[1] * 0.8)
                predicted = (predicted[0] + velocity[0], predicted[1] + velocity[1])
            continue

        (bx, by, bw, bh), (cx, cy), area = hit
        rows.append({"frame": number, "found": True, "box_x": bx, "box_y": by,
                     "box_w": bw, "box_h": bh, "centroid_x": cx,
                     "centroid_y": cy, "area": area})

        velocity = (cx - previous[0], cy - previous[1]) if previous else (0.0, 0.0)
        predicted = (cx + velocity[0], cy + velocity[1])
        previous = (cx, cy)
        coasting = 0
        if len(early_areas) < area_anchor_samples:
            early_areas.append(float(area))

    table = pd.DataFrame(rows, columns=BOX_COLUMNS)
    if progress:
        found = int(table["found"].sum())
        print(f"Tracked {found}/{len(table)} frames ({found / max(len(table), 1):.0%}).")
    return table


def draw_preview(video_path: str, boxes: pd.DataFrame, out_video: str) -> None:
    """Write a copy of the video with the tracked box drawn on it."""
    width, height = frame_size(video_path)
    writer = cv2.VideoWriter(
        out_video, cv2.VideoWriter_fourcc(*"mp4v"), video_fps(video_path), (width, height)
    )
    by_frame = boxes.set_index("frame")
    try:
        for number, frame in iter_frames(video_path):
            if number in by_frame.index:
                row = by_frame.loc[number]
                if bool(row["found"]):
                    x, y = int(row["box_x"]), int(row["box_y"])
                    w, h = int(row["box_w"]), int(row["box_h"])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (int(row["centroid_x"]), int(row["centroid_y"])),
                               4, (0, 0, 255), -1)
                else:
                    cv2.putText(frame, "LOST", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Preview video saved to: {os.path.abspath(out_video)}")


def contact_sheet(
    video_path: str,
    boxes: pd.DataFrame,
    out_path: str,
    stride: int = 10,
    columns: int = 8,
    cell: Tuple[int, int] = (160, 160),
    pad: int = 40,
) -> None:
    """
    Grid of cropped thumbnails, every `stride` frames, so a whole clip's
    tracking can be eyeballed at once -- far faster than scrubbing a
    preview video to find the frames where tracking let go.
    """
    by_frame = boxes.set_index("frame")
    crops = []
    for number, frame in iter_frames(video_path, stride=stride):
        if number not in by_frame.index or not bool(by_frame.loc[number, "found"]):
            crops.append((number, np.zeros((cell[1], cell[0], 3), dtype=np.uint8)))
            continue
        row = by_frame.loc[number]
        cx, cy = float(row["centroid_x"]), float(row["centroid_y"])
        height, width = frame.shape[:2]
        x, y, w, h = fixed_box(cx, cy, cell[0] + pad, cell[1] + pad, width, height)
        crops.append((number, cv2.resize(frame[y:y + h, x:x + w], cell)))

    if not crops:
        raise RuntimeError("No frames to build a contact sheet from.")

    rows = (len(crops) + columns - 1) // columns
    sheet = np.zeros((rows * (cell[1] + 18), columns * cell[0], 3), dtype=np.uint8)
    for index, (number, crop) in enumerate(crops):
        r, c = divmod(index, columns)
        top = r * (cell[1] + 18)
        sheet[top:top + cell[1], c * cell[0]:(c + 1) * cell[0]] = crop
        cv2.putText(sheet, str(number), (c * cell[0] + 4, top + cell[1] + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(out_path, sheet)
    print(f"Contact sheet ({len(crops)} frames) saved to: {os.path.abspath(out_path)}")
