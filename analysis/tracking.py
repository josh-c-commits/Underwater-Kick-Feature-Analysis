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
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .frames import iter_frames, median_background

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
    # robust extents of the blob (low/high percentile of its pixel x-coords) --
    # the leading edge is whichever one faces the direction of travel
    "edge_left",
    "edge_right",
    # camera offset from the background plate this frame; 0 without
    # stabilization, or when stabilization found no real camera motion
    "cam_dx",
    "cam_dy",
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


def correlation_window(shape: Tuple[int, int]) -> np.ndarray:
    """
    Hanning taper for phase correlation, sized to an image of `shape` (h, w).

    Phase correlation treats the image as periodic, so the hard jump where the
    right edge meets the left (and top meets bottom) becomes a strong feature --
    one that never moves, because the image border is fixed in frame
    coordinates. Untapered, that feature dominates whenever the scene itself is
    weak (a median plate smeared by drift) and drags every estimate toward zero:
    on footage with a known 47px drift it produced a 14.5px median error. With
    the taper fading the borders out, the same measurement came to 1.1px.
    """
    return cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)


def estimate_translation(
    frame_gray: np.ndarray,
    reference_gray: np.ndarray,
    window: Optional[np.ndarray] = None,
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
        np.float32(reference_gray), np.float32(frame_gray), window
    )
    return float(shift[0]), float(shift[1]), float(response)


def estimate_translation_consensus(
    frame_gray: np.ndarray,
    reference_gray: np.ndarray,
    grid: Tuple[int, int] = (4, 2),
    min_response: float = 0.5,
    min_fraction: float = 0.5,
) -> Tuple[float, float, float]:
    """
    (dx, dy, agreement): the shift mapping `reference_gray` onto `frame_gray`,
    as the median over a grid of tiles, where agreement is the fraction of
    tiles that correlated confidently. dx and dy are NaN when fewer than
    `min_fraction` of tiles did.

    Why tiles: a single whole-frame correlation reports whatever moves most
    coherently. With little background texture, that is the swimmer -- and
    their motion then gets "corrected" away as if it were the camera's, the
    swimmer is absorbed into the background plate, and tracking silently
    loses them. Real camera motion moves every tile at once, while a swimmer
    occupies one or two and is outvoted by the median. And if too few tiles
    carry texture to say anything, the honest answer is "unmeasurable", not
    a guess. On real footage tiles correlated at 0.62-0.95 (10th percentile
    to median); textureless ones sat near 0.3, hence the 0.5 default.
    """
    height, width = frame_gray.shape[:2]
    cols, rows = grid
    tile_h, tile_w = height // rows, width // cols
    if tile_h < 16 or tile_w < 16:
        cols, rows = 1, 1
        tile_h, tile_w = height, width
    window = correlation_window((tile_h, tile_w))
    reference = np.float32(reference_gray)
    frame = np.float32(frame_gray)

    shifts = []
    for row in range(rows):
        for col in range(cols):
            ys = slice(row * tile_h, (row + 1) * tile_h)
            xs = slice(col * tile_w, (col + 1) * tile_w)
            (dx, dy), response = cv2.phaseCorrelate(reference[ys, xs], frame[ys, xs], window)
            if response >= min_response:
                shifts.append((dx, dy))

    agreement = len(shifts) / float(rows * cols)
    if agreement < min_fraction:
        return float("nan"), float("nan"), agreement
    dx, dy = np.median(np.asarray(shifts), axis=0)
    return float(dx), float(dy), agreement


def _prepared_gray(frame: np.ndarray, blur: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 0)
    return gray.astype(np.float32)


def measure_camera_motion(
    video_path: str,
    plate_gray: np.ndarray,
    blur: int = 5,
    min_response: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two independent measurements of where the camera is on each frame, taken
    in a single read of the video, as (relative, absolute) arrays of shape (n, 2).

    relative: frame-to-frame shifts (tile consensus), summed from frame 1.
        Precise from one frame to the next (~0.04px), but tiny per-step biases
        accumulate -- about 7px over a 10-second clip, even with no camera
        motion at all.
    absolute: each frame against the plain median plate (whole frame, tapered).
        Never accumulates, but noisy frame to frame, and garbage whenever the
        plate is featureless or smeared by steady drift.

    Neither is usable alone; fuse_camera_path combines their strengths.
    """
    relative = [(0.0, 0.0)]
    absolute = []
    previous = None
    window = None
    for _, frame in iter_frames(video_path):
        gray = _prepared_gray(frame, blur)
        if window is None:
            window = correlation_window(gray.shape)
        (ax, ay), _ = cv2.phaseCorrelate(plate_gray, gray, window)
        absolute.append((ax, ay))
        if previous is not None:
            dx, dy, _ = estimate_translation_consensus(gray, previous, min_response=min_response)
            if not np.isfinite(dx):
                dx, dy = 0.0, 0.0  # no measurable camera motion this step
            last = relative[-1]
            relative.append((last[0] + dx, last[1] + dy))
        previous = gray
    return np.asarray(relative, dtype=float), np.asarray(absolute, dtype=float)


def fuse_camera_path(
    relative: np.ndarray,
    absolute: np.ndarray,
    window: int = 61,
    max_spread: float = 2.0,
) -> np.ndarray:
    """
    Complementary filter: the relative path's frame-to-frame precision, anchored
    by the absolute measurement so it can't wander off.

    The gap between them (absolute - relative) is the relative path's slowly
    accumulated error plus the absolute measurement's fast noise. A rolling
    median of that gap keeps the slow part, which is exactly the correction the
    relative path needs, and discards the noise.

    The gate is what makes this safe. Where the absolute measurement is valid,
    the gap barely moves within a window: a spread of 0.1-0.3px on real footage.
    Where it's garbage -- a featureless plate, or one smeared by drift -- the
    gap scatters by 9-60px. Scattering windows are discarded and bridged from
    trustworthy neighbours; if nothing is trustworthy, the relative path stands
    alone. Without this gate, a single textureless clip produced corrections
    that were off by 2500px.
    """
    gap = pd.DataFrame(absolute - relative)
    min_periods = max(3, window // 4)
    center = gap.rolling(window, center=True, min_periods=min_periods).median()
    spread = (gap - center).abs().rolling(window, center=True, min_periods=min_periods).median()
    trusted = (spread.max(axis=1) <= max_spread).to_numpy()

    correction = center.copy()
    correction.loc[~trusted, :] = np.nan
    correction = correction.interpolate(limit_direction="both").fillna(0.0)
    return relative + correction.to_numpy()


def camera_moved(
    offsets: np.ndarray, motion_floor: float = 4.0, min_moved_frames: int = 5
) -> bool:
    """
    Whether an estimated camera path shows motion beyond what water alone
    produces on a camera that isn't moving.

    Ripple fools phase correlation slightly on every frame. On a camera known to
    be still, estimated offsets had a median of 0.6px and never exceeded 3.3px;
    on genuinely drifting footage they sat around 12-14px. Applying ripple-noise
    offsets isn't harmless -- warping the plate to chase water cost 13 of 558
    tracked frames on a still clip -- so a clip counts as moving only if a few
    frames are displaced well past that noise. A count rather than a single
    maximum means one freak spike can't switch it on, while a real bump of a
    handful of frames still does.
    """
    centred = offsets - np.median(offsets, axis=0)
    magnitude = np.hypot(centred[:, 0], centred[:, 1])
    return int(np.sum(magnitude > motion_floor)) >= min_moved_frames


def stabilized_background(
    video_path: str,
    max_samples: int = 120,
    blur: int = 5,
    min_response: float = 0.5,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (plate, offsets): a background plate for a camera that moves, and each
    frame's camera offset from it, shape (n, 2).

    Both tracking and calibration build their reference through this, so they
    share one coordinate frame. A calibration clicked on one plate and positions
    measured against a different one would disagree by however far apart the
    two plates sit.
    """
    if progress:
        print("Measuring camera motion (pass 1 of 2)...")
    plain = median_background(video_path, max_samples=max_samples)
    relative, absolute = measure_camera_motion(
        video_path, _prepared_gray(plain, blur), blur, min_response
    )
    first_path = fuse_camera_path(relative, absolute)

    if not camera_moved(first_path):
        # Nothing to correct: the estimates are ripple noise. Returning the plain
        # plate and zero offsets makes stabilizing a steady clip an exact no-op
        # rather than a small, systematic degradation.
        if progress:
            print("No camera motion beyond water noise; treating the camera as fixed.")
        return plain, np.zeros_like(first_path)

    if progress:
        print(f"Building aligned background from up to {max_samples} frames...")
    plate = aligned_background(video_path, first_path, max_samples=max_samples)

    # Refinement. The first anchor was measured against the plain median, which
    # drift smears -- and matching against a smeared image pulls every estimate
    # toward its centre (it reported 94% of the true motion). Re-measuring
    # against the aligned plate removes that, and also expresses each offset
    # directly relative to the plate the offsets will be applied to. On footage
    # with a known 47px drift this took the median error from 0.81px to 0.52px
    # and the worst case from 5.1px to 3.4px.
    if progress:
        print("Measuring camera motion (pass 2 of 2)...")
    plate_gray = _prepared_gray(plate, blur)
    window = correlation_window(plate_gray.shape)
    refined_absolute = np.asarray(
        [cv2.phaseCorrelate(plate_gray, _prepared_gray(frame, blur), window)[0]
         for _, frame in iter_frames(video_path)],
        dtype=float,
    )
    return plate, fuse_camera_path(relative, refined_absolute)


def aligned_background(
    video_path: str,
    path: np.ndarray,
    max_samples: int = 120,
    min_align: float = 1.0,
    home: Optional[Tuple[float, float]] = None,
    start_frame: int = 1,
    end_frame: Optional[int] = None,
) -> np.ndarray:
    """
    Background plate built from frames shifted back into register first.

    A plain median of a drifting clip averages the scene across every camera
    position, so every edge is smeared. Undoing each sample's offset before
    taking the median keeps the plate sharp. Samples are aligned to the path's
    *median* position rather than frame 1: the opening frames are often the
    camera still being aimed, which makes frame 1 an unrepresentative home.

    Samples within min_align of home are used as-is. Resampling them would only
    blur the plate by interpolation for a sub-pixel gain, and on a steady camera
    -- where every offset is sub-pixel noise -- that blur alone cost 11 of 558
    tracked frames. With the threshold, a steady camera gets exactly the plain
    median back.

    home: where to align to instead of the path's median. Pass (0, 0) with the
    offsets from stabilized_background to land in that plate's coordinates.
    start_frame/end_frame: build from part of the clip only (1-indexed,
    inclusive). Together these give a calibration reference from just the
    frames where markers were down, in the same coordinates tracking uses.
    """
    last = len(path) if end_frame is None else min(end_frame, len(path))
    stride = max(1, (last - start_frame + 1) // max_samples)
    home = np.median(path, axis=0) if home is None else np.asarray(home, dtype=float)
    samples = []
    for number, frame in iter_frames(video_path, start_frame, last, stride):
        dx, dy = path[number - 1] - home
        if np.hypot(dx, dy) < min_align:
            samples.append(frame)
        else:
            height, width = frame.shape[:2]
            matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
            samples.append(cv2.warpAffine(frame, matrix, (width, height),
                                          borderMode=cv2.BORDER_REFLECT))
        if len(samples) >= max_samples:
            break
    if not samples:
        raise RuntimeError(f"No frames could be read from {video_path}.")
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def calibration_reference(
    video_path: str,
    frames: Optional[Tuple[int, int]] = None,
    stabilize: bool = False,
    max_samples: int = 120,
    progress: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    (reference, plate): the image calibration marks are clicked on, and the
    plate `track` measures positions against -- or None for the plate when it
    would take an extra pass to build and isn't the reference itself.

    frames: (first, last) to build the reference from only those frames, for
    markers that were on the pool floor for part of the clip. With stabilize,
    each of those frames is shifted into the stabilized plate's coordinates
    first, so the marks land where tracking will measure positions even if the
    camera moved between placing the markers and the swim.
    """
    if stabilize:
        plate, offsets = stabilized_background(video_path, max_samples=max_samples,
                                               progress=progress)
        if frames is None:
            return plate, plate
        if progress:
            print(f"Building the reference from frames {frames[0]}-{frames[1]}...")
        reference = aligned_background(video_path, offsets, max_samples=max_samples,
                                       home=(0.0, 0.0), start_frame=frames[0],
                                       end_frame=frames[1])
        return reference, plate
    if frames is None:
        if progress:
            print(f"Building a median reference image from {max_samples} frames...")
        reference = median_background(video_path, max_samples=max_samples)
        return reference, reference
    if progress:
        print(f"Building the reference from frames {frames[0]}-{frames[1]}...")
    return median_background(video_path, max_samples, frames[0], frames[1]), None


def plate_offset(image: np.ndarray, plate: np.ndarray, blur: int = 5) -> Optional[Tuple[float, float]]:
    """
    (dx, dy) of `image` relative to `plate`, or None when it can't be measured
    (too little texture, or the tiles disagree).

    Used to check that a calibration reference built from part of a clip sits
    where the tracking plate does. A camera nudged while markers were placed or
    removed offsets every clicked mark by the same amount, and nothing
    downstream would notice.
    """
    dx, dy, _ = estimate_translation_consensus(
        _prepared_gray(image, blur), _prepared_gray(plate, blur)
    )
    if not np.isfinite(dx):
        return None
    return dx, dy


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
    camera: Optional[Tuple[float, float]] = None,
    min_shift: float = 2.5,
) -> np.ndarray:
    """Foreground mask for one frame, given the camera's offset from the plate
    when stabilizing (None when the camera is taken as fixed)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 0)

    shift = (0.0, 0.0)
    if camera is not None and np.isfinite(camera[0]) and np.hypot(*camera) >= min_shift:
        # The *plate* moves onto the frame, never the reverse: shifting the
        # frame would leave every detection in displaced coordinates needing an
        # inverse correction, and any error there would quietly bias every
        # position downstream.
        #
        # Below min_shift the plate is left alone. Warping resamples (slightly
        # blurring) the plate and blanking the border discards usable frame; on
        # a steady camera that cost buys nothing, and acting on sub-2px offsets
        # measurably worsened detection there (94% -> 90% tracked). Positions
        # are still corrected by the full offset downstream -- this gate is only
        # about whether warping helps *detection*.
        background_gray = shift_image(background_gray, camera[0], camera[1])
        shift = (camera[0], camera[1])

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
    max_area_ratio: float,
    min_area_ratio: float,
    edge_percentile: float = 98.0,
):
    """
    Return a dict describing the chosen blob -- box, centroid, area, and
    robust left/right edges -- or None if nothing passes the gates.

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
        # different object, however close it happens to be right now. The two
        # bounds guard against different things: the upper one blocks
        # defection to a bigger (nearer) swimmer; the lower one rejects body
        # *fragments* -- an arm or the legs detected on their own, whose centroid
        # sits well off the body's. Measured, loosening it from 1/3 to 1/4 only
        # admitted detections a median 20px from the swimmer's true path, and
        # 1/5 derailed tracking on drifting footage (83 frames lost, not gained).
        if area_reference is not None:
            ratio = area / area_reference
            if ratio > max_area_ratio or ratio < min_area_ratio:
                continue
        box = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        candidates.append((box, (cx, cy), area, i))

    if not candidates:
        return None
    if predicted is None:
        chosen = max(candidates, key=lambda c: c[2])
    else:
        px, py = predicted
        chosen = min(candidates, key=lambda c: np.hypot(c[1][0] - px, c[1][1] - py))
        if np.hypot(chosen[1][0] - px, chosen[1][1] - py) > max_jump:
            return None

    (x, y, w, h), centroid, area, label = chosen
    # Percentiles of the blob's pixel columns rather than its extreme columns:
    # the box edge is set by the single outermost pixel, so it jumps with any
    # ripple that happens to touch the silhouette's boundary.
    columns = np.nonzero(labels[y:y + h, x:x + w] == label)[1] + x
    left, right = np.percentile(columns, [100.0 - edge_percentile, edge_percentile])
    return {"box": (x, y, w, h), "centroid": centroid, "area": area,
            "edges": (float(left), float(right))}


def detect_boxes(
    video_path: str,
    background: Optional[np.ndarray] = None,
    roi: Optional[Tuple[int, int]] = None,
    seed_point: Optional[Tuple[int, int]] = None,
    min_area: int = 80,
    threshold: Optional[int] = None,
    sigma: float = 6.0,
    max_jump: float = 60.0,
    max_area_ratio: float = 3.0,
    min_area_ratio: float = 1.0 / 3.0,
    area_anchor_samples: int = 15,
    edge_percentile: float = 98.0,
    max_radius_factor: float = 4.0,
    stabilize: bool = False,
    min_response: float = 0.5,
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
    offsets = None
    if stabilize:
        if background is not None:
            raise ValueError(
                "Pass background=None with stabilize=True: the plate has to be built "
                "from the estimated camera path, or offsets and plate won't agree."
            )
        background, offsets = stabilized_background(
            video_path, max_samples, blur, min_response, progress
        )
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
        if offsets is not None and number - 1 < len(offsets):
            cam_dx, cam_dy = float(offsets[number - 1][0]), float(offsets[number - 1][1])
            camera = (cam_dx, cam_dy)
        else:
            cam_dx, cam_dy, camera = 0.0, 0.0, None
        binary = _binary_diff(frame, background_gray, threshold, blur, roi, sigma,
                              camera=camera, min_shift=min_shift)
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
            binary, min_area, roi, predicted, radius, area_reference,
            max_area_ratio, min_area_ratio, edge_percentile,
        )

        if hit is None:
            # The camera offset is still recorded on lost frames: the camera
            # moved whether or not the swimmer was found, and an overlay drawn
            # on this frame needs to know where the pool is.
            rows.append({"frame": number, "found": False, "box_x": None, "box_y": None,
                         "box_w": None, "box_h": None, "centroid_x": None,
                         "centroid_y": None, "area": 0, "edge_left": None,
                         "edge_right": None, "cam_dx": cam_dx, "cam_dy": cam_dy})
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

        (bx, by, bw, bh), (cx, cy), area = hit["box"], hit["centroid"], hit["area"]
        rows.append({"frame": number, "found": True, "box_x": bx, "box_y": by,
                     "box_w": bw, "box_h": bh, "centroid_x": cx,
                     "centroid_y": cy, "area": area, "edge_left": hit["edges"][0],
                     "edge_right": hit["edges"][1], "cam_dx": cam_dx, "cam_dy": cam_dy})

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
