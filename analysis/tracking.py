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

from .frames import fps as video_fps
from .frames import frame_size, iter_frames, median_background, stack_median

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
    # how the row was obtained: "detected", "keyframe" (marked by hand) or "lost"
    "source",
    # forward and backward tracking chose different blobs here; worth reviewing
    "conflict",
    # blob area over the swimmer's lower-quartile area within 3 s either side
    "size_ratio",
    # bigger and taller than the swimmer alone: another swimmer has probably
    # overlapped the subject and merged into the same blob (association._flag_merges)
    "merged",
]

# Every blob that differs from the background, per frame (see detect_candidates).
CANDIDATE_COLUMNS = ["frame", "box_x", "box_y", "box_w", "box_h", "centroid_x", "centroid_y",
                     "area", "edge_left", "edge_right"]


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
    the taper fading the borders out, the same measurement came to 0.5px.
    """
    return cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)


def _phase_correlate(
    reference: np.ndarray, image: np.ndarray, window: Optional[np.ndarray] = None
) -> Tuple[Tuple[float, float], float]:
    """
    cv2.phaseCorrelate without its side effect: given a window, OpenCV
    multiplies *both input arrays* by it in place before correlating.

    Callers here reuse their images. The plate is compared against every frame,
    so it was faded once per frame until only a patch at its centre survived,
    and each frame reached the next frame-to-frame match already tapered. On a
    steady 1080p clip that turned a few pixels of drift into 35px of invented
    camera motion. Tapering copies instead leaves the inputs as they were.
    """
    reference, image = np.float32(reference), np.float32(image)
    if window is not None:
        reference, image = reference * window, image * window
    return cv2.phaseCorrelate(reference, image)


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
    shift, response = _phase_correlate(reference_gray, frame_gray, window)
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

    shifts = []
    for row in range(rows):
        for col in range(cols):
            ys = slice(row * tile_h, (row + 1) * tile_h)
            xs = slice(col * tile_w, (col + 1) * tile_w)
            (dx, dy), response = _phase_correlate(reference_gray[ys, xs], frame_gray[ys, xs],
                                                  window)
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
    ignore_above: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two independent measurements of where the camera is on each frame, taken
    in a single read of the video, as (relative, absolute) arrays of shape (n, 2).

    relative: frame-to-frame shifts (tile consensus), summed from frame 1.
        Precise from one frame to the next (~0.04px), but biased: caustics and
        surface shimmer move a fraction of a pixel per frame, too close to the
        static pool's zero to separate, so each match is pulled slightly toward
        the water's motion. Summed, that's ~10px over a 10-second cropped clip
        and ~60px over the full 1080p frame, with no camera motion at all.
    absolute: each frame against the plain median plate (whole frame, tapered).
        Never accumulates, but noisy frame to frame, and garbage whenever the
        plate is featureless or smeared by steady drift.

    Neither is usable alone; fuse_camera_path combines their strengths.

    ignore_above: rows above this are left out of both measurements. The
    water surface moves constantly and coherently, which is exactly what
    pulls these estimates off; the camera's motion shows just as well below it.
    """
    top = max(0, ignore_above or 0)
    plate_gray = plate_gray[top:]
    relative = [(0.0, 0.0)]
    absolute = []
    previous = None
    window = None
    for _, frame in iter_frames(video_path):
        gray = _prepared_gray(frame, blur)[top:]
        if window is None:
            window = correlation_window(gray.shape)
        (ax, ay), _ = _phase_correlate(plate_gray, gray, window)
        absolute.append((ax, ay))
        if previous is not None:
            dx, dy, _ = estimate_translation_consensus(gray, previous, min_response=min_response)
            if not np.isfinite(dx):
                dx, dy = 0.0, 0.0  # no measurable camera motion this step
            last = relative[-1]
            relative.append((last[0] + dx, last[1] + dy))
        previous = gray
    return np.asarray(relative, dtype=float), np.asarray(absolute, dtype=float)


def _robust_line_at(values: np.ndarray, index: int, half: int) -> np.ndarray:
    """
    A Theil-Sen line through values[index - half : index + half + 1] (clipped to
    the array), read off at `index`, per column. The slope is the median of all
    pairwise slopes and the level the median of what's left after removing it,
    so a few garbage points can't drag either.
    """
    lo, hi = max(0, index - half), min(len(values), index + half + 1)
    t = np.arange(lo, hi, dtype=float) - index
    first, second = np.triu_indices(hi - lo, 1)
    levels = []
    for column in values[lo:hi].T:
        slope = np.median((column[second] - column[first]) / (t[second] - t[first]))
        levels.append(np.median(column - slope * t))
    return np.asarray(levels)


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

    Within half a window of either end of the clip, the window is cut off on
    one side, and a median of a drifting gap lags behind: on the first frame it
    reports the gap as it stood ~15 frames later. The relative path always
    drifts (see measure_camera_motion), so those frames use a robust line
    through the available window instead, read at the frame itself. On a steady full-frame clip the median alone left
    ~4px of error at the ends, nearly enough to declare the camera moving; on
    footage with a known drift, the line cut the worst error from 3.4px to
    2.7px and the last 30 frames' from 1.0px to 0.4px. A clip shorter than the
    window is all ends, and on a 40-frame one the median error fell from 1.4px
    to 0.4px.
    """
    gap = pd.DataFrame(np.asarray(absolute, dtype=float) - np.asarray(relative, dtype=float))
    min_periods = max(3, window // 4)
    center = gap.rolling(window, center=True, min_periods=min_periods).median()
    count, half = len(gap), window // 2
    if count >= min_periods:
        values = gap.to_numpy()
        for index in sorted(set(range(min(half, count))) | set(range(max(0, count - half), count))):
            center.iloc[index] = _robust_line_at(values, index, half)
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

    Ripple fools phase correlation slightly on every frame. On cameras known to
    be still, cropped or full frame, estimated offsets never exceeded 2.9px;
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
    ignore_above: Optional[int] = None,
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
    top = max(0, ignore_above or 0)
    plain = median_background(video_path, max_samples=max_samples, ignore_above=ignore_above)
    relative, absolute = measure_camera_motion(
        video_path, _prepared_gray(plain, blur), blur, min_response, ignore_above
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
    plate = aligned_background(video_path, first_path, max_samples=max_samples,
                               ignore_above=ignore_above)

    # Refinement. The first anchor was measured against the plain median, which
    # drift smears -- and matching against a smeared image pulls every estimate
    # toward its centre (it reported 94% of the true motion). Re-measuring
    # against the aligned plate removes that, and also expresses each offset
    # directly relative to the plate the offsets will be applied to. On footage
    # with a known 47px drift this took the median error from 0.68px to 0.45px
    # and the worst case from 5.3px to 2.9px.
    if progress:
        print("Measuring camera motion (pass 2 of 2)...")
    plate_gray = _prepared_gray(plate, blur)[top:]
    window = correlation_window(plate_gray.shape)
    refined_absolute = np.asarray(
        [_phase_correlate(plate_gray, _prepared_gray(frame, blur)[top:], window)[0]
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
    ignore_above: Optional[int] = None,
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
    ignore_above: as in frames.median_background.
    """
    last = len(path) if end_frame is None else min(end_frame, len(path))
    stride = max(1, (last - start_frame + 1) // max_samples)
    expected = min(max_samples, len(range(start_frame, last + 1, stride)))
    home = np.median(path, axis=0) if home is None else np.asarray(home, dtype=float)
    top = max(0, ignore_above or 0)
    plate, stack, count = None, None, 0
    for number, frame in iter_frames(video_path, start_frame, last, stride):
        dx, dy = path[number - 1] - home
        if np.hypot(dx, dy) >= min_align:
            height, width = frame.shape[:2]
            matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
            frame = cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
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


def calibration_reference(
    video_path: str,
    frames: Optional[Tuple[int, int]] = None,
    stabilize: bool = False,
    max_samples: int = 120,
    progress: bool = False,
    ignore_above: Optional[int] = None,
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
                                               progress=progress, ignore_above=ignore_above)
        if frames is None:
            return plate, plate
        if progress:
            print(f"Building the reference from frames {frames[0]}-{frames[1]}...")
        reference = aligned_background(video_path, offsets, max_samples=max_samples,
                                       home=(0.0, 0.0), start_frame=frames[0],
                                       end_frame=frames[1], ignore_above=ignore_above)
        return reference, plate
    if frames is None:
        if progress:
            print(f"Building a median reference image from {max_samples} frames...")
        reference = median_background(video_path, max_samples=max_samples,
                                      ignore_above=ignore_above)
        return reference, reference
    if progress:
        print(f"Building the reference from frames {frames[0]}-{frames[1]}...")
    return median_background(video_path, max_samples, frames[0], frames[1], ignore_above), None


def plate_offset(
    image: np.ndarray, plate: np.ndarray, blur: int = 5, ignore_above: Optional[int] = None
) -> Optional[Tuple[float, float]]:
    """
    (dx, dy) of `image` relative to `plate`, or None when it can't be measured
    (too little texture, or the tiles disagree).

    Used to check that a calibration reference built from part of a clip sits
    where the tracking plate does. A camera nudged while markers were placed or
    removed offsets every clicked mark by the same amount, and nothing
    downstream would notice.
    """
    top = max(0, ignore_above or 0)
    dx, dy, _ = estimate_translation_consensus(
        _prepared_gray(image, blur)[top:], _prepared_gray(plate, blur)[top:]
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
    kernel_size: int = 5,
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def suggest_roi(
    video_path: str,
    max_samples: int = 60,
    busy_ratio: float = 4.0,
    ignore_above: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Report the rows worth searching, by measuring how much each image row
    moves across the clip.

    A swaying lane rope or the water surface moves across its whole row in
    every frame; the swimmer covers a few columns of theirs. So each row's
    motion is the *median* change across its columns, which a swimmer-sized
    minority of columns can't move. Rows more than `busy_ratio` times busier
    than a typical row are rope-like and excluded, and the largest calm band
    left is returned. (A mean across columns, as before, let a swimmer on 4K
    footage -- a tenth of the frame's width for most of the clip -- push their
    own rows out of the band.) Rows above `ignore_above` are never part of it.
    """
    background = median_background(video_path, max_samples=max_samples,
                                   ignore_above=ignore_above)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    energies = []
    for _, frame in iter_frames(video_path, stride=max(1, 300 // max_samples)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        energies.append(np.median(cv2.absdiff(gray, background_gray), axis=1))
        if len(energies) >= max_samples:
            break

    row_energy = np.mean(np.stack(energies), axis=0)
    top = max(0, ignore_above or 0)
    cutoff = busy_ratio * max(float(np.median(row_energy[top:])), 1.0)
    row_energy[:top] = np.inf

    best, current = (0, 0), None
    for y, value in enumerate(row_energy):
        if value <= cutoff:
            current = (current[0], y + 1) if current else (y, y + 1)
            if current[1] - current[0] > best[1] - best[0]:
                best = current
        else:
            current = None
    return best


def frame_candidates(
    binary: np.ndarray,
    min_area: int,
    roi: Optional[Tuple[int, int]] = None,
    edge_percentile: float = 98.0,
) -> List[dict]:
    """
    Every blob in one frame's foreground mask that is big enough to matter:
    its box, centroid, area, and robust left/right edges.

    Which of them is the swimmer is decided later (association.associate), with
    the whole clip and the keyframes in view.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for i in range(1, count):  # 0 is the background label
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        if roi is not None and not (roi[0] <= cy <= roi[1]):
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        # Percentiles of the blob's pixel columns rather than its extreme columns:
        # the box edge is set by the single outermost pixel, so it jumps with any
        # ripple that happens to touch the silhouette's boundary.
        columns = np.nonzero(labels[y:y + h, x:x + w] == i)[1] + x
        left, right = np.percentile(columns, [100.0 - edge_percentile, edge_percentile])
        blobs.append({"box_x": x, "box_y": y, "box_w": w, "box_h": h, "centroid_x": cx,
                      "centroid_y": cy, "area": area, "edge_left": float(left),
                      "edge_right": float(right)})
    return blobs


def _odd(value: float) -> int:
    return max(1, int(round(value)) // 2 * 2 + 1)


def detect_candidates(
    video_path: str,
    background: Optional[np.ndarray] = None,
    roi: Optional[Tuple[int, int]] = None,
    min_area: Optional[int] = None,
    threshold: Optional[int] = None,
    sigma: float = 6.0,
    edge_percentile: float = 98.0,
    stabilize: bool = False,
    min_response: float = 0.5,
    min_shift: float = 2.5,
    blur: Optional[int] = None,
    max_samples: int = 120,
    progress: bool = True,
    ignore_above: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    (candidates, camera): every blob that differs from the background in every
    frame (CANDIDATE_COLUMNS), and each frame's camera offset, shape (n, 2).

    roi: (y_min, y_max) band to search, which is the main defence against
    locking onto a swaying lane rope -- see suggest_roi(). It gates the
    threshold calculation too, not just which blobs are kept.
    sigma: threshold in robust deviations (MAD) above the ROI's median
    difference. Lower it if the swimmer is being missed, raise it if
    ripple is being picked up.
    stabilize: compensate camera drift by phase-correlating each frame
    against the background plate before differencing. Background
    subtraction assumes a fixed camera; without this, a camera that
    wanders even 25px turns every high-contrast edge in the scene --
    lane ropes, floor lines, tile borders -- into false motion.
    ignore_above: rows above this are never searched, and are left out of the
    background plate and camera-motion measurement -- for the water surface,
    whose constant motion is noise with nothing stable in it.
    min_area, blur: default to values tuned at 1920 px wide, scaled up for
    larger frames, so specks of ripple at 4K aren't taken for blobs.
    """
    width, _ = frame_size(video_path)
    scale = max(1.0, width / 1920.0)
    blur = blur if blur is not None else _odd(5 * scale)
    min_area = min_area if min_area is not None else int(round(80 * scale ** 2))
    kernel_size = _odd(5 * scale)

    offsets = None
    if stabilize:
        if background is not None:
            raise ValueError(
                "Pass background=None with stabilize=True: the plate has to be built "
                "from the estimated camera path, or offsets and plate won't agree."
            )
        background, offsets = stabilized_background(
            video_path, max_samples, blur, min_response, progress, ignore_above
        )
    if background is None:
        if progress:
            print(f"Building median background from up to {max_samples} frames...")
        background = median_background(video_path, max_samples=max_samples,
                                       ignore_above=ignore_above)
    if ignore_above:
        top = int(ignore_above)
        roi = (max(roi[0], top), roi[1]) if roi is not None else (top, background.shape[0])
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        background_gray = cv2.GaussianBlur(background_gray, (blur | 1, blur | 1), 0)

    rows, camera_path_ = [], []
    for number, frame in iter_frames(video_path):
        if offsets is not None and number - 1 < len(offsets):
            cam_dx, cam_dy = float(offsets[number - 1][0]), float(offsets[number - 1][1])
            camera = (cam_dx, cam_dy)
        else:
            cam_dx, cam_dy, camera = 0.0, 0.0, None
        binary = _binary_diff(frame, background_gray, threshold, blur, roi, sigma,
                              camera=camera, min_shift=min_shift, kernel_size=kernel_size)
        for blob in frame_candidates(binary, min_area, roi, edge_percentile):
            rows.append({"frame": number, **blob})
        camera_path_.append((cam_dx, cam_dy))
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS), np.asarray(camera_path_, dtype=float)


def boxes_from_candidates(
    blobs: pd.DataFrame,
    camera: np.ndarray,
    fps: float,
    width: int,
    keyframes=(),
    motion=None,
    max_area_ratio: float = 3.0,
    min_area_ratio: float = 1.0 / 3.0,
) -> pd.DataFrame:
    """Association on already-detected candidates, as a BOX_COLUMNS table. Shared
    by detect_boxes and the reviewer, which reruns it after every keyframe edit."""
    from .association import associate

    table = associate(blobs, len(camera), fps, keyframes, motion, max_area_ratio,
                      min_area_ratio, default_length=0.05 * width)
    table["cam_dx"], table["cam_dy"] = camera[:, 0], camera[:, 1]
    table = table.reindex(columns=BOX_COLUMNS)
    table["area"] = table["area"].fillna(0)
    return table


def candidates_path(boxes_csv: str) -> str:
    """Where `track` keeps the candidates behind a boxes table, for the reviewer."""
    return os.path.splitext(boxes_csv)[0] + "_candidates.csv"


def detect_boxes(
    video_path: str,
    background: Optional[np.ndarray] = None,
    roi: Optional[Tuple[int, int]] = None,
    seed_point: Optional[Tuple[float, float]] = None,
    keyframes=None,
    min_area: Optional[int] = None,
    threshold: Optional[int] = None,
    sigma: float = 6.0,
    max_area_ratio: float = 3.0,
    min_area_ratio: float = 1.0 / 3.0,
    edge_percentile: float = 98.0,
    stabilize: bool = False,
    min_response: float = 0.5,
    min_shift: float = 2.5,
    blur: Optional[int] = None,
    max_samples: int = 120,
    progress: bool = True,
    ignore_above: Optional[int] = None,
    motion=None,
    candidates: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    One row per video frame (BOX_COLUMNS): the swimmer's box, centroid and
    area, with found=False on frames where nothing plausible was seen.

    Detection (detect_candidates, see its arguments) finds every moving blob;
    association.associate then decides which is the swimmer, following them
    forwards and backwards from each keyframe.

    keyframes: association.Keyframe list marking the swimmer (or their
    absence) on any frames. seed_point: (x, y) on frame 1, a shorthand for one
    keyframe there. Without either, the largest blob wins, which on footage
    with more than one lane occupied is usually the wrong person -- whoever
    is nearest the camera looks biggest. Strongly recommended.
    max_area_ratio / min_area_ratio: blobs more than this many times bigger or
    smaller than the swimmer at the keyframe are someone else, or a fragment.
    candidates: detect_candidates' output, to re-associate without reading
    the video again.
    """
    from .association import Keyframe

    if candidates is None:
        candidates = detect_candidates(
            video_path, background, roi, min_area, threshold, sigma, edge_percentile,
            stabilize, min_response, min_shift, blur, max_samples, progress, ignore_above,
        )
    blobs, camera = candidates
    keys = list(keyframes or [])
    if seed_point is not None and not any(k.frame == 1 for k in keys):
        keys.append(Keyframe(1, float(seed_point[0]), float(seed_point[1])))
    width, _ = frame_size(video_path)
    table = boxes_from_candidates(blobs, camera, video_fps(video_path), width, keys, motion,
                                  max_area_ratio, min_area_ratio)
    if progress:
        found = int(table["found"].sum())
        conflicts = int(table["conflict"].sum())
        print(f"Tracked {found}/{len(table)} frames ({found / max(len(table), 1):.0%})"
              + (f"; {conflicts} frames where forward and backward tracking disagreed."
                 if conflicts else "."))
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
