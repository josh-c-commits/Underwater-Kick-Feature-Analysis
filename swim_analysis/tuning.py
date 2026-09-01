"""Bayesian optimization over MediaPipe Pose Landmarker hyperparameters.

Goal: find the (min_pose_detection_confidence, min_pose_presence_confidence,
min_tracking_confidence) combination that gets landmarks to both (a) show up
on screen and (b) land inside the SAM2 outline, by minimizing a point
penalty summed over a video:

    for each judged landmark, for each frame in range:
        + missing_penalty                        if the landmark wasn't detected that frame
        + exp(distance_penalty_scale * dist) - 1  if detected outside the outline mask,
                                                   where dist = pixels from the mask boundary
        + 0                                       if detected inside the outline, or no mask
                                                   exists that frame (can't judge containment)

Two things are deliberately fixed *before* optimization starts, not
re-derived per trial:

  * The outline masks (outline_capture.generate_outline_masks). Mask
    quality has nothing to do with pose-landmarker settings, so there's
    no reason to regenerate SAM2 masks on every Bayesian-optimization
    trial -- generate them once and reuse the same masks_dir throughout.

  * The set of judged landmarks (select_judged_landmarks). Which
    landmarks are worth scoring is a property of the swimmer/camera
    setup (e.g. an ankle that's never visible from this camera angle
    shouldn't tank every trial's score), not of the hyperparameters
    being searched. Pick it from a rank-visibility table computed
    ahead of time (see ranking.rank_by_visibility / the CLI's
    rank-visibility command) and hold it fixed across all trials.

Each trial *does* have to re-run MediaPipe extraction, since the
three confidence thresholds are baked into PoseLandmarkerOptions at
landmarker-creation time -- there's no way to sweep them without
re-running detection.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import pandas as pd

from .duration import resolve_frame_bounds
from .landmarks import (
    DEFAULT_VISIBILITY_THRESHOLD,
    LANDMARK_NAMES,
    detect_column_style,
    landmark_columns,
)
from .outline import load_mask, outside_distance_transform
from .pose_extraction import extract

HYPERPARAM_NAMES = (
    "min_pose_detection_confidence",
    "min_pose_presence_confidence",
    "min_tracking_confidence",
)


def select_judged_landmarks(
    ranking_table: pd.DataFrame,
    threshold: Optional[float] = None,
    top_n: Optional[int] = None,
) -> List[int]:
    """
    Pick which landmark indices should count toward the tuning score,
    from a rank_by_visibility()-shaped table (columns 'landmark',
    'avg_visibility'). Exactly one of threshold/top_n must be given:
    threshold keeps every landmark with avg_visibility >= threshold,
    top_n keeps the N most-visible landmarks regardless of their
    absolute visibility.
    """
    if (threshold is None) == (top_n is None):
        raise ValueError("Provide exactly one of threshold= or top_n=, not both/neither.")

    table = ranking_table.sort_values("avg_visibility", ascending=False)
    if top_n is not None:
        chosen_names = table["landmark"].head(top_n).tolist()
    else:
        chosen_names = table.loc[table["avg_visibility"] >= threshold, "landmark"].tolist()

    name_to_index = {name: i for i, name in enumerate(LANDMARK_NAMES)}
    indices, unmatched = [], []
    for name in chosen_names:
        if name in name_to_index:
            indices.append(name_to_index[name])
        else:
            unmatched.append(name)
    if unmatched:
        print(
            f"Warning: {len(unmatched)} ranked landmark name(s) don't match the "
            f"built-in landmark names (likely from a custom --mapping file used "
            f"when ranking) and will be skipped: {unmatched}"
        )
    if not indices:
        raise ValueError(
            "No judged landmarks selected -- check threshold/top_n against the ranking table."
        )
    return sorted(indices)


def score_extraction(
    csv_path: str,
    masks_dir: str,
    judged_landmarks: Sequence[int],
    frame_width: int,
    frame_height: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    missing_penalty: float = 1.0,
    distance_penalty_scale: float = 0.05,
) -> float:
    """
    Total point penalty for one extraction CSV against a fixed outline
    mask directory and landmark subset. Lower is better; 0 means every
    judged landmark was visible and inside the outline on every frame
    in range.
    """
    df = pd.read_csv(csv_path)
    style = detect_column_style(list(df.columns))
    df = df.set_index("frame")

    lo = start_frame if start_frame is not None else int(df.index.min())
    hi = end_frame if end_frame is not None else int(df.index.max())

    total = 0.0
    for frame in range(lo, hi + 1):
        mask = load_mask(masks_dir, frame)
        dist_field = outside_distance_transform(mask) if mask is not None else None

        if frame not in df.index:
            total += missing_penalty * len(judged_landmarks)
            continue

        row = df.loc[frame]
        for idx in judged_landmarks:
            x_col, y_col, _, vis_col = landmark_columns(idx, style)
            x, y, vis = row.get(x_col), row.get(y_col), row.get(vis_col)
            if pd.isna(x) or pd.isna(y) or pd.isna(vis) or float(vis) < visibility_threshold:
                total += missing_penalty
                continue
            if dist_field is None:
                continue  # no mask this frame -- can't judge containment

            x_px = min(max(int(round(float(x) * frame_width)), 0), frame_width - 1)
            y_px = min(max(int(round(float(y) * frame_height)), 0), frame_height - 1)
            dist = float(dist_field[y_px, x_px])
            if dist > 0:
                total += math.exp(distance_penalty_scale * dist) - 1.0

    return total


def _video_dimensions(video_path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path} with OpenCV.")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return width, height


def _extract_for_trial(
    video_path: str,
    model_path: str,
    params: Sequence[float],
    num_poses: int,
    use_cpu: bool,
) -> Tuple[str, str]:
    """
    Run one MediaPipe extraction with a given hyperparameter triple.
    video_path is assumed already normalized (normalize=False here) --
    re-running ffmpeg on every trial would be pure waste since the
    video itself never changes between trials, only the confidences.
    Returns (csv_path, tmp_dir); caller must rmtree(tmp_dir) when done.
    """
    tmp_dir = tempfile.mkdtemp(prefix="swim_tuning_")
    csv_path = os.path.join(tmp_dir, "trial.csv")
    extract(
        video_path,
        csv_path,
        model_path,
        num_poses=num_poses,
        min_pose_detection_confidence=params[0],
        min_pose_presence_confidence=params[1],
        min_tracking_confidence=params[2],
        normalize=False,
        use_cpu=use_cpu,
    )
    return csv_path, tmp_dir


@dataclass
class TuningResult:
    best_params: Dict[str, float]
    best_score: float
    all_scores: List[float] = field(default_factory=list)
    all_params: List[Tuple[float, float, float]] = field(default_factory=list)


def optimize_pose_hyperparameters(
    video_path: str,
    masks_dir: str,
    model_path: str,
    judged_landmarks: Sequence[int],
    n_calls: int = 20,
    n_initial_points: int = 5,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    missing_penalty: float = 1.0,
    distance_penalty_scale: float = 0.05,
    num_poses: int = 1,
    use_cpu: bool = True,
    random_state: int = 0,
    progress: bool = True,
) -> TuningResult:
    """
    Bayesian-optimize the three pose-landmarker confidence thresholds
    (each searched over [0.05, 0.95]) against the penalty score in
    score_extraction(), using scikit-optimize's Gaussian-process
    minimizer. video_path should be the same normalized video the
    masks in masks_dir were generated from (see generate_outline_masks
    / the generate-outline CLI command), so frame numbering lines up.
    """
    from skopt import gp_minimize
    from skopt.space import Real

    start_frame, end_frame = resolve_frame_bounds(
        start_frame, end_frame, start_time, end_time, fps
    )
    width, height = _video_dimensions(video_path)

    space = [Real(0.05, 0.95, name=name) for name in HYPERPARAM_NAMES]

    trial_num = [0]

    def objective(params: Sequence[float]) -> float:
        trial_num[0] += 1
        csv_path, tmp_dir = _extract_for_trial(video_path, model_path, params, num_poses, use_cpu)
        try:
            score = score_extraction(
                csv_path,
                masks_dir,
                judged_landmarks,
                width,
                height,
                start_frame=start_frame,
                end_frame=end_frame,
                visibility_threshold=visibility_threshold,
                missing_penalty=missing_penalty,
                distance_penalty_scale=distance_penalty_scale,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if progress:
            param_str = " ".join(f"{n}={v:.3f}" for n, v in zip(HYPERPARAM_NAMES, params))
            print(f"[trial {trial_num[0]}/{n_calls}] {param_str} -> score={score:.2f}")
        return score

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=min(n_initial_points, n_calls),
        random_state=random_state,
    )

    best_params = dict(zip(HYPERPARAM_NAMES, result.x))
    return TuningResult(
        best_params=best_params,
        best_score=result.fun,
        all_scores=list(result.func_vals),
        all_params=[tuple(x) for x in result.x_iters],
    )
