"""Generate per-frame SAM2 outline masks for a video clip.

Uses SAM2 (via the `ultralytics` package) to segment and track the
swimmer across a video clip, writing one binary mask PNG per frame in
the naming convention outline.py expects: frame_000001.png,
frame_000002.png, ...

Why SAM2 here (and not SAM3): SAM3's weights are gated behind a
Hugging Face access request and manual download, while SAM2's weights
auto-download from a public GitHub release with no account needed --
confirmed by actually running it, not just reading the docs. SAM3's
text-prompt convenience ("segment the swimmer" with no click) is
real, but the access friction wasn't worth it given the setup pain
already spent on this project. Revisit if that gating situation
changes.

Model variants (smallest/fastest to largest/most accurate):
sam2_t.pt (tiny, default here), sam2_s.pt, sam2_b.pt, sam2_l.pt.
Larger variants are more accurate but slower -- especially relevant
on CPU, where video tracking (the propagation step after the first
prompted frame) is the expensive part, confirmed to be CPU-bound and
correct but slow in constrained environments.

IMPORTANT -- why this drives ultralytics.models.sam.predict.SAM2VideoPredictor
directly instead of just calling SAM(...)(video, points=..., stream=True):
that convenience call silently resolves (via SAM.task_map) to the
image-only SAM2Predictor, never the memory-based SAM2VideoPredictor,
even for a video source. Confirmed by reading ultralytics' own
task_map/predict dispatch (there is no source-type check that
switches to the video predictor). The image predictor consumes the
prompt dict via `self.prompts.pop(...)` on the very first frame; every
later frame calls inference() with no points/bboxes/masks at all,
which trips SAM's `generate()` fallback -- full unprompted
"segment everything" via a dense grid of point prompts -- on *every
single subsequent frame*. That's why the old point-once approach was
both slow (grid-search segmentation every frame instead of one cheap
memory-augmented inference) and inaccurate (picking masks.data[0] out
of that grid has no continuity with the previous frame's swimmer --
it's effectively a random object each frame).

SAM2VideoPredictor.inference() has no such fallback; it always uses
its memory-attention tracking state. Driving it directly also exposes
add_new_prompts(obj_id, points/labels, frame_idx), which lets us plant
corrective click prompts at additional frames chosen by the caller
(not just frame 1) -- each one is treated as a fresh conditioning
frame the tracker re-anchors to, which is exactly what's needed when
memory-based tracking drifts onto the wrong object partway through a
clip. This is the same "multiple clicks at different points in the
video" workflow SAM2's own interactive demo uses; here it's just
supplied up front instead of interactively.

This does mean this module leans on ultralytics internals
(SAM2VideoPredictor, add_new_prompts, _prepare_prompts) that aren't
part of its public docs -- confirmed to exist and behave as described
by reading the installed ultralytics==8.4.123 source directly. Pin
the ultralytics version and re-verify this module against
predict.py's SAM2VideoPredictor if upgrading.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .outline import MASK_FILENAME_PATTERN

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def _first_frame(video_path: str) -> np.ndarray:
    return _frame_at(video_path, 1)


def _frame_at(video_path: str, frame_number: int) -> np.ndarray:
    """1-indexed frame read, matching this project's frame numbering everywhere else."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path} with OpenCV.")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, frame = cap.read()
        if not success:
            raise RuntimeError(f"Could not read frame {frame_number} of {video_path}.")
        return frame
    finally:
        cap.release()


def _frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path} with OpenCV.")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def pick_point_interactively(video_path: str, frame_number: int = 1) -> Tuple[int, int]:
    """
    Show a frame (frame 1 by default) in a matplotlib window and let
    the user click once on the swimmer. Returns (x, y) in pixel
    coordinates. Requires a display -- use --point/--reprompt over
    SSH/headless.
    """
    import matplotlib.pyplot as plt

    frame_bgr = _frame_at(video_path, frame_number)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(frame_rgb)
    ax.set_title(f"Click once on the swimmer (frame {frame_number}), then close this window")
    pts = fig.ginput(1, timeout=0)  # blocks until one click
    plt.close(fig)

    if not pts:
        raise RuntimeError("No point was clicked.")
    x, y = pts[0]
    return int(round(x)), int(round(y))


def save_frame_preview(video_path: str, frame_number: int, out_path: str) -> None:
    """Write a single frame to out_path, for headless workflows where
    you'd rather eyeball pixel coordinates in an image viewer than
    click interactively. Used both for the initial prompt (frame 1)
    and for picking coordinates for a --reprompt at a later frame."""
    frame = _frame_at(video_path, frame_number)
    cv2.imwrite(out_path, frame)


def _build_video_predictor(model_variant: str, device: Optional[str], imgsz: Optional[int]):
    """Instantiate ultralytics' memory-tracking SAM2 video predictor directly.

    See the module docstring for why this bypasses the SAM(...) convenience
    wrapper: that wrapper never selects this class for a video source.
    """
    from ultralytics import SAM
    from ultralytics.models.sam.predict import SAM2VideoPredictor

    sam = SAM(model_variant)
    if not getattr(sam, "is_sam2", False):
        raise ValueError(
            f"{model_variant!r} is not a SAM2 checkpoint (expected e.g. sam2_t.pt) -- "
            "the video predictor this module relies on is SAM2-specific."
        )

    overrides = {"conf": 0.25, "imgsz": imgsz if imgsz is not None else 1024}
    if device is not None:
        overrides["device"] = device

    predictor = SAM2VideoPredictor(overrides=overrides)
    predictor.setup_model(model=sam.model, verbose=False)
    return predictor


def _prepare_prompt_tensors(predictor, frame_bgr: np.ndarray, point: Optional[Point], bbox: Optional[BBox]):
    """Preprocess frame_bgr the same way the streaming pipeline would, and
    scale the given pixel-space point/bbox into the model's internal
    coordinate space -- returns (points, labels) tensors ready for
    add_new_prompts(), and the preprocessed image tensor to stash into
    inference_state["im"] beforehand."""
    im_tensor = predictor.preprocess([frame_bgr])
    dst_shape = im_tensor.shape[2:]
    src_shape = frame_bgr.shape[:2]

    points_in = [list(point)] if point is not None else None
    labels_in = [1] if point is not None else None
    bboxes_in = [list(bbox)] if bbox is not None else None

    points, labels, _masks = predictor._prepare_prompts(
        dst_shape, src_shape, bboxes=bboxes_in, points=points_in, labels=labels_in, masks=None
    )
    return im_tensor, points, labels


def _add_prompt_at_frame(
    predictor,
    video_path: str,
    frame_number: int,
    point: Optional[Point],
    bbox: Optional[BBox],
    progress: bool,
) -> None:
    frame_bgr = _frame_at(video_path, frame_number)
    im_tensor, points, labels = _prepare_prompt_tensors(predictor, frame_bgr, point, bbox)
    predictor.inference_state["im"] = im_tensor
    pred_masks, _scores = predictor.add_new_prompts(
        obj_id=0, points=points, labels=labels, frame_idx=frame_number
    )
    if progress:
        px = int((pred_masks > predictor.model.mask_threshold).sum())
        prompt_desc = f"point {point}" if point is not None else f"bbox {bbox}"
        print(f"Anchor added at frame {frame_number} ({prompt_desc}): ~{px} px in low-res mask")


def generate_outline_masks(
    video_path: str,
    masks_dir: str,
    point: Optional[Point] = None,
    bbox: Optional[BBox] = None,
    reprompts: Optional[Sequence[Tuple[int, Point]]] = None,
    reprompt_bboxes: Optional[Sequence[Tuple[int, BBox]]] = None,
    model_variant: str = "sam2_t.pt",
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    progress: bool = True,
) -> None:
    """
    Run SAM2 video segmentation over video_path, prompted by a point
    (x, y) or bounding box (x1, y1, x2, y2) on frame 1, and write one
    mask PNG per frame to masks_dir. Frames where SAM2 loses the
    subject get no file written (outline.py's load_mask already
    treats a missing file as "no mask that frame").

    Exactly one of point / bbox must be given for frame 1.

    reprompts / reprompt_bboxes: optional additional (frame_number,
    point-or-bbox) anchors, each 1-indexed to match this project's
    frame numbering. Each one plants a corrective click/box on the
    swimmer at that frame *before* propagation runs, so memory-based
    tracking re-anchors there instead of carrying forward whatever
    (possibly wrong) object it locked onto earlier. Use this when a
    single frame-1 prompt drifts onto the wrong swimmer/lane line/etc
    partway through the clip -- pick a few frames where that happens
    (e.g. via save_frame_preview) and add a corrective point there.

    device: 'cpu', 'mps' (Apple Silicon), 'cuda' (NVIDIA), or a
    specific CUDA index like '0'. Left as None, ultralytics picks
    automatically -- which is exactly the setting that can silently
    pick a partially-broken accelerator backend (e.g. some SAM2 ops
    on MPS aren't implemented and PyTorch falls back to CPU per-op,
    which can end up *slower* than plain CPU due to the constant
    CPU<->GPU data shuffling). Set this explicitly to compare.

    imgsz: the resolution SAM2 actually processes internally
    (defaults to 1024x1024 regardless of source video resolution --
    so shrinking the *source* video does NOT shrink this on its own).
    Passing a smaller value (e.g. 512) is what actually reduces
    compute and memory per frame.
    """
    if (point is None) == (bbox is None):
        raise ValueError("Provide exactly one of point= or bbox= for frame 1, not both/neither.")

    reprompts = list(reprompts or [])
    reprompt_bboxes = list(reprompt_bboxes or [])

    anchor_frames = [1] + [f for f, _ in reprompts] + [f for f, _ in reprompt_bboxes]
    if len(anchor_frames) != len(set(anchor_frames)):
        raise ValueError(f"Duplicate anchor frame numbers in prompts: {anchor_frames}")

    total_frames = _frame_count(video_path)
    for f in anchor_frames:
        if not (1 <= f <= total_frames):
            raise ValueError(f"Anchor frame {f} is out of range for a {total_frames}-frame video.")

    os.makedirs(masks_dir, exist_ok=True)

    if progress:
        import torch

        print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"torch.backends.mps.is_available() = {torch.backends.mps.is_available()}")
        print(f"Requested device = {device!r} (None = ultralytics auto-selects)")
        print(f"Requested imgsz = {imgsz if imgsz is not None else 1024}")
        print(f"Anchor frames: {sorted(anchor_frames)}")

    predictor = _build_video_predictor(model_variant, device, imgsz)
    predictor.setup_source(video_path)  # builds predictor.dataset + inference_state

    _add_prompt_at_frame(predictor, video_path, 1, point, bbox, progress)
    for frame_number, reprompt_point in reprompts:
        _add_prompt_at_frame(predictor, video_path, frame_number, reprompt_point, None, progress)
    for frame_number, reprompt_bbox in reprompt_bboxes:
        _add_prompt_at_frame(predictor, video_path, frame_number, None, reprompt_bbox, progress)

    results = predictor(video_path, stream=True)

    written, missing = 0, 0
    for frame_num, r in enumerate(results, start=1):
        if r.masks is None or len(r.masks) == 0:
            missing += 1
            if progress:
                print(f"frame {frame_num}: no mask (subject not found)")
            continue

        mask = r.masks.data[0].cpu().numpy() > 0.5
        mask_img = (mask.astype(np.uint8)) * 255
        out_path = os.path.join(masks_dir, MASK_FILENAME_PATTERN.format(frame=frame_num))
        cv2.imwrite(out_path, mask_img)
        written += 1
        if progress:
            print(f"frame {frame_num}: mask written ({mask.sum()} px)")

    print(f"Done. {written} masks written, {missing} frames had no detected mask.")
    print(f"Masks saved to: {os.path.abspath(masks_dir)}")
