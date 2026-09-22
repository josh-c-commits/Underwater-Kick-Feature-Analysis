"""video -> csv: run MediaPipe Pose Landmarker over a video and write
one row per frame to a CSV, using named columns (e.g. left_shoulder_x).

Every video frame gets a row -- even if no pose was detected in it,
in which case the x/y/z/visibility fields are left blank. This keeps
CSV row N aligned with video frame N, which annotate.py depends on.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .landmarks import named_header, NUM_LANDMARKS
from .tracking import crop_to_frame_norm, fixed_box


def normalize_video(input_path: str, output_path: str) -> None:
    """
    Re-encode with ffmpeg so rotation metadata is baked into the pixel
    data and the codec is one OpenCV decodes reliably. This is what
    fixes NORM_RECT/IMAGE_DIMENSIONS errors and the intermittent frame
    corruption that comes from reading .mov/HEVC directly in cv2.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. `winget install ffmpeg` "
            "or download from ffmpeg.org) and make sure it's on PATH."
        )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed:\n{result.stderr}")


def build_row(
    frame_number: int,
    pose_landmarks,
    world_landmarks,
    box: Optional[Tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
    include_world: bool,
) -> list:
    """
    Assemble one CSV row, converting crop-relative landmark coordinates
    back to full-frame normalized coordinates.

    Split out from the extraction loop purely so the coordinate handling is
    testable without running MediaPipe -- getting the resituation wrong would
    silently place every landmark in the wrong part of the frame, which is
    exactly the kind of bug that doesn't announce itself.
    """
    row: list = [frame_number]
    if box is not None:
        row += [box[0], box[1], box[2], box[3]]

    if not pose_landmarks:
        row += [""] * (NUM_LANDMARKS * 4)
        if include_world:
            row += [""] * (NUM_LANDMARKS * 3)
        return row

    for landmark in pose_landmarks:
        if box is None:
            x, y = landmark.x, landmark.y
        else:
            x, y = crop_to_frame_norm(landmark.x, landmark.y, box, frame_width, frame_height)
        row += [
            f"{x:.8f}",
            f"{y:.8f}",
            f"{landmark.z:.8f}",
            f"{landmark.visibility:.8f}",
        ]

    if include_world:
        if world_landmarks:
            for landmark in world_landmarks:
                row += [f"{landmark.x:.8f}", f"{landmark.y:.8f}", f"{landmark.z:.8f}"]
        else:
            row += [""] * (NUM_LANDMARKS * 3)
    return row


def extract(
    input_video: str,
    csv_path: str,
    model_path: str,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.4,
    min_pose_presence_confidence: float = 0.6,
    min_tracking_confidence: float = 0.6,
    normalize: bool = True,
    use_cpu: bool = True,
    boxes_csv: Optional[str] = None,
    crop_size: Optional[Tuple[int, int]] = None,
    include_world: bool = True,
) -> None:
    """
    Run pose detection on input_video and write results to csv_path.

    boxes_csv: output of `track`. When given, each frame is cropped to a
    fixed-size window around the tracked centroid before detection, and the
    resulting landmarks are converted back to full-frame coordinates. This is
    what makes MediaPipe viable here at all: the swimmer is ~0.3% of the frame,
    and MediaPipe's person detector sees the whole frame downscaled to a few
    hundred pixels, so uncropped the subject is only a handful of pixels wide
    and detection never fires.

    crop_size: (width, height) of that window, fixed for the whole clip so
    landmark coordinates stay comparable frame to frame. Defaults to a
    generous multiple of the median tracked box.
    """
    tmp_dir = None
    try:
        if normalize:
            tmp_dir = tempfile.mkdtemp(prefix="analysis_")
            video_path = os.path.join(tmp_dir, "normalized_input.mp4")
            normalize_video(input_video, video_path)
        else:
            video_path = input_video

        vid_capture = cv2.VideoCapture(video_path)
        if not vid_capture.isOpened():
            raise RuntimeError(f"Could not open {video_path} with OpenCV.")

        fps = vid_capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:  # covers 0 and NaN
            print("Warning: FPS read as invalid, defaulting to 30.")
            fps = 30.0

        # On macOS, the Tasks API's GPU/Metal delegate path
        # (DrishtiMetalHelper) is known to hard-crash with
        # "Check failed: service_ Service is unavailable." on some
        # setups (see google-ai-edge/mediapipe#5568), so use_cpu
        # defaults to True. Only set use_cpu=False if you've confirmed
        # GPU actually works on your machine.
        delegate = (
            python.BaseOptions.Delegate.CPU if use_cpu
            else python.BaseOptions.Delegate.GPU
        )
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )

        frame_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        boxes = None
        if boxes_csv is not None:
            boxes = pd.read_csv(boxes_csv).set_index("frame")
            if crop_size is None:
                found = boxes[boxes["found"]]
                if found.empty:
                    raise RuntimeError(f"{boxes_csv} has no tracked frames to crop around.")
                # Pad generously around the median tracked box: limbs leave the
                # tight detection box during a kick, and a crop that clips them
                # costs exactly the landmarks the analysis is about.
                crop_size = (
                    int(min(frame_width, found["box_w"].median() * 3)),
                    int(min(frame_height, found["box_h"].median() * 3)),
                )
            print(f"Cropping to {crop_size[0]}x{crop_size[1]} around the tracked centroid.")

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(named_header(world=include_world, box=boxes is not None))

            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                frame_count = 0
                while True:
                    success, frame = vid_capture.read()
                    if not success:
                        break
                    frame_count += 1

                    box = None
                    if boxes is not None:
                        if frame_count not in boxes.index or not bool(
                            boxes.loc[frame_count, "found"]
                        ):
                            writer.writerow(
                                build_row(frame_count, None, None, (0, 0, 0, 0),
                                          frame_width, frame_height, include_world)
                            )
                            continue
                        row_data = boxes.loc[frame_count]
                        box = fixed_box(
                            float(row_data["centroid_x"]), float(row_data["centroid_y"]),
                            crop_size[0], crop_size[1], frame_width, frame_height,
                        )
                        region = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                    else:
                        region = frame

                    rgb_frame = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int((frame_count / fps) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                    world = (
                        result.pose_world_landmarks[0]
                        if include_world and result.pose_world_landmarks
                        else None
                    )
                    writer.writerow(
                        build_row(frame_count, landmarks, world, box,
                                  frame_width, frame_height, include_world)
                    )

        vid_capture.release()
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"CSV saved to: {os.path.abspath(csv_path)}")
