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

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .landmarks import named_header, NUM_LANDMARKS


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
) -> None:
    """Run pose detection on input_video and write results to csv_path."""
    tmp_dir = None
    try:
        if normalize:
            tmp_dir = tempfile.mkdtemp(prefix="swim_analysis_")
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

        empty_fields = [""] * (NUM_LANDMARKS * 4)

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(named_header())

            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                frame_count = 0
                while True:
                    success, frame = vid_capture.read()
                    if not success:
                        break
                    frame_count += 1

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int((frame_count / fps) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    row = [frame_count]
                    if result.pose_landmarks:
                        for lm in result.pose_landmarks[0]:
                            row.extend([
                                f"{lm.x:.8f}", f"{lm.y:.8f}",
                                f"{lm.z:.8f}", f"{lm.visibility:.8f}",
                            ])
                    else:
                        row.extend(empty_fields)
                    writer.writerow(row)

        vid_capture.release()
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"CSV saved to: {os.path.abspath(csv_path)}")
