"""
Videos of the tracked swimmer, for watching a swim rather than checking numbers.

Two views of the same tracking (the overlay command is the one for checking):

  * spotlight -- the whole frame, greyed out except for a soft-edged oval around
    the swimmer, so they stay in the context of the pool and the other
    swimmers;
  * follow -- a virtual camera that glides along with the swimmer at a fixed
    zoom, showing only them.

Positions come from analysis.kinematics, so the oval and the camera follow the
same smoothed, gap-filled path the speed is measured on. The oval's colour says
how each frame's position was obtained: detected, filled in across a gap, merged
with another swimmer, or a keyframe click; a frame with no position at all is
left grey and labelled lost.
"""

from __future__ import annotations

import os
import subprocess
import sys
from fractions import Fraction
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .analysis import camera_path, direction_of_travel, kinematics
from .frames import fps as video_fps
from .frames import frame_count, frame_size, iter_frames

# BGR
STATUS_COLOURS = {
    "detected": (90, 220, 90),
    "keyframe": (255, 190, 60),
    "filled": (0, 215, 255),
    "merged": (0, 140, 255),
    "lost": (60, 60, 230),
}


# ---------- where the swimmer is ----------

def swimmer_path(boxes: pd.DataFrame, fps: float, calibration=None) -> pd.DataFrame:
    """
    One row per frame: the swimmer's centre (x, y) in frame coordinates, a
    steadied box size (w, h), how the position was obtained (`status`), and the
    speed in the direction of travel. Position is NaN where there's none to give.

    The size is a rolling median over about a second, so the oval doesn't pulse
    as the legs fold and extend.
    """
    kin = kinematics(boxes, fps, calibration)
    cam_dx, cam_dy = camera_path(boxes)
    x = kin["x_smooth"].to_numpy(float) + cam_dx
    y = kin["y_smooth"].to_numpy(float) + cam_dy

    found = boxes["found"].fillna(False).astype(bool).to_numpy()
    merged = (boxes["merged"].fillna(False).astype(bool).to_numpy() if "merged" in boxes
              else np.zeros(len(boxes), bool))
    source = boxes["source"].to_numpy(object) if "source" in boxes else np.full(len(boxes), None)
    filled = kin["filled"].to_numpy(bool)

    # a merge too long to fill keeps its (biased) centroid rather than vanishing
    raw_x = boxes["centroid_x"].to_numpy(float)
    raw_y = boxes["centroid_y"].to_numpy(float)
    missing = ~np.isfinite(x) & found
    x = np.where(missing, raw_x, x)
    y = np.where(missing, raw_y, y)

    status = np.where(found, "detected", np.where(filled, "filled", "lost")).astype(object)
    status[found & merged] = "merged"
    status[found & (source == "keyframe")] = "keyframe"
    status[~np.isfinite(x)] = "lost"

    clean = found & ~merged & boxes["box_w"].notna().to_numpy() if "box_w" in boxes else found & False
    window = int(round(fps)) | 1
    w = pd.Series(np.where(clean, boxes["box_w"].to_numpy(float), np.nan)) if "box_w" in boxes else pd.Series(np.full(len(boxes), np.nan))
    h = pd.Series(np.where(clean, boxes["box_h"].to_numpy(float), np.nan)) if "box_h" in boxes else pd.Series(np.full(len(boxes), np.nan))
    w = w.rolling(window, center=True, min_periods=1).median().interpolate(limit_direction="both")
    h = h.rolling(window, center=True, min_periods=1).median().interpolate(limit_direction="both")

    speed_column = "speed_m_s" if "speed_m_s" in kin else "speed_px_s"
    direction = direction_of_travel(kin["world_x_m" if speed_column == "speed_m_s" else "x_smooth"]) or 1
    return pd.DataFrame({
        "frame": boxes["frame"].to_numpy(int), "x": x, "y": y,
        "w": w.to_numpy(float), "h": h.to_numpy(float), "status": status,
        "speed": kin[speed_column].to_numpy(float) * direction,
        "speed_units": "m/s" if speed_column == "speed_m_s" else "px/s",
        "depth": kin["depth_m"].to_numpy(float) if "depth_m" in kin else np.nan,
    })


def camera_track(path: pd.DataFrame, fps: float, smooth_seconds: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Where the follow camera points on each frame: the swimmer's path, carried
    straight across frames with no position and held at the ends, then smoothed
    so the camera glides instead of twitching with every kick. The smoothing is
    centred in time, so the camera doesn't lag behind.
    """
    from scipy.ndimage import gaussian_filter1d

    frames = path["frame"].to_numpy(float)
    known = np.isfinite(path["x"].to_numpy(float))
    if not known.any():
        raise ValueError("The tracking has no positions to follow.")
    cx = np.interp(frames, frames[known], path["x"].to_numpy(float)[known])
    cy = np.interp(frames, frames[known], path["y"].to_numpy(float)[known])
    sigma = max(smooth_seconds * fps, 1e-6)
    return gaussian_filter1d(cx, sigma, mode="nearest"), gaussian_filter1d(cy, sigma, mode="nearest")


# ---------- drawing ----------

def greyed(frame: np.ndarray, brightness: float = 0.55) -> np.ndarray:
    """The frame without colour and dimmed, as the backdrop for the spotlight."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.convertScaleAbs(grey, alpha=brightness)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def spotlight(frame: np.ndarray, centre, size, colour, brightness: float = 0.55,
              margin: float = 1.4, feather: float = 0.3) -> np.ndarray:
    """
    The frame greyed out except a soft-edged oval around `centre`, `margin` times
    the box `size` (w, h), outlined in `colour`. With no centre, all grey.
    """
    out = greyed(frame, brightness)
    if centre is None or not np.all(np.isfinite(centre)) or not np.all(np.isfinite(size)):
        return out
    height, width = frame.shape[:2]
    cx, cy = centre
    a = max(0.5 * size[0] * margin, 8.0)
    b = max(0.5 * size[1] * margin * 1.3, 0.35 * a)  # a long thin swimmer still gets room
    reach_x, reach_y = a * (1 + feather), b * (1 + feather)
    x0, x1 = int(max(0, cx - reach_x)), int(min(width, cx + reach_x + 1))
    y0, y1 = int(max(0, cy - reach_y)), int(min(height, cy + reach_y + 1))
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    distance = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
    weight = np.clip((1 + feather - distance) / feather, 0.0, 1.0)
    weight = (weight * weight * (3 - 2 * weight))[..., None]  # smoothstep: no hard rim
    region = frame[y0:y1, x0:x1].astype(np.float32)
    backdrop = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (weight * region + (1 - weight) * backdrop).astype(np.uint8)
    thickness = max(1, int(round(height / 540)))
    cv2.ellipse(out, (int(round(cx)), int(round(cy))), (int(round(a * (1 + feather / 2))),
                int(round(b * (1 + feather / 2)))), 0, 0, 360, colour, thickness, cv2.LINE_AA)
    return out


def follow_crop(frame: np.ndarray, centre, crop_size: Tuple[int, int]) -> np.ndarray:
    """The crop_size (w, h) window centred on `centre`, slid inward at the frame's edges."""
    height, width = frame.shape[:2]
    cw, ch = crop_size
    x0 = int(round(min(max(centre[0] - cw / 2, 0), width - cw)))
    y0 = int(round(min(max(centre[1] - ch / 2, 0), height - ch)))
    return frame[y0:y0 + ch, x0:x0 + cw]


def draw_hud(image: np.ndarray, text: str, colour) -> None:
    """A line of text on a dark band at the bottom left, with a status dot."""
    height = image.shape[0]
    scale = height / 1080 * 0.9
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = int(round(th * 0.6))
    dot = int(round(th * 0.45))
    x, y = pad, height - pad
    band = image[y - th - 2 * pad:height, 0:tw + 4 * pad + 2 * dot]
    band[:] = (band * 0.35).astype(np.uint8)
    cv2.circle(image, (x + dot, y - th // 2), dot, colour, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x + 2 * dot + pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def hud_text(row, fps: float) -> str:
    seconds = (row["frame"] - 1) / fps
    parts = [f"{seconds:5.2f} s", f"frame {int(row['frame'])}", str(row["status"])]
    if row["status"] != "lost" and np.isfinite(row["speed"]):
        value = f"{row['speed']:.2f}" if row["speed_units"] == "m/s" else f"{row['speed']:.0f}"
        parts.append(f"{value} {row['speed_units']}")
    if row["status"] != "lost" and np.isfinite(row.get("depth", np.nan)):
        parts.append(f"{row['depth']:.2f} m deep")
    return "   ".join(parts)


# ---------- writing ----------

class VideoOut:
    """Frames piped to ffmpeg as H.264, which plays anywhere (QuickTime included)."""

    def __init__(self, path: str, width: int, height: int, fps: float, crf: int = 20):
        rate = Fraction(fps).limit_denominator(1001)
        self.size = (width, height)
        self.process = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
             "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-movflags", "+faststart", path],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the video.")


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def export_video(
    video_path: str,
    boxes: pd.DataFrame,
    out_video: str,
    mode: str = "spotlight",
    width: Optional[int] = None,
    zoom: float = 3.0,
    calibration=None,
    hud: bool = True,
    progress: bool = True,
) -> str:
    """
    Write the spotlight or follow video of a tracked clip.

    `width` is the output width (default 1920 for spotlight, 1280 for follow).
    `zoom` sets the follow camera's view: this many of the swimmer's lengths
    across, fixed for the whole clip so the zoom never pumps.
    """
    if mode not in ("spotlight", "follow"):
        raise ValueError("mode must be 'spotlight' or 'follow'")
    rate = video_fps(video_path)
    frame_w, frame_h = frame_size(video_path)
    total = frame_count(video_path)
    path = swimmer_path(boxes, rate, calibration).set_index("frame", drop=False)

    if mode == "spotlight":
        out_w = _even(min(width or 1920, frame_w))
        out_h = _even(frame_h * out_w / frame_w)
        scale = out_w / frame_w
    else:
        clean = path.loc[path["status"].isin(["detected", "keyframe"]), "w"]
        length = float(np.nanmedian(clean)) if len(clean) else 0.05 * frame_w
        out_w = _even(width or 1280)
        out_h = _even(out_w * 9 / 16)
        crop_w = int(min(max(zoom * length, 64), frame_w))
        crop_h = int(round(crop_w * out_h / out_w))
        if crop_h > frame_h:
            crop_h, crop_w = frame_h, int(round(frame_h * out_w / out_h))
        cam_x, cam_y = camera_track(path, rate)
        camera = dict(zip(path["frame"].to_numpy(int), zip(cam_x, cam_y)))

    writer = VideoOut(out_video, out_w, out_h, rate)
    try:
        for number, frame in iter_frames(video_path):
            row = path.loc[number] if number in path.index else None
            status = row["status"] if row is not None else "lost"
            colour = STATUS_COLOURS[status]
            if mode == "spotlight":
                small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                centre = size = None
                if row is not None and status != "lost":
                    centre = (row["x"] * scale, row["y"] * scale)
                    size = (row["w"] * scale, row["h"] * scale)
                image = spotlight(small, centre, size, colour)
            else:
                centre = camera.get(number)
                if centre is None:  # past the end of the tracking table
                    centre = (cam_x[-1], cam_y[-1])
                image = cv2.resize(follow_crop(frame, centre, (crop_w, crop_h)), (out_w, out_h),
                                   interpolation=cv2.INTER_AREA if crop_w >= out_w else cv2.INTER_CUBIC)
                if status == "lost":
                    image = greyed(image, 0.8)
            if hud:
                text = hud_text(row, rate) if row is not None else f"frame {number}   lost"
                draw_hud(image, text, colour)
            writer.write(image)
            if progress and (number % 30 == 0 or number == total):
                sys.stdout.write(f"\rWriting {mode} video: frame {number}/{total}")
                sys.stdout.flush()
    finally:
        writer.close()
    if progress:
        print()
    print(f"{mode.capitalize()} video saved to: {os.path.abspath(out_video)}")
    return out_video
