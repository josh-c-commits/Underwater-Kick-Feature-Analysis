"""Re-encode raw footage once, into the form every later stage reads.

Everything here changes pixel values, so it happens in one place, once:

  * HDR to SDR. iPhones record HLG HDR: 10-bit, BT.2020 colour primaries. Read
    as ordinary video, those values come out hazy and desaturated; floor-line
    contrast was 15% lower on our footage. The standard ITU-R BT.2408
    conversion is baked into a 3D lookup table that ffmpeg applies natively.
  * Levelling. A camera that isn't level tilts the swimmer's path and every
    line in the pool. The frame is rotated about its centre, so the lens centre
    stays where it was, and nothing is cropped, so the geometry stays known.
  * A sidecar file (`<video>.json`) records what was done, plus any region to
    ignore, so later stages can read it instead of being told again.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import List, Optional

import cv2
import numpy as np

from .lens import camera_lens

# HLG (ARIB STD-B67) constants
_A, _B, _C = 0.17883277, 0.28466892, 0.55991073
_BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593])
_BT709_LUMA = np.array([0.2126, 0.7152, 0.0722])
# linear BT.2020 RGB -> linear BT.709 RGB
_BT2020_TO_BT709 = np.array([
    [1.6605, -0.5876, -0.0728],
    [-0.1246, 1.1329, -0.0083],
    [-0.0182, -0.1006, 1.1187],
])
_PEAK_NITS = 1000.0       # nominal HLG display peak; system gamma 1.2 belongs to it
_REFERENCE_WHITE = 203.0  # nits; HLG reference white per ITU-R BT.2408
# Where reference white lands in SDR (1.0 = full white). Sunlit white floor tiles
# sit above reference white: placing it at full white clipped 16% of a test frame,
# losing their texture; at 0.70, 0.5% clipped, and mid-water contrast was still
# 1.7x the naive conversion's (floor-line contrast 2x).
_WHITE_LEVEL = 0.70
_KNEE = 0.8               # SDR luminance above which highlights are compressed


@dataclass
class ClipInfo:
    """What normalization did to a clip, stored next to it as `<video>.json`."""

    source: str
    width: int
    height: int
    rotation_deg: float = 0.0          # counter-clockwise rotation applied to level it
    # Two points (source pixels) clicked on a line that's horizontal in the pool,
    # when the rotation came from them.
    level_line: Optional[List[List[float]]] = None
    tone_mapped: Optional[str] = None  # "hlg" when HDR was converted to SDR
    # Rows above this are left out of background plates, camera-motion
    # measurement and the search for the swimmer: e.g. the water surface, whose
    # constant motion is noise with nothing stable to latch onto.
    ignore_above: Optional[int] = None
    # The camera's own name for the lens (e.g. "... back camera 2.22mm f/2.2" for
    # the 0.5x) and the source's width: together they pick the lens model that
    # undoes the underwater window's distortion (see lens.py).
    camera_lens: Optional[str] = None
    source_width: Optional[int] = None

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "ClipInfo":
        with open(path) as handle:
            return cls(**json.load(handle))


def info_path(video_path: str) -> str:
    return os.path.splitext(video_path)[0] + ".json"


def read_clip_info(video_path: str) -> Optional[ClipInfo]:
    """The clip's sidecar, or None for clips normalized before sidecars existed."""
    path = info_path(video_path)
    return ClipInfo.load(path) if os.path.exists(path) else None


def ignore_above(video_path: str) -> Optional[int]:
    info = read_clip_info(video_path)
    return info.ignore_above if info else None


# ---------- HDR to SDR ----------

def hlg_to_sdr(rgb: np.ndarray) -> np.ndarray:
    """
    HLG-encoded BT.2020 RGB in [0, 1] -> gamma-encoded BT.709 RGB in [0, 1].

    The BT.2408 display-light route: undo the HLG curve to scene light, apply
    the HLG system gamma to get display light in nits, place HLG reference white
    (203 nits) at 70% of SDR white to leave headroom for the sunlit floor, then
    fit the result into BT.709.

    Two choices are about measurement rather than looks. Underwater blue-cyan
    lies outside BT.709, and clipping it turns the whole pool vivid cyan; so
    out-of-gamut colours are pulled toward grey *at their own luminance*, which
    keeps every brightness difference the tracker relies on. Highlights above
    the knee are compressed on luminance alone, for the same reason.
    """
    e = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    scene = np.where(e <= 0.5, e * e / 3.0, (np.exp((e - _C) / _A) + _B) / 12.0)
    luma = np.maximum(np.einsum("...c,c->...", scene, _BT2020_LUMA), 1e-12)[..., None]
    display = _PEAK_NITS * luma ** 0.2 * scene / _REFERENCE_WHITE * _WHITE_LEVEL
    linear = np.einsum("...c,rc->...r", display, _BT2020_TO_BT709)

    y = np.maximum(np.einsum("...c,c->...", linear, _BT709_LUMA), 0.0)[..., None]
    low = linear.min(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = np.where(low < 0, y / np.maximum(y - low, 1e-12), 1.0)
    linear = y + np.clip(pull, 0.0, 1.0) * (linear - y)

    knee_y = np.where(y <= _KNEE, y, _KNEE + (1 - _KNEE) * np.tanh((y - _KNEE) / (1 - _KNEE)))
    with np.errstate(divide="ignore", invalid="ignore"):
        linear = np.where(y > 0, linear * (knee_y / np.maximum(y, 1e-12)), 0.0)
    high = linear.max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        squeeze = np.where(high > 1.0, (1.0 - knee_y) / np.maximum(high - knee_y, 1e-12), 1.0)
    linear = knee_y + np.clip(squeeze, 0.0, 1.0) * (linear - knee_y)
    return np.clip(linear, 0.0, 1.0) ** (1 / 2.4)


def write_cube(path: str, size: int = 65) -> None:
    """hlg_to_sdr sampled on a size^3 grid, as a .cube file for ffmpeg's lut3d
    (red varies fastest)."""
    axis = np.linspace(0.0, 1.0, size)
    blue, green, red = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([red, green, blue], axis=-1).reshape(-1, 3)
    out = hlg_to_sdr(grid)
    with open(path, "w") as handle:
        handle.write(f"LUT_3D_SIZE {size}\n")
        np.savetxt(handle, out, fmt="%.6f")


# ---------- levelling ----------

def roll_from_points(first, second) -> float:
    """
    Clockwise lean, in degrees, of the line through two clicked points that lie
    on something horizontal in the pool, such as the lane rope.

    Clicked rather than detected: automatic measurement locked onto floor
    cross-lines in three of four clips, and those tilt with the camera's
    sideways aim even when it is level. Two clicks ~3000 px apart fix the angle
    to about 0.05 degrees.
    """
    (x0, y0), (x1, y1) = first, second
    if x1 < x0:
        (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
    if x1 == x0:
        raise ValueError("The two points are directly above one another; click along the line.")
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def median_frame(video_path: str, samples: int = 9) -> np.ndarray:
    """A median of frames spread over the clip, at full resolution: the swimmer
    and surface ripple fade out, leaving the pool's lines to click on."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path} with OpenCV.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for number in np.linspace(0, max(total - 1, 0), samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}.")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


# ---------- the re-encode ----------

def probe(video_path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,pix_fmt,color_transfer,color_primaries", "-of", "json",
         video_path],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)["streams"][0]


def filter_chain(hlg: bool, rotate: Optional[float], width: Optional[int], cube: str) -> str:
    """The ffmpeg -vf chain: HDR conversion, then levelling, then scaling."""
    filters = []
    if hlg:
        filters += ["scale=in_color_matrix=bt2020:in_range=limited:out_range=full",
                    "format=rgb48le", f"lut3d=file={cube}:interp=tetrahedral"]
    if rotate:
        # ffmpeg's rotate is clockwise for positive angles; ours is counter-clockwise
        filters.append(f"rotate=-({rotate})*PI/180:ow=iw:oh=ih:c=black")
    if width:
        filters.append(f"scale={int(width)}:-2:flags=area")
    filters += ["scale=out_color_matrix=bt709:out_range=limited", "format=yuv420p"]
    return ",".join(filters)


def normalize_video(
    input_path: str,
    output_path: str,
    rotate: Optional[float] = None,
    level_line: Optional[List[List[float]]] = None,
    tone_map: bool = True,
    ignore_top: Optional[float] = None,
    width: Optional[int] = None,
    crf: int = 18,
    preset: str = "fast",
) -> ClipInfo:
    """
    Re-encode so every later stage decodes identical, SDR, upright frames, and
    write the sidecar describing it.

    rotate: degrees to rotate counter-clockwise.
    level_line: two points on a line that's horizontal in the pool; the clip is
    rotated to make it level (instead of giving `rotate`).
    ignore_top: fraction of the frame height, from the top, for later stages to
    ignore (stored in the sidecar; the pixels are kept).
    width: downscale to this width; full resolution by default.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`, "
            "`winget install ffmpeg`, or from ffmpeg.org) and make sure it's on PATH."
        )
    if level_line is not None and rotate is not None:
        raise ValueError("Give either a rotation or a level line, not both.")
    stream = probe(input_path)
    hlg = tone_map and stream.get("color_transfer") == "arib-std-b67"
    if level_line is not None:
        roll = roll_from_points(*level_line)
        # Below a twentieth of a degree, rotating would only soften the image.
        rotate = roll if abs(roll) >= 0.05 else 0.0

    with tempfile.TemporaryDirectory() as tmp:
        cube = os.path.join(tmp, "hlg_to_sdr.cube")
        if hlg:
            write_cube(cube)
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", filter_chain(hlg, rotate, width, cube),
               "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-color_range", "tv", "-an", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed:\n{result.stderr}")

    cap = cv2.VideoCapture(output_path)
    out_w, out_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    info = ClipInfo(
        source=input_path, width=out_w, height=out_h, rotation_deg=float(rotate or 0.0),
        level_line=[[float(v) for v in point] for point in level_line] if level_line else None,
        tone_mapped="hlg" if hlg else None,
        ignore_above=int(round(ignore_top * out_h)) if ignore_top else None,
        camera_lens=camera_lens(input_path), source_width=int(stream["width"]),
    )
    info.save(info_path(output_path))
    return info
