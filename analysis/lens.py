"""
Undo the refraction at a camera's flat window underwater.

A phone lens is designed for air. Underwater, light reaches it through flat
glass, bending at the water/air boundary by Snell's law -- not at all head-on,
more and more toward the edges of the view -- so the picture's edges are
stretched outward ("pincushion"). Measured on the iPhone 17 Pro's 0.5x lens, a
metre of pool near the left and right edges covers ~55% more pixels than at the
centre, and every straight floor line bows.

The physics gives the correction exactly, with one number per lens and video
mode, the focal length in pixels. A point r pixels from the lens centre arrived
at angle atan(r / f) inside the phone; Snell's law gives the angle it had in
the water; an ideal camera would have put it at f * n * tan(that angle). After
undistorting, straight pool lines are straight and plain perspective holds: at
a given distance, a metre spans the same number of pixels horizontally and
vertically anywhere in the frame.

The focal length was fitted as the value that straightens the floor lines: 1674
px on one 4K clip, 1666 on another, so 1670 for the 0.5x lens in 4K video. The
lens centre the phone records with every frame sits within a few pixels of the
image centre on these clips, so the image centre is used.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

WATER = 1.333
# iPhone 17 Pro 0.5x ("back camera 2.22mm f/2.2"), 4K video, measured on two clips
ULTRAWIDE_FOCAL_4K = 1670.0
_ULTRAWIDE_NAMES = ("2.22mm",)


@dataclass
class FlatPort:
    """A camera behind flat glass underwater. `focal` is the lens's focal length
    in pixels at this frame width; (cx, cy) the lens centre."""

    focal: float
    cx: float
    cy: float
    n: float = WATER

    @property
    def ideal_focal(self) -> float:
        """Focal length of the ideal camera the undistorted image comes from."""
        return self.focal * self.n

    def undistort(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Filmed image points -> where an ideal camera in water puts them."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        dx, dy = x - self.cx, y - self.cy
        r = np.hypot(dx, dy)
        safe = np.maximum(r, 1e-9)
        ideal = self.ideal_focal * np.tan(np.arcsin(np.sin(np.arctan(safe / self.focal)) / self.n))
        scale = np.where(r > 1e-9, ideal / safe, 1.0)
        return self.cx + dx * scale, self.cy + dy * scale

    def distort(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """The inverse: ideal-camera points -> where they appear in the filmed
        image. NaN beyond the edge of the window's view."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        dx, dy = x - self.cx, y - self.cy
        r = np.hypot(dx, dy)
        safe = np.maximum(r, 1e-9)
        s = np.sin(np.arctan(safe / self.ideal_focal)) * self.n
        filmed = np.where(s < 1, self.focal * np.tan(np.arcsin(np.minimum(s, 1.0 - 1e-12))), np.nan)
        scale = np.where(r > 1e-9, filmed / safe, 1.0)
        return self.cx + dx * scale, self.cy + dy * scale

    def to_dict(self) -> Dict:
        return {"model": "flat_port", "focal": self.focal, "centre": [self.cx, self.cy], "n": self.n}

    @classmethod
    def from_dict(cls, data: Dict) -> "FlatPort":
        if data.get("model") != "flat_port":
            raise ValueError(f"Unknown lens model {data.get('model')!r}")
        cx, cy = data["centre"]
        return cls(float(data["focal"]), float(cx), float(cy), float(data.get("n", WATER)))


def for_frame(width: int, height: int, focal_4k: float = ULTRAWIDE_FOCAL_4K) -> FlatPort:
    """The 0.5x lens model for a frame of this size (the focal length scales
    with the frame's width, for clips normalized smaller than 4K)."""
    return FlatPort(focal_4k * width / 3840.0, width / 2.0, height / 2.0)


# ---------- which lens shot a clip ----------

def _atoms(data: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", data[i:i + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[i + 8:i + 16])[0]
            header = 16
        elif size == 0:
            size = end - i
        if size < header:
            return
        yield kind, i + header, i + size
        i += size


def _moov(path: str) -> Optional[bytes]:
    """The contents of the file's moov atom (its index, a few hundred KB), found
    by hopping between top-level atoms instead of reading the whole video."""
    with open(path, "rb") as handle:
        position = 0
        while True:
            handle.seek(position)
            header = handle.read(16)
            if len(header) < 8:
                return None
            size, kind = struct.unpack(">I4s", header[:8])
            length = 8
            if size == 1:
                size = struct.unpack(">Q", header[8:16])[0]
                length = 16
            elif size == 0:  # the last atom, running to the end of the file
                if kind != b"moov":
                    return None
                handle.seek(position + length)
                return handle.read()
            if size < length:
                return None
            if kind == b"moov":
                handle.seek(position + length)
                return handle.read(size - length)
            position += size


def quicktime_metadata(path: str) -> Dict[str, object]:
    """Apple's mdta key/value metadata from every level of a QuickTime file
    (movie and tracks), e.g. com.apple.quicktime.camera.lens_model."""
    try:
        moov = _moov(path)
    except OSError:
        return {}
    if not moov:
        return {}
    found: Dict[str, object] = {}

    def walk(start: int, end: int) -> None:
        for kind, s, e in _atoms(moov, start, end):
            if kind == b"meta":
                keys, values = [], {}
                for k2, s2, e2 in _atoms(moov, s, e):
                    if k2 == b"keys":
                        count = struct.unpack(">I", moov[s2 + 4:s2 + 8])[0]
                        i = s2 + 8
                        for _ in range(count):
                            size = struct.unpack(">I", moov[i:i + 4])[0]
                            keys.append(moov[i + 8:i + size].decode("utf8", "replace"))
                            i += size
                    elif k2 == b"ilst":
                        for k3, s3, e3 in _atoms(moov, s2, e2):
                            index = struct.unpack(">I", k3)[0]
                            for k4, s4, e4 in _atoms(moov, s3, e3):
                                if k4 == b"data":
                                    kind_code = struct.unpack(">I", moov[s4:s4 + 4])[0] & 0xFFFFFF
                                    payload = moov[s4 + 8:e4]
                                    values[index] = (payload.decode("utf8", "replace")
                                                     if kind_code == 1 else payload)
                for index, value in values.items():
                    if 1 <= index <= len(keys):
                        found[keys[index - 1]] = value
            elif kind in (b"trak", b"udta", b"mdia", b"minf"):
                walk(s, e)

    walk(0, len(moov))
    return found


def camera_lens(path: str) -> Optional[str]:
    """The lens a clip was shot with, as the phone names it, if recorded."""
    value = quicktime_metadata(path).get("com.apple.quicktime.camera.lens_model")
    return value if isinstance(value, str) else None


def is_ultrawide(lens_name: Optional[str]) -> bool:
    return bool(lens_name) and any(tag in lens_name for tag in _ULTRAWIDE_NAMES)


def lens_for_video(video: str) -> Optional[FlatPort]:
    """
    The lens model for a normalized clip: from its sidecar's record of the
    camera, or failing that the raw file it came from. None when the lens isn't
    the 0.5x this model was measured on.
    """
    import os

    from .normalize import read_clip_info

    info = read_clip_info(video)
    if info is None:
        return None
    name, source_width = info.camera_lens, info.source_width
    if info.source and os.path.exists(info.source):
        if name is None:
            name = camera_lens(info.source)
        if source_width is None:
            from .normalize import probe

            source_width = int(probe(info.source)["width"])
    # Measured on 4K recordings only: other video modes may crop the sensor
    # differently, which changes the focal length.
    if not is_ultrawide(name) or source_width != 3840:
        return None
    return for_frame(info.width, info.height)
