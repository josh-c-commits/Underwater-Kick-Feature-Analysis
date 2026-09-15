"""Image position -> real-world distance along the pool.

The camera is wide, so a single pixels-per-metre number is wrong: measuring
lane-rope colour blocks across one of these frames gives spacings that vary by
roughly 20% between the centre and the edges. Anything derived from raw pixel
displacement inherits that error.

Rather than model the camera (intrinsics, distortion, pose), this measures
where known world positions actually land in the image and interpolates
between them. That is deliberately empirical: an underwater housing with a
flat port refracts at the air/glass/water interfaces, which inflates effective
focal length by ~1.33 and adds radial distortion that an in-air calibration
would get wrong. Reference marks photographed through the same water and the
same port absorb all of it for free.

Model
-----
A calibration is a set of *reference lines* -- typically the two lane ropes
bounding the swimmer's lane -- each carrying knots of the form
(image_x, image_y, world_x). Along a line, world_x is interpolated between
knots; between lines, results are blended by image_y. So the same image column
maps to different world positions depending on how deep in the frame it sits,
which is the point: a mark 15m down the pool projects to a different column on
the near rope than on the far one.

Interpolation between knots is piecewise linear, which is only adequate
because knots are meant to be dense. Two knots cannot describe a curve *and*
leave any residual to check accuracy against; lane-rope colour-block
boundaries give 15-20 across a frame, and at that spacing linear segments are
fine. Outside a line's knot range the result is NaN unless extrapolation is
explicitly requested -- silently clamping would hand back confident numbers
for the part of the pool that was never calibrated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CALIBRATION_VERSION = 1


@dataclass
class ReferenceLine:
    """Knots along one physical line in the scene (e.g. a single lane rope)."""

    name: str
    # (image_x, image_y, world_x); world_x in metres from the start wall
    knots: List[Tuple[float, float, float]] = field(default_factory=list)

    def sorted_knots(self) -> List[Tuple[float, float, float]]:
        return sorted(self.knots, key=lambda k: k[0])

    def arrays(self):
        knots = self.sorted_knots()
        xs = np.array([k[0] for k in knots], dtype=float)
        ys = np.array([k[1] for k in knots], dtype=float)
        ws = np.array([k[2] for k in knots], dtype=float)
        return xs, ys, ws

    def world_x_at(self, image_x: float, extrapolate: bool = False) -> float:
        xs, _, ws = self.arrays()
        if len(xs) < 2:
            return float("nan")
        if not extrapolate and (image_x < xs[0] or image_x > xs[-1]):
            return float("nan")
        if extrapolate and image_x < xs[0]:
            slope = (ws[1] - ws[0]) / (xs[1] - xs[0])
            return float(ws[0] + (image_x - xs[0]) * slope)
        if extrapolate and image_x > xs[-1]:
            slope = (ws[-1] - ws[-2]) / (xs[-1] - xs[-2])
            return float(ws[-1] + (image_x - xs[-1]) * slope)
        return float(np.interp(image_x, xs, ws))

    def image_y_at(self, image_x: float) -> float:
        xs, ys, _ = self.arrays()
        if len(xs) == 0:
            return float("nan")
        if len(xs) == 1:
            return float(ys[0])
        return float(np.interp(image_x, xs, ys))


@dataclass
class Calibration:
    lines: List[ReferenceLine] = field(default_factory=list)
    frame_size: Optional[Tuple[int, int]] = None
    video: Optional[str] = None
    notes: str = ""

    # ---- persistence ----

    def to_dict(self) -> Dict:
        return {
            "version": CALIBRATION_VERSION,
            "video": self.video,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "notes": self.notes,
            "lines": [
                {
                    "name": line.name,
                    "knots": [
                        {"image_x": k[0], "image_y": k[1], "world_x": k[2]}
                        for k in line.sorted_knots()
                    ],
                }
                for line in self.lines
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Calibration":
        version = data.get("version")
        if version != CALIBRATION_VERSION:
            raise ValueError(
                f"Unsupported calibration version {version!r} "
                f"(this build writes and reads version {CALIBRATION_VERSION})."
            )
        lines = [
            ReferenceLine(
                name=entry["name"],
                knots=[
                    (float(k["image_x"]), float(k["image_y"]), float(k["world_x"]))
                    for k in entry["knots"]
                ],
            )
            for entry in data.get("lines", [])
        ]
        size = data.get("frame_size")
        return cls(
            lines=lines,
            frame_size=tuple(size) if size else None,
            video=data.get("video"),
            notes=data.get("notes", ""),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as handle:
            return cls.from_dict(json.load(handle))

    # ---- the actual mapping ----

    def world_x(self, image_x: float, image_y: float, extrapolate: bool = False) -> float:
        """
        World distance along the pool (metres) for an image point.

        With one reference line this is just that line's mapping, and carries
        no depth correction at all. With two or more, results are blended by
        image_y, so a swimmer higher or lower in the frame is read off an
        appropriately interpolated line.
        """
        usable = [line for line in self.lines if len(line.knots) >= 2]
        if not usable:
            raise ValueError("Calibration has no reference line with at least 2 knots.")

        values, ys = [], []
        for line in usable:
            world = line.world_x_at(image_x, extrapolate=extrapolate)
            if not np.isnan(world):
                values.append(world)
                ys.append(line.image_y_at(image_x))
        if not values:
            return float("nan")
        if len(values) == 1:
            return values[0]

        order = np.argsort(ys)
        ys_sorted = np.array(ys, dtype=float)[order]
        values_sorted = np.array(values, dtype=float)[order]
        return float(np.interp(image_y, ys_sorted, values_sorted))

    def depth_ambiguity(self, image_x: float) -> float:
        """
        Spread (metres) between what each reference line says about the same
        image column.

        This is a direct, model-free measurement of the error introduced by not
        knowing where across the lane the swimmer actually is: the near and far
        ropes bound their possible depth, so the disagreement between the two
        bounds the positional uncertainty. It needs no focal length, no camera
        standoff, and no assumptions -- just the same mark clicked on both
        ropes. Returns NaN if fewer than two lines cover this column.
        """
        values = [
            line.world_x_at(image_x)
            for line in self.lines
            if len(line.knots) >= 2
        ]
        values = [v for v in values if not np.isnan(v)]
        if len(values) < 2:
            return float("nan")
        return float(max(values) - min(values))

    def covered_image_x(self) -> Tuple[float, float]:
        """Image-x range where at least one line has knots -- outside this,
        every reading is extrapolation."""
        spans = [line.arrays()[0] for line in self.lines if len(line.knots) >= 2]
        if not spans:
            raise ValueError("Calibration has no reference line with at least 2 knots.")
        return (float(min(s[0] for s in spans)), float(max(s[-1] for s in spans)))


def series_world_x(
    calibration: Calibration,
    image_x: Sequence[float],
    image_y: Sequence[float],
    extrapolate: bool = False,
) -> np.ndarray:
    """Vectorized world_x over paired image coordinate sequences."""
    return np.array(
        [
            calibration.world_x(float(x), float(y), extrapolate=extrapolate)
            if np.isfinite(x) and np.isfinite(y)
            else float("nan")
            for x, y in zip(image_x, image_y)
        ],
        dtype=float,
    )
