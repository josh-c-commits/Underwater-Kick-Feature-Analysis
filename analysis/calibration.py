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
A calibration is a set of *reference lines*, each carrying knots of the form
(image_x, image_y, world_x). Along a line, world_x is interpolated between
knots; between lines, results are blended by image_y. So the same image column
can map to different world positions depending on how high in the frame it
sits, which is the point: a mark 15m down the pool projects to a different
column on a line near the camera than on one far from it.

A line's marks should share a plane with the swimmer, because the swimmer is
what gets measured against them. The best line is a row of markers placed at
measured distances along the floor of the swimmer's lane: every mark then
lies in the swimmer's own vertical plane, as densely as markers were placed.
Existing pool features are a fallback. A lane rope sits far nearer the camera
than the swimmer, so its marks have a different scale altogether.

Interpolation between knots is piecewise linear, which is only adequate
because knots are meant to be dense. Two knots cannot describe a curve *and*
leave any residual to check accuracy against; lane-rope colour-block
boundaries give 15-20 across a frame, and at that spacing linear segments are
fine. Outside a line's knot range the result is NaN unless extrapolation is
explicitly requested -- silently clamping would hand back confident numbers
for the part of the pool that was never calibrated.

Dense marks also check each other. bend_report() uses them to measure how far
the lens bends the scale, how well interpolation between marks holds up, and
which mark, if any, looks mis-clicked.

The lens model
--------------
For clips whose lens is known (lens.py: the iPhone 0.5x behind its flat window
underwater), the refraction is undone first, and then no interpolation is
needed: through an ideal camera, distance X along a straight pool line and
image position u relate exactly by X = (p0 + p1 u) / (1 + p2 u). Three numbers
per line, fitted to all its marks at once, so click errors average out, a
mis-click stands out against the fit, and readings stay right beyond the
outermost marks. Sparse marks are enough: the pool's own floor lines, 9 ft
apart, crossing the floor line the swimmer swam along.

That line sits at the swimmer's distance from the camera, so its ruler reads
the swimmer directly. A level, square-on camera sees the swimmer's vertical
plane at a single scale, the same horizontally and vertically, so the ruler's
metres-per-pixel also converts vertical pixels: depth below the surface is the
height of the swimmer below the lane rope (which floats on the surface), in
those units. The floor's own depth never enters -- a floor that dips moves its
lines up or down in the picture, not sideways -- which matters because pool
floors slope and nobody measures them. With `under_rope` the swimmer swam
directly beneath the rope: two lane lines are clicked, one either side, and the
swimmer is read halfway between them, with the rope at exactly their distance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lens import FlatPort

# 2 added the lens model, the surface (lane rope) and the swimmer's-lane options;
# version 1 files still load and work as before.
CALIBRATION_VERSION = 2


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

    def image_x_for(self, world_x: float) -> Optional[float]:
        """Inverse of world_x_at: the image column where this line reads
        `world_x`, or None if that distance lies outside its knots."""
        xs, _, ws = self.arrays()
        if len(xs) < 2 or not (min(ws) <= world_x <= max(ws)):
            return None
        if ws[-1] < ws[0]:  # distances decrease left-to-right: flip for np.interp
            xs, ws = xs[::-1], ws[::-1]
        return float(np.interp(world_x, ws, xs))

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
    # Which image the marks were clicked on: "median" or "stabilized". Anything
    # drawn or measured against this calibration has to use the same one, or it
    # sits offset by however far apart the two images are.
    reference: str = "median"
    # (first, last) frames the reference image was built from, when markers
    # were only down for part of the clip; None means the whole clip.
    frames: Optional[Tuple[int, int]] = None
    # The lens model (see the module docstring). Without one, marks are
    # interpolated as described above.
    lens: Optional[FlatPort] = None
    # Points clicked along the lane rope, which floats on the surface: the
    # reference for depth, and for which way is level.
    surface: List[Tuple[float, float]] = field(default_factory=list)
    # The swimmer swam directly beneath the rope, between the first two lines.
    under_rope: bool = False
    # How much nearer the camera the rope is than the swimmer (m), and how deep
    # the camera was (m): together they correct depth for the rope not being at
    # the swimmer's distance. 0 = assume it is.
    rope_offset: float = 0.0
    camera_depth: float = 0.5

    # ---- persistence ----

    def to_dict(self) -> Dict:
        return {
            "version": CALIBRATION_VERSION,
            "video": self.video,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "notes": self.notes,
            "reference": self.reference,
            "frames": list(self.frames) if self.frames else None,
            "lens": self.lens.to_dict() if self.lens else None,
            "surface": [list(point) for point in self.surface],
            "under_rope": self.under_rope,
            "rope_offset": self.rope_offset,
            "camera_depth": self.camera_depth,
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
        if version not in (1, CALIBRATION_VERSION):
            raise ValueError(
                f"Unsupported calibration version {version!r} "
                f"(this build reads versions 1 and {CALIBRATION_VERSION})."
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
        frames = data.get("frames")
        return cls(
            lines=lines,
            frame_size=tuple(size) if size else None,
            video=data.get("video"),
            notes=data.get("notes", ""),
            reference=data.get("reference", "median"),
            frames=(int(frames[0]), int(frames[1])) if frames else None,
            lens=FlatPort.from_dict(data["lens"]) if data.get("lens") else None,
            surface=[(float(x), float(y)) for x, y in data.get("surface", [])],
            under_rope=bool(data.get("under_rope", False)),
            rope_offset=float(data.get("rope_offset", 0.0)),
            camera_depth=float(data.get("camera_depth", 0.5)),
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
        if self.lens is not None:
            return float(self._lens_world_x(image_x, image_y))

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
        if self.lens is not None:
            return float("nan")  # the lens model reads the swimmer's own lane line
        values = [
            line.world_x_at(image_x)
            for line in self.lines
            if len(line.knots) >= 2
        ]
        values = [v for v in values if not np.isnan(v)]
        if len(values) < 2:
            return float("nan")
        return float(max(values) - min(values))

    def world_range(self) -> Tuple[float, float]:
        """Smallest and largest distance any usable line has a knot at."""
        worlds = [w for line in self.lines if len(line.knots) >= 2 for w in line.arrays()[2]]
        if not worlds:
            raise ValueError("Calibration has no reference line with at least 2 knots.")
        return float(min(worlds)), float(max(worlds))

    def isoline(
        self, world_x: float, frame_height: int
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], bool]]:
        """
        Where the calibration reads `world_x`, as image-space segments
        (start, end, interpolated) in plate coordinates.

        This traces what world_x() actually computes rather than an idealised
        line. Between reference lines the result is interpolated, so those
        segments are marked True. Above the top line and below the bottom one,
        world_x() holds the nearest line's value, which makes the true isoline
        vertical there; those segments are marked False. That distinction is the
        point of drawing it: on a tilted camera the real pool markings lean while
        the held segments stay vertical, so the gap between them is visible.
        """
        if self.lens is not None:
            return self._lens_isoline(world_x, frame_height)
        points = []
        for line in self.lines:
            if len(line.knots) < 2:
                continue
            column = line.image_x_for(world_x)
            if column is not None:
                points.append((column, line.image_y_at(column)))
        if not points:
            return []

        points.sort(key=lambda point: point[1])
        top, bottom = points[0], points[-1]
        segments = [((top[0], 0.0), top, False)]
        segments += [(a, b, True) for a, b in zip(points, points[1:])]
        segments.append((bottom, (bottom[0], float(frame_height - 1)), False))
        return segments

    def covered_image_x(self) -> Tuple[float, float]:
        """Image-x range where at least one line has knots -- outside this,
        every reading is extrapolation."""
        spans = [line.arrays()[0] for line in self.lines if len(line.knots) >= 2]
        if not spans:
            raise ValueError("Calibration has no reference line with at least 2 knots.")
        return (float(min(s[0] for s in spans)), float(max(s[-1] for s in spans)))


    # ---- the lens model ----

    def roll(self) -> float:
        """Radians the (undistorted) lane rope leans: the camera's leftover roll.
        0 without a surface."""
        if self.lens is None or len(self.surface) < 2:
            return 0.0
        xs, ys = self.lens.undistort(*np.array(self.surface, dtype=float).T)
        return float(np.arctan(np.polyfit(xs, ys, 1)[0]))

    def ideal(self, image_x, image_y) -> Tuple[np.ndarray, np.ndarray]:
        """Image points -> an ideal, level camera's coordinates: the lens undone,
        then turned so the surface is horizontal."""
        ux, uy = self.lens.undistort(image_x, image_y)
        angle = -self.roll()
        if angle:
            dx, dy = ux - self.lens.cx, uy - self.lens.cy
            c, s = np.cos(angle), np.sin(angle)
            ux, uy = self.lens.cx + c * dx - s * dy, self.lens.cy + s * dx + c * dy
        return ux, uy

    def _from_ideal(self, ux, uy) -> Tuple[np.ndarray, np.ndarray]:
        angle = self.roll()
        if angle:
            dx, dy = ux - self.lens.cx, uy - self.lens.cy
            c, s = np.cos(angle), np.sin(angle)
            ux, uy = self.lens.cx + c * dx - s * dy, self.lens.cy + s * dx + c * dy
        return self.lens.distort(ux, uy)

    def ruler(self, line: ReferenceLine) -> "Ruler":
        xs, ys, ws = line.arrays()
        ux, _ = self.ideal(xs, ys)
        width = (self.frame_size[0] if self.frame_size else 2 * self.lens.cx)
        edge_x, _ = self.ideal(np.array([0.0, width]), np.array([self.lens.cy, self.lens.cy]))
        return Ruler.fit(ux, ws, span=(float(min(edge_x)), float(max(edge_x))))

    def swimmer_rulers(self) -> List["Ruler"]:
        usable = [line for line in self.lines if len(line.knots) >= 2]
        return [self.ruler(line) for line in usable[:2 if self.under_rope else 1]]

    def _lens_world_x(self, image_x, image_y):
        ux, _ = self.ideal(image_x, image_y)
        return np.mean([ruler(ux) for ruler in self.swimmer_rulers()], axis=0)

    def metres_per_px(self, image_x, image_y):
        """The scale at the swimmer's distance, the same horizontally and
        vertically for a square-on camera."""
        ux, _ = self.ideal(image_x, image_y)
        return np.abs(np.mean([ruler.slope(ux) for ruler in self.swimmer_rulers()], axis=0))

    def swimmer_distance(self) -> float:
        """How far the swimmer's lane is from the camera (m), from the ruler's
        scale and the lens's focal length."""
        ruler = self.swimmer_rulers()[0]
        middle = float(np.mean(ruler.u))
        scales = [abs(r.slope(middle)) for r in self.swimmer_rulers()]
        return float(np.mean(scales) * self.lens.ideal_focal)

    def depth(self, image_x, image_y):
        """
        Depth below the water surface (m) of image points on the swimmer: their
        height below the lane rope in the level ideal image, in the ruler's
        metres-per-pixel. NaN without a lens model and a surface.

        The rope floats at the surface but may not be at the swimmer's distance;
        rope_offset (how much nearer it is, m) with camera_depth corrects for
        that: from below, a nearer stretch of surface appears higher, by
        camera_depth * (distance ratio - 1) once scaled.
        """
        image_x = np.asarray(image_x, dtype=float)
        if self.lens is None or len(self.surface) < 2:
            return np.full(image_x.shape, np.nan)
        ux, uy = self.ideal(image_x, image_y)
        sx, sy = self.ideal(*np.array(self.surface, dtype=float).T)
        surface_row = float(np.mean(sy))
        depth = (uy - surface_row) * self.metres_per_px(image_x, image_y)
        if self.rope_offset and not self.under_rope:
            distance = self.swimmer_distance()
            depth = depth - self.camera_depth * (distance / (distance - self.rope_offset) - 1)
        return depth

    def _lens_isoline(self, world_x: float, frame_height: int):
        rulers = self.swimmer_rulers()
        lo, hi = rulers[0].span
        grid = np.linspace(lo, hi, 4001)
        values = np.mean([ruler(grid) for ruler in rulers], axis=0)
        order = np.argsort(values)
        if not (values.min() <= world_x <= values.max()):
            return []
        column = float(np.interp(world_x, values[order], grid[order]))
        low, high = self.world_range()
        interpolated = low <= world_x <= high
        rows = np.linspace(-0.2 * frame_height, 1.2 * frame_height, 60)
        xs, ys = self._from_ideal(np.full(rows.shape, column), rows)
        points = [(float(x), float(y)) for x, y in zip(xs, ys)
                  if np.isfinite(x) and np.isfinite(y) and 0 <= y <= frame_height - 1]
        return [(a, b, interpolated) for a, b in zip(points, points[1:])]


@dataclass
class Ruler:
    """
    Distance along one straight pool line from ideal-camera image position:
    X = (p0 + p1 t) / (1 + p2 t) with t = u / 1000 -- exactly how a straight
    line maps through an ideal camera (p2 is perspective from the camera being
    turned; ~0 when it's square-on). Fitted to all the line's marks at once.
    """

    params: Tuple[float, float, float]
    u: np.ndarray
    metres: np.ndarray
    span: Tuple[float, float]

    @classmethod
    def fit(cls, u, metres, span) -> "Ruler":
        u, metres = np.asarray(u, dtype=float), np.asarray(metres, dtype=float)
        t = u / 1000.0
        linear = np.polyfit(t, metres, 1)
        params = (float(linear[1]), float(linear[0]), 0.0)
        if len(u) >= 4:
            design = np.column_stack([np.ones_like(t), t, -t * metres])
            p0, p1, p2 = np.linalg.lstsq(design, metres, rcond=None)[0]
            edges = np.array(span) / 1000.0
            # keep it only if it's a plausible camera: no pole across the frame
            # and a perspective term that changes the scale by under 30% there
            if np.all(np.abs(p2 * edges) < 0.3):
                params = (float(p0), float(p1), float(p2))
        return cls(params, u, metres, span)

    def __call__(self, u):
        t = np.asarray(u, dtype=float) / 1000.0
        p0, p1, p2 = self.params
        return (p0 + p1 * t) / (1 + p2 * t)

    def slope(self, u):
        """dX/du in metres per pixel."""
        t = np.asarray(u, dtype=float) / 1000.0
        p0, p1, p2 = self.params
        return (p1 - p0 * p2) / (1 + p2 * t) ** 2 / 1000.0

    def residuals(self) -> np.ndarray:
        """Each mark's distance from the fit (m)."""
        return self.metres - self(self.u)

    def left_out_residuals(self) -> np.ndarray:
        """Each mark's miss when the fit is made without it: an honest check,
        and the way a single mis-click shows up."""
        misses = []
        for i in range(len(self.u)):
            keep = np.arange(len(self.u)) != i
            if keep.sum() < 2:
                return np.full(len(self.u), np.nan)
            other = Ruler.fit(self.u[keep], self.metres[keep], self.span)
            misses.append(float(self.metres[i] - other(self.u[i])))
        return np.array(misses)


def ruler_report(calibration: Calibration, line: ReferenceLine) -> List[str]:
    """How one line's marks fit the lens model, and any mark that looks mis-clicked."""
    ruler = calibration.ruler(line)
    misses = ruler.left_out_residuals()
    scale = abs(float(ruler.slope(np.mean(ruler.u))))
    lines = [f"{line.name}: {len(ruler.u)} marks; {1 / scale:.0f} px per metre at this line."]
    # Leaving one out needs enough marks for the rest to fit the full formula.
    if len(ruler.u) >= 5 and np.isfinite(misses).all():
        lines.append(f"  each mark predicted from the others: within {np.abs(misses).max() * 100:.1f} cm "
                     f"(typically {np.median(np.abs(misses)) * 100:.1f} cm)")
        typical = np.median(np.abs(misses))
        for metres, miss in zip(ruler.metres, misses):
            if abs(miss) > max(3 * typical, 0.05):
                lines.append(f"  the {metres:g} m mark is {abs(miss) * 100:.0f} cm off the others: "
                             "probably mis-clicked, or the wrong distance. Re-click it with --edit.")
    else:
        fitted = ruler.residuals()
        lines.append(f"  marks sit within {np.abs(fitted).max() * 100:.1f} cm of the fit; click 5 or more "
                     "to check each against the others.")
    return lines


def calibration_path(video: str, folder: str = os.path.join("data", "calibrations")) -> str:
    """Where a clip's calibration lives by default: data/calibrations/<clip>.json."""
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(folder, stem + ".json")


def find_calibration(video: Optional[str]) -> Optional[str]:
    """The clip's calibration file if it has one at the default place."""
    if not video:
        return None
    path = calibration_path(video)
    return path if os.path.exists(path) else None


def coincident_lines(lines: Sequence[ReferenceLine], tolerance: float = 10.0) -> List[Tuple[str, str]]:
    """
    Pairs of reference lines whose shared marks all sit within `tolerance`
    pixels of each other -- i.e. the same physical line clicked twice.

    A second line only helps if it runs somewhere else in the image: its job is
    to show how the distance scale changes with image height. Two copies of one
    line tell the calibration nothing new, yet would still make depth ambiguity
    read as ~0, which looks like a good calibration rather than an empty one.
    """
    pairs = []
    for i, first in enumerate(lines):
        for second in lines[i + 1:]:
            marks = {w: (x, y) for x, y, w in first.knots}
            shared = [(marks[w], (x, y)) for x, y, w in second.knots if w in marks]
            if shared and all(np.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance for a, b in shared):
                pairs.append((first.name, second.name))
    return pairs


def reference_line_from_clicks(
    name: str,
    placed: Dict[str, Optional[Tuple[float, float]]],
    names: Sequence[str],
    marks: Sequence[float],
) -> Tuple[Optional[ReferenceLine], str]:
    """
    Turn keypoint-labeller output into a ReferenceLine.

    Returns (line, "") on success, or (None, reason) when the clicks cannot
    define a usable line -- so the caller can drop that one line and keep the
    rest, rather than discarding every line because one mark wasn't visible.
    """
    knots = [
        (float(placed[label][0]), float(placed[label][1]), float(mark))
        for label, mark in zip(names, marks)
        if placed.get(label) is not None
    ]
    if len(knots) < 2:
        return None, (
            f"only {len(knots)} mark(s) placed -- a line needs two known points "
            "to define a scale along it"
        )

    ordered = sorted(knots, key=lambda k: k[0])
    image_steps = np.diff([k[0] for k in ordered])
    world_steps = np.diff([k[2] for k in ordered])
    # Along one physical line, distance must change monotonically with image
    # position. Anything else is a mis-click (a mark placed on the wrong side of
    # another), and interpolating through it would silently produce nonsense.
    if np.any(image_steps <= 0) or not (np.all(world_steps > 0) or np.all(world_steps < 0)):
        return None, (
            "marks aren't in distance order along the line (or two share a column) "
            "-- most likely a mis-click"
        )
    return ReferenceLine(name=name, knots=knots), ""


def straight_ruler_error(line: ReferenceLine) -> Optional[Tuple[float, float, float]]:
    """
    (metres, pixels, at): the furthest any inner mark sits from a straight
    ruler drawn through the line's two end marks, and the distance it happens
    at. None with fewer than three marks.

    This is how wrong a two-mark calibration would have been. A wide lens
    squeezes the scale towards the frame edges, so a straight ruler matches
    at its ends and drifts in between; only marks in between reveal how much.
    """
    xs, _, ws = line.arrays()
    if len(xs) < 3:
        return None
    slope = (ws[-1] - ws[0]) / (xs[-1] - xs[0])  # metres per pixel
    errors = ws[0] + (xs - xs[0]) * slope - ws
    worst = int(np.argmax(np.abs(errors[1:-1]))) + 1
    return float(abs(errors[worst])), float(abs(errors[worst] / slope)), float(ws[worst])


def _neighbour_misses(xs: np.ndarray, ws: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """(distance, metres, signed pixels, gap) per inner mark. The pixel miss is
    signed (the mark's column minus where its neighbours put it) because a
    mis-click's echo on the next mark has the opposite sign."""
    misses = []
    for k in range(1, len(xs) - 1):
        slope = (ws[k + 1] - ws[k - 1]) / (xs[k + 1] - xs[k - 1])
        metres = ws[k - 1] + (xs[k] - xs[k - 1]) * slope - ws[k]
        gap = (xs[k + 1] - xs[k - 1]) / 2.0
        misses.append((float(ws[k]), float(abs(metres)), float(metres / slope), float(gap)))
    return misses


def neighbour_misses(line: ReferenceLine) -> List[Tuple[float, float, float, float]]:
    """
    (distance, metres, pixels, gap) for each inner mark: how far it sits from
    the straight line joining its two neighbours, and their average spacing
    in pixels.

    This is leave-one-out on the calibration's own interpolation. Drop a mark,
    read its distance off its neighbours, and compare. It measures what
    interpolation gets wrong (lens curvature plus click error) at twice the
    real mark spacing, so readings between the actual marks do better.
    """
    xs, _, ws = line.arrays()
    return [(world, metres, abs(pixels), gap)
            for world, metres, pixels, gap in _neighbour_misses(xs, ws)]


def suspect_marks(
    line: ReferenceLine,
    fraction: float = 0.15,
    noise_factor: float = 5.0,
    min_marks: int = 6,
) -> List[Tuple[float, float, float]]:
    """
    Likely mis-clicks, as (distance, metres, pixels), worst first.

    A mis-click is a spike: the mark misses the line through its neighbours by
    the full error, and each neighbour misses by about half as much. Lens bend
    instead changes smoothly from mark to mark. So the worst mark is flagged
    only if it misses by more than `fraction` of the local mark spacing *and*
    by `noise_factor` times the typical miss among marks it doesn't disturb.
    Then it's set aside and the rest are re-checked.

    Simulated on a lens whose scale changes ~20% across the frame, as this
    footage's does, it caught every inner mark clicked 30% of a gap off, with
    marks anywhere from 1m to 2.5m apart, and at most 1% false flags at 4px of
    click noise. On a far more strongly bent lens it needs marks every metre to
    stay reliable: with sparse marks there, bend and mis-clicks look alike.
    It needs six marks.

    An end mark has no neighbour beyond it, so it can't be checked directly.
    But a mis-clicked end mark drags the prediction for the mark next to it,
    which then looks like the culprit. The two cases differ one mark further
    in: a mis-clicked inner mark skews its other neighbour's prediction too,
    at about half strength, while a mis-clicked end mark leaves that mark
    alone. That difference, including the echo's opposite sign, decides which
    of the two gets flagged. It catches an end mark about half the time when
    it is 30% of a gap off, and nearly always at 50%.
    """
    xs, _, ws = line.arrays()
    keep = list(range(len(xs)))
    flagged: List[Tuple[float, float, float]] = []
    while len(keep) >= min_marks and len(flagged) < max(1, len(xs) // 4):
        misses = _neighbour_misses(xs[keep], ws[keep])
        worst = int(np.argmax([abs(pixels) / gap for _, _, pixels, gap in misses]))
        # Neighbours of the worst mark carry half its error, so they can't
        # vouch for what a normal miss looks like.
        others = [abs(m[2]) for i, m in enumerate(misses) if abs(i - worst) > 1]
        typical = float(np.median(others)) if others else 0.0
        world, metres, signed, gap = misses[worst]
        pixels = abs(signed)
        if pixels <= max(fraction * gap, noise_factor * typical):
            break
        culprit = worst + 1  # misses[i] describes keep[i + 1]
        last = len(keep) - 1
        if worst in (0, len(misses) - 1):
            inner = 1 if worst == 0 else len(misses) - 2
            echo = misses[inner][2]
            if not (echo * signed < 0 and abs(echo) >= 0.25 * pixels):
                # The end mark's error reached its neighbour through the weight
                # it carries in that neighbour's interpolation.
                w = ws[keep]
                if worst == 0:
                    culprit, weight = 0, (w[2] - w[1]) / (w[2] - w[0])
                else:
                    culprit, weight = last, (w[last - 1] - w[last - 2]) / (w[last] - w[last - 2])
                world, metres, pixels = float(w[culprit]), metres / weight, pixels / weight
        flagged.append((world, metres, pixels))
        del keep[culprit]
    return flagged


def _length(metres: float) -> str:
    if metres < 0.01:
        return "under 1cm"
    if metres < 1.0:
        return f"{metres * 100:.0f}cm"
    return f"{metres:.2f}m"


def bend_report(line: ReferenceLine) -> List[str]:
    """
    Plain-language check of one reference line's marks against each other:
    how much the lens bends the scale, how well interpolation between marks
    holds up, and which marks look mis-clicked.
    """
    xs, _, ws = line.arrays()
    count = len(xs)
    if count < 3:
        return [f"{line.name}: {count} marks, so there's no way to check this ruler is "
                "straight. A third mark between them is the first test of lens bend."]

    report = [f"{line.name} ({count} marks, {ws.min():g}m to {ws.max():g}m):"]
    suspects = suspect_marks(line) if count >= 6 else []
    # A mis-click would otherwise pose as lens bend, so the figures below are
    # measured without it.
    flagged = {world for world, _, _ in suspects}
    clean = ReferenceLine(line.name, [k for k in line.knots if k[2] not in flagged])
    aside = " (leaving out the possible mis-click below)" if suspects else ""

    metres, pixels, at = straight_ruler_error(clean)
    report.append(f"  lens bend: a straight ruler through just the end marks would be off "
                  f"by up to {_length(metres)} ({pixels:.0f}px), at {at:g}m{aside}. The "
                  "marks in between correct for that.")
    if len(clean.knots) >= 4:
        misses = neighbour_misses(clean)
        typical = float(np.median([m[1] for m in misses]))
        worst = max(misses, key=lambda m: m[1])
        report.append(f"  between marks: each inner mark sits about {_length(typical)} from "
                      f"the line joining its two neighbours ({_length(worst[1])} at worst, "
                      f"at {worst[0]:g}m). That's the ruler's error with marks twice as far "
                      "apart as yours, so readings between your marks should do better.")
    for world, metres, pixels in suspects:
        report.append(f"  possible mis-click: {world:g}m is {_length(metres)} "
                      f"({pixels:.0f}px) out of line with the marks around it, far more "
                      "than the rest. Check its ring with `overlay --still`; "
                      "`calibrate ... --edit` reopens your clicks to fix it.")
    if count >= 6 and not suspects:
        report.append("  clicks: no mark stands out from its neighbours.")
    return report


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
