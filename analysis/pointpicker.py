"""Zoomable/pannable click UI for picking a pixel coordinate on a frame.

The swimmer in these clips is roughly 35x84 px inside a 1624x320 frame
-- about 0.3% of the image. Rendered full-width in a plain matplotlib
window that's a target a few pixels across, so hand jitter easily puts
the click on open water instead of on the body, and there's no way to
tell that you missed.

So this exists to make the click accurate: scroll to zoom around the
cursor, drag to pan, click to place, and the placed point can be
nudged until it's right before confirming.

Controls:
    scroll          zoom in/out around the cursor
    left-drag       pan
    left-click      place (or move) the point
    u / backspace   clear the placed point
    r               reset the view to the whole frame
    enter           confirm and close
    escape          cancel (returns None)
    close window    same as enter
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]

# Mouse movement (in screen pixels) below which a press/release pair counts
# as a click rather than a pan drag.
_CLICK_TOLERANCE_PX = 5
_ZOOM_STEP = 1.3
_MIN_VISIBLE_PX = 8.0  # don't let zoom-in go below this many image pixels across
_MAX_VIEW_SCALE = 4.0  # don't let zoom-out go beyond this multiple of the image


class _ZoomPanView:
    def __init__(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "",
    ):
        import matplotlib.pyplot as plt

        self._plt = plt
        self._cancelled = False
        self._press = None  # (display_x, display_y, xlim_at_press, ylim_at_press)
        self._title = title
        self._marker = None

        display = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if mask is not None and mask.any():
            red = np.zeros_like(display)
            red[..., 0] = 255
            blended = (0.5 * display + 0.5 * red).astype(np.uint8)
            display = np.where(mask[..., None], blended, display)

        self._height, self._width = display.shape[:2]
        fig_w = 14.0
        fig_h = max(3.0, min(9.0, fig_w * self._height / self._width + 1.5))
        self.fig, self.ax = plt.subplots(figsize=(fig_w, fig_h))
        self.ax.imshow(display)
        self._home = (self.ax.get_xlim(), self.ax.get_ylim())
        # subclasses call _redraw() once their own state is in place

    # ---- view helpers ----

    def _zoom_about(self, xdata: float, ydata: float, factor: float) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        width, height = x1 - x0, y1 - y0

        new_width, new_height = width * factor, height * factor
        if abs(new_width) < _MIN_VISIBLE_PX or abs(new_height) < _MIN_VISIBLE_PX:
            return
        if abs(new_width) > self._width * _MAX_VIEW_SCALE:
            return

        # keep whatever is under the cursor pinned in place
        rel_x = (xdata - x0) / width
        rel_y = (ydata - y0) / height
        self.ax.set_xlim(xdata - new_width * rel_x, xdata + new_width * (1 - rel_x))
        self.ax.set_ylim(ydata - new_height * rel_y, ydata + new_height * (1 - rel_y))
        self.fig.canvas.draw_idle()

    def _reset_view(self) -> None:
        self.ax.set_xlim(*self._home[0])
        self.ax.set_ylim(*self._home[1])
        self.fig.canvas.draw_idle()

    # ---- subclass hooks ----

    def _on_click(self, x: int, y: int) -> None:
        """A real click (not a drag) landed at image pixel (x, y)."""

    def _on_key(self, event) -> None:
        """Key press other than 'r', which the base class already handles."""

    def _draw_overlay(self) -> None:
        """Redraw whatever markers this view shows."""

    def _result(self):
        raise NotImplementedError

    def _finish(self, cancelled: bool = False) -> None:
        self._cancelled = cancelled
        self._plt.close(self.fig)

    def _redraw(self) -> None:
        self._draw_overlay()
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self) -> None:
        self.ax.set_title(self._title, fontsize=9)

    # ---- event handlers ----

    def on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 1 / _ZOOM_STEP if event.button == "up" else _ZOOM_STEP
        self._zoom_about(event.xdata, event.ydata, factor)

    def on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1:
            return
        self._press = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def on_motion(self, event) -> None:
        if self._press is None or event.x is None:
            return
        press_x, press_y, (x0, x1), (y0, y1) = self._press
        if np.hypot(event.x - press_x, event.y - press_y) < _CLICK_TOLERANCE_PX:
            return  # not (yet) a drag; leave the view alone so a click stays a click

        bbox = self.ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        # Pan is computed against the limits captured at press time, so it stays
        # stable even though the data under the cursor moves as we redraw.
        dx = (event.x - press_x) * (x1 - x0) / bbox.width
        dy = (event.y - press_y) * (y1 - y0) / bbox.height
        self.ax.set_xlim(x0 - dx, x1 - dx)
        self.ax.set_ylim(y0 - dy, y1 - dy)
        self.fig.canvas.draw_idle()

    def on_release(self, event) -> None:
        if self._press is None:
            return
        press_x, press_y, _, _ = self._press
        self._press = None
        if event.button != 1 or event.inaxes is not self.ax or event.xdata is None:
            return
        if np.hypot(event.x - press_x, event.y - press_y) >= _CLICK_TOLERANCE_PX:
            return  # that was a pan, not a click

        x = min(max(int(round(event.xdata)), 0), self._width - 1)
        y = min(max(int(round(event.ydata)), 0), self._height - 1)
        self._on_click(x, y)

    def on_key(self, event) -> None:
        if event.key == "r":
            self._reset_view()
        else:
            self._on_key(event)

    # ---- driver ----

    def run(self):
        connect = self.fig.canvas.mpl_connect
        connect("scroll_event", self.on_scroll)
        connect("button_press_event", self.on_press)
        connect("motion_notify_event", self.on_motion)
        connect("button_release_event", self.on_release)
        connect("key_press_event", self.on_key)

        self._plt.show()  # blocks until the window is closed
        return self._result()


class _PointPicker(_ZoomPanView):
    """Place a single point."""

    def __init__(self, image_bgr, mask=None, title="", initial_point=None):
        self.point: Optional[Point] = initial_point
        self._marker = None
        super().__init__(image_bgr, mask=mask, title=title)
        self._redraw()

    def _on_click(self, x: int, y: int) -> None:
        self.point = (x, y)
        self._redraw()

    def _on_key(self, event) -> None:
        if event.key in ("u", "backspace"):
            self.point = None
            self._redraw()
        elif event.key == "enter":
            self._finish()
        elif event.key == "escape":
            self._finish(cancelled=True)

    def _draw_overlay(self) -> None:
        if self._marker is not None:
            for artist in self._marker:
                artist.remove()
            self._marker = None
        if self.point is not None:
            x, y = self.point
            self._marker = [
                self.ax.axhline(y, color="lime", lw=0.6, alpha=0.8),
                self.ax.axvline(x, color="lime", lw=0.6, alpha=0.8),
                self.ax.plot(x, y, marker="o", ms=9, mfc="none", mec="lime", mew=1.5)[0],
            ]

    # kept as the public name the rest of the codebase and tests use
    def _draw_marker(self) -> None:
        self._redraw()

    def _update_title(self) -> None:
        placed = f"point: {self.point}" if self.point is not None else "point: none yet"
        self.ax.set_title(
            f"{self._title}\n"
            "scroll=zoom  drag=pan  click=place  r=reset  u=clear  enter=confirm  esc=cancel\n"
            f"{placed}",
            fontsize=9,
        )

    def _result(self) -> Optional[Point]:
        return None if self._cancelled else self.point


class _KeypointLabeler(_ZoomPanView):
    """Place a named series of keypoints on one frame.

    Skipping is a first-class outcome, not an absence: a landmark that is
    genuinely not visible underwater ("skipped") is a different fact from one
    the labeller simply hasn't reached yet, and conflating them would make the
    ground truth silently wrong wherever a limb is occluded.
    """

    def __init__(self, image_bgr, names, mask=None, title="", existing=None):
        if not names:
            raise ValueError("names must not be empty")
        self.names: List[str] = list(names)
        self.points: Dict[str, Optional[Point]] = dict(existing or {})
        self.index = 0
        self._artists: List = []
        super().__init__(image_bgr, mask=mask, title=title)
        self._redraw()

    @property
    def current(self) -> str:
        return self.names[self.index]

    def _advance(self, step: int = 1) -> None:
        self.index = max(0, min(len(self.names) - 1, self.index + step))

    def _on_click(self, x: int, y: int) -> None:
        self.points[self.current] = (x, y)
        if self.index < len(self.names) - 1:
            self._advance()
        self._redraw()

    def _on_key(self, event) -> None:
        if event.key in ("n", " ", "right"):
            self._advance(1)
        elif event.key in ("p", "left"):
            self._advance(-1)
        elif event.key in ("u", "backspace"):
            self.points.pop(self.current, None)
        elif event.key == "s":
            self.points[self.current] = None  # explicitly not visible
            self._advance(1)
        elif event.key == "enter":
            self._finish()
            return
        elif event.key == "escape":
            self._finish(cancelled=True)
            return
        self._redraw()

    def _draw_overlay(self) -> None:
        for artist in self._artists:
            artist.remove()
        self._artists = []
        for name, point in self.points.items():
            if point is None:
                continue
            is_current = name == self.current
            colour = "yellow" if is_current else "lime"
            self._artists.append(
                self.ax.plot(point[0], point[1], marker="o", ms=10 if is_current else 7,
                             mfc="none", mec=colour, mew=1.6)[0]
            )
            self._artists.append(
                self.ax.annotate(name, point, textcoords="offset points", xytext=(6, 5),
                                 color=colour, fontsize=7)
            )

    def _update_title(self) -> None:
        placed = sum(1 for v in self.points.values() if v is not None)
        skipped = sum(1 for v in self.points.values() if v is None)
        self.ax.set_title(
            f"{self._title}\n"
            f"[{self.index + 1}/{len(self.names)}]  now placing: {self.current}   "
            f"({placed} placed, {skipped} skipped)\n"
            "click=place  n/p=next/prev  s=not visible  u=clear  r=reset  "
            "enter=done  esc=cancel",
            fontsize=9,
        )

    def _result(self) -> Optional[Dict[str, Optional[Point]]]:
        return None if self._cancelled else dict(self.points)


def pick_point(
    image_bgr: np.ndarray,
    title: str = "",
    mask: Optional[np.ndarray] = None,
    initial_point: Optional[Point] = None,
) -> Optional[Point]:
    """
    Open a zoomable window on image_bgr and return the clicked (x, y) in
    original-image pixel coordinates, or None if nothing was placed (or
    the user pressed escape). If mask is given it's tinted red over the
    image, so an existing segmentation can be judged before correcting it.
    Requires a display.
    """
    return _PointPicker(image_bgr, mask=mask, title=title, initial_point=initial_point).run()


def label_keypoints(
    image_bgr: np.ndarray,
    names,
    title: str = "",
    existing: Optional[Dict[str, Optional[Point]]] = None,
) -> Optional[Dict[str, Optional[Point]]]:
    """
    Walk through `names`, placing each keypoint on image_bgr. Returns
    {name: (x, y)} for placed points, {name: None} for ones explicitly
    marked not visible, and omits names never reached. Returns None if
    the labeller cancelled. Requires a display.
    """
    return _KeypointLabeler(image_bgr, names, title=title, existing=existing).run()
