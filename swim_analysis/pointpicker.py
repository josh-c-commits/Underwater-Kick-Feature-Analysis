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

from typing import Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]

# Mouse movement (in screen pixels) below which a press/release pair counts
# as a click rather than a pan drag.
_CLICK_TOLERANCE_PX = 5
_ZOOM_STEP = 1.3
_MIN_VISIBLE_PX = 8.0  # don't let zoom-in go below this many image pixels across
_MAX_VIEW_SCALE = 4.0  # don't let zoom-out go beyond this multiple of the image


class _PointPicker:
    def __init__(
        self,
        image_bgr: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "",
        initial_point: Optional[Point] = None,
    ):
        import matplotlib.pyplot as plt

        self._plt = plt
        self.point: Optional[Point] = initial_point
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
        self._draw_marker()

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

    def _draw_marker(self) -> None:
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
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self) -> None:
        placed = f"point: {self.point}" if self.point is not None else "point: none yet"
        self.ax.set_title(
            f"{self._title}\n"
            "scroll=zoom  drag=pan  click=place  r=reset  u=clear  enter=confirm  esc=cancel\n"
            f"{placed}",
            fontsize=9,
        )

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
        self.point = (x, y)
        self._draw_marker()

    def on_key(self, event) -> None:
        if event.key == "r":
            self._reset_view()
        elif event.key in ("u", "backspace"):
            self.point = None
            self._draw_marker()
        elif event.key == "enter":
            self._plt.close(self.fig)
        elif event.key == "escape":
            self._cancelled = True
            self._plt.close(self.fig)

    # ---- driver ----

    def run(self) -> Optional[Point]:
        connect = self.fig.canvas.mpl_connect
        connect("scroll_event", self.on_scroll)
        connect("button_press_event", self.on_press)
        connect("motion_notify_event", self.on_motion)
        connect("button_release_event", self.on_release)
        connect("key_press_event", self.on_key)

        self._plt.show()  # blocks until the window is closed
        return None if self._cancelled else self.point


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
