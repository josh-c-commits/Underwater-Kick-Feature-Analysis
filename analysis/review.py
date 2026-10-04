"""Frame-by-frame review of a tracked clip, with keyframe fixes applied live.

One window shows the frame, a timeline of the whole clip coloured by how each
frame was tracked, and the swimmer's position and speed with a cursor on the
current frame. Problems are one key away: lost frames, frames where forward and
backward tracking disagreed, and frames where the blob is about twice the
swimmer's size (usually another swimmer overlapping). Clicking the swimmer adds
a keyframe and re-tracks the whole clip from the cached candidates -- under a
second, no video re-read -- so the effect of each fix shows immediately.

Controls:
    left / right      previous / next frame
    down / up         10 frames back / forward
    home / end        first / last frame
    n / p             next / previous problem stretch
    click (frame)     the swimmer is here: add or move this frame's keyframe
    a                 the swimmer isn't visible on this frame (absent keyframe)
    d / backspace     delete this frame's keyframe
    click (timeline)  jump to that frame
    scroll, drag, r   zoom, pan, reset the view
    s                 save keyframes and the re-tracked boxes
    enter             save and close
    escape            close without saving
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

from .pointpicker import _ZoomPanView

PROBLEMS = ("lost", "conflict", "merged")
COLOURS = {  # RGB
    "detected": (0.20, 0.70, 0.30),
    "keyframe": (0.10, 0.55, 0.95),
    "merged": (0.95, 0.80, 0.10),
    "conflict": (1.00, 0.50, 0.00),
    "lost": (0.85, 0.15, 0.15),
    "absent": (0.55, 0.55, 0.55),
}
EXPLAIN = {
    "detected": "tracked",
    "keyframe": "keyframe",
    "merged": "probably merged with another swimmer",
    "conflict": "forward and backward tracking disagree",
    "lost": "lost",
    "absent": "marked absent",
}


def frame_status(table: pd.DataFrame, keyframes=()) -> np.ndarray:
    """Each frame's status, for the timeline and problem navigation."""
    found = table["found"].astype(bool).to_numpy()
    status = np.where(found, "detected", "lost").astype(object)
    if "merged" in table:
        status[found & table["merged"].fillna(False).astype(bool).to_numpy()] = "merged"
    status[table["conflict"].fillna(False).astype(bool).to_numpy()] = "conflict"
    status[(table["source"] == "keyframe").to_numpy()] = "keyframe"
    for key in keyframes:
        if key.absent and 1 <= key.frame <= len(status):
            status[key.frame - 1] = "absent"
    return status


def next_problem(status: np.ndarray, frame: int, step: int = 1) -> Optional[int]:
    """First frame of the next (step=1) or previous (step=-1) stretch of
    problem frames, skipping the rest of the stretch `frame` is in."""
    problem = np.isin(status, PROBLEMS)
    i = frame - 1
    while 0 <= i < len(status) and problem[i]:
        i += step
    while 0 <= i < len(status) and not problem[i]:
        i += step
    if not 0 <= i < len(status):
        return None
    if step < 0:  # land on the start of that stretch, not its end
        while i - 1 >= 0 and problem[i - 1]:
            i -= 1
    return i + 1


class FrameSource:
    """Frames by number, downscaled for display, read sequentially when stepping
    forward (seeking only on long jumps) and kept in a small cache.

    A seek decodes from the keyframe before the target, which in a 4K H.264
    file can be a hundred frames or more, so a short hop forward is cheaper
    done by reading through: `max_skip` frames is the crossover."""

    def __init__(self, video_path: str, display_width: int = 1920, cache_size: int = 48,
                 max_skip: int = 90):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {video_path} with OpenCV.")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = max(1.0, self.width / float(display_width))  # full-res px per display px
        self._next = 1
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size
        self._max_skip = max_skip

    def get(self, number: int) -> np.ndarray:
        if number in self._cache:
            self._cache.move_to_end(number)
            return self._cache[number]
        if self._next < number <= self._next + self._max_skip:
            while self._next < number:
                if not self.cap.grab():
                    raise RuntimeError(f"Could not read frame {self._next}.")
                self._next += 1
        elif number != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {number}.")
        self._next = number + 1
        if self.scale > 1.0:
            frame = cv2.resize(frame, (int(round(self.width / self.scale)),
                                       int(round(self.height / self.scale))),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._cache[number] = rgb
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return rgb

    def close(self) -> None:
        self.cap.release()


class Reviewer(_ZoomPanView):
    def __init__(
        self,
        video_path: str,
        boxes_csv: str,
        blobs: pd.DataFrame,
        keyframes,
        keyframes_file: str,
        fps: float,
        calibration=None,
        display_width: int = 1920,
        every: float = 1.0,
    ):
        import matplotlib.pyplot as plt

        from .tracking import boxes_from_candidates

        self._associate = boxes_from_candidates
        self.video_path, self.boxes_csv = video_path, boxes_csv
        self.keyframes_file, self.fps = keyframes_file, fps
        self.calibration, self.blobs = calibration, blobs
        self.keyframes = sorted(keyframes, key=lambda k: k.frame)
        self.source = FrameSource(video_path, display_width)
        self.scale = self.source.scale
        boxes = pd.read_csv(boxes_csv)
        self.camera = boxes[["cam_dx", "cam_dy"]].fillna(0.0).to_numpy(float)
        self.count = len(boxes)
        self.blobs_by_frame = {int(f): g for f, g in blobs.groupby("frame")} if len(blobs) else {}
        self.frame, self.saved, self.dirty = 1, False, False

        # base-class state, set up here because the layout differs
        self._plt, self._cancelled, self._press, self._marker = plt, False, None, None
        self._title = "review"
        first = self.source.get(1)
        self._height, self._width = first.shape[:2]
        self.fig = plt.figure(figsize=(15, 10.5))
        grid = self.fig.add_gridspec(3, 2, height_ratios=[8, 0.45, 2.0], hspace=0.28, wspace=0.12,
                                     left=0.05, right=0.98, top=0.93, bottom=0.05)
        self.ax = self.fig.add_subplot(grid[0, :])
        self.ax_time = self.fig.add_subplot(grid[1, :])
        self.ax_pos = self.fig.add_subplot(grid[2, 0])
        self.ax_speed = self.fig.add_subplot(grid[2, 1])
        self._image = self.ax.imshow(first)
        self.ax.set_axis_off()
        self._home = (self.ax.get_xlim(), self.ax.get_ylim())
        self._disable_default_keymap()
        self._artists: List = []
        self._lines = []
        if calibration is not None:
            from .overlay import _distance_lines

            self._lines = _distance_lines(calibration, self.source.height, every)
        # Re-tracked from the candidates rather than trusting the CSV, which may
        # predate the current keyframes or columns.
        self._retrack()

    # ---- tracking ----

    def _retrack(self) -> None:
        from .analysis import direction_of_travel, kinematics

        self.table = self._associate(self.blobs, self.camera, self.fps, self.source.width,
                                     self.keyframes)
        self.status = frame_status(self.table, self.keyframes)
        self.kin = kinematics(self.table, self.fps, calibration=self.calibration)
        # speed in the direction of travel, so a right-to-left swimmer reads positive
        self.direction = direction_of_travel(self.kin["x_smooth"]) or 1
        self._draw_timeline()
        self._draw_plots()
        self._redraw()

    def _set_keyframe(self, keyframe) -> None:
        from .association import merge_keyframes

        self.keyframes = merge_keyframes(self.keyframes, [keyframe])
        self.dirty = True
        self._retrack()

    def _delete_keyframe(self) -> None:
        kept = [k for k in self.keyframes if k.frame != self.frame]
        if len(kept) != len(self.keyframes):
            self.keyframes, self.dirty = kept, True
            self._retrack()

    def save(self) -> None:
        from .association import save_keyframes

        save_keyframes(self.keyframes_file, self.keyframes, video=self.video_path)
        self.table.to_csv(self.boxes_csv, index=False)
        self.saved, self.dirty = True, False
        self._redraw()

    # ---- navigation ----

    def go(self, frame: int) -> None:
        self.frame = int(min(max(frame, 1), self.count))
        self._image.set_data(self.source.get(self.frame))
        self._redraw()

    def _on_key(self, event) -> None:
        from .association import Keyframe

        steps = {"right": 1, "left": -1, "up": 10, "down": -10}
        key = event.key
        if key in steps:
            self.go(self.frame + steps[key])
        elif key == "home":
            self.go(1)
        elif key == "end":
            self.go(self.count)
        elif key in ("n", "p"):
            target = next_problem(self.status, self.frame, 1 if key == "n" else -1)
            if target is not None:
                self.go(target)
        elif key == "a":
            self._set_keyframe(Keyframe(self.frame))
        elif key in ("d", "backspace", "delete"):
            self._delete_keyframe()
        elif key == "s":
            self.save()
        elif key == "enter":
            self.save()
            self._finish()
        elif key == "escape":
            self._finish(cancelled=True)

    def _on_click(self, x: int, y: int) -> None:
        from .association import Keyframe

        self._set_keyframe(Keyframe(self.frame, x * self.scale, y * self.scale))

    def on_release(self, event) -> None:
        if event.inaxes is self.ax_time and event.xdata is not None and event.button == 1:
            self._press = None
            self.go(int(round(event.xdata)))
            return
        super().on_release(event)

    # ---- drawing ----

    def _draw_timeline(self) -> None:
        ax = self.ax_time
        ax.clear()
        colours = np.array([COLOURS[s] for s in self.status])[None, :, :]
        ax.imshow(colours, aspect="auto", extent=(0.5, self.count + 0.5, 0, 1),
                  interpolation="nearest")
        present = [k.frame for k in self.keyframes if not k.absent]
        if present:
            ax.plot(present, [0.5] * len(present), "v", color="white", ms=7, mec="black")
        ax.set_yticks([])
        ax.set_xlim(0.5, self.count + 0.5)
        ax.tick_params(labelsize=8)
        counts = {s: int(np.sum(self.status == s)) for s in COLOURS}
        ax.set_title("  ".join(f"{EXPLAIN[s]}: {n}" for s, n in counts.items() if n),
                     fontsize=8, loc="left")
        self._time_cursor = ax.axvline(self.frame, color="black", lw=1.5)

    def _draw_plots(self) -> None:
        frames = self.kin["frame"].to_numpy()
        metres = "world_x_m" in self.kin
        position = self.kin["world_x_m"] if metres else self.kin["x_smooth"]
        speed = (self.kin["speed_m_s"] if metres else self.kin["speed_px_s"]) * self.direction
        for ax, values, label in ((self.ax_pos, position, "distance (m)" if metres else "x (px)"),
                                  (self.ax_speed, speed, "speed (m/s)" if metres else "speed (px/s)")):
            ax.clear()
            ax.plot(frames, values, lw=1, color="tab:blue")
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_xlim(1, self.count)
        self._plot_cursors = [self.ax_pos.axvline(self.frame, color="black", lw=1),
                              self.ax_speed.axvline(self.frame, color="black", lw=1)]

    def _draw_overlay(self) -> None:
        for artist in self._artists:
            artist.remove()
        self._artists = []
        add = self._artists.append
        s = self.scale
        dx, dy = self.camera[self.frame - 1] if self.frame - 1 < len(self.camera) else (0.0, 0.0)

        for distance, segments in self._lines:  # calibration lines follow the camera
            for (x0, y0), (x1, y1), solid in segments:
                add(self.ax.plot([(x0 + dx) / s, (x1 + dx) / s], [(y0 + dy) / s, (y1 + dy) / s],
                                 color="orange", lw=0.8, ls="-" if solid else "--", alpha=0.8)[0])

        for _, blob in self.blobs_by_frame.get(self.frame, pd.DataFrame()).iterrows():
            add(self.ax.add_patch(self._rect(blob, "white", 0.6, 0.35)))

        start = max(1, self.frame - 30)
        trail = self.table[(self.table.frame >= start) & (self.table.frame <= self.frame)
                           & self.table.found]
        if len(trail) > 1:
            add(self.ax.plot(trail.centroid_x / s, trail.centroid_y / s, "-", color="cyan",
                             lw=1.2, alpha=0.7)[0])

        row = self.table.iloc[self.frame - 1]
        status = self.status[self.frame - 1]
        colour = COLOURS[status]
        if row["found"]:
            if pd.notna(row["box_x"]):
                add(self.ax.add_patch(self._rect(row, colour, 2.0, 1.0)))
            add(self.ax.plot(row["centroid_x"] / s, row["centroid_y"] / s, "o", color=colour,
                             ms=6, mec="black")[0])
        key = next((k for k in self.keyframes if k.frame == self.frame and not k.absent), None)
        if key is not None:
            add(self.ax.plot(key.x / s, key.y / s, "+", color="white", ms=14, mew=2)[0])

        if hasattr(self, "_time_cursor"):
            self._time_cursor.set_xdata([self.frame, self.frame])
            for cursor in self._plot_cursors:
                cursor.set_xdata([self.frame, self.frame])

    def _rect(self, box, colour, width, alpha):
        from matplotlib.patches import Rectangle

        s = self.scale
        return Rectangle((box["box_x"] / s, box["box_y"] / s), box["box_w"] / s, box["box_h"] / s,
                         fill=False, ec=colour, lw=width, alpha=alpha)

    def _update_title(self) -> None:
        row = self.table.iloc[self.frame - 1]
        status = self.status[self.frame - 1]
        parts = [f"frame {self.frame}/{self.count} ({(self.frame - 1) / self.fps:.2f} s)",
                 EXPLAIN[status]]
        if row["found"]:
            parts.append(f"({row['centroid_x']:.0f}, {row['centroid_y']:.0f}) px")
            if pd.notna(row.get("size_ratio")):
                parts.append(f"size x{row['size_ratio']:.2f}")
        kin = self.kin.iloc[self.frame - 1]
        if "world_x_m" in self.kin and pd.notna(kin.get("world_x_m")):
            parts.append(f"{kin['world_x_m']:.2f} m, {kin['speed_m_s'] * self.direction:.2f} m/s")
        elif pd.notna(kin.get("speed_px_s")):
            parts.append(f"{kin['speed_px_s'] * self.direction:.0f} px/s")
        state = "saved" if self.saved and not self.dirty else ("unsaved changes" if self.dirty else "")
        self.ax.set_title(
            "  |  ".join(parts) + (f"   [{state}]" if state else "") + "\n"
            "left/right=frame  up/down=10  n/p=next/prev problem  click=swimmer here  "
            "a=absent  d=delete keyframe  timeline click=jump  s=save  enter=save+close  esc=quit",
            fontsize=9,
        )

    def _result(self) -> bool:
        self.source.close()
        return self.saved


def run_review(video_path: str, boxes_csv: str, keyframes_file: str, calibration=None,
               display_width: int = 1920, every: float = 1.0) -> bool:
    """Open the reviewer; returns True if anything was saved."""
    from .association import load_keyframes
    from .frames import fps as video_fps
    from .tracking import candidates_path

    path = candidates_path(boxes_csv)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No candidates file next to {boxes_csv} ({path}). Re-run `track`, which saves it."
        )
    reviewer = Reviewer(video_path, boxes_csv, pd.read_csv(path), load_keyframes(keyframes_file),
                        keyframes_file, video_fps(video_path), calibration, display_width, every)
    return reviewer.run()
