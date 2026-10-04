# Filming guide

How to record footage the pipeline can measure well.

## Camera

- **Side-on and level.** Mount the camera underwater, looking across the swimmer's
  lane, square-on to it, with the horizon level. Distance readings are most
  trustworthy for a camera that doesn't look along the lane or tilt up or down.
- **Fixed mount.** Stabilization corrects a camera that slides a few pixels, but not
  one that rotates or zooms. Clamp it if you can.
- **Frame rate.** 60 fps or more. A dolphin kick takes around half a second, so
  60 fps gives about 30 samples per kick.
- **Keep one format.** Run every clip through `normalize-video` once and use that file
  for every later step, so frame numbers stay consistent. Use `--level` to straighten
  the frame (click two points far apart on the lane rope), and `--ignore-top 0.33` to
  leave the water surface out of measurement, as long as the swimmer stays below it
  until the breakout.
- **Turn off the phone's video stabilization** once the camera is on a tripod. It shifts
  and warps each frame slightly, which moves the lens centre around.

## Calibration

With the 0.5× lens in 4K, the pool's own floor lines are the ruler and no markers are
needed.

1. **Swim directly above a floor line**, your lane's centre line. Its distance from
   the camera sets the scale, and the calibration measures it from your clicks on
   that line. Drifting 30 cm toward or away from the camera costs about 2%.
2. **Keep a lane rope in view** for depth below the surface: it floats on the surface,
   so depth is measured down from it. For the most accurate depth, swim directly
   beneath a rope and calibrate with `--under-rope`.
3. **Note roughly how deep the phone is** (within 25 cm), in case the rope you click
   isn't at your lane's distance and you want `--rope-offset` to correct for it.
4. **Calibrate each clip** unless the camera stayed put between clips:

   ```bash
   uv run python -m analysis calibrate clip.mp4 --line lane --mark-range 6 69 9 --feet --surface
   ```

   Click where each floor line crosses the line you swam along, starting from the
   wall you pushed off from (`s` skips a line that's out of view), then 3–5 points
   spread along the rope.

**Other footage** (another lens, or 1080p, where the lens model isn't measured) still
uses markers at known distances along the floor of your lane, about a metre apart.
If they can't stay in during the swim, film them for a few seconds first and
calibrate from those frames with `--seconds 0 5` (add `--stabilize` if the camera may
have been knocked).

## Checking a calibration

- The report printed after clicking says how well the marks fit the lens model (each
  predicted from the others), names any that look mis-clicked, and gives your lane's
  distance from the camera. To fix a single mark without redoing the rest, re-run
  the same command with `--edit`.
- `overlay ... --still` draws the distance lines and your clicked marks onto the
  reference image. Every ring should sit on its marker.
