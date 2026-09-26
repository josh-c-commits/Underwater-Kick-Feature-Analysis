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
- **Start with the swimmer in view.** For now the swimmer can only be selected on
  frame 1, so trim each clip to start once they are clearly visible.
- **Keep one format.** Run every clip through `normalize-video` once and use that file
  for every later step, so frame numbers stay consistent.

## Calibration markers

Distances come from markers placed at known positions in the swimmer's lane.

1. **Place them along the lane's centre line on the floor.** That puts them in the
   same vertical plane as the swimmer. Lane ropes and wall marks sit at different
   distances from the camera and give the wrong scale.
2. **Measure distances horizontally from the wall.** Where the floor slopes, a tape
   laid along it reads long.
3. **Space them about a metre apart** across the whole distance you want to
   measure, e.g. 0–15 m. Dense marks correct lens distortion and let the
   calibration check its own clicks.
4. **Add a second row at swimmer depth if you can**, e.g. floats tied a fixed height
   above weights. A single floor row assumes a level, square-on camera; a second
   row measures any tilt.
5. **Leave the markers in while swimming, if possible.** Tracking ignores anything
   that doesn't move, and every clip then carries its own calibration. If they must
   come out, film them for a few seconds first without touching the camera, then
   calibrate from just those frames:

   ```bash
   uv run python -m analysis calibrate clip.mp4 pool.json --line floor \
       --mark-range 0 15 1 --seconds 0 5
   ```

   If the camera might have been knocked while removing them, add `--stabilize`
   here and when tracking, so both use the same corrected coordinates.

## Checking a calibration

- The report printed after clicking gives the lens bend, the error between marks,
  and any mark that looks mis-clicked. To fix a single mark without redoing the
  rest, re-run the same command with `--edit`.
- `overlay ... --still` draws the distance lines and your clicked marks onto the
  reference image. Every ring should sit on its marker.
