# Underwater Kick Feature Analysis

Measures dolphin-kick kinematics (speed, kick frequency, distance per kick) from
underwater video, tracking the swimmer with classical computer vision rather than a
trained detector.

The end goal is to find which features of an underwater kick, such as rate,
amplitude and body-wave phase lag, make a swimmer fast. Most of the work so far is
the measurement underneath that question: following a small swimmer through moving
caustics, lane ropes and other people, holding the camera still in software, and
turning pixels into metres through a wide lens and a flat port.

## Status

| Part | Status |
|---|---|
| Camera stabilization | Validated: ~0.4 px median error against injected, known camera drift; steady footage passes through unchanged |
| Swimmer tracking | Implemented; accuracy against hand-labelled positions not yet measured |
| Speed, kick frequency, distance per kick, velocity fluctuation index | Implemented |
| Distance calibration | Built, with self-checks tested in simulation; awaiting footage with floor markers |
| Pose landmarks (MediaPipe) | Extraction implemented; phase lag and amplitude analysis planned |

## How it works

1. **Track.** A per-pixel median over the clip gives an empty-pool background plate,
   and subtracting it leaves whatever moves. The swimmer is picked out from ripple,
   lane ropes and other swimmers by a search band, noise-robust thresholds, and
   continuity of position and size.
2. **Stabilize** (optional). Phase correlation measures camera drift two ways: frame
   to frame (precise but drifting) and frame to background plate (noisy but
   anchored). A gated complementary filter fuses the two, and positions are
   reported in pool-fixed coordinates.
3. **Calibrate.** Click markers placed at measured distances along the swimmer's
   lane. Distances between them are interpolated, so lens distortion and flat-port
   refraction are measured rather than modelled.
4. **Analyze.** Smoothed positions give speed, kick frequency (the dominant
   frequency of the vertical oscillation), distance per kick, and the velocity
   fluctuation index.
5. **Check.** Overlays draw the tracked box and calibrated distance lines back onto
   the video, so every result can be checked by eye.

The reasoning behind each choice is in [docs/design.md](docs/design.md).

## Quickstart

Requires Python 3.9+, [uv](https://docs.astral.sh/uv/), and ffmpeg on your `PATH`.

```bash
uv sync

# Re-encode once, so every stage decodes identical frames
uv run python -m analysis normalize-video raw.mov data/normalized/clip.mp4

# Track: click the swimmer on frame 1 (add --stabilize if the camera may have moved)
uv run python -m analysis track data/normalized/clip.mp4 out/clip/boxes.csv \
    --pick-seed --preview out/clip/preview.mp4

# Calibrate from floor markers every metre out to 15 m, then check it on a still
uv run python -m analysis calibrate data/normalized/clip.mp4 data/calibrations/pool.json \
    --line floor --mark-range 0 15 1
uv run python -m analysis overlay data/normalized/clip.mp4 out/clip/calibration.png \
    --calibration data/calibrations/pool.json --still

# Kinematics
uv run python -m analysis analyze out/clip/boxes.csv out/clip/kinematics.csv \
    --video data/normalized/clip.mp4 --calibration data/calibrations/pool.json
```

Without a calibration, speeds come out in px/s; kick frequency and the velocity
fluctuation index don't need one. `uv run python -m analysis --help` lists the other
commands, including pose extraction and hand-labelling ground truth.

How to set up the camera and calibration markers: [docs/filming.md](docs/filming.md).

## Known limitations

- The swimmer can only be selected on frame 1, so clips should start with them
  clearly visible.
- After losing the swimmer, the tracker widens its search and can pick up someone
  else.
- Stabilization corrects sliding, not rotation or zoom.
- With one row of markers, readings assume a level camera square-on to the lane. A
  second row at swimmer depth measures any tilt.

## Development

```bash
uv run pytest
```

222 tests cover tracking, stabilization, calibration, kinematics and the
command-line interface.

| Module | Role |
|---|---|
| `analysis/tracking.py` | Background plate, swimmer detection, camera stabilization |
| `analysis/calibration.py` | Image-to-world mapping and its self-checks |
| `analysis/analysis.py` | Kinematics and summary metrics |
| `analysis/overlay.py` | Review videos and calibration stills |
| `analysis/pointpicker.py` | Zoomable click windows for seeding and calibration |
| `analysis/pose_extraction.py` | MediaPipe landmarks on tracked crops |
| `analysis/frames.py` | Video reading and background compositing |
| `analysis/cli.py` | Command-line interface |
