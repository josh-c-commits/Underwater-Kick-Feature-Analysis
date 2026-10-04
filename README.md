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
| Swimmer tracking | Measured against hand labels on two clips: right swimmer on 72–84% of labelled frames from one click, mean speed 2–5% low (the box centre isn't the hip) |
| Speed, kick frequency, distance per kick, velocity fluctuation index | Implemented |
| Distance and depth calibration | Lens model measured on two clips (same focal length within 0.5%); floor-line ruler and depth below the surface built, tested on exact synthetic geometry and on real floor lines; first calibration of a clip needs your clicks |
| Pose landmarks (MediaPipe) | Extraction implemented; phase lag and amplitude analysis planned |

## How it works

1. **Normalize.** iPhone HDR footage is converted to standard range (roughly twice
   the floor-line contrast of a naive conversion), the frame is levelled from two
   clicks on the lane rope, and a region such as the water surface can be marked for
   later stages to ignore.
2. **Track.** A per-pixel median over the clip gives an empty-pool background plate,
   and subtracting it leaves whatever moves. You click the swimmer on any frame (a
   keyframe), and a Kalman filter follows them forwards and backwards from every
   keyframe, ruling out blobs that are the wrong size or out of reach. A frame with
   nothing plausible is reported lost rather than guessed.
3. **Stabilize** (optional). Phase correlation measures camera drift two ways: frame
   to frame (precise but drifting) and frame to background plate (noisy but
   anchored). A gated complementary filter fuses the two, and positions are
   reported in pool-fixed coordinates.
4. **Calibrate.** The refraction at the phone's flat window, which stretches the
   picture's edges underwater, is undone with one measured number per lens. Then you
   click where the pool's floor lines cross the line you swam along; one formula
   through all the clicks gives distance from the wall, and a few clicks on the lane
   rope, which floats on the surface, give depth below it. Clips without a known lens
   interpolate between clicked markers instead.
5. **Analyze.** Smoothed positions give speed, kick frequency (the dominant
   frequency of the vertical oscillation), distance per kick, and the velocity
   fluctuation index.
6. **Check.** `review` steps through a tracked clip frame by frame, with a timeline
   of problem frames (lost, merged with another swimmer, or where forward and backward
   tracking disagree). Clicking the swimmer adds a keyframe and re-tracks the clip
   instantly. Overlays draw the tracked box and calibrated distance lines back onto
   the video.

The reasoning behind each choice is in [docs/design.md](docs/design.md).

## Quickstart

Requires Python 3.9+, [uv](https://docs.astral.sh/uv/), and ffmpeg on your `PATH`.

```bash
uv sync

# Re-encode once: HDR to standard range, levelled by two clicks on the lane rope,
# and the water surface (top third) left out of later measurement
uv run python -m analysis normalize-video raw.mov data/normalized/clip.mp4 \
    --level --ignore-top 0.33

# Track: click the swimmer on a frame where they're clearly visible (here 90);
# add --stabilize if the camera may have moved
uv run python -m analysis track data/normalized/clip.mp4 out/clip/boxes.csv \
    --pick-seed 90 --preview out/clip/preview.mp4

# Step through it frame by frame; click the swimmer wherever tracking went wrong
uv run python -m analysis review data/normalized/clip.mp4 out/clip/boxes.csv

# Ground truth: click the nose, hip and toes on 40 frames spread through the clip
# (--side is the side facing the camera), then score the tracking against them
uv run python -m analysis label data/normalized/clip.mp4 --side right
uv run python -m analysis evaluate-track data/normalized/clip.mp4 out/clip/boxes.csv

# Watch it: the full frame greyed out except the swimmer, or a camera that follows them
uv run python -m analysis export data/normalized/clip.mp4 out/clip/boxes.csv out/clip/spotlight.mp4
uv run python -m analysis export data/normalized/clip.mp4 out/clip/boxes.csv out/clip/follow.mp4 \
    --mode follow

# Calibrate: click where the floor lines (6 ft from the wall, then every 9 ft) cross
# the line you swam along, then a few points on the lane rope; check it on a still.
# Saved as data/calibrations/clip.json, which later commands find on their own.
uv run python -m analysis calibrate data/normalized/clip.mp4 --line lane \
    --mark-range 6 69 9 --feet --surface
uv run python -m analysis overlay data/normalized/clip.mp4 out/clip/calibration.png --still

# Kinematics: speed in m/s, distance from the wall, and depth below the surface
uv run python -m analysis analyze out/clip/boxes.csv out/clip/kinematics.csv \
    --video data/normalized/clip.mp4
```

Without a calibration, speeds come out in px/s; kick frequency and the velocity
fluctuation index don't need one. `uv run python -m analysis --help` lists the other
commands, including pose extraction.

How to set up the camera and calibration markers: [docs/filming.md](docs/filming.md).

## Known limitations

- When another swimmer overlaps the subject in the image, the two merge into one blob
  and the position is pulled toward the other person. Such frames are flagged by a
  `size_ratio` near 2; pose landmarks will separate them.
- The box centre isn't the hip: it shifts with posture (arms forward in the glide,
  head and arms fading near the surface), so mean speed reads 2–5% low against
  hand-labelled hips. Tracking is weakest at the ends of a clip (push-off beside the
  wall, rising toward the surface), so review those.
- Stabilization corrects sliding, not rotation or zoom.
- Distances assume you swam above the floor line you clicked along: drifting 30 cm
  toward or away from the camera changes the scale by ~2%. A camera tilted 3° adds up
  to ~0.5% at the frame edges. Depth assumes the rope you clicked is at your lane's
  distance unless you give `--rope-offset`; swimming directly beneath it
  (`--under-rope`) makes depth exact.

## Development

```bash
uv run pytest
```

313 tests cover normalization, tracking, stabilization, the lens model and
calibration, kinematics, ground-truth scoring, video export and the command-line
interface.

| Module | Role |
|---|---|
| `analysis/normalize.py` | HDR conversion, levelling, and each clip's sidecar file |
| `analysis/tracking.py` | Background plate, swimmer detection, camera stabilization |
| `analysis/association.py` | Keyframes, the Kalman search, and gap filling |
| `analysis/review.py` | The frame-by-frame review window |
| `analysis/groundtruth.py` | Hand labels and scoring tracking against them |
| `analysis/lens.py` | Undoing the underwater window's refraction; which lens shot a clip |
| `analysis/calibration.py` | Image-to-world mapping, depth below the surface, self-checks |
| `analysis/analysis.py` | Kinematics and summary metrics |
| `analysis/overlay.py` | Checking overlays and calibration stills |
| `analysis/export.py` | Spotlight and follow videos of the swimmer |
| `analysis/pointpicker.py` | Zoomable click windows for seeding, calibration and labelling |
| `analysis/pose_extraction.py` | MediaPipe landmarks on tracked crops |
| `analysis/frames.py` | Video reading and background compositing |
| `analysis/cli.py` | Command-line interface |
