# Design notes

Why the pipeline works the way it does, stage by stage. The code's docstrings go
into more detail; this is the overview.

## Normalization

**HDR to standard range.** iPhones record HLG HDR: 10-bit, with BT.2020 colour
primaries. Read as ordinary video, those values come out hazy and desaturated.
Normalization applies the ITU-R BT.2408 conversion instead, baked into a 3D lookup
table that ffmpeg applies natively. Two choices favour measurement over looks:

- HLG reference white lands at 70% of SDR white, not 100%. Sunlit white floor tiles
  sit above reference white, and at 100% they clipped: 16% of a test frame lost its
  texture. At 70%, 0.5% did.
- Underwater blue-cyan lies outside the standard colour range. Instead of being
  clipped, which changes brightness, it is pulled toward grey at its own brightness,
  keeping every brightness difference the tracker relies on.

On a test frame this roughly doubled floor-line contrast (p95−p5 of 138 against 70)
and raised mid-water contrast 1.7×.

**Levelling.** The frame is rotated about its centre, so the lens centre stays put,
and nothing is cropped. The angle comes from two clicks on a line that's horizontal in
the pool, such as the lane rope. Automatic measurement was tried and rejected: it
locked onto rows of floor cross-lines, which tilt with the camera's sideways aim even
when the camera is level, and got three of four clips wrong.

**A sidecar per clip.** `<video>.json` records the rotation, the conversion, and any
rows to ignore (e.g. the water surface). Background plates, camera-motion measurement
and the swimmer search all read it.

**Memory.** Background plates are built in one preallocated array, with the median
taken a strip of rows at a time. 120 samples of 4K take about 3 GB this way, rather
than ~9 GB, and build in 24–36 s on an 8 GB laptop.

## Tracking

**Background subtraction instead of a trained detector.** Off-the-shelf person
detectors are trained mostly on upright people in air. An underwater swimmer is
small, horizontal, blurred and tinted blue, which is far from that training data.
But the camera is fixed and the swimmer is the main thing moving, so a per-pixel
median over the clip gives a clean "empty pool" plate. Subtracting it finds the
swimmer at native resolution, with no training data.

**Detection, then association.** Every blob that differs from the background is found
in every frame first (one pass over the video). Deciding which blob is the swimmer
comes second, over the whole clip at once, so it can be rerun instantly, for example
after a keyframe is added.

- *Search band.* Rows that move across their whole width, like a swaying lane rope or
  the surface, are excluded. Each row's motion is the *median* change across its
  columns, which a swimmer covering a tenth of the width can't move. A mean, used
  before, let a swimmer on 4K footage push their own rows out of the band.
- *Robust threshold.* A pixel counts as changed if it differs from the plate by more
  than the band's median difference plus 6 median absolute deviations. That tracks
  the footage's own noise level, so ripple and flicker need no hand-tuned threshold.
- *Keyframes.* You mark the swimmer on any frames, typically one where they first
  appear clearly. Tracking runs forwards and backwards from each, so a swimmer who
  isn't visible on frame 1 is still followed from the moment they are. A keyframe can
  also mark them absent, which stops tracking from inventing them there.
- *Kalman filter.* It keeps a best estimate of the swimmer's position and velocity,
  with its uncertainty. Candidates are scored by how surprising they would be under
  that estimate and how well their size matches the swimmer's at the keyframe. While
  no candidate is plausible, the estimate coasts and the search widens with its
  uncertainty, but never past what the swimmer could reach without suddenly going
  much faster. After 1.5 s lost, a pass gives up. Settings are in body lengths, so
  they work at any resolution or distance.
- *Lost, not wrong.* With nothing plausible, the frame is reported lost. Where the
  forward and backward passes chose different blobs, the frame is flagged as a
  conflict and the pass from the nearer keyframe is kept.
- *Leading edge.* Besides the centroid, each blob's 98th-percentile columns give a
  stable front edge in the direction of travel.
- *Pieces put back together.* Near the surface the head and arms often come away
  from the body as a separate blob, and following only the bigger piece puts the
  centre on the hips and legs. Pieces that sit level with the chosen blob, within
  0.2 body lengths of it, no bigger than it, and that keep the whole within 1.15 body
  lengths, are joined to it: the box is their union and the centre their
  area-weighted mean. The body length is the upper quartile of the chosen blobs'
  widths, since split bodies measure short and merges long.
- *Expected size.* The size check is anchored to the keyframe's blob together with
  the first 15 detections after it. A keyframe blob alone can be far off: on one
  clip it was twice the usual size, swollen by bubbles and a second swimmer, and the
  subject then failed the size check a few seconds later.

**Measured against hand labels** (40 frames each on two 4K clips, one keyframe seeded
mid-clip, `evaluate-track`):

| | `josh_front_01` | `josh_back_01` |
|---|---|---|
| Right swimmer, where visible | 84% (32 of 38) | 72% (28 of 39) |
| Centre vs hip, spread along the swim | ±15 px (0.04 body lengths) | ±23 px (0.08) |
| Mean speed vs the labelled hip | −2.1% | −4.6% |

Mid-clip the tracking is solid; the misses are at the ends. At the start, the swimmer
pushes off beside the wall and other swimmers, and a backward pass coasting past
the last detection can latch onto someone resting there. At the end, the swimmer
rises toward the surface, fragments, and on `josh_back_01` merges with a second
swimmer above. One more keyframe near the end of `josh_back_01` recovered its last
1.5 s and halved its speed error to −2.4%.

The remaining speed bias is the centroid itself: it sits ahead of the hip in the
streamlined glide after the push-off (arms out front) and behind it near the
surface, where the head and arms fade, so over a swim it reads 2–5% slow. A hip
landmark from pose estimation removes that. Seeding on a different frame of
`josh_back_01` moved its speed error between −4.6% and −17.7%, so on clips with a
swimmer passing close by, review the ends with `evaluate-track` or by eye.

`josh_front_01` also showed why `--stabilize` exists: the phone settled by about
100 px over the first 1.5 s (measured directly on the floor lines; the stabilizer's
estimate agreed to a few pixels), which in frame coordinates would add ~44 px/s to
the push-off speed. Stabilized, the same one-click run scored worse (−8.2%), but
only because its backward pass latched onto someone at the wall before the swimmer
appeared and the gap filling bridged from there: the start needs a review click.

Known weakness: when another swimmer overlaps the subject in the image, the two merge
into one blob, roughly doubling its area and pulling the centroid toward the other
person. Frames where the blob is both bigger (1.8×) and taller (1.5×) than the
swimmer's lower-quartile size within 3 s are flagged `merged`; size alone flagged
far too many, since kicking and distance swing a lone swimmer's area by half. Pose
landmarks will separate overlapping swimmers properly.

**Review.** `review` shows the clip frame by frame, with a timeline coloured by how
each frame was tracked and position and speed plots with a cursor. "Next problem" jumps
between lost, merged and conflicting stretches. Clicking the swimmer adds a keyframe,
and association reruns on the cached candidates in under a second, so each fix shows
immediately.

**Ground truth.** `label` collects hand clicks on frames spread evenly through a clip
(40 by default): the nose, hip and toes on the side facing the camera, or "not in
view". `evaluate-track` scores a tracking table against them:

- *Which swimmer:* the tracker is on target when its centre is within half a body
  length of the labelled hip. Otherwise the frame is wrong, lost, or (when the
  swimmer was marked out of view) a false alarm. Keyframe frames are left out, since
  they would only score the click that made them.
- *Position and speed:* the position the speed calculation actually uses, gap-filled
  and smoothed, against the hip. A constant offset is harmless for speed; its spread
  isn't. Mean speed is the slope of position against time, so it uses every
  labelled frame rather than just the two ends.
- *Coverage:* whether the box contains the nose and toes, which decides whether a
  crop of it will hold the whole swimmer for pose estimation.

Labels are saved after every frame and a rerun resumes where it stopped. Each frame
opens zoomed on where the labeller's own previous clicks put the swimmer, never on
the tracker's output, so the labels can't drift toward agreeing with it. The file
has MediaPipe's layout, so the same labels, extended with `--preset body`, will
score pose estimation.

**Export.** `export` writes a video of the tracked swimmer in one of two modes:
*spotlight*, the whole frame greyed out except a soft oval around the swimmer, and
*follow*, a virtual camera at a fixed zoom (3 body lengths across) that glides along
with them. Both follow the smoothed, gap-filled path the speed is measured on. The
follow camera's path is smoothed over ±0.3 s, centred in time so it doesn't lag, and
carried straight across frames with no position. The oval's colour shows whether the
position was detected, filled, merged or a keyframe, and a line at the bottom gives
time, status and speed. Frames are piped to ffmpeg as H.264, so the files play in
QuickTime.

## Camera stabilization

Background subtraction assumes the camera doesn't move. A camera drifting by even a
few pixels turns every high-contrast edge in the pool into false motion.

**Two measurements, fused.** Both use phase correlation, which finds the shift
between two images from a single peak in the frequency domain.

- *Frame to frame:* the shift between consecutive frames, measured on a 4×2 grid of
  tiles, taking the median of the tiles that match confidently. This is precise
  from one frame to the next, but moving light (caustics, surface shimmer) pulls
  every match slightly toward the water's motion. Summed over a clip, that bias
  reaches ~10 px on a 10-second cropped clip and ~60 px on a full 1080p frame, with
  no camera motion at all.
- *Frame to plate:* each frame against the background plate. This never
  accumulates error, but it is noisy frame to frame, and unreliable when the plate
  is featureless.

A complementary filter takes the frame-to-frame path and corrects it with a rolling
median of the gap between the two (61 frames). Where that gap scatters by more than
2 px, the frame-to-plate measurement is judged unreliable, so those windows are
bridged from their neighbours. Near the ends of a clip, where the window is cut off,
a robust line replaces the median so the correction keeps up with the drift.

**Leaving steady footage alone.** Water fools phase correlation slightly on every
frame, and applying those tiny corrections would only degrade a steady clip. So a
clip counts as moving only if at least five frames sit more than 4 px from the
median camera position; otherwise stabilization returns the plain plate and zero
offsets. When the camera did move, the plate is rebuilt from realigned frames and
every frame is measured again against it.

**Validation.** Footage was shifted along known camera paths, and the recovered path
compared against the truth:

| Footage | Median error | Worst error |
|---|---|---|
| Cropped clip, known drift | 0.45 px | 2.85 px |
| Full 1080p frame, known drift | 0.40 px | 2.74 px |
| Cropped clip, a (6, 3) px knock for 100 frames | 0.56 px | 2.85 px |
| 40-frame synthetic drift | 0.40 px | 1.02 px |

Steady clips (cropped and full frame) and a textureless synthetic clip came back as
exact no-ops.

One implementation note: given a window, OpenCV's `phaseCorrelate` multiplies both
input arrays by it *in place*. Reused across frames, that silently faded the
background plate and produced 35 px of phantom camera motion on a steady 1080p clip,
so every call now tapers copies instead.

## Calibration

**The lens, modelled exactly.** The new footage was shot on the iPhone 17 Pro's 0.5×
lens, through the phone's flat cover glass, underwater. Light bends at that glass by
Snell's law, more toward the edges of the view, which stretches the picture outward
(pincushion distortion). It was first mistaken for a curved pool floor. Three
measurements settled it:

- Five floor lines bow in proportion to their distance from the image centre
  (bow/offset 0.105 ± 0.003). A curved floor would bow nearer lines *less*.
- The lane rope, above the centre, bows the opposite way: −4.3 px measured, −5.1
  predicted.
- Snell's law with one fitted number, the focal length, straightens all six floor
  lines to within 0.5 px, and the rope too. Fitted separately, two clips gave 837
  and 833 px at 1920 wide, so it's a property of the lens: 1670 px at 4K. The lens
  is read from each raw file's metadata ("back camera 2.22mm"), and the model is
  applied automatically to clips recorded in 4K.

The phone also records the lens centre for every frame; it sits within a few pixels
of the image centre, so the image centre is used.

**The floor lines as the ruler.** Once the lens is undone, the camera is ideal:
distance X along a straight pool line and image position u relate exactly by
X = (p0 + p1 u)/(1 + p2 u), and p2 is ~0 for a square-on camera. This pool's floor
lines are 6 ft from the wall and every 9 ft after (8 lanes of 9 ft in a 75 ft width).
After undistortion they cross the floor evenly (gaps vary by 1.4–2%), consistent
with that layout. Clicking where they cross the floor line the swimmer swam along
gives a ruler at exactly the swimmer's distance. All clicks are fitted at once, so
errors average out, and each mark is checked against a fit of the others. Simulated
clicks on real crossings fit within 0.6–2 cm. Interpolating between the same lines
instead would be off by up to 12 cm near the frame edges, where the lens raises the
scale from 81 to 127 px/m.

**Why not a full floor map.** A homography of the whole floor would assume it's
flat. It isn't: the floor lines converge above the lane rope, which is only
possible if they rise away from the camera. Distance along the swim doesn't need
the floor's depth anyway. A deeper or shallower floor moves its lines up or down in
the picture, hardly sideways, so only the lines' spacing matters.

**Depth from the surface.** The lane rope floats on the flat surface. For a
square-on camera, the swimmer's vertical plane has one scale, the same vertically
and horizontally, so the ruler's metres-per-pixel converts the swimmer's height
below the rope into depth below the surface. The rope also measures any leftover
camera roll, which is taken out of every reading. If the rope isn't at the
swimmer's distance, a nearer stretch of surface appears higher. `--rope-offset`
(how much nearer) with `--camera-depth` corrects for that. Swimming directly beneath
the rope (`--under-rope`, with the floor lines either side clicked) removes the
issue entirely.

**Error budget** (from exact synthetic geometry):

| Source | Effect |
|---|---|
| Swimmer 30 cm off their lane line | ~2% on distances and depth (any single camera) |
| Camera tilted 3° | up to ~0.5% on distances at the frame edges; depth within 1 cm |
| Floor 30 cm deeper under one end | ~0.1% |
| Rope bordering the lane, uncorrected | a few cm of depth |

**Interpolated marks (other footage).** Without a known lens, the calibration records
where marks at known distances appear and interpolates between them, so the lens,
the window and the water are absorbed by measurement. Marks must share the
swimmer's plane: in the older side-on footage, the lane rope sits 4–6 times closer to
the camera than the swimmer, so its marks say nothing about the swimmer. After
clicking, these calibrations report the lens bend, each mark predicted from its
neighbours, likely mis-clicks, and, when markers were only down for part of the
clip, whether the camera moved.

## Metrics

Metrics are chosen so that as many as possible don't depend on the calibration:

| Metric | Definition | Needs |
|---|---|---|
| Kick frequency | Strongest frequency (0.5–8 Hz) of the detrended vertical oscillation, by Lomb–Scargle periodogram on observed frames only | Frame rate only |
| Velocity fluctuation index | (max − min) / mean speed | Nothing: dimensionless, so any constant scale error cancels |
| Speed, distance per kick | Smoothed speed in the direction of travel; mean speed / kick frequency | Calibration, for metres |

**Filling gaps.** Lost frames, and frames flagged `merged` (whose centroid is pulled
toward another swimmer), are filled for gaps of up to 1 s by a Kalman smoother. It
runs forwards, then backwards, so each fill matches the swimmer's position and
velocity at both ends. Filled frames are flagged and count toward speed only: kick
frequency comes from observed frames alone. The Lomb–Scargle periodogram handles
their uneven spacing directly. An FFT on the spliced samples, as before, missed a
simulated 2 Hz kick by a median 0.04 Hz (worst 0.25); Lomb–Scargle by 0.001 Hz.

**What the box can't measure.** On real 4K footage the blob's centroid moves backwards
and forwards within each kick as the legs fold and extend, so its speed swings far
more than the swimmer's (−92 to +584 px/s around a 200 px/s mean). Mean speed and
distance are sound; within-kick velocity (and so the velocity fluctuation index) and a
dependable kick rhythm need pose landmarks or hand-labelled ground truth.

Positions are smoothed with a Savitzky–Golay filter, which keeps the kick extremes
that a moving average would flatten. Speeds are taken in the swimmer's direction of
travel, so "peak speed" means the same thing whichever way they swim.
