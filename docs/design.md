# Design notes

Why the pipeline works the way it does, stage by stage. The code's docstrings go
into more detail; this is the overview.

## Tracking

**Background subtraction instead of a trained detector.** Off-the-shelf person
detectors are trained mostly on upright people in air. An underwater swimmer is
small, horizontal, blurred and tinted blue, which is far from that training data.
But the camera is fixed and the swimmer is the main thing moving, so a per-pixel
median over the clip gives a clean "empty pool" plate. Subtracting it finds the
swimmer at native resolution, with no training data.

**Picking the swimmer out of everything else that moves:**

- *Search band.* Each image row's motion is measured across the clip. Rows that move
  constantly, like a swaying lane rope, are excluded, and the tracker searches the
  largest band that remains.
- *Robust threshold.* A pixel counts as changed if it differs from the plate by
  more than the band's median difference plus 6 median absolute deviations. That
  tracks the footage's own noise level, so ripple and flicker don't need a
  hand-tuned threshold.
- *Identity gates.* A candidate must be near where the swimmer was predicted to be
  (60 px per frame by default) and between a third and three times the size of the
  swimmer's first detections. When nothing passes, the frame is reported lost
  rather than matched to the wrong object.
- *Leading edge.* Besides the centroid, the tracker records the 98th-percentile
  columns of each blob, which give a stable front edge in the direction of travel.

Known weakness: while the swimmer is lost, the prediction slows and the search
radius grows (up to 240 px), which can pick up another person. The swimmer can also
only be selected on frame 1.

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

**Measured, not modelled.** An underwater housing's flat port refracts light,
magnifying the image by roughly 1.33 and adding radial distortion. An in-air lens
calibration would get both wrong. Instead, the calibration records where marks at
known distances actually appear in the footage and interpolates between them, so
the lens, the port and the water are all accounted for by measurement.

**Marks must share the swimmer's plane.** The scale of a mark depends on its
distance from the camera. In the test footage, the lane rope sits 4–6 times closer
to the camera than the swimmer, so distances read off rope floats say nothing about
the swimmer. The calibration therefore uses markers placed at measured distances
along the floor of the swimmer's own lane.

**Model.** Each reference line carries marks of the form (image x, image y,
distance). Distance is interpolated linearly between marks along a line, and blended
by image height between lines. Beyond the outermost lines, the nearest line's value
is held; outside the marked range, the result is undefined rather than extrapolated.

**Self-checks.** After clicking, the calibration reports:

- *Lens bend:* how far a straight ruler through the two end marks would be from the
  marks in between.
- *Error between marks:* each inner mark predicted from its two neighbours, which is
  the interpolation error at twice the actual spacing.
- *Likely mis-clicks:* marks that spike away from their neighbours. On a simulated
  lens whose scale changes about 20% across the frame, this caught every inner mark
  clicked 30% of a marker spacing off, at spacings of 1–2.5 m, with at most 1% false
  alarms.
- *Camera position:* when markers were only down for part of the clip, whether the
  camera sat in the same place then as during the rest of the clip.

## Metrics

Metrics are chosen so that as many as possible don't depend on the calibration:

| Metric | Definition | Needs |
|---|---|---|
| Kick frequency | Strongest frequency (0.5–8 Hz) of the detrended vertical oscillation | Frame rate only |
| Velocity fluctuation index | (max − min) / mean speed | Nothing: dimensionless, so any constant scale error cancels |
| Speed, distance per kick | Smoothed speed in the direction of travel; mean speed / kick frequency | Calibration, for metres |

Positions are smoothed with a Savitzky–Golay filter, which keeps the kick extremes
that a moving average would flatten. Speeds are taken in the swimmer's direction of
travel, so "peak speed" means the same thing whichever way they swim.
