import cv2
import numpy as np
import pytest

from analysis.tracking import (
    _blank_border,
    detect_boxes,
    estimate_translation,
    shift_image,
)

from conftest import write_video


def textured(height=200, width=320, seed=0):
    """Random texture, blurred. Phase correlation needs something to lock onto;
    a flat image has no peak to find."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (height, width), dtype=np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


# ---------- shift estimation ----------

@pytest.mark.parametrize("dx,dy", [(12, 7), (-9, 4), (0, -15), (3, 0)])
def test_estimate_translation_recovers_a_known_shift(dx, dy):
    reference = textured()
    moved = shift_image(reference, dx, dy)

    est_dx, est_dy, response = estimate_translation(moved, reference)
    assert est_dx == pytest.approx(dx, abs=0.5)
    assert est_dy == pytest.approx(dy, abs=0.5)
    assert response > 0.1


def test_sign_convention_maps_reference_onto_frame():
    """phaseCorrelate(a, b) gives the shift taking a to b -- so feeding
    (reference, frame) yields the offset to apply to the reference."""
    reference = textured()
    frame = shift_image(reference, 10, -6)

    dx, dy, _ = estimate_translation(frame, reference)
    realigned = shift_image(reference, dx, dy)

    # after applying the estimate to the reference, it should match the frame
    before = cv2.absdiff(frame, reference).mean()
    after = cv2.absdiff(frame, realigned).mean()
    assert after < before / 2, f"realignment made it worse: {before:.1f} -> {after:.1f}"


def test_estimate_translation_is_subpixel():
    reference = textured()
    moved = shift_image(reference, 4.5, -2.25)
    dx, dy, _ = estimate_translation(moved, reference)
    assert dx == pytest.approx(4.5, abs=0.3)
    assert dy == pytest.approx(-2.25, abs=0.3)


# ---------- border handling ----------

def test_blank_border_zeroes_the_side_the_shift_came_from():
    diff = np.full((100, 100), 255, dtype=np.uint8)
    _blank_border(diff, dx=5, dy=0)
    assert diff[:, :5].sum() == 0, "left strip should be blanked for a positive dx"
    assert diff[:, 10:].min() == 255, "interior must be untouched"


def test_blank_border_handles_every_sign_combination():
    for dx, dy in [(6, 4), (-6, 4), (6, -4), (-6, -4)]:
        diff = np.full((80, 80), 255, dtype=np.uint8)
        _blank_border(diff, dx, dy)
        blanked = (diff == 0).sum()
        assert blanked > 0, f"nothing blanked for dx={dx} dy={dy}"
        assert diff[40, 40] == 255, "centre must survive"


def test_blank_border_survives_a_shift_larger_than_the_image():
    diff = np.full((20, 20), 255, dtype=np.uint8)
    _blank_border(diff, dx=999, dy=-999)  # must not raise or wrap
    assert diff.sum() == 0


# ---------- integration ----------

def drifting_video(tmp_path, drift_per_frame=1.5, count=40):
    """A moving target over a textured background, with the whole frame
    translating -- i.e. a swimmer plus a camera that won't sit still."""
    base = cv2.cvtColor(textured(200, 320), cv2.COLOR_GRAY2BGR)
    frames = []
    for i in range(count):
        frame = base.copy()
        frame[90:110, 40 + i * 5:60 + i * 5] = 0  # the "swimmer"
        frames.append(shift_image(frame, i * drift_per_frame, i * drift_per_frame * 0.4))
    return write_video(tmp_path / "drift.mp4", frames)


def test_stabilized_tracking_follows_the_target_on_drifting_footage(tmp_path):
    path = drifting_video(tmp_path)
    table = detect_boxes(path, stabilize=True, seed_point=(50, 100), progress=False)

    found = table[table["found"]]
    assert len(found) >= 20, "stabilized tracking should hold the target"
    # target moves +5px/frame and the camera drags it a further +1.5px/frame
    assert found["centroid_x"].is_monotonic_increasing


def test_stabilize_does_not_disturb_a_static_camera(moving_square_video):
    """On footage that never moves, enabling stabilization should change
    essentially nothing -- the estimated shift is ~0."""
    plain = detect_boxes(moving_square_video, progress=False)
    stabilized = detect_boxes(moving_square_video, stabilize=True, progress=False)

    assert stabilized["found"].sum() >= plain["found"].sum() - 2
    both = plain[plain["found"]].merge(
        stabilized[stabilized["found"]], on="frame", suffixes=("_p", "_s")
    )
    assert len(both) > 10
    assert (both["centroid_x_p"] - both["centroid_x_s"]).abs().max() < 3.0


def test_low_response_estimates_are_ignored(tmp_path):
    """A flat, textureless scene gives phase correlation nothing to lock onto;
    the resulting garbage shift must be rejected rather than applied."""
    flat = np.full((160, 240, 3), 128, dtype=np.uint8)
    frames = []
    for i in range(20):
        frame = flat.copy()
        frame[70:90, 30 + i * 6:50 + i * 6] = 0
        frames.append(frame)
    path = write_video(tmp_path / "flat.mp4", frames)

    # min_response=1.1 is unreachable, so every estimate must be discarded and
    # the result must match the unstabilized path exactly
    gated = detect_boxes(path, stabilize=True, min_response=1.1, progress=False)
    plain = detect_boxes(path, progress=False)
    assert gated["found"].tolist() == plain["found"].tolist()


# ---------- drift must not leak into measured motion ----------

def test_camera_offset_is_recorded_every_frame(tmp_path):
    path = drifting_video(tmp_path)
    table = detect_boxes(path, stabilize=True, seed_point=(50, 100), progress=False)
    # recorded on lost frames too -- the camera moved regardless
    assert table["cam_dx"].notna().mean() > 0.8
    # camera drifts +1.5 px/frame; the plate is the median, so offsets are
    # relative to mid-clip, but their slope must be the drift rate
    frames = table["frame"].to_numpy(float)
    ok = table["cam_dx"].notna().to_numpy()
    slope = np.polyfit(frames[ok], table["cam_dx"].to_numpy(float)[ok], 1)[0]
    assert slope == pytest.approx(1.5, abs=0.25)


def test_camera_drift_is_removed_from_measured_speed(tmp_path):
    from analysis.analysis import kinematics

    path = drifting_video(tmp_path)  # scene motion +5 px/frame, camera +1.5 px/frame
    table = detect_boxes(path, stabilize=True, seed_point=(50, 100), progress=False)
    kin = kinematics(table, fps=30.0)
    found = table["found"].to_numpy()

    frame_speed = np.nanmedian(np.diff(table["centroid_x"].to_numpy(float)))
    pool_speed = np.nanmedian(kin["speed_px_s"].to_numpy(float)[found]) / 30.0

    assert frame_speed == pytest.approx(6.5, abs=0.6), "raw detections include the camera"
    assert pool_speed == pytest.approx(5.0, abs=0.6), "derived speed must not"


def test_without_stabilization_the_camera_is_taken_as_fixed(moving_square_video):
    table = detect_boxes(moving_square_video, progress=False)
    assert (table["cam_dx"] == 0).all() and (table["cam_dy"] == 0).all()


# ---------- deciding whether the camera moved at all ----------

from analysis.tracking import camera_moved, fuse_camera_path, stabilized_background  # noqa: E402


def test_ripple_level_noise_is_not_camera_motion():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.6, (600, 2))  # still camera: ~0.6px median, like real footage
    assert not camera_moved(noise)


def test_a_single_spike_does_not_switch_stabilization_on():
    offsets = np.zeros((600, 2))
    offsets[300] = (9.0, 0.0)
    assert not camera_moved(offsets)


def test_a_short_real_bump_does():
    offsets = np.zeros((600, 2))
    offsets[300:310] = (9.0, 0.0)
    assert camera_moved(offsets)


def test_steady_drift_does():
    ramp = np.stack([np.linspace(0, 40, 600), np.zeros(600)], axis=1)
    assert camera_moved(ramp)


# ---------- the complementary filter ----------

def test_fusion_removes_the_relative_paths_accumulated_drift():
    n = 300
    truth = np.stack([np.linspace(0, 30, n), np.zeros(n)], axis=1)
    relative = truth + np.stack([np.linspace(0, 8, n), np.zeros(n)], axis=1)  # wanders 8px
    rng = np.random.default_rng(1)
    absolute = truth + rng.normal(0, 0.5, (n, 2))  # anchored but noisy
    fused = fuse_camera_path(relative, absolute)
    assert np.abs(fused - truth)[30:-30].max() < 1.0


def test_fusion_ignores_an_absolute_measurement_that_scatters():
    # a featureless plate gives garbage "absolute" readings; they must not be
    # allowed to drag the path around (ungated, this was off by 2500px)
    n = 300
    relative = np.stack([np.linspace(0, 30, n), np.zeros(n)], axis=1)
    rng = np.random.default_rng(2)
    garbage = rng.uniform(-500, 500, (n, 2))
    fused = fuse_camera_path(relative, garbage)
    assert np.allclose(fused - fused[0], relative - relative[0], atol=1e-9)


def test_stabilizing_a_still_textureless_clip_is_an_exact_no_op(moving_square_video):
    plate, offsets = stabilized_background(moving_square_video, max_samples=40)
    assert (offsets == 0).all()
    plain = detect_boxes(moving_square_video, progress=False)
    stabilized = detect_boxes(moving_square_video, stabilize=True, progress=False)
    assert plain["found"].tolist() == stabilized["found"].tolist()


def test_passing_a_plate_with_stabilize_is_rejected(moving_square_video):
    from analysis.frames import median_background

    with pytest.raises(ValueError, match="background=None"):
        detect_boxes(moving_square_video, stabilize=True,
                     background=median_background(moving_square_video), progress=False)


# ---------- calibration reference from part of a drifting clip ----------

from analysis.tracking import calibration_reference, plate_offset  # noqa: E402


def test_marker_frames_land_in_the_tracking_plates_coordinates(tmp_path):
    """Markers down for frames 1-10 of a drifting clip: the reference built from
    just those frames has to line up with the plate tracking measures against,
    not with where the camera happened to be while the markers were down."""
    world = cv2.cvtColor(textured(200, 320), cv2.COLOR_GRAY2BGR)
    frames = []
    for i in range(40):
        scene = world.copy()
        if i < 10:
            scene[90:110, 150:170] = 0  # a marker on the pool floor
        frames.append(shift_image(scene, i * 1.5, i * 0.6))
    path = write_video(tmp_path / "markers_drift.mp4", frames)

    reference, plate = calibration_reference(path, frames=(1, 10), stabilize=True,
                                             max_samples=40)
    # Tracking maps a frame position to the plate as position - offset. A click
    # on the marker has to give the same answer tracking would for an object
    # sitting there -- whatever the offsets' own accuracy.
    _, offsets = stabilized_background(path, max_samples=40)
    in_frames = np.array([(159.5 + i * 1.5, 99.5 + i * 0.6) for i in range(10)])
    expected = np.median(in_frames - offsets[:10], axis=0)

    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray[40:180, 100:260] < 40)
    assert xs.mean() + 100 == pytest.approx(expected[0], abs=1.0)
    assert ys.mean() + 40 == pytest.approx(expected[1], abs=1.0)

    # and without the alignment the marker frames would sit far off the plate:
    # the camera was up to ~30px from its median position while they were shot
    assert np.abs(offsets[:10]).max() > 15
    shift = plate_offset(reference, plate)
    assert shift is not None and np.hypot(*shift) < 2.0, "calibrate would warn"


# ---------- phase correlation must not touch its inputs ----------

from analysis.tracking import (  # noqa: E402
    correlation_window,
    estimate_translation_consensus,
    measure_camera_motion,
)


def test_phase_correlation_leaves_its_inputs_alone():
    """OpenCV multiplies both inputs by the window in place. Every caller here
    reuses its images -- the plate against every frame -- so the side effect
    compounded into a faded plate and 35px of invented motion on a steady clip."""
    reference = textured().astype(np.float32)
    frame = shift_image(reference, 4, -3)
    before = (reference.copy(), frame.copy())

    estimate_translation(frame, reference, window=correlation_window(reference.shape))
    estimate_translation_consensus(frame, reference)

    assert np.array_equal(reference, before[0])
    assert np.array_equal(frame, before[1])


def test_measuring_camera_motion_leaves_the_plate_alone(moving_square_video):
    plate = cv2.cvtColor(textured(160, 320), cv2.COLOR_GRAY2BGR)
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    before = plate_gray.copy()
    measure_camera_motion(moving_square_video, plate_gray)
    assert np.array_equal(plate_gray, before)


# ---------- fusion at the ends of a clip ----------

def test_fusion_follows_a_drifting_relative_path_right_to_the_ends():
    """Moving light pulls every frame-to-frame match slightly, so the relative
    path drifts even on a still camera. A centred median can't see past the
    clip's ends and lags that drift there -- far enough, at 0.3px/frame, to
    make a still camera look like it moved."""
    rng = np.random.default_rng(0)
    frames = 300
    relative = np.outer(np.arange(frames), [0.3, -0.2])       # bias, no real motion
    absolute = rng.normal(0.0, 0.5, (frames, 2))               # honest but noisy

    fused = fuse_camera_path(relative, absolute)

    error = np.hypot(fused[:, 0], fused[:, 1])
    assert error[:30].max() < 1.5 and error[-30:].max() < 1.5, "the ends lag the drift"
    assert error.max() < 1.5
    assert not camera_moved(fused)


def test_fusion_on_a_clip_shorter_than_its_window():
    """Every frame of a 40-frame clip is within half a window of an end."""
    rng = np.random.default_rng(1)
    truth = np.outer(np.arange(40), [1.5, 0.6])
    relative = truth + np.outer(np.arange(40), [0.1, 0.1])    # slightly biased steps
    absolute = truth + rng.normal(0.0, 0.5, (40, 2))

    fused = fuse_camera_path(relative, absolute)

    assert np.abs(fused - truth).max() < 1.5
