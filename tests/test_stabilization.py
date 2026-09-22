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
