import pandas as pd
import pytest

from analysis.association import (
    Keyframe,
    MotionModel,
    SwimmerFilter,
    associate,
    keyframes_path,
    load_keyframes,
    merge_keyframes,
    save_keyframes,
)
from analysis.tracking import detect_boxes

from conftest import moving_square_frames, write_video

FPS = 60.0
LENGTH = 100.0  # the swimmer blob's length in px


def blob(frame, x, y, length=LENGTH, area=2000.0):
    return {"frame": frame, "box_x": x - length / 2, "box_y": y - 10, "box_w": length,
            "box_h": 20, "centroid_x": x, "centroid_y": y, "area": area,
            "edge_left": x - length / 2, "edge_right": x + length / 2}


def swimmer_path(frames, start=500.0, speed=2.0, y=300.0):
    """A swimmer gliding right at `speed` px/frame (1.2 body lengths/s here)."""
    return {f: (start + speed * (f - 1), y) for f in frames}


def table(rows):
    return pd.DataFrame(rows)


def positions(result):
    found = result[result.found]
    return dict(zip(found.frame, zip(found.centroid_x, found.centroid_y)))


# ---------- keyframes on disk ----------

def test_keyframes_round_trip(tmp_path):
    path = str(tmp_path / "k" / "clip.json")
    save_keyframes(path, [Keyframe(90, 2954.0, 1162.0), Keyframe(10)], video="clip.mp4")
    loaded = load_keyframes(path)
    assert [k.frame for k in loaded] == [10, 90]
    assert loaded[0].absent and not loaded[1].absent
    assert (loaded[1].x, loaded[1].y) == (2954.0, 1162.0)


def test_missing_keyframes_file_means_none(tmp_path):
    assert load_keyframes(str(tmp_path / "nope.json")) == []


def test_new_keyframes_replace_old_ones_on_the_same_frame():
    merged = merge_keyframes([Keyframe(5, 1, 1), Keyframe(9, 2, 2)], [Keyframe(9, 3, 3)])
    assert [(k.frame, k.x) for k in merged] == [(5, 1), (9, 3)]


def test_default_keyframes_file_is_named_after_the_clip():
    assert keyframes_path("data/normalized/josh_back_01.mp4") == "data/keyframes/josh_back_01.json"


# ---------- the motion model ----------

def test_the_search_envelope_grows_while_the_swimmer_is_lost():
    f = SwimmerFilter(0.0, 0.0, LENGTH, FPS, MotionModel())
    f.update((0.0, 0.0))
    point = (40.0, 0.0)
    sizes = []
    for _ in range(5):
        f.predict()
        sizes.append(f.distance2(point))
        f.coasting += 1
    assert all(a > b for a, b in zip(sizes, sizes[1:])), "the same offset should look ever less surprising"


def test_reach_is_capped_by_speed():
    f = SwimmerFilter(0.0, 0.0, LENGTH, FPS, MotionModel())
    f.seen, f.anchor_speed = True, 2.0  # 2 px/frame
    assert f.reachable((20.0, 0.0))
    assert not f.reachable((200.0, 0.0)), "100 px a frame is not a swimmer"


# ---------- association ----------

def test_one_keyframe_mid_clip_covers_both_directions():
    path = swimmer_path(range(1, 101))
    rows = [blob(f, *p) for f, p in path.items()]
    rows += [blob(f, 1500.0, 300.0) for f in range(1, 101)]  # a look-alike far away
    result = associate(table(rows), 100, FPS, [Keyframe(50, *path[50])])
    got = positions(result)
    assert len(got) == 100
    assert all(abs(got[f][0] - path[f][0]) < 1 for f in got), "should never switch to the look-alike"
    assert result.set_index("frame").loc[50, "source"] == "keyframe"


def test_a_swimmer_invisible_at_the_start_is_tracked_from_where_they_appear():
    path = swimmer_path(range(30, 101))
    rows = [blob(f, *p) for f, p in path.items()]
    result = associate(table(rows), 100, FPS, [Keyframe(60, *path[60])])
    found = result[result.found].frame
    assert found.min() == 30 and found.max() == 100


def test_lost_rather_than_wrong():
    """The swimmer vanishes for 10 frames while a same-sized blob shows up 3 body
    lengths ahead: those frames must be reported lost, not given to the decoy."""
    path = swimmer_path(range(1, 61))
    rows = [blob(f, *p) for f, p in path.items() if not 30 <= f < 40]
    rows += [blob(f, path[f][0] + 3 * LENGTH, 300.0) for f in range(30, 40)]
    result = associate(table(rows), 60, FPS, [Keyframe(1, *path[1])]).set_index("frame")
    assert not result.loc[30:39, "found"].any()
    assert result.loc[40:60, "found"].all(), "and picked up again where they reappear"


def test_a_pass_gives_up_after_coasting_too_long():
    path = swimmer_path(range(1, 301))
    rows = [blob(f, *p) for f, p in path.items() if f < 50 or f > 200]
    result = associate(table(rows), 300, FPS, [Keyframe(1, *path[1])],
                       model=MotionModel(max_coast=1.0))
    assert not result.set_index("frame").loc[201:, "found"].any(), "2.5 s lost: stop guessing"


def test_absent_keyframe_stops_tracking_there():
    path = swimmer_path(range(1, 41))
    rows = [blob(f, *p) for f, p in path.items()]
    result = associate(table(rows), 40, FPS, [Keyframe(1, *path[1]), Keyframe(20)]).set_index("frame")
    assert not result.loc[20, "found"]
    assert result.loc[1:19, "found"].all()
    assert not result.loc[21:, "found"].any(), "no keyframe after the absence to track from"


def test_disagreeing_passes_are_flagged_and_the_nearer_keyframe_wins():
    """Two look-alikes side by side; the keyframes say A at frame 1 and B at 40."""
    a = swimmer_path(range(1, 41), start=500)
    b = swimmer_path(range(1, 41), start=500 + 2 * LENGTH)
    rows = [blob(f, *a[f]) for f in a] + [blob(f, *b[f]) for f in b]
    result = associate(table(rows), 40, FPS, [Keyframe(1, *a[1]), Keyframe(40, *b[40])]).set_index("frame")
    assert result.loc[2:39, "conflict"].all()
    assert result.loc[10, "centroid_x"] == pytest.approx(a[10][0]), "frame 10 is nearer frame 1"
    assert result.loc[35, "centroid_x"] == pytest.approx(b[35][0]), "frame 35 is nearer frame 40"


def test_a_much_bigger_blob_is_someone_else():
    path = swimmer_path(range(1, 31))
    rows = [blob(f, *p) for f, p in path.items() if f != 15]
    rows.append(blob(15, *path[15], area=20000.0))  # 10x the swimmer, right where they should be
    result = associate(table(rows), 30, FPS, [Keyframe(1, *path[1])]).set_index("frame")
    assert not result.loc[15, "found"]


def test_a_bigger_and_taller_blob_is_flagged_as_merged():
    path = swimmer_path(range(1, 31))
    rows = [blob(f, *p) for f, p in path.items()]
    rows[19].update(area=4000.0, box_h=40)  # frame 20: twice the size, twice as tall
    rows[24].update(area=4000.0)            # frame 25: bigger only (e.g. arms out, bubbles)
    result = associate(table(rows), 30, FPS, [Keyframe(1, *path[1])]).set_index("frame")
    assert result.loc[20, "size_ratio"] == pytest.approx(2.0)
    assert result.loc[20, "merged"]
    assert not result.loc[25, "merged"], "size alone isn't enough"
    assert not result.loc[19, "merged"]


def test_a_click_where_detection_missed_the_swimmer_is_kept_and_tracked_from():
    path = swimmer_path(range(1, 21))
    rows = [blob(f, *p) for f, p in path.items() if f != 10]  # not detected on frame 10
    click = (path[10][0] + 3, path[10][1] - 2)
    result = associate(table(rows), 20, FPS, [Keyframe(10, *click)],
                       default_length=LENGTH).set_index("frame")
    assert result.loc[10, "found"] and result.loc[10, "source"] == "keyframe"
    assert (result.loc[10, "centroid_x"], result.loc[10, "centroid_y"]) == click
    assert pd.isna(result.loc[10, "box_x"])
    assert result.drop(index=10)["found"].all(), "both directions carry on from the click"


def test_without_keyframes_the_largest_blob_on_the_first_frame_is_followed():
    small = swimmer_path(range(1, 21), start=500)
    big = swimmer_path(range(1, 21), start=900)
    rows = [blob(f, *small[f], area=1000.0) for f in small] + [blob(f, *big[f], area=3000.0) for f in big]
    got = positions(associate(table(rows), 20, FPS))
    assert all(abs(got[f][0] - big[f][0]) < 1 for f in got)


# ---------- through detect_boxes, on video ----------

def test_detect_boxes_tracks_backwards_from_a_later_keyframe(tmp_path):
    frames = moving_square_frames(count=40, step=3)
    for frame in frames[:10]:
        frame[:] = 200  # the target isn't there yet
    path = write_video(tmp_path / "late.mp4", frames)
    x20 = 40 + 3 * 19 + 12  # the square's centre on frame 20
    result = detect_boxes(path, keyframes=[Keyframe(20, x20, 92)], progress=False)
    found = result[result.found].frame
    assert found.min() <= 12 and found.max() == 40


# ---------- the track command ----------

from analysis.association import load_keyframes as _load  # noqa: E402
from analysis.cli import main  # noqa: E402
from analysis.tracking import BOX_COLUMNS  # noqa: E402


def test_track_saves_and_accumulates_keyframes(tmp_path, moving_square_video):
    keys = str(tmp_path / "keys.json")
    out = str(tmp_path / "boxes.csv")
    main(["track", moving_square_video, out, "--no-auto-roi", "--keyframes", keys,
          "--seed-at", "20", "121", "92"])
    main(["track", moving_square_video, out, "--no-auto-roi", "--keyframes", keys,
          "--absent", "5"])
    saved = _load(keys)
    assert [(k.frame, k.absent) for k in saved] == [(5, True), (20, False)]
    boxes = pd.read_csv(out)
    assert list(boxes.columns) == BOX_COLUMNS
    assert not boxes.set_index("frame").loc[5, "found"]


# ---------- pieces of one swimmer ----------

def split_swimmer(frames=30, split_from=21, extra=None):
    """Whole for the first frames (one blob 100 px long); from `split_from` on, the
    head and arms come away as a small blob just in front of the body."""
    rows = []
    for f in range(1, frames + 1):
        x = 500.0 + 2.0 * (f - 1)
        if f < split_from:
            rows.append(blob(f, x, 300.0, length=100.0, area=1700.0))
        else:
            rows.append(blob(f, x + 5, 300.0, length=70.0, area=1400.0))   # body and legs
            rows.append(blob(f, x - 40, 300.0, length=20.0, area=300.0))   # head and arms
            if extra:
                rows.append(extra(f, x))
    return rows


def test_a_swimmer_split_in_two_is_put_back_together():
    result = associate(table(split_swimmer()), 30, FPS, [Keyframe(1, 500.0, 300.0)]).set_index("frame")
    x = 500.0 + 2.0 * 24
    assert result.loc[25, "centroid_x"] == pytest.approx(x + (1400 * 5 - 300 * 40) / 1700), \
        "the area-weighted centre of both pieces, not the bigger piece's"
    assert result.loc[25, "box_x"] == pytest.approx(x - 50) and result.loc[25, "box_w"] == pytest.approx(90)
    assert result.loc[25, "area"] == 1700


def test_a_neighbouring_swimmer_is_not_joined_on():
    neighbour = lambda f, x: blob(f, x - 130, 300.0, length=70.0, area=1400.0)  # noqa: E731
    result = associate(table(split_swimmer(extra=neighbour)), 30, FPS,
                       [Keyframe(1, 500.0, 300.0)]).set_index("frame")
    assert result.loc[25, "box_w"] == pytest.approx(90), "too far away to be part of the swimmer"


def test_a_piece_above_or_below_the_body_is_not_joined():
    bubbles = lambda f, x: blob(f, x + 5, 240.0, length=30.0, area=200.0)  # noqa: E731
    result = associate(table(split_swimmer(extra=bubbles)), 30, FPS,
                       [Keyframe(1, 500.0, 300.0)]).set_index("frame")
    assert result.loc[25, "box_y"] == pytest.approx(290.0), "the box doesn't reach up to it"


def test_the_expected_size_comes_from_the_keyframe_and_the_detections_after_it():
    """The keyframe's blob is twice the swimmer's usual size (bubbles stuck to it).
    Anchored to that alone, a swimmer at a third of it would fail the size check
    later; anchored to the first detections too, they don't."""
    path = swimmer_path(range(1, 61))
    rows = []
    for f, (x, y) in path.items():
        area = 4000.0 if f == 1 else (2000.0 if f < 40 else 1250.0)
        rows.append(blob(f, x, y, area=area))
    result = associate(table(rows), 60, FPS, [Keyframe(1, *path[1])]).set_index("frame")
    assert result.loc[40:, "found"].all()
