import pytest

from analysis.landmarks import (
    LANDMARK_NAMES,
    NUM_LANDMARKS,
    POSE_CONNECTIONS,
    detect_column_style,
    display_name,
    landmark_columns,
    load_index_to_name_map,
    named_header,
    numbered_header,
)


def test_landmark_set_is_the_mediapipe_33():
    assert NUM_LANDMARKS == 33
    assert len(LANDMARK_NAMES) == 33
    assert len(set(LANDMARK_NAMES)) == 33, "landmark names must be unique"


def test_pose_connections_reference_real_landmarks():
    for a, b in POSE_CONNECTIONS:
        assert 0 <= a < NUM_LANDMARKS
        assert 0 <= b < NUM_LANDMARKS


def test_named_header_shape():
    header = named_header()
    assert header[0] == "frame"
    assert len(header) == 1 + NUM_LANDMARKS * 4
    assert "left_shoulder_x" in header
    assert "right_foot_index_visibility" in header


def test_numbered_header_shape():
    header = numbered_header()
    assert header[0] == "frame"
    assert len(header) == 1 + NUM_LANDMARKS * 4
    assert "11_x" in header


def test_detect_column_style_round_trips_both_headers():
    assert detect_column_style(named_header()) == "named"
    assert detect_column_style(numbered_header()) == "numbered"


def test_detect_column_style_rejects_unrecognized_columns():
    with pytest.raises(ValueError, match="Could not detect column style"):
        detect_column_style(["frame", "something_else"])


def test_landmark_columns_named_and_numbered():
    assert landmark_columns(11, "named") == (
        "left_shoulder_x",
        "left_shoulder_y",
        "left_shoulder_z",
        "left_shoulder_visibility",
    )
    assert landmark_columns(11, "numbered") == ("11_x", "11_y", "11_z", "11_visibility")


def test_landmark_columns_rejects_unknown_style():
    with pytest.raises(ValueError, match="Unknown column style"):
        landmark_columns(0, "bogus")


def test_landmark_columns_agree_with_headers():
    header = named_header()
    for i in range(NUM_LANDMARKS):
        for col in landmark_columns(i, "named"):
            assert col in header


def test_default_index_to_name_map_is_the_builtin_order():
    mapping = load_index_to_name_map(None)
    assert mapping[0] == "nose"
    assert mapping[11] == "left_shoulder"
    assert len(mapping) == NUM_LANDMARKS


def test_index_to_name_map_from_file_skips_header_and_blank_rows(tmp_path):
    path = tmp_path / "mapping.csv"
    path.write_text("index,name\n0,snoot\n\n11,shoulder_L\nnot_an_int,junk\n")
    mapping = load_index_to_name_map(str(path))
    assert mapping == {0: "snoot", 11: "shoulder_L"}


def test_display_name_prefers_builtin_for_named_style():
    mapping = {11: "custom"}
    assert display_name(11, "named", mapping) == "left_shoulder"
    assert display_name(11, "numbered", mapping) == "custom"


def test_display_name_falls_back_to_index_when_unmapped():
    assert display_name(7, "numbered", {}) == "7"
