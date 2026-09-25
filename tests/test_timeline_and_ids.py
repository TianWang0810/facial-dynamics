"""
Tests for timestamp policy and clip identity.

Synthetic timelines only -- analyze_timeline() and tracker_timestamps_ms() are
pure functions over a PTS array, so no video is needed.

Usage:
    python tests/test_timeline_and_ids.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.common.ids import clip_id, infer_grouping
from src.observation.decode import FrameTimingIssue, analyze_timeline, tracker_timestamps_ms


def _codes(timeline):
    return {issue["code"] for issue in timeline["issues"]}


def test_missing_pts_is_kept_in_place_not_dropped():
    """The silent-corruption case: dropping a frame shifts every later index."""
    pts = [0.00, 0.04, float("nan"), 0.12, 0.16]
    timeline = analyze_timeline(pts, 25.0)

    assert timeline["n_frames_decoded"] == 5, "the frame must still be counted"
    assert len(timeline["pts_sec"]) == 5, "the array must keep its length and positions"
    assert timeline["pts_sec"][2] is None, "the missing timestamp is null, not filled in"
    assert timeline["pts_sec"][3] == 0.12, "frame 3 must still be frame 3"
    assert timeline["n_missing_pts"] == 1
    assert FrameTimingIssue.MISSING_PTS in _codes(timeline)
    assert not timeline["is_strictly_monotonic"], "a clip with a missing timestamp is not clean"


def test_backward_and_duplicate_timestamps_are_reported_not_repaired():
    backward = analyze_timeline([0.0, 0.08, 0.04, 0.12], 25.0)
    assert FrameTimingIssue.NON_MONOTONIC in _codes(backward)
    assert backward["pts_sec"][2] == 0.04, "the decoded value must survive; repairing would hide the damage"
    assert not backward["is_strictly_monotonic"]

    duplicate = analyze_timeline([0.0, 0.04, 0.04, 0.08], 25.0)
    assert FrameTimingIssue.DUPLICATE_PTS in _codes(duplicate)
    assert not duplicate["is_strictly_monotonic"]

    clean = analyze_timeline([0.0, 0.04, 0.08, 0.12], 25.0)
    assert clean["issues"] == []
    assert clean["is_strictly_monotonic"]
    assert abs(clean["fps_from_pts"] - 25.0) < 1e-6


def test_gap_before_the_final_frame_has_its_own_code():
    tail = analyze_timeline([0.0, 0.04, 0.08, 0.12, 0.24], 25.0)
    assert _codes(tail) == {FrameTimingIssue.TAIL_GAP}, _codes(tail)
    assert tail["large_gaps"] == [{"after_frame": 3, "gap_sec": tail["large_gaps"][0]["gap_sec"]}]
    both = analyze_timeline([0.0, 0.04, 0.30, 0.34, 0.50], 25.0)
    assert set(_codes(both)) == {FrameTimingIssue.LARGE_GAP, FrameTimingIssue.TAIL_GAP}


def test_large_gap_is_located_not_just_counted():
    pts = [0.0, 0.04, 0.08, 0.60, 0.64]
    timeline = analyze_timeline(pts, 25.0)
    assert FrameTimingIssue.LARGE_GAP in _codes(timeline)
    assert timeline["large_gaps"][0]["after_frame"] == 2, "the gap must be located, not merely flagged"
    assert abs(timeline["large_gaps"][0]["gap_sec"] - 0.52) < 1e-9


def test_tracker_clock_is_monotonic_without_overwriting_pts():
    """MediaPipe needs strictly increasing ms; the data keeps its real timestamps."""
    pts = [0.0, 0.0005, 0.001, float("nan"), 0.002]
    clock = tracker_timestamps_ms(pts)

    assert np.all(np.diff(clock) > 0), f"tracker clock must strictly increase: {clock}"
    assert clock[0] == 0
    # 0.0005s and 0.001s both round to a millisecond that would collide; the
    # clock is nudged, while pts stays whatever the container said.
    assert clock[1] == 1 and clock[2] == 2
    assert clock[3] == 3, "a frame with no PTS still advances the tracker by one"
    assert list(pts) == [0.0, 0.0005, 0.001, pts[3], 0.002] or np.isnan(pts[3])

    ordinary = tracker_timestamps_ms([0.0, 0.04, 0.08])
    assert list(ordinary) == [0, 40, 80], "a clean timeline should map straight to milliseconds"


def test_same_basename_in_different_directories_does_not_collide():
    """The bug: recursive glob plus basename keys merged two different clips."""
    root = "/data/clips"
    first = clip_id("/data/clips/speaker01/take1.mp4", root)
    second = clip_id("/data/clips/speaker02/take1.mp4", root)

    assert first != second, "identically-named files in different folders must stay distinct"
    assert first == "speaker01__take1"
    assert second == "speaker02__take1"
    assert clip_id("/data/clips/take1.mp4", root) == "take1"


def test_speaker_id_is_null_unless_the_layout_provides_one():
    root = "/data/clips"
    path = "/data/clips/speaker07/session2/take1.mp4"

    unknown = infer_grouping(path, root)
    assert unknown["speaker_id"] is None, "a speaker must never be guessed"
    assert unknown["speaker_id_source"] is None
    assert unknown["source_video_id"] == "speaker07/session2/take1"

    parent = infer_grouping(path, root, speaker_from="parent_dir")
    assert parent["speaker_id"] == "session2"
    top = infer_grouping(path, root, speaker_from="top_dir")
    assert top["speaker_id"] == "speaker07"
    assert top["speaker_id_source"] == "top_dir"

    flat = infer_grouping("/data/clips/take1.mp4", root, speaker_from="top_dir")
    assert flat["speaker_id"] is None, "a flat layout carries no speaker information"


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} timeline/id tests.")


if __name__ == "__main__":
    main()
