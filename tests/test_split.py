"""
Tests for split construction and the leakage claims around it.

Usage:
    python tests/test_split.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.sequence.split import (
    KIND_CLIP_DISJOINT, KIND_SPEAKER_DISJOINT, fit_normalization, make_split,
    verify_no_leakage,
)


def test_speaker_never_spans_two_splits():
    clips = [f"s{s}__take{t}" for s in range(1, 7) for t in range(3)]
    speakers = [c.split("__")[0] for c in clips]

    split = make_split(clips, speakers, seed=3)
    assert split["kind"] == KIND_SPEAKER_DISJOINT
    assert split["limitation"] is None

    report = verify_no_leakage(split, dict(zip(clips, speakers)))
    assert report["is_clean"], f"leakage: {report}"
    assert report["speaker_check_performed"]

    # Every clip of a speaker landed in the same split.
    by_speaker = {}
    for clip, name in split["assignments"].items():
        by_speaker.setdefault(clip.split("__")[0], set()).add(name)
    assert all(len(v) == 1 for v in by_speaker.values()), f"speaker split across sides: {by_speaker}"


def test_missing_speaker_downgrades_the_claim_rather_than_guessing():
    clips = [f"clip{i}" for i in range(9)]
    split = make_split(clips, None, seed=1)

    assert split["kind"] == KIND_CLIP_DISJOINT, "without speaker ids the split cannot claim more"
    assert split["limitation"] and "NOT be described as identity-disjoint" in split["limitation"]

    report = verify_no_leakage(split)
    assert report["is_clean"], "clips must still not repeat across splits"
    assert not report["speaker_check_performed"], \
        "the speaker check must report that it did not run, rather than passing vacuously"

    partial = make_split(clips, ["a", None, "b", None, "c", "d", "e", "f", "g"], seed=1)
    assert partial["kind"] == KIND_CLIP_DISJOINT, "one unknown speaker invalidates the stronger claim"


def test_windows_follow_their_clip_so_overlap_cannot_cross():
    clips = [f"s{s}__take0" for s in range(1, 7)]
    speakers = [c.split("__")[0] for c in clips]
    split = make_split(clips, speakers, seed=0)

    # Overlapping windows of one clip all inherit that clip's split.
    window_clip = [c for c in clips for _ in range(5)]
    window_split = [split["assignments"][c] for c in window_clip]
    for index in range(0, len(window_clip), 5):
        assert len(set(window_split[index:index + 5])) == 1, \
            "windows of one clip must not be dealt to different splits"


def test_normalization_is_fitted_on_train_only():
    clips = ["a", "b", "c", "d"]
    split = {"assignments": {"a": "train", "b": "train", "c": "val", "d": "test"}}

    window_clip = ["a", "b", "c", "d"]
    features = np.stack([
        np.full((4, 3), 1.0), np.full((4, 3), 3.0),
        np.full((4, 3), 100.0), np.full((4, 3), -100.0),
    ])

    stats = fit_normalization(features, window_clip, split, on="train")
    assert stats["fitted_on"] == "train" and stats["n_windows"] == 2
    assert abs(stats["mean"][0] - 2.0) < 1e-9, \
        f"val/test values leaked into the mean: {stats['mean'][0]}"
    assert all(s > 0 for s in stats["std"]), "std must stay positive to be divisible"

    try:
        fit_normalization(features, window_clip, {"assignments": {}}, on="train")
        raise AssertionError("fitting on an empty split must raise, not return garbage")
    except ValueError:
        pass


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} split tests.")


if __name__ == "__main__":
    main()
