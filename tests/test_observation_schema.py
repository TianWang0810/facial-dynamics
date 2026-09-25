"""
Schema guard for the Observation layer's analyze_video() / merge_results() output.

Originally written to prove that the read_stream_timing() refactor (commit
07350ac) left analyze_video() bit-identical. It still asserts exactly that, but
is now also the record of every *deliberate* change to the Observation output
schema: each intentional addition is whitelisted by name, so any unintended
field appearing or disappearing still fails loudly.

The "before" side is loaded straight out of git at a pinned baseline commit, so
the comparison stays reproducible as history moves on.

Checked here:
  1. Every baseline field is still present, under its baseline name or the
     recorded replacement, and carries the same value.
  2. Every added or renamed field is whitelisted with the decision behind it;
     anything else appearing or disappearing still fails loudly.
  3. pts_sec is internally consistent with the aggregates derived from it.
  4. pts_sec does NOT leak into quality.json, which must stay lightweight.

Note on the v2 timestamp policy break. The baseline dropped frames whose PTS the
container did not report, which shifted every later frame's index while leaving
the array length plausible. Such frames are now kept in place with a null
timestamp, so on a clip containing one, pts_sec is LONGER than the baseline's and
the aggregates differ. That is the point of the change, not a regression, and
BROKEN_BY_DESIGN_ON records the condition under which the comparison is expected
to differ -- the guard skips value comparison for those clips and says so, rather
than being weakened for every clip.

Plain asserts rather than pytest: env-v0.1 is frozen and ships no test runner
(design doc section 6.4).

Usage:
    python tests/test_observation_schema.py \
        --input_dir ~/facial-dynamics/data/raw_clips_hpc_test3
"""
import argparse
import glob
import importlib.util
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.observation.merge import merge_results
from src.observation.pts import analyze_video as analyze_video_current

BASELINE_COMMIT = "5f26025"
PTS_MODULE_PATH = "src/observation/pts.py"

# Fields present at BASELINE_COMMIT whose name and value are unchanged.
INHERITED_FIELDS = (
    "file", "resolution", "fps_declared", "n_frames_decoded", "quality_label",
)

# Baseline field -> current field. Renamed to say what they actually mean; the
# value must still match on a clip with no timing defects.
RENAMED_FIELDS = {
    "duration_from_pts": "duration_from_pts_sec",
    "first_pts": "first_pts_sec",
    "last_pts": "last_pts_sec",
    "max_frame_gap": "max_frame_gap_sec",
    "min_frame_gap": "min_frame_gap_sec",
    # "monotonic" said nothing about missing timestamps; the new name is the
    # property actually required of a usable clip.
    "is_monotonic": "is_strictly_monotonic",
}

# Present under the SAME name in both, but carrying a different representation.
#   issues  free-text strings at the baseline, structured {code, n_frames,
#           frame_indices, detail} records now. The two are only comparable when
#           both are empty -- which is the case on a clip with no timing defects,
#           and a clip with defects is skipped from value comparison anyway.
REPRESENTATION_CHANGED_FIELDS = ("issues",)

# Deliberate additions since the baseline, one entry per decision.
#   pts_sec            full per-frame array, so the Dynamics layer never
#                      reconstructs time as frame_idx / fps
#   n_missing_pts      frames the container gave no timestamp for; they are now
#                      retained rather than dropped, so the count must be visible
#   fps_from_pts       measured rate, used to resample onto the model grid
#   median_step_sec    the measured step fps_from_pts is the reciprocal of, kept
#                      so a consumer reads the raw quantity, not only its inverse
#   large_gaps         located, not merely counted, so a caller can segment
#   width / height     the numeric pair behind the "WxH" resolution string, so no
#                      consumer has to parse that string back apart
#   path               absolute source path, for traceability
INTENTIONALLY_ADDED_FIELDS = {
    "pts_sec", "n_missing_pts", "fps_from_pts", "median_step_sec",
    "large_gaps", "width", "height", "path",
    # Per-stream container timing (video/audio start, A/V offset, audio format),
    # read from headers only; added for the 04_audio stage, which needs it to
    # cut audio by real time. See src/observation/decode.py:read_stream_timing.
    "streams",
}

# Fields whose value is asserted identical across the baseline: the unchanged
# ones plus the renamed ones read through the rename map.
COMPARED_FIELDS = len(INHERITED_FIELDS) + len(RENAMED_FIELDS)

# Values are compared only on clips free of these conditions; see the module
# docstring. A clip WITH one is reported, not silently passed.
BROKEN_BY_DESIGN_ON = "a clip containing frames with no PTS"


def load_baseline_analyze_video(commit: str):
    source = subprocess.run(
        ["git", "show", f"{commit}:{PTS_MODULE_PATH}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        temp_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("pts_baseline", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.unlink(temp_path)
    return module.analyze_video


def check_key_sets(before: dict, after: dict, clip: str) -> None:
    """Every baseline key is accounted for, and every current key is whitelisted.

    A renamed field is NOT a removal plus an addition: it is one decision, so it
    is matched through RENAMED_FIELDS on both sides. Comparing the raw key sets
    without that map made every rename look like both a disappearance and an
    unexplained addition, which is what this guard is supposed to distinguish.
    """
    expected_before = set(INHERITED_FIELDS) | set(RENAMED_FIELDS) | set(REPRESENTATION_CHANGED_FIELDS)
    assert set(before) == expected_before, (
        f"baseline keys drifted from this test's expectations: {sorted(set(before) ^ expected_before)}"
    )

    expected_after = (set(INHERITED_FIELDS) | set(RENAMED_FIELDS.values())
                      | set(REPRESENTATION_CHANGED_FIELDS) | INTENTIONALLY_ADDED_FIELDS)
    disappeared = expected_after - set(after)
    unexpected = set(after) - expected_after
    assert not disappeared, (
        f"{clip}: fields disappeared from analyze_video(): {sorted(disappeared)}"
    )
    assert not unexpected, (
        f"{clip}: unexpected schema change. {sorted(unexpected)} appeared with no whitelist entry. "
        f"If the addition is deliberate, add it to INTENTIONALLY_ADDED_FIELDS -- or to "
        f"RENAMED_FIELDS if it replaces a baseline field -- with a comment explaining why."
    )


def check_inherited_fields_unchanged(before: dict, after: dict, clip: str) -> list:
    check_key_sets(before, after, clip)

    if after.get("n_missing_pts"):
        # The baseline silently dropped those frames, so its aggregates describe a
        # different, shorter clip. Comparing them would assert the old bug.
        return [(clip, "__skipped__", BROKEN_BY_DESIGN_ON,
                 f"{after['n_missing_pts']} frame(s) with no PTS; value comparison skipped")]

    mismatches = [(clip, f, before[f], after[f]) for f in INHERITED_FIELDS if before[f] != after[f]]
    mismatches += [(clip, f"{old}->{new}", before[old], after[new])
                   for old, new in RENAMED_FIELDS.items() if before[old] != after[new]]
    # A representation-changed field is only comparable when both sides are
    # empty. On a clip that got here, neither side found a defect, so anything
    # non-empty on either side is a genuine disagreement about the same clip.
    mismatches += [(clip, f, before[f], after[f]) for f in REPRESENTATION_CHANGED_FIELDS
                   if bool(before[f]) or bool(after[f])]
    return mismatches


def check_pts_sec_consistency(record: dict, clip: str):
    pts = record["pts_sec"]
    assert isinstance(pts, list) and pts, f"{clip}: pts_sec must be a non-empty list"
    assert len(pts) == record["n_frames_decoded"], (
        f"{clip}: pts_sec has {len(pts)} entries but n_frames_decoded={record['n_frames_decoded']}"
    )
    present = [v for v in pts if v is not None]
    assert len(pts) - len(present) == record["n_missing_pts"], (
        f"{clip}: nulls in pts_sec disagree with n_missing_pts"
    )
    if present:
        assert present[0] == record["first_pts_sec"], f"{clip}: first present pts != first_pts_sec"
        assert present[-1] == record["last_pts_sec"], f"{clip}: last present pts != last_pts_sec"
    strictly_increasing = all(present[i] < present[i + 1] for i in range(len(present) - 1))
    expected = strictly_increasing and record["n_missing_pts"] == 0
    assert expected == record["is_strictly_monotonic"], (
        f"{clip}: pts_sec disagrees with the is_strictly_monotonic flag"
    )


def check_quality_json_stays_lightweight():
    """quality.json must never carry the per-frame array -- it is a summary artifact."""
    pts_results = [{
        "file": "fake.mp4", "pts_sec": [0.0, 0.04, 0.08], "resolution": "1920x1080",
        "fps_declared": 25.0, "n_frames_decoded": 3, "n_missing_pts": 0,
        "duration_from_pts_sec": 0.08, "first_pts_sec": 0.0, "last_pts_sec": 0.08,
        "is_strictly_monotonic": True, "large_gaps": [],
        "max_frame_gap_sec": 0.04, "min_frame_gap_sec": 0.04, "issues": [], "quality_label": "usable",
    }]


    video_metadata_list, quality_list = merge_results(pts_results)

    assert "pts_sec" in video_metadata_list[0], "video_metadata.json must carry pts_sec"
    serialized = repr(quality_list)
    assert "pts_sec" not in serialized, "pts_sec leaked into quality.json"
    assert "0.04, 0.08" not in serialized, "the per-frame PTS array leaked into quality.json"
    print("  quality.json stays lightweight: no pts_sec, no per-frame array [OK]")


def main():
    parser = argparse.ArgumentParser(description="Guard the Observation-layer output schema.")
    parser.add_argument("--input_dir", required=True, help="Directory containing mp4 clips (searched recursively).")
    parser.add_argument("--commit", default=BASELINE_COMMIT, help="Commit holding the baseline pts.py.")
    args = parser.parse_args()

    analyze_video_baseline = load_baseline_analyze_video(args.commit)
    mp4_files = sorted(glob.glob(os.path.join(os.path.expanduser(args.input_dir), "**", "*.mp4"), recursive=True))
    if not mp4_files:
        print(f"No mp4 files found under {args.input_dir}")
        sys.exit(1)

    print(f"Baseline {args.commit} vs working tree: {COMPARED_FIELDS} carried-over fields "
          f"({len(INHERITED_FIELDS)} same-name, {len(RENAMED_FIELDS)} renamed) x {len(mp4_files)} clips, "
          f"plus {sorted(INTENTIONALLY_ADDED_FIELDS)}\n")

    mismatches, n_compared_clips = [], 0
    for mp4_path in mp4_files:
        clip = os.path.basename(mp4_path)
        before = analyze_video_baseline(mp4_path)
        after = analyze_video_current(mp4_path)

        found = check_inherited_fields_unchanged(before, after, clip)
        mismatches.extend(found)
        check_pts_sec_consistency(after, clip)

        skipped = [f for f in found if f[1] == "__skipped__"]
        if skipped:
            print(f"  {clip}: keys whitelisted, value comparison SKIPPED ({skipped[0][3]}), "
                  f"pts_sec {len(after['pts_sec'])} entries consistent")
        else:
            n_compared_clips += 1
            print(f"  {clip}: {COMPARED_FIELDS}/{COMPARED_FIELDS} carried-over fields identical, "
                  f"pts_sec {len(after['pts_sec'])} entries consistent [OK]")

    # A skip is reported, not counted as a failure: it is the documented
    # consequence of the v2 timestamp policy, not a regression.
    real_mismatches = [m for m in mismatches if m[1] != "__skipped__"]
    for clip, field, b, a in real_mismatches:
        print(f"MISMATCH {clip}.{field}: before={b!r} after={a!r}")
    assert not real_mismatches, (
        f"{len(real_mismatches)} field mismatches -- schema change is NOT backward-compatible"
    )

    print()
    check_quality_json_stays_lightweight()
    print(f"\nPASS: {COMPARED_FIELDS * n_compared_clips} carried-over-field comparisons exact "
          f"on {n_compared_clips}/{len(mp4_files)} clips, pts_sec consistent on {len(mp4_files)} clips, "
          f"quality.json uncontaminated.")


if __name__ == "__main__":
    main()
