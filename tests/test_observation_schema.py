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
  1. All fields that existed at the baseline are still byte-identical.
  2. The only new field is pts_sec -- nothing else was added or removed.
  3. pts_sec is internally consistent with the aggregates derived from it.
  4. pts_sec does NOT leak into quality.json, which must stay lightweight.

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

# Fields present at BASELINE_COMMIT. These must never change value.
INHERITED_FIELDS = (
    "file", "resolution", "fps_declared", "n_frames_decoded", "duration_from_pts",
    "first_pts", "last_pts", "is_monotonic", "max_frame_gap", "min_frame_gap",
    "issues", "quality_label",
)

# Deliberate schema changes since the baseline, one entry per decision.
# pts_sec: full per-frame PTS array, so the Dynamics layer never has to
# reconstruct time as frame_idx / fps. See
# docs/engineering_log/video_metadata_pts_retention.md
INTENTIONALLY_ADDED_FIELDS = {"pts_sec"}


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


def check_inherited_fields_unchanged(before: dict, after: dict, clip: str) -> list:
    assert set(before) == set(INHERITED_FIELDS), (
        f"baseline keys drifted from this test's expectations: {set(before) ^ set(INHERITED_FIELDS)}"
    )
    added = set(after) - set(before)
    removed = set(before) - set(after)
    assert removed == set(), f"{clip}: fields disappeared from analyze_video(): {removed}"
    assert added == INTENTIONALLY_ADDED_FIELDS, (
        f"{clip}: unexpected schema change. Added {added}, expected exactly "
        f"{INTENTIONALLY_ADDED_FIELDS}. If this addition is deliberate, add it to "
        f"INTENTIONALLY_ADDED_FIELDS with a comment explaining why."
    )
    return [(clip, f, before[f], after[f]) for f in INHERITED_FIELDS if before[f] != after[f]]


def check_pts_sec_consistency(record: dict, clip: str):
    pts = record["pts_sec"]
    assert isinstance(pts, list) and pts, f"{clip}: pts_sec must be a non-empty list"
    assert len(pts) == record["n_frames_decoded"], (
        f"{clip}: pts_sec has {len(pts)} entries but n_frames_decoded={record['n_frames_decoded']}"
    )
    assert pts[0] == record["first_pts"], f"{clip}: pts_sec[0] != first_pts"
    assert pts[-1] == record["last_pts"], f"{clip}: pts_sec[-1] != last_pts"
    assert all(pts[i] < pts[i + 1] for i in range(len(pts) - 1)) == record["is_monotonic"], (
        f"{clip}: pts_sec monotonicity disagrees with the is_monotonic flag"
    )


def check_quality_json_stays_lightweight():
    """quality.json must never carry the per-frame array -- it is a summary artifact."""
    pts_results = [{
        "file": "fake.mp4", "pts_sec": [0.0, 0.04, 0.08], "resolution": "1920x1080",
        "fps_declared": 25.0, "n_frames_decoded": 3, "duration_from_pts": 0.08,
        "first_pts": 0.0, "last_pts": 0.08, "is_monotonic": True,
        "max_frame_gap": 0.04, "min_frame_gap": 0.04, "issues": [], "quality_label": "usable",
    }]
    face_results = [{"file": "fake.mp4", "face_detection_rate": 1.0, "avg_confidence": 1.0,
                     "min_confidence": 1.0, "clip_quality_label": "usable"}]

    video_metadata_list, quality_list = merge_results(pts_results, face_results)

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

    print(f"Baseline {args.commit} vs working tree: {len(INHERITED_FIELDS)} inherited fields "
          f"x {len(mp4_files)} clips, plus {sorted(INTENTIONALLY_ADDED_FIELDS)}\n")

    mismatches = []
    for mp4_path in mp4_files:
        clip = os.path.basename(mp4_path)
        before = analyze_video_baseline(mp4_path)
        after = analyze_video_current(mp4_path)

        mismatches.extend(check_inherited_fields_unchanged(before, after, clip))
        check_pts_sec_consistency(after, clip)
        print(f"  {clip}: {len(INHERITED_FIELDS)}/{len(INHERITED_FIELDS)} inherited identical, "
              f"pts_sec {len(after['pts_sec'])} entries consistent [OK]")

    for clip, field, b, a in mismatches:
        print(f"MISMATCH {clip}.{field}: before={b!r} after={a!r}")
    assert not mismatches, f"{len(mismatches)} field mismatches -- schema change is NOT backward-compatible"

    print()
    check_quality_json_stays_lightweight()
    print(f"\nPASS: {len(INHERITED_FIELDS) * len(mp4_files)} inherited-field comparisons exact, "
          f"pts_sec consistent on {len(mp4_files)} clips, quality.json uncontaminated.")


if __name__ == "__main__":
    main()
