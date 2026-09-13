"""
Equivalence check for the src/observation/pts.py PTS-exposure refactor.

The refactor split read_stream_timing() out of analyze_video() and added
extract_pts(), so the Dynamics layer can differentiate against real per-frame
PTS (see docs/engineering_log/video_metadata_pts_retention.md). analyze_video()
was required to keep its exact previous behaviour. This script proves that by
loading the pre-refactor implementation directly out of git at a pinned commit,
running both versions over the same clips, and asserting field-by-field equality
on the *complete* returned dict.

Comparing the full dict matters: video_metadata.json only carries 6 of these
fields, so a check against that file alone cannot see a regression in
is_monotonic, max_frame_gap, min_frame_gap, issues or quality_label (those flow
into quality.json instead).

Usage:
    python tests/test_pts_refactor_equivalence.py \
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

from src.observation.pts import analyze_video as analyze_video_current

PRE_REFACTOR_COMMIT = "5f26025"
PTS_MODULE_PATH = "src/observation/pts.py"
COMPARED_FIELDS = (
    "file", "resolution", "fps_declared", "n_frames_decoded", "duration_from_pts",
    "first_pts", "last_pts", "is_monotonic", "max_frame_gap", "min_frame_gap",
    "issues", "quality_label",
)


def load_pre_refactor_analyze_video(commit: str):
    source = subprocess.run(
        ["git", "show", f"{commit}:{PTS_MODULE_PATH}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        temp_path = f.name
    try:
        spec = importlib.util.spec_from_file_location("pts_pre_refactor", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.unlink(temp_path)
    return module.analyze_video


def main():
    parser = argparse.ArgumentParser(description="Assert the pts.py refactor left analyze_video() output unchanged.")
    parser.add_argument("--input_dir", required=True, help="Directory containing mp4 clips (searched recursively).")
    parser.add_argument("--commit", default=PRE_REFACTOR_COMMIT, help="Commit holding the pre-refactor pts.py.")
    args = parser.parse_args()

    analyze_video_before = load_pre_refactor_analyze_video(args.commit)
    mp4_files = sorted(glob.glob(os.path.join(os.path.expanduser(args.input_dir), "**", "*.mp4"), recursive=True))
    if not mp4_files:
        print(f"No mp4 files found under {args.input_dir}")
        sys.exit(1)

    print(f"Comparing analyze_video() at {args.commit} (before) vs working tree (after)")
    print(f"{len(COMPARED_FIELDS)} fields x {len(mp4_files)} clips\n")

    mismatches = []
    for mp4_path in mp4_files:
        before = analyze_video_before(mp4_path)
        after = analyze_video_current(mp4_path)

        assert set(before) == set(after), f"key set changed: {set(before) ^ set(after)}"
        assert set(before) == set(COMPARED_FIELDS), (
            f"analyze_video() keys drifted from this test's expectations: {set(before) ^ set(COMPARED_FIELDS)}"
        )

        for field in COMPARED_FIELDS:
            if before[field] != after[field]:
                mismatches.append((os.path.basename(mp4_path), field, before[field], after[field]))
        status = "OK" if not mismatches else "MISMATCH"
        print(f"  {os.path.basename(mp4_path)}: {len(COMPARED_FIELDS)}/{len(COMPARED_FIELDS)} fields identical [{status}]")

    print()
    for clip, field, b, a in mismatches:
        print(f"MISMATCH {clip}.{field}: before={b!r} after={a!r}")
    assert not mismatches, f"{len(mismatches)} field mismatches -- refactor is NOT behaviour-preserving"

    print(f"PASS: refactor is lossless across {len(mp4_files)} clips "
          f"({len(COMPARED_FIELDS) * len(mp4_files)} field comparisons, exact equality).")


if __name__ == "__main__":
    main()
