"""
Unit tests for the run layout and artifact naming.

Self-contained: builds run directories under a temporary root, touches files, and
checks the derived paths. No clips, no models, no subprocesses.

The point under test is that paths are DERIVED, never assembled by a caller, so a
stage cannot write somewhere the next stage does not look.

Usage:
    python tests/test_pipeline_layout.py
"""
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.pipeline.layout import (
    STAGE_ARTIFACTS, STAGE_DIRS, STAGE_FEATURES, STAGE_GEOMETRY, STAGE_OBSERVATION,
    STAGES, RunLayout, latest_run, make_run_id, parse_run_id, sanitize_tag,
)


def _layout(root: str, run_id: str = "demo__20260913T181500Z__abc1234") -> RunLayout:
    return RunLayout(root=root, run_id=run_id)


def test_run_id_is_unique_sortable_and_traceable():
    moment = datetime(2026, 9, 13, 18, 15, 0, tzinfo=timezone.utc)
    run_id = make_run_id("talkvid sample!", REPO_ROOT, timestamp=moment)

    parts = parse_run_id(run_id)
    assert parts["timestamp"] == "20260913T181500Z"
    assert parts["tag"] == "talkvid-sample", f"tag was not sanitised: {parts['tag']}"
    assert parts["revision"], "a run id must record the code revision it came from"

    # Fixed-width UTC basic format means lexicographic order is chronological,
    # which is what latest_run() relies on.
    earlier = make_run_id("x", REPO_ROOT, timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    later = make_run_id("x", REPO_ROOT, timestamp=datetime(2026, 11, 2, 3, 4, 5, tzinfo=timezone.utc))
    assert earlier < later, "run ids must sort chronologically as plain strings"


def test_tag_sanitisation_keeps_paths_safe():
    assert sanitize_tag("a/b\\c") == "a-b-c", "separators must not survive into a path component"
    assert sanitize_tag("  ..weird..  ") == "weird"
    assert sanitize_tag("") == "run", "an empty tag must still yield a usable directory name"
    assert sanitize_tag("!!!") == "run"
    assert len(sanitize_tag("x" * 200)) <= 48
    assert ".." not in sanitize_tag("../../etc/passwd"), "a tag must not be able to escape the run root"


def test_paths_are_derived_and_typo_proof():
    root = tempfile.mkdtemp()
    try:
        layout = _layout(root)
        assert layout.run_dir == os.path.join(root, layout.run_id)
        assert layout.stage_dir(STAGE_GEOMETRY).endswith("01_geometry")
        assert layout.artifact(STAGE_GEOMETRY, "geometry.parquet") == os.path.join(
            layout.run_dir, "01_geometry", "geometry.parquet"
        )

        # A filename that is not a declared artifact of the stage must not resolve.
        for stage, filename in ((STAGE_GEOMETRY, "geometry.parquett"),
                                (STAGE_OBSERVATION, "geometry.parquet")):
            try:
                layout.artifact(stage, filename)
                raise AssertionError(f"{filename!r} should not resolve under {stage}")
            except KeyError:
                pass

        try:
            layout.stage_dir("nonexistent")
            raise AssertionError("an unknown stage must raise")
        except KeyError:
            pass
    finally:
        shutil.rmtree(root)


def test_stage_directories_sort_in_execution_order():
    names = [STAGE_DIRS[stage] for stage in STAGES]
    assert names == sorted(names), f"numeric prefixes must keep folders in pipeline order: {names}"
    assert len(set(names)) == len(names)


def test_completeness_requires_every_artifact_and_rejects_truncation():
    root = tempfile.mkdtemp()
    try:
        layout = _layout(root)
        layout.create_directories()
        assert not layout.is_complete(STAGE_OBSERVATION)

        artifacts = layout.stage_artifacts(STAGE_OBSERVATION)
        assert len(artifacts) == len(STAGE_ARTIFACTS[STAGE_OBSERVATION])

        with open(artifacts[0], "w") as handle:
            handle.write("{}")
        assert not layout.is_complete(STAGE_OBSERVATION), "one of two artifacts is not complete"
        assert len(layout.missing_artifacts(STAGE_OBSERVATION)) == 1

        # A crash mid-write leaves a zero-length file; resuming onto it is worse
        # than redoing the stage, so it must not count as complete.
        open(artifacts[1], "w").close()
        assert not layout.is_complete(STAGE_OBSERVATION), "a zero-length artifact must not count"

        with open(artifacts[1], "w") as handle:
            handle.write("{}")
        assert layout.is_complete(STAGE_OBSERVATION)
        assert layout.missing_artifacts(STAGE_OBSERVATION) == []
    finally:
        shutil.rmtree(root)


def test_latest_run_filters_by_tag_and_ignores_foreign_directories():
    root = tempfile.mkdtemp()
    try:
        moments = [datetime(2026, 3, 1, tzinfo=timezone.utc), datetime(2026, 7, 1, tzinfo=timezone.utc)]
        ids = [make_run_id("alpha", REPO_ROOT, timestamp=m) for m in moments]
        other = make_run_id("beta", REPO_ROOT, timestamp=datetime(2026, 12, 1, tzinfo=timezone.utc))
        for name in ids + [other, "not-a-run-dir", "scratch"]:
            os.makedirs(os.path.join(root, name))

        assert latest_run(root, "alpha") == ids[1], "must pick the most recent run of that tag"
        assert latest_run(root, "beta") == other
        assert latest_run(root) == other, "unfiltered, the newest run overall wins"
        assert latest_run(root, "missing") is None
        assert latest_run(os.path.join(root, "nope")) is None
    finally:
        shutil.rmtree(root)


def test_every_stage_declares_artifacts_and_a_log():
    root = tempfile.mkdtemp()
    try:
        layout = _layout(root)
        for stage in STAGES:
            assert STAGE_ARTIFACTS[stage], f"{stage} declares no artifacts, so completeness is undefined"
            assert layout.log_path(stage).startswith(layout.log_dir)
            assert layout.log_path(stage).endswith(".log")
        logs = {layout.log_path(stage) for stage in STAGES}
        assert len(logs) == len(STAGES), "each stage needs its own log file"
    finally:
        shutil.rmtree(root)


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} pipeline-layout tests.")


if __name__ == "__main__":
    main()
