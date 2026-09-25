"""
One-command entrypoint: mp4 directory in, windowed training tensors out.

Runs Observation -> Geometry -> Features -> Dynamics -> Audio in order, wiring each
stage's outputs into the next so no path is ever passed by hand. Every artifact
lands in a timestamped run directory laid out by src/pipeline/layout.py.

Usage:
    python scripts/run_pipeline.py \
        --input_dir <dir_of_mp4> \
        --model_path <face_landmarker.task> \
        --output_root runs/

    # restart after a failure without redoing the expensive extraction
    python scripts/run_pipeline.py ... --resume

    # re-run one stage against an existing run
    python scripts/run_pipeline.py ... --run_id <id> --stages features --force

Stages run as subprocesses of the same per-stage CLI a person would call by
hand. That is deliberate: there is exactly one implementation of each stage, so
the pipeline and the standalone scripts cannot drift apart, and a stage that
fails can be re-run alone with the command printed in the log.

Audio (04) cuts a conditioning sidecar aligned row-for-row with features.npz by
real window times; a clip without an audio stream yields invalid audio, not a
failure. It never changes the 63-dim tensor.

Dynamics is a validation stage, not a data-producing one -- it recomputes
derivatives and reports on them. It is included by default because a silent
change in motion statistics is worth noticing, and it writes only a summary.
"""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.pipeline.layout import (
    STAGE_AUDIO, STAGE_DYNAMICS, STAGE_FEATURES, STAGE_GEOMETRY, STAGE_OBSERVATION, STAGES,
    RunLayout, latest_run, make_run_id,
)
from src.schema import feature_vector as schema


def _stage_command(stage: str, layout: RunLayout, args) -> list:
    """The exact argv for one stage, with every path derived from the layout."""
    python = sys.executable
    scripts = os.path.join(REPO_ROOT, "scripts")

    if stage == STAGE_OBSERVATION:
        command = [python, os.path.join(scripts, "run_observation.py"),
                   "--input_dir", args.input_dir,
                   "--output_dir", layout.stage_dir(STAGE_OBSERVATION)]
        if args.speaker_from:
            command += ["--speaker_from", args.speaker_from]
        return command

    if stage == STAGE_GEOMETRY:
        command = [python, os.path.join(scripts, "run_geometry.py"),
                   "--input_dir", args.input_dir,
                   "--model_path", args.model_path,
                   "--output_dir", layout.stage_dir(STAGE_GEOMETRY),
                   "--gaze_backend", args.gaze_backend]
        if args.gaze_model_path:
            command += ["--gaze_model_path", args.gaze_model_path]
        if args.gaze_tanh:
            command += ["--gaze_tanh"]
        if args.speaker_from:
            command += ["--speaker_from", args.speaker_from]
        return command

    if stage == STAGE_FEATURES:
        command = [python, os.path.join(scripts, "run_features.py"),
                   "--geometry_parquet", layout.artifact(STAGE_GEOMETRY, "geometry.parquet"),
                   "--video_metadata", layout.artifact(STAGE_OBSERVATION, "video_metadata.json"),
                   "--observation_quality", layout.artifact(STAGE_OBSERVATION, "quality.json"),
                   "--output", layout.artifact(STAGE_FEATURES, "features.npz"),
                   "--target_hz", str(args.target_hz),
                   "--max_gap_seconds", str(args.max_gap_seconds),
                   "--window_seconds", str(args.window_seconds),
                   "--overlap_ratio", str(args.overlap_ratio),
                   "--min_detection_rate", str(args.min_detection_rate)]
        if args.symmetric_blendshape:
            command += ["--symmetric_blendshape"]
        if args.allow_timeline_issues:
            command += ["--allow_timeline_issues"]
        return command

    if stage == STAGE_DYNAMICS:
        return [python, os.path.join(scripts, "run_dynamics.py"),
                "--geometry_parquet", layout.artifact(STAGE_GEOMETRY, "geometry.parquet"),
                "--video_metadata", layout.artifact(STAGE_OBSERVATION, "video_metadata.json"),
                "--signal", args.dynamics_signal,
                "--report_json", layout.artifact(STAGE_DYNAMICS, "dynamics_report.json")]

    if stage == STAGE_AUDIO:
        return [python, os.path.join(scripts, "run_audio.py"),
                "--features_npz", layout.artifact(STAGE_FEATURES, "features.npz"),
                "--video_metadata", layout.artifact(STAGE_OBSERVATION, "video_metadata.json"),
                "--input_dir", args.input_dir,
                "--geometry_metadata", layout.artifact(STAGE_GEOMETRY, "geometry_metadata.json"),
                "--output", layout.artifact(STAGE_AUDIO, "audio.npz")]

    raise KeyError("no command defined for stage %r" % stage)


# Packages whose version changes what the pipeline produces, not merely how fast
# it runs: av decodes the PTS every downstream timestamp is derived from,
# mediapipe produces the landmarks, and numpy the arithmetic on both.
PROVENANCE_PACKAGES = ("av", "mediapipe", "numpy", "pandas", "pyarrow")


def _environment_record() -> dict:
    """Which interpreter and which package versions actually produced this run.

    The run id names the code revision, but code is only half of what determines
    the output. environment/env-v0.1 locks the other half; recording what ran is
    what makes a deviation from that lock visible in the manifest instead of
    having to be reconstructed later from whoever remembers which venv was active.
    """
    versions = {}
    for name in PROVENANCE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": versions,
        "note": "compare against environment/env-v0.1/conda-lock.yml; this records what ran, "
                "it does not enforce the lock",
    }


def _run_stage(stage: str, command: list, log_path: str) -> dict:
    """Run one stage, teeing its output to the console and to its log file."""
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ %s\n\n" % " ".join(command))
        log.flush()
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write("    " + line)
            log.write(line)
        process.wait()

    return {
        "stage": stage, "command": command, "returncode": process.returncode,
        "seconds": round(time.time() - started, 2), "log": log_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the full facial-dynamics pipeline: mp4 -> windowed 63-dim tensors.")
    parser.add_argument("--input_dir", required=True, help="Directory of mp4 clips, searched recursively.")
    parser.add_argument("--model_path", help="Path to face_landmarker.task. Required unless the geometry stage is skipped.")
    parser.add_argument("--output_root", default=os.path.join(REPO_ROOT, "runs"),
                        help="Directory that run folders are created under.")
    parser.add_argument("--tag", default=None,
                        help="Dataset label used in the run id. Defaults to the input directory's name.")
    parser.add_argument("--run_id", default=None,
                        help="Reuse an existing run directory instead of creating one. Use with --stages or --resume.")
    parser.add_argument("--resume_latest", action="store_true",
                        help="Reuse the most recent run with the same tag.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                        help="Stages to run, in pipeline order regardless of the order given.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages whose artifacts already exist and are non-empty.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run selected stages even if their artifacts exist. Overrides --resume.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands that would run and exit without executing or creating a run directory.")
    parser.add_argument("--keep_going", action="store_true",
                        help="Continue with later stages after one fails. Off by default, since a later stage would read an incomplete artifact.")

    geometry = parser.add_argument_group("geometry stage")
    geometry.add_argument("--gaze_backend", choices=("iris", "external"), default="iris")
    geometry.add_argument("--gaze_model_path", default=None)
    geometry.add_argument("--gaze_tanh", action="store_true")
    geometry.add_argument("--speaker_from", choices=("parent_dir", "top_dir"), default=None,
                          help="Where to read a speaker id from the directory layout. Without it, "
                               "speaker_id is null and any split is clip-disjoint only.")

    features = parser.add_argument_group("features stage")
    features.add_argument("--target_hz", type=float,
                          default=float(schema.WINDOWING["resampling"]["default_target_hz"]),
                          help="Uniform model sample rate.")
    features.add_argument("--max_gap_seconds", type=float,
                          default=float(schema.WINDOWING["resampling"]["max_gap_seconds_default"]),
                          help="Unobserved stretches longer than this split a clip into segments.")
    features.add_argument("--window_seconds", type=float,
                          default=float(schema.WINDOWING["default_target_seconds"]))
    features.add_argument("--overlap_ratio", type=float,
                          default=float(schema.WINDOWING["default_overlap_ratio"]))
    features.add_argument("--min_detection_rate", type=float, default=0.7)
    features.add_argument("--allow_timeline_issues", action="store_true")
    features.add_argument("--symmetric_blendshape", action="store_true")

    dynamics = parser.add_argument_group("dynamics stage")
    dynamics.add_argument("--dynamics_signal", default="blendshapes",
                          help="Geometry column the validation stage differentiates.")

    args = parser.parse_args()

    selected = [stage for stage in STAGES if stage in set(args.stages)]
    if not os.path.isdir(args.input_dir):
        parser.error("--input_dir %r is not a directory" % args.input_dir)

    tag = args.tag or os.path.basename(os.path.abspath(args.input_dir))
    if args.run_id:
        run_id = args.run_id
    elif args.resume_latest:
        run_id = latest_run(args.output_root, tag)
        if not run_id:
            parser.error("no existing run with tag %r under %s" % (tag, args.output_root))
        print(f"Resuming latest run: {run_id}")
    else:
        run_id = make_run_id(tag, REPO_ROOT)

    layout = RunLayout(root=os.path.abspath(args.output_root), run_id=run_id)

    # Checked here rather than at parse time: a geometry stage that --resume will
    # skip needs no model, and demanding one would block restarting a run on a
    # machine where the model file is not mounted.
    will_run_geometry = STAGE_GEOMETRY in selected and not (
        args.resume and not args.force and layout.is_complete(STAGE_GEOMETRY)
    )
    if will_run_geometry and not args.model_path:
        parser.error("--model_path is required when the geometry stage runs "
                     "(it would be skipped only with --resume and existing artifacts)")

    print(f"Run id:    {run_id}")
    print(f"Run dir:   {layout.run_dir}")
    print(f"Input:     {os.path.abspath(args.input_dir)}")
    print(f"Stages:    {' -> '.join(selected)}")
    print(f"Schema:    v{schema.SCHEMA_VERSION} ({schema.TOTAL_DIM}-dim)")
    print()

    if args.dry_run:
        for stage in selected:
            print(f"[{stage}]")
            print("    " + " ".join(_stage_command(stage, layout, args)))
        print("\nDry run: nothing executed, no directories created.")
        return

    layout.create_directories()

    results = []
    failed = None
    for stage in selected:
        if layout.is_complete(stage) and args.resume and not args.force:
            print(f"[{stage}] SKIPPED (artifacts present; --force to redo)")
            results.append({"stage": stage, "skipped": True, "returncode": 0})
            continue

        command = _stage_command(stage, layout, args)
        print(f"[{stage}] running")
        result = _run_stage(stage, command, layout.log_path(stage))
        missing = layout.missing_artifacts(stage)
        result["missing_artifacts"] = missing
        results.append(result)

        if result["returncode"] != 0:
            print(f"[{stage}] FAILED (exit {result['returncode']}), see {result['log']}")
            failed = stage
        elif missing:
            print(f"[{stage}] FAILED: exited cleanly but did not produce {missing}")
            result["returncode"] = 1
            failed = stage
        else:
            print(f"[{stage}] ok ({result['seconds']}s)")

        if failed and not args.keep_going:
            print(f"\nStopping: a later stage would read an incomplete {failed} artifact. "
                  f"Use --keep_going to override, or fix and re-run with --resume.")
            break
        failed = None if args.keep_going else failed

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_dir": os.path.abspath(args.input_dir),
        "model_path": os.path.abspath(args.model_path) if args.model_path else None,
        "schema_version": schema.SCHEMA_VERSION,
        "environment": _environment_record(),
        "stages_selected": selected,
        "settings": {k: v for k, v in sorted(vars(args).items()) if k not in ("stages",)},
        "results": results,
        "artifacts": {
            stage: {
                "dir": os.path.relpath(layout.stage_dir(stage), layout.run_dir),
                "complete": layout.is_complete(stage),
            } for stage in STAGES
        },
    }
    with open(layout.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("\n--- Pipeline summary ---")
    for result in results:
        if result.get("skipped"):
            print(f"  {result['stage']:<12} skipped")
        else:
            status = "ok" if result["returncode"] == 0 else f"FAILED ({result['returncode']})"
            print(f"  {result['stage']:<12} {status:<14} {result.get('seconds', 0)}s")
    print(f"\nRun dir:  {layout.run_dir}")
    print(f"Manifest: {layout.manifest_path}")

    if any(r["returncode"] != 0 for r in results):
        sys.exit(1)
    if STAGE_FEATURES in selected and layout.is_complete(STAGE_FEATURES):
        print(f"Tensors:  {layout.artifact(STAGE_FEATURES, 'features.npz')}")


if __name__ == "__main__":
    main()
