# Decision: one entrypoint, one run directory, derived paths

## Context

The four stages were previously run by hand, each with its own `--input_dir` /
`--output_dir` / `--geometry_parquet` flags, and outputs landed wherever the
operator pointed them. That has three failure modes, all of which cost time
rather than producing errors:

- Stage 3 reads a `geometry.parquet` from a different run than the
  `video_metadata.json` it is paired with. Nothing detects this; the frame counts
  happen to match and the features are silently built against the wrong clip.
- A result cannot be traced back to the code that produced it.
- Re-running after a crash means redoing the expensive extraction, because there
  is no notion of which stages already completed.

## Decision

`scripts/run_pipeline.py` is the entrypoint. It runs Observation -> Geometry ->
Features -> Dynamics, wiring each stage's outputs into the next so no path is
passed by hand.

`src/pipeline/layout.py` owns every path, the same way
`schemas/geometry.schema.json` owns the feature layout. Nothing joins path
strings ad hoc.

### Naming: unique outside, canonical inside

The run directory name is unique -- `<tag>__<UTC timestamp>__<git sha>` -- so runs
never overwrite each other and every result names the revision that produced it.
The timestamp is fixed-width UTC basic format, which makes lexicographic sort
chronological; `latest_run()` depends on that.

The artifacts inside are canonically named at a fixed depth: `geometry.parquet`
is always `01_geometry/geometry.parquet`. A stage never has to guess what the
previous stage called its output, and a human can predict a path without parsing
a run name. Uniqueness lives in exactly one place -- the run directory -- rather
than being smeared across every filename.

Numeric directory prefixes (`00_`, `01_`) make execution order visible in a
listing and keep the folders sorted in production order.

`RunLayout.artifact()` validates the filename against the stage's declared
artifacts, so a typo raises where it is written instead of producing a file
nobody downstream looks for.

### Stages run as subprocesses

Each stage is invoked as the same per-stage CLI a person would run by hand. There
is therefore exactly one implementation of each stage: the pipeline and the
standalone scripts cannot drift. The exact command is written at the top of the
stage's log, so a failing stage can be re-run alone by copying one line.

The cost is process startup per stage, which is irrelevant next to decoding
video.

### Completion is defined by artifacts, not by exit code

A stage is complete when every declared artifact exists AND is non-empty.
Zero-length files are treated as incomplete, because that is exactly what a crash
mid-write leaves behind, and resuming onto a truncated artifact is worse than
redoing the stage.

The orchestrator additionally treats "exited 0 but did not produce its artifacts"
as a failure. A stage that skips every clip for a legitimate reason still exits
cleanly today, and that must not be mistaken for success.

Execution stops at the first failure unless `--keep_going` is passed, since the
next stage would otherwise read an incomplete artifact.

## Consequences

- `--resume` skips completed stages, so a run can be restarted after a failure
  without redoing extraction. `--force` overrides it, `--stages` selects a subset.
- `--model_path` is required only when the geometry stage will actually run.
  Checking it at parse time would block resuming a run on a machine where the
  model file is not mounted.
- `--dry_run` prints the exact commands and creates nothing, so the wiring can be
  inspected before a long job.
- `run_manifest.json` records the run id, inputs, schema version, every setting,
  and per-stage exit code and duration.
- `runs/` is gitignored. A run is reproducible from its inputs plus the revision
  in its name; `geometry.parquet` alone is far too large to commit.
