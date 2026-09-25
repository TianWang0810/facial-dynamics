"""
Run directory layout and artifact naming.

Single source of truth for where every pipeline artifact lands and what it is
called. The orchestrator, each stage and anything reading the results derive
their paths from here rather than joining strings themselves -- the same reason
schemas/geometry.schema.json owns the feature layout.

Directory shape
---------------
    <output_root>/<run_id>/
        run_manifest.json          what ran, from what, with which settings
        00_observation/
            video_metadata.json
            quality.json
        01_geometry/
            geometry.parquet
            geometry_metadata.json
            geometry_quality.json
        02_features/
            features.npz
            features_manifest.json
        03_dynamics/
            dynamics_report.json
        04_audio/
            audio.npz              conditioning sidecar, row-aligned with features.npz
            audio_manifest.json
        logs/
            00_observation.log ... 04_audio.log

Two naming rules, pulling in opposite directions on purpose:

The run directory is UNIQUE. Its name carries the dataset tag, a UTC timestamp
and the code revision, so two runs never overwrite each other and a result can
be traced back to the commit that produced it.

The artifacts inside are CANONICAL. geometry.parquet is always called
geometry.parquet, at a fixed depth. Downstream code and humans can therefore
predict a path without parsing a run name, and a stage never has to guess what
the previous stage called its output. Uniqueness lives in exactly one place.

The numeric prefixes (00_, 01_) make execution order visible in a directory
listing and keep the folders sorted in the order they were produced.
"""
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

# Stage order is meaningful: each consumes the previous stage's artifacts.
STAGE_OBSERVATION = "observation"
STAGE_GEOMETRY = "geometry"
STAGE_FEATURES = "features"
STAGE_DYNAMICS = "dynamics"
# Audio runs after Dynamics because it consumes features.npz window times; it is
# numbered 04 rather than inserted earlier so the canonical 00-03 paths of every
# existing run stay valid.
STAGE_AUDIO = "audio"

STAGES = (STAGE_OBSERVATION, STAGE_GEOMETRY, STAGE_FEATURES, STAGE_DYNAMICS, STAGE_AUDIO)

STAGE_DIRS = {
    STAGE_OBSERVATION: "00_observation",
    STAGE_GEOMETRY: "01_geometry",
    STAGE_FEATURES: "02_features",
    STAGE_DYNAMICS: "03_dynamics",
    STAGE_AUDIO: "04_audio",
}

# Artifacts each stage must have produced to count as complete. Used for --resume:
# a stage whose outputs all exist is skipped, so a run can be restarted after a
# failure without redoing the expensive extraction.
STAGE_ARTIFACTS = {
    STAGE_OBSERVATION: ("video_metadata.json", "quality.json"),
    STAGE_GEOMETRY: ("geometry.parquet", "geometry_metadata.json", "geometry_quality.json"),
    STAGE_FEATURES: ("features.npz", "features_manifest.json"),
    STAGE_DYNAMICS: ("dynamics_report.json",),
    STAGE_AUDIO: ("audio.npz", "audio_manifest.json"),
}

RUN_MANIFEST = "run_manifest.json"
LOG_DIR = "logs"

# Run ids must survive being used as a directory name on any platform and as a
# key in a report, so the dataset tag is restricted rather than trusted.
_TAG_ALLOWED = re.compile(r"[^A-Za-z0-9._-]+")
_TAG_MAX_LEN = 48
RUN_ID_PATTERN = re.compile(r"^(?P<tag>.+)__(?P<timestamp>\d{8}T\d{6}Z)__(?P<revision>[0-9a-f]{7,40}|nogit)$")


def sanitize_tag(tag: str) -> str:
    """Reduce a free-form dataset name to something safe as a path component."""
    cleaned = _TAG_ALLOWED.sub("-", str(tag)).strip("-._")
    cleaned = cleaned[:_TAG_MAX_LEN].strip("-._")
    return cleaned or "run"


def git_revision(repo_root: str) -> str:
    """Short commit sha of the code being run, or 'nogit' outside a checkout.

    Recorded in the run id because a result that cannot be traced to a revision
    cannot be reproduced, and this pipeline's behaviour is defined by its code.
    """
    try:
        completed = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "--short=7", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "nogit"


def make_run_id(tag: str, repo_root: str, timestamp: datetime = None) -> str:
    """<tag>__<UTC timestamp>__<git sha>, e.g. talkvid-sample__20260913T181500Z__5f26025.

    UTC because runs get compared across machines in different timezones, and a
    sortable basic-format timestamp so a directory listing is chronological.
    """
    moment = timestamp or datetime.now(timezone.utc)
    return "%s__%s__%s" % (
        sanitize_tag(tag), moment.strftime("%Y%m%dT%H%M%SZ"), git_revision(repo_root),
    )


def parse_run_id(run_id: str) -> dict:
    """Split a run id back into its parts; raises if it was not produced by make_run_id."""
    match = RUN_ID_PATTERN.match(run_id)
    if not match:
        raise ValueError("%r is not a run id of the form <tag>__<timestamp>__<revision>" % run_id)
    return match.groupdict()


@dataclass(frozen=True)
class RunLayout:
    """Every path in one run, derived -- never assembled ad hoc by a caller."""
    root: str
    run_id: str

    @property
    def run_dir(self) -> str:
        return os.path.join(self.root, self.run_id)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.run_dir, RUN_MANIFEST)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.run_dir, LOG_DIR)

    def stage_dir(self, stage: str) -> str:
        if stage not in STAGE_DIRS:
            raise KeyError("unknown stage %r; known stages are %s" % (stage, list(STAGES)))
        return os.path.join(self.run_dir, STAGE_DIRS[stage])

    def artifact(self, stage: str, filename: str) -> str:
        """Path to one named artifact of a stage.

        The filename is checked against that stage's declared artifacts, so a
        typo fails here rather than producing a file nobody downstream looks for.
        """
        if filename not in STAGE_ARTIFACTS[stage]:
            raise KeyError(
                "%r is not an artifact of stage %r; declared artifacts are %s"
                % (filename, stage, list(STAGE_ARTIFACTS[stage]))
            )
        return os.path.join(self.stage_dir(stage), filename)

    def log_path(self, stage: str) -> str:
        return os.path.join(self.log_dir, "%s.log" % STAGE_DIRS[stage])

    def stage_artifacts(self, stage: str) -> tuple:
        return tuple(self.artifact(stage, name) for name in STAGE_ARTIFACTS[stage])

    def is_complete(self, stage: str) -> bool:
        """True when every declared artifact of the stage exists and is non-empty.

        Zero-length files count as incomplete: a crash mid-write leaves exactly
        that, and resuming onto a truncated artifact is worse than redoing it.
        """
        return all(
            os.path.exists(path) and os.path.getsize(path) > 0
            for path in self.stage_artifacts(stage)
        )

    def missing_artifacts(self, stage: str) -> list:
        return [
            path for path in self.stage_artifacts(stage)
            if not (os.path.exists(path) and os.path.getsize(path) > 0)
        ]

    def create_directories(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        for stage in STAGES:
            os.makedirs(self.stage_dir(stage), exist_ok=True)


def latest_run(root: str, tag: str = None) -> str:
    """Most recent run id under root, optionally filtered by dataset tag.

    Sorting is lexicographic, which is chronological here because the timestamp
    is fixed-width UTC basic format -- that is why it was chosen.
    """
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        try:
            parts = parse_run_id(name)
        except ValueError:
            continue
        if tag and parts["tag"] != sanitize_tag(tag):
            continue
        candidates.append((parts["timestamp"], name))
    return max(candidates)[1] if candidates else None
