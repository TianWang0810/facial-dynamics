"""
Timeline validation for the Observation layer.

Decoding and the timestamp policy live in src/observation/decode.py, which the
Geometry layer uses as well -- one decoder, one definition of what a frame is and
what its timestamp is. This module adds only the clip-level verdict.

The complete per-frame array is published in video_metadata.json as pts_sec
(design doc section 1.1). Entries are null where the container reported no
timestamp; the frame is still counted and still occupies its position, because
dropping it would shift every later index while leaving the array length
plausible.

Reusable core logic; CLI entrypoint is in scripts/run_observation.py
"""
from src.observation.decode import read_timeline


def analyze_video(mp4_path: str) -> dict:
    """Decode timing and return the clip's timeline plus a verdict.

    quality_label is "usable" only when the timeline is strictly monotonic with
    no missing timestamps and no oversized gaps. Anything else is needs_review:
    a timing defect is a hard error about the data, distinct from a visual
    quality threshold, and the two are kept separate on purpose.
    """
    timeline = read_timeline(mp4_path)
    timeline["file"] = mp4_path.split("/")[-1]
    timeline["path"] = mp4_path
    timeline["quality_label"] = "usable" if not timeline["issues"] else "needs_review"
    return timeline
