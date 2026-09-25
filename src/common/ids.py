"""
Stable clip identity.

A clip is identified by its path RELATIVE to the input root, with separators
replaced, rather than by its basename. Two files named clip01.mp4 in different
subdirectories are different clips; keying on the basename silently merged them,
and because the Observation and Geometry layers built their keys independently,
the merge could differ between stages.

The relative path is also the only identifier available that survives a move of
the dataset root, which is what makes it usable in a manifest that outlives the
machine it was produced on.

source_video_id and speaker_id are recorded when the directory layout makes them
recoverable, and left null otherwise. A null speaker_id is a stated limitation --
see infer_grouping() -- and must never be filled in with a guess, because a wrong
speaker id silently converts a speaker-disjoint split into a leaky one.
"""
import os

SEPARATOR = "__"


def clip_id(mp4_path: str, input_root: str) -> str:
    """Stable identifier: the path relative to input_root, flattened.

    data/s01/take2.mp4 under root data/ becomes s01__take2.
    """
    relative = os.path.relpath(os.path.abspath(mp4_path), os.path.abspath(input_root))
    stem, _ = os.path.splitext(relative)
    parts = [part for part in stem.replace("\\", "/").split("/") if part not in ("", ".")]
    parts = [part for part in parts if part != ".."]
    return SEPARATOR.join(parts) if parts else stem


def relative_path(mp4_path: str, input_root: str) -> str:
    """The clip's path relative to the input root, kept verbatim for traceability."""
    return os.path.relpath(os.path.abspath(mp4_path), os.path.abspath(input_root)).replace("\\", "/")


def infer_grouping(mp4_path: str, input_root: str, speaker_from: str = None) -> dict:
    """Identity metadata for a clip, with unknowns left null.

    speaker_from selects how a speaker is identified:
        None          -- unknown; speaker_id is null and any split built on this
                         data is clip-disjoint only, NOT identity-disjoint
        "parent_dir"  -- the immediate parent directory names the speaker, the
                         usual layout for per-subject folders
        "top_dir"     -- the first path component names the speaker

    source_video_id is the relative path without the extension, i.e. the clip's
    provenance before any windowing, so windows from one source video can be kept
    on one side of a split.
    """
    relative = relative_path(mp4_path, input_root)
    stem, _ = os.path.splitext(relative)
    parts = [part for part in stem.split("/") if part]

    speaker = None
    if speaker_from == "parent_dir" and len(parts) >= 2:
        speaker = parts[-2]
    elif speaker_from == "top_dir" and len(parts) >= 2:
        speaker = parts[0]

    return {
        "clip_id": clip_id(mp4_path, input_root),
        "relative_path": relative,
        "source_video_id": stem,
        "speaker_id": speaker,
        "speaker_id_source": speaker_from if speaker else None,
    }
