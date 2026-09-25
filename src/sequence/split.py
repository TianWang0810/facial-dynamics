"""
Train/val/test splitting, and the honesty constraints around it.

Two rules, both about leakage:

Overlapping windows must not cross a split. Windows are cut with 50% overlap, so
two adjacent windows share half their frames. Putting one in train and the other
in val means validation is scored on frames the model trained on. Splitting is
therefore done at the CLIP level and windows follow their clip -- never by
shuffling windows.

A speaker must not appear on both sides. Generalisation to a new face is the
claim this project eventually wants to make, and a split that keeps the same
speaker in train and val cannot support it: the model can memorise that face.
When speaker_id is available, splitting is speaker-disjoint.

When speaker_id is NOT available -- the default, because the directory layout
usually does not encode it -- make_split() still produces a usable clip-disjoint
split, but labels it `clip_disjoint` and records a limitation string. Describing
that split as identity-disjoint would be a false claim about what was measured,
and is the single most likely way for this pipeline to produce a result that
looks better than it is.

Dataset-level normalisation statistics are fitted on the TRAIN split only.
fit_normalization() enforces that by taking the split it is fitted on; computing
mean and variance over everything leaks the evaluation distribution into the
inputs, which is subtle enough to survive review.
"""
import numpy as np

SPLIT_NAMES = ("train", "val", "test")

KIND_SPEAKER_DISJOINT = "speaker_disjoint"
KIND_CLIP_DISJOINT = "clip_disjoint"

CLIP_DISJOINT_LIMITATION = (
    "speaker_id was not available, so this split is clip-disjoint only. The same person may appear "
    "in more than one split. Results from it must NOT be described as identity-disjoint or as "
    "evidence of cross-identity generalisation."
)


def make_split(clip_ids, speaker_ids=None, ratios=(0.7, 0.15, 0.15), seed: int = 0) -> dict:
    """Assign clips to train/val/test, grouping by speaker when one is known.

    Grouping is by speaker when every clip has one, and by clip otherwise. Groups
    are shuffled with a fixed seed and dealt out to meet the ratios by clip count,
    so the split is reproducible from the seed alone.
    """
    clip_ids = list(clip_ids)
    if not clip_ids:
        return {"kind": KIND_CLIP_DISJOINT, "assignments": {}, "limitation": CLIP_DISJOINT_LIMITATION}

    have_speakers = speaker_ids is not None and all(s is not None for s in speaker_ids)
    if have_speakers:
        groups = {}
        for clip, speaker in zip(clip_ids, speaker_ids):
            groups.setdefault(speaker, []).append(clip)
        kind = KIND_SPEAKER_DISJOINT
    else:
        groups = {clip: [clip] for clip in clip_ids}
        kind = KIND_CLIP_DISJOINT

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    total = len(clip_ids)
    targets = [int(round(total * ratio)) for ratio in ratios]
    targets[0] = total - targets[1] - targets[2]

    assignments, counts = {}, [0, 0, 0]
    for key in keys:
        # Whichever split is furthest below its target takes the next group, so a
        # large speaker cannot overshoot one split and starve another.
        deficits = [targets[i] - counts[i] for i in range(3)]
        index = int(np.argmax(deficits))
        for clip in groups[key]:
            assignments[clip] = SPLIT_NAMES[index]
        counts[index] += len(groups[key])

    result = {
        "kind": kind,
        "seed": seed,
        "ratios": list(ratios),
        "assignments": assignments,
        "n_groups": len(keys),
        "counts": {name: counts[i] for i, name in enumerate(SPLIT_NAMES)},
        "limitation": None if have_speakers else CLIP_DISJOINT_LIMITATION,
    }
    return result


def verify_no_leakage(split: dict, clip_to_speaker: dict = None) -> dict:
    """Check that no clip, and no speaker when known, spans two splits."""
    assignments = split["assignments"]
    by_split = {name: {c for c, s in assignments.items() if s == name} for name in SPLIT_NAMES}

    overlapping_clips = set()
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            overlapping_clips |= by_split[a] & by_split[b]

    overlapping_speakers = set()
    if clip_to_speaker and all(v is not None for v in clip_to_speaker.values()):
        speakers = {}
        for clip, name in assignments.items():
            speakers.setdefault(clip_to_speaker[clip], set()).add(name)
        overlapping_speakers = {s for s, names in speakers.items() if len(names) > 1}

    return {
        "clip_leakage": sorted(overlapping_clips),
        "speaker_leakage": sorted(overlapping_speakers),
        "is_clean": not overlapping_clips and not overlapping_speakers,
        "speaker_check_performed": bool(clip_to_speaker and all(v is not None for v in clip_to_speaker.values())),
    }


def fit_normalization(features: np.ndarray, window_clip_ids, split: dict,
                      on: str = "train") -> dict:
    """Per-dimension mean and std, fitted on one split only.

    Takes the split rather than an already-filtered array so the restriction is
    part of the call and cannot be forgotten. Fitting over everything leaks the
    evaluation distribution into the inputs.
    """
    assignments = split["assignments"]
    mask = np.array([assignments.get(clip) == on for clip in window_clip_ids], dtype=bool)
    if not mask.any():
        raise ValueError("no windows belong to split %r; cannot fit normalisation" % on)

    subset = np.asarray(features)[mask]
    flat = subset.reshape(-1, subset.shape[-1])
    return {
        "fitted_on": on,
        "n_windows": int(mask.sum()),
        "mean": flat.mean(axis=0).tolist(),
        "std": np.maximum(flat.std(axis=0), 1e-8).tolist(),
    }
