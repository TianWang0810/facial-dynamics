"""
L1 atomic text layer: a deterministic codec between one window of the TARGET
tensor and structured, per-channel segment text.

    encode: (window_frames, TOTAL_DIM) + the four mask signals -> text dict
    decode: text dict -> (window_frames, TOTAL_DIM) + the four mask signals

Every threshold comes from schemas/l1_rules.json, which is loaded and validated
at import exactly as src/schema/feature_vector.py loads the feature schema.
Nothing here is fitted to data; the same window always yields the same text.
That is what lets L1 serve as the near-lossless baseline the L2/L3 layers are
measured against (docs/plans/multilayer_semantic_representation.md section 2).

Text channels
-------------
Each schema channel fans out into named text channels:

    blendshape            one per blendshape name (_neutral excluded)
    headpose_translation  head_tx, head_ty, head_tz
    headpose_rotation_6d  head_rotation, three axes quantized JOINTLY
    gaze                  gaze_pitch, gaze_yaw

A text channel's value at a frame is quantized to a level. For signed channels
the level carries the sign ("+slight", "-moderate"). Rotation is the exception to
element-wise handling: the 6-vector is orthonormalised, converted to a
sign-aligned quaternion, and expressed as a rotation vector in degrees relative
to a per-window reference rotation stored in the header (log map at the
reference). Each rotation-vector component is quantized, the three levels form
one tuple, and decode goes back through the exponential map. The 6-vector
itself is never binned or averaged element-wise, and no Euler angle appears.

Segmentation
------------
Levels are cut into maximal runs, then runs shorter than min_segment_seconds are
merged into a neighbour by a fixed rule (see _merge_short_runs). The first step
is exactly lossless with respect to per-frame levels; only the merge loses
information, which scripts/l1_report.py measures.

Masks
-----
A frame is described in a schema channel's text iff loss_weight() is > 0 there.
Padding, gap-filled and unobserved frames are recorded only as header intervals,
never as channel text, and decode restores all four mask signals from them.
"""
import json
import os

import numpy as np

from src.schema.feature_vector import CHANNELS, SCHEMA_VERSION, TOTAL_DIM, channel
from src.sequence.contract import loss_weight, neutral_vector

RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "schemas", "l1_rules.json",
)

ROTATION_CHANNEL = "headpose_rotation_6d"
# How far a time may sit from a grid point, in frames, before decode refuses it
# rather than rounding it onto a frame the author did not mean. Must exceed the
# rounding the text itself applies (0.5e-4 s at 4 decimals is 0.0015 frames at
# 30 Hz) and stay far below half a frame.
TIME_TOLERANCE_FRAMES = 0.05


def _load_rules(path: str = RULES_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate_quantizer(name: str, q: dict, n_levels: int) -> None:
    edges, reps = q["edges"], q["representatives"]
    if len(edges) != n_levels - 1 or len(reps) != n_levels:
        raise ValueError("quantizer %r needs %d edges and %d representatives" % (name, n_levels - 1, n_levels))
    if edges[0] <= 0 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("quantizer %r edges must be positive and strictly increasing: %s" % (name, edges))
    bounds = [0.0] + list(edges) + [np.inf]
    for index, rep in enumerate(reps):
        if not bounds[index] <= rep < bounds[index + 1]:
            raise ValueError("quantizer %r representative %s lies outside its bin [%s, %s)"
                             % (name, rep, bounds[index], bounds[index + 1]))


def _validate(rules: dict) -> None:
    n_levels = len(rules["levels"])
    for name, q in rules["quantizers"].items():
        for unit, sub in (q["by_unit"].items() if "by_unit" in q else [(None, q)]):
            _validate_quantizer(name if unit is None else "%s[%s]" % (name, unit), sub, n_levels)
    missing = [spec.name for spec in CHANNELS if spec.name not in rules["channels"]]
    if missing:
        raise ValueError("l1_rules.json has no text mapping for enabled schema channels %s" % missing)
    for spec in CHANNELS:
        mapping = rules["channels"][spec.name]
        if "text_names" in mapping and len(mapping["text_names"]) != spec.dim:
            raise ValueError("channel %r has dim %d but %d text names"
                             % (spec.name, spec.dim, len(mapping["text_names"])))


RULES = _load_rules()
_validate(RULES)
RULES_VERSION = RULES["rules_version"]
LEVELS = tuple(RULES["levels"])
TIME_DECIMALS = int(RULES["time"]["decimals"])
MIN_SEGMENT_SECONDS = float(RULES["segmentation"]["min_segment_seconds"])


def quantizer(name: str, gaze_unit: str = None) -> dict:
    """The quantizer for a schema channel, resolving gaze by backend unit."""
    q = RULES["quantizers"][RULES["channels"][name]["quantizer"]]
    if "by_unit" in q:
        if gaze_unit not in q["by_unit"]:
            raise ValueError("no L1 %s quantizer for unit %r; known units are %s"
                             % (name, gaze_unit, sorted(q["by_unit"])))
        q = q["by_unit"][gaze_unit]
    return q


# ---------------------------------------------------------------------------
# Scalar quantization. A level is an int: 0..n-1 unsigned, -(n-1)..(n-1) signed.

def quantize(values, q: dict) -> np.ndarray:
    """Values -> integer levels. A value exactly on an edge goes to the upper bin.

    Non-finite input is refused: the target tensor is sanitized upstream, so a
    NaN here is a bug, and searchsorted would silently file it as 'strong'. An
    unsigned channel is clipped at 0 so a stray negative cannot be read as its
    magnitude.
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("cannot quantize non-finite values; sanitize the tensor first")
    if not q["signed"]:
        values = np.maximum(values, 0.0)
    magnitude = np.searchsorted(np.asarray(q["edges"]), np.abs(values), side="right")
    return (np.sign(values) * magnitude).astype(np.int64) if q["signed"] else magnitude.astype(np.int64)


def dequantize(levels, q: dict) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.int64)
    reps = np.asarray(q["representatives"], dtype=np.float64)
    return np.sign(levels) * reps[np.abs(levels)] if q["signed"] else reps[levels]


def level_label(level: int, q: dict) -> str:
    if level == 0 or not q["signed"]:
        return LEVELS[abs(int(level))]
    return ("+" if level > 0 else "-") + LEVELS[abs(int(level))]


def parse_label(label: str, q: dict) -> int:
    """Inverse of level_label(). Strict, because the label is what a person edits:
    a signed channel's non-zero level must carry its sign, and 'none' never does."""
    signed_form = label[:1] in ("+", "-")
    name = label[1:] if signed_form else label
    if name not in LEVELS:
        raise ValueError("unknown L1 level %r; levels are %s" % (label, list(LEVELS)))
    magnitude = LEVELS.index(name)
    if signed_form != bool(q["signed"] and magnitude):
        raise ValueError("level %r: %s" % (label, "a signed channel's non-zero level needs a +/- sign"
                                           if q["signed"] and magnitude else "this level takes no sign"))
    return (-1 if label[0] == "-" else 1) * magnitude if signed_form else magnitude


# ---------------------------------------------------------------------------
# Rotation: 6D <-> rotation vector (degrees), always through the manifold.
#
# Every rotation in the text is a rotation vector relative to a reference R_ref:
# R = R_ref @ exp(v). The reference is a header value (see rotation_reference),
# so the levels describe head MOTION about the window's own pose rather than a
# static camera-relative posture -- the same reason translation and gaze are
# baseline-relative in the target tensor.

def _matrix_from_rotvec_deg(rotvec_deg) -> np.ndarray:
    from src.sequence.resample import _matrix_from_quaternion

    vector = np.radians(np.asarray(rotvec_deg, dtype=np.float64).reshape(3))
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    return _matrix_from_quaternion(np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * vector / angle]))


def _rotvec_deg_from_quaternion(q) -> np.ndarray:
    """Log map. q is put on the w >= 0 hemisphere, which selects the angle in
    [0, 180] degrees and makes q and -q (one rotation) give the same vector."""
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    sin_half = np.linalg.norm(q[1:])
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, q[0])
    return np.degrees(q[1:] / sin_half * angle)


def _quaternions(r6, reference_deg=None) -> list:
    """Sign-aligned unit quaternions of R_ref^T @ R for each 6-vector."""
    from src.geometry.headpose import rotation_6d_to_matrix
    from src.sequence.resample import _quaternion_from_matrix

    inverse_reference = _matrix_from_rotvec_deg(reference_deg if reference_deg is not None else np.zeros(3)).T
    out = []
    for vector in np.asarray(r6, dtype=np.float64).reshape(-1, 6):
        q = _quaternion_from_matrix(inverse_reference @ rotation_6d_to_matrix(vector))
        if out and np.dot(q, out[0]) < 0.0:
            q = -q
        out.append(q)
    return out


def rotvec_deg_from_r6(r6, reference_deg=None) -> np.ndarray:
    """(T, 6) -> (T, 3): rotation vector (degrees) of each rotation relative to the reference."""
    quaternions = _quaternions(r6, reference_deg)
    return np.array([_rotvec_deg_from_quaternion(q) for q in quaternions]).reshape(-1, 3)


def r6_from_rotvec_deg(rotvec_deg, reference_deg=None) -> np.ndarray:
    """(T, 3) rotation vectors (degrees) about the reference -> (T, 6), via the exponential map."""
    from src.geometry.headpose import rotation_matrix_to_6d

    reference = _matrix_from_rotvec_deg(reference_deg if reference_deg is not None else np.zeros(3))
    rotvec = np.asarray(rotvec_deg, dtype=np.float64).reshape(-1, 3)
    return np.stack([rotation_matrix_to_6d(reference @ _matrix_from_rotvec_deg(v)) for v in rotvec]).astype(np.float64)


def mean_rotvec_deg(r6, reference_deg=None) -> np.ndarray:
    """Mean rotation (relative to the reference) as a rotation vector.

    The normalised mean of hemisphere-aligned quaternions is the chordal L2 mean
    on the rotation manifold -- the multi-sample form of the nlerp used by
    src/sequence/resample.py -- rather than an element-wise mean of 6-vectors.
    """
    mean = np.mean(_quaternions(r6, reference_deg), axis=0)
    return _rotvec_deg_from_quaternion(mean / max(np.linalg.norm(mean), 1e-12))


def rotation_reference(r6, described) -> np.ndarray:
    """The window's reference rotation as declared by the rules, rounded as the text stores it.

    Rounding happens HERE, before any level is computed, so encode quantizes
    against exactly the reference decode will read back.
    """
    reference = RULES["channels"][ROTATION_CHANNEL]["reference"]
    described = np.asarray(described, dtype=bool)
    if reference["method"] == "identity" or not described.any():
        return np.zeros(3)
    if reference["method"] != "window_mean":
        raise ValueError("unknown rotation reference method %r" % reference["method"])
    return np.round(mean_rotvec_deg(np.asarray(r6)[described]), int(reference["decimals"]))


# ---------------------------------------------------------------------------
# Segmentation over integer level rows.

def _runs(levels: np.ndarray, described: np.ndarray) -> list:
    """Maximal runs [start, stop) of identical level rows inside described stretches.

    Each run carries a block id: runs in the same block are temporally adjacent,
    runs in different blocks are separated by undescribed frames and never merge.
    """
    runs, block = [], -1
    previous = None
    for frame in np.flatnonzero(described):
        row = tuple(levels[frame])
        if previous is None or frame != previous + 1:
            block += 1
            runs.append([int(frame), int(frame) + 1, row, block])
        elif row == runs[-1][2]:
            runs[-1][1] = int(frame) + 1
        else:
            runs.append([int(frame), int(frame) + 1, row, block])
        previous = frame
    return runs


def _merge_short_runs(runs: list, levels: np.ndarray, min_frames: int) -> list:
    """Relabel runs shorter than min_frames with a neighbour's level.

    A run may only take a neighbour's level if that level is within ONE step of
    every frame's own quantized level, on every axis. Flicker across a bin edge
    is a one-step alternation and is absorbed; a short excursion of two or more
    levels is a real event (a blink onset, a brief jaw drop) and is kept even
    though it is short. This bounds what segmentation can add: no frame is ever
    decoded more than one level away from its per-frame quantization.

    Deterministic by construction: always the shortest mergeable run first
    (earliest on a tie), into the eligible neighbour with the smaller summed
    level distance, then the longer neighbour, then the left one.
    """
    runs = [list(run) for run in runs]

    def eligible(index, j):
        start, stop = runs[index][0], runs[index][1]
        return int(np.max(np.abs(levels[start:stop] - np.asarray(runs[j][2])))) <= 1

    while True:
        candidates = []
        for index, (start, stop, row, block) in enumerate(runs):
            if stop - start >= min_frames:
                continue
            neighbours = [j for j in (index - 1, index + 1)
                          if 0 <= j < len(runs) and runs[j][3] == block and eligible(index, j)]
            if neighbours:
                candidates.append((stop - start, start, index, neighbours))
        if not candidates:
            return runs
        _, _, index, neighbours = min(candidates)

        def preference(j):
            distance = sum(abs(a - b) for a, b in zip(runs[j][2], runs[index][2]))
            return (distance, -(runs[j][1] - runs[j][0]), 0 if j < index else 1)

        target = min(neighbours, key=preference)
        runs[index][2] = runs[target][2]
        merged = []
        for run in runs:
            if merged and merged[-1][3] == run[3] and merged[-1][2] == run[2] and merged[-1][1] == run[0]:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        runs = merged


def segment(levels, described, min_frames: int) -> list:
    """(start, stop, level_row) segments of an (T, k) integer level array."""
    levels = np.asarray(levels, dtype=np.int64)
    if levels.ndim == 1:
        levels = levels[:, None]
    runs = _runs(levels, np.asarray(described, dtype=bool))
    if min_frames > 1:
        runs = _merge_short_runs(runs, levels, min_frames)
    return [(start, stop, row) for start, stop, row, _ in runs]


def min_segment_frames(hz: float, min_segment_seconds: float = None) -> int:
    seconds = MIN_SEGMENT_SECONDS if min_segment_seconds is None else float(min_segment_seconds)
    return max(1, int(np.floor(seconds * hz + 0.5)))


# ---------------------------------------------------------------------------
# Channel naming and masks.

def blendshape_names_from_metadata(metadata: dict) -> list:
    """The 52 blendshape column names persisted by the Geometry layer."""
    names = metadata.get("tracks", {}).get("A_landmarks_blendshapes", {}).get("blendshape_names")
    if not names:
        raise ValueError("geometry_metadata.json has no blendshape_names; it predates the field. "
                         "Re-run the geometry stage (scripts/run_pipeline.py --stages geometry features --force).")
    if len(names) != channel("blendshape").dim:
        raise ValueError("expected %d blendshape names, got %d" % (channel("blendshape").dim, len(names)))
    return list(names)


def gaze_unit_from_metadata(metadata: dict) -> str:
    return metadata["tracks"]["C_gaze"]["unit"]


def text_channels(blendshape_names: list) -> list:
    """(text name, schema channel, dimension offsets) for every described text channel."""
    out = []
    for spec in CHANNELS:
        mapping = RULES["channels"][spec.name]
        if spec.name == "blendshape":
            for offset, name in enumerate(blendshape_names):
                if name not in mapping["exclude"]:
                    out.append((name, spec.name, (offset,)))
        elif spec.name == ROTATION_CHANNEL:
            out.append((mapping["text_name"], spec.name, tuple(range(spec.dim))))
        else:
            for offset, name in enumerate(mapping["text_names"]):
                out.append((name, spec.name, (offset,)))
    return out


def describable(observed, quality, is_gap_filled, is_padding) -> np.ndarray:
    """(T, n_channels) bool: loss_weight() > 0, the frames L1 describes."""
    return loss_weight(observed, quality, is_gap_filled, is_padding) > 0.0


def _time(frame: int, hz: float) -> float:
    return round(frame / hz, TIME_DECIMALS)


def _frame(t: float, hz: float) -> int:
    exact = float(t) * hz
    frame = int(np.floor(exact + 0.5))
    if abs(exact - frame) > TIME_TOLERANCE_FRAMES:
        raise ValueError("time %s does not fall on the %.3f Hz grid" % (t, hz))
    return frame


def _intervals(mask, hz: float) -> list:
    mask = np.asarray(mask, dtype=bool)
    out, start = [], None
    for frame, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = frame
        elif not flag and start is not None:
            out.append([_time(start, hz), _time(frame, hz)])
            start = None
    return out


def _mask_from_intervals(intervals, n_frames: int, hz: float) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in intervals:
        mask[_frame(start, hz):_frame(stop, hz)] = True
    return mask


# ---------------------------------------------------------------------------
# Encode / decode one window.

def _consistent(row, hint, q) -> list:
    """Clip each axis of a segment's value_hint into its level's bin.

    A merged segment's mean can fall outside the bin of the level it was merged
    into (a blink peak labelled 'strong' whose absorbed moderate shoulders pull
    the mean to 0.48). The LEVEL is kept -- it carries the meaning, and here the
    peak really is strong -- and the hint is clipped to the nearest value the
    level admits. Unedited text therefore always has its hint inside its level's
    bin, which is what lets the refined decode tell an unedited segment from an
    edited level. Relabelling to the mean's level was measured as equally
    accurate but turned blink peaks into 'moderate', so it was rejected.
    """
    out = []
    for level, value in zip(row, hint):
        low, high = bin_bounds(level, q)
        out.append(float(np.clip(value, low, high)))
    return out


def channel_levels(features, described, blendshape_names: list, gaze_unit: str):
    """Per text channel: (text name, schema channel, quantizer, described mask,
    raw block, quantized-space values, per-frame levels, rotation reference).

    The single place per-frame levels are computed, shared by encode_window() and
    by the report's one-level-guarantee check so the two cannot disagree.
    """
    features = np.asarray(features, dtype=np.float64)
    channel_index = {spec.name: k for k, spec in enumerate(CHANNELS)}
    for text_name, schema_name, offsets in text_channels(blendshape_names):
        spec = channel(schema_name)
        q = quantizer(schema_name, gaze_unit)
        mask = np.asarray(described, dtype=bool)[:, channel_index[schema_name]]
        block = features[:, spec.start:spec.end]
        reference = None
        if schema_name == ROTATION_CHANNEL:
            reference = rotation_reference(block, mask)
            values = rotvec_deg_from_r6(block, reference)
        else:
            values = block[:, list(offsets)]
        yield text_name, schema_name, q, mask, block, values, quantize(values, q), reference


def encode_window(features, quality, observed, is_gap_filled, is_padding, hz: float,
                  blendshape_names: list, gaze_unit: str, min_segment_seconds: float = None,
                  header: dict = None) -> dict:
    """One (T, TOTAL_DIM) target window -> L1 text dict."""
    features = np.asarray(features, dtype=np.float64)
    n_frames = features.shape[0]
    if features.shape[1] != TOTAL_DIM:
        raise ValueError("window must be (T, %d), got %s" % (TOTAL_DIM, features.shape))
    described = describable(observed, quality, is_gap_filled, is_padding)
    min_frames = min_segment_frames(hz, min_segment_seconds)

    channels, references = {}, {}
    for text_name, schema_name, q, mask, block, values, levels, reference in \
            channel_levels(features, described, blendshape_names, gaze_unit):
        if schema_name == ROTATION_CHANNEL:
            axes = RULES["channels"][schema_name]["axes"]
            references[text_name] = {axis: float(reference[k]) for k, axis in enumerate(axes)}

        segments = []
        for start, stop, row in segment(levels, mask, min_frames):
            entry = {"t": [_time(start, hz), _time(stop, hz)]}
            if schema_name == ROTATION_CHANNEL:
                hint = [round(float(v), 3) for v in mean_rotvec_deg(block[start:stop], reference)]
            else:
                hint = [round(float(values[start:stop, 0].mean()), 4)]
            hint = _consistent(row, hint, q)
            if schema_name == ROTATION_CHANNEL:
                entry["level"] = {axis: level_label(row[k], q) for k, axis in enumerate(axes)}
                entry["value_hint"] = {axis: hint[k] for k, axis in enumerate(axes)}
            else:
                entry["level"] = level_label(row[0], q)
                entry["value_hint"] = hint[0]
            segments.append(entry)
        channels[text_name] = segments

    out = dict(header or {})
    out.update({
        "hz": float(hz),
        "n_frames": int(n_frames),
        "reference": references,
        "masks": {
            "is_padding": _intervals(is_padding, hz),
            "is_gap_filled": _intervals(is_gap_filled, hz),
            "unobserved": _intervals(~np.asarray(observed, dtype=bool), hz),
        },
        "channels": channels,
    })
    return out


DECODE_SOURCES = ("level", "value_hint", "refined")


def bin_bounds(level: int, q: dict) -> tuple:
    """[low, high] of the values that quantize to this level (inclusive bounds, for clipping tests)."""
    edges = [0.0] + list(q["edges"]) + [np.inf]
    magnitude = abs(int(level))
    low, high = edges[magnitude], edges[magnitude + 1]
    if not q["signed"]:
        return low, high
    if magnitude == 0:
        return -edges[1], edges[1]
    return (low, high) if level > 0 else (-high, -low)


def _segment_value(level: int, hint: float, q: dict, source: str) -> float:
    """One segment value on one axis under the chosen decode source.

    refined: the recorded segment mean when it still lies inside the level's
    bin, otherwise the level's representative. An unedited segment therefore
    keeps its measured intensity, while an edited level -- whose old hint now
    sits in another bin -- decodes to the new level as intended. The level stays
    authoritative; the hint only refines within it.
    """
    if source == "level":
        return float(dequantize(level, q))
    if source == "value_hint":
        return float(hint)
    low, high = bin_bounds(level, q)
    return float(hint) if low <= hint <= high else float(dequantize(level, q))


def _whittaker(values: np.ndarray, covered: np.ndarray, lam: float) -> np.ndarray:
    """Whittaker smoother: argmin ||y - x||^2 + lam * ||D2 y||^2, per covered stretch.

    Deterministic, no free parameters beyond lam, and it never reaches across an
    uncovered frame, so a smoothed value can never borrow from padding, a gap
    or another segment id's absence. Stretches shorter than 3 frames have no
    second difference and are returned unchanged.
    """
    out = values.copy()
    frames = np.flatnonzero(covered)
    if frames.size == 0:
        return out
    breaks = np.flatnonzero(np.diff(frames) > 1) + 1
    for block in np.split(frames, breaks):
        n = block.size
        if n < 3:
            continue
        second = np.diff(np.eye(n), 2, axis=0)
        out[block] = np.linalg.solve(np.eye(n) + lam * second.T @ second, values[block])
    return out


def decode_window(text: dict, blendshape_names: list, gaze_unit: str, source: str = None,
                  smooth: bool = None) -> dict:
    """L1 text dict -> features (T, TOTAL_DIM), quality (T, n_channels), and masks.

    source and smooth default to the rules' decode block. With smooth on, each
    text channel's piecewise-constant values are passed through _whittaker() over
    its covered stretches -- for rotation on the rotation-vector components about
    the window reference, i.e. in the tangent space, before the exponential map,
    never on the 6-vector.
    """
    decode_rules = RULES["decode"]
    source = decode_rules["default_source"] if source is None else source
    smooth = decode_rules["smoothing"]["method"] != "none" if smooth is None else bool(smooth)
    if source not in DECODE_SOURCES:
        raise ValueError("source must be one of %s, got %r" % (DECODE_SOURCES, source))
    lam = float(decode_rules["smoothing"]["lambda"])
    hz, n_frames = float(text["hz"]), int(text["n_frames"])
    features = np.tile(neutral_vector(), (n_frames, 1))
    covered = np.zeros((n_frames, TOTAL_DIM), dtype=bool)

    blendshape = channel("blendshape")
    excluded = RULES["channels"]["blendshape"]
    for offset, name in enumerate(blendshape_names):
        if name in excluded["exclude"]:
            features[:, blendshape.start + offset] = excluded["excluded_fill"]
            covered[:, blendshape.start + offset] = True

    # The text is hand-editable, so decode is strict: a misspelled or missing
    # channel would otherwise silently zero the quality of its whole schema
    # channel, and an overlapping segment would silently overwrite another.
    expected = [name for name, _, _ in text_channels(blendshape_names)]
    if set(text["channels"]) != set(expected):
        raise ValueError("text channels do not match the rules: missing %s, unknown %s"
                         % (sorted(set(expected) - set(text["channels"])), sorted(set(text["channels"]) - set(expected))))

    for text_name, schema_name, offsets in text_channels(blendshape_names):
        spec = channel(schema_name)
        q = quantizer(schema_name, gaze_unit)
        axes = RULES["channels"][schema_name]["axes"] if schema_name == ROTATION_CHANNEL else None
        values = np.zeros((n_frames, len(axes) if axes else 1))
        written = np.zeros(n_frames, dtype=bool)
        for entry in text["channels"][text_name]:
            start, stop = _frame(entry["t"][0], hz), _frame(entry["t"][1], hz)
            if not 0 <= start < stop <= n_frames:
                raise ValueError("%s segment %s lies outside the %d-frame window" % (text_name, entry["t"], n_frames))
            if written[start:stop].any():
                raise ValueError("%s segment %s overlaps an earlier segment" % (text_name, entry["t"]))
            written[start:stop] = True
            labels = [entry["level"][a] for a in axes] if axes else [entry["level"]]
            hints = [entry["value_hint"][a] for a in axes] if axes else [entry["value_hint"]]
            values[start:stop] = [_segment_value(parse_label(label, q), float(hint), q, source)
                                  for label, hint in zip(labels, hints)]
        if smooth:
            values = _whittaker(values, written, lam)

        rows = np.flatnonzero(written)
        if axes:
            if text_name not in text.get("reference", {}):
                raise ValueError("window header has no reference for %s" % text_name)
            reference = np.array([text["reference"][text_name][a] for a in axes], dtype=np.float64)
            if rows.size:
                features[rows, spec.slice] = r6_from_rotvec_deg(values[rows], reference)
            covered[rows, spec.slice] = True
        else:
            column = spec.start + offsets[0]
            features[rows, column] = values[rows, 0]
            covered[rows, column] = True

    quality = np.stack([covered[:, spec.slice].all(axis=1) for spec in CHANNELS], axis=1).astype(np.float64)
    masks = text["masks"]
    is_padding = _mask_from_intervals(masks["is_padding"], n_frames, hz)
    is_gap_filled = _mask_from_intervals(masks["is_gap_filled"], n_frames, hz)
    observed = ~_mask_from_intervals(masks["unobserved"], n_frames, hz)
    return {"features": features, "quality": quality, "observed": observed,
            "is_gap_filled": is_gap_filled, "is_padding": is_padding}


def quantize_window(features, blendshape_names: list, gaze_unit: str, described=None) -> np.ndarray:
    """Per-frame quantize -> dequantize with no segmentation: the pure quantization baseline.

    described (T, n_channels) only matters for the rotation reference, which is
    the mean over described frames exactly as encode_window() computes it.
    """
    features = np.asarray(features, dtype=np.float64)
    if described is None:
        described = np.ones((features.shape[0], len(CHANNELS)), dtype=bool)
    rotation_index = [spec.name for spec in CHANNELS].index(ROTATION_CHANNEL)
    out = features.copy()
    blendshape = channel("blendshape")
    excluded = RULES["channels"]["blendshape"]
    for offset, name in enumerate(blendshape_names):
        if name in excluded["exclude"]:
            out[:, blendshape.start + offset] = excluded["excluded_fill"]
    for _, schema_name, offsets in text_channels(blendshape_names):
        spec = channel(schema_name)
        q = quantizer(schema_name, gaze_unit)
        if schema_name == ROTATION_CHANNEL:
            reference = rotation_reference(features[:, spec.slice], described[:, rotation_index])
            rotvec = dequantize(quantize(rotvec_deg_from_r6(features[:, spec.slice], reference), q), q)
            out[:, spec.slice] = r6_from_rotvec_deg(rotvec, reference)
        else:
            column = spec.start + offsets[0]
            out[:, column] = dequantize(quantize(features[:, column], q), q)
    return out


# ---------------------------------------------------------------------------
# Whole features.npz <-> one L1 document.

def encode_features(npz: dict, blendshape_names: list, gaze_unit: str,
                    min_segment_seconds: float = None, source: dict = None) -> dict:
    """Encode every window of a features.npz (as a dict of arrays) into one document."""
    if str(npz["schema_version"]) != RULES["applies_to_schema_version"]:
        raise ValueError("l1_rules.json targets schema %s but the tensor is schema %s"
                         % (RULES["applies_to_schema_version"], npz["schema_version"]))
    hz = float(npz["target_hz"])
    windows = []
    for w in range(npz["features"].shape[0]):
        header = {
            "window_id": "%s@%d" % (npz["clip_id"][w], int(npz["window_start"][w])),
            "clip_id": str(npz["clip_id"][w]),
            "window_start": int(npz["window_start"][w]),
            "segment_id": int(npz["segment_id"][w]),
        }
        windows.append(encode_window(
            npz["features"][w], npz["quality"][w], npz["observed"][w], npz["is_gap_filled"][w],
            npz["is_padding"][w], hz, blendshape_names, gaze_unit,
            min_segment_seconds=min_segment_seconds, header=header,
        ))
    return {
        "format": "l1_text",
        "rules_version": RULES_VERSION,
        "schema_version": SCHEMA_VERSION,
        "min_segment_seconds": MIN_SEGMENT_SECONDS if min_segment_seconds is None else float(min_segment_seconds),
        "gaze_unit": gaze_unit,
        "blendshape_names": list(blendshape_names),
        "target_hz": hz,
        "source": source or {},
        "windows": windows,
    }


def decode_document(document: dict, source: str = None, smooth: bool = None) -> dict:
    """Inverse of encode_features(): arrays shaped like features.npz."""
    if document.get("format") != "l1_text":
        raise ValueError("not an L1 text document")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("document is schema %s, this code is schema %s" % (document["schema_version"], SCHEMA_VERSION))
    decoded = [decode_window(w, document["blendshape_names"], document["gaze_unit"], source=source, smooth=smooth)
               for w in document["windows"]]
    stacked = {key: np.stack([d[key] for d in decoded]) for key in decoded[0]}
    stacked["features"] = stacked["features"].astype(np.float32)
    stacked["quality"] = stacked["quality"].astype(np.float32)
    stacked.update({
        "clip_id": np.array([w["clip_id"] for w in document["windows"]]),
        "window_start": np.array([w["window_start"] for w in document["windows"]], dtype=np.int32),
        "segment_id": np.array([w["segment_id"] for w in document["windows"]], dtype=np.int32),
        "channel_names": np.array([spec.name for spec in CHANNELS]),
        "schema_version": np.array(SCHEMA_VERSION),
        "target_hz": np.array(document["target_hz"]),
    })
    return stacked
