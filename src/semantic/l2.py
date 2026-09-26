"""
L2 action-phrase layer on top of a TRACK-level L1 text.

    tracks:  features.npz windows -> one track per (clip_id, segment_id). Windows of
             a segment are slices of one resampled series, so overlapping frames are
             identical; stitch_tracks() asserts that instead of assuming it.
    encode:  track-level L1 text -> L2 track: rest values, detected phrases (each with
             the exact L1 segments inside its span as `detail`), and the residual L1
             segments nothing explains.
    decode:  exact        residual + every phrase's detail -> the same L1 text, verbatim
             phrase_only  residual + a generated fill of each phrase's span (a fixed
                          template now; exemplar retrieval later, plan section 6)

Design: docs/plans/l2_knowledge_base_retrieval.md. Every threshold comes from
schemas/l2_vocabulary.json, loaded and validated at import like l1_rules.json.

Why a track-level L1 rather than the window texts: an action crosses windows
(browInnerUp runs exceed 1 s 40% of the time) and each window has its own rotation
reference. l1.encode_window() is length-agnostic, so the same codec, run once over
the whole track, gives one timeline with one reference and consistent masks.
"""
import json
import os

import numpy as np

from src.semantic import l1

VOCABULARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "schemas", "l2_vocabulary.json",
)
ROTATION_TEXT = l1.RULES["channels"][l1.ROTATION_CHANNEL]["text_name"]
ROTATION_AXES = tuple(l1.RULES["channels"][l1.ROTATION_CHANNEL]["axes"])


def _load_vocabulary(path: str = VOCABULARY_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate(vocabulary: dict) -> None:
    if vocabulary["applies_to_l1_rules_version"] != l1.RULES_VERSION:
        raise ValueError("l2_vocabulary.json targets L1 rules %s but l1_rules.json is %s"
                         % (vocabulary["applies_to_l1_rules_version"], l1.RULES_VERSION))
    if vocabulary["rest"]["method"] != "median":
        raise ValueError("unknown rest method %r" % vocabulary["rest"]["method"])
    for name, spec in vocabulary["phrases"].items():
        if spec["detector"] not in DETECTORS:
            raise ValueError("phrase %r: unknown detector %r" % (name, spec["detector"]))
        if spec["template"]["method"] != "trapezoid":
            raise ValueError("phrase %r: unknown template %r" % (name, spec["template"]["method"]))
        if spec["sign"] not in ("positive", "signed"):
            raise ValueError("phrase %r: sign must be positive or signed" % name)
        missing = [ref for ref in spec["drivers"] if ref not in spec["members"]]
        if missing:
            raise ValueError("phrase %r: drivers %s are not members" % (name, missing))
        for ref in spec["members"]:
            text, axis = split_ref(ref)
            if (text == ROTATION_TEXT) != (axis is not None) or (axis and axis not in ROTATION_AXES):
                raise ValueError("phrase %r: bad channel reference %r" % (name, ref))
        for pair in spec.get("pairs", []):
            if not set(pair) <= set(spec["drivers"]):
                raise ValueError("phrase %r: pair %s must be drivers" % (name, pair))
        if "dominant_names" in spec and set(spec["dominant_names"]) != set(spec["drivers"]):
            raise ValueError("phrase %r: dominant_names must name every driver" % name)
        for ref, names in spec.get("direction_names", {}).items():
            if spec["sign"] != "signed" or ref not in spec["drivers"] or set(names) != {"+", "-"}:
                raise ValueError("phrase %r: direction_names needs a signed driver and both '+' and '-'" % name)
        d = spec["detect"]
        if not 0 < d["dev_low"] <= d["dev_high"]:
            raise ValueError("phrase %r: need 0 < dev_low <= dev_high" % name)
        if not 0 < d["min_seconds"] < d["max_seconds"]:
            raise ValueError("phrase %r: need 0 < min_seconds < max_seconds" % name)
        t = spec["template"]
        if not (0 < t["onset_fraction"] and 0 < t["offset_fraction"] and t["onset_fraction"] + t["offset_fraction"] < 1):
            raise ValueError("phrase %r: template fractions must be positive and sum below 1" % name)


def _round(x) -> float:
    return round(float(x), VOCABULARY["rest"]["decimals"])


# ---------------------------------------------------------------------------
# Tracks: features.npz windows -> contiguous per-segment series.

def stitch_tracks(npz: dict) -> list:
    """One dict per (clip_id, segment_id), in first-appearance order."""
    order, groups = [], {}
    for row, key in enumerate(zip(npz["clip_id"].tolist(), npz["segment_id"].tolist())):
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)
    arrays = ("features", "quality", "observed", "is_gap_filled")
    tracks = []
    for clip_id, segment_id in order:
        rows = groups[(clip_id, segment_id)]
        starts = [int(npz["window_start"][r]) for r in rows]
        valid = [int((~npz["is_padding"][r]).sum()) for r in rows]
        t0, t1 = min(starts), max(s + v for s, v in zip(starts, valid))
        out = {k: np.zeros((t1 - t0,) + npz[k].shape[2:], dtype=npz[k].dtype) for k in arrays}
        filled = np.zeros(t1 - t0, dtype=bool)
        for r, s, v in zip(rows, starts, valid):
            span = slice(s - t0, s - t0 + v)
            for k in arrays:
                piece = npz[k][r, :v]
                seen = filled[span]
                if seen.any() and not np.array_equal(out[k][span][seen], piece[seen]):
                    raise ValueError("%s segment %d: overlapping windows disagree on %s" % (clip_id, segment_id, k))
                out[k][span] = piece
            filled[span] = True
        if not filled.all():
            raise ValueError("%s segment %d: windows leave frames uncovered" % (clip_id, segment_id))
        first = rows[int(np.argmin(starts))]
        tracks.append({"track_id": "%s#%d" % (clip_id, segment_id), "clip_id": clip_id, "segment_id": int(segment_id),
                       "track_start": t0, "time_sec": float(npz["window_time_sec"][first]), "rows": rows, **out})
    return tracks


def track_l1(track: dict, hz: float, blendshape_names: list, gaze_unit: str) -> dict:
    """The L1 text of a whole track: l1.encode_window over its full length."""
    n = track["features"].shape[0]
    header = {k: track[k] for k in ("track_id", "clip_id", "segment_id", "track_start", "time_sec")}
    return l1.encode_window(track["features"], track["quality"], track["observed"], track["is_gap_filled"],
                            np.zeros(n, dtype=bool), hz, blendshape_names, gaze_unit, header=header)


# ---------------------------------------------------------------------------
# L1 segments as per-frame values, and segment surgery.

def _axes(text_name: str):
    return ROTATION_AXES if text_name == ROTATION_TEXT else None


def channel_values(text: dict, blendshape_names: list, gaze_unit: str) -> dict:
    """text channel -> (values (T, k), covered (T,)): the refined L1 value per frame, unsmoothed."""
    hz, n = float(text["hz"]), int(text["n_frames"])
    out = {}
    for text_name, schema_name, _ in l1.text_channels(blendshape_names):
        q = l1.quantizer(schema_name, gaze_unit)
        axes = _axes(text_name)
        values = np.zeros((n, len(axes) if axes else 1))
        covered = np.zeros(n, dtype=bool)
        for entry in text["channels"][text_name]:
            s, e = l1._frame(entry["t"][0], hz), l1._frame(entry["t"][1], hz)
            labels = [entry["level"][a] for a in axes] if axes else [entry["level"]]
            hints = [entry["value_hint"][a] for a in axes] if axes else [entry["value_hint"]]
            values[s:e] = [l1._segment_value(l1.parse_label(lab, q), float(h), q, "refined") for lab, h in zip(labels, hints)]
            covered[s:e] = True
        out[text_name] = (values, covered)
    return out


def rest_values(values: dict) -> dict:
    rest = {}
    for text_name, (v, covered) in values.items():
        axes = _axes(text_name)
        med = np.median(v[covered], axis=0) if covered.any() else np.zeros(v.shape[1])
        rest[text_name] = {a: _round(m) for a, m in zip(axes, med)} if axes else _round(med[0])
    return rest


def split_segments(segments: list, spans: list, hz: float) -> tuple:
    """Cut L1 segments at span boundaries: (parts inside any span, parts outside). Level and hint are kept."""
    inside, outside = [], []
    for entry in segments:
        s, e = l1._frame(entry["t"][0], hz), l1._frame(entry["t"][1], hz)
        cuts = sorted({s, e} | {b for a, z in spans for b in (a, z) if s < b < e})
        for a, z in zip(cuts[:-1], cuts[1:]):
            part = dict(entry, t=[l1._time(a, hz), l1._time(z, hz)])
            (inside if any(x <= a and z <= y for x, y in spans) else outside).append(part)
    return inside, outside


def merge_segments(segments: list, hz: float) -> list:
    """Sort, then coalesce touching segments with equal level and hint (inverse of split_segments)."""
    out = []
    for entry in sorted(segments, key=lambda x: x["t"][0]):
        if (out and l1._frame(out[-1]["t"][1], hz) == l1._frame(entry["t"][0], hz)
                and out[-1]["level"] == entry["level"] and out[-1]["value_hint"] == entry["value_hint"]):
            out[-1] = dict(out[-1], t=[out[-1]["t"][0], entry["t"][1]])
        else:
            out.append(dict(entry))
    for a, b in zip(out[:-1], out[1:]):
        if l1._frame(a["t"][1], hz) > l1._frame(b["t"][0], hz):
            raise ValueError("segments overlap at %s / %s" % (a["t"], b["t"]))
    return out


# ---------------------------------------------------------------------------
# Channel references. A phrase names channels as "<text channel>" or, for the
# jointly quantized rotation, "<text channel>.<axis>" (e.g. "head_rotation.x").

TRANSLATION_TEXT = tuple(l1.RULES["channels"]["headpose_translation"]["text_names"])
GAZE_TEXT = tuple(l1.RULES["channels"]["gaze"]["text_names"])


def split_ref(ref: str) -> tuple:
    text, _, axis = ref.partition(".")
    return text, (axis or None)


def schema_of(text_name: str) -> str:
    if text_name == ROTATION_TEXT:
        return l1.ROTATION_CHANNEL
    if text_name in TRANSLATION_TEXT:
        return "headpose_translation"
    if text_name in GAZE_TEXT:
        return "gaze"
    return "blendshape"


def ref_series(values: dict, ref: str) -> tuple:
    text, axis = split_ref(ref)
    v, covered = values[text]
    return v[:, ROTATION_AXES.index(axis) if axis else 0], covered


def ref_rest(rest: dict, ref: str) -> float:
    text, axis = split_ref(ref)
    return float(rest[text][axis] if axis else rest[text])


def text_channels_of(spec: dict) -> list:
    return sorted({split_ref(ref)[0] for ref in spec["members"]})


# ---------------------------------------------------------------------------
# Detector: excursions of the driver channels away from rest.

def _runs(mask: np.ndarray) -> list:
    out, start = [], None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    return out


def detect_excursion(name: str, spec: dict, values: dict, rest: dict, hz: float, gaze_unit: str) -> tuple:
    """(phrases without id/detail, rejected candidates).

    Per driver: runs of frames at >= dev_low away from rest (one sign at a time for a
    signed phrase) that reach dev_high. Overlapping runs of all drivers form one
    candidate; it becomes a phrase if its duration is within bounds and every member
    channel is described over the whole span. Parameters are read at the strongest
    driver's extreme: the level there, and every member's deviation over those frames.
    """
    d = spec["detect"]
    signs = (1.0, -1.0) if spec["sign"] == "signed" else (1.0,)
    events = []
    for ref in spec["drivers"]:
        series, covered = ref_series(values, ref)
        dev = series - ref_rest(rest, ref)
        for sign in signs:
            for s, e in _runs((sign * dev >= d["dev_low"]) & covered):
                if (sign * dev[s:e]).max() >= d["dev_high"]:
                    events.append((s, e, ref))
    events.sort()
    clusters = []
    for s, e, ref in events:
        if clusters and s < clusters[-1][1]:
            clusters[-1] = (clusters[-1][0], max(clusters[-1][1], e), clusters[-1][2] | {ref})
        else:
            clusters.append((s, e, {ref}))

    phrases, rejected = [], []
    for s, e, active in clusters:
        t = [l1._time(s, hz), l1._time(e, hz)]
        seconds = (e - s) / hz
        if seconds < d["min_seconds"] - 1e-9 or seconds > d["max_seconds"] + 1e-9:
            rejected.append({"type": name, "t": t, "reason": "too_short" if seconds < d["min_seconds"] else "too_long"})
            continue
        if not all(values[text][1][s:e].all() for text in text_channels_of(spec)):
            rejected.append({"type": name, "t": t, "reason": "not_fully_described"})
            continue
        devs = {ref: ref_series(values, ref)[0][s:e] - ref_rest(rest, ref) for ref in spec["members"]}
        top = max(sorted(active), key=lambda ref: np.abs(devs[ref]).max())
        extreme = np.abs(devs[top]).max()
        at_peak = np.flatnonzero(np.abs(devs[top]) == extreme)
        text, _ = split_ref(top)
        q = l1.quantizer(schema_of(text), gaze_unit)
        peak_value = float(ref_series(values, top)[0][s + at_peak[0]])
        params = {"driver": top, "peak": l1.level_label(int(l1.quantize([peak_value], q)[0]), q)}
        if spec["sign"] == "signed":
            params["direction"] = "+" if devs[top][at_peak[0]] > 0 else "-"
        for left, right in spec.get("pairs", []):
            sides = [side for side, ref in (("left", left), ("right", right)) if ref in active]
            params["symmetry"] = "both" if len(sides) == 2 else sides[0] if sides else "none"
        if "dominant_names" in spec:
            params["dominant"] = spec["dominant_names"][top]
        if top in spec.get("direction_names", {}):
            params["motion"] = spec["direction_names"][top][params["direction"]]
        params.update({"onset": _round(at_peak[0] / hz), "offset": _round((e - s - 1 - at_peak[-1]) / hz),
                       "channel_dev": {ref: _round(devs[ref][at_peak].mean()) for ref in spec["members"]}})
        phrases.append({"type": name, "t": t, "params": params})
    return phrases, rejected


DETECTORS = {"excursion": detect_excursion}


# ---------------------------------------------------------------------------
# Generation: phrase -> per-frame value profile of each member over its span.

def _scale_for_peak(phrase: dict, rest: dict, gaze_unit: str) -> float:
    """An edited 'peak' level wins over stale channel deviations: scale them all so the
    driver lands on the new level's representative (refined-decode logic)."""
    p = phrase["params"]
    driver, dev = p["driver"], p["channel_dev"][p["driver"]]
    q = l1.quantizer(schema_of(split_ref(driver)[0]), gaze_unit)
    level = l1.parse_label(p["peak"], q)
    low, high = l1.bin_bounds(level, q)
    value = ref_rest(rest, driver) + dev
    if low <= value <= high or abs(dev) < 1e-9:
        return 1.0
    return (float(l1.dequantize(level, q)) - ref_rest(rest, driver)) / dev


def trapezoid_shape(n: int, on: int, off: int) -> np.ndarray:
    on, off = min(on, n - 1), min(off, max(0, n - 1 - min(on, n - 1)))
    shape = np.ones(n)
    shape[:on] = (np.arange(on) + 1) / (on + 1)
    if off:
        shape[n - off:] = np.arange(off, 0, -1) / (off + 1)
    return shape


def keypoint_shape(n: int, on: int, off: int, curve: dict) -> np.ndarray:
    """A three-piece normalized curve (rise / hold / fall, each resampled) laid onto n frames."""
    on, off = min(on, n - 1), min(off, max(0, n - 1 - min(on, n - 1)))
    hold = n - on - off
    pieces = []
    for part, length in (("rise", on), ("hold", hold), ("fall", off)):
        if length:
            y = np.asarray(curve[part], dtype=np.float64)
            pieces.append(np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(y)), y))
    return np.concatenate(pieces)


def generate(phrase: dict, spec: dict, rest: dict, hz: float, gaze_unit: str, kb: dict = None) -> dict:
    """member ref -> (n,) profile. Uses the knowledge-base shape of this type when given, else the template."""
    s, e = l1._frame(phrase["t"][0], hz), l1._frame(phrase["t"][1], hz)
    n, p, t = e - s, phrase["params"], spec["template"]
    on = int(round(p["onset"] * hz)) if "onset" in p else int(round(t["onset_fraction"] * n))
    off = int(round(p["offset"] * hz)) if "offset" in p else int(round(t["offset_fraction"] * n))
    entry = (kb or {}).get("types", {}).get(phrase["type"], {})
    shapes, ratios = entry.get("shape", {}), entry.get("ratio", {})
    channel_dev = dict(p.get("channel_dev", {}))
    if p["driver"] not in channel_dev:
        # a hand-written phrase: the driver's deviation is its peak level's representative above rest
        q = l1.quantizer(schema_of(split_ref(p["driver"])[0]), gaze_unit)
        channel_dev[p["driver"]] = float(l1.dequantize(l1.parse_label(p["peak"], q), q)) - ref_rest(rest, p["driver"])
    for ref in spec["members"]:
        if ref not in channel_dev:  # unspecified members follow the driver by the kb ratio (0 without a kb)
            channel_dev[ref] = ratios.get(ref, 0.0) * channel_dev[p["driver"]]
    phrase = dict(phrase, params=dict(p, channel_dev=channel_dev))
    scale = _scale_for_peak(phrase, rest, gaze_unit)
    out = {}
    for ref in spec["members"]:
        dev = scale * channel_dev[ref]
        shape = keypoint_shape(n, on, off, shapes[ref]) if ref in shapes else trapezoid_shape(n, on, off)
        profile = ref_rest(rest, ref) + dev * shape
        out[ref] = np.clip(profile, 0.0, 1.0) if schema_of(split_ref(ref)[0]) == "blendshape" else profile
    return out


def _segments_from_profile(values: np.ndarray, start: int, hz: float, q: dict, axes=None) -> list:
    """Quantize and segment a generated (n, k) profile exactly as L1 encode does."""
    values = values.reshape(len(values), -1)
    levels = l1.quantize(values, q)
    out = []
    for a, z, row in l1.segment(levels, np.ones(len(values), dtype=bool), l1.min_segment_frames(hz)):
        hint = l1._consistent(row, [round(float(v), 4) for v in values[a:z].mean(0)], q)
        entry = {"t": [l1._time(start + a, hz), l1._time(start + z, hz)]}
        if axes:
            entry["level"] = {ax: l1.level_label(row[k], q) for k, ax in enumerate(axes)}
            entry["value_hint"] = {ax: hint[k] for k, ax in enumerate(axes)}
        else:
            entry["level"], entry["value_hint"] = l1.level_label(row[0], q), hint[0]
        out.append(entry)
    return out


def phrase_segments(phrase: dict, rest: dict, hz: float, gaze_unit: str, kb: dict = None, generator=None) -> dict:
    """text channel -> generated L1 segments for one phrase's span.

    generator(phrase, rest, hz) -> {member ref: profile} or None overrides the built-in
    shape (exemplar retrieval plugs in here); None falls back to kb shape / template.
    """
    spec = VOCABULARY["phrases"][phrase["type"]]
    profiles = generator(phrase, rest, hz) if generator else None
    if profiles is None:
        profiles = generate(phrase, spec, rest, hz, gaze_unit, kb)
    start = l1._frame(phrase["t"][0], hz)
    n = l1._frame(phrase["t"][1], hz) - start
    out = {}
    for text in text_channels_of(spec):
        q = l1.quantizer(schema_of(text), gaze_unit)
        if text == ROTATION_TEXT:
            block = np.stack([profiles.get("%s.%s" % (text, ax), np.full(n, rest[text][ax])) for ax in ROTATION_AXES], 1)
            out[text] = _segments_from_profile(block, start, hz, q, ROTATION_AXES)
        else:
            out[text] = _segments_from_profile(profiles[text], start, hz, q)
    return out


# ---------------------------------------------------------------------------
# Encode / decode one track.

L1_HEADER = ("hz", "n_frames", "reference", "masks")


def encode_track(text: dict, blendshape_names: list, gaze_unit: str) -> tuple:
    """Track-level L1 text -> (L2 track dict, rejected candidates)."""
    hz = float(text["hz"])
    values = channel_values(text, blendshape_names, gaze_unit)
    rest = rest_values(values)
    phrases, rejected, claimed = [], [], {}
    for name, spec in sorted(VOCABULARY["phrases"].items(), key=lambda kv: (-kv[1]["priority"], kv[0])):
        found, rej = DETECTORS[spec["detector"]](name, spec, values, rest, hz, gaze_unit)
        rejected += rej
        for phrase in found:
            span = (l1._frame(phrase["t"][0], hz), l1._frame(phrase["t"][1], hz))
            if any(a < span[1] and span[0] < b for ch in text_channels_of(spec) for a, b in claimed.get(ch, [])):
                rejected.append({"type": name, "t": phrase["t"], "reason": "claimed_by_higher_priority"})
                continue
            for ch in text_channels_of(spec):
                claimed.setdefault(ch, []).append(span)
            phrases.append(phrase)
    phrases.sort(key=lambda p: p["t"][0])

    residual = {}
    for ch, segments in text["channels"].items():
        spans = claimed.get(ch, [])
        inside, residual[ch] = split_segments(segments, spans, hz) if spans else ([], [dict(x) for x in segments])
        for phrase in phrases:
            if ch in text_channels_of(VOCABULARY["phrases"][phrase["type"]]):
                s, e = l1._frame(phrase["t"][0], hz), l1._frame(phrase["t"][1], hz)
                phrase.setdefault("detail", {})[ch] = [x for x in inside if s <= l1._frame(x["t"][0], hz) < e]
    for index, phrase in enumerate(phrases):
        phrase["id"] = index
    track = {k: text[k] for k in ("track_id", "clip_id", "segment_id", "track_start", "time_sec") if k in text}
    track.update({k: text[k] for k in L1_HEADER})
    track.update({"rest": rest, "phrases": [{"id": p["id"], "type": p["type"], "t": p["t"], "params": p["params"],
                                              "detail": p["detail"]} for p in phrases],
                  "residual": residual})
    return track, rejected


def decode_track(track: dict, gaze_unit: str, mode: str = "exact", kb: dict = None, generator=None) -> dict:
    """L2 track -> track-level L1 text. exact uses every phrase's detail; phrase_only regenerates
    each span (knowledge-base shapes when kb is given, else the vocabulary's template)."""
    if mode not in ("exact", "phrase_only"):
        raise ValueError("mode must be exact or phrase_only, got %r" % mode)
    hz = float(track["hz"])
    channels = {ch: list(segments) for ch, segments in track["residual"].items()}
    for phrase in track["phrases"]:
        if mode == "exact":
            if "detail" not in phrase:
                raise ValueError("phrase %s has no detail (edited?); use mode='phrase_only'" % phrase["id"])
            for ch, segments in phrase["detail"].items():
                channels[ch] += segments
            continue
        for ch, segments in phrase_segments(phrase, track["rest"], hz, gaze_unit, kb, generator).items():
            channels[ch] += segments
    text = {k: track[k] for k in L1_HEADER}
    text["channels"] = {ch: merge_segments(segments, hz) for ch, segments in channels.items()}
    return text


def encode_document(npz: dict, blendshape_names: list, gaze_unit: str, source: dict = None) -> tuple:
    """features.npz arrays -> (L2 document, track-level L1 texts, rejected candidates per track)."""
    hz = float(npz["target_hz"])
    tracks, l1_texts, rejected = [], [], []
    for track in stitch_tracks(npz):
        text = track_l1(track, hz, blendshape_names, gaze_unit)
        l2_track, rej = encode_track(text, blendshape_names, gaze_unit)
        tracks.append(l2_track)
        l1_texts.append(text)
        rejected.append(rej)
    document = {"format": "l2_text", "vocabulary_version": VOCABULARY_VERSION, "l1_rules_version": l1.RULES_VERSION,
                "schema_version": l1.SCHEMA_VERSION, "gaze_unit": gaze_unit, "blendshape_names": list(blendshape_names),
                "target_hz": hz, "source": source or {}, "tracks": tracks}
    return document, l1_texts, rejected


VOCABULARY = _load_vocabulary()
_validate(VOCABULARY)
VOCABULARY_VERSION = VOCABULARY["vocabulary_version"]
