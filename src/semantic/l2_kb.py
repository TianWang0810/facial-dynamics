"""
Layer 3 of the L2 knowledge base: statistics of real phrases, fitted on TRAIN speakers.

    shape   per phrase type and member channel, the mean normalized curve of a phrase,
            cut at its keypoints into rise / hold / fall so that phrases of different
            durations and onset/offset times average without smearing. Normalization
            is by the member's own plateau deviation (params.channel_dev), so the
            curve is a shape and the amplitude stays in the text.
    ratio   per member: median of channel_dev[member] / channel_dev[driver], used to
            fill members a new or hand-written phrase does not specify.
    params  quantiles of duration, driver deviation, onset and offset.

l2.decode_track(..., kb=...) uses `shape` (and `ratio` for missing members) in
phrase-only decode; without a kb it falls back to the vocabulary's trapezoid.
Nothing here is used by encode or by exact decode, which stay rule-only.
"""
import numpy as np

from src.semantic import l1, l2

POINTS = 8
MIN_PLATEAU = 1e-3


def _pieces(dev: np.ndarray, on: int, off: int) -> dict:
    n = len(dev)
    first, last = min(on, n - 1), max(min(on, n - 1), n - 1 - off)
    parts = {"rise": dev[:first], "hold": dev[first:last + 1], "fall": dev[last + 1:]}
    return {k: np.interp(np.linspace(0, 1, POINTS), np.linspace(0, 1, len(v)), v) for k, v in parts.items() if len(v)}


def build(document: dict, l1_texts: list, track_ids: set) -> dict:
    """KB from the phrases of the given tracks (pass TRAIN tracks only)."""
    names, unit, hz = document["blendshape_names"], document["gaze_unit"], float(document["target_hz"])
    curves, ratios, stats = {}, {}, {}  # curves[type][ref][part] = [sum of signed deviations, sum of |plateau|]
    for track, text in zip(document["tracks"], l1_texts):
        if track["track_id"] not in track_ids or not track["phrases"]:
            continue
        values = l2.channel_values(text, names, unit)
        for phrase in track["phrases"]:
            spec = l2.VOCABULARY["phrases"][phrase["type"]]
            p = phrase["params"]
            s, e = l1._frame(phrase["t"][0], hz), l1._frame(phrase["t"][1], hz)
            on, off = int(round(p["onset"] * hz)), int(round(p["offset"] * hz))
            driver_dev = p["channel_dev"][p["driver"]]
            st = stats.setdefault(phrase["type"], {"duration": [], "driver_dev": [], "onset": [], "offset": []})
            for key, v in (("duration", (e - s) / hz), ("driver_dev", abs(driver_dev)), ("onset", p["onset"]), ("offset", p["offset"])):
                st[key].append(v)
            for ref in spec["members"]:
                plateau = p["channel_dev"][ref]
                if abs(driver_dev) > MIN_PLATEAU:
                    ratios.setdefault(phrase["type"], {}).setdefault(ref, []).append(plateau / driver_dev)
                if abs(plateau) < MIN_PLATEAU:
                    continue
                # plateau-weighted: sum(dev * sign) / sum(|plateau|), so a member that barely moves in one
                # exemplar cannot blow up the mean shape by a division by its near-zero plateau
                dev = (l2.ref_series(values, ref)[0][s:e] - l2.ref_rest(track["rest"], ref)) * np.sign(plateau)
                for part, y in _pieces(dev, on, off).items():
                    acc = curves.setdefault(phrase["type"], {}).setdefault(ref, {}).setdefault(part, [np.zeros(POINTS), 0.0])
                    acc[0] += y
                    acc[1] += abs(plateau)
    kb = {"points": POINTS, "n_tracks": len(track_ids), "types": {}}
    for kind, st in stats.items():
        shape = {}
        for ref, parts in curves.get(kind, {}).items():
            # a part no exemplar had (e.g. a rise when every onset was 0) falls back to the trapezoid's line
            default = {"rise": np.linspace(1 / (POINTS + 1), POINTS / (POINTS + 1), POINTS),
                       "hold": np.ones(POINTS), "fall": np.linspace(POINTS / (POINTS + 1), 1 / (POINTS + 1), POINTS)}
            shape[ref] = {part: [round(float(v), 4) for v in (parts[part][0] / parts[part][1] if part in parts else default[part])]
                          for part in ("rise", "hold", "fall")}
        kb["types"][kind] = {
            "n": len(st["duration"]),
            "shape": shape,
            "ratio": {ref: round(float(np.median(r)), 4) for ref, r in ratios.get(kind, {}).items()},
            "params": {k: {q: round(float(np.percentile(v, pct)), 4) for q, pct in (("p10", 10), ("median", 50), ("p90", 90))}
                       for k, v in st.items()},
        }
    return kb
