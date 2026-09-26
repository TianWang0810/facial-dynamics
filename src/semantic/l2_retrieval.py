"""
Exemplar retrieval for phrase-only L2 -> L1 decode (docs/plans/l2_knowledge_base_retrieval.md section 6).

First version: per-phrase TARGET cost only (join cost / Viterbi over a track is the
next step); the k lowest-cost exemplars are averaged, with per-type weights and k fitted
on VALIDATION speakers (fit_weights). k = 1 is pure retrieval; large k tends to the mean shape. For a query phrase, candidates are the bank's exemplars of the same type
(TRAIN speakers only -- the caller builds the bank from train tracks):

    hard filter  duration ratio within [1/alpha, alpha]; relaxed to all exemplars of
                 the type if none survive (recorded)
    target cost  |log(d_q / d_e)| / sd                            duration
                 + sum_m |cdev_q[m] - cdev_e[m]| / sd_m / n_m    member deviations
                 + (|onset_q - onset_e| + |offset_q - offset_e|) / sd_t   keypoints
                 + |rest_q[driver] - rest_e[driver]| / sd_r      resting-face compatibility
                 each term normalized by its spread in the bank; tie -> lowest exemplar index
    transform    the exemplar's per-member deviation curve, cut at its keypoints into
                 rise / hold / fall and stretched onto the query's, scaled per member by
                 cdev_q / cdev_e (clipped to [1/4, 4]); a member whose exemplar plateau is
                 below its min_plateau keeps the exemplar curve shifted by the plateau
                 difference instead of scaled

Parameters live in schemas/l2_retrieval.json. Deterministic: same bank, same query,
same output.
"""
import json
import os

import numpy as np

from src.semantic import l1, l2

RETRIEVAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "schemas", "l2_retrieval.json",
)
with open(RETRIEVAL_PATH, encoding="utf-8") as _handle:
    RETRIEVAL = json.load(_handle)


def _keypoints(phrase: dict, hz: float) -> tuple:
    s, e = l1._frame(phrase["t"][0], hz), l1._frame(phrase["t"][1], hz)
    n = e - s
    on = min(int(round(phrase["params"]["onset"] * hz)), n - 1)
    off = min(int(round(phrase["params"]["offset"] * hz)), max(0, n - 1 - on))
    return s, e, on, off


def build_bank(document: dict, l1_texts: list, track_ids: set) -> dict:
    """type -> dict of arrays + per-exemplar curves, from the given (TRAIN) tracks."""
    names, unit, hz = document["blendshape_names"], document["gaze_unit"], float(document["target_hz"])
    bank = {}
    for track, text in zip(document["tracks"], l1_texts):
        if track["track_id"] not in track_ids or not track["phrases"]:
            continue
        values = l2.channel_values(text, names, unit)
        for phrase in track["phrases"]:
            spec = l2.VOCABULARY["phrases"][phrase["type"]]
            s, e, on, off = _keypoints(phrase, hz)
            p = phrase["params"]
            b = bank.setdefault(phrase["type"], {"duration": [], "cdev": [], "onset": [], "offset": [], "rest": [],
                                                 "curves": [], "keys": [], "source": []})
            b["duration"].append((e - s) / hz)
            b["cdev"].append([p["channel_dev"][ref] for ref in spec["members"]])
            b["onset"].append(p["onset"])
            b["offset"].append(p["offset"])
            b["rest"].append(l2.ref_rest(track["rest"], p["driver"]))
            b["curves"].append({ref: l2.ref_series(values, ref)[0][s:e] - l2.ref_rest(track["rest"], ref) for ref in spec["members"]})
            b["keys"].append((on, off))
            b["source"].append({"track_id": track["track_id"], "t": phrase["t"]})
    for b in bank.values():
        for k in ("duration", "cdev", "onset", "offset", "rest"):
            b[k] = np.asarray(b[k], dtype=np.float64)
        b["sd"] = {"log_duration": max(float(np.log(b["duration"]).std()), 1e-3),
                   "cdev": np.maximum(b["cdev"].std(0), 1e-4),
                   "time": max(float(np.concatenate([b["onset"], b["offset"]]).std()), 1.0 / hz),
                   "rest": max(float(b["rest"].std()), 1e-4)}
    return bank


COMPONENTS = ("duration", "channel_dev", "keypoints", "rest")


def cost_components(bank_type: dict, phrase: dict, rest: dict, hz: float) -> tuple:
    """(dict of per-exemplar cost terms, filter mask) for one query; each term normalized by its bank spread."""
    spec = l2.VOCABULARY["phrases"][phrase["type"]]
    p = phrase["params"]
    s, e, _, _ = _keypoints(phrase, hz)
    d_q = (e - s) / hz
    cdev_q = np.array([p["channel_dev"][ref] for ref in spec["members"]])
    sd = bank_type["sd"]
    terms = {"duration": np.abs(np.log(d_q / bank_type["duration"])) / sd["log_duration"],
             "channel_dev": (np.abs(cdev_q - bank_type["cdev"]) / sd["cdev"]).mean(1),
             "keypoints": (np.abs(p["onset"] - bank_type["onset"]) + np.abs(p["offset"] - bank_type["offset"])) / sd["time"],
             "rest": np.abs(l2.ref_rest(rest, p["driver"]) - bank_type["rest"]) / sd["rest"]}
    ratio = d_q / bank_type["duration"]
    alpha = RETRIEVAL["filter"]["max_duration_ratio"]
    ok = (ratio <= alpha) & (ratio >= 1.0 / alpha)
    return terms, (ok if ok.any() else np.ones_like(ok))


def target_costs(bank_type: dict, phrase: dict, rest: dict, hz: float, weights: dict = None) -> tuple:
    """(cost per exemplar, relaxed?) for one query; filtered exemplars cost inf."""
    weights = weights or RETRIEVAL["weights"]
    terms, ok = cost_components(bank_type, phrase, rest, hz)
    cost = sum(weights[k] * terms[k] for k in COMPONENTS)
    return np.where(ok, cost, np.inf), bool(ok.all() and not _passes(bank_type, phrase, hz).any())


def _passes(bank_type: dict, phrase: dict, hz: float) -> np.ndarray:
    s, e, _, _ = _keypoints(phrase, hz)
    ratio = ((e - s) / hz) / bank_type["duration"]
    alpha = RETRIEVAL["filter"]["max_duration_ratio"]
    return (ratio <= alpha) & (ratio >= 1.0 / alpha)


def _warp(curve: np.ndarray, keys: tuple, n: int, on: int, off: int) -> np.ndarray:
    """Exemplar curve cut at its keypoints (rise / hold / fall) and stretched onto the query's."""
    m = len(curve)
    e_on, e_off = keys
    first, last = e_on, max(e_on, m - 1 - e_off)
    src = {"rise": curve[:first], "hold": curve[first:last + 1], "fall": curve[last + 1:]}
    dst = {"rise": on, "hold": n - on - off, "fall": off}
    out = []
    for part in ("rise", "hold", "fall"):
        if not dst[part]:
            continue
        y = src[part]
        if not len(y):  # the exemplar has no such part: bridge from the neighbouring value
            y = np.array([curve[first] if part == "rise" else curve[last]])
        out.append(np.interp(np.linspace(0, 1, dst[part]), np.linspace(0, 1, len(y)), y) if len(y) > 1 else np.full(dst[part], y[0]))
    return np.concatenate(out)


def profiles_from(bank_type: dict, index: int, phrase: dict, rest: dict, hz: float) -> dict:
    """member ref -> (n,) profile of the query span, built from exemplar `index`."""
    spec = l2.VOCABULARY["phrases"][phrase["type"]]
    _, _, on, off = _keypoints(phrase, hz)
    s, e = l1._frame(phrase["t"][0], hz), l1._frame(phrase["t"][1], hz)
    n = e - s
    lo, hi = RETRIEVAL["transform"]["scale_clip"]
    out = {}
    for k, ref in enumerate(spec["members"]):
        curve = _warp(bank_type["curves"][index][ref], bank_type["keys"][index], n, on, off)
        cq, ce = phrase["params"]["channel_dev"][ref], bank_type["cdev"][index, k]
        schema = l2.schema_of(l2.split_ref(ref)[0])
        if abs(ce) >= RETRIEVAL["transform"]["min_plateau"][schema]:
            curve = curve * float(np.clip(cq / ce, lo, hi))
        else:
            curve = curve + (cq - ce)
        profile = l2.ref_rest(rest, ref) + curve
        out[ref] = np.clip(profile, 0.0, 1.0) if schema == "blendshape" else profile
    return out


def _average(profiles: list) -> dict:
    return {ref: np.mean([p[ref] for p in profiles], 0) for ref in profiles[0]}


def retrieve(bank: dict, phrase: dict, rest: dict, hz: float, fit: dict = None) -> tuple:
    """(profiles, provenance): mean of the k lowest-cost exemplars' transformed profiles.

    fit (from fit_weights) gives per-type weights and k; without it, the JSON weights and k = 1.
    Ties are broken by exemplar index (stable sort), so the result is deterministic.
    """
    b = bank.get(phrase["type"])
    if not b:
        return None, None
    f = (fit or {}).get(phrase["type"], {})
    cost, relaxed = target_costs(b, phrase, rest, hz, f.get("weights"))
    order = [int(i) for i in np.argsort(cost, kind="stable")[:f.get("k", 1)] if np.isfinite(cost[i])]
    profiles = _average([profiles_from(b, i, phrase, rest, hz) for i in order])
    return profiles, {"exemplars": [b["source"][i] for i in order], "cost": float(cost[order[0]]), "relaxed": relaxed}


def fit_weights(bank: dict, queries: dict, hz: float, gaze_unit: str, pool: int = 150,
                grid=(0.0, 0.5, 1.0, 2.0, 4.0), ks=(1, 3, 5, 10, 20, 50, 100), rounds: int = 2) -> dict:
    """Per phrase type, the weights and k that minimize the per-frame level mismatch on the given
    (VALIDATION) queries: {type: [(phrase, rest, truth {ref: values})]}. Coordinate descent over the
    weight grid, then k; candidates are each query's `pool` lowest-cost exemplars under the JSON weights,
    whose transformed profiles are computed once."""
    out = {}
    for kind, items in queries.items():
        b = bank.get(kind)
        if not b or not items:
            continue
        spec = l2.VOCABULARY["phrases"][kind]
        qs = {ref: l1.quantizer(l2.schema_of(l2.split_ref(ref)[0]), gaze_unit) for ref in spec["members"]}
        prepared = []
        for phrase, rest, truth in items:
            terms, ok = cost_components(b, phrase, rest, hz)
            base = np.where(ok, sum(RETRIEVAL["weights"][k] * terms[k] for k in COMPONENTS), np.inf)
            cand = [int(i) for i in np.argsort(base, kind="stable")[:pool] if np.isfinite(base[i])]
            profs = [profiles_from(b, i, phrase, rest, hz) for i in cand]
            stacked = {ref: np.stack([p[ref] for p in profs]) for ref in spec["members"]}
            truth_levels = {ref: l1.quantize(truth[ref], qs[ref]) for ref in spec["members"]}
            prepared.append(({k: terms[k][cand] for k in COMPONENTS}, stacked, truth_levels))

        def mismatch(weights, k):
            miss = total = 0
            for terms, stacked, truth_levels in prepared:
                order = np.argsort(sum(weights[c] * terms[c] for c in COMPONENTS), kind="stable")[:k]
                for ref, arr in stacked.items():
                    lev = l1.quantize(arr[order].mean(0), qs[ref])
                    miss += int((lev != truth_levels[ref]).sum())
                    total += lev.size
            return miss / max(total, 1)

        weights, k = dict(RETRIEVAL["weights"]), 1
        best = mismatch(weights, k)
        for _ in range(rounds):
            for c in COMPONENTS:
                for v in grid:
                    trial = dict(weights, **{c: v})
                    score = mismatch(trial, k)
                    if score < best - 1e-12:
                        best, weights = score, trial
            for kk in ks:
                score = mismatch(weights, kk)
                if score < best - 1e-12:
                    best, k = score, kk
        out[kind] = {"weights": weights, "k": k, "val_level_mismatch": best, "n_val": len(items),
                     "val_level_mismatch_json_weights_k1": mismatch(dict(RETRIEVAL["weights"]), 1)}
    return out
