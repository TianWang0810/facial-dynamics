"""
CLI: encode a run into L2 text (docs/plans/l2_knowledge_base_retrieval.md) and report on it.

Writes <output_dir>/l2_text.json, l2_report.json and l2_report.md. Checks and numbers:

    exact round trip  every track's L2 -> L1 must equal its track-level L1 verbatim
    track-level L1    round-trip error vs the original tensor (the new L1 timeline
                      L2 sits on; compare with the window-level l1_report)
    phrases           count, rate, duration, peak, symmetry, onset/offset; rejected
                      candidates by reason
    coverage          frames where a phrase channel is >= rest + dev_high that lie
                      inside a phrase of that type
    compression       residual segments + phrases vs track-level L1 segments
    phrase_only       template fill vs the exact L1 inside phrase spans: per-frame
                      level agreement and decoded value error

Usage:
    python scripts/l2_report.py --run_dir ../runs/<id>
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.semantic import l1, l2, l2_kb, l2_retrieval  # noqa: E402
from src.semantic.l1_eval import value_errors  # noqa: E402
from src.semantic.l1_io import add_input_arguments, load_inputs  # noqa: E402
from src.sequence.split import make_split, verify_no_leakage  # noqa: E402


def quantiles(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "p10": float(np.percentile(x, 10)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "mean": float(x.mean())}


ORACLE_TOP = 50


def oracle_choices(document: dict, test: list, test_values: list, bank: dict, unit: str, hz: float) -> dict:
    """(track_id, phrase id) -> the exemplar among the ORACLE_TOP lowest-cost candidates whose transformed
    profile best matches the true per-frame L1 levels (then values). An upper bound on what a better
    cost function could pick from the same bank; it uses the answer, so it is never a decode method."""
    out = {}
    for k, i in enumerate(test):
        track = document["tracks"][i]
        for phrase in track["phrases"]:
            b = bank.get(phrase["type"])
            if not b:
                continue
            spec = l2.VOCABULARY["phrases"][phrase["type"]]
            s, e = l1._frame(phrase["t"][0], hz), l1._frame(phrase["t"][1], hz)
            truth = {ref: l2.ref_series(test_values[k], ref)[0][s:e] for ref in spec["members"]}
            qs = {ref: l1.quantizer(l2.schema_of(l2.split_ref(ref)[0]), unit) for ref in spec["members"]}
            cost, _ = l2_retrieval.target_costs(b, phrase, track["rest"], hz)
            best = None
            for index in np.argsort(cost, kind="stable")[:ORACLE_TOP]:
                if not np.isfinite(cost[index]):
                    break
                prof = l2_retrieval.profiles_from(b, int(index), phrase, track["rest"], hz)
                miss = sum(int((l1.quantize(prof[r], qs[r]) != l1.quantize(truth[r], qs[r])).sum()) for r in prof)
                err = sum(float(np.abs(prof[r] - truth[r]).mean()) / b["sd"]["cdev"][j] for j, r in enumerate(spec["members"]))
                if best is None or (miss, err) < best[0]:
                    best = ((miss, err), int(index))
            if best:
                out[(track["track_id"], phrase["id"])] = best[1]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_input_arguments(parser)
    parser.add_argument("--output_dir", default=None, help="Default: <run_dir>/l2")
    args = parser.parse_args()
    npz, names, unit, source = load_inputs(args)
    out_dir = args.output_dir or os.path.join(os.path.abspath(args.run_dir), "l2")
    os.makedirs(out_dir, exist_ok=True)
    hz = float(npz["target_hz"])
    q = l1.quantizer("blendshape", unit)

    document, l1_texts, rejected = l2.encode_document(npz, names, unit, source=source)
    with open(os.path.join(out_dir, "l2_text.json"), "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    tracks = l2.stitch_tracks(npz)

    report = {"source": source, "vocabulary_version": l2.VOCABULARY_VERSION, "n_tracks": len(tracks),
              "described_seconds": 0.0, "checks": {}, "phrases": {}}

    exact_ok, originals, recons, described = 0, [], [], []
    for track, text, l2_track in zip(tracks, l1_texts, document["tracks"]):
        back = l2.decode_track(l2_track, unit, "exact")
        exact_ok += int(back["channels"] == text["channels"] and all(back[k] == text[k] for k in l2.L1_HEADER))
        originals.append(track["features"])
        recons.append(l1.decode_window(text, names, unit)["features"])
        described.append(l1.describable(track["observed"], track["quality"], track["is_gap_filled"],
                                        np.zeros(len(track["features"]), dtype=bool)))
    d_all = np.concatenate(described)[None]
    report["described_seconds"] = float(d_all[..., 0].sum() / hz)
    report["checks"]["exact_round_trip"] = {"tracks_ok": exact_ok, "tracks": len(tracks), "pass": exact_ok == len(tracks)}
    v = value_errors(np.concatenate(originals)[None], np.concatenate(recons)[None], d_all, names)
    report["track_level_l1"] = {"bs_mae": v["blendshape"]["mae"], "bs_max": v["blendshape"]["max_abs"],
                                "trans_mae": v["headpose_translation"]["mae"],
                                "rot_mean_deg": v[l1.ROTATION_CHANNEL]["mean"], "rot_max_deg": v[l1.ROTATION_CHANNEL]["max"],
                                "gaze_mae": v["gaze"]["mae"]}

    l1_segments = sum(len(s) for t in l1_texts for s in t["channels"].values())
    residual_segments = sum(len(s) for t in document["tracks"] for s in t["residual"].values())
    n_phrases = sum(len(t["phrases"]) for t in document["tracks"])
    report["compression"] = {"l1_segments": l1_segments, "residual_segments": residual_segments, "phrases": n_phrases,
                             "l2_units_over_l1": (residual_segments + n_phrases) / l1_segments}

    # speaker-disjoint split: the knowledge base is fitted on TRAIN speakers, phrase-only decode scored on TEST
    clips = sorted({t["clip_id"] for t in tracks})
    speaker = {c: c.split("__")[0] for c in clips}
    split = make_split(clips, [speaker[c] for c in clips], seed=0)
    if not verify_no_leakage(split, speaker)["is_clean"]:
        raise ValueError("split leaks speakers")
    part = {t["track_id"]: split["assignments"][t["clip_id"]] for t in document["tracks"]}
    kb = l2_kb.build(document, l1_texts, {k for k, v in part.items() if v == "train"})
    kb["split"] = {"kind": split["kind"], "seed": split["seed"], "counts": split["counts"]}
    os.makedirs(os.path.join(out_dir, "kb"), exist_ok=True)
    with open(os.path.join(out_dir, "kb", "kb_shapes.json"), "w", encoding="utf-8") as handle:
        json.dump(kb, handle, indent=2)
    report["split"] = kb["split"]
    test = [i for i, t in enumerate(document["tracks"]) if part[t["track_id"]] == "test"]
    bank = l2_retrieval.build_bank(document, l1_texts, {k for k, v in part.items() if v == "train"})
    test_values = [l2.channel_values(l1_texts[i], names, unit) for i in test]
    oracle_choice = oracle_choices(document, test, test_values, bank, unit, hz)

    # retrieval weights and k fitted on VALIDATION speakers, then frozen for the test numbers
    val_queries = {}
    for i, t_l2 in enumerate(document["tracks"]):
        if part[t_l2["track_id"]] != "val":
            continue
        values = l2.channel_values(l1_texts[i], names, unit)
        for phrase in t_l2["phrases"]:
            s0, e0 = l1._frame(phrase["t"][0], hz), l1._frame(phrase["t"][1], hz)
            spec = l2.VOCABULARY["phrases"][phrase["type"]]
            truth = {ref: l2.ref_series(values, ref)[0][s0:e0] for ref in spec["members"]}
            val_queries.setdefault(phrase["type"], []).append((phrase, t_l2["rest"], truth))
    fit = l2_retrieval.fit_weights(bank, val_queries, hz, unit)
    with open(os.path.join(out_dir, "kb", "retrieval_fit.json"), "w", encoding="utf-8") as handle:
        json.dump(fit, handle, indent=2)
    report["retrieval_fit"] = fit

    def retrieval_gen(phrase, rest, hz_):
        return l2_retrieval.retrieve(bank, phrase, rest, hz_)[0]

    def retrieval_fit_gen(phrase, rest, hz_):
        return l2_retrieval.retrieve(bank, phrase, rest, hz_, fit)[0]

    def oracle_gen(track_id):
        def gen(phrase, rest, hz_):
            index = oracle_choice.get((track_id, phrase["id"]))
            return None if index is None else l2_retrieval.profiles_from(bank[phrase["type"]], index, phrase, rest, hz_)
        return gen

    tr = [document["tracks"][i] for i in test]
    generated = {"template": [l2.decode_track(t, unit, "phrase_only") for t in tr],
                 "kb_shape": [l2.decode_track(t, unit, "phrase_only", kb=kb) for t in tr],
                 "retrieval_k1": [l2.decode_track(t, unit, "phrase_only", kb=kb, generator=retrieval_gen) for t in tr],
                 "retrieval_fit": [l2.decode_track(t, unit, "phrase_only", kb=kb, generator=retrieval_fit_gen) for t in tr],
                 "oracle@%d" % ORACLE_TOP: [l2.decode_track(t, unit, "phrase_only", kb=kb, generator=oracle_gen(t["track_id"]))
                                            for t in tr]}
    generated_values = {m: [l2.channel_values(g, names, unit) for g in gs] for m, gs in generated.items()}

    seconds_by_speaker = {}
    for track in tracks:
        spk = speaker[track["clip_id"]]
        seconds_by_speaker[spk] = seconds_by_speaker.get(spk, 0.0) + len(track["features"]) / hz
    for name, spec in l2.VOCABULARY["phrases"].items():
        found = [(t, p) for t in document["tracks"] for p in t["phrases"] if p["type"] == name]
        count_by_speaker = {s: 0 for s in seconds_by_speaker}
        for t, _ in found:
            count_by_speaker[speaker[t["clip_id"]]] += 1
        reasons = {}
        for rej in rejected:
            for r in rej:
                if r["type"] == name:
                    reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        params = [p["params"] for _, p in found]
        entry = {
            "count": len(found),
            "per_minute_per_speaker": quantiles([60 * count_by_speaker[s] / seconds_by_speaker[s] for s in seconds_by_speaker]),
            "duration_s": quantiles([p["t"][1] - p["t"][0] for _, p in found]),
            "driver_dev_abs": quantiles([abs(x["channel_dev"][x["driver"]]) for x in params]),
            "onset_s": quantiles([x["onset"] for x in params]), "offset_s": quantiles([x["offset"] for x in params]),
            "rejected": reasons,
        }
        for key in ("symmetry", "direction", "dominant"):
            if params and key in params[0]:
                entry[key] = {v: sum(x[key] == v for x in params) for v in sorted({x[key] for x in params})}

        # coverage: frames where some driver is >= dev_high from rest, explained by ANY phrase claiming that
        # driver's channel (blink and lids_lowered share the eyelids and split them by duration)
        hit = total = 0
        driver_texts = {l2.split_ref(ref)[0] for ref in spec["drivers"]}
        for text, t_l2 in zip(l1_texts, document["tracks"]):
            values = l2.channel_values(text, names, unit)
            inside = np.zeros(int(text["n_frames"]), dtype=bool)
            for p in t_l2["phrases"]:
                if driver_texts & set(l2.text_channels_of(l2.VOCABULARY["phrases"][p["type"]])):
                    inside[l1._frame(p["t"][0], hz):l1._frame(p["t"][1], hz)] = True
            strong = np.zeros_like(inside)
            for ref in spec["drivers"]:
                series, cov = l2.ref_series(values, ref)
                dev = series - l2.ref_rest(t_l2["rest"], ref)
                strong |= cov & ((np.abs(dev) if spec["sign"] == "signed" else dev) >= spec["detect"]["dev_high"])
            hit += int((strong & inside).sum())
            total += int(strong.sum())
        entry["coverage_of_dev_high_frames"] = hit / max(total, 1)

        # phrase-only decode vs the exact L1 inside this type's spans, TEST speakers only
        entry["phrase_only_test"] = {}
        for method in generated:
            agree = n = 0
            errors = {}
            for k, i in enumerate(test):
                t_l2 = document["tracks"][i]
                for p in t_l2["phrases"]:
                    if p["type"] != name:
                        continue
                    s0, e0 = l1._frame(p["t"][0], hz), l1._frame(p["t"][1], hz)
                    for ref in spec["members"]:
                        text_name, _ = l2.split_ref(ref)
                        q = l1.quantizer(l2.schema_of(text_name), unit)
                        a = l2.ref_series(test_values[k], ref)[0][s0:e0]
                        b = l2.ref_series(generated_values[method][k], ref)[0][s0:e0]
                        agree += int((l1.quantize(a, q) == l1.quantize(b, q)).sum())
                        n += e0 - s0
                        errors.setdefault(l2.schema_of(text_name), []).append(np.abs(a - b))
            entry["phrase_only_test"][method] = {
                "level_agree": agree / max(n, 1),
                "value_mae": {k: float(np.concatenate(v).mean()) for k, v in errors.items()}}
        entry["kb"] = {"n_train": kb["types"].get(name, {}).get("n", 0), "params": kb["types"].get(name, {}).get("params")}
        report["phrases"][name] = entry

    with open(os.path.join(out_dir, "l2_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    md = markdown(report)
    with open(os.path.join(out_dir, "l2_report.md"), "w", encoding="utf-8") as handle:
        handle.write(md)
    print(md)


def markdown(r: dict) -> str:
    c, t = r["checks"]["exact_round_trip"], r["track_level_l1"]
    lines = ["# L2 report", "",
             "%d tracks, %.0f described seconds, vocabulary %s; split %s %s." % (
                 r["n_tracks"], r["described_seconds"], r["vocabulary_version"], r["split"]["kind"], r["split"]["counts"]), "",
             "- exact round trip: **%s** (%d / %d tracks verbatim)" % ("PASS" if c["pass"] else "FAIL", c["tracks_ok"], c["tracks"]),
             "- track-level L1 vs original: bs MAE %.4f (max %.3f), translation MAE %.3f, rotation %.2f° (max %.1f°), gaze MAE %.4f"
             % (t["bs_mae"], t["bs_max"], t["trans_mae"], t["rot_mean_deg"], t["rot_max_deg"], t["gaze_mae"]),
             "- compression: %d L1 segments -> %d residual segments + %d phrases (%.3f)"
             % (r["compression"]["l1_segments"], r["compression"]["residual_segments"], r["compression"]["phrases"],
                r["compression"]["l2_units_over_l1"]), "",
             "| phrase | count | per min (median) | duration s p10/med/p90 | coverage |", "|---|---|---|---|---|"]
    for name, e in r["phrases"].items():
        d = e["duration_s"]
        lines.append("| %s | %d | %.1f | %s | %.1f%% |" % (
            name, e["count"], e["per_minute_per_speaker"]["median"],
            "%.2f / %.2f / %.2f" % (d["p10"], d["median"], d["p90"]) if d["n"] else "-", 100 * e["coverage_of_dev_high_frames"]))
    methods = list(next(iter(r["phrases"].values()))["phrase_only_test"])
    lines += ["", "Phrase-only decode vs exact L1 inside phrase spans, TEST speakers (knowledge base and exemplar bank "
              "from TRAIN speakers). Level agreement per frame and member; value MAE by channel family.", "",
              "| phrase | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for name, e in r["phrases"].items():
        po = e["phrase_only_test"]
        cells = []
        for m in methods:
            mae = ", ".join("%s %.4f" % (k.replace("headpose_rotation_6d", "rot°").replace("blendshape", "bs"), v)
                            for k, v in po[m]["value_mae"].items())
            cells.append("%.1f%% (%s)" % (100 * po[m]["level_agree"], mae))
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    lines += ["", "Retrieval fit on VALIDATION speakers (per type): " + "; ".join(
        "%s k=%d w=%s val mismatch %.3f (json weights, k=1: %.3f)" % (kind, f["k"], f["weights"], f["val_level_mismatch"],
                                                                    f["val_level_mismatch_json_weights_k1"])
        for kind, f in r.get("retrieval_fit", {}).items())]
    lines += ["", "Details per phrase (symmetry / direction / dominant axis, rejected candidates):", ""]
    for name, e in r["phrases"].items():
        extra = {k: e[k] for k in ("symmetry", "direction", "dominant") if k in e}
        lines.append("- **%s**: %s; rejected %s" % (name, extra, e["rejected"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
