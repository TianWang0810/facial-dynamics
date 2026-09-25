"""
CLI entrypoint for the L1 round-trip error report.

Answers the acceptance questions of docs/plans/multilayer_semantic_representation.md
section 4.4 on a real run, with every number attributable to one step:

    A  per_frame          quantize -> dequantize each frame, no text, no segments.
                          The pure quantization cost: L1's lower bound for levels.
    B  segmented_step     encode -> JSON text -> decode from level representatives,
                          no smoothing. B - A is what segmentation adds.
    D  segmented          the same text through the DEFAULT decode (rules: refined
                          source + Whittaker smoothing). D - B is what the
                          post-processing buys, with the text unchanged.
    C  segment_means      value_hint, no smoothing: the piecewise-constant floor.
    sweep                 D at several min_segment_seconds (level check at 0 uses B).

Checks that must hold (the report exits non-zero if any fails):
    - run-length segmentation alone (min 0) reproduces A exactly
    - the four mask signals survive the round trip and loss_weight() on the
      decoded arrays reproduces the described mask
    - every channel has described frames (otherwise the checks above are vacuous)
    - no segment is more than one level from any of its frames' own level
    - every value_hint lies inside its level's bin (so refined decode keeps it)
    - --text (a file written by l1_encode.py in another process) is identical,
      as a whole document, to a fresh encode: the real determinism check

Every derivative number comes from src/dynamics/derive.py via src/semantic/l1_eval.py.

Usage:
    python scripts/l1_report.py --run_dir ../runs/<run_id> --output_dir <dir>
    python scripts/l1_report.py --run_dir ... --output_dir <dir> --text <dir>/l1_text.json
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.schema.feature_vector import CHANNELS, channel
from src.semantic.l1 import (
    MIN_SEGMENT_SECONDS, RULES_VERSION, _frame, bin_bounds, channel_levels, decode_document, describable,
    encode_features, parse_label, quantize_window, quantizer, text_channels,
)
from src.semantic.l1_eval import dynamics_errors, text_statistics, value_errors
from src.semantic.l1_io import add_input_arguments, load_inputs

SWEEP_SECONDS = (0.0, 2.0 / 30.0, 0.1, 0.2)
BLINK_CHANNELS = ("eyeBlinkLeft", "eyeBlinkRight")


def _roundtrip(document: dict, source: str = None, smooth: bool = None) -> dict:
    """Serialise to JSON text and back before decoding, so the text IS the interface."""
    return decode_document(json.loads(json.dumps(document)), source=source, smooth=smooth)


def _get(tree: dict, *path):
    """Nested lookup that yields None for a channel that is disabled or had no described frames."""
    for key in path:
        if not isinstance(tree, dict) or tree.get(key) is None:
            return None
        tree = tree[key]
    return tree


def _headline(values: dict, dynamics: dict) -> dict:
    return {
        "blendshape_mae": _get(values, "blendshape", "mae"),
        "blendshape_max_abs": _get(values, "blendshape", "max_abs"),
        "blendshape_bias": _get(values, "blendshape", "bias"),
        "translation_mae": _get(values, "headpose_translation", "mae"),
        "rotation_mean_deg": _get(values, "headpose_rotation_6d", "mean"),
        "rotation_max_deg": _get(values, "headpose_rotation_6d", "max"),
        "gaze_mae": _get(values, "gaze", "mae"),
        "blendshape_velocity_mae": _get(dynamics, "blendshape", "velocity", "mae"),
        "blendshape_velocity_mean_abs_recon": _get(dynamics, "blendshape", "velocity", "mean_abs_recon"),
        "blendshape_velocity_mean_abs_original": _get(dynamics, "blendshape", "velocity", "mean_abs_original"),
    }


def _difference(a, b):
    return None if a is None or b is None else a - b


def _fmt(value, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _hints_consistent(document: dict, names: list) -> bool:
    """Unedited text: every value_hint lies inside its level's bin (what refined decode relies on)."""
    for window in document["windows"]:
        for text_name, schema_name, _ in text_channels(names):
            q = quantizer(schema_name, document["gaze_unit"])
            for entry in window["channels"][text_name]:
                levels = entry["level"].values() if isinstance(entry["level"], dict) else [entry["level"]]
                hints = entry["value_hint"].values() if isinstance(entry["value_hint"], dict) else [entry["value_hint"]]
                for label, hint in zip(levels, hints):
                    low, high = bin_bounds(parse_label(label, q), q)
                    if not low <= hint <= high:
                        return False
    return True


def _max_level_shift(npz: dict, document: dict, described: np.ndarray, names: list, gaze_unit: str) -> int:
    """Largest distance between a segment's level and any of its frames' own per-frame level."""
    worst = 0
    for w, window in enumerate(document["windows"]):
        hz = float(window["hz"])
        for text_name, _, q, _, _, _, levels, _ in channel_levels(npz["features"][w], described[w], names, gaze_unit):
            for entry in window["channels"][text_name]:
                start, stop = _frame(entry["t"][0], hz), _frame(entry["t"][1], hz)
                labels = entry["level"].values() if isinstance(entry["level"], dict) else [entry["level"]]
                row = np.array([parse_label(label, q) for label in labels])
                worst = max(worst, int(np.abs(levels[start:stop] - row).max()))
    return worst


def _blink_example(npz: dict, document: dict, names: list) -> dict:
    """The window with the largest eyeBlinkLeft peak, and its blink text."""
    column = channel("blendshape").start + names.index(BLINK_CHANNELS[0])
    w = int(np.argmax(npz["features"][:, :, column].max(axis=1)))
    return {
        "window_id": document["windows"][w]["window_id"],
        "original_eyeBlinkLeft": [round(float(v), 3) for v in npz["features"][w, :, column]],
        "text": {name: document["windows"][w]["channels"][name] for name in BLINK_CHANNELS},
    }


def _markdown(report: dict) -> str:
    lines = ["# L1 round-trip report", "",
             f"- run: `{report['source']['features_npz']}`",
             f"- rules v{report['rules_version']}, min_segment_seconds={report['min_segment_seconds']}",
             f"- windows: {report['text']['n_windows']}, text channels: {report['text']['n_text_channels']}, "
             f"segments/channel/window: {report['text']['segments_per_channel_window_mean']:.2f}",
             "", "## Checks", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'} {name}" for name, ok in report["checks"].items()]
    lines += ["", "## Value and dynamics error (described frames only)", "",
              "| config | bs MAE | bs max | bs bias | trans MAE | rot mean deg | rot max deg | gaze MAE | bs vel MAE | bs mean abs vel recon / orig |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, h in report["headline"].items():
        gaze = _fmt(h["gaze_mae"], ".4f")
        lines.append(f"| {name} | {_fmt(h['blendshape_mae'], '.4f')} | {_fmt(h['blendshape_max_abs'], '.3f')} | "
                     f"{_fmt(h['blendshape_bias'], '+.4f')} | {_fmt(h['translation_mae'], '.4f')} | "
                     f"{_fmt(h['rotation_mean_deg'], '.2f')} | {_fmt(h['rotation_max_deg'], '.2f')} | {gaze} | "
                     f"{_fmt(h['blendshape_velocity_mae'], '.3f')} | {_fmt(h['blendshape_velocity_mean_abs_recon'], '.3f')} / "
                     f"{_fmt(h['blendshape_velocity_mean_abs_original'], '.3f')} |")
    lines += ["", "## Segmentation cost (segmented_step - per_frame)", ""]
    for key, value in report["segmentation_cost"].items():
        lines.append(f"- {key}: {_fmt(value, '+.5f')}")
    lines += ["", "## Post-processing effect (segmented - segmented_step; negative = better)", ""]
    for key, value in report["post_processing_gain"].items():
        lines.append(f"- {key}: {_fmt(value, '+.5f')}")
    lines += ["", "## Blink spot check", "", f"window `{report['blink_example']['window_id']}`", "",
              "original eyeBlinkLeft: " + ", ".join(str(v) for v in report["blink_example"]["original_eyeBlinkLeft"]), ""]
    for name, segments in report["blink_example"]["text"].items():
        lines.append(f"- {name}: " + " | ".join(f"{s['t'][0]:.3f}-{s['t'][1]:.3f} {s['level']} ({s['value_hint']})" for s in segments))
    lines += ["", "## Worst blendshapes by MAE (segmented)", ""]
    per_name = report["configs"]["segmented"]["values"].get("blendshape", {}).get("per_name_mae", {})
    worst = sorted(((n, v) for n, v in per_name.items() if v is not None), key=lambda kv: -kv[1])[:10]
    lines += [f"- {n}: {v:.4f}" for n, v in worst]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Measure L1 encode/decode round-trip error on a pipeline run.")
    add_input_arguments(parser)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text", default=None, help="An l1_text.json from scripts/l1_encode.py to cross-check against a fresh encode.")
    args = parser.parse_args()

    npz, names, gaze_unit, source = load_inputs(args)
    original = npz["features"].astype(np.float64)
    hz = float(npz["target_hz"])
    described = np.stack([describable(npz["observed"][w], npz["quality"][w], npz["is_gap_filled"][w],
                                      npz["is_padding"][w]) for w in range(original.shape[0])])

    per_frame = np.stack([quantize_window(original[w], names, gaze_unit, described[w])
                          for w in range(original.shape[0])])
    document = encode_features(npz, names, gaze_unit, source=source)
    step = _roundtrip(document, "level", smooth=False)
    segmented = _roundtrip(document)
    means = _roundtrip(document, "value_hint", smooth=False)

    recon = {"per_frame": per_frame, "segmented_step": step["features"], "segmented": segmented["features"],
             "segment_means": means["features"]}
    configs = {name: {"values": value_errors(original, array, described, names),
                      "dynamics": dynamics_errors(original, array, described, hz, names)}
               for name, array in recon.items()}

    sweep = {}
    run_length_only = None
    for seconds in SWEEP_SECONDS:
        doc = encode_features(npz, names, gaze_unit, min_segment_seconds=seconds)
        decoded = _roundtrip(doc, "level", smooth=False)["features"]
        default_decode = _roundtrip(doc)["features"]
        if seconds == 0.0:
            run_length_only = decoded
        values = value_errors(original, default_decode, described, names)
        sweep["%.4f" % seconds] = {
            "blendshape_mae": _get(values, "blendshape", "mae"), "blendshape_bias": _get(values, "blendshape", "bias"),
            "rotation_mean_deg": _get(values, "headpose_rotation_6d", "mean"),
            "segments_per_channel_window": text_statistics(doc)["segments_per_channel_window_mean"],
        }

    decoded_described = np.stack([describable(segmented["observed"][w], segmented["quality"][w],
                                              segmented["is_gap_filled"][w], segmented["is_padding"][w])
                                  for w in range(original.shape[0])])
    dims_described = np.concatenate([np.repeat(described[..., [k]], spec.dim, axis=-1)
                                     for k, spec in enumerate(CHANNELS)], axis=-1)
    checks = {
        "every_channel_has_described_frames": bool(described.any(axis=(0, 1)).all()),
        "run_length_segmentation_equals_per_frame": bool(dims_described.any() and np.allclose(
            run_length_only[dims_described], per_frame[dims_described].astype(np.float32), atol=1e-6)),
        "segmentation_moves_no_frame_more_than_one_level":
            _max_level_shift(npz, document, described, names, gaze_unit) <= 1,
        "every_hint_lies_in_its_level_bin": _hints_consistent(document, names),
        "masks_round_trip_exactly": bool(
            np.array_equal(segmented["is_padding"], npz["is_padding"])
            and np.array_equal(segmented["is_gap_filled"], npz["is_gap_filled"])
            and np.array_equal(segmented["observed"], npz["observed"])),
        "loss_weight_reproduces_described_mask": bool(np.array_equal(decoded_described, described)),
    }
    if args.text:
        with open(args.text, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        # Written by another process (scripts/l1_encode.py): this is the real
        # determinism check, over the whole document, not just the windows.
        checks["text_file_matches_fresh_encode"] = on_disk == json.loads(json.dumps(document))

    per_frame_values = configs["per_frame"]["values"]
    step_values = configs["segmented_step"]["values"]
    segmented_values = configs["segmented"]["values"]
    cost_paths = (("blendshape_mae", ("blendshape", "mae")), ("blendshape_max_abs", ("blendshape", "max_abs")),
                  ("blendshape_bias", ("blendshape", "bias")), ("translation_mae", ("headpose_translation", "mae")),
                  ("rotation_mean_deg", ("headpose_rotation_6d", "mean")))
    report = {
        "rules_version": RULES_VERSION,
        "min_segment_seconds": MIN_SEGMENT_SECONDS,
        "source": source,
        "n_described_frames_per_channel": {spec.name: int(described[..., k].sum()) for k, spec in enumerate(CHANNELS)},
        "checks": checks,
        "headline": {name: _headline(c["values"], c["dynamics"]) for name, c in configs.items()},
        "segmentation_cost": {key: _difference(_get(step_values, *path), _get(per_frame_values, *path))
                              for key, path in cost_paths},
        "post_processing_gain": {key: _difference(_get(segmented_values, *path), _get(step_values, *path))
                                 for key, path in cost_paths},
        "sweep_min_segment_seconds": sweep,
        "text": text_statistics(document),
        "blink_example": _blink_example(npz, document, names),
        "configs": configs,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "l1_report.json")
    md_path = os.path.join(args.output_dir, "l1_report.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_markdown(report))
    print(_markdown(report))
    print(f"Written: {json_path}")
    print(f"Written: {md_path}")
    if not all(checks.values()):
        print("FAIL: " + ", ".join(name for name, ok in checks.items() if not ok))
        sys.exit(1)


if __name__ == "__main__":
    main()
