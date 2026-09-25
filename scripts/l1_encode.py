"""
CLI entrypoint for L1 encode: features.npz (TARGET space) -> L1 text document.

Reads the tensor and the blendshape names / gaze unit the Geometry layer
persisted, and writes one JSON document holding the per-window, per-channel
segment text. Rules come from schemas/l1_rules.json; nothing is fitted, so the
same run always produces byte-identical text.

Usage:
    python scripts/l1_encode.py --run_dir ../runs/<run_id> --output <dir>/l1_text.json
    python scripts/l1_encode.py --features_npz <f>.npz --geometry_metadata <m>.json --output <t>.json
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.semantic.l1 import encode_features
from src.semantic.l1_io import add_input_arguments, load_inputs


def main():
    parser = argparse.ArgumentParser(description="Encode features.npz windows into L1 structured segment text.")
    add_input_arguments(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_segment_seconds", type=float, default=None,
                        help="Override the rules' minimum segment duration; 0 disables merging (pure run-length).")
    args = parser.parse_args()

    npz, names, gaze_unit, source = load_inputs(args)
    document = encode_features(npz, names, gaze_unit, min_segment_seconds=args.min_segment_seconds, source=source)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1, ensure_ascii=False)
    n_segments = sum(len(s) for w in document["windows"] for s in w["channels"].values())
    print(f"Encoded {len(document['windows'])} windows, {n_segments} segments "
          f"(rules v{document['rules_version']}, min_segment_seconds={document['min_segment_seconds']})")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
