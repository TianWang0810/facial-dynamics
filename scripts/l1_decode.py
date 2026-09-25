"""
CLI entrypoint for L1 decode: L1 text document -> tensor shaped like features.npz.

The output carries the same arrays as features.npz (features, quality, observed,
is_gap_filled, is_padding, clip_id, window_start, segment_id, channel_names,
schema_version, target_hz), so anything that consumes features.npz -- including
src/sequence/contract.py:loss_weight() -- consumes the decoded tensor unchanged.
quality is 1.0 where the text describes a frame and 0.0 elsewhere: the extractor's
quality proxy is not a facial attribute and is not carried by the text.

Usage:
    python scripts/l1_decode.py --text <dir>/l1_text.json --output <dir>/l1_decoded.npz
    python scripts/l1_decode.py ... --source level --no_smooth   # pure quantization view
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.semantic.l1 import DECODE_SOURCES, RULES, decode_document


def main():
    parser = argparse.ArgumentParser(description="Decode an L1 text document back into a features.npz-shaped tensor.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", choices=DECODE_SOURCES, default=None,
                        help="Default from schemas/l1_rules.json (refined: segment mean inside the level's bin, "
                             "else the level's representative). level: representatives only. "
                             "value_hint: raw segment means, for auditing.")
    parser.add_argument("--no_smooth", action="store_true",
                        help="Skip the rules' post-decode smoothing and emit the piecewise-constant tensor.")
    args = parser.parse_args()

    with open(args.text, encoding="utf-8") as handle:
        document = json.load(handle)
    decoded = decode_document(document, source=args.source, smooth=False if args.no_smooth else None)
    source = args.source or RULES["decode"]["default_source"]
    smoothing = "none" if args.no_smooth else RULES["decode"]["smoothing"]["method"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **decoded)
    print(f"Decoded {decoded['features'].shape[0]} windows -> features {decoded['features'].shape} (source={source}, smoothing={smoothing})")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
