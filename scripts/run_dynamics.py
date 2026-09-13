"""
CLI entrypoint for the Dynamics layer.
Recomputes velocity / acceleration / jerk from a geometry.parquet plus the
matching video_metadata.json, and prints a validation summary (mean / max /
NaN checks) per clip.

Per design doc section 3 the Dynamics layer is derived on demand and never
persisted, so this script deliberately writes no per-frame output; only the
summary is written, and only when --report_json is given.

Real per-frame PTS come from the pts_sec array in video_metadata.json, so no
source mp4 is re-decoded and timing is never reconstructed as frame_idx / fps.
A video_metadata.json produced before pts_sec existed will be rejected with a
message telling you to re-run run_observation.py.

Usage:
    python scripts/run_dynamics.py \
        --geometry_parquet <dir>/geometry.parquet \
        --video_metadata <dir>/video_metadata.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.dynamics.derive import summarize_dynamics

SIGNAL_SHAPES = {"landmarks": (-1, 3), "blendshapes": (-1,)}


def main():
    parser = argparse.ArgumentParser(description="Recompute Dynamics-layer derivatives and print a validation summary.")
    parser.add_argument("--geometry_parquet", required=True, help="Path to geometry.parquet produced by run_geometry.py")
    parser.add_argument("--video_metadata", required=True, help="Path to video_metadata.json produced by run_observation.py; supplies the per-frame pts_sec array.")
    parser.add_argument("--signal", choices=sorted(SIGNAL_SHAPES), default="landmarks", help="Which geometry column to differentiate.")
    parser.add_argument("--clip_id", action="append", default=None,
                        help="Restrict to clip ids starting with this prefix; repeatable. "
                             "Use --clip_id=-04ZSRBGcsk form, since clip ids start with a dash.")
    parser.add_argument("--report_json", default=None, help="Optional path to write the summary as JSON. No per-frame data is ever written.")
    args = parser.parse_args()

    with open(args.video_metadata, encoding="utf-8") as f:
        metadata_by_clip = {os.path.splitext(m["file"])[0]: m for m in json.load(f)}

    df = pd.read_parquet(args.geometry_parquet)
    clip_ids = sorted(df["clip_id"].unique())
    if args.clip_id:
        clip_ids = [c for c in clip_ids if any(c.startswith(prefix) for prefix in args.clip_id)]
    if not clip_ids:
        print(f"No matching clips in {args.geometry_parquet}")
        sys.exit(1)

    summaries = []
    for clip_id in clip_ids:
        print(f"Processing: {clip_id}")
        meta = metadata_by_clip.get(clip_id)
        if meta is None:
            print(f"  SKIPPED: no entry for {clip_id} in {args.video_metadata}")
            continue

        if not meta.get("pts_sec"):
            print(f"  SKIPPED: no pts_sec array for {clip_id} -- regenerate {os.path.basename(args.video_metadata)} with run_observation.py")
            continue

        try:
            timestamps = np.asarray(meta["pts_sec"], dtype=np.float64)
            clip_df = df[df["clip_id"] == clip_id].sort_values("frame_idx")
            values = np.array(clip_df[args.signal].tolist(), dtype=np.float64)
            values = values.reshape((len(clip_df),) + SIGNAL_SHAPES[args.signal])

            if len(timestamps) != len(clip_df):
                print(f"  SKIPPED: {len(clip_df)} geometry frames vs {len(timestamps)} PTS entries -- refusing to guess the alignment")
                continue
            if meta.get("n_frames_decoded") not in (None, len(timestamps)):
                print(f"  WARNING: video_metadata reports n_frames_decoded={meta['n_frames_decoded']} but pts_sec has {len(timestamps)} entries")

            summary = summarize_dynamics(values, timestamps)
            summary["clip_id"] = clip_id
            summary["signal"] = args.signal
            summaries.append(summary)
        except Exception as e:
            print(f"  FAILED: {e}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\n--- Summary ---")
    for s in summaries:
        step = s["timestep_sec"]
        print(f"\n{s['clip_id']}  ({s['signal']}, {s['n_frames']} frames, {s['duration_sec']:.3f}s)")
        print(f"  timestep: min={step['min']:.6f}s max={step['max']:.6f}s uniform={step['is_uniform']}")
        print(f"  input NaN frames: {s['n_nan_frames_input']}")
        for name in ("velocity", "acceleration", "jerk"):
            d = s[name]
            print(f"  {name:<12} mean={d['mean']:.6f} max={d['max']:.6f}  "
                  f"interior mean={d['interior_mean']:.6f} max={d['interior_max']:.6f}  "
                  f"NaN frames={d['n_nan_frames']} all_finite={d['all_finite']}")
    print(f"\nSummarized {len(summaries)}/{len(clip_ids)} clips (no per-frame data written).")
    if args.report_json:
        print(f"Written: {args.report_json}")


if __name__ == "__main__":
    main()
