"""
CLI entrypoint for the Geometry layer.
Runs landmark/blendshape/head-pose extraction + canonicalization on all mp4
files under a directory, then writes geometry.parquet, geometry_metadata.json,
and geometry_quality.json.

Usage:
    python scripts/run_geometry.py --input_dir <dir> --model_path <face_landmarker.task>
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.geometry.extract import extract_geometry
from src.geometry.canonicalization import canonicalize
from src.geometry.summarize import load_and_flatten, build_geometry_metadata, build_quality_report


def main():
    parser = argparse.ArgumentParser(description="Run Geometry-layer extraction on a directory of mp4 clips.")
    parser.add_argument("--input_dir", required=True, help="Directory containing downloaded mp4 clips (searched recursively).")
    parser.add_argument("--model_path", required=True, help="Path to face_landmarker.task")
    parser.add_argument("--output_dir", default=None, help="Where to write geometry outputs. Defaults to --input_dir.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.mp4"), recursive=True))
    if not mp4_files:
        print(f"No mp4 files found under {args.input_dir}")
        sys.exit(1)

    all_rows = []
    quality_per_clip = []

    for mp4_path in mp4_files:
        clip_id = os.path.splitext(os.path.basename(mp4_path))[0]
        print(f"Processing: {clip_id}")
        try:
            raw = extract_geometry(mp4_path, args.model_path)
            canon = canonicalize(raw["landmarks"])
            canon["blendshapes"] = raw["blendshapes"]
            canon["headpose"] = raw["headpose"]
            canon["detected"] = raw["detected"]

            rows = load_and_flatten(canon, clip_id)
            all_rows.extend(rows)

            n_frames = len(rows)
            n_detected = sum(1 for r in rows if r["detected"])
            tracking_failure_rate = 1.0 - (n_detected / n_frames if n_frames > 0 else 0.0)

            if tracking_failure_rate <= 0.05:
                clip_label = "usable"
            elif tracking_failure_rate <= 0.3:
                clip_label = "low-quality"
            else:
                clip_label = "reject"

            quality_per_clip.append({
                "clip_id": clip_id, "n_frames": n_frames, "n_tracked": n_detected,
                "tracking_failure_rate": round(tracking_failure_rate, 4),
                "clip_quality_label": clip_label,
            })
        except Exception as e:
            print(f"  FAILED: {e}")

    df = pd.DataFrame(all_rows)
    parquet_path = os.path.join(output_dir, "geometry.parquet")
    df.to_parquet(parquet_path, index=False)

    with open(os.path.join(output_dir, "geometry_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(build_geometry_metadata(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "geometry_quality.json"), "w", encoding="utf-8") as f:
        json.dump(build_quality_report(quality_per_clip), f, indent=2, ensure_ascii=False)

    print("\n--- Summary ---")
    for q in quality_per_clip:
        print(f"{q['clip_id']}: tracking_failure_rate={q['tracking_failure_rate']:.4f}, label={q['clip_quality_label']}")
    print(f"\nWritten: {parquet_path} ({len(df)} rows)")
    print(f"Written: {output_dir}/geometry_metadata.json")
    print(f"Written: {output_dir}/geometry_quality.json")


if __name__ == "__main__":
    main()
