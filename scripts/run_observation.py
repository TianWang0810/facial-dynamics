"""
CLI entrypoint for the Observation layer.
Runs PTS validation + face-detection quality check on all mp4 files under
a directory, then writes video_metadata.json and quality.json.

Usage:
    python scripts/run_observation.py --input_dir <dir_of_downloaded_clips>
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.observation.pts import analyze_video
from src.observation.quality import analyze_video_faces
from src.observation.merge import merge_results


def main():
    parser = argparse.ArgumentParser(description="Run Observation-layer validation on a directory of mp4 clips.")
    parser.add_argument("--input_dir", required=True, help="Directory containing downloaded mp4 clips (searched recursively).")
    parser.add_argument("--output_dir", default=None, help="Where to write video_metadata.json / quality.json. Defaults to --input_dir.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.mp4"), recursive=True))
    if not mp4_files:
        print(f"No mp4 files found under {args.input_dir}")
        sys.exit(1)

    pts_results, face_results = [], []
    for mp4_path in mp4_files:
        print(f"Processing: {mp4_path}")
        try:
            pts_results.append(analyze_video(mp4_path))
            face_results.append(analyze_video_faces(mp4_path))
        except Exception as e:
            print(f"  FAILED: {e}")

    video_metadata_list, quality_list = merge_results(pts_results, face_results)

    with open(os.path.join(output_dir, "video_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(video_metadata_list, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "quality.json"), "w", encoding="utf-8") as f:
        json.dump(quality_list, f, indent=2, ensure_ascii=False)

    print("\n--- Summary ---")
    for q in quality_list:
        print(f"{q['file']}: {q['clip_quality_label']}")
    print(f"\nWritten: {output_dir}/video_metadata.json")
    print(f"Written: {output_dir}/quality.json")


if __name__ == "__main__":
    main()
