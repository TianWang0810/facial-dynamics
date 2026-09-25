"""
CLI entrypoint for the Observation layer.

Validates the TIME AXIS of every mp4 under a directory and writes
video_metadata.json (with the full per-frame PTS array) and quality.json.

This layer no longer looks at pixels. It decodes timing only -- no RGB
conversion, no face detection -- which makes it a single cheap pass over each
container. The visual verdict comes from the Geometry layer's MediaPipe tracking
rate; see docs/decisions/observation_timeline_only.md for why the Haar-based
check that used to live here was removed rather than kept as a second opinion.

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

from src.common.ids import infer_grouping
from src.observation.pts import analyze_video
from src.observation.merge import merge_results


def main():
    parser = argparse.ArgumentParser(description="Validate the timeline of a directory of mp4 clips.")
    parser.add_argument("--input_dir", required=True, help="Directory containing downloaded mp4 clips (searched recursively).")
    parser.add_argument("--output_dir", default=None, help="Where to write video_metadata.json / quality.json. Defaults to --input_dir.")
    parser.add_argument("--speaker_from", choices=("parent_dir", "top_dir"), default=None,
                        help="Where to read a speaker id from in the directory layout; left unset, speaker_id is null.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.mp4"), recursive=True))
    if not mp4_files:
        print(f"No mp4 files found under {args.input_dir}")
        sys.exit(1)

    pts_results, skipped = [], []
    for mp4_path in mp4_files:
        identity = infer_grouping(mp4_path, args.input_dir, speaker_from=args.speaker_from)
        clip_id = identity["clip_id"]
        print(f"Processing: {clip_id}")
        try:
            result = analyze_video(mp4_path)
            result["file"] = clip_id
            result.update(identity)
            pts_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            skipped.append({"clip_id": clip_id, "relative_path": identity["relative_path"],
                            "reason": "observation_failed", "detail": str(e)})

    video_metadata_list, quality_list = merge_results(pts_results)

    with open(os.path.join(output_dir, "video_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(video_metadata_list, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "quality.json"), "w", encoding="utf-8") as f:
        json.dump({"per_clip": quality_list, "skipped_clips": skipped,
                   "speaker_from": args.speaker_from,
                   "scope": "timeline only; the visual verdict is the Geometry layer's tracking rate"},
                  f, indent=2, ensure_ascii=False)

    print("\n--- Summary ---")
    for q in quality_list:
        reasons = (" [" + ", ".join(q["reject_reasons"]) + "]") if q["reject_reasons"] else ""
        print(f"{q['file']}: timeline {q['timeline_label']}{reasons}")
    if skipped:
        print(f"\nSkipped {len(skipped)} clip(s):")
        for record in skipped:
            print(f"  {record['clip_id']}: {record['detail']}")
    print(f"\nWritten: {output_dir}/video_metadata.json")
    print(f"Written: {output_dir}/quality.json")


if __name__ == "__main__":
    main()
