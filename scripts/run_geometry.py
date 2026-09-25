"""
CLI entrypoint for the Geometry layer.

Runs the three-track extraction (landmarks+blendshapes / head pose / gaze) over
all mp4 files under a directory and writes geometry.parquet,
geometry_metadata.json and geometry_quality.json.

Usage:
    python scripts/run_geometry.py --input_dir <dir> --model_path <face_landmarker.task>

    # with a dedicated gaze model instead of the iris-geometry fallback
    python scripts/run_geometry.py --input_dir <dir> --model_path <face_landmarker.task> \
        --gaze_backend external --gaze_model_path <gaze.onnx>
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.common.ids import infer_grouping
from src.geometry.confidence import build_clip_mask_report
from src.geometry.gaze import build_backend
from src.geometry.pipeline import run_clip
from src.geometry.summarize import build_geometry_metadata, build_quality_report, load_and_flatten


def main():
    parser = argparse.ArgumentParser(description="Run three-track Geometry-layer extraction on a directory of mp4 clips.")
    parser.add_argument("--input_dir", required=True, help="Directory containing downloaded mp4 clips (searched recursively).")
    parser.add_argument("--model_path", required=True, help="Path to face_landmarker.task")
    parser.add_argument("--output_dir", default=None, help="Where to write geometry outputs. Defaults to --input_dir.")
    parser.add_argument("--gaze_backend", choices=("iris", "external"), default="iris",
                        help="iris: zero-dependency iris-geometry fallback inside env-v0.1. "
                             "external: dedicated onnxruntime gaze model (needs env-v0.2).")
    parser.add_argument("--gaze_model_path", default=None, help="Gaze model file, required when --gaze_backend external.")
    parser.add_argument("--gaze_tanh", action="store_true",
                        help="Compress zero-meaned gaze angles into the physiological range with a tanh. Requires a backend whose output is in radians.")
    parser.add_argument("--speaker_from", choices=("parent_dir", "top_dir"), default=None,
                        help="Where to read a speaker id from in the directory layout. Left unset, speaker_id is null and any split built on this data is clip-disjoint only, NOT identity-disjoint.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.mp4"), recursive=True))
    if not mp4_files:
        print(f"No mp4 files found under {args.input_dir}")
        sys.exit(1)

    gaze_backend = build_backend(args.gaze_backend, args.gaze_model_path)
    print(f"Gaze backend: {gaze_backend.name} (unit: {gaze_backend.unit}, calibrated: {gaze_backend.is_calibrated})")

    clip_reports = []
    skipped = []
    timelines = {}
    blendshape_names = None
    tanh_applied = args.gaze_tanh
    parquet_path = os.path.join(output_dir, "geometry.parquet")

    # Written clip by clip through a single open ParquetWriter rather than
    # accumulating every row in memory. At ~1500 floats per frame, holding a
    # whole dataset before the first write is what limits how much data this can
    # process; a clip at a time is bounded by the longest clip instead.
    writer = None
    n_rows = 0
    try:
        for mp4_path in mp4_files:
            identity = infer_grouping(mp4_path, args.input_dir, speaker_from=args.speaker_from)
            clip_id = identity["clip_id"]
            print(f"Processing: {clip_id}")
            try:
                clip_result = run_clip(mp4_path, args.model_path, gaze_backend, apply_gaze_tanh=args.gaze_tanh)
                rows = load_and_flatten(clip_result, clip_id, identity=identity)
                table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema)
                writer.write_table(table)
                n_rows += len(rows)

                clip_reports.append(build_clip_mask_report(
                    clip_result["confidences"], clip_result["detected"], clip_id))
                timelines[clip_id] = clip_result["timeline"]
                # Every clip must report the same category order; a mismatch means
                # the 52 positional columns would mean different things per clip.
                names = clip_result["blendshape_names"]
                if names:
                    if blendshape_names is None:
                        blendshape_names = names
                    elif names != blendshape_names:
                        raise RuntimeError("blendshape category order differs from earlier clips")
                tanh_applied = clip_result["gaze_tanh_applied"]
            except Exception as e:
                print(f"  FAILED: {e}")
                skipped.append({"clip_id": clip_id, "relative_path": identity["relative_path"],
                                "reason": "extraction_failed", "detail": str(e)})
    finally:
        if writer is not None:
            writer.close()
        gaze_backend.close()

    if writer is None:
        print("No clip produced any rows; writing an empty geometry.parquet would hide that.")
        with open(os.path.join(output_dir, "geometry_quality.json"), "w", encoding="utf-8") as f:
            json.dump({"per_clip": [], "skipped": skipped, "note": "no clip succeeded"}, f, indent=2)
        sys.exit(1)

    metadata = build_geometry_metadata(gaze_backend.name, tanh_applied,
                                       gaze_backend.unit, gaze_backend.is_calibrated,
                                       blendshape_names=blendshape_names or [])
    metadata["identity"] = {
        "clip_id_rule": "path relative to --input_dir, separators replaced by '__'",
        "speaker_from": args.speaker_from,
        "speaker_id_available": bool(args.speaker_from),
        "limitation": None if args.speaker_from else
                      "speaker_id is null: a split over these clips is clip-disjoint only and must NOT be described as identity-disjoint",
    }
    with open(os.path.join(output_dir, "geometry_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    quality = build_quality_report(clip_reports)
    quality["skipped_clips"] = skipped
    quality["timelines"] = timelines
    with open(os.path.join(output_dir, "geometry_quality.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2, ensure_ascii=False)

    print("\n--- Summary ---")
    for report in clip_reports:
        signals = report["per_signal_confidence"]
        print(f"\n{report['clip_id']}  ({report['n_frames']} frames, "
              f"tracking_failure_rate={report['tracking_failure_rate']:.4f}, label={report['clip_quality_label']})")
        for name, stats in signals.items():
            mean = stats["mean"]
            std = stats["std"]
            print(f"  conf_{name:<12} mean={mean if mean is None else round(mean, 4)} "
                  f"std={std if std is None else round(std, 4)} "
                  f"zero={stats['n_zero']} low_but_tracked={stats['n_low_but_tracked']}")

    if skipped:
        print(f"\nSkipped {len(skipped)} clip(s):")
        for record in skipped:
            print(f"  {record['clip_id']}: {record['reason']} -- {record['detail']}")

    print(f"\nWritten: {parquet_path} ({n_rows} rows, {len(clip_reports)} clips)")
    print(f"Written: {output_dir}/geometry_metadata.json")
    print(f"Written: {output_dir}/geometry_quality.json")


if __name__ == "__main__":
    main()
