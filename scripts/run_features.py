"""
CLI entrypoint for the feature-assembly step.

geometry.parquet -> uniformly-sampled, windowed tensors for a Continuous Temporal
Autoencoder v0, with the provenance needed to weight the loss correctly.

Pipeline within this stage:

    1. QC     drop clips by explicit, recorded policy. Timeline defects (from the
              Observation layer) are hard errors about the file; the detection-rate
              threshold is a configurable visual policy, measured by MediaPipe --
              the same model that produced the features, so the gate and the data
              agree by construction. Every rejection is counted with its reason;
              nothing is filtered silently.
    2. Resample  onto a uniform grid (30 Hz by default) from the real per-frame
              PTS carried in geometry.parquet, splitting at gaps longer than
              --max_gap_seconds so nothing is interpolated across unobserved time.
    3. Window slice the uniform series into fixed-length overlapping windows.
              Because every clip now shares one sample rate, all windows share one
              frame count and stack into a single tensor.

What is stored is the TARGET space. x_input is derived at load time by
src/sequence/contract.py:build_input(); storing both would double the disk and
guarantee divergence. Four provenance signals travel with the tensor -- observed,
quality, is_gap_filled, is_padding -- and are combined into a loss weight by
contract.loss_weight() rather than being pre-blended here, so the policy stays
visible and changeable.

Usage:
    python scripts/run_features.py \
        --geometry_parquet <dir>/geometry.parquet \
        --video_metadata <dir>/video_metadata.json \
        --output <dir>/features.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.schema import feature_vector as schema
from src.sequence.contract import describe_contract, sanitize
from src.sequence.resample import antialias_warning, resample_clip
from src.sequence.window import frames_for_seconds, hop_for_window, slice_windows

DEFAULT_TARGET_HZ = 30.0


def _required_columns() -> list:
    columns = ["clip_id", "frame_idx", "pts_sec", "detected"]
    for spec in schema.CHANNELS:
        source = spec.source_column if isinstance(spec.source_column, list) else [spec.source_column]
        columns.extend(source)
        columns.append(spec.quality_column)
    return sorted(set(columns))


def _load_quality_report(path: str) -> dict:
    """Clip-level labels from the Observation layer, when available."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("per_clip", payload) if isinstance(payload, dict) else payload
    return {record["file"]: record for record in records} if isinstance(records, list) else {}


def main():
    parser = argparse.ArgumentParser(description="Assemble uniformly-sampled, windowed feature tensors from geometry.parquet.")
    parser.add_argument("--geometry_parquet", required=True)
    parser.add_argument("--video_metadata", required=True, help="Supplies per-clip timeline context; per-frame PTS come from geometry.parquet.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation_quality", default=None,
                        help="Path to the Observation layer's quality.json, which supplies the timeline verdict. Without it, only the detection-rate policy applies.")

    resampling = parser.add_argument_group("resampling")
    resampling.add_argument("--target_hz", type=float, default=DEFAULT_TARGET_HZ,
                            help=f"Uniform sample rate for the model grid. Default {DEFAULT_TARGET_HZ}; a v0 engineering choice, not a theoretical requirement.")
    resampling.add_argument("--max_gap_seconds", type=float, default=0.2,
                            help="Unobserved stretches longer than this split the clip into separate segments; nothing is interpolated across one.")

    qc = parser.add_argument_group("quality control")
    qc.add_argument("--min_detection_rate", type=float, default=0.7,
                    help="Reject a clip whose MediaPipe-tracked fraction falls below this. A configurable visual policy, not a hard error.")
    qc.add_argument("--keep_tail_gap_frame", action="store_true",
                    help="Use the final frame of a clip whose only timeline issue is tail_frame_gap. "
                         "Off by default: that frame is excluded and the clip kept.")
    qc.add_argument("--allow_timeline_issues", action="store_true",
                    help="Keep clips whose PTS is non-monotonic or incomplete. Off by default: those are hard errors about the file.")
    qc.add_argument("--min_segment_seconds", type=float, default=None,
                    help="Drop resampled segments shorter than this. Defaults to one window.")

    windowing = parser.add_argument_group("windowing")
    windowing.add_argument("--window_seconds", type=float,
                           default=float(schema.WINDOWING["default_target_seconds"]))
    windowing.add_argument("--overlap_ratio", type=float,
                           default=float(schema.WINDOWING["default_overlap_ratio"]))
    windowing.add_argument("--symmetric_blendshape", action="store_true",
                           help="Record that the encoder input should use the [-1,1] blendshape map. Stored targets stay in [0,1]; the map is applied at load time.")
    windowing.add_argument("--clip_id", action="append", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.geometry_parquet)
    missing = [column for column in _required_columns() if column not in df.columns]
    if missing:
        print(f"geometry.parquet is missing columns required by schema v{schema.SCHEMA_VERSION}: {missing}")
        print("Regenerate it with scripts/run_geometry.py.")
        sys.exit(1)

    with open(args.video_metadata, encoding="utf-8") as handle:
        metadata_by_clip = {m["file"]: m for m in json.load(handle)}
    observation_quality = _load_quality_report(args.observation_quality)

    clip_ids = sorted(df["clip_id"].unique())
    if args.clip_id:
        clip_ids = [c for c in clip_ids if any(c.startswith(prefix) for prefix in args.clip_id)]
    if not clip_ids:
        print(f"No matching clips in {args.geometry_parquet}")
        sys.exit(1)

    window = frames_for_seconds(args.window_seconds, args.target_hz)
    hop = hop_for_window(window, args.overlap_ratio)
    min_segment_frames = (frames_for_seconds(args.min_segment_seconds, args.target_hz)
                          if args.min_segment_seconds else window)
    print(f"Grid: {args.target_hz} Hz | window {window} frames ({args.window_seconds}s), hop {hop}")
    print(f"QC: min_detection_rate={args.min_detection_rate}, "
          f"timeline issues {'allowed' if args.allow_timeline_issues else 'rejected'}, "
          f"max_gap={args.max_gap_seconds}s")
    print()

    kept, rejected = [], []
    feature_windows, quality_windows = [], []
    gap_windows, padding_windows, observed_windows = [], [], []
    window_clip, window_start, window_segment, window_time = [], [], [], []

    for clip_id in clip_ids:
        clip_df = df[df["clip_id"] == clip_id].sort_values("frame_idx")
        meta = metadata_by_clip.get(clip_id, {})
        report = observation_quality.get(clip_id, {})

        def reject(reason, detail=""):
            rejected.append({"clip_id": clip_id, "reason": reason, "detail": detail})
            print(f"  REJECTED {clip_id}: {reason} {detail}")

        timeline_issues = report.get("pts_completeness", {}).get("issues", [])
        # A gap before the final frame only (tail_frame_gap) is a cut artefact on one
        # frame. Policy: keep the clip and EXCLUDE that frame from use -- the timeline
        # itself is untouched and the exclusion is recorded. Any other timeline issue
        # still rejects the whole clip unless --allow_timeline_issues.
        blocking = [i for i in timeline_issues if i["code"] != "tail_frame_gap"]
        exclude_tail = len(blocking) < len(timeline_issues) and not args.keep_tail_gap_frame
        if blocking and not args.allow_timeline_issues:
            reject("timeline_defect", ",".join(i["code"] for i in timeline_issues))
            continue

        times = np.asarray(pd.to_numeric(clip_df["pts_sec"], errors="coerce"), dtype=np.float64)
        detected = np.asarray(clip_df["detected"], dtype=bool)
        if times.shape[0] < 2:
            reject("too_short", f"{times.shape[0]} frames")
            continue

        detection_rate = float(detected.mean())
        if detection_rate < args.min_detection_rate:
            reject("low_detection_rate", f"{detection_rate:.4f} < {args.min_detection_rate}")
            continue

        features, quality = schema.from_dataframe(clip_df)
        features, was_invalid = sanitize(features)

        # Frame-level observation drives segmentation; per-channel invalidity does
        # NOT. A blink makes gaze unavailable for a few frames while landmarks,
        # blendshapes and head pose stay perfectly tracked. Folding that into a
        # frame-level "invalid" would treat a blink as unobserved time, split the
        # clip at every blink, and throw away good supervision for three channels
        # to protect one. Per-channel loss is exactly what the quality mask is for,
        # so an invalid channel has its quality zeroed and the frame stays whole.
        valid = detected & np.isfinite(times)
        if exclude_tail:
            valid[-1] = False
        quality = np.where(was_invalid, 0.0, quality)
        if valid.sum() < 2:
            reject("no_observed_frames", f"{int(valid.sum())} observed frames")
            continue

        source_hz = None
        finite_steps = np.diff(times[np.isfinite(times)])
        if finite_steps.size:
            median_step = float(np.median(finite_steps))
            source_hz = (1.0 / median_step) if median_step > 0 else None
        warning = antialias_warning(source_hz, args.target_hz)
        if warning:
            print(f"  WARNING {clip_id}: {warning}")

        resampled = resample_clip(features, quality, times, target_hz=args.target_hz,
                                  max_gap_seconds=args.max_gap_seconds, valid=valid)
        if resampled["n_segments"] == 0:
            reject("no_continuous_segment", f"no stretch of >=2 valid frames within {args.max_gap_seconds}s gaps")
            continue

        # Observed-ness after resampling: a grid sample inherits the detection
        # state of the source frame nearest it, so a stretch that was never
        # tracked cannot be laundered into "observed" by interpolation.
        grid_observed = detected[resampled["source_index"]]

        clip_windows = 0
        for segment_id, (start, stop) in enumerate(resampled["segment_spans"]):
            length = stop - start
            if length < min_segment_frames:
                rejected.append({"clip_id": clip_id, "reason": "segment_too_short",
                                 "detail": f"segment {segment_id}: {length} < {min_segment_frames} frames"})
                continue

            segment_features = resampled["features"][start:stop]
            segment_quality = resampled["quality"][start:stop]
            segment_gap = resampled["is_gap_filled"][start:stop]
            segment_observed = grid_observed[start:stop]

            sliced = slice_windows(segment_features, segment_quality, window=window, hop=hop)
            if sliced["features"].shape[0] == 0:
                continue

            # Gap and observed masks ride through the same slicing call as the
            # features so they stay frame-aligned.
            extras = np.stack([
                segment_gap.astype(np.float64),
                segment_observed.astype(np.float64),
            ], axis=1)
            extra_windows = slice_windows(extras, segment_quality, window=window, hop=hop)["features"]

            # Padding is computed from the window geometry, NOT carried through
            # slice_windows: that function pads features by REPEATING the edge
            # frame, so a boolean smuggled in as a feature column comes back as
            # the repeated edge value rather than as a padding marker. Deriving it
            # from the start offsets is exact and cannot be fooled.
            starts = np.asarray(sliced["starts"], dtype=np.int64)
            offsets = np.arange(window, dtype=np.int64)[None, :]
            padding_mask = (starts[:, None] + offsets) >= length

            feature_windows.append(sliced["features"])
            quality_windows.append(sliced["quality"])
            gap_windows.append((extra_windows[:, :, 0] > 0.5) & ~padding_mask)
            # A padded frame was not observed; the edge value repeated into it says
            # nothing about whether a face was there.
            observed_windows.append((extra_windows[:, :, 1] > 0.5) & ~padding_mask)
            padding_windows.append(padding_mask)
            window_clip.extend([clip_id] * len(sliced["starts"]))
            window_start.extend([start + s for s in sliced["starts"]])
            # The grid time of each window's first frame, on the container timeline
            # the video PTS came from. It is what lets a sidecar stream (audio) be
            # cut for this window by real time instead of by position.
            window_time.extend([float(resampled["times"][start + s]) for s in sliced["starts"]])
            window_segment.extend([segment_id] * len(sliced["starts"]))
            clip_windows += len(sliced["starts"])

        if clip_windows == 0:
            reject("no_window_fits", f"{resampled['n_segments']} segment(s), none >= {window} frames")
            continue

        kept.append({
            "clip_id": clip_id,
            "speaker_id": (clip_df["speaker_id"].iloc[0] if "speaker_id" in clip_df.columns else None),
            "source_video_id": (clip_df["source_video_id"].iloc[0] if "source_video_id" in clip_df.columns else None),
            "n_source_frames": int(times.shape[0]), "n_valid_frames": int(valid.sum()),
            "detection_rate": round(detection_rate, 4),
            "source_hz": round(source_hz, 4) if source_hz else None,
            "n_segments": resampled["n_segments"], "n_resampled_frames": int(resampled["features"].shape[0]),
            "tail_frame_excluded": bool(exclude_tail),
            "n_gap_filled": int(resampled["is_gap_filled"].sum()), "n_windows": clip_windows,
            "fps_declared": meta.get("fps_declared"),
        })
        print(f"  {clip_id}: {clip_windows} windows from {resampled['n_segments']} segment(s), "
              f"{int(resampled['is_gap_filled'].sum())} gap-filled samples")

    if not feature_windows:
        print("\nNo clip produced windows.")
        summary = {"kept": kept, "rejected": rejected}
        with open(os.path.splitext(args.output)[0] + "_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        sys.exit(1)

    features = np.concatenate(feature_windows).astype(np.float32)
    quality = np.concatenate(quality_windows).astype(np.float32)
    is_gap_filled = np.concatenate(gap_windows)
    is_padding = np.concatenate(padding_windows)
    observed = np.concatenate(observed_windows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features, quality=quality, observed=observed,
        is_gap_filled=is_gap_filled, is_padding=is_padding,
        clip_id=np.array(window_clip), window_start=np.array(window_start, dtype=np.int32),
        segment_id=np.array(window_segment, dtype=np.int32),
        window_time_sec=np.array(window_time, dtype=np.float64),
        channel_names=np.array(schema.CHANNEL_NAMES), schema_version=np.array(schema.SCHEMA_VERSION),
        target_hz=np.array(args.target_hz),
    )

    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "schema_file": os.path.relpath(schema.SCHEMA_PATH, REPO_ROOT),
        "total_dim": schema.TOTAL_DIM,
        "channels": [{"name": s.name, "slice": [s.start, s.end], "dim": s.dim} for s in schema.CHANNELS],
        "grid": {"target_hz": args.target_hz, "window_frames": window, "hop_frames": hop,
                 "window_seconds": args.window_seconds, "overlap_ratio": args.overlap_ratio,
                 "max_gap_seconds": args.max_gap_seconds},
        "contract": describe_contract(symmetric_blendshape=args.symmetric_blendshape),
        "tensor_shapes": {"features": list(features.shape), "quality": list(quality.shape)},
        "qc": {
            "detection_rate_source": "MediaPipe Face Landmarker `detected` column in geometry.parquet",
            "policy": {"min_detection_rate": args.min_detection_rate,
                       "allow_timeline_issues": args.allow_timeline_issues,
                       "min_segment_frames": min_segment_frames},
            "n_clips_seen": len(clip_ids), "n_clips_kept": len(kept), "n_rejections": len(rejected),
            "rejections": rejected,
        },
        "per_clip": kept,
        "limitations": [
            "speaker_id is null unless run_geometry.py was given --speaker_from; without it a split "
            "over these windows is clip-disjoint only and must not be called identity-disjoint",
            "head-pose translation and gaze are clip-mean-subtracted, which removes any sustained "
            "pose or gaze offset along with the camera placement; a persistent head tilt is not recoverable "
            "from headpose_t alone (headpose_t_raw in geometry.parquet retains it)",
            "no channel is identity-disentangled: blendshape amplitude and head translation both carry "
            "speaker-specific structure, and landmark canonicalization does not change that",
        ],
    }
    manifest_path = os.path.splitext(args.output)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("\n--- Summary ---")
    print(f"kept {len(kept)}/{len(clip_ids)} clips, {features.shape[0]} windows of {window} frames @ {args.target_hz} Hz")
    if rejected:
        counts = {}
        for record in rejected:
            counts[record["reason"]] = counts.get(record["reason"], 0) + 1
        print("rejections: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"gap-filled samples: {int(is_gap_filled.sum())}, padded: {int(is_padding.sum())}, "
          f"unobserved: {int((~observed).sum())}")
    print(f"\nfeatures {features.shape}  (schema v{schema.SCHEMA_VERSION})")
    print(f"Written: {args.output}")
    print(f"Written: {manifest_path}")


if __name__ == "__main__":
    main()
