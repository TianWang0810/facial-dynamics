"""
CLI entrypoint for the Audio stage (04_audio): an audio conditioning sidecar
aligned window-for-window with features.npz.

For each clip, the audio stream is decoded once (src/audio/features.py), each
chunk placed at its own PTS, and every features.npz window of that clip gets the
audio covering [window_time_sec, window_time_sec + window_seconds) -- cut by the
same container-timeline times the video grid uses, never by position. Row i of
audio.npz is the audio for row i of features.npz, and audio.npz carries
clip_id / window_start / window_time_sec so a reader can assert that.

Audio is conditioning, not target: it does not touch the 63-dim tensor or
schemas/geometry.schema.json. A clip with no audio stream is not an error; its
windows get audio_valid = False everywhere and the manifest says so.

Usage:
    python scripts/run_audio.py --features_npz <run>/02_features/features.npz \
        --video_metadata <run>/00_observation/video_metadata.json \
        --input_dir <mp4 root> --output <run>/04_audio/audio.npz
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.audio import features as audio


def main():
    parser = argparse.ArgumentParser(description="Cut an audio conditioning sidecar aligned with features.npz windows.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--video_metadata", required=True, help="Supplies each clip's relative_path and stream timing.")
    parser.add_argument("--input_dir", required=True, help="The mp4 root the pipeline was run on.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry_metadata", default=None,
                        help="Optional; supplies blendshape names for the jawOpen/audio sync diagnostic.")
    args = parser.parse_args()

    with np.load(args.features_npz) as data:
        clip_ids = data["clip_id"]
        window_start = data["window_start"]
        is_padding = data["is_padding"]
        hz = float(data["target_hz"])
        if "window_time_sec" not in data.files:
            print("features.npz has no window_time_sec; it predates the field. Re-run the features stage.")
            sys.exit(1)
        window_time = data["window_time_sec"]
    window_frames = is_padding.shape[1]
    window_seconds = window_frames / hz

    with open(args.video_metadata, encoding="utf-8") as handle:
        metadata = {m["file"]: m for m in json.load(handle)}
    jaw_column = None
    if args.geometry_metadata:
        from src.schema.feature_vector import channel
        with open(args.geometry_metadata, encoding="utf-8") as handle:
            names = json.load(handle)["tracks"]["A_landmarks_blendshapes"].get("blendshape_names") or []
        if "jawOpen" in names:
            jaw_column = channel("blendshape").start + names.index("jawOpen")
    with np.load(args.features_npz) as data:
        features = data["features"] if jaw_column is not None else None
        observed = data["observed"]

    n = clip_ids.shape[0]
    n_mel = int(round(window_seconds * audio.FRAMES_PER_SECOND))
    n_wave = int(round(window_seconds * audio.SAMPLE_RATE))
    mel = np.zeros((n, n_mel, audio.N_MELS), np.float32)
    wave = np.zeros((n, n_wave), np.int16)
    valid = np.zeros((n, n_mel), bool)
    per_clip = []

    for clip_id in dict.fromkeys(clip_ids.tolist()):
        rows = np.flatnonzero(clip_ids == clip_id)
        meta = metadata.get(clip_id, {})
        relative = meta.get("relative_path")
        record = {"clip_id": clip_id, "n_windows": int(rows.size), "streams": meta.get("streams")}
        decoded = {"has_audio": False}
        if not relative:
            record["status"] = "no_relative_path_in_video_metadata"
        else:
            try:
                decoded = audio.decode_audio(os.path.join(args.input_dir, relative))
                record["status"] = "ok" if decoded["has_audio"] else "no_audio_stream"
            except Exception as error:  # a broken audio stream must not take the clip's video with it
                record["status"] = "decode_failed"
                record["detail"] = str(error)[:300]
        if decoded.get("has_audio"):
            pts = meta.get("pts_sec") or []
            finite = [p for p in pts if p is not None]
            steps = np.diff(finite) if len(finite) > 2 else np.zeros(0)
            tail_gap = bool(steps.size and steps[-1] > 1.5 * np.median(steps))
            video_seconds = audio.video_duration(pts, exclude_tail_gap=tail_gap)
            audio_seconds = audio.covered_duration(decoded)
            mismatch = (None if video_seconds is None or audio_seconds is None
                        else round(video_seconds - audio_seconds, 4))
            record.update({
                "audio_start_sec": decoded["start_sec"],
                "decoded_seconds": round(decoded["wave"].size / audio.SAMPLE_RATE, 4),
                "video_seconds": None if video_seconds is None else round(video_seconds, 4),
                "video_minus_audio_sec": mismatch,
                "source_rate": decoded["source_rate"],
                "discontinuities": decoded["discontinuities"],
            })
            if mismatch is None or abs(mismatch) > audio.AV_DURATION_TOL:
                record["status"] = "av_duration_mismatch"
                decoded = {"has_audio": False}
        for row in rows:
            n_real = int((~is_padding[row]).sum())
            cut = audio.window_audio(decoded, float(window_time[row]), window_seconds,
                                     padding_from_sec=float(window_time[row]) + n_real / hz)
            mel[row], wave[row], valid[row] = cut["mel"], cut["wave"], cut["valid"]
        record["valid_fraction"] = round(float(valid[rows].mean()), 4)
        if features is not None and valid[rows].any():
            motion_t, motion, audio_t, energy = [], [], [], []
            for row in rows:
                t0 = float(window_time[row])
                ok = observed[row] & ~is_padding[row]
                motion_t += list(t0 + np.flatnonzero(ok) / hz)
                motion += list(features[row, ok, jaw_column])
                v = valid[row]
                audio_t += list(t0 + (np.flatnonzero(v) + 0.5) / audio.FRAMES_PER_SECOND)
                energy += list(np.log(np.exp(mel[row, v].astype(np.float64)).sum(axis=1)))
            mt, i = np.unique(motion_t, return_index=True)
            at, j = np.unique(audio_t, return_index=True)
            lags, corr = audio.sync_curve(mt, np.asarray(motion)[i], at, np.asarray(energy)[j])
            record["sync_curve"] = [None if not np.isfinite(c) else round(float(c), 4) for c in corr]
            if np.isfinite(corr).any():
                record["sync_peak_lag_ms"] = int(round(1000 * lags[int(np.nanargmax(corr))]))
                record["sync_peak_r"] = round(float(np.nanmax(corr)), 4)
        per_clip.append(record)
        print(f"  {clip_id}: {record['status']}, valid {record['valid_fraction']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output, mel=mel, wave=wave, audio_valid=valid,
        clip_id=clip_ids, window_start=window_start, window_time_sec=window_time,
        sample_rate=np.array(audio.SAMPLE_RATE), frames_per_second=np.array(audio.FRAMES_PER_SECOND),
        audio_features_version=np.array(audio.SPEC["audio_features_version"]),
    )
    manifest = {
        "audio_features_version": audio.SPEC["audio_features_version"],
        "spec_file": os.path.relpath(audio.SPEC_PATH, REPO_ROOT),
        "features_npz": os.path.abspath(args.features_npz),
        "alignment": "row i of audio.npz is the audio for row i of features.npz; both carry clip_id, window_start, window_time_sec",
        "shapes": {"mel": list(mel.shape), "wave": list(wave.shape), "audio_valid": list(valid.shape)},
        "window_seconds": window_seconds,
        "n_clips": len(per_clip),
        "n_clips_with_audio": sum(1 for r in per_clip if r["status"] == "ok"),
        "valid_fraction": round(float(valid.mean()), 4) if valid.size else None,
        "per_clip": per_clip,
    }
    curves = [r["sync_curve"] for r in per_clip if r.get("sync_curve") and None not in r["sync_curve"]]
    if curves:
        pooled = np.mean(np.asarray(curves, dtype=np.float64), axis=0)
        lags = np.arange(pooled.size) * 0.01 - audio.SYNC_MAX_LAG
        manifest["sync_check"] = {
            "what": "jawOpen vs audio log energy cross-correlation; positive lag = audio later than mouth. "
                    "Diagnostic only.",
            "n_clips": len(curves),
            "pooled_peak_lag_ms": int(round(1000 * lags[int(pooled.argmax())])),
            "pooled_peak_r": round(float(pooled.max()), 4),
            "pooled_r_at_zero": round(float(pooled[pooled.size // 2]), 4),
            "median_clip_peak_lag_ms": int(np.median([r["sync_peak_lag_ms"] for r in per_clip if "sync_peak_lag_ms" in r])),
        }
    for record in per_clip:
        record.pop("sync_curve", None)
    manifest["n_clips_av_duration_mismatch"] = sum(1 for r in per_clip if r["status"] == "av_duration_mismatch")
    manifest_path = os.path.splitext(args.output)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"\n{manifest['n_clips_with_audio']}/{manifest['n_clips']} clips with usable audio "
          f"({manifest['n_clips_av_duration_mismatch']} av_duration_mismatch); valid fraction {manifest['valid_fraction']}")
    if "sync_check" in manifest:
        sc = manifest["sync_check"]
        print(f"sync check: pooled peak lag {sc['pooled_peak_lag_ms']:+d} ms (r={sc['pooled_peak_r']}), "
              f"median clip lag {sc['median_clip_peak_lag_ms']:+d} ms over {sc['n_clips']} clips")
    print(f"Written: {args.output}")
    print(f"Written: {manifest_path}")


if __name__ == "__main__":
    main()
