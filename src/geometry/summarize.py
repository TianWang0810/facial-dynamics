"""
Aggregate the three tracks into geometry.parquet + geometry_metadata.json
+ geometry_quality.json.

One parquet row per frame, carrying all three tracks and the shared quality mask
side by side, so a consumer never has to join across files to know how much to
trust a frame:

    track A   landmarks (478*3 flattened), blendshapes (52), eye_distance
    track B   headpose_r6 (6), headpose_t (3), headpose_t_raw (3)
    track C   gaze_pitch, gaze_yaw
    mask      conf_landmarks, conf_blendshapes, conf_headpose, conf_gaze

headpose_t_raw is kept next to the normalised headpose_t so the clip-level
reference subtraction stays invertible and auditable after the fact.
"""
import numpy as np
from datetime import datetime

from src.geometry import confidence as confidence_module
from src.geometry import gaze as gaze_module
from src.geometry import headpose as headpose_module
from src.geometry import landmarks as landmarks_module

MEDIAPIPE_VERSION = "0.10.21"


def _optional_float(value):
    return float(value) if np.isfinite(value) else None


def load_and_flatten(clip_result: dict, clip_id: str, identity: dict = None) -> list:
    """Turn one clip's track arrays into frame-aligned parquet rows.

    Every row carries source_frame_idx (position in decode order) and pts_sec
    (the timestamp the container reported for that frame, or None when it
    reported none), so a row can be traced to a specific frame of a specific file
    without relying on array position matching across files.
    """
    canonical_landmarks = clip_result["canonical_landmarks"]
    blendshapes = clip_result["blendshapes"]
    headpose_r6 = clip_result["headpose_r6"]
    headpose_t = clip_result["headpose_t"]
    headpose_t_raw = clip_result["headpose_t_raw"]
    gaze_pitch = clip_result["gaze_pitch"]
    gaze_yaw = clip_result["gaze_yaw"]
    detected = clip_result["detected"]
    eye_distance = clip_result["eye_distance"]
    confidences = clip_result["confidences"]

    pts_sec = clip_result.get("pts_sec")
    tracker_ms = clip_result.get("tracker_timestamp_ms")
    identity = identity or {}

    rows = []
    for i in range(canonical_landmarks.shape[0]):
        rows.append({
            "clip_id": clip_id,
            "source_video_id": identity.get("source_video_id"),
            "speaker_id": identity.get("speaker_id"),
            "relative_path": identity.get("relative_path"),
            "frame_idx": i,
            "source_frame_idx": i,
            "pts_sec": _optional_float(pts_sec[i]) if pts_sec is not None else None,
            "tracker_timestamp_ms": int(tracker_ms[i]) if tracker_ms is not None else None,
            "detected": bool(detected[i]),
            "eye_distance": _optional_float(eye_distance[i]),
            "landmarks": canonical_landmarks[i].flatten().tolist(),
            "blendshapes": blendshapes[i].tolist(),
            "headpose_r6": headpose_r6[i].tolist(),
            "headpose_t": headpose_t[i].tolist(),
            "headpose_t_raw": headpose_t_raw[i].tolist(),
            "gaze_pitch": _optional_float(gaze_pitch[i]),
            "gaze_yaw": _optional_float(gaze_yaw[i]),
            "conf_landmarks": float(confidences["landmarks"][i]),
            "conf_blendshapes": float(confidences["blendshapes"][i]),
            "conf_headpose": float(confidences["headpose"][i]),
            "conf_gaze": float(confidences["gaze"][i]),
        })
    return rows


def build_geometry_metadata(gaze_backend_name: str, gaze_tanh_applied: bool,
                            gaze_unit: str, gaze_is_calibrated: bool,
                            blendshape_names: list) -> dict:
    """Provenance for every choice a consumer would otherwise have to guess.

    The gaze unit is a REQUIRED argument rather than a constant, because it is a
    property of the active backend and not of this layer: the external model
    emits radians, the iris fallback a dimensionless offset proxy. This block
    previously hard-coded "radians" and had the real unit patched in afterwards
    under a second key, so the file declared both at once -- exactly the
    ambiguity docs/decisions/v0_training_data_contract.md ("Honest units and
    honest scores") exists to prevent. There is one key now, and it is filled by
    the caller from the backend.

    blendshape_names is likewise required: the 52 blendshape columns of
    geometry.parquet are positional, and their meaning is whatever order this
    MediaPipe build's category_name table happens to have. Persisting the names
    read off the model output is what lets a consumer (the L1 text layer, which
    names channels) resolve a column without assuming that order.
    """
    # Empty when no clip produced a single detection: the names come off the
    # tracker output, so there is nothing to read. Recorded as null rather than
    # failing the stage -- Features rejects such clips by detection rate anyway,
    # and a consumer that needs the names (L1) refuses a null explicitly.
    if blendshape_names and len(blendshape_names) != landmarks_module.N_BLENDSHAPES:
        raise ValueError("expected %d blendshape names from the tracker, got %d"
                         % (landmarks_module.N_BLENDSHAPES, len(blendshape_names)))
    return {
        "tracks": {
            "A_landmarks_blendshapes": {
                "tool": "MediaPipe Face Landmarker",
                "joint_extraction": True,
                "note": "478 3D landmarks and 52 ARKit-named blendshapes come from one joint inference; no separate landmark-fitting then blendshape-solving step",
                "output_fields": {"landmarks": "478 x 3 (canonical coordinates)", "blendshapes": "52-dim, ARKit-compatible naming"},
                "blendshape_names": list(blendshape_names) or None,
                "blendshape_names_note": "category_name of each blendshape column, in column order, as read off the tracker output -- not a restated constant",
            },
            "B_headpose": {
                "dimensions": 9,
                "rotation_parameterisation": "6D continuous representation (first two columns of R; Zhou et al. 2019)",
                "rotation_rationale": "Euler angles suffer gimbal lock and a wrap-around discontinuity; unit quaternions double-cover the rotation group (q == -q). Both inject artificial jumps that the Dynamics layer would differentiate into spurious velocity spikes.",
                "rotation_recovery": "Gram-Schmidt, see src/geometry/headpose.py:rotation_6d_to_matrix",
                "translation_normalisation": "per-clip mean head position over tracked frames subtracted, removing camera-distance offset; raw translation retained as headpose_t_raw",
            },
            "C_gaze": {
                "dimensions": 2,
                "backend": gaze_backend_name,
                "unit": gaze_unit,
                "is_calibrated": bool(gaze_is_calibrated),
                "note": "estimated by track C's own estimator, deliberately NOT derived from the eyeLookUp/Down/In/Out blendshape coefficients (insufficient accuracy, heavily discretised)",
                "normalisation": "zero-mean per clip over frames with non-zero confidence",
                "tanh_compression": {
                    "applied": gaze_tanh_applied,
                    "pitch_limit_rad": gaze_module.PITCH_LIMIT_RAD,
                    "yaw_limit_rad": gaze_module.YAW_LIMIT_RAD,
                },
            },
        },
        "canonicalization_method": {
            "description": "Scale by the Euclidean distance between the two outer eye corners; translate to their midpoint as origin",
            "reference_landmarks": {"point_a": 33, "point_b": 263},
        },
        "quality_mask": {
            "granularity": "per frame, per channel",
            "track_signals": list(confidence_module.TRACK_SIGNALS),
            "columns": list(confidence_module.CONFIDENCE_COLUMNS),
            "low_confidence_threshold": confidence_module.LOW_CONFIDENCE_THRESHOLD,
            "reported_statistics": "mean, std, var, min, max, p05, p50 plus explicit tracked-but-low-confidence counts, so degraded frames that never failed outright stay visible",
            "methods": dict(
                list(landmarks_module.CONFIDENCE_METHOD.items())
                + list(headpose_module.CONFIDENCE_METHOD.items())
                + list(gaze_module.CONFIDENCE_METHOD.items())
            ),
        },
        "tracker": {
            "name": "MediaPipe Face Landmarker", "version": MEDIAPIPE_VERSION,
            "note": "Pinned to 0.10.x to avoid a known Metal-related crash in 1.0.x on macOS",
            "delegate": "CPU", "model_file": "face_landmarker.task (float16, v1)",
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_quality_report(clip_reports: list) -> dict:
    """Roll the per-clip mask reports up, keeping the per-channel view intact."""
    signal_means = {}
    for signal in confidence_module.TRACK_SIGNALS:
        means = [r["per_signal_confidence"][signal]["mean"] for r in clip_reports
                 if r["per_signal_confidence"][signal]["mean"] is not None]
        low_rates = [r["per_signal_confidence"][signal]["low_but_tracked_rate"] for r in clip_reports
                     if r["per_signal_confidence"][signal]["low_but_tracked_rate"] is not None]
        signal_means[signal] = {
            "mean_of_clip_means": round(float(np.mean(means)), 4) if means else None,
            "std_of_clip_means": round(float(np.std(means)), 4) if means else None,
            "worst_clip_mean": round(float(np.min(means)), 4) if means else None,
            "mean_low_but_tracked_rate": round(float(np.mean(low_rates)), 4) if low_rates else None,
        }

    return {
        "per_clip": clip_reports,
        "per_signal_across_clips": signal_means,
        "note": "Small-sample validation stage; per-subgroup statistics deferred to the large-scale processing stage",
        "filtered_clips": [r["clip_id"] for r in clip_reports if r["clip_quality_label"] == "reject"],
    }
