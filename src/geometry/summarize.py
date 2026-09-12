"""
Aggregate canonicalized results into geometry.parquet + geometry_metadata.json
+ geometry_quality.json, matching design doc section 2.4.
"""
import numpy as np
from datetime import datetime

MEDIAPIPE_VERSION = "0.10.21"


def load_and_flatten(canonical_data: dict, clip_id: str) -> list:
    canonical_landmarks = canonical_data["canonical_landmarks"]
    blendshapes = canonical_data["blendshapes"]
    headpose = canonical_data["headpose"]
    detected = canonical_data["detected"]
    eye_distance = canonical_data["eye_distance"]

    n_frames = canonical_landmarks.shape[0]
    rows = []
    for i in range(n_frames):
        rows.append({
            "clip_id": clip_id, "frame_idx": i, "detected": bool(detected[i]),
            "eye_distance": float(eye_distance[i]) if not np.isnan(eye_distance[i]) else None,
            "landmarks": canonical_landmarks[i].flatten().tolist(),
            "blendshapes": blendshapes[i].tolist(),
            "headpose": headpose[i].flatten().tolist(),
        })
    return rows


def build_geometry_metadata() -> dict:
    return {
        "canonicalization_method": {
            "description": "Scale by the Euclidean distance between the two outer eye corners; translate to their midpoint as origin",
            "reference_landmarks": {"point_a": 33, "point_b": 263},
        },
        "tracker": {
            "name": "MediaPipe Face Landmarker", "version": MEDIAPIPE_VERSION,
            "note": "Pinned to 0.10.x to avoid a known Metal-related crash in 1.0.x on macOS",
            "delegate": "CPU", "model_file": "face_landmarker.task (float16, v1)",
        },
        "output_fields": {
            "landmarks": "478 x 3 (canonical coordinates)", "blendshapes": "52-dim, ARKit-compatible naming",
            "headpose": "4x4 transformation matrix, flattened to 16-dim",
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_quality_report(quality_per_clip: list) -> dict:
    return {
        "per_clip": quality_per_clip,
        "note": "Small-sample validation stage; per-subgroup statistics deferred to the large-scale processing stage",
        "filtered_clips": [q["clip_id"] for q in quality_per_clip if q["clip_quality_label"] == "reject"],
    }
