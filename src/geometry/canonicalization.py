"""
Identity canonicalization: use the Euclidean distance between the two outer eye
corners (landmarks 33 and 263) as the scale reference, and their midpoint as the
translation origin. Method recorded in geometry_metadata.json (see summarize.py).
"""
import numpy as np

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def canonicalize(landmarks: np.ndarray) -> dict:
    n_frames = landmarks.shape[0]
    canonical = np.full_like(landmarks, np.nan)
    eye_distances = np.full(n_frames, np.nan, dtype=np.float32)
    origins = np.full((n_frames, 3), np.nan, dtype=np.float32)

    for i in range(n_frames):
        frame_lm = landmarks[i]
        if np.isnan(frame_lm).any():
            continue
        p_left = frame_lm[LEFT_EYE_OUTER]
        p_right = frame_lm[RIGHT_EYE_OUTER]
        eye_dist = np.linalg.norm(p_left - p_right)
        origin = (p_left + p_right) / 2.0
        if eye_dist < 1e-6:
            continue
        canonical[i] = (frame_lm - origin) / eye_dist
        eye_distances[i] = eye_dist
        origins[i] = origin

    return {
        "canonical_landmarks": canonical,
        "eye_distance": eye_distances,
        "origin": origins,
        "reference_points": {"left": LEFT_EYE_OUTER, "right": RIGHT_EYE_OUTER},
    }
