"""
Track A: joint landmark + blendshape extraction.

MediaPipe Face Landmarker emits 478 3D landmarks and 52 ARKit-compatible
blendshape coefficients from a single inference -- the two are computed jointly
inside the model, so there is deliberately no in-house "fit landmarks -> solve
for blendshapes" two-step here.

Track B (src/geometry/headpose.py) consumes the facial transformation matrix
produced by that same inference call, so tracks A and B share one forward pass
per frame while remaining separate modules. Track C (src/geometry/gaze.py) runs
its own model. The shared decode loop lives in src/geometry/pipeline.py.

Confidence (see src/geometry/confidence.py for how the four channels combine):
MediaPipe Face Landmarker leaves NormalizedLandmark.visibility / .presence at
zero, so there is no native per-landmark score to report. Both signals below are
therefore derived proxies, and are labelled as such in geometry_metadata.json:

  in_frame_fraction -- MediaPipe extrapolates landmarks beyond the image border
    when the face is partially out of frame; those coordinates are inferred, not
    observed, so the fraction of landmarks inside [0, 1] is a genuine measure of
    how much of the output is supported by pixels.
  size_factor -- a face occupying very few pixels yields landmarks quantised to
    a coarse grid. Saturates at MIN_FACE_SPAN so normally framed faces are not
    penalised.

Blendshape confidence is the same proxy. An earlier version additionally zeroed
it when the 52 coefficients summed below a threshold, on the theory that an
all-near-zero vector meant the model could not read the expression. That test was
removed: a genuinely neutral face IS an all-near-zero ARKit vector, so the rule
marked correct output on exactly the expression the model reproduces best. There
is no way to tell a neutral face from a failed read by the coefficient sum alone,
and a false "unusable" is worse than no signal -- it silently deletes neutral
frames from supervision and biases the data towards expressive ones.

None of these numbers are probabilities. They are quality proxies with no
calibration, and geometry_metadata.json labels them as such.
"""
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

N_LANDMARKS = 478
N_BLENDSHAPES = 52

# A face whose landmark bounding box spans at least this fraction of the shorter
# image side is considered fully resolved; below it, confidence falls off linearly.
MIN_FACE_SPAN = 0.15

CONFIDENCE_METHOD = {
    "landmarks": "quality proxy, not a probability: in_frame_fraction * size_factor (MediaPipe Face Landmarker exposes no native per-landmark score)",
    "blendshapes": "quality proxy, not a probability: same as the landmark proxy; a near-zero coefficient vector is NOT penalised, because that is what a neutral face looks like",
}


def build_options(model_path: str) -> FaceLandmarkerOptions:
    """Options for the single shared inference that feeds tracks A and B."""
    return FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )


def empty_frame() -> dict:
    """Undetected frame: NaN payload, zero confidence, frame index preserved.

    Nothing is dropped or interpolated -- a tracking dropout stays a dropout all
    the way into geometry.parquet so the Dynamics layer can propagate it.
    """
    return {
        "detected": False,
        "landmarks": np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32),
        "blendshapes": np.full((N_BLENDSHAPES,), np.nan, dtype=np.float32),
        "conf_landmarks": 0.0,
        "conf_blendshapes": 0.0,
    }


def parse_frame(landmarks_proto, blendshapes_proto) -> dict:
    """Convert one MediaPipe detection into track-A arrays plus confidences."""
    landmarks = np.array([[p.x, p.y, p.z] for p in landmarks_proto], dtype=np.float32)
    blendshapes = np.array([c.score for c in blendshapes_proto], dtype=np.float32)

    xy = landmarks[:, :2]
    in_frame_fraction = float(np.all((xy >= 0.0) & (xy <= 1.0), axis=1).mean())
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    size_factor = float(min(1.0, span / MIN_FACE_SPAN))
    conf_landmarks = in_frame_fraction * size_factor

    # Deliberately the same proxy as the landmarks: the two come from one
    # inference, and no coefficient-magnitude test can separate a neutral face
    # from a failed read. See the module docstring.
    conf_blendshapes = conf_landmarks

    return {
        "detected": True,
        "landmarks": landmarks,
        "blendshapes": blendshapes,
        "conf_landmarks": conf_landmarks,
        "conf_blendshapes": conf_blendshapes,
    }


def blendshape_names(blendshapes_proto) -> list:
    return [c.category_name for c in blendshapes_proto]
