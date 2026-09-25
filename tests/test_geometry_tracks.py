"""
Unit tests for the three-track Geometry layer.

Self-contained and dependency-light on purpose: every case is synthetic, so this
needs no downloaded clips, no face_landmarker.task and no MediaPipe install. It
therefore covers exactly the modules that are pure numpy -- headpose.py, gaze.py
and confidence.py -- which is also where the parameterisation and normalisation
decisions that would be expensive to get wrong actually live. Plain asserts
rather than pytest, matching tests/test_dynamics_derive.py, because env-v0.1 is
frozen and ships no test runner.

The case that matters most is test_6d_rotation_is_continuous_at_180_degrees():
it is the concrete evidence for choosing the 6D representation over Euler angles
for a signal the Dynamics layer differentiates.

Usage:
    python tests/test_geometry_tracks.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.geometry.confidence import LOW_CONFIDENCE_THRESHOLD, signal_statistics
from src.geometry.gaze import (
    LEFT_EYE, LEFT_IRIS_CENTER, RIGHT_EYE, RIGHT_IRIS_CENTER,
    PITCH_LIMIT_RAD, UNIT_IRIS_PROXY, UNIT_RADIANS, YAW_LIMIT_RAD,
    IrisGeometryBackend, normalise_gaze,
)
from src.geometry.headpose import (
    decompose_similarity, normalise_translation, orthonormality_residual,
    parse_frame, rotation_6d_to_matrix, rotation_matrix_to_6d,
)


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_6d_rotation_roundtrips():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rotation = _random_rotation(rng)
        recovered = rotation_6d_to_matrix(rotation_matrix_to_6d(rotation))
        assert np.allclose(rotation, recovered, atol=1e-6), "6D -> matrix round-trip lost the rotation"


def test_6d_rotation_is_continuous_at_180_degrees():
    """Euler yaw jumps by ~2*pi across the wrap point; the 6D form does not.

    This is the property that matters downstream: the Dynamics layer
    differentiates head pose, so a representation that jumps at +/-180 degrees
    would manufacture an enormous velocity spike out of a continuous head turn.
    """
    epsilon = 1e-3
    before = _rotation_z(np.pi - epsilon)
    after = _rotation_z(-np.pi + epsilon)  # the same physical pose, other side of the wrap

    euler_before = np.arctan2(before[1, 0], before[0, 0])
    euler_after = np.arctan2(after[1, 0], after[0, 0])
    euler_jump = abs(euler_after - euler_before)

    sixd_jump = float(np.linalg.norm(rotation_matrix_to_6d(after) - rotation_matrix_to_6d(before)))

    assert euler_jump > 6.0, "expected the Euler parameterisation to wrap here"
    assert sixd_jump < 1e-2, f"6D representation should stay continuous, jumped {sixd_jump}"


def test_decompose_similarity_splits_scale_out():
    rng = np.random.default_rng(1)
    rotation = _random_rotation(rng)
    scale = 2.75
    translation = np.array([1.0, -3.0, 12.0])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * scale
    matrix[:3, 3] = translation

    recovered_r, recovered_t, recovered_s = decompose_similarity(matrix)
    assert abs(recovered_s - scale) < 1e-6, "similarity scale not recovered"
    assert np.allclose(recovered_r, rotation, atol=1e-6), "rotation not recovered after scale removal"
    assert np.allclose(recovered_t, translation, atol=1e-6), "translation not recovered"
    assert orthonormality_residual(recovered_r) < 1e-6


def test_headpose_confidence_drops_on_degenerate_matrix():
    rng = np.random.default_rng(2)
    clean = np.eye(4)
    clean[:3, :3] = _random_rotation(rng) * 1.5
    assert parse_frame(clean)["conf_headpose"] > 0.99, "a clean similarity transform should score high"

    skewed = np.eye(4)
    skewed[:3, :3] = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.3], [0.2, 0.0, 1.0]])
    assert parse_frame(skewed)["conf_headpose"] < 0.5, "a non-orthonormal block should be penalised"

    collapsed = np.zeros((4, 4))
    assert parse_frame(collapsed)["conf_headpose"] == 0.0, "a degenerate matrix must score zero"


def test_translation_normalisation_ignores_dropouts():
    translations = np.array([
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
        [np.nan, np.nan, np.nan],   # tracking dropout
        [5.0, 6.0, 7.0],
    ])
    detected = np.array([True, True, False, True])

    normalised, reference = normalise_translation(translations, detected)
    assert np.allclose(reference, [3.0, 4.0, 5.0]), f"dropout leaked into the reference: {reference}"
    assert np.allclose(normalised[0], [-2.0, -2.0, -2.0])
    assert np.isnan(normalised[2]).all(), "a dropout must stay a dropout"

    none_tracked = normalise_translation(translations, np.zeros(4, dtype=bool))
    assert np.isnan(none_tracked[1]).all(), "with nothing tracked the reference must be NaN, not 0"


def test_gaze_zero_mean_and_tanh_bounds():
    pitch = np.array([0.1, 0.2, 0.3, 5.0])
    yaw = np.array([-0.2, 0.0, 0.2, -5.0])
    confidence = np.array([1.0, 1.0, 1.0, 0.0])  # last frame is a blink, excluded

    centred_pitch, centred_yaw, reference = normalise_gaze(pitch, yaw, confidence)
    assert abs(reference[0] - 0.2) < 1e-9, f"blink frame contaminated the pitch baseline: {reference}"
    assert abs(reference[1] - 0.0) < 1e-9, f"blink frame contaminated the yaw baseline: {reference}"
    assert abs(float(np.mean(centred_pitch[:3]))) < 1e-6, "tracked frames should be zero-mean"

    compressed_pitch, compressed_yaw, _ = normalise_gaze(pitch, yaw, confidence, apply_tanh=True)
    assert abs(compressed_pitch[3]) < PITCH_LIMIT_RAD, "tanh must bound the outlier pitch"
    assert abs(compressed_yaw[3]) < YAW_LIMIT_RAD, "tanh must bound the outlier yaw"
    assert abs(compressed_pitch[0] - centred_pitch[0]) < 5e-3, "tanh should be near-identity for small angles"


def _synthetic_face(iris_offset_x: float = 0.0, eye_aperture: float = 0.35) -> np.ndarray:
    """478 landmarks with only the eye/iris points set to meaningful values."""
    landmarks = np.zeros((478, 3), dtype=np.float64)
    for eye, iris_index, centre_x in ((LEFT_EYE, LEFT_IRIS_CENTER, 0.35), (RIGHT_EYE, RIGHT_IRIS_CENTER, 0.65)):
        half_width = 0.05
        landmarks[eye["outer"]] = [centre_x - half_width, 0.5, 0.0]
        landmarks[eye["inner"]] = [centre_x + half_width, 0.5, 0.0]
        landmarks[eye["upper"]] = [centre_x, 0.5 - eye_aperture * half_width, 0.0]
        landmarks[eye["lower"]] = [centre_x, 0.5 + eye_aperture * half_width, 0.0]
        landmarks[iris_index] = [centre_x + iris_offset_x * half_width, 0.5, 0.0]
    return landmarks


def test_iris_backend_tracks_offset_and_rejects_blinks():
    backend = IrisGeometryBackend()
    shape = (720, 720, 3)  # square, so the normalised space is isotropic

    centred = backend.estimate(None, _synthetic_face(iris_offset_x=0.0), shape)
    assert abs(centred[1]) < 1e-9, "a centred iris should read as zero yaw"
    assert centred[2] > 0.0, "an open eye should have non-zero confidence"

    looking_aside = backend.estimate(None, _synthetic_face(iris_offset_x=0.6), shape)
    assert looking_aside[1] > centred[1], "an offset iris should produce a larger yaw"

    blinking = backend.estimate(None, _synthetic_face(eye_aperture=0.02), shape)
    assert blinking[2] == 0.0, "a closed eye must report zero gaze confidence"
    assert np.isnan(blinking[0]) and np.isnan(blinking[1]), "a blink must not report an angle"

    undetected = backend.estimate(None, None, shape)
    assert undetected[2] == 0.0 and np.isnan(undetected[0])


def test_signal_statistics_surface_tracked_but_low_confidence():
    """The silent-contamination case: frames that tracked fine but scored badly."""
    scores = np.array([0.95, 0.92, 0.30, 0.25, 0.0])
    detected = np.array([True, True, True, True, False])

    stats = signal_statistics(scores, detected)
    assert stats["n_low_but_tracked"] == 2, "the two degraded-but-tracked frames must be counted"
    assert stats["n_zero"] == 1, "the dropout should be counted separately from degraded frames"
    assert stats["low_but_tracked_rate"] == round(2 / 5, 4)
    assert stats["std"] > 0.0, "variance must be reported, not just the mean"
    assert stats["min"] == 0.0 and stats["max"] == 0.95

    flat = signal_statistics(np.full(5, 0.9), np.ones(5, dtype=bool))
    assert flat["n_low_but_tracked"] == 0
    assert flat["std"] == 0.0, "a uniformly confident channel should show zero spread"
    assert flat["mean"] > LOW_CONFIDENCE_THRESHOLD


def test_gram_schmidt_survives_degenerate_input():
    """A network early in training, or a neutral fill, emits exactly these."""
    for label, bad in (("zero vector", np.zeros(6)),
                       ("parallel columns", np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])),
                       ("nan", np.full(6, np.nan)),
                       ("inf", np.array([np.inf, 0, 0, 0, 1.0, 0]))):
        recovered = rotation_6d_to_matrix(bad)
        assert np.isfinite(recovered).all(), f"{label} produced non-finite output"
        assert np.allclose(recovered.T @ recovered, np.eye(3), atol=1e-8), \
            f"{label} produced a non-rotation"
        assert abs(np.linalg.det(recovered) - 1.0) < 1e-8, f"{label} produced a reflection"


def test_geodesic_angle_error_is_available_for_evaluation():
    """Raw 6D distance has no physical unit; angular error does."""
    rng = np.random.default_rng(11)
    reference = _random_rotation(rng)
    for angle in (0.0, 0.01, 0.5, np.pi / 2):
        rotated = reference @ _rotation_z(angle)
        recovered_a = rotation_6d_to_matrix(rotation_matrix_to_6d(reference))
        recovered_b = rotation_6d_to_matrix(rotation_matrix_to_6d(rotated))
        relative = recovered_a.T @ recovered_b
        # Clipped before arccos: float error pushes the trace fractionally outside
        # [-1, 1] for near-identity rotations, and unclipped arccos returns NaN
        # exactly where the error is smallest.
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        measured = float(np.arccos(cosine))
        assert np.isfinite(measured), f"angle {angle} gave a non-finite error"
        assert abs(measured - angle) < 1e-6, f"expected {angle}, measured {measured}"


def test_gaze_units_and_axis_order_are_explicit():
    backend = IrisGeometryBackend()
    assert backend.unit == UNIT_IRIS_PROXY, "the iris backend must not claim radians"
    assert backend.is_calibrated is False

    shape = (720, 720, 3)
    # Axis order is (pitch, yaw). Moving the iris up must raise pitch.
    centred = backend.estimate(None, _synthetic_face(), shape)
    face_up = _synthetic_face()
    for iris, eye in ((LEFT_IRIS_CENTER, LEFT_EYE), (RIGHT_IRIS_CENTER, RIGHT_EYE)):
        face_up[iris][1] -= 0.01          # image y decreases upward
    looking_up = backend.estimate(None, face_up, shape)
    assert looking_up[0] > centred[0], "positive pitch must mean looking up"

    looking_right = backend.estimate(None, _synthetic_face(iris_offset_x=0.6), shape)
    assert looking_right[1] > centred[1], "positive yaw must mean looking image-right"

    # The proxy is dimensionless, so radian-scaled compression must be refused.
    values = np.array([0.1, 0.2, 0.3])
    try:
        normalise_gaze(values, values, np.ones(3), apply_tanh=True, unit=UNIT_IRIS_PROXY)
        raise AssertionError("tanh at radian limits must be refused for a dimensionless proxy")
    except ValueError:
        pass
    pitch, _, _ = normalise_gaze(values, values, np.ones(3), apply_tanh=True, unit=UNIT_RADIANS)
    assert np.isfinite(pitch).all(), "a radian backend may use the compression"


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} geometry-track tests.")


if __name__ == "__main__":
    main()
