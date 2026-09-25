"""
Track B: head pose as 9 dimensions -- 6D rotation + 3D normalised translation.

Rotation parameterisation: the 6D continuous representation of Zhou et al. 2019
("On the Continuity of Rotation Representations in Neural Networks"), i.e. the
first two columns of the rotation matrix, with the third recovered by
Gram-Schmidt. Euler angles are rejected because of gimbal lock and because the
wrap-around at +/-180 degrees creates artificial discontinuities that the
Dynamics layer would differentiate into enormous spurious velocities; unit
quaternions are rejected because of double cover (q and -q are the same
rotation, so a sign flip between adjacent frames is likewise a fake jump). The
6D form is continuous everywhere, which is exactly the property a layer that
differentiates the signal needs. This choice is recorded in
geometry_metadata.json via build_geometry_metadata() in summarize.py.

Translation normalisation: MediaPipe's facial transformation matrix places the
head in camera space, so its translation mixes head motion with how far the
subject happened to sit from the lens. Subtracting a per-clip reference (the
mean head position over the frames where tracking succeeded) removes that camera
distance offset and leaves motion relative to the clip's own baseline. The raw
translation is kept alongside it so the normalisation stays invertible and
auditable.

Confidence: MediaPipe returns a similarity transform, so the upper-left 3x3
block carries a uniform scale that must be divided out before it can be read as a
rotation. The residual is measured AFTER that rescaling and BEFORE any
orthonormalization -- nothing in this module orthogonalizes the block first, so
the number does reflect the matrix as delivered.

What it does and does not mean: a near-zero residual says the delivered block was
a clean scaled rotation, which catches a corrupt or degenerate transform. It does
NOT say the pose is accurate. MediaPipe constructs this matrix to be rigid, so
the residual is expected to sit near zero on most frames and carries little
information when it does; it is a validity check, not an accuracy estimate, and
like every other score here it is an uncalibrated proxy rather than a probability.
"""
import numpy as np

ORTHONORMALITY_TOL = 0.05  # residual at which head-pose confidence reaches zero
# Vector norm below which Gram-Schmidt is undefined and the identity is returned.
DEGENERATE_NORM_TOL = 1e-8

CONFIDENCE_METHOD = {
    "headpose": "measured: 1 - min(1, ||R^T R - I||_F / %.2f) after dividing out the similarity scale; zero if det(R) <= 0" % ORTHONORMALITY_TOL,
}


def decompose_similarity(matrix_4x4: np.ndarray) -> tuple:
    """Split MediaPipe's 4x4 facial transformation into (rotation, translation, scale).

    The upper-left block is a scaled rotation; the scale is the mean column norm.
    Returns (R, t, scale) with R rescaled to unit columns.
    """
    matrix = np.asarray(matrix_4x4, dtype=np.float64)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    column_norms = np.linalg.norm(linear, axis=0)
    scale = float(column_norms.mean())
    if not np.isfinite(scale) or scale < 1e-9:
        return np.full((3, 3), np.nan), np.full(3, np.nan), float("nan")
    return linear / scale, translation, scale


def rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened -- the continuous 6D representation."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return rotation[:, :2].T.reshape(6).astype(np.float32)


def rotation_6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Inverse of rotation_matrix_to_6d via Gram-Schmidt.

    Column order matches rotation_matrix_to_6d exactly: that function emits
    R[:, :2].T.reshape(6), i.e. the first column's three components followed by
    the second's, and this reads them back in the same order. The pair is
    round-tripped in tests, because a silent transposition here would corrupt
    every rotation while still producing a valid-looking matrix.

    Degenerate input is handled rather than allowed to divide by zero. A network
    early in training, or an all-zero fill, can easily produce a first vector of
    zero length or two parallel vectors; Gram-Schmidt is undefined for both. Such
    input falls back to the identity rotation, which is a valid rotation and an
    honest "no information", instead of NaNs that propagate into the loss.
    """
    r6 = np.asarray(r6, dtype=np.float64).reshape(2, 3)
    a1, a2 = r6[0], r6[1]
    if not np.isfinite(r6).all():
        return np.eye(3)

    norm1 = np.linalg.norm(a1)
    if norm1 < DEGENERATE_NORM_TOL:
        return np.eye(3)
    b1 = a1 / norm1

    b2 = a2 - np.dot(b1, a2) * b1
    norm2 = np.linalg.norm(b2)
    if norm2 < DEGENERATE_NORM_TOL:
        # a2 is parallel to a1, so it carries no second axis. Pick any direction
        # orthogonal to b1 -- the rotation is underdetermined and this at least
        # stays on the manifold.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(b1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        b2 = fallback - np.dot(b1, fallback) * b1
        norm2 = np.linalg.norm(b2)
    b2 = b2 / norm2

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def orthonormality_residual(rotation: np.ndarray) -> float:
    """Frobenius norm of (R^T R - I): 0 for a perfect rotation."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if not np.isfinite(rotation).all():
        return float("nan")
    return float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))


def empty_frame() -> dict:
    return {
        "headpose_r6": np.full((6,), np.nan, dtype=np.float32),
        "headpose_t_raw": np.full((3,), np.nan, dtype=np.float32),
        "headpose_scale": float("nan"),
        "conf_headpose": 0.0,
    }


def parse_frame(matrix_4x4: np.ndarray) -> dict:
    """Convert one facial transformation matrix into the 6D + translation form."""
    rotation, translation, scale = decompose_similarity(matrix_4x4)
    if not np.isfinite(rotation).all():
        return empty_frame()

    residual = orthonormality_residual(rotation)
    determinant = float(np.linalg.det(rotation))
    if not np.isfinite(residual) or determinant <= 0.0:
        confidence = 0.0
    else:
        confidence = 1.0 - min(1.0, residual / ORTHONORMALITY_TOL)

    return {
        "headpose_r6": rotation_matrix_to_6d(rotation),
        "headpose_t_raw": translation.astype(np.float32),
        "headpose_scale": scale,
        "conf_headpose": float(confidence),
    }


def normalise_translation(translations: np.ndarray, detected: np.ndarray) -> tuple:
    """Subtract the clip-level reference position from every raw translation.

    The reference is the mean over tracked frames only, so dropouts (NaN) neither
    shift nor poison it. Returns (normalised, reference); if no frame tracked,
    the reference is NaN and the output stays NaN.
    """
    translations = np.asarray(translations, dtype=np.float64)
    detected = np.asarray(detected, dtype=bool)
    usable = detected & np.isfinite(translations).all(axis=1)
    if not usable.any():
        return np.full_like(translations, np.nan, dtype=np.float32), np.full(3, np.nan)
    reference = translations[usable].mean(axis=0)
    return (translations - reference).astype(np.float32), reference
