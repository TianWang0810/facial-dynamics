"""
Derive velocity / acceleration / jerk from a canonicalized Geometry sequence by
differentiating against real per-frame PTS timestamps.

Design doc section 3 defines this layer as a pure geometry -> dynamics transform.
Nothing here reads files, prints, or keeps state, and no per-frame derivative is
persisted; the CLI entrypoint (scripts/run_dynamics.py) owns all I/O and
recomputes on demand.

Differentiation scheme: timestamps are the Observation layer's real PTS, never
frame_idx / fps, because variable-frame-rate and re-encoded clips break the
constant-step assumption (design doc section 1). Every derivative therefore uses
the non-uniform form of np.gradient, which consumes the coordinate array instead
of a scalar step: interior frames get a second-order central difference over the
two unequal neighbouring intervals, boundary frames a first-order one-sided
difference.

Boundary frames (first / last): nothing is dropped and nothing is padded, so the
output always has the same length as the input and result[i] lines up with frame
i of geometry.parquet. The cost is that frame 0 and frame T-1 come from a
one-sided difference and are lower-order, hence noisier, than the interior. Each
additional derivative order widens that edge -- acceleration's boundary values
are built from already-one-sided velocities, jerk's from already-degraded
accelerations -- so for order k the outermost k frames at each end should be read
as lower-confidence. summarize_dynamics() reports interior-only statistics
alongside the full-sequence ones to keep that gap visible.

NaN frames: frames where tracking failed arrive as NaN from
src/geometry/canonicalization.py and stay NaN. A central difference reads its two
neighbours but not the frame itself, so at order 1 a NaN at frame i poisons
frames i-1 and i+1 while frame i comes back finite. Each further order widens the
hole by one frame on each side and flips its parity: order 2 spoils i-2, i and
i+2, order 3 spoils i-3, i-1, i+1 and i+3. The propagation is deliberate --
interpolating across a tracking dropout would invent motion that was never
observed -- so summarize_dynamics() reports n_nan_frames per derivative order
instead of assuming it matches the input count.
"""
import numpy as np

EDGE_ORDER = 1
UNIFORM_STEP_TOL_SEC = 1e-6
DERIVATIVE_ORDERS = ((1, "velocity"), (2, "acceleration"), (3, "jerk"))

# A derivative's unit is the differentiated signal's own unit per second^order,
# so it is a property of WHICH geometry column was passed in -- not of this
# module. summarize_dynamics() previously hard-coded the landmark unit, which
# labelled a blendshape run (dimensionless coefficients) as eye-distance units.
# The mapping lives here so the report and anything reading dynamics_report.json
# share one vocabulary.
SIGNAL_UNITS = {
    "landmarks": "canonical eye-distance units per second^order",
    "blendshapes": "dimensionless ARKit blendshape coefficient per second^order",
}
UNSPECIFIED_UNITS = "unspecified: the caller did not name the differentiated signal"


def units_for_signal(signal: str = None) -> str:
    """Unit string for a signal, or an explicit "unspecified" for an unknown one.

    Deliberately not a guess: an unrecognised signal gets a string that says the
    unit is unknown, rather than the unit of whichever column happens to be the
    default.
    """
    return SIGNAL_UNITS.get(signal, UNSPECIFIED_UNITS)


def _differentiate(values: np.ndarray, timestamps: np.ndarray, order: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be 1-D, got shape {timestamps.shape}")
    if values.ndim < 1 or values.shape[0] != timestamps.shape[0]:
        raise ValueError(f"values has {values.shape[0]} frames but timestamps has {timestamps.shape[0]}")
    if timestamps.shape[0] < 2:
        raise ValueError("at least 2 frames are required to differentiate")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing; check Observation-layer PTS monotonicity")

    derivative = values
    for _ in range(order):
        derivative = np.gradient(derivative, timestamps, axis=0, edge_order=EDGE_ORDER)
    return derivative


def compute_velocity(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """First time derivative of a geometry signal, in the signal's own units per second.
    Same shape as values; frames 0 and T-1 are one-sided (see module docstring)."""
    return _differentiate(values, timestamps, order=1)


def compute_acceleration(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Second time derivative, in the signal's own units per second squared.
    The outermost 2 frames at each end are degraded by repeated one-sided edges."""
    return _differentiate(values, timestamps, order=2)


def compute_jerk(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Third time derivative, in the signal's own units per second cubed.
    The outermost 3 frames at each end are degraded by repeated one-sided edges."""
    return _differentiate(values, timestamps, order=3)


def compute_magnitude(derivative: np.ndarray) -> np.ndarray:
    """Per-frame, per-element magnitude of a derivative. A 3-D (frames, points,
    coords) array such as a landmark derivative is reduced with an L2 norm over
    the coordinate axis, giving one scalar speed per landmark. A 2-D
    (frames, channels) array such as a blendshape derivative has no vector axis
    to reduce, so the absolute value is used instead."""
    derivative = np.asarray(derivative, dtype=np.float64)
    if derivative.ndim >= 3:
        return np.linalg.norm(derivative, axis=-1)
    return np.abs(derivative)


def _finite_stat(reducer, array: np.ndarray):
    finite = array[np.isfinite(array)]
    return float(reducer(finite)) if finite.size else None


def _nan_frame_count(array: np.ndarray) -> int:
    mask = ~np.isfinite(np.asarray(array, dtype=np.float64))
    if mask.ndim == 1:
        return int(mask.sum())
    return int(mask.any(axis=tuple(range(1, mask.ndim))).sum())


def _magnitude_stats(magnitude: np.ndarray, order: int) -> dict:
    n_frames = magnitude.shape[0]
    interior = magnitude[order:n_frames - order] if n_frames > 2 * order else magnitude[:0]
    return {
        "mean": _finite_stat(np.mean, magnitude),
        "max": _finite_stat(np.max, magnitude),
        "interior_mean": _finite_stat(np.mean, interior),
        "interior_max": _finite_stat(np.max, interior),
        "n_boundary_frames_excluded": 2 * order,
        "n_nan_frames": _nan_frame_count(magnitude),
        "all_finite": bool(np.isfinite(magnitude).all()),
    }


def summarize_dynamics(values: np.ndarray, timestamps: np.ndarray, signal: str = None) -> dict:
    """Validation summary (mean / max / NaN checks) for the first three derivatives.
    Pure: the same (values, timestamps, signal) always yields the same dict.
    "interior" statistics exclude the outermost `order` frames at each end, i.e.
    exactly the frames whose values come from one-sided differences.

    `signal` names the geometry column being differentiated and is used only to
    label the reported units; see units_for_signal()."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    steps = np.diff(timestamps)

    summary = {
        "n_frames": int(timestamps.shape[0]),
        "duration_sec": float(timestamps[-1] - timestamps[0]),
        "timestep_sec": {
            "min": float(steps.min()), "max": float(steps.max()), "mean": float(steps.mean()),
            "is_uniform": bool(np.ptp(steps) <= UNIFORM_STEP_TOL_SEC),
        },
        "n_nan_frames_input": _nan_frame_count(values),
        "units": units_for_signal(signal),
    }
    for order, name in DERIVATIVE_ORDERS:
        summary[name] = _magnitude_stats(compute_magnitude(_differentiate(values, timestamps, order)), order)
    return summary
