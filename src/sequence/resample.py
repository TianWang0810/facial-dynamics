"""
Resampling an irregular PTS timeline onto a uniform grid.

A TCN with no explicit time input assumes its samples are evenly spaced. Source
clips are not: they arrive at 25 or 30fps, with dropped frames and re-encoding
artefacts. v0 therefore resamples every clip onto one configurable rate (30 Hz by
default) before windowing. That is an engineering choice for this model, not a
claim that all models need it -- a model that consumes timestamps directly should
skip this step.

Segments, not one long interpolation
------------------------------------
Interpolating across a long gap invents motion that was never observed, and the
invented stretch is smooth, so it looks like clean data downstream. Before
resampling, the timeline is split into continuous SEGMENTS at every gap longer
than max_gap_seconds, and each segment is resampled independently. Nothing is
ever interpolated from one side of a gap to the other. The gap limit is in
seconds because it describes the physical world -- how long a face may be
unobserved before the motion between the two ends is unknowable -- and a frame
count would silently change meaning with the source frame rate.

Per-channel interpolation
-------------------------
Different channels need different rules and the schema records which is which:

    linear      blendshape coefficients, head translation, gaze angles --
                continuous scalars where a straight line between two samples is
                a defensible estimate.
    rotation    head-pose 6D. Element-wise linear interpolation of a raw 6-vector
                does NOT stay on the rotation manifold: the interpolant leaves
                the set of valid rotations between samples, and the error grows
                with the angle between them. Rotations are therefore converted to
                matrices, interpolated on the manifold (nlerp via quaternions,
                which is a geodesic-consistent shortest-path blend for the small
                inter-frame angles seen here), and converted back to 6D.

Two distinct notions of "made up"
---------------------------------
resample_clip() reports them separately and they must not be merged:

    is_resampled    the sample sits on the uniform grid rather than on an original
                    frame time. This is normal and applies to most output frames;
                    it is not a defect.
    is_gap_filled   the sample lies inside a stretch where the source had no valid
                    observation. This is a repair and must be down-weighted.
"""
import numpy as np

from src.schema.feature_vector import CHANNELS

# Resample points closer than this to an original frame time are treated as
# landing on it, which keeps float noise from marking an exact hit as resampled.
GRID_SNAP_SEC = 1e-9


def _quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), branch-stable."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def interpolate_rotation_6d(r6_values: np.ndarray, source_times: np.ndarray,
                            target_times: np.ndarray) -> np.ndarray:
    """Interpolate 6D rotations through the rotation manifold, not element-wise.

    Each source 6-vector is orthonormalised to a matrix, converted to a
    quaternion, blended with its neighbour, then converted back to 6D. Quaternion
    sign is aligned to the previous sample before blending, since q and -q are the
    same rotation but blending across the sign flip would sweep the long way
    round.
    """
    from src.geometry.headpose import rotation_6d_to_matrix, rotation_matrix_to_6d

    r6_values = np.asarray(r6_values, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)

    quaternions = np.zeros((r6_values.shape[0], 4))
    previous = None
    for index in range(r6_values.shape[0]):
        vector = r6_values[index]
        if not np.isfinite(vector).all():
            quaternions[index] = previous if previous is not None else np.array([1.0, 0.0, 0.0, 0.0])
            continue
        quaternion = _quaternion_from_matrix(rotation_6d_to_matrix(vector))
        if previous is not None and np.dot(quaternion, previous) < 0.0:
            quaternion = -quaternion
        quaternions[index] = quaternion
        previous = quaternion

    output = np.zeros((target_times.shape[0], 6), dtype=np.float64)
    positions = np.interp(target_times, source_times, np.arange(source_times.shape[0]))
    for index, position in enumerate(positions):
        low = int(np.floor(position))
        high = min(low + 1, quaternions.shape[0] - 1)
        weight = float(position - low)
        blended = (1.0 - weight) * quaternions[low] + weight * quaternions[high]
        norm = np.linalg.norm(blended)
        blended = blended / norm if norm > 1e-12 else quaternions[low]
        output[index] = rotation_matrix_to_6d(_matrix_from_quaternion(blended))
    return output


def find_segments(times: np.ndarray, valid: np.ndarray, max_gap_seconds: float) -> list:
    """Continuous stretches of valid observation, split at gaps over the limit.

    Returns (start_index, stop_index) half-open pairs into the original arrays.
    A stretch with fewer than two valid samples cannot be interpolated and is
    dropped, with the caller told how many frames that cost.
    """
    times = np.asarray(times, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(times)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []

    segments = []
    start = indices[0]
    for previous, current in zip(indices[:-1], indices[1:]):
        if times[current] - times[previous] > max_gap_seconds:
            segments.append((int(start), int(previous) + 1))
            start = current
    segments.append((int(start), int(indices[-1]) + 1))
    return [(a, b) for a, b in segments if b - a >= 2]


def resample_clip(features: np.ndarray, quality: np.ndarray, times: np.ndarray,
                  target_hz: float, max_gap_seconds: float,
                  valid: np.ndarray = None) -> dict:
    """Resample one clip's (T, D) features onto a uniform target_hz grid.

    Each continuous segment is resampled independently; output segments are
    concatenated and their boundaries reported, so a window can be prevented from
    straddling a gap. Returns arrays plus per-sample provenance.
    """
    features = np.asarray(features, dtype=np.float64)
    quality = np.asarray(quality, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(times) & np.isfinite(features).all(axis=1)
    valid = np.asarray(valid, dtype=bool)

    step = 1.0 / float(target_hz)
    segments = find_segments(times, valid, max_gap_seconds)

    out_features, out_quality, out_times = [], [], []
    out_segment_id, out_source_index, out_gap_filled = [], [], []
    segment_spans = []

    for segment_id, (start, stop) in enumerate(segments):
        segment_valid = np.flatnonzero(valid[start:stop]) + start
        source_times = times[segment_valid]
        first, last = source_times[0], source_times[-1]
        n_samples = int(np.floor((last - first) / step + 1e-9)) + 1
        if n_samples < 1:
            continue
        grid = first + step * np.arange(n_samples)

        resampled = np.empty((n_samples, features.shape[1]), dtype=np.float64)
        for spec in CHANNELS:
            block = features[segment_valid, spec.slice]
            if spec.name == "headpose_rotation_6d":
                resampled[:, spec.slice] = interpolate_rotation_6d(block, source_times, grid)
            else:
                for offset in range(spec.dim):
                    resampled[:, spec.start + offset] = np.interp(grid, source_times, block[:, offset])

        resampled_quality = np.empty((n_samples, quality.shape[1]), dtype=np.float64)
        for column in range(quality.shape[1]):
            resampled_quality[:, column] = np.interp(grid, source_times, quality[segment_valid, column])

        # A grid point inside a stretch the source did not observe is a repair,
        # not a normal resample, and the two must not be conflated: ordinary
        # resampling puts most output samples between two source samples, which is
        # expected and fine.
        #
        # The test is therefore in two parts. First, is the source interval
        # bracketing this grid point a genuine hole -- longer than the segment's
        # typical step? Ordinary 25Hz-to-30Hz resampling never satisfies this,
        # because every interval is the normal one. Second, is the point actually
        # inside that hole rather than sitting on one of its endpoints?
        #
        # Anchoring on the segment's median step rather than on the grid spacing
        # keeps the answer independent of the target rate.
        source_steps = np.diff(source_times)
        median_step = float(np.median(source_steps)) if source_steps.size else step
        hole_threshold = median_step * 1.5
        endpoint_tolerance = median_step * 0.5

        nearest = np.searchsorted(source_times, grid).clip(1, source_times.shape[0] - 1)
        left_time = source_times[nearest - 1]
        right_time = source_times[nearest]
        local_step = right_time - left_time
        distance_to_source = np.minimum(grid - left_time, right_time - grid)
        gap_filled = (local_step > hole_threshold) & (distance_to_source > endpoint_tolerance + GRID_SNAP_SEC)

        source_index = segment_valid[np.abs(source_times[:, None] - grid[None, :]).argmin(axis=0)]

        segment_spans.append((len(out_times), len(out_times) + n_samples))
        out_features.append(resampled)
        out_quality.append(resampled_quality)
        out_times.append(grid)
        out_segment_id.append(np.full(n_samples, segment_id, dtype=np.int32))
        out_source_index.append(source_index.astype(np.int32))
        out_gap_filled.append(gap_filled)

    if not out_features:
        return {
            "features": np.zeros((0, features.shape[1])), "quality": np.zeros((0, quality.shape[1])),
            "times": np.zeros(0), "segment_id": np.zeros(0, dtype=np.int32),
            "source_index": np.zeros(0, dtype=np.int32), "is_gap_filled": np.zeros(0, dtype=bool),
            "segment_spans": [], "n_segments": 0, "target_hz": float(target_hz),
            "n_source_frames": int(features.shape[0]), "n_source_valid": int(valid.sum()),
        }

    return {
        "features": np.concatenate(out_features),
        "quality": np.concatenate(out_quality),
        "times": np.concatenate(out_times),
        "segment_id": np.concatenate(out_segment_id),
        "source_index": np.concatenate(out_source_index),
        "is_gap_filled": np.concatenate(out_gap_filled),
        "segment_spans": segment_spans,
        "n_segments": len(segment_spans),
        "target_hz": float(target_hz),
        "n_source_frames": int(features.shape[0]),
        "n_source_valid": int(valid.sum()),
    }


def antialias_warning(source_hz: float, target_hz: float) -> str:
    """Whether this resample decimates, and therefore needs a low-pass first.

    np.interp does not filter, so resampling 60 Hz to 30 Hz aliases any content
    above 15 Hz back into the retained band. For the source rates this project
    actually sees (24-30 Hz to 30 Hz) the operation is interpolation or a very
    mild decimation and no filter is applied; a genuinely high-rate source needs
    one, and this returns the message that says so rather than filtering silently.
    """
    if source_hz and target_hz and source_hz > target_hz * 1.5:
        return ("source %.3f Hz decimated to %.3f Hz without an anti-alias filter; "
                "content above %.3f Hz will alias" % (source_hz, target_hz, target_hz / 2.0))
    return None
