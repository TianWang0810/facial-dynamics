"""
Round-trip error accounting for the L1 text layer.

Pure functions over (N, T, TOTAL_DIM) arrays; scripts/l1_report.py owns the I/O.
Errors are measured only on DESCRIBED frames -- where loss_weight() > 0 for the
channel -- because those are the only frames L1 claims to represent and the only
frames a model is supervised on. Undescribed frames decode to a neutral fill by
design and would otherwise dominate the numbers with an error nobody pays.

Two families of numbers:

    value     per-channel MAE / RMSE / max / signed mean (bias). Rotation is
              reported as geodesic angle in degrees after orthonormalisation,
              as the schema's rotation_evaluation_requirement demands, never as
              a raw-6D difference.
    dynamics  velocity / acceleration / jerk recomputed with the repo's own
              src/dynamics/derive.py, over each contiguous described run of each
              window. A piecewise-constant decode has zero derivative inside a
              segment and a spike at each boundary, so this is where the cost of
              segmenting shows up even when the value error looks small.
"""
import numpy as np

from src.dynamics.derive import compute_acceleration, compute_jerk, compute_velocity
from src.schema.feature_vector import CHANNELS, channel, geodesic_angle_error
from src.semantic.l1 import RULES

ROTATION_CHANNEL = "headpose_rotation_6d"
DERIVATIVES = (("velocity", compute_velocity), ("acceleration", compute_acceleration), ("jerk", compute_jerk))


def _channel_index(name: str) -> int:
    return [spec.name for spec in CHANNELS].index(name)


def _columns(name: str, blendshape_names: list) -> list:
    spec = channel(name)
    if name == "blendshape":
        excluded = RULES["channels"]["blendshape"]["exclude"]
        return [spec.start + k for k, n in enumerate(blendshape_names) if n not in excluded]
    return list(range(spec.start, spec.end))


def _scalar_stats(error: np.ndarray) -> dict:
    if error.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "max_abs": None, "bias": None}
    return {
        "n": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_abs": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def value_errors(original, recon, described, blendshape_names: list) -> dict:
    """Value-space round-trip error per schema channel on described frames."""
    original = np.asarray(original, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    described = np.asarray(described, dtype=bool)
    out = {}
    for spec in CHANNELS:
        mask = described[..., _channel_index(spec.name)]
        if spec.name == ROTATION_CHANNEL:
            a, b = original[mask][:, spec.slice], recon[mask][:, spec.slice]
            angles = np.degrees([geodesic_angle_error(x, y) for x, y in zip(a, b)])
            out[spec.name] = {
                "n_frames": int(mask.sum()), "unit": "degrees (geodesic, after Gram-Schmidt)",
                "mean": float(np.mean(angles)) if angles.size else None,
                "p95": float(np.percentile(angles, 95)) if angles.size else None,
                "max": float(np.max(angles)) if angles.size else None,
            }
            continue
        columns = _columns(spec.name, blendshape_names)
        error = recon[mask][:, columns] - original[mask][:, columns]
        entry = {"n_frames": int(mask.sum()), "n_dims": len(columns), **_scalar_stats(error)}
        if spec.name == "blendshape":
            names = [n for n in blendshape_names if n not in RULES["channels"]["blendshape"]["exclude"]]
            entry["per_name_mae"] = {n: (float(np.mean(np.abs(error[:, k]))) if error.size else None)
                                   for k, n in enumerate(names)}
        else:
            entry["per_dim"] = [_scalar_stats(error[:, k]) for k in range(error.shape[1])]
        out[spec.name] = entry
    return out


def _described_runs(mask: np.ndarray) -> list:
    runs, start = [], None
    for frame, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = frame
        elif not flag and start is not None:
            runs.append((start, frame))
            start = None
    return [(a, b) for a, b in runs if b - a >= 2]


def dynamics_errors(original, recon, described, hz: float, blendshape_names: list) -> dict:
    """Derivative statistics of original vs reconstruction, via src/dynamics/derive.py.

    Timestamps are the uniform grid the windows were cut on (features.npz
    target_hz) -- real resampled times, not a frame_idx / fps reconstruction of
    a raw clip. Rotation is differentiated as the raw 6-vector, which the schema
    names a representation-space smoothness term, NOT angular velocity.
    """
    original = np.asarray(original, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    described = np.asarray(described, dtype=bool)
    out = {}
    for spec in CHANNELS:
        columns = _columns(spec.name, blendshape_names)
        collected = {name: {"orig": [], "recon": []} for name, _ in DERIVATIVES}
        for w in range(original.shape[0]):
            for start, stop in _described_runs(described[w, :, _channel_index(spec.name)]):
                times = np.arange(start, stop, dtype=np.float64) / hz
                for name, fn in DERIVATIVES:
                    collected[name]["orig"].append(fn(original[w, start:stop][:, columns], times).ravel())
                    collected[name]["recon"].append(fn(recon[w, start:stop][:, columns], times).ravel())
        entry = {"space": "raw 6D representation (not angular velocity)" if spec.name == ROTATION_CHANNEL
                 else "channel units per second^order"}
        for name, _ in DERIVATIVES:
            if not collected[name]["orig"]:
                entry[name] = None
                continue
            a = np.concatenate(collected[name]["orig"])
            b = np.concatenate(collected[name]["recon"])
            entry[name] = {
                "mean_abs_original": float(np.mean(np.abs(a))),
                "mean_abs_recon": float(np.mean(np.abs(b))),
                "mae": float(np.mean(np.abs(b - a))),
                "max_abs_recon": float(np.max(np.abs(b))),
            }
        out[spec.name] = entry
    return out


def text_statistics(document: dict) -> dict:
    """How compact the text is: segments per text channel per window."""
    counts = [len(segments) for w in document["windows"] for segments in w["channels"].values()]
    per_channel = {}
    for w in document["windows"]:
        for name, segments in w["channels"].items():
            per_channel.setdefault(name, []).append(len(segments))
    busiest = sorted(per_channel.items(), key=lambda item: -np.mean(item[1]))[:8]
    return {
        "n_windows": len(document["windows"]),
        "n_text_channels": len(per_channel),
        "segments_total": int(np.sum(counts)),
        "segments_per_channel_window_mean": float(np.mean(counts)),
        "segments_per_channel_window_max": int(np.max(counts)),
        "busiest_channels_mean_segments": {name: float(np.mean(c)) for name, c in busiest},
    }
