"""
Temporal slicing and re-assembly, applied between the Geometry layer and the
encoder.

Sequences are cut into overlapping windows for training, and at inference time
the per-window predictions are stitched back with overlap-add blending so the
seams do not show up as steps in the reconstructed motion. The defaults (target
duration, overlap, tail handling) come from the `windowing` block of
schemas/geometry.schema.json rather than being restated here.

Window length is a DURATION, not a frame count. window_frames_for_fps() converts
the schema's target_seconds against the rate of the timeline being cut, so a
25fps and a 30fps clip get 25- and 30-frame windows that both cover 1.0s.
Holding the frame count constant instead would make the same "30 frame" window
mean 1.0s in one clip and 1.2s in another -- the opposite of what the contract
asks for.

Which rate that is depends on the caller. scripts/run_features.py resamples every
clip onto one grid (src/sequence/resample.py) BEFORE windowing and therefore
passes the grid rate, so all its windows share a frame count and stack into a
single tensor. A caller that windows raw, un-resampled clips passes each clip's
own PTS-measured rate instead, and gets windows of differing lengths that cannot
stack -- those have to be grouped into per-length buckets. This module does not
choose between the two; it converts whichever rate it is given.

Three further things this module is careful about:

Padding is never mistaken for data. The final window of a clip is usually short.
It is padded by repeating the edge frame, and the corresponding quality entries
are forced to zero, so any consumer honouring the mask ignores the padding
automatically -- whether it uses the mask as an attention mask or as a loss
weight.

Blending is weighted, not averaged. Overlap-add uses a raised-cosine taper so a
frame near a window's edge -- where the model has least context -- contributes
less than a frame at its centre. A plain mean over overlapping windows would give
the least reliable predictions equal say.

Interpolation is always flagged. fill_low_confidence() returns the filled array
and an is_interpolated mask together, never the array alone. The TCN consumption
path in the schema requires that flag so the loss can down-weight invented
frames; handing back filled data without it would erase the distinction between
observed and reconstructed motion.
"""
import numpy as np

from src.schema.feature_vector import CHANNELS, WINDOWING

# Decimal places the seconds * fps product is quantised to before rounding. A
# PTS-measured frame rate carries float noise far below this, so quantising first
# keeps clips of the same nominal rate on the same side of a rounding boundary.
FRAME_COUNT_QUANTISE_DECIMALS = 9


def frames_for_seconds(seconds: float, fps: float) -> int:
    """Window length in frames, floored at the schema's min_window_frames.

    Two rounding hazards are handled here, both of which fragment the output into
    spurious buckets when left alone:

    Half-to-even. Python's round() would make 0.5s at 25fps 12 frames but 0.5s at
    27fps 14, flipping on the parity of an exact half. This rounds half up.

    Measurement noise. fps comes from decoded PTS, so a nominally 25fps clip
    arrives as 24.999999999999979 and 0.5s lands at 12.499999999999989 -- just
    below the half, while a neighbouring clip with a clean 25.0 lands exactly on
    it. Rounding straight away would put two clips of the same real frame rate in
    different window buckets for no physical reason, so the product is quantised
    to FRAME_COUNT_QUANTISE_DECIMALS first. The noise is ~1e-14; the quantum is
    1e-9; a genuinely different frame rate is far coarser than either.
    """
    exact = round(seconds * fps, FRAME_COUNT_QUANTISE_DECIMALS)
    frames = int(np.floor(exact + 0.5))
    return max(int(WINDOWING["min_window_frames"]), frames)


def window_frames_for_fps(fps: float, target_seconds: float = None) -> int:
    """Frames covering target_seconds at this clip's own measured frame rate.

    The schema's default_target_seconds is used when none is given. This is the
    only sanctioned way to choose a window length -- there is deliberately no
    project-wide default frame count to fall back on.
    """
    if target_seconds is None:
        target_seconds = float(WINDOWING["default_target_seconds"])
    return frames_for_seconds(target_seconds, fps)


def hop_for_window(window: int, overlap_ratio: float = None) -> int:
    """Stride between consecutive windows, from the schema's default overlap."""
    if overlap_ratio is None:
        overlap_ratio = float(WINDOWING["default_overlap_ratio"])
    return max(1, int(round(window * (1.0 - overlap_ratio))))


def fps_from_pts(pts_sec) -> float:
    """Effective frame rate from the Observation layer's real PTS array.

    The median step is used rather than the mean so a single dropped frame does
    not drag the estimate; clips whose declared fps disagrees with their actual
    PTS are exactly the case this project refuses to paper over.
    """
    pts = np.asarray(pts_sec, dtype=np.float64)
    if pts.size < 2:
        raise ValueError("need at least 2 timestamps to estimate fps")
    step = float(np.median(np.diff(pts)))
    if step <= 0:
        raise ValueError("non-positive median PTS step; check Observation-layer monotonicity")
    return 1.0 / step


def window_starts(n_frames: int, window: int, hop: int, tail: str = "pad") -> list:
    """Start indices of every window covering an n_frames sequence."""
    if window < 1 or hop < 1:
        raise ValueError("window and hop must be >= 1")
    if n_frames <= 0:
        return []
    if n_frames <= window:
        return [0] if tail == "pad" else []

    starts = list(range(0, n_frames - window + 1, hop))
    covered = starts[-1] + window
    if covered < n_frames and tail == "pad":
        starts.append(starts[-1] + hop)
    return starts


def slice_windows(features: np.ndarray, quality: np.ndarray, window: int,
                  hop: int = None, tail: str = "pad") -> dict:
    """Cut (T, D) features and (T, C) quality into overlapping windows.

    `window` is required and comes from window_frames_for_fps() for this clip --
    there is no fallback constant, because a shared default frame count is exactly
    the assumption the duration-based contract rejects.

    Returns starts plus stacked (N, window, D) and (N, window, C) arrays. Padded
    tail frames carry zero quality, so padding is self-identifying.
    """
    features = np.asarray(features)
    quality = np.asarray(quality, dtype=np.float64)
    if features.shape[0] != quality.shape[0]:
        raise ValueError(
            "features has %d frames but quality has %d" % (features.shape[0], quality.shape[0])
        )

    hop = hop or hop_for_window(window)

    n_frames = features.shape[0]
    starts = window_starts(n_frames, window, hop, tail=tail)
    if not starts:
        empty_features = np.empty((0, window, features.shape[1]), dtype=features.dtype)
        empty_quality = np.empty((0, window, quality.shape[1]), dtype=np.float64)
        return {"starts": [], "features": empty_features, "quality": empty_quality,
                "window": window, "hop": hop, "n_frames": n_frames}

    feature_windows, quality_windows = [], []
    for start in starts:
        stop = start + window
        if stop <= n_frames:
            feature_windows.append(features[start:stop])
            quality_windows.append(quality[start:stop])
            continue

        n_valid = max(0, n_frames - start)
        n_pad = window - n_valid
        edge = features[-1:] if n_valid == 0 else features[start:n_frames][-1:]
        padded_features = np.concatenate([features[start:n_frames], np.repeat(edge, n_pad, axis=0)])
        padded_quality = np.concatenate([
            quality[start:n_frames], np.zeros((n_pad, quality.shape[1]), dtype=np.float64)
        ])
        feature_windows.append(padded_features)
        quality_windows.append(padded_quality)

    return {
        "starts": starts,
        "features": np.stack(feature_windows),
        "quality": np.stack(quality_windows),
        "window": window,
        "hop": hop,
        "n_frames": n_frames,
    }


def _taper(window: int) -> np.ndarray:
    """Raised-cosine blend weights, strictly positive so no frame gets zero say."""
    if window < 3:
        return np.ones(window, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(window) + 0.5) / window)
    return ramp + 1e-6


def overlap_add(windows: np.ndarray, starts, n_frames: int) -> np.ndarray:
    """Stitch (N, window, D) predictions back into (n_frames, D).

    Contributions are weighted by a raised-cosine taper and normalised by the
    accumulated weight, so overlap regions blend smoothly instead of stepping at
    the seam. Frames no window covered come back as NaN rather than zero.
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3:
        raise ValueError("windows must be (N, window, D), got %s" % (windows.shape,))
    n_windows, window, n_dims = windows.shape
    if len(starts) != n_windows:
        raise ValueError("%d starts for %d windows" % (len(starts), n_windows))

    accumulator = np.zeros((n_frames, n_dims), dtype=np.float64)
    weights = np.zeros((n_frames, 1), dtype=np.float64)
    taper = _taper(window)[:, None]

    for index, start in enumerate(starts):
        stop = min(start + window, n_frames)
        span = stop - start
        if span <= 0:
            continue
        accumulator[start:stop] += windows[index, :span] * taper[:span]
        weights[start:stop] += taper[:span]

    uncovered = weights[:, 0] <= 0.0
    weights[uncovered] = 1.0
    result = accumulator / weights
    result[uncovered] = np.nan
    return result


def fill_low_confidence(features: np.ndarray, quality: np.ndarray, threshold: float) -> tuple:
    """Linearly interpolate frames whose channel confidence is at or below threshold.

    Returns (filled, is_interpolated) where is_interpolated is (T, n_channels).
    Interpolation runs per channel, so a gaze dropout never rewrites the
    blendshape block. Frames outside the first/last reliable observation are held
    at the nearest reliable value rather than extrapolated. A channel with no
    reliable frame at all is left untouched and fully flagged.
    """
    features = np.array(features, dtype=np.float64, copy=True)
    quality = np.asarray(quality, dtype=np.float64)
    n_frames = features.shape[0]
    is_interpolated = np.zeros((n_frames, len(CHANNELS)), dtype=bool)
    frame_index = np.arange(n_frames)

    for channel_index, spec in enumerate(CHANNELS):
        bad = quality[:, channel_index] <= threshold
        if not bad.any():
            continue
        good = ~bad
        is_interpolated[:, channel_index] = bad
        if not good.any():
            continue
        for dim in range(spec.start, spec.end):
            features[bad, dim] = np.interp(frame_index[bad], frame_index[good], features[good, dim])

    return features, is_interpolated
