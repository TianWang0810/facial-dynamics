"""
Shared quality mask across the three geometry tracks.

Each track produces its own confidence signal (see the CONFIDENCE_METHOD dict in
landmarks.py / headpose.py / gaze.py). This module is where those signals are
combined into the per-frame, per-signal scores that land in geometry.parquet.

None of these scores is a calibrated probability. They are quality PROXIES:
monotonic in how trustworthy a frame looks, with no claim that 0.8 means an 80%
chance of anything. They are named and reported as proxies so a consumer does not
threshold them as if they were likelihoods.

Two properties are deliberate:

Per frame and per channel, not one label per clip. A clip-level "usable" hides
the fact that head pose can be solid while gaze is unusable through a 40-frame
stretch of blinks. Every frame therefore carries conf_landmarks,
conf_blendshapes, conf_headpose and conf_gaze independently, and a consumer can
mask one channel without discarding the others.

Mean AND variance, not just success/failure. A binary tracked/not-tracked flag
misses the frames that were tracked but poorly -- partially out of frame, tiny
face, near-degenerate pose fit. Those frames are the ones that quietly
contaminate training, because nothing marks them as suspect. Reporting the
distribution (mean, std, min, percentiles) plus an explicit count of
"tracked but low-confidence" frames keeps them visible: a channel with mean 0.9
and std 0.3 is a different object from one with mean 0.9 and std 0.02, even
though both look fine on the mean alone.
"""
import numpy as np

# Named TRACK_SIGNALS, not CHANNELS. These are the four signals the extraction
# tracks emit, which is NOT the same four-tuple as the schema's feature channels:
# landmarks has a signal here but no feature channel (it stays a sidecar), while
# head pose has one signal here but two feature channels sharing it. Both tuples
# happen to have length four, both produce a (T, 4) array, and reusing one name
# for both invited exactly the confusion of indexing one with the other's order.
TRACK_SIGNALS = ("landmarks", "blendshapes", "headpose", "gaze")
CONFIDENCE_COLUMNS = tuple("conf_%s" % signal for signal in TRACK_SIGNALS)

# A frame at or below this score is "low confidence". Frames that are ALSO
# detected are the silent-contamination case this module exists to surface.
LOW_CONFIDENCE_THRESHOLD = 0.5


def _percentile(values: np.ndarray, q: float):
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else None


def signal_statistics(scores: np.ndarray, detected: np.ndarray) -> dict:
    """Distribution of one track signal's per-frame confidence over a clip."""
    scores = np.asarray(scores, dtype=np.float64)
    detected = np.asarray(detected, dtype=bool)
    finite = scores[np.isfinite(scores)]
    n_frames = int(scores.shape[0])

    low = np.isfinite(scores) & (scores <= LOW_CONFIDENCE_THRESHOLD)
    low_but_tracked = int((low & detected & (scores > 0.0)).sum())

    return {
        "n_frames": n_frames,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "var": float(finite.var()) if finite.size else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "p05": _percentile(scores, 5),
        "p50": _percentile(scores, 50),
        "n_zero": int((np.isfinite(scores) & (scores <= 0.0)).sum()),
        "n_low_confidence": int(low.sum()),
        "n_low_but_tracked": low_but_tracked,
        "low_but_tracked_rate": round(low_but_tracked / n_frames, 4) if n_frames else None,
    }


def build_clip_mask_report(confidences: dict, detected: np.ndarray, clip_id: str) -> dict:
    """Per-clip quality record: per-channel distributions plus a coarse label.

    The label is retained for triage convenience only; the per-channel
    distributions above it are the actual deliverable, and downstream code should
    filter on the per-frame scores rather than on this string.
    """
    detected = np.asarray(detected, dtype=bool)
    n_frames = int(detected.shape[0])
    n_tracked = int(detected.sum())
    tracking_failure_rate = 1.0 - (n_tracked / n_frames if n_frames else 0.0)

    per_signal = {name: signal_statistics(confidences[name], detected) for name in TRACK_SIGNALS}

    if tracking_failure_rate <= 0.05:
        label = "usable"
    elif tracking_failure_rate <= 0.3:
        label = "low-quality"
    else:
        label = "reject"

    return {
        "clip_id": clip_id,
        "n_frames": n_frames,
        "n_tracked": n_tracked,
        "tracking_failure_rate": round(tracking_failure_rate, 4),
        "clip_quality_label": label,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "per_signal_confidence": per_signal,
    }
