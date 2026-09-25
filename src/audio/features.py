"""
Audio conditioning features: decode once, log-mel, slice per video window by real time.

Every constant comes from schemas/audio_features.json. The feature math is pure
numpy (no librosa / torch), so it runs inside the frozen env-v0.1 and is
testable on synthetic signals; only decode_audio() touches PyAV.

Time
----
Audio and video PTS share one container timeline. Every audio sample is placed
by its own frame's PTS and every video window is addressed by its own grid time,
so a difference in stream start times (reported by the Observation layer) is
absorbed without any offset being assumed. Nothing is aligned by array position.

What PTS cannot reveal is a stream whose CONTENT is shifted while both start at
0; its tell is a video/audio duration mismatch. A clip whose durations differ by
more than av_duration_tolerance_sec gets its audio marked invalid, never
re-aligned by a guessed offset. (On the 16-clip TalkVid pilot none did: all were
within 0.06 s. Use the video's real PTS for its duration -- the container's
declared average rate was wrong for at least one clip.)

Gaps
----
A discontinuity in the audio PTS is reported and left as a hole: samples stay
at their own timestamps, and any audio frame the decoded samples do not cover
is marked invalid. The same "never repair a damaged timeline" rule as video.
"""
import json
import os

import numpy as np

SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "schemas", "audio_features.json",
)
with open(SPEC_PATH, encoding="utf-8") as _handle:
    SPEC = json.load(_handle)

SAMPLE_RATE = int(SPEC["decode"]["sample_rate"])
AV_DURATION_TOL = float(SPEC["decode"]["av_duration_tolerance_sec"])
SYNC_MAX_LAG = float(SPEC["decode"]["sync_check_max_lag_sec"])
DISCONTINUITY_TOL = float(SPEC["decode"]["discontinuity_tolerance_sec"])
MEL = SPEC["log_mel"]
WIN, HOP, N_FFT, N_MELS = int(MEL["win_length"]), int(MEL["hop_length"]), int(MEL["n_fft"]), int(MEL["n_mels"])
LOG_FLOOR = float(MEL["log_floor"])
FRAMES_PER_SECOND = int(SPEC["windowing"]["frames_per_second"])
if SAMPLE_RATE % HOP or SAMPLE_RATE // HOP != FRAMES_PER_SECOND:
    raise ValueError("audio_features.json: sample_rate / hop_length must equal frames_per_second")
if N_FFT < WIN:
    raise ValueError("audio_features.json: n_fft must be >= win_length")


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = None, fmax: float = None) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangular HTK-mel filters, unit peak."""
    fmin = float(MEL["fmin_hz"]) if fmin is None else fmin
    fmax = float(MEL["fmax_hz"]) if fmax is None else fmax
    bins = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    lower, centre, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bins[None, :] - lower) / (centre - lower)
    falling = (upper - bins[None, :]) / (upper - centre)
    return np.clip(np.minimum(rising, falling), 0.0, None)


_FILTERS = mel_filterbank()
_WINDOW = np.hanning(WIN + 1)[:-1]  # periodic Hann


def log_mel(wave: np.ndarray) -> np.ndarray:
    """(n_samples,) float waveform in [-1, 1] -> (n_frames, n_mels) log power mel.

    Frame k covers samples [k * HOP, k * HOP + WIN); frames that would run past
    the end are not produced, so every frame is built from real samples only.
    """
    wave = np.asarray(wave, dtype=np.float64)
    if wave.shape[0] < WIN:
        return np.zeros((0, N_MELS))
    n_frames = 1 + (wave.shape[0] - WIN) // HOP
    index = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    spectrum = np.abs(np.fft.rfft(wave[index] * _WINDOW, n=N_FFT, axis=1)) ** float(MEL["power"])
    return np.log(np.maximum(spectrum @ _FILTERS.T, LOG_FLOOR))


def frame_centres(n_frames: int, audio_start_sec: float) -> np.ndarray:
    """Container time of each log-mel frame's centre."""
    return audio_start_sec + (np.arange(n_frames) * HOP + WIN / 2.0) / SAMPLE_RATE


def place_frames(frames: list, sample_rate: int = SAMPLE_RATE) -> tuple:
    """Lay decoded (pts_sec, samples) chunks onto one sample grid by their own PTS.

    Returns (wave, covered, start_sec, discontinuities). The grid starts at the
    first chunk's PTS; each later chunk is written at round((pts - start) * sr).
    A chunk that does not continue where the previous one ended (beyond the
    tolerance) is a discontinuity: it is recorded, the uncovered stretch stays
    zero with covered = False, and overlapping samples keep the earlier chunk.
    """
    frames = [(float(pts), np.asarray(samples, dtype=np.float32)) for pts, samples in frames
              if samples is not None and len(samples)]
    if not frames:
        return np.zeros(0, np.float32), np.zeros(0, bool), None, []
    start = frames[0][0]
    positions = [int(round((pts - start) * sample_rate)) for pts, _ in frames]
    total = max(p + len(s) for p, (_, s) in zip(positions, frames))
    wave = np.zeros(total, np.float32)
    covered = np.zeros(total, bool)
    discontinuities = []
    expected_end = None
    for position, (pts, samples) in zip(positions, frames):
        if expected_end is not None and abs(position - expected_end) > DISCONTINUITY_TOL * sample_rate:
            discontinuities.append({"at_sec": round(pts, 6),
                                    "jump_sec": round((position - expected_end) / sample_rate, 6)})
        stop = position + len(samples)
        fresh = ~covered[position:stop]
        wave[position:stop][fresh] = samples[fresh]
        covered[position:stop] = True
        expected_end = stop
    return wave, covered, start, discontinuities


def decode_audio(mp4_path: str) -> dict:
    """Decode the first audio stream to SAMPLE_RATE mono, each chunk at its own PTS.

    Returns has_audio False (and nothing else) for a file with no audio stream;
    the caller marks every window's audio invalid rather than failing the clip.
    """
    import av

    with av.open(mp4_path) as container:
        if not container.streams.audio:
            return {"has_audio": False}
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            for out in resampler.resample(frame):
                pts = out.pts * out.time_base if out.pts is not None else frame.pts * frame.time_base
                chunks.append((float(pts), out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0))
        for out in resampler.resample(None):
            if out.pts is not None:
                chunks.append((float(out.pts * out.time_base), out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0))
    wave, covered, start, discontinuities = place_frames(chunks)
    return {"has_audio": True, "wave": wave, "covered": covered, "start_sec": start,
            "discontinuities": discontinuities, "source_rate": int(stream.rate)}


def window_audio(decoded: dict, window_time_sec: float, window_seconds: float, padding_from_sec: float = None) -> dict:
    """Audio for one video window [t0, t0 + window_seconds), on that window's own grid.

    mel (F, N_MELS), wave (S,) int16 and a per-mel-frame validity mask. A mel
    frame is valid only if all samples it was built from were decoded and it
    lies before padding_from_sec (the start of the window's padded video frames).
    """
    n_mel = int(round(window_seconds * FRAMES_PER_SECOND))
    n_wave = int(round(window_seconds * SAMPLE_RATE))
    mel = np.full((n_mel, N_MELS), np.log(LOG_FLOOR), np.float32)
    wave = np.zeros(n_wave, np.int16)
    valid = np.zeros(n_mel, bool)
    if not decoded.get("has_audio") or decoded["wave"].size == 0:
        return {"mel": mel, "wave": wave, "valid": valid}

    offset = int(round((window_time_sec - decoded["start_sec"]) * SAMPLE_RATE))
    # Waveform: copy the covered samples that fall inside the window.
    src_lo, src_hi = max(0, offset), min(decoded["wave"].size, offset + n_wave)
    if src_hi > src_lo:
        chunk = decoded["wave"][src_lo:src_hi] * decoded["covered"][src_lo:src_hi]
        wave[src_lo - offset:src_hi - offset] = np.clip(np.round(chunk * 32767.0), -32768, 32767).astype(np.int16)

    # Mel: frames of this window start at the window's first sample; centre-padded
    # by half a window so frame j is centred at t0 + (j + 0.5) / FRAMES_PER_SECOND.
    lead = WIN // 2 - HOP // 2
    lo = offset - lead
    need = (n_mel - 1) * HOP + WIN
    segment = np.zeros(need, np.float32)
    seg_cov = np.zeros(need, bool)
    a, b = max(0, lo), min(decoded["wave"].size, lo + need)
    if b > a:
        segment[a - lo:b - lo] = decoded["wave"][a:b]
        seg_cov[a - lo:b - lo] = decoded["covered"][a:b]
    mel[:] = log_mel(segment).astype(np.float32)
    index = np.arange(WIN)[None, :] + HOP * np.arange(n_mel)[:, None]
    valid = seg_cov[index].all(axis=1)
    if padding_from_sec is not None:
        centres = window_time_sec + (np.arange(n_mel) + 0.5) / FRAMES_PER_SECOND
        valid &= centres < padding_from_sec
    mel[~valid] = np.log(LOG_FLOOR)
    return {"mel": mel, "wave": wave, "valid": valid}


def video_duration(pts_sec, exclude_tail_gap: bool) -> float:
    """First to last frame plus one median step, from the video's own PTS."""
    pts = np.asarray([p for p in pts_sec if p is not None], dtype=np.float64)
    if pts.size < 2:
        return None
    if exclude_tail_gap:
        pts = pts[:-1]
    return float(pts[-1] - pts[0] + np.median(np.diff(pts)))


def covered_duration(decoded: dict) -> float:
    covered = np.flatnonzero(decoded["covered"])
    return None if covered.size == 0 else float((covered[-1] + 1 - covered[0]) / SAMPLE_RATE)


def sync_curve(motion_times, motion, audio_times, energy, max_lag: float = None, step: float = 0.01) -> tuple:
    """(lags_sec, correlation) of mouth motion vs audio energy on a common grid.

    Positive lag = audio later than motion. Both series are interpolated onto a
    `step` grid over their common span and z-scored; the correlation at each lag
    is the mean product over the overlap.
    """
    max_lag = SYNC_MAX_LAG if max_lag is None else max_lag
    start, stop = max(motion_times[0], audio_times[0]), min(motion_times[-1], audio_times[-1])
    grid = np.arange(start, stop, step)
    lags = np.arange(-int(round(max_lag / step)), int(round(max_lag / step)) + 1)
    if grid.size <= 2 * lags.max() + 2:
        return lags * step, np.full(lags.size, np.nan)
    x = np.interp(grid, motion_times, motion)
    y = np.interp(grid, audio_times, energy)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    out = [np.mean(x[:x.size - lag] * y[lag:]) if lag >= 0 else np.mean(x[-lag:] * y[:y.size + lag]) for lag in lags]
    return lags * step, np.array(out)
