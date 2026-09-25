"""
Unit tests for the audio conditioning sidecar (src/audio/features.py).

Self-contained: synthetic signals only, no mp4, no PyAV decode. Plain asserts,
runnable with or without pytest.

The case that matters most is test_alignment_is_by_time_not_by_position(): the
same sound placed at a different container start time, cut for windows at the
correspondingly shifted times, must give identical features. That is the
property that lets a clip whose video and audio streams start at different
times (normal for cuts made without re-encoding) stay in sync.

Usage:
    python tests/test_audio_features.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.audio import features as audio

SR = audio.SAMPLE_RATE


def _decoded(wave, start=0.0):
    wave = np.asarray(wave, np.float32)
    return {"has_audio": True, "wave": wave, "covered": np.ones(wave.size, bool), "start_sec": start,
            "discontinuities": []}


def test_filterbank_shape_and_peaks():
    bank = audio.mel_filterbank()
    assert bank.shape == (audio.N_MELS, audio.N_FFT // 2 + 1)
    # Low HTK filters are narrower than the 31.25 Hz FFT bin spacing, so their sampled
    # peak is below 1; what matters is that no filter is empty (every band sees energy).
    assert (bank >= 0).all() and (bank.max(axis=1) <= 1.0 + 1e-9).all()
    assert (bank.max(axis=1) > 0.3).all(), "an empty or near-empty mel filter"
    peaks = bank.argmax(axis=1)
    assert (np.diff(peaks) >= 0).all(), "filters are ordered by frequency"


def test_sine_lands_in_the_right_mel_band():
    t = np.arange(SR) / SR
    for hz in (300.0, 1000.0, 4000.0):
        mel = audio.log_mel(0.5 * np.sin(2 * np.pi * hz * t))
        band = int(np.median(mel.argmax(axis=1)))
        centres = audio._mel_to_hz(np.linspace(audio._hz_to_mel(0.0), audio._hz_to_mel(8000.0), audio.N_MELS + 2))[1:-1]
        assert abs(centres[band] - hz) / hz < 0.15, (hz, centres[band])


def test_frame_times_locate_a_click_within_one_frame():
    wave = np.zeros(2 * SR, np.float32)
    wave[int(0.5 * SR):int(0.5 * SR) + 32] = 1.0
    start = 10.0
    mel = audio.log_mel(wave)
    peak = audio.frame_centres(mel.shape[0], start)[mel.sum(axis=1).argmax()]
    assert abs(peak - 10.5) <= 1.0 / audio.FRAMES_PER_SECOND, peak

    cut = audio.window_audio(_decoded(wave, start), 10.3, 1.0)
    j = int(cut["mel"].sum(axis=1).argmax())
    centre = 10.3 + (j + 0.5) / audio.FRAMES_PER_SECOND
    assert abs(centre - 10.5) <= 1.0 / audio.FRAMES_PER_SECOND, centre
    assert cut["valid"].all()


def test_alignment_is_by_time_not_by_position():
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.1, 3 * SR).astype(np.float32)
    a = audio.window_audio(_decoded(wave, 0.0), 1.2, 1.0)
    b = audio.window_audio(_decoded(wave, 0.137), 1.2 + 0.137, 1.0)
    assert np.allclose(a["mel"], b["mel"], atol=1e-4) and np.array_equal(a["wave"], b["wave"])


def test_waveform_is_the_windowed_source():
    wave = (np.arange(3 * SR) % 1000 / 1000.0 - 0.5).astype(np.float32)
    cut = audio.window_audio(_decoded(wave, 2.0), 3.0, 1.0)
    expected = np.round(wave[SR:2 * SR] * 32767).astype(np.int16)
    assert np.array_equal(cut["wave"], expected)


def test_uncovered_and_padded_frames_are_invalid():
    wave = np.random.default_rng(1).normal(0, 0.1, SR).astype(np.float32)
    # Window reaches 0.5 s past the end of the audio.
    cut = audio.window_audio(_decoded(wave, 0.0), 0.5, 1.0)
    centres = 0.5 + (np.arange(100) + 0.5) / 100
    assert cut["valid"][centres < 0.97].all() and not cut["valid"][centres > 1.01].any()
    assert np.all(cut["mel"][~cut["valid"]] == np.float32(np.log(audio.LOG_FLOOR)))
    # Padding: video frames from 0.8 s on are padding, so audio there is masked too.
    padded = audio.window_audio(_decoded(wave, 0.0), 0.0, 1.0, padding_from_sec=0.8)
    assert not padded["valid"][centres - 0.5 >= 0.8].any()


def test_no_audio_stream_is_all_invalid_not_an_error():
    cut = audio.window_audio({"has_audio": False}, 0.0, 1.0)
    assert not cut["valid"].any() and cut["mel"].shape == (100, audio.N_MELS) and not cut["wave"].any()


def test_discontinuity_is_reported_and_left_as_a_hole():
    chunk = np.ones(1024, np.float32) * 0.1
    frames = [(0.0, chunk), (1024 / SR, chunk), (1024 / SR * 2 + 0.05, chunk)]
    wave, covered, start, jumps = audio.place_frames(frames)
    assert start == 0.0 and len(jumps) == 1 and abs(jumps[0]["jump_sec"] - 0.05) < 1e-3
    hole = slice(2048, 2048 + int(0.05 * SR))
    assert not covered[hole].any() and not wave[hole].any(), "a gap is never filled"
    cut = audio.window_audio({"has_audio": True, "wave": wave, "covered": covered, "start_sec": 0.0,
                              "discontinuities": jumps}, 0.0, 0.3)
    centres = (np.arange(30) + 0.5) / 100
    in_hole = (centres > 2048 / SR + 0.02) & (centres < 2048 / SR + 0.03)
    assert not cut["valid"][in_hole].any()


def test_continuous_chunks_have_no_discontinuity():
    chunk = np.ones(1024, np.float32)
    frames = [(k * 1024 / SR, chunk) for k in range(5)]
    wave, covered, _, jumps = audio.place_frames(frames)
    assert not jumps and covered.all() and wave.size == 5 * 1024


def test_sync_curve_recovers_a_known_delay():
    rng = np.random.default_rng(2)
    t = np.arange(0, 20, 1 / 30)
    motion = np.convolve(rng.normal(size=t.size), np.ones(5) / 5, mode="same")
    at = np.arange(0, 20, 0.01)
    energy = np.interp(at - 0.12, t, motion)  # audio 120 ms later than the mouth
    lags, corr = audio.sync_curve(t, motion, at, energy)
    assert abs(lags[np.nanargmax(corr)] - 0.12) <= 0.02


def test_video_duration_excludes_a_tail_gap_frame():
    pts = [0.0, 0.04, 0.08, 0.12, 0.24]
    assert abs(audio.video_duration(pts, exclude_tail_gap=True) - 0.16) < 1e-9
    assert abs(audio.video_duration(pts, exclude_tail_gap=False) - 0.28) < 1e-9


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} audio feature tests.")


if __name__ == "__main__":
    main()
