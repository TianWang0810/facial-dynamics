"""
Tests for resampling onto the model grid and for the training contract.

Synthetic signals only. The two cases that matter most:

test_same_motion_at_25_and_30fps_agrees_after_resampling() is the reason the
resampling step exists -- the same physical motion sampled at two rates must land
on the same grid within tolerance, or the model is learning the camera.

test_gap_is_not_bridged_and_derivatives_stop_at_the_seam() is the reason segments
exist -- interpolating across unobserved time invents smooth motion that looks
like clean data.

Usage:
    python tests/test_resample_and_contract.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.geometry.headpose import rotation_6d_to_matrix, rotation_matrix_to_6d
from src.schema.feature_vector import CHANNELS, TOTAL_DIM, channel, slice_for
from src.sequence.contract import (
    build_input, derivative_weights, loss_weight, neutral_vector, sanitize,
)
from src.sequence.resample import (
    antialias_warning, find_segments, interpolate_rotation_6d, resample_clip,
)

N_CHANNELS = len(CHANNELS)


def _rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _synthetic_clip(fps, duration=2.0):
    """One deterministic motion, sampled at whatever rate is asked for."""
    times = np.arange(0.0, duration, 1.0 / fps)
    features = np.zeros((times.shape[0], TOTAL_DIM))
    # Smooth, band-limited motion so two sample rates can agree.
    features[:, slice_for("blendshape")] = np.sin(2 * np.pi * 0.5 * times)[:, None] * 0.4 + 0.5
    features[:, slice_for("headpose_translation")] = np.cos(2 * np.pi * 0.3 * times)[:, None] * 5.0
    for index, t in enumerate(times):
        features[index, slice_for("headpose_rotation_6d")] = rotation_matrix_to_6d(
            _rotation_z(0.4 * np.sin(2 * np.pi * 0.25 * t))
        )
    features[:, slice_for("gaze")] = np.sin(2 * np.pi * 0.4 * times)[:, None] * 0.1
    quality = np.ones((times.shape[0], N_CHANNELS))
    return features, quality, times


def test_same_motion_at_25_and_30fps_agrees_after_resampling():
    a_features, a_quality, a_times = _synthetic_clip(25.0)
    b_features, b_quality, b_times = _synthetic_clip(30.0)

    a = resample_clip(a_features, a_quality, a_times, target_hz=30.0, max_gap_seconds=0.2)
    b = resample_clip(b_features, b_quality, b_times, target_hz=30.0, max_gap_seconds=0.2)

    n = min(a["features"].shape[0], b["features"].shape[0])
    assert n > 40, f"expected a couple of seconds of grid, got {n} samples"
    assert np.allclose(a["times"][:n], b["times"][:n], atol=1e-9), "both must land on the same grid"

    for spec in CHANNELS:
        difference = np.abs(a["features"][:n, spec.slice] - b["features"][:n, spec.slice]).max()
        tolerance = 0.02 if spec.name != "headpose_translation" else 0.2
        assert difference < tolerance, \
            f"{spec.name} differs by {difference:.4f} between 25fps and 30fps sources"


def test_gap_is_not_bridged_and_derivatives_stop_at_the_seam():
    times = np.array([0.0, 0.04, 0.08, 1.00, 1.04, 1.08])   # ~0.9s hole
    features = np.zeros((6, TOTAL_DIM))
    features[:, 0] = [0.0, 0.1, 0.2, 5.0, 5.1, 5.2]
    features[:, slice_for("headpose_rotation_6d")] = rotation_matrix_to_6d(np.eye(3))
    quality = np.ones((6, N_CHANNELS))

    segments = find_segments(times, np.ones(6, dtype=bool), max_gap_seconds=0.2)
    assert len(segments) == 2, f"the hole must split the clip: {segments}"

    result = resample_clip(features, quality, times, target_hz=30.0, max_gap_seconds=0.2)
    assert result["n_segments"] == 2
    assert result["features"][:, 0].max() <= 5.2 + 1e-6
    # Nothing may land between 0.2 and 5.0 -- that range exists only if the gap
    # was interpolated across.
    interior = result["features"][:, 0]
    assert not ((interior > 0.3) & (interior < 4.9)).any(), "motion was invented across the gap"

    weight = np.ones((result["features"].shape[0], N_CHANNELS))
    derivatives = derivative_weights(weight, segment_id=result["segment_id"])
    boundary = np.flatnonzero(np.diff(result["segment_id"]) != 0)
    assert derivatives["velocity"][boundary].sum() == 0.0, \
        "a first difference must not be supervised across a segment boundary"


def test_normal_resampling_is_not_confused_with_gap_filling():
    """Two different things: sitting between samples vs bridging unobserved time."""
    def run(fps, dropout=None):
        n = 60
        times = np.arange(n) / fps
        features = np.zeros((n, TOTAL_DIM))
        features[:, 0] = np.arange(n)
        features[:, slice_for("headpose_rotation_6d")] = rotation_matrix_to_6d(np.eye(3))
        quality = np.ones((n, N_CHANNELS))
        valid = np.ones(n, dtype=bool)
        if dropout:
            valid[dropout] = False
        return resample_clip(features, quality, times, target_hz=30.0,
                             max_gap_seconds=0.2, valid=valid)

    assert run(30.0)["is_gap_filled"].sum() == 0, "an on-grid source needs no repair"
    assert run(25.0)["is_gap_filled"].sum() == 0, \
        "25Hz to 30Hz puts most samples between source frames; that is resampling, not repair"

    short_hole = run(30.0, dropout=slice(20, 23))
    assert short_hole["n_segments"] == 1, "0.1s is under the 0.2s split threshold"
    assert short_hole["is_gap_filled"].sum() > 0, \
        "a hole bridged by interpolation must be flagged even when it does not split the clip"


def test_one_bad_channel_does_not_split_the_clip():
    """A blink makes gaze unavailable while every other channel stays tracked.

    Found on real data: six blink frames at 24fps span 0.25s, over the 0.2s gap
    limit, so folding per-channel invalidity into frame validity split the clip in
    two and discarded good blendshape and head-pose supervision either side.
    """
    n, fps = 72, 24.0
    times = np.arange(n) / fps
    features = np.zeros((n, TOTAL_DIM))
    features[:, 0] = np.arange(n) * 0.01
    features[:, slice_for("headpose_rotation_6d")] = rotation_matrix_to_6d(np.eye(3))
    quality = np.ones((n, N_CHANNELS))

    blink = slice(30, 36)                      # 6 frames = 0.25s > 0.2s
    gaze_index = [spec.name for spec in CHANNELS].index("gaze")
    quality[blink, gaze_index] = 0.0

    observed = np.ones(n, dtype=bool)           # the face was visible throughout
    result = resample_clip(features, quality, times, target_hz=30.0,
                           max_gap_seconds=0.2, valid=observed)

    assert result["n_segments"] == 1, \
        "a gaze dropout must not split the clip -- other channels were tracked fine"
    assert result["is_gap_filled"].sum() == 0, "nothing was unobserved, so nothing was repaired"

    # The gaze channel is still suppressed, via quality rather than segmentation.
    blink_grid = (result["times"] >= times[31]) & (result["times"] <= times[34])
    assert result["quality"][blink_grid, gaze_index].max() < 0.5, \
        "gaze quality must stay low through the blink"
    assert result["quality"][blink_grid, 0].min() > 0.9, \
        "blendshape supervision must survive the blink untouched"


def test_rotation_interpolates_on_the_manifold_not_element_wise():
    """Element-wise linear on raw 6D leaves the rotation manifold."""
    start, end = _rotation_z(0.0), _rotation_z(np.pi / 2)
    r6 = np.stack([rotation_matrix_to_6d(start), rotation_matrix_to_6d(end)])
    source_times = np.array([0.0, 1.0])
    target_times = np.array([0.5])

    manifold = interpolate_rotation_6d(r6, source_times, target_times)[0]
    recovered = rotation_6d_to_matrix(manifold)
    assert np.allclose(recovered.T @ recovered, np.eye(3), atol=1e-8), "result must be a rotation"
    angle = np.arccos(np.clip((np.trace(recovered) - 1) / 2, -1, 1))
    assert abs(angle - np.pi / 4) < 1e-6, f"midpoint should be 45 degrees, got {np.rad2deg(angle)}"

    # The naive alternative: linear on the raw 6-vector, then measure how far the
    # result is from a valid rotation before orthonormalization rescues it.
    naive = 0.5 * (r6[0] + r6[1])
    columns = naive.reshape(2, 3)
    norms = np.linalg.norm(columns, axis=1)
    assert np.abs(norms - 1.0).max() > 0.2, \
        "element-wise blending should visibly leave the manifold; if not, the test is not exercising the point"


def test_degenerate_rotation_input_does_not_crash():
    for bad in (np.zeros(6), np.array([1.0, 0, 0, 1.0, 0, 0]), np.full(6, np.nan)):
        r6 = np.stack([bad, rotation_matrix_to_6d(np.eye(3))])
        out = interpolate_rotation_6d(r6, np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0]))
        assert out.shape == (3, 6)
        assert np.isfinite(out).all(), f"degenerate input {bad} produced non-finite output"


def test_symmetric_input_does_not_move_the_stored_target():
    target = np.zeros((4, TOTAL_DIM))
    target[:, slice_for("blendshape")] = 0.25
    target[:, slice_for("gaze")] = 0.1
    original = target.copy()

    x_input = build_input(target, symmetric_blendshape=True)
    assert np.allclose(target, original), "build_input must not mutate the stored target"
    assert np.allclose(x_input[:, slice_for("blendshape")], -0.5), "[0,1] 0.25 maps to [-1,1] -0.5"
    assert np.allclose(x_input[:, slice_for("gaze")], target[:, slice_for("gaze")]), \
        "only the blendshape block is remapped"
    assert target[:, slice_for("blendshape")].min() >= 0.0, "the sigmoid target stays in [0,1]"


def test_non_finite_becomes_a_real_number_before_the_network():
    features = np.zeros((3, TOTAL_DIM))
    features[1, slice_for("gaze")] = np.nan
    features[2, slice_for("headpose_rotation_6d")] = np.inf

    clean, was_invalid = sanitize(features)
    assert np.isfinite(clean).all(), "no non-finite value may reach the network"
    assert was_invalid[1, 3] and was_invalid[2, 2]
    assert not was_invalid[0].any()

    # Rotation's neutral is the identity 6D vector; zeros are not a rotation.
    rotation = clean[2, slice_for("headpose_rotation_6d")]
    assert np.allclose(rotation, neutral_vector()[slice_for("headpose_rotation_6d")])
    recovered = rotation_6d_to_matrix(rotation)
    assert np.allclose(recovered, np.eye(3), atol=1e-9), "the rotation fill must be a valid rotation"


def test_loss_weight_separates_padding_gap_and_dropout():
    observed = np.array([True, True, False, True, True])
    quality = np.tile(np.array([1.0, 0.6, 0.0, 0.9, 0.9])[:, None], (1, N_CHANNELS))
    gap = np.array([False, False, False, True, False])
    padding = np.array([False, False, False, False, True])

    weight = loss_weight(observed, quality, gap, padding)
    assert weight[0, 0] == 1.0
    assert abs(weight[1, 0] - 0.6) < 1e-9, "a low-quality but observed frame keeps proportional weight"
    assert weight[2, 0] == 0.0, "an unobserved frame carries a neutral fill, not a measurement"
    assert weight[3, 0] == 0.0, "gap-filled samples are not supervision by default"
    assert weight[4, 0] == 0.0, "padding is never data"

    partial = loss_weight(observed, quality, gap, padding, gap_fill_weight=0.25)
    assert abs(partial[3, 0] - 0.25) < 1e-9, "the gap-fill weight must be configurable"

    # Channel granularity: one bad channel must not silence the others.
    per_channel = quality.copy()
    per_channel[0, 3] = 0.0
    weight2 = loss_weight(observed, per_channel, gap, padding)
    assert weight2[0, 3] == 0.0 and weight2[0, 0] == 1.0


def test_derivative_weights_need_two_and_three_valid_frames():
    weight = np.ones((5, N_CHANNELS))
    weight[2] = 0.0                      # one bad frame in the middle

    derivatives = derivative_weights(weight)
    velocity, acceleration = derivatives["velocity"], derivatives["acceleration"]
    assert velocity.shape[0] == 4 and acceleration.shape[0] == 3, "lengths must match a difference"
    assert velocity[1, 0] == 0.0 and velocity[2, 0] == 0.0, "both differences touching frame 2 die"
    assert velocity[0, 0] == 1.0 and velocity[3, 0] == 1.0
    assert acceleration[0, 0] == 0.0 and acceleration[1, 0] == 0.0 and acceleration[2, 0] == 0.0, \
        "every second difference spanning frame 2 must be zero"

    short = derivative_weights(np.ones((1, N_CHANNELS)))
    assert short["velocity"].shape[0] == 0 and short["acceleration"].shape[0] == 0


def test_antialias_warning_fires_only_when_decimating():
    assert antialias_warning(60.0, 30.0) is not None, "2x decimation needs a stated alias risk"
    assert antialias_warning(25.0, 30.0) is None, "upsampling does not alias"
    assert antialias_warning(30.0, 30.0) is None
    assert antialias_warning(None, 30.0) is None


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} resample/contract tests.")


if __name__ == "__main__":
    main()
