"""
Unit tests for src/dynamics/derive.py.

Self-contained: every case is synthetic, so this needs no downloaded clips and
no geometry.parquet. Plain asserts rather than pytest, because env-v0.1 is
frozen and does not ship a test runner (design doc section 6.4 -- adding one
would require opening env-v0.2).

The case that matters most is test_dropped_frame_vs_constant_fps(): it is the
concrete evidence for the design doc section 1 rule that timing must come from
real PTS and never from frame_idx / fps.

Usage:
    python tests/test_dynamics_derive.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.dynamics.derive import (
    compute_acceleration, compute_jerk, compute_magnitude, compute_velocity,
    summarize_dynamics,
)


def test_dropped_frame_vs_constant_fps():
    """A dropped frame makes the constant-fps assumption wrong by 100%.

    Six frames of a point moving at a constant 2.0 units/s. The PTS show a gap
    between frames 2 and 3 -- 0.12s instead of 0.04s -- exactly what a dropped
    or re-encoded frame looks like. True velocity is 2.0 at every frame.
    """
    timestamps = np.array([0.0, 0.04, 0.08, 0.20, 0.24, 0.28])
    positions = (2.0 * timestamps)[:, None]

    pts_based = compute_velocity(positions, timestamps).ravel()
    constant_fps = np.gradient(positions, 0.04, axis=0).ravel()

    assert np.allclose(pts_based, 2.0), f"PTS-based velocity should be exactly 2.0, got {pts_based}"
    assert np.allclose(constant_fps[2:4], 4.0), "constant-fps is expected to double across the gap"

    worst_error = np.abs(constant_fps - 2.0).max()
    assert worst_error == 2.0, f"expected a 100% error across the gap, got {worst_error}"
    print(f"  PTS-based       : {np.round(pts_based, 4).tolist()}  (correct)")
    print(f"  constant-fps    : {np.round(constant_fps, 4).tolist()}  (100% error at the gap)")


def test_non_uniform_matches_analytic_derivatives():
    """On randomly spaced timestamps, interior values track the analytic derivative."""
    rng = np.random.default_rng(0)
    t = np.unique(np.sort(rng.uniform(0.0, 4.0, 400)))
    values = np.stack([np.sin(2 * t), np.cos(3 * t), 0.5 * t ** 2], axis=-1)[:, None, :]

    true_velocity = np.stack([2 * np.cos(2 * t), -3 * np.sin(3 * t), t], axis=-1)[:, None, :]
    true_acceleration = np.stack([-4 * np.sin(2 * t), -9 * np.cos(3 * t), np.ones_like(t)], axis=-1)[:, None, :]

    velocity_error = np.abs(compute_velocity(values, t) - true_velocity)
    acceleration_error = np.abs(compute_acceleration(values, t) - true_acceleration)

    assert velocity_error[1:-1].max() < 1e-2, velocity_error[1:-1].max()
    assert acceleration_error[2:-2].max() < 1.0, acceleration_error[2:-2].max()
    # Boundary frames use a one-sided difference and must be visibly worse.
    assert velocity_error[[0, -1]].max() > velocity_error[1:-1].mean()
    print(f"  interior velocity max err={velocity_error[1:-1].max():.2e}, "
          f"boundary max err={velocity_error[[0, -1]].max():.2e} (one-sided, expected worse)")


def test_linear_signal_is_exact_including_boundaries():
    """A one-sided difference is still exact on a linear signal, at any spacing."""
    t = np.array([0.0, 0.013, 0.077, 0.078, 0.31, 0.5])
    values = (3.0 * t + 1.0)[:, None]
    assert np.allclose(compute_velocity(values, t), 3.0)


def test_output_length_and_alignment_preserved():
    """No frame is dropped or padded, so result[i] lines up with geometry frame i."""
    t = np.linspace(0.0, 1.0, 17)
    values = np.random.default_rng(1).normal(size=(17, 478, 3))
    for fn in (compute_velocity, compute_acceleration, compute_jerk):
        assert fn(values, t).shape == values.shape


def test_nan_propagation_width_and_parity():
    """A NaN at frame i spoils i +/- k, i +/- (k-2), ... but not frame i at order 1."""
    t = np.arange(10.0)
    values = np.arange(10.0)[:, None]
    values[5] = np.nan

    expected = {1: [4, 6], 2: [3, 5, 7], 3: [2, 4, 6, 8]}
    for order, fn in ((1, compute_velocity), (2, compute_acceleration), (3, compute_jerk)):
        spoiled = np.where(~np.isfinite(fn(values, t).ravel()))[0].tolist()
        assert spoiled == expected[order], f"order {order}: expected {expected[order]}, got {spoiled}"
    print(f"  NaN at frame 5 -> order1 {expected[1]}, order2 {expected[2]}, order3 {expected[3]}")


def test_magnitude_reduction_rule():
    """3-D landmark derivatives reduce with an L2 norm; 2-D blendshapes use abs."""
    landmark_like = np.array([[[3.0, 4.0, 0.0]]])
    assert compute_magnitude(landmark_like).shape == (1, 1)
    assert np.allclose(compute_magnitude(landmark_like), 5.0)

    blendshape_like = np.array([[-2.0, 1.0]])
    assert np.allclose(compute_magnitude(blendshape_like), [[2.0, 1.0]])


def test_invalid_input_raises():
    """Bad timing data fails loudly instead of silently producing inf."""
    cases = {
        "non-monotonic timestamps": (np.zeros((3, 1)), np.array([0.0, 0.0, 1.0])),
        "length mismatch": (np.zeros((3, 1)), np.array([0.0, 1.0])),
        "single frame": (np.zeros((1, 1)), np.array([0.0])),
        "decreasing timestamps": (np.zeros((3, 1)), np.array([0.0, 2.0, 1.0])),
    }
    for description, (values, timestamps) in cases.items():
        try:
            compute_velocity(values, timestamps)
        except ValueError:
            continue
        raise AssertionError(f"{description}: expected ValueError, none raised")


def test_summarize_is_pure_and_reports_boundary_split():
    """Same input -> same output, and interior stats exclude the one-sided frames."""
    rng = np.random.default_rng(2)
    t = np.cumsum(rng.uniform(0.03, 0.05, 60))
    values = rng.normal(size=(60, 5, 3))

    first, second = summarize_dynamics(values, t), summarize_dynamics(values, t)
    assert first == second, "summarize_dynamics must be deterministic"
    assert first["timestep_sec"]["is_uniform"] is False
    for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        assert first[name]["n_boundary_frames_excluded"] == 2 * order
        assert first[name]["all_finite"] is True


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} dynamics tests.")


if __name__ == "__main__":
    main()
