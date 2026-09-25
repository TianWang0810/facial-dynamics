"""
Unit tests for the 63-dim feature schema and the windowing that feeds the encoder.

Self-contained: synthetic arrays only, no clips, no parquet, no MediaPipe.

The layout assertions in test_schema_layout_is_the_published_contract() restate
the index ranges on purpose. Everywhere else in the codebase, duplicating those
numbers is the exact mistake this schema exists to prevent -- here the
duplication IS the test: it is a tripwire that fires if schemas/geometry.schema.json
is edited without the four downstream consumers being told.

Usage:
    python tests/test_feature_schema.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.schema.feature_vector import (
    CHANNELS, CHANNEL_NAMES, SCHEMA, TOTAL_DIM, assemble, channel, decoder_scale,
    expand_quality, from_dataframe, geodesic_angle_error, rotation_loss_contract,
    slice_for, split, target_space, to_symmetric_blendshape,
)
from src.sequence.window import (
    fill_low_confidence, fps_from_pts, frames_for_seconds, hop_for_window,
    overlap_add, slice_windows, window_frames_for_fps, window_starts,
)

N_CHANNELS = len(CHANNELS)


def _synthetic_channels(n_frames: int = 40) -> dict:
    rng = np.random.default_rng(7)
    return {
        "blendshape": rng.uniform(0.0, 1.0, size=(n_frames, 52)),
        "headpose_translation": rng.normal(0.0, 12.0, size=(n_frames, 3)),
        "headpose_rotation_6d": rng.normal(0.0, 1.0, size=(n_frames, 6)),
        "gaze": rng.normal(0.0, 0.2, size=(n_frames, 2)),
    }


def test_schema_layout_is_the_published_contract():
    assert TOTAL_DIM == 63, "with every channel enabled the vector is 63 wide"
    assert CHANNEL_NAMES == ("blendshape", "headpose_translation", "headpose_rotation_6d", "gaze")

    expected = {
        "blendshape": (0, 52),
        "headpose_translation": (52, 55),
        "headpose_rotation_6d": (55, 61),
        "gaze": (61, 63),
    }
    for name, (start, end) in expected.items():
        spec = channel(name)
        assert (spec.start, spec.end) == (start, end), f"{name} moved to [{spec.start}:{spec.end}]"
        assert slice_for(name) == slice(start, end)
        assert spec.dim == end - start

    assert channel("blendshape").decoder_head["activation"] == "sigmoid"
    assert channel("headpose_translation").decoder_head["activation"] == "identity"
    assert channel("headpose_rotation_6d").decoder_head["post_process"] == "gram_schmidt_orthonormalization"
    assert channel("gaze").decoder_head["activation"] == "tanh"


def test_assemble_split_roundtrip():
    channels = _synthetic_channels()
    features = assemble(channels)
    assert features.shape == (40, TOTAL_DIM)

    recovered = split(features)
    for name, array in channels.items():
        assert np.allclose(recovered[name], array, atol=1e-6), f"{name} did not survive the round-trip"


def test_assemble_refuses_missing_or_misshaped_channels():
    channels = _synthetic_channels()

    incomplete = dict(channels)
    incomplete.pop("gaze")
    try:
        assemble(incomplete)
        raise AssertionError("a missing channel must raise, never be zero-filled")
    except KeyError:
        pass

    wrong = dict(channels)
    wrong["headpose_rotation_6d"] = np.zeros((40, 4))  # quaternion-sized by mistake
    try:
        assemble(wrong)
        raise AssertionError("a wrong channel width must raise")
    except ValueError:
        pass

    ragged = dict(channels)
    ragged["gaze"] = np.zeros((39, 2))
    try:
        assemble(ragged)
        raise AssertionError("a frame-count mismatch must raise")
    except ValueError:
        pass


def test_from_dataframe_matches_the_schema_order():
    """A dict-of-lists stand-in for a geometry.parquet slice; no pandas needed."""
    n_frames = 5
    rng = np.random.default_rng(3)
    frame = {
        "blendshapes": [list(rng.uniform(0, 1, 52)) for _ in range(n_frames)],
        "headpose_t": [list(rng.normal(0, 5, 3)) for _ in range(n_frames)],
        "headpose_r6": [list(rng.normal(0, 1, 6)) for _ in range(n_frames)],
        "gaze_pitch": list(rng.normal(0, 0.1, n_frames)),
        "gaze_yaw": list(rng.normal(0, 0.1, n_frames)),
        "conf_blendshapes": [1.0, 1.0, 0.2, 1.0, 1.0],
        "conf_headpose": [1.0, 1.0, 1.0, 1.0, 1.0],
        "conf_gaze": [1.0, 0.0, 1.0, 1.0, 1.0],
    }

    class _Frame(dict):
        def __len__(self):
            return n_frames

    features, quality = from_dataframe(_Frame(frame))
    assert features.shape == (n_frames, TOTAL_DIM)
    assert quality.shape == (n_frames, N_CHANNELS)
    assert np.allclose(features[:, slice_for("blendshape")], np.array(frame["blendshapes"]), atol=1e-6)
    assert np.allclose(features[:, slice_for("gaze")][:, 0], frame["gaze_pitch"], atol=1e-6)
    # Both head-pose channels share one confidence column, per the schema.
    assert np.allclose(quality[:, 1], quality[:, 2])


def test_expand_quality_broadcasts_per_channel():
    quality = np.array([[0.9, 0.5, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0]])
    expanded = expand_quality(quality)
    assert expanded.shape == (2, TOTAL_DIM)
    assert np.allclose(expanded[0, slice_for("blendshape")], 0.9)
    assert np.allclose(expanded[0, slice_for("gaze")], 0.0)


def test_symmetric_blendshape_map_touches_only_its_block():
    features = assemble(_synthetic_channels(3))
    mapped = to_symmetric_blendshape(features)
    block = slice_for("blendshape")
    assert mapped[:, block].min() >= -1.0 and mapped[:, block].max() <= 1.0
    assert np.allclose(mapped[:, 52:], features[:, 52:]), "the non-blendshape channels must not move"


def test_window_starts_and_padding_is_masked():
    features = np.arange(70 * TOTAL_DIM, dtype=np.float64).reshape(70, TOTAL_DIM)
    quality = np.ones((70, N_CHANNELS))

    result = slice_windows(features, quality, window=30, hop=15)
    assert result["starts"] == [0, 15, 30, 45], f"unexpected starts: {result['starts']}"
    assert result["features"].shape == (4, 30, TOTAL_DIM)

    # The final window runs past frame 70 and must be padded with zero quality.
    tail_quality = result["quality"][-1]
    assert tail_quality[:25].all(), "real frames in the tail window must stay unmasked"
    assert not tail_quality[25:].any(), "padded frames must carry zero quality"
    assert np.allclose(result["features"][-1, 25:], features[-1]), "padding should repeat the edge frame"

    short = slice_windows(features[:10], quality[:10], window=30, hop=15)
    assert short["features"].shape == (1, 30, TOTAL_DIM)
    assert not short["quality"][0, 10:].any()

    assert window_starts(70, 30, 15, tail="drop") == [0, 15, 30]


def test_feature_padding_repeats_the_edge_so_booleans_cannot_ride_along():
    """Why is_padding is derived from window geometry, not carried as a column.

    slice_windows pads FEATURES by repeating the edge frame and pads QUALITY with
    zeros. A boolean smuggled through as a feature column therefore comes back as
    the repeated edge value, not as a padding marker -- which silently reported
    "padded: 0" on real data that had 14 padded samples.
    """
    n_frames, window, hop = 40, 30, 15
    features = np.zeros((n_frames, TOTAL_DIM))
    flag_column = np.ones(n_frames)          # a boolean riding as a feature
    features[:, 0] = flag_column
    quality = np.ones((n_frames, N_CHANNELS))

    result = slice_windows(features, quality, window=window, hop=hop)
    tail = result["features"][-1]
    assert tail[-1, 0] == 1.0, \
        "feature padding repeats the edge, so a flag column comes back True in the padding"
    assert result["quality"][-1][-1].max() == 0.0, "quality padding is zeroed, unlike features"

    # The geometric derivation run_features.py uses instead.
    starts = np.asarray(result["starts"])
    offsets = np.arange(window)[None, :]
    padding_mask = (starts[:, None] + offsets) >= n_frames
    assert padding_mask[:-1].sum() == 0, "only the tail window is padded here"
    assert padding_mask[-1].sum() == (starts[-1] + window) - n_frames
    assert padding_mask.sum() == 5, f"30-frame window at start 15 over 40 frames pads 5, got {padding_mask.sum()}"
    # The mask agrees with where quality was zeroed, which is the independent signal.
    assert np.array_equal(padding_mask[-1], result["quality"][-1].max(axis=1) == 0.0)


def test_overlap_add_reconstructs_without_seams():
    n_frames, n_dims = 100, 4
    signal = np.linspace(0.0, 1.0, n_frames)[:, None] * np.arange(1, n_dims + 1)[None, :]
    quality = np.ones((n_frames, N_CHANNELS))

    result = slice_windows(signal, quality, window=30, hop=15)
    stitched = overlap_add(result["features"], result["starts"], n_frames)

    assert stitched.shape == (n_frames, n_dims)
    assert np.allclose(stitched, signal, atol=1e-9), "overlap-add should be lossless on identical windows"

    sparse = overlap_add(result["features"][:1], result["starts"][:1], n_frames)
    assert np.isnan(sparse[40]).all(), "frames no window covered must be NaN, not silently zero"


def test_fill_low_confidence_is_per_channel_and_always_flagged():
    n_frames = 10
    features = np.tile(np.arange(n_frames, dtype=np.float64)[:, None], (1, TOTAL_DIM))
    quality = np.ones((n_frames, N_CHANNELS))
    quality[4:6, 3] = 0.0        # gaze dropout only

    filled, is_interpolated = fill_low_confidence(features, quality, threshold=0.5)

    assert is_interpolated[:, 3].sum() == 2, "the two gaze frames must be flagged"
    assert not is_interpolated[:, 0].any(), "blendshape was never low-confidence"
    assert np.allclose(filled[:, slice_for("blendshape")], features[:, slice_for("blendshape")]), \
        "a gaze dropout must not rewrite the blendshape block"
    assert np.allclose(filled[4:6, slice_for("gaze")], features[4:6, slice_for("gaze")], atol=1e-9), \
        "linear interpolation of a linear ramp should recover the original values"

    dead = np.zeros((n_frames, N_CHANNELS))
    _, all_flagged = fill_low_confidence(features, dead, threshold=0.5)
    assert all_flagged.all(), "a channel with no reliable frame must be fully flagged"


def test_fps_and_window_length_come_from_real_pts():
    pts = np.arange(0, 2.0, 1.0 / 25.0)
    assert abs(fps_from_pts(pts) - 25.0) < 1e-6

    dropped = np.delete(pts, 10)  # one missing frame must not move the median
    assert abs(fps_from_pts(dropped) - 25.0) < 1e-6

    assert frames_for_seconds(1.0, 25.0) == 25
    assert frames_for_seconds(0.5, 30.0) == 15
    assert frames_for_seconds(0.01, 25.0) == 2, "window must stay long enough to difference"


def test_window_length_holds_duration_not_frame_count():
    """The contract fixes seconds; frames follow from each clip's own rate."""
    assert window_frames_for_fps(25.0, 1.0) == 25
    assert window_frames_for_fps(30.0, 1.0) == 30, "a 30fps clip needs 30 frames for the same second"
    assert window_frames_for_fps(25.0, 0.5) == 13
    assert window_frames_for_fps(30.0, 0.5) == 15

    # A PTS-measured rate carries float noise; clips of the same real frame rate
    # must not land in different window buckets because of it.
    for noisy in (24.999999999999979, 25.0, 25.000000000000004):
        assert window_frames_for_fps(noisy, 0.5) == 13, f"{noisy!r} fragmented the 0.5s bucket"
        assert window_frames_for_fps(noisy, 1.0) == 25, f"{noisy!r} fragmented the 1.0s bucket"

    # Both cover one second, which is the invariant that matters.
    for fps in (24.0, 25.0, 29.97, 30.0, 60.0):
        frames = window_frames_for_fps(fps, 1.0)
        assert abs(frames / fps - 1.0) < 0.03, f"{fps}fps window drifted to {frames / fps:.3f}s"

    # No project-wide frame-count default exists to fall back on.
    assert "default_window_frames" not in SCHEMA["windowing"], \
        "a fixed default frame count would reintroduce the fps assumption"
    assert window_frames_for_fps(30.0) == 30, "the default target duration should come from the schema"

    assert hop_for_window(30, 0.5) == 15
    assert hop_for_window(25, 0.5) == 12


def test_blendshape_input_map_never_touches_the_target_space():
    """Enabling the symmetric input map must not move the reconstruction target."""
    spec = channel("blendshape")
    assert spec.input_normalization["optional_symmetric_map"]["range"] == [-1.0, 1.0]
    assert spec.target_normalization["range"] == [0.0, 1.0], \
        "the target must stay in [0,1] whatever the input map does"
    assert spec.decoder_head["output_range"] == spec.target_normalization["range"], \
        "a sigmoid head and its target must span the same interval or the loss can never reach zero"
    assert "ENCODER INPUT" in spec.input_normalization["optional_symmetric_map"]["scope"].upper()

    # Gaze is the deliberate asymmetry: its compression DOES apply to both sides,
    # because the tanh head cannot reach a target beyond its own limit.
    gaze_scope = channel("gaze").input_normalization["optional_tanh_compression"]["scope"].upper()
    assert "TARGET" in gaze_scope, "gaze compression must be declared as applying to the target too"


def test_gaze_decoder_scale_is_per_axis():
    scale = decoder_scale("gaze")
    assert scale.shape == (2,), "the gaze scale must be a 2-vector, not a scalar"
    assert not np.isclose(scale[0], scale[1]), \
        "pitch and yaw limits differ (35 vs 50 degrees); a single shared value would distort the gaze field"
    assert abs(np.rad2deg(scale[0]) - 35.0) < 1e-4  # float32 storage
    assert abs(np.rad2deg(scale[1]) - 50.0) < 1e-4
    assert channel("gaze").decoder_head["scale_dim_names"] == ["pitch", "yaw"]

    # Channels without an explicit scale get ones, so element-wise use is uniform.
    assert np.allclose(decoder_scale("blendshape"), 1.0)
    assert decoder_scale("headpose_rotation_6d").shape == (6,)


def test_rotation_contract_allows_raw_6d_baseline_and_requires_angle_evaluation():
    contract = rotation_loss_contract()
    assert contract["baseline_is_raw_6d"], "raw 6D reconstruction must remain a legitimate v0 baseline"
    assert contract["requires_geodesic_evaluation"], \
        "angular error must be reported whatever the training loss is -- raw-6D MSE has no physical unit"
    assert contract["derivative_is_representation_space_only"], \
        "a raw-6D difference must not be described as angular velocity"
    # The earlier absolute claim that GS may never appear in a loss is gone.
    assert "NOT the only correct" in SCHEMA["loss_contract"]["rotation_6d_loss_space"]


def test_geodesic_angle_error_is_stable_at_both_extremes():
    from src.geometry.headpose import rotation_matrix_to_6d

    def rotation_z(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # 6D vectors are stored as float32, and arccos near zero amplifies a cosine
    # error of eps into an angle error of sqrt(2*eps): ~3e-8 rad at float32
    # precision. The property under test is that it is finite and small, not that
    # it is bit-exact.
    identical = rotation_matrix_to_6d(rotation_z(0.3))
    measured = geodesic_angle_error(identical, identical)
    assert np.isfinite(measured), "identical rotations must not give NaN from an unclipped arccos"
    assert measured < 1e-6, f"identical rotations should measure ~0, got {measured}"

    for angle in (1e-6, 0.25, np.pi / 2, np.pi - 1e-3):
        measured = geodesic_angle_error(rotation_matrix_to_6d(rotation_z(0.0)),
                                        rotation_matrix_to_6d(rotation_z(angle)))
        assert np.isfinite(measured), f"angle {angle} produced NaN"
        assert abs(measured - angle) < 1e-4, f"expected {angle}, got {measured}"

    assert np.isfinite(geodesic_angle_error(np.zeros(6), np.zeros(6))), \
        "degenerate input must not produce NaN"


def test_headpose_channels_deliberately_share_one_confidence_column():
    mask = SCHEMA["sidecar"]["quality_mask"]
    columns = mask["columns"]
    assert len(columns) == len(CHANNELS), "one mask column per channel, in channel order"
    assert columns[1] == columns[2] == "conf_headpose", \
        "translation and rotation come from one pose estimate and share its confidence"
    assert "INTENTIONAL" in mask["columns_note"].upper(), \
        "the duplication needs an explicit note or a reviewer will 'fix' it"
    assert channel("headpose_translation").quality_column == channel("headpose_rotation_6d").quality_column


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} feature-schema tests.")


if __name__ == "__main__":
    main()
