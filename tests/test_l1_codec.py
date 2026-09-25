"""
Unit tests for the L1 atomic text layer (src/semantic/l1.py, l1_eval.py).

Self-contained: synthetic windows only, no run directory, no MediaPipe. Plain
asserts, runnable with or without pytest, matching the rest of tests/.

The cases that matter most:
    test_run_length_segmentation_is_exactly_per_frame_quantization -- segmentation
        by itself adds no error; only the min-duration merge may.
    test_masks_are_absent_from_text_and_restored_exactly -- padding / gap-filled /
        unobserved frames never become text, and loss_weight() on the decoded
        arrays reproduces the original describable mask.
    test_rotation_goes_through_the_manifold -- rotation is quantized as a rotation
        vector reached via quaternions, never element-wise on the 6-vector.
    test_editing_one_segment_changes_only_that_channel_and_span -- the locality the
        L3 "adjust one axis, nothing else moves" goal ultimately depends on.

Usage:
    python tests/test_l1_codec.py
"""
import copy
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.geometry.headpose import rotation_6d_to_matrix, rotation_matrix_to_6d
from src.schema.feature_vector import CHANNELS, TOTAL_DIM, geodesic_angle_error, slice_for
from src.semantic import l1
from src.semantic.l1_eval import dynamics_errors, value_errors
from src.sequence.contract import IDENTITY_ROTATION_6D, loss_weight

HZ = 30.0
T = 30
IRIS = "dimensionless_iris_offset_proxy"
# Labels only; the real names come from geometry_metadata.json at run time.
NAMES = ["_neutral"] + ["bs%02d" % i for i in range(1, 52)]
NAMES[9], NAMES[10], NAMES[25] = "eyeBlinkLeft", "eyeBlinkRight", "jawOpen"
BS = slice_for("blendshape").start


def _window(n_frames: int = T) -> dict:
    features = np.zeros((n_frames, TOTAL_DIM))
    features[:, slice_for("headpose_rotation_6d")] = IDENTITY_ROTATION_6D
    return {
        "features": features,
        "quality": np.ones((n_frames, len(CHANNELS))),
        "observed": np.ones(n_frames, dtype=bool),
        "is_gap_filled": np.zeros(n_frames, dtype=bool),
        "is_padding": np.zeros(n_frames, dtype=bool),
    }


def _encode(w: dict, **kwargs) -> dict:
    return l1.encode_window(w["features"], w["quality"], w["observed"], w["is_gap_filled"],
                            w["is_padding"], HZ, NAMES, IRIS, **kwargs)


def _decode(text: dict, source: str = None, smooth: bool = None) -> dict:
    return l1.decode_window(text, NAMES, IRIS, source=source, smooth=smooth)


def _rotation_y(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Rules and quantization

def test_rules_are_validated():
    good = copy.deepcopy(l1.RULES)
    l1._validate(good)

    bad = copy.deepcopy(good)
    bad["quantizers"]["blendshape"]["representatives"][1] = 0.9
    try:
        l1._validate(bad)
        raise AssertionError("a representative outside its bin must be rejected")
    except ValueError:
        pass

    bad = copy.deepcopy(good)
    bad["quantizers"]["headpose_translation"]["edges"] = [0.3, 0.1, 0.8]
    try:
        l1._validate(bad)
        raise AssertionError("non-increasing edges must be rejected")
    except ValueError:
        pass

    bad = copy.deepcopy(good)
    del bad["channels"]["gaze"]
    try:
        l1._validate(bad)
        raise AssertionError("an enabled schema channel without a text mapping must be rejected")
    except ValueError:
        pass


def test_quantize_edges_go_to_the_upper_bin_and_keep_sign():
    q = l1.quantizer("headpose_translation")
    edges = q["edges"]
    assert list(l1.quantize([0.0, edges[0] - 1e-9, edges[0], -edges[1], 10.0], q)) == [0, 0, 1, -2, 3]
    unsigned = l1.quantizer("blendshape")
    assert list(l1.quantize([0.0, unsigned["edges"][2], 1.0], unsigned)) == [0, 3, 3]


def test_per_frame_error_is_bounded_by_the_bin():
    q = l1.quantizer("blendshape")
    values = np.random.default_rng(0).uniform(0.0, 1.0, 5000)
    error = np.abs(l1.dequantize(l1.quantize(values, q), q) - values)
    bounds = [0.0] + q["edges"] + [1.0]
    worst = max(max(rep - lo, hi - rep) for rep, lo, hi in zip(q["representatives"], bounds[:-1], bounds[1:]))
    assert error.max() <= worst + 1e-12


def test_labels_round_trip_and_are_strict():
    for name in ("blendshape", "headpose_translation", "headpose_rotation_6d"):
        q = l1.quantizer(name)
        span = range(-3, 4) if q["signed"] else range(4)
        for level in span:
            assert l1.parse_label(l1.level_label(level, q), q) == level
    signed, unsigned = l1.quantizer("headpose_translation"), l1.quantizer("blendshape")
    for label, q in (("slight", signed), ("+none", signed), ("-none", signed), ("+slight", unsigned), ("huge", unsigned)):
        try:
            l1.parse_label(label, q)
            raise AssertionError("%r must be rejected" % label)
        except ValueError:
            pass


def test_gaze_thresholds_follow_the_backend_unit():
    assert l1.quantizer("gaze", IRIS)["edges"] != l1.quantizer("gaze", "radians")["edges"]
    try:
        l1.quantizer("gaze", "degrees")
        raise AssertionError("an unknown gaze unit must not silently borrow another unit's thresholds")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Segmentation

def test_run_length_segmentation_is_exactly_per_frame_quantization():
    rng = np.random.default_rng(1)
    w = _window()
    w["features"][:, BS:BS + 52] = rng.uniform(0.0, 0.7, (T, 52))
    w["features"][:, slice_for("headpose_translation")] = rng.normal(0.0, 0.4, (T, 3))
    w["features"][:, slice_for("gaze")] = rng.normal(0.0, 0.03, (T, 2))
    w["features"][:, slice_for("headpose_rotation_6d")] = np.stack(
        [rotation_matrix_to_6d(_rotation_y(a) @ _rotation_z(b)) for a, b in rng.normal(0.0, 8.0, (T, 2))])

    decoded = _decode(_encode(w, min_segment_seconds=0.0), "level", smooth=False)["features"]
    expected = l1.quantize_window(w["features"], NAMES, IRIS)
    assert np.allclose(decoded, expected, atol=1e-6), "run-length segmentation alone must be lossless w.r.t. levels"


def test_merge_removes_flicker_keeps_events_and_is_deterministic():
    levels = np.array([0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0])
    described = np.ones(levels.size, dtype=bool)
    segments = l1.segment(levels, described, min_frames=3)
    assert [(a, b, r[0]) for a, b, r in segments] == [(0, 7, 0), (7, 11, 2), (11, 14, 0)]
    assert segments == l1.segment(levels, described, min_frames=3)


def test_merge_keeps_short_runs_two_levels_away():
    """A 2-frame blink onset next to a 'strong' peak is an event, not flicker."""
    levels = np.array([1, 1, 2, 3, 3, 3, 3, 3, 3])
    segments = l1.segment(levels, np.ones(levels.size, dtype=bool), min_frames=3)
    assert segments[0] == (0, 2, (1,)), segments


def test_segmentation_never_moves_a_frame_more_than_one_level():
    rng = np.random.default_rng(3)
    for _ in range(200):
        levels = rng.integers(-3, 4, size=(40, rng.integers(1, 4)))
        described = rng.uniform(size=40) > 0.1
        for start, stop, row in l1.segment(levels, described, min_frames=int(rng.integers(2, 6))):
            assert np.abs(levels[start:stop] - np.asarray(row)).max() <= 1
            assert described[start:stop].all()


def test_merge_prefers_the_closer_level():
    levels = np.array([0, 0, 0, 0, 2, 3, 3, 3, 3])
    segments = l1.segment(levels, np.ones(levels.size, dtype=bool), min_frames=2)
    assert [(a, b, r[0]) for a, b, r in segments] == [(0, 4, 0), (4, 9, 3)]


def test_merge_never_crosses_undescribed_frames():
    levels = np.array([0, 0, 0, 0, 3, 0, 0, 0])
    described = np.ones(levels.size, dtype=bool)
    described[3] = described[5] = False
    segments = l1.segment(levels, described, min_frames=3)
    assert (4, 5, (3,)) in segments, "an isolated short run has no neighbour to merge into and must stay"
    assert all(not (a <= 3 < b) and not (a <= 5 < b) for a, b, _ in segments)


def test_blink_is_a_rise_then_recovery():
    w = _window()
    blink = np.full(T, 0.03)
    blink[10:18] = [0.12, 0.34, 0.55, 0.6, 0.58, 0.5, 0.3, 0.12]
    w["features"][:, BS + 9] = blink
    labels = [s["level"] for s in _encode(w)["channels"]["eyeBlinkLeft"]]
    assert labels[0] == "none" and labels[-1] == "none", labels
    assert "strong" in labels and labels.index("strong") > 0, labels
    peak = labels.index("strong")
    order = {name: k for k, name in enumerate(l1.LEVELS)}
    assert all(order[a] <= order[b] for a, b in zip(labels[:peak], labels[1:peak + 1])), "monotone rise"
    assert all(order[a] >= order[b] for a, b in zip(labels[peak:], labels[peak + 1:])), "monotone recovery"


# ---------------------------------------------------------------------------
# Masks

def test_masks_are_absent_from_text_and_restored_exactly():
    w = _window()
    w["features"][:, BS + 25] = np.linspace(0.0, 0.6, T)
    w["is_padding"][25:] = True
    w["quality"][25:] = 0.0
    w["is_gap_filled"][12:14] = True
    w["observed"][20] = False

    text = _encode(w)
    hidden = w["is_padding"] | w["is_gap_filled"] | ~w["observed"]
    for segments in text["channels"].values():
        for s in segments:
            frames = range(l1._frame(s["t"][0], HZ), l1._frame(s["t"][1], HZ))
            assert not any(hidden[f] for f in frames), "masked frames must never be described"

    decoded = _decode(json.loads(json.dumps(text)))
    for key in ("is_padding", "is_gap_filled", "observed"):
        assert np.array_equal(decoded[key], w[key]), key
    original_weight = loss_weight(w["observed"], w["quality"], w["is_gap_filled"], w["is_padding"]) > 0
    decoded_weight = loss_weight(decoded["observed"], decoded["quality"], decoded["is_gap_filled"], decoded["is_padding"]) > 0
    assert np.array_equal(original_weight, decoded_weight)


def test_per_channel_invalidity_only_hides_that_channel():
    w = _window()
    gaze = [spec.name for spec in CHANNELS].index("gaze")
    w["quality"][5:9, gaze] = 0.0
    text = _encode(w)
    covered = lambda name: {f for s in text["channels"][name]
                            for f in range(l1._frame(s["t"][0], HZ), l1._frame(s["t"][1], HZ))}
    assert covered("gaze_pitch") == set(range(T)) - set(range(5, 9))
    assert covered("jawOpen") == set(range(T)), "a gaze dropout must not silence the blendshape text"


# ---------------------------------------------------------------------------
# Rotation

def test_rotation_goes_through_the_manifold():
    r6 = rotation_matrix_to_6d(_rotation_y(10.0))
    assert np.allclose(l1.rotvec_deg_from_r6(r6)[0], [0.0, 10.0, 0.0], atol=1e-5)

    q = np.array([np.cos(np.radians(5.0)), 0.0, np.sin(np.radians(5.0)), 0.0])
    assert np.allclose(l1._rotvec_deg_from_quaternion(q), l1._rotvec_deg_from_quaternion(-q)), "q and -q are one rotation"

    # Half the window at 0 degrees, half at 20 degrees about y: the window
    # reference is the 10 degree manifold mean, and each half deviates by 10.
    w = _window()
    rotation = slice_for("headpose_rotation_6d")
    w["features"][:15, rotation] = rotation_matrix_to_6d(np.eye(3))
    w["features"][15:, rotation] = rotation_matrix_to_6d(_rotation_y(20.0))
    text = _encode(w)
    assert np.allclose([text["reference"]["head_rotation"][a] for a in "xyz"], [0.0, 10.0, 0.0])
    levels = [s["level"] for s in text["channels"]["head_rotation"]]
    assert levels == [{"x": "none", "y": "-moderate", "z": "none"}, {"x": "none", "y": "+moderate", "z": "none"}], levels

    decoded = _decode(text, "level", smooth=False)["features"][:, rotation]
    rep = l1.quantizer("headpose_rotation_6d")["representatives"][2]
    angle = np.degrees(geodesic_angle_error(decoded[0], decoded[-1]))
    assert abs(angle - 2 * rep) < 1e-4, "the reference cancels: the two halves differ by exactly two representatives"
    for vector in decoded[[0, -1]]:
        matrix = rotation_6d_to_matrix(vector)
        assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6), "decoded rotation must be orthonormal"
        assert np.allclose(vector, rotation_matrix_to_6d(matrix), atol=1e-6), "decoded 6D is already on the manifold"


def test_rotation_reference_identity_option_describes_absolute_pose():
    w = _window()
    w["features"][:, slice_for("headpose_rotation_6d")] = rotation_matrix_to_6d(_rotation_y(10.0))
    assert _encode(w)["channels"]["head_rotation"][0]["level"]["y"] == "none", "static pose is the reference"

    reference = l1.RULES["channels"]["headpose_rotation_6d"]["reference"]
    saved = reference["method"]
    reference["method"] = "identity"
    try:
        text = _encode(w)
    finally:
        reference["method"] = saved
    assert text["channels"]["head_rotation"][0]["level"] == {"x": "none", "y": "+moderate", "z": "none"}
    assert text["reference"]["head_rotation"] == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_constant_rotation_round_trips_to_the_rounded_reference():
    w = _window()
    rotation = slice_for("headpose_rotation_6d")
    w["features"][:, rotation] = rotation_matrix_to_6d(_rotation_y(16.43) @ _rotation_z(4.3))
    decoded = _decode(_encode(w))["features"][:, rotation]
    error = np.degrees(geodesic_angle_error(w["features"][0, rotation], decoded[0]))
    decimals = l1.RULES["channels"]["headpose_rotation_6d"]["reference"]["decimals"]
    assert error <= np.sqrt(3) * 0.5 * 10.0 ** -decimals + 1e-6, error


def test_rotation_mean_is_a_manifold_mean():
    r6 = np.stack([rotation_matrix_to_6d(_rotation_z(20.0)), rotation_matrix_to_6d(_rotation_z(-20.0))])
    assert np.allclose(l1.mean_rotvec_deg(r6), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Text as the interface

def test_json_round_trip_is_lossless_and_neutral_is_excluded():
    rng = np.random.default_rng(2)
    w = _window()
    w["features"][:, BS:BS + 52] = rng.uniform(0.0, 0.7, (T, 52))
    text = _encode(w)
    assert "_neutral" not in text["channels"]
    assert len(text["channels"]) == 51 + 3 + 1 + 2
    direct = _decode(text)["features"]
    via_json = _decode(json.loads(json.dumps(text)))["features"]
    assert np.array_equal(direct, via_json)
    assert np.all(direct[:, BS] == l1.RULES["channels"]["blendshape"]["excluded_fill"])


def test_editing_one_segment_changes_only_that_channel_and_span():
    w = _window()
    w["features"][:, BS + 25] = np.where(np.arange(T) < 15, 0.1, 0.4)
    text = _encode(w)
    edited = copy.deepcopy(text)
    target = edited["channels"]["jawOpen"][0]
    assert target["level"] == "slight"
    target["level"] = "strong"
    start, stop = l1._frame(target["t"][0], HZ), l1._frame(target["t"][1], HZ)

    # Without smoothing the change is exactly the edited span.
    changed = np.argwhere(~np.isclose(_decode(text, smooth=False)["features"], _decode(edited, smooth=False)["features"]))
    assert changed.size and set(changed[:, 1]) == {BS + 25}
    assert set(changed[:, 0]) == set(range(start, stop))

    # With the default decode, smoothing may spread it in time, but never to another channel,
    # and the edited span still decodes into the new level's bin on average.
    before, after = _decode(text)["features"], _decode(edited)["features"]
    changed = np.argwhere(~np.isclose(before, after))
    assert set(changed[:, 1]) == {BS + 25}
    low, _ = l1.bin_bounds(3, l1.quantizer("blendshape"))
    assert after[start:stop, BS + 25].mean() >= low, "an edit to 'strong' must decode as strong"


def test_value_hint_decodes_segment_means():
    w = _window()
    w["features"][:, BS + 25] = np.linspace(0.21, 0.29, T)
    text = _encode(w)
    assert len(text["channels"]["jawOpen"]) == 1
    decoded = _decode(text, source="value_hint", smooth=False)["features"][:, BS + 25]
    assert np.allclose(decoded, np.linspace(0.21, 0.29, T).mean(), atol=1e-4)


def test_merged_segments_keep_level_and_hint_consistent():
    q = l1.quantizer("blendshape")
    # Shoulders merged into a 'strong' peak drag the mean below 0.5: the level stays, the hint is clipped.
    assert l1._consistent((3,), [0.45], q) == [0.5]
    assert l1._consistent((1,), [0.12], q) == [0.12], "a consistent hint is untouched"
    rng = np.random.default_rng(4)
    w = _window()
    w["features"][:, BS:BS + 52] = np.clip(np.cumsum(rng.normal(0, 0.08, (T, 52)), axis=0) + 0.3, 0, 1)
    for name, segments in _encode(w)["channels"].items():
        if name.startswith("bs") or name in ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen"):
            for seg in segments:
                low, high = l1.bin_bounds(l1.parse_label(seg["level"], q), q)
                assert low <= seg["value_hint"] <= high, (name, seg)


def test_refined_source_uses_the_hint_only_inside_its_level():
    q = l1.quantizer("blendshape")
    assert l1._segment_value(2, 0.3, q, "refined") == 0.3, "an unedited hint inside its bin is kept"
    assert l1._segment_value(3, 0.3, q, "refined") == l1.dequantize(3, q), "an edited level overrides a stale hint"
    signed = l1.quantizer("headpose_translation")
    assert l1._segment_value(-1, -0.15, signed, "refined") == -0.15
    assert l1._segment_value(1, -0.15, signed, "refined") == l1.dequantize(1, signed)


def test_default_decode_is_closer_than_the_step_decode():
    w = _window()
    w["features"][:, BS + 25] = 0.3 + 0.25 * np.sin(np.linspace(0.0, 2 * np.pi, T))
    text = _encode(w)
    truth = w["features"][:, BS + 25]
    step = _decode(text, "level", smooth=False)["features"][:, BS + 25]
    default = _decode(text)["features"][:, BS + 25]
    assert np.abs(default - truth).mean() < np.abs(step - truth).mean()
    assert np.abs(np.diff(default)).max() < np.abs(np.diff(step)).max(), "smoothing removes the step jumps"


def test_smoothing_keeps_constants_and_never_crosses_uncovered_frames():
    covered = np.ones(12, dtype=bool)
    covered[6] = False
    constant = np.full((12, 1), 0.4)
    assert np.allclose(l1._whittaker(constant, covered, 2.0)[covered], 0.4)
    left = np.zeros((12, 1))
    left[7:] = 1.0
    smoothed = l1._whittaker(left, covered, 2.0)
    assert np.allclose(smoothed[:6], 0.0) and np.allclose(smoothed[7:], 1.0), "the two stretches are smoothed independently"


def test_off_grid_time_is_rejected():
    text = _encode(_window())
    text["channels"]["jawOpen"][0]["t"][1] = 0.4567
    try:
        _decode(text)
        raise AssertionError("a time between grid points must be rejected, not rounded")
    except ValueError:
        pass


def _expect_value_error(fn, why: str):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(why)


def test_decode_rejects_malformed_edits():
    text = _encode(_window())

    overlapping = copy.deepcopy(text)
    overlapping["channels"]["jawOpen"] = [{"t": [0.0, 1.0], "level": "strong", "value_hint": 0.8},
                                          {"t": [0.0, 0.5], "level": "none", "value_hint": 0.0}]
    _expect_value_error(lambda: _decode(overlapping), "overlapping segments must not silently overwrite")

    misspelled = copy.deepcopy(text)
    misspelled["channels"]["jawOpn"] = misspelled["channels"].pop("jawOpen")
    _expect_value_error(lambda: _decode(misspelled), "an unknown/missing channel must not zero the channel's quality")

    no_reference = copy.deepcopy(text)
    del no_reference["reference"]["head_rotation"]
    _expect_value_error(lambda: _decode(no_reference), "a missing rotation reference must not default to identity")


def test_quantize_refuses_non_finite_and_clips_unsigned():
    unsigned, signed = l1.quantizer("blendshape"), l1.quantizer("headpose_translation")
    _expect_value_error(lambda: l1.quantize([np.nan], unsigned), "NaN must not quantize to 'strong'")
    _expect_value_error(lambda: l1.quantize([np.inf], signed), "inf must not quantize")
    assert list(l1.quantize([-0.3], unsigned)) == [0], "a negative activation is not a 'moderate' one"


def test_blendshape_names_must_be_persisted():
    try:
        l1.blendshape_names_from_metadata({"tracks": {"A_landmarks_blendshapes": {}}})
        raise AssertionError("missing names must fail with a regenerate message")
    except ValueError as error:
        assert "Re-run" in str(error)
    meta = {"tracks": {"A_landmarks_blendshapes": {"blendshape_names": NAMES}}}
    assert l1.blendshape_names_from_metadata(meta) == NAMES


def test_document_round_trip_and_schema_guard():
    windows = [_window() for _ in range(3)]
    npz = {key: np.stack([w[key] for w in windows]) for key in windows[0]}
    npz.update({"clip_id": np.array(["c"] * 3), "window_start": np.array([0, 15, 30]),
                "segment_id": np.zeros(3, dtype=int), "schema_version": np.array("2.0.0"),
                "target_hz": np.array(HZ)})
    document = l1.encode_features(npz, NAMES, IRIS)
    decoded = l1.decode_document(json.loads(json.dumps(document)))
    assert decoded["features"].shape == (3, T, TOTAL_DIM)
    assert list(decoded["window_start"]) == [0, 15, 30]
    npz["schema_version"] = np.array("1.0.0")
    try:
        l1.encode_features(npz, NAMES, IRIS)
        raise AssertionError("rules for another schema version must not be applied")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Evaluation

def test_eval_reports_zero_for_a_perfect_reconstruction():
    w = _window()
    w["features"][:, BS + 25] = np.linspace(0.0, 0.5, T)
    original = w["features"][None]
    described = np.ones((1, T, len(CHANNELS)), dtype=bool)
    values = value_errors(original, original, described, NAMES)
    assert values["blendshape"]["mae"] == 0.0 and values["headpose_rotation_6d"]["max"] < 1e-5
    dynamics = dynamics_errors(original, original, described, HZ, NAMES)
    assert dynamics["blendshape"]["velocity"]["mae"] == 0.0
    assert dynamics["blendshape"]["velocity"]["mean_abs_original"] > 0.0


def main():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nPASS: {len(tests)}/{len(tests)} L1 codec tests.")


if __name__ == "__main__":
    main()
