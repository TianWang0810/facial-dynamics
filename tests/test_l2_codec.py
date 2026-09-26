"""
L2 phrase layer: track stitching, blink detection, exact and phrase-only decode.

Self-contained (synthetic tensors, no clips, no model).

Usage:
    python -m pytest tests/test_l2_codec.py -q
    python tests/test_l2_codec.py
"""
import copy
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.schema.feature_vector import channel  # noqa: E402
from src.semantic import l1, l2, l2_kb, l2_retrieval  # noqa: E402
from src.sequence.contract import neutral_vector  # noqa: E402

HZ = 30.0
UNIT = "dimensionless_iris_offset_proxy"
# MediaPipe's 52 category names in their order (CLAUDE.md table); L2 phrases reference them by name
NAMES = ("_neutral browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight cheekPuff cheekSquintLeft "
         "cheekSquintRight eyeBlinkLeft eyeBlinkRight eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight "
         "eyeLookOutLeft eyeLookOutRight eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight eyeWideLeft "
         "eyeWideRight jawForward jawLeft jawOpen jawRight mouthClose mouthDimpleLeft mouthDimpleRight mouthFrownLeft "
         "mouthFrownRight mouthFunnel mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight "
         "mouthPucker mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper mouthSmileLeft "
         "mouthSmileRight mouthStretchLeft mouthStretchRight mouthUpperUpLeft mouthUpperUpRight noseSneerLeft "
         "noseSneerRight").split()
BLINK_L, BLINK_R, SQUINT_L = (NAMES.index(n) for n in ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft"))
BS = channel("blendshape").start
BLINK_BOTH, BLINK_RIGHT, CLOSURE, NOD = (30, 37), (80, 86), (95, 120), (50, 62)


def series(n=120):
    x = np.tile(neutral_vector(), (n, 1)).astype(np.float32)
    x[:, BS + BLINK_L] = x[:, BS + BLINK_R] = 0.12
    x[:, BS + SQUINT_L] = 0.3
    for eye in (BLINK_L, BLINK_R):
        x[BLINK_BOTH[0]:BLINK_BOTH[1], BS + eye] = 0.7
        x[CLOSURE[0]:CLOSURE[1], BS + eye] = 0.8
    x[BLINK_RIGHT[0]:BLINK_RIGHT[1], BS + BLINK_R] = 0.6
    rotvec = np.zeros((n, 3))
    rotvec[NOD[0]:NOD[1], 0] = -9.0
    rotvec[NOD[0]:NOD[1], 1] = 1.0
    x[:, channel("headpose_rotation_6d").slice] = l1.r6_from_rotvec_deg(rotvec)
    return x


def windows(x, window=30, hop=15):
    n = x.shape[0]
    starts = list(range(0, n - window + 1, hop))
    return {
        "features": np.stack([x[s:s + window] for s in starts]),
        "quality": np.ones((len(starts), window, 4), dtype=np.float32),
        "observed": np.ones((len(starts), window), dtype=bool),
        "is_gap_filled": np.zeros((len(starts), window), dtype=bool),
        "is_padding": np.zeros((len(starts), window), dtype=bool),
        "clip_id": np.array(["spk__vid__0.0_4.0"] * len(starts)),
        "segment_id": np.zeros(len(starts), dtype=np.int32),
        "window_start": np.array(starts, dtype=np.int32),
        "window_time_sec": np.array([10.0 + s / HZ for s in starts]),
        "target_hz": np.array(HZ), "schema_version": np.array(l1.SCHEMA_VERSION),
    }


def encoded():
    doc, texts, rejected = l2.encode_document(windows(series()), NAMES, UNIT)
    return doc["tracks"][0], texts[0], rejected[0]


def test_stitch_restores_the_series_and_refuses_disagreeing_overlaps():
    x = series()
    npz = windows(x)
    (track,) = l2.stitch_tracks(npz)
    np.testing.assert_array_equal(track["features"], x)
    assert track["track_start"] == 0 and track["time_sec"] == 10.0
    npz["features"][1, 0, BS + SQUINT_L] += 0.1  # frame 15 now disagrees between windows 0 and 1
    try:
        l2.stitch_tracks(npz)
    except ValueError as err:
        assert "disagree" in str(err)
    else:
        raise AssertionError("disagreeing overlap was accepted")


def test_split_then_merge_is_identity():
    _, text, _ = encoded()
    segments = text["channels"]["eyeBlinkLeft"]
    inside, outside = l2.split_segments(segments, [(3, 31), (50, 61)], HZ)
    assert l2.merge_segments(inside + outside, HZ) == segments


def test_detects_blinks_a_long_closure_and_a_signed_head_move():
    track, _, rejected = encoded()
    got = [(p["type"], p["params"].get("symmetry", p["params"].get("dominant")), p["t"]) for p in track["phrases"]]
    assert got == [("blink", "both", [1.0, 1.2333]), ("head_move", "nod", [1.6667, 2.0667]),
                   ("blink", "right", [2.6667, 2.8667]), ("lids_lowered", "both", [3.1667, 4.0])]
    assert sorted((r["type"], r["reason"]) for r in rejected) == [
        ("blink", "too_long"), ("lids_lowered", "too_short"), ("lids_lowered", "too_short")]
    p = track["phrases"][0]["params"]
    assert p["peak"] == "strong" and p["driver"] == "eyeBlinkLeft"
    assert abs(p["channel_dev"]["eyeBlinkLeft"] - 0.58) < 1e-6 and abs(p["channel_dev"]["eyeBlinkRight"] - 0.58) < 1e-6
    nod = track["phrases"][1]["params"]
    assert nod["direction"] == "-" and nod["motion"] == "up" and nod["driver"] == "head_rotation.x"
    assert nod["channel_dev"]["head_rotation.x"] < -6
    assert track["rest"]["eyeBlinkLeft"] == 0.12


def test_exact_decode_reproduces_the_track_l1_verbatim():
    track, text, _ = encoded()
    decoded = l2.decode_track(track, UNIT, "exact")
    assert decoded["channels"] == text["channels"]
    for key in l2.L1_HEADER:
        assert decoded[key] == text[key]


def test_residual_is_empty_inside_claimed_spans():
    track, _, _ = encoded()
    for phrase in track["phrases"]:
        s, e = l1._frame(phrase["t"][0], HZ), l1._frame(phrase["t"][1], HZ)
        for ch in l2.text_channels_of(l2.VOCABULARY["phrases"][phrase["type"]]):
            covered = sum(l1._frame(x["t"][1], HZ) - l1._frame(x["t"][0], HZ) for x in phrase["detail"][ch])
            assert covered == e - s
            assert not any(s < l1._frame(x["t"][1], HZ) and l1._frame(x["t"][0], HZ) < e for x in track["residual"][ch])


def _levels(text, ch):
    values, _ = l2.channel_values(text, NAMES, UNIT)[ch]
    return l1.quantize(values[:, 0], l1.quantizer("blendshape", UNIT))


def test_phrase_only_changes_only_the_phrase_spans_and_reaches_the_peak():
    track, text, _ = encoded()
    generated = l2.decode_track(track, UNIT, "phrase_only")
    for ch in text["channels"]:
        if ch not in ("eyeBlinkLeft", "eyeBlinkRight", "head_rotation"):
            assert generated["channels"][ch] == text["channels"][ch]
    rot, _ = l2.channel_values(generated, NAMES, UNIT)["head_rotation"]
    rot_true, _ = l2.channel_values(text, NAMES, UNIT)["head_rotation"]
    assert abs(rot[NOD[0]:NOD[1], 0].min() - rot_true[NOD[0]:NOD[1], 0].min()) < 1.0
    s, e = BLINK_BOTH
    assert _levels(generated, "eyeBlinkLeft")[s:e].max() == 3
    outside = np.ones(120, dtype=bool)
    for p in track["phrases"]:
        outside[l1._frame(p["t"][0], HZ):l1._frame(p["t"][1], HZ)] = False
    np.testing.assert_array_equal(_levels(generated, "eyeBlinkRight")[outside], _levels(text, "eyeBlinkRight")[outside])


def test_an_edited_peak_level_is_followed():
    track, _, _ = encoded()
    edited = copy.deepcopy(track)
    edited["phrases"][0]["params"]["peak"] = "moderate"
    del edited["phrases"][0]["detail"]
    try:
        l2.decode_track(edited, UNIT, "exact")
    except ValueError as err:
        assert "phrase_only" in str(err)
    else:
        raise AssertionError("exact decode of an edited phrase should refuse")
    s, e = BLINK_BOTH
    assert _levels(l2.decode_track(edited, UNIT, "phrase_only"), "eyeBlinkLeft")[s:e].max() == 2


def test_vocabulary_validation_rejects_bad_thresholds():
    bad = copy.deepcopy(l2.VOCABULARY)
    bad["phrases"]["blink"]["detect"]["dev_low"] = 0.5
    try:
        l2._validate(bad)
    except ValueError as err:
        assert "dev_low" in str(err)
    else:
        raise AssertionError("dev_low > dev_high was accepted")
    bad = copy.deepcopy(l2.VOCABULARY)
    bad["phrases"]["head_move"]["members"].append("head_rotation")
    try:
        l2._validate(bad)
    except ValueError as err:
        assert "reference" in str(err)
    else:
        raise AssertionError("an axis-less rotation reference was accepted")


def _doc():
    doc, texts, _ = l2.encode_document(windows(series()), NAMES, UNIT)
    return doc, texts, {doc["tracks"][0]["track_id"]}


def test_retrieving_a_phrase_from_a_bank_that_holds_it_reproduces_its_levels():
    doc, texts, ids = _doc()
    track, text = doc["tracks"][0], texts[0]
    bank = l2_retrieval.build_bank(doc, texts, ids)
    phrase = track["phrases"][0]
    profiles, info = l2_retrieval.retrieve(bank, phrase, track["rest"], HZ)
    assert info["exemplars"][0]["t"] == phrase["t"] and info["cost"] == 0.0
    s, e = l1._frame(phrase["t"][0], HZ), l1._frame(phrase["t"][1], HZ)
    for ref, profile in profiles.items():
        np.testing.assert_allclose(profile, l2.ref_series(l2.channel_values(text, NAMES, UNIT), ref)[0][s:e], atol=1e-9)
    again, _ = l2_retrieval.retrieve(bank, phrase, track["rest"], HZ)
    assert all(np.array_equal(again[r], profiles[r]) for r in profiles)


def test_a_hand_written_phrase_decodes_with_kb_ratios():
    doc, texts, ids = _doc()
    kb = l2_kb.build(doc, texts, ids)
    assert set(kb["types"]) == {"blink", "head_move", "lids_lowered"}
    track = copy.deepcopy(doc["tracks"][0])
    written = {"id": 99, "type": "blink", "t": [0.2, 0.4333], "params": {"driver": "eyeBlinkLeft", "peak": "strong"}}
    for ch in ("eyeBlinkLeft", "eyeBlinkRight"):  # make room: the residual must not overlap the new span
        track["residual"][ch] = l2.split_segments(track["residual"][ch], [(6, 13)], HZ)[1]
    track["phrases"].append(written)
    out = l2.decode_track(track, UNIT, "phrase_only", kb=kb)
    values, _ = l2.channel_values(out, NAMES, UNIT)["eyeBlinkRight"]
    assert l1.quantize(values[6:13, 0], l1.quantizer("blendshape", UNIT)).max() == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
