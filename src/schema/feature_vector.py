"""
Loader for the 63-dimensional feature schema.

schemas/geometry.schema.json is the single source of truth. This module reads it
at import time and derives every constant from it, so no index, width or channel
order is ever written twice. If you need the blendshape slice, ask
slice_for("blendshape") -- do not type 0:52 anywhere, in this repo or in the
modelling repo.

Slices are DERIVED from the enabled channels' widths, not read from the file, so
disabling a channel narrows the vector and leaves no hole. TOTAL_DIM follows; it
is never 63 by assumption. When every channel is enabled the derived slices must
equal the declared_slice values, and that is asserted on import -- a schema edit
that breaks the invariant fails loudly rather than silently misaligning a
training tensor, which is the failure mode this whole arrangement exists to
prevent.

Assembly and splitting are inverse operations over the same spec, so the four
consumers named in the schema -- extraction, model input, decoder heads, loss --
can all round-trip through the identical definition:

    features = assemble({"blendshape": ..., "headpose_translation": ..., ...})
    channels = split(features)          # exactly recovers the inputs
"""
import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "schemas", "geometry.schema.json",
)


@dataclass(frozen=True)
class ChannelSpec:
    """One channel's contract, mirrored verbatim from the JSON."""
    name: str
    start: int
    end: int
    dim: int
    track: str
    source_column: object
    quality_column: str
    units: str
    input_normalization: dict
    target_normalization: dict
    decoder_head: dict
    projection_dim: int
    loss: dict

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


def _load_schema(path: str = SCHEMA_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _build_channels(raw: dict) -> Tuple[ChannelSpec, ...]:
    channels = []
    cursor = 0
    for entry in raw["channels"]:
        if not entry.get("enabled", True):
            continue
        start, end = cursor, cursor + entry["dim"]
        cursor = end
        channels.append(ChannelSpec(
            name=entry["name"], start=start, end=end, dim=entry["dim"],
            track=entry["track"], source_column=entry["source_column"],
            quality_column=entry["quality_column"], units=entry["units"],
            input_normalization=entry["input_normalization"],
            target_normalization=entry["target_normalization"],
            decoder_head=entry["decoder_head"],
            projection_dim=entry["projection_dim"], loss=entry["loss"],
        ))
    return tuple(channels)


def _validate(raw: dict, channels: Tuple[ChannelSpec, ...]) -> None:
    """Check the derived layout, and that it agrees with the file when nothing is off."""
    cursor = 0
    for spec in channels:
        if spec.start != cursor or spec.end - spec.start != spec.dim:
            raise ValueError(
                "derived slice for %r is [%d, %d) with dim=%d; derivation is broken"
                % (spec.name, spec.start, spec.end, spec.dim)
            )
        cursor = spec.end

    if all(entry.get("enabled", True) for entry in raw["channels"]):
        if cursor != raw["total_dim_all_enabled"]:
            raise ValueError(
                "every channel is enabled but the derived width is %d, not total_dim_all_enabled=%d"
                % (cursor, raw["total_dim_all_enabled"])
            )
        declared = {entry["name"]: tuple(entry["declared_slice"]) for entry in raw["channels"]}
        for spec in channels:
            if declared[spec.name] != (spec.start, spec.end):
                raise ValueError(
                    "channel %r derives to [%d, %d) but declared_slice says %s; the file and the "
                    "derivation disagree" % (spec.name, spec.start, spec.end, declared[spec.name])
                )


SCHEMA = _load_schema()
SCHEMA_VERSION = SCHEMA["schema_version"]
DTYPE = np.dtype(SCHEMA["dtype"])
CHANNELS = _build_channels(SCHEMA)
CHANNEL_NAMES = tuple(spec.name for spec in CHANNELS)
QUALITY_COLUMNS = tuple(spec.quality_column for spec in CHANNELS)
WINDOWING = SCHEMA["windowing"]
SIDECAR = SCHEMA["sidecar"]

_validate(SCHEMA, CHANNELS)

# Derived from the enabled channels, never assumed. Disabling gaze makes this 61.
TOTAL_DIM = CHANNELS[-1].end if CHANNELS else 0
DISABLED_CHANNELS = tuple(
    entry["name"] for entry in SCHEMA["channels"] if not entry.get("enabled", True)
)

_BY_NAME = {spec.name: spec for spec in CHANNELS}


def channel(name: str) -> ChannelSpec:
    if name not in _BY_NAME:
        raise KeyError("unknown channel %r; schema defines %s" % (name, list(CHANNEL_NAMES)))
    return _BY_NAME[name]


def slice_for(name: str) -> slice:
    """The only sanctioned way to index into the 63-dim vector."""
    return channel(name).slice


def assemble(channel_arrays: dict) -> np.ndarray:
    """Concatenate per-channel arrays into (T, TOTAL_DIM) in schema order.

    Every channel must be present and shaped (T, dim); a missing or mis-shaped
    channel raises rather than being silently zero-filled, because a zero-filled
    channel is indistinguishable from a genuinely neutral pose downstream.
    """
    missing = [name for name in CHANNEL_NAMES if name not in channel_arrays]
    if missing:
        raise KeyError("missing channels for assembly: %s" % missing)

    n_frames = None
    for spec in CHANNELS:
        array = np.asarray(channel_arrays[spec.name])
        if array.ndim != 2 or array.shape[1] != spec.dim:
            raise ValueError(
                "channel %r must be (T, %d), got %s" % (spec.name, spec.dim, array.shape)
            )
        if n_frames is None:
            n_frames = array.shape[0]
        elif array.shape[0] != n_frames:
            raise ValueError(
                "channel %r has %d frames but earlier channels have %d"
                % (spec.name, array.shape[0], n_frames)
            )

    features = np.empty((n_frames, TOTAL_DIM), dtype=DTYPE)
    for spec in CHANNELS:
        features[:, spec.slice] = np.asarray(channel_arrays[spec.name], dtype=DTYPE)
    return features


def split(features: np.ndarray) -> dict:
    """Inverse of assemble(): (T, TOTAL_DIM) -> {channel name: (T, dim)}."""
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[-1] != TOTAL_DIM:
        raise ValueError("features must be (T, %d), got %s" % (TOTAL_DIM, features.shape))
    return {spec.name: features[:, spec.slice] for spec in CHANNELS}


def from_dataframe(df) -> Tuple[np.ndarray, np.ndarray]:
    """Build (features, quality) for one clip from geometry.parquet rows.

    The frame order is the caller's responsibility -- sort by frame_idx first.
    Returns features (T, TOTAL_DIM) and quality (T, n_channels), the latter in
    the same channel order, so quality[:, k] masks the slice of CHANNELS[k].
    """
    channel_arrays = {}
    for spec in CHANNELS:
        columns = spec.source_column if isinstance(spec.source_column, list) else [spec.source_column]
        parts = []
        for column in columns:
            values = np.asarray(list(df[column]), dtype=np.float64)
            parts.append(values.reshape(len(df), -1))
        channel_arrays[spec.name] = np.concatenate(parts, axis=1)

    quality = np.stack(
        [np.asarray(list(df[spec.quality_column]), dtype=np.float64) for spec in CHANNELS], axis=1
    )
    return assemble(channel_arrays), quality


def expand_quality(quality: np.ndarray) -> np.ndarray:
    """Broadcast a per-channel mask (T, n_channels) to per-dimension (T, TOTAL_DIM).

    Useful for element-wise loss weighting; the compact per-channel form stays the
    canonical one, since all dimensions of a channel share a confidence.
    """
    quality = np.asarray(quality, dtype=np.float64)
    if quality.ndim != 2 or quality.shape[1] != len(CHANNELS):
        raise ValueError("quality must be (T, %d), got %s" % (len(CHANNELS), quality.shape))
    expanded = np.empty((quality.shape[0], TOTAL_DIM), dtype=np.float64)
    for index, spec in enumerate(CHANNELS):
        expanded[:, spec.slice] = quality[:, index][:, None]
    return expanded


def to_symmetric_blendshape(features: np.ndarray) -> np.ndarray:
    """Apply the schema's optional [0,1] -> [-1,1] map to the blendshape block.

    ENCODER INPUT ONLY. The reconstruction target stays in the raw [0,1] space the
    sigmoid head emits, so this must never be applied to ground truth -- see the
    channel's target_normalization. Assembling targets is a separate call that
    simply does not use this function.
    """
    spec = channel("blendshape")
    low, high = spec.input_normalization["optional_symmetric_map"]["range"]
    features = np.array(features, dtype=DTYPE, copy=True)
    features[..., spec.slice] = low + (high - low) * features[..., spec.slice]
    return features


def decoder_scale(name: str) -> np.ndarray:
    """Per-dimension output scale for a channel's decoder head.

    Returns an array of length spec.dim, so a consumer multiplies element-wise and
    can never collapse a per-axis scale into a single scalar by accident. The gaze
    head is the case that matters: pitch and yaw have different physiological
    limits (35 and 50 degrees), and applying one of them to both axes would
    distort the gaze field rather than merely mis-scale it.
    """
    spec = channel(name)
    scale = spec.decoder_head.get("scale")
    if scale is None:
        return np.ones(spec.dim, dtype=DTYPE)
    scale = np.asarray(scale, dtype=DTYPE).reshape(-1)
    if scale.shape[0] != spec.dim:
        raise ValueError(
            "channel %r declares a %d-element decoder scale but has dim %d"
            % (name, scale.shape[0], spec.dim)
        )
    return scale


def target_space(name: str) -> dict:
    """The space a channel's reconstruction target and loss live in.

    Deliberately a separate accessor from the input normalisation: the two differ
    for blendshape (input may be symmetrically mapped, the target never is) and
    the pair must not be read through one field.
    """
    return channel(name).target_normalization


def rotation_loss_contract() -> dict:
    """The v0 rotation loss contract, as declared, for training code to assert on.

    baseline_is_raw_6d says the v0 reconstruction loss compares raw 6-vectors, as
    in Zhou et al. 2019. That is a defensible baseline, not the only valid choice:
    a loss on the orthonormalized rotation (geodesic angle) is legitimate too, and
    this deliberately does not forbid it.

    requires_geodesic_evaluation is the part that is NOT optional. Raw-6D error has
    no physical unit, and two models with equal 6D error can differ substantially
    in actual angular error, so angular error in degrees must be reported
    alongside whatever the training loss happens to be.
    """
    contract = SCHEMA["loss_contract"]
    return {
        "baseline_is_raw_6d": "Raw 6-vector reconstruction loss is the v0 baseline" in contract["rotation_6d_loss_space"],
        "requires_geodesic_evaluation": bool(contract.get("rotation_evaluation_requirement")),
        "derivative_is_representation_space_only": "NOT angular velocity" in contract.get("rotation_derivative_naming", ""),
        "space": contract["rotation_6d_loss_space"],
    }


def geodesic_angle_error(r6_a, r6_b) -> float:
    """Angular error in radians between two 6D rotations, after orthonormalization.

    The evaluation metric rotation_loss_contract() requires. The trace is clipped
    before arccos because float error pushes it fractionally outside [-1, 1] for
    near-identical rotations -- exactly where the error is smallest and an
    unclipped arccos would return NaN.
    """
    from src.geometry.headpose import rotation_6d_to_matrix

    a = rotation_6d_to_matrix(np.asarray(r6_a, dtype=np.float64))
    b = rotation_6d_to_matrix(np.asarray(r6_b, dtype=np.float64))
    cosine = np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))
