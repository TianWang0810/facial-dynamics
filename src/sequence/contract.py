"""
The training contract: how stored tensors become x_input, y_target and weights.

The stored tensor is the TARGET space. x_input is derived from it
deterministically at load time rather than stored a second time -- two copies of
one array is a fifth of the disk and a guaranteed source of drift the first time
one path is updated and the other is not.

    y_target = stored features, per-channel target_normalization
    x_input  = build_input(y_target), per-channel input_normalization
    weight   = loss_weight(validity, quality, is_gap_filled, is_padding)

Four distinct per-frame boolean/scalar signals, deliberately never merged:

    observed        the source frame produced a real detection
    quality         [0,1] proxy score from the extractor (NOT a probability)
    is_gap_filled   the sample was invented to bridge unobserved time
    is_padding      the sample exists only to fill out a window to fixed length

They answer different questions and a consumer may weigh them differently. A
single pre-blended "mask" column cannot be taken apart again, so the blending
happens here, explicitly, where the policy is visible.

Non-finite values never reach the network. sanitize() replaces them with a
per-channel neutral value and returns the mask of what it replaced; relying on
NaN * 0 == 0 is wrong -- in IEEE 754 that is NaN, and one such value poisons an
entire batch's gradient.

Derivative supervision respects gaps. A first difference needs two consecutive
valid frames, a second difference needs three, and neither may span a segment
boundary or padding. derivative_weights() returns the masks that enforce this,
rather than leaving each loss implementation to rediscover the rule.
"""
import numpy as np

from src.schema.feature_vector import CHANNELS, TOTAL_DIM, channel

# Per-channel neutral fill used when a non-finite value has to be replaced. Zero
# is correct for every current channel: a zero blendshape coefficient is a
# neutral expression, zero translation is the clip baseline, zero gaze is the
# clip mean direction. Rotation is the exception -- the 6D vector of the identity
# rotation is not zero, and a zero 6-vector is not a rotation at all.
IDENTITY_ROTATION_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def neutral_vector() -> np.ndarray:
    """The 63-dim vector used to replace non-finite samples, per channel."""
    neutral = np.zeros(TOTAL_DIM, dtype=np.float64)
    neutral[channel("headpose_rotation_6d").slice] = IDENTITY_ROTATION_6D
    return neutral


def sanitize(features: np.ndarray) -> tuple:
    """Replace non-finite entries with the neutral vector.

    Returns (clean, was_invalid) where was_invalid is (T, n_channels): a channel
    is marked for a frame if ANY of its dimensions was non-finite, since a
    partially-NaN channel is not usable either.
    """
    features = np.array(features, dtype=np.float64, copy=True)
    neutral = neutral_vector()
    bad = ~np.isfinite(features)
    was_invalid = np.zeros((features.shape[0], len(CHANNELS)), dtype=bool)
    for index, spec in enumerate(CHANNELS):
        block = bad[:, spec.slice]
        was_invalid[:, index] = block.any(axis=1)
        if block.any():
            rows = np.flatnonzero(was_invalid[:, index])
            features[np.ix_(rows, np.arange(spec.start, spec.end))] = neutral[spec.slice]
    return features, was_invalid


def build_input(target: np.ndarray, symmetric_blendshape: bool = False) -> np.ndarray:
    """Derive the encoder input from the stored target tensor.

    The only transform that currently differs between the two spaces is the
    optional blendshape map; it is applied here and never to the stored array, so
    the sigmoid head keeps a [0,1] target whatever the input looks like.
    """
    features = np.array(target, dtype=np.float64, copy=True)
    if symmetric_blendshape:
        spec = channel("blendshape")
        low, high = spec.input_normalization["optional_symmetric_map"]["range"]
        features[..., spec.slice] = low + (high - low) * features[..., spec.slice]
    return features


def loss_weight(observed: np.ndarray, quality: np.ndarray, is_gap_filled: np.ndarray,
                is_padding: np.ndarray, gap_fill_weight: float = 0.0,
                use_quality: bool = True) -> np.ndarray:
    """Combine the four signals into a per-frame, per-channel loss weight.

    Policy, in order of severity:
        padding      -> 0, always. It is not data.
        gap filled   -> gap_fill_weight, 0 by default. Supervising a model on
                        motion that was interpolated teaches it the interpolator.
        not observed -> 0. The value present is a neutral fill, not a measurement.
        otherwise    -> the quality proxy, or 1 when use_quality is off.

    Returns (T, n_channels). Channel granularity matters: a gaze dropout must not
    silence the blendshape supervision for the same frame.
    """
    observed = np.asarray(observed, dtype=bool)
    quality = np.asarray(quality, dtype=np.float64)
    is_gap_filled = np.asarray(is_gap_filled, dtype=bool)
    is_padding = np.asarray(is_padding, dtype=bool)

    weight = quality.copy() if use_quality else np.ones_like(quality)
    weight = np.clip(weight, 0.0, 1.0)
    weight[~observed] = 0.0
    if is_gap_filled.ndim == 1:
        is_gap_filled = is_gap_filled[:, None]
    if is_padding.ndim == 1:
        is_padding = is_padding[:, None]
    weight = np.where(np.broadcast_to(is_gap_filled, weight.shape), gap_fill_weight, weight)
    weight = np.where(np.broadcast_to(is_padding, weight.shape), 0.0, weight)
    return weight


def derivative_weights(weight: np.ndarray, segment_id: np.ndarray = None) -> dict:
    """Weights for first- and second-difference supervision.

    A first difference at i uses frames i and i+1, so it is supervised only where
    both are. A second difference at i uses i, i+1 and i+2. Neither may cross a
    segment boundary: the two sides of a gap are not adjacent in time, and
    differencing across one produces a velocity from a jump that never happened.

    Returned arrays are shorter than the input by 1 and 2 along time, matching
    what a difference produces, so a caller cannot accidentally align them wrong.
    """
    weight = np.asarray(weight, dtype=np.float64)
    n_frames = weight.shape[0]
    if n_frames < 2:
        empty = np.zeros((0,) + weight.shape[1:])
        return {"velocity": empty, "acceleration": empty}

    same_segment_1 = np.ones(n_frames - 1, dtype=bool)
    same_segment_2 = np.ones(max(0, n_frames - 2), dtype=bool)
    if segment_id is not None:
        segment_id = np.asarray(segment_id)
        same_segment_1 = segment_id[:-1] == segment_id[1:]
        if n_frames >= 3:
            same_segment_2 = (segment_id[:-2] == segment_id[1:-1]) & (segment_id[1:-1] == segment_id[2:])

    velocity = np.minimum(weight[:-1], weight[1:]) * same_segment_1[:, None]
    if n_frames >= 3:
        acceleration = np.minimum(np.minimum(weight[:-2], weight[1:-1]), weight[2:]) * same_segment_2[:, None]
    else:
        acceleration = np.zeros((0,) + weight.shape[1:])
    return {"velocity": velocity, "acceleration": acceleration}


def channel_loss_normalizer() -> dict:
    """Per-channel divisor so a channel does not dominate by dimension count.

    A plain sum of squared error over 63 dimensions is 52/63 blendshape before
    any weighting: the model would optimise expression and neglect head pose and
    gaze simply because there are more blendshape numbers. Dividing each channel's
    summed error by its dimension makes the channels commensurate, and the
    schema's per-channel loss weights then express actual priority rather than
    fighting the dimension count.
    """
    return {spec.name: float(spec.dim) for spec in CHANNELS}


def describe_contract(symmetric_blendshape: bool = False) -> dict:
    """Machine-readable summary of the contract, for the run manifest."""
    return {
        "stored_space": "target",
        "input_derivation": "build_input(target, symmetric_blendshape=%s)" % symmetric_blendshape,
        "symmetric_blendshape_applied_to_input": symmetric_blendshape,
        "symmetric_blendshape_applied_to_target": False,
        "signals": {
            "observed": "source frame produced a real detection",
            "quality": "[0,1] extractor proxy score, NOT a calibrated probability",
            "is_gap_filled": "sample invented to bridge unobserved time",
            "is_padding": "sample exists only to fill a window to fixed length",
        },
        "weight_policy": "padding -> 0; gap-filled -> gap_fill_weight (default 0); "
                         "unobserved -> 0; else quality",
        "derivative_rule": "velocity needs 2 consecutive valid in-segment frames, "
                           "acceleration needs 3; never across a segment boundary or padding",
        "non_finite_policy": "replaced with a per-channel neutral before the network; "
                             "rotation uses the identity 6D vector, not zeros",
        "channel_normalizer": channel_loss_normalizer(),
    }
