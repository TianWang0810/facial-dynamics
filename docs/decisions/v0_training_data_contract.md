# Decision: what the pipeline guarantees for a Continuous Temporal Autoencoder v0

Scope: making the extracted data reliable enough to train and evaluate a v0
autoencoder. NOT in scope, and deliberately not implemented: VQ, identity
encoders or adversaries, neutral estimation, avatar or robot retargeting.

## Frames and timestamps come from one decode

Timing previously came from a PyAV pass and pixels from a separate OpenCV pass,
related only by array position. Equal lengths do not verify that correspondence:
the two decoders disagree about what counts as a frame. `src/observation/decode.py`
now decodes once and a frame carries its own PTS, so there is nothing to
misalign. Every geometry row records `source_frame_idx` and `pts_sec`.

A frame with no PTS is kept in place with a null timestamp. Dropping it -- the
previous behaviour -- shifted every later index while leaving the array length
plausible, which is the hardest kind of corruption to notice downstream.
Duplicate and backward timestamps are likewise preserved and reported: rewriting
a damaged stream into something monotonic hides the damage.

MediaPipe needs a strictly increasing millisecond clock. That is a tracker
constraint, so `tracker_timestamps_ms()` derives one FROM the real PTS into its
own column; `pts_sec` is never overwritten to satisfy it.

## One sample rate for the model grid

A TCN with no time input assumes even spacing. `src/sequence/resample.py` puts
every clip on a configurable grid (30 Hz default) using real PTS. This is a v0
engineering choice, not a theoretical requirement.

Timelines split into segments at gaps longer than `--max_gap_seconds`, and
nothing is interpolated across one. The limit is in seconds because it describes
how long a face may be unobserved before the motion between the two ends is
unknowable; a frame count would silently change meaning with the source rate.

Rotation is interpolated through the rotation manifold (quaternion nlerp,
sign-aligned), not element-wise on the raw 6-vector -- a linear blend of two
6-vectors leaves the set of valid rotations.

Two provenance flags that must not be merged: `is_resampled` (the sample is on
the grid rather than an original frame time -- normal) and `is_gap_filled` (the
sample bridges unobserved time -- a repair). No anti-alias filter is applied; a
source above 1.5x the target rate warns with the alias band named.

## Input, target and weight are separate things

The stored tensor is the TARGET. `x_input` is derived at load time by
`src/sequence/contract.py:build_input()`; storing both would double the disk and
guarantee divergence. The blendshape `[-1,1]` map applies to the input only, so
the sigmoid head keeps its `[0,1]` target.

Four signals travel separately -- `observed`, `quality`, `is_gap_filled`,
`is_padding` -- and are combined by `loss_weight()`. A single pre-blended mask
cannot be taken apart again.

Non-finite values are replaced by a per-channel neutral before the network
(`identity` for rotation, not zeros, since a zero 6-vector is not a rotation).
Relying on `NaN * 0` is wrong: in IEEE 754 that is `NaN` and poisons the batch.

`derivative_weights()` enforces that a first difference needs two consecutive
valid in-segment frames and a second difference three, never crossing a segment
boundary or padding. `channel_loss_normalizer()` divides each channel's error by
its dimension so blendshape does not dominate by being 52 of 63 numbers.

## Honest units and honest scores

The iris gaze backend emits a DIMENSIONLESS proxy -- normalised iris displacement
times an uncalibrated gain -- not radians. The schema previously declared radians
for both backends. Backends now carry `unit` and `is_calibrated`, and requesting
the radian-scaled tanh compression for the proxy raises rather than dividing a
ratio by a radian limit. Output from the two backends is not the same quantity
and must not be pooled as one training target. Axis order is (pitch, yaw);
positive pitch is looking up, which required negating the raw image-space offset.

Every confidence is a quality PROXY, not a probability, and is named as such. The
head-pose residual is measured after scale removal and before any
orthonormalization, so it is a genuine validity check -- but MediaPipe constructs
that matrix to be rigid, so it sits near zero on most frames and says little
about pose ACCURACY.

Removed: the `blendshape.sum() < 0.05` degeneracy test. A genuinely neutral face
IS an all-near-zero ARKit vector, so the rule zeroed confidence on exactly the
expression the model reproduces best, silently biasing supervision towards
expressive frames.

## QC is explicit and counted

The Features stage applies QC by stated policy and records every rejection with a
reason. Timeline defects are hard errors about the file; the detection-rate
threshold is a configurable visual policy. The two are reported separately so one
can be overridden without the other.

The detection rate comes from MediaPipe's `detected` column -- the same model that
produces the features -- not from a second, independent detector. The Haar cascade
that used to supply it was removed after disagreeing with MediaPipe by 59
percentage points on the first real clip; see
docs/decisions/observation_timeline_only.md.

## Identity, splits and what may be claimed

`clip_id` is the path relative to the input root, not the basename -- two files
named `take1.mp4` in different folders were previously merged into one.

`speaker_id` is populated only when `--speaker_from` says where to read it, and
is null otherwise. `src/sequence/split.py` produces a speaker-disjoint split when
speakers are known and a clip-disjoint one otherwise, LABELLED as such with a
limitation string. Splitting is at clip level so overlapping windows cannot
straddle a boundary. Normalisation statistics are fitted on train only.

## Rotation loss: the earlier claim was too absolute

The previous contract said Gram-Schmidt may never appear in a training loss. That
is wrong as stated. Raw-6D reconstruction is the v0 baseline, following Zhou et
al. 2019; a loss on the orthonormalized rotation is also legitimate. What is
required is that angular error in degrees be reported alongside, because raw-6D
MSE has no physical unit. A finite difference of raw 6D values is a
representation-space smoothness term and must not be called angular velocity. v0
does not require rotation acceleration or jerk terms.

## Stated limitations

- Clip-mean subtraction removes sustained head pose and gaze offsets along with
  camera placement. A persistent head tilt is not recoverable from `headpose_t`;
  `headpose_t_raw` retains it.
- No channel is identity-disentangled. Blendshape amplitude and head translation
  both carry speaker-specific structure, and landmark canonicalization does not
  change that for the other channels.
- There is no neutral-pose ground truth. The clip's first frames and its
  low-motion frames are NOT neutral and must not be treated as such.
