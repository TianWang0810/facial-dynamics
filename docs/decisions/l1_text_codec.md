# Decision: the L1 atomic text codec — rules in one JSON, deterministic, measured

Implements L1 of `docs/plans/multilayer_semantic_representation.md` (section 4).
This record holds the trade-offs made while building it; the plan keeps only the intent.

## Context

L1 has to be a rule-driven, deterministic, near-lossless mapping between one window of
the TARGET tensor (`features.npz`, 30 frames × 63) and human-readable, editable text.
It is the error baseline that L2 and L3 are measured against, so each step's loss has
to be attributable rather than lumped together.

## Decisions

**1. Rules live in `schemas/l1_rules.json`; `src/semantic/l1.py` derives everything from
it.** This follows the pattern of `feature_vector_single_source_of_truth.md`. The JSON is
validated on import: edges strictly increasing, one representative per level, each
representative inside its own bin, and a text mapping for every enabled schema channel.
Adding a schema channel therefore makes L1 fail at import rather than silently skip
the channel.

**2. Four levels (none/slight/moderate/strong) with absolute edges, shared across all
blendshapes.** "slight" means the same activation for `browInnerUp` and `jawOpen`.
Per-channel scaling fitted to data was rejected, because it would make L1 a learned
codec and break the "no training" premise. Signed channels (translation, rotation,
gaze) carry the sign in the label (`+slight`, `-moderate`). Gaze thresholds are keyed
by the backend unit, because the iris proxy and radians are different quantities and
must never share a quantizer (schema `unit_contract`).

**3. Decode: the level is authoritative, the hint refines within it, then smooth.**
(Revised 2026-09-23. The first version decoded every segment to its level's fixed
representative, as a flat step. That gave a blendshape max error of 0.53 and velocity
spikes at every boundary, and was not acceptable.)

The default decode (`decode.default_source = "refined"`, `decode.smoothing`) works in
two steps:
- *refined*: a segment decodes to its `value_hint` (the segment mean) when the hint
  lies inside the level's bin, otherwise to the level's representative. Encode clips
  every hint into its segment level's bin (segmentation step 4), so unedited text
  always keeps its measured intensity. When a person edits the level, the old hint
  now sits in another bin, so the segment decodes to the new level. The level stays
  what a person edits. Blendshape `none`'s representative is exactly 0.0, because
  "no action" should reconstruct as zero activation.
- *smoothing*: a Whittaker smoother (λ = 2) per text channel over each covered
  stretch. It never crosses an undescribed frame. Rotation is smoothed on its
  rotation-vector components about the window reference (tangent space), then
  mapped back through the exponential map.

The text format did not change; this is pure post-processing. `source = "level"`
without smoothing is kept as the pure-quantization view, which the report uses to
prove run-length segmentation is lossless.

Alternatives measured on the validation clip, same text:

| decode | bs MAE | bs max | rot mean/max | bs velocity MAE |
|---|---|---|---|---|
| level, step (old) | 0.0229 | 0.528 | 1.36 / 4.43 | 0.195 |
| hint, step | 0.0097 | 0.366 | 0.94 / 4.94 | 0.143 |
| hint, linear between segment centres | 0.0116 | 0.332 | 1.17 / 5.49 | 0.143 |
| smoothing, but hint not clipped into its bin | 0.0095 | 0.339 | 0.90 / 2.33 | 0.128 |
| **refined + clipped hint + Whittaker (default)** | **0.0091** | **0.183** | **0.90 / 2.33** | **0.126** |

The result is insensitive to λ between 0.5 and 8 (MAE 0.0095–0.0097, velocity
0.128–0.132).

For the hint I first tried relabelling a merged segment to the level of its mean
instead of clipping. It was equally accurate, but it turned blink peaks (mean 0.48,
dragged down by the absorbed shoulders) into `moderate`, so it was rejected.

**4. Segmentation = run-length, then a bounded min-duration merge.** Step 1 cuts
maximal runs of identical level, which is exactly lossless with respect to per-frame
levels. The report verifies this on real data (`run_length_segmentation_equals_per_frame`).

Step 2 merges runs shorter than `min_segment_seconds` (0.1 s = 3 frames). A run may
only be merged into a neighbour that is at most one level away from *every* frame of
the run, on every axis. The first version had no such bound. On the validation clip
it absorbed a 2-frame blink onset (value 0.08) into the adjacent `strong` peak, a
0.67 error. With the bound, flicker across a bin edge is absorbed, a short excursion
of two or more levels is kept as an event, and segmentation can never move a frame by
more than one level. That guarantee is pinned by a property test. The merge order is
fully determined (shortest first, then earliest; smaller level distance, then longer
neighbour, then left).

**5. Rotation: rotation vector relative to a per-window reference.** Rotation is
never quantized element-wise on the 6-vector and never expressed as Euler angles. The
path is 6D → Gram-Schmidt matrix → sign-aligned quaternion → rotation vector in
degrees (log map). Decode goes back through the exponential map, and the three axes
are quantized and segmented jointly as one level tuple. Segment means and the
reference use the sign-aligned quaternion mean (the multi-sample form of the nlerp in
`resample.py`).

The reference is the window's manifold mean, stored in the window header at 0.1°. It
exists because translation and gaze are baseline-relative in the target tensor and
rotation is not. On the validation clip rotation sits at a constant ~16° about x, and
with an absolute (identity) reference the levels spent their whole range on that static
posture: mean round-trip error 5.29°. With the window reference it is 1.31°.
`reference.method = "identity"` restores the absolute description. The axes are named
only `x/y/z`: nod/turn/tilt naming waits until the sign convention has been checked on
video.

**6. Masks are not text.** A frame appears in a channel's segments iff
`contract.loss_weight() > 0` for that channel, reusing the existing rule rather than
inventing one. Padding, gap-filled and unobserved stretches are header intervals, and
decode restores all four signals from them. A decoded channel's quality is 1.0 where
its text covers the frame and 0.0 elsewhere, so `loss_weight()` on the decoded arrays
reproduces the original describable mask exactly (checked). The extractor's quality
*value* is not a facial attribute and is not carried.

**6b. Decode is strict, because the text is hand-edited.** A missing or unknown
text channel, overlapping segments, an off-grid time, a missing rotation reference, or
an ambiguous label (a signed level without its sign) all raise an error. They are
never repaired. Without this, a misspelled `jawOpen` would silently zero the quality
of all 51 blendshapes. `quantize()` refuses non-finite input and clips unsigned
channels at 0.

**7. `_neutral` is excluded by name** (`channels.blendshape.exclude`) and decodes to 0.0.
This leaves 51 + 3 + 1 (rotation) + 2 = 57 text channels covering the 62 live
dimensions.

**8. L1 sits outside the four-stage pipeline.** `scripts/l1_encode.py`,
`l1_decode.py` and `l1_report.py` read a run through `RunLayout`, but no stage or
artifact was added to `src/pipeline/layout.py`, so the pipeline contract is unchanged.
Outputs go wherever `--output`/`--output_dir` says (by convention `<run_dir>/l1/`).

**9. Prerequisite fixed: blendshape names are persisted.**
`build_geometry_metadata()` now requires `blendshape_names` and writes them under
`tracks.A_landmarks_blendshapes.blendshape_names`, and `run_geometry.py` checks that
every clip reports the same order. If no clip produced a single detection, the field
is `null` and the stage still completes (Features rejects those clips by detection
rate). This is additive. Runs older than
`talkvid-age__20260923T182430Z__24ffdd5` lack the field, and `l1_encode.py` refuses them
with a re-run message rather than guessing the order.

## Measured (validation clip, run `talkvid-age__20260923T182430Z__24ffdd5`, 10 windows)

Described frames only; rules v1.0.0; full numbers in `<run_dir>/l1/l1_report.json`.

| config | bs MAE | bs max | bs bias | trans MAE | rot mean / max (deg) | gaze MAE | bs \|v\| recon / orig |
|---|---|---|---|---|---|---|---|
| A per-frame levels | 0.0222 | 0.249 | −0.0005 | 0.070 | 1.31 / 4.43 | 0.0057 | 0.178 / 0.158 |
| B segmented, level step | 0.0229 | 0.528 | −0.0011 | 0.092 | 1.36 / 4.43 | 0.0061 | 0.114 / 0.158 |
| **D segmented, default decode** | **0.0091** | **0.183** | **+0.0001** | **0.044** | **0.90 / 2.33** | **0.0032** | 0.067 / 0.158 |
| floor: Whittaker on the *original* | 0.0025 | 0.124 | — | 0.018 | 0.20 / 1.00 | 0.0013 | velocity MAE 0.070 |

- Segmentation cost (B − A): blendshape MAE +0.0008, bias −0.0006, rotation +0.05°.
  Segmentation introduces no systematic bias.
- The default decode beats even per-frame levels (A): the hint carries the
  within-level intensity that levels alone discard.
- Text size: 1.66 segments per channel per window (2.33 without the merge).
- Blink spot check (window @60): `eyeBlinkLeft` = slight → strong (0.60–0.83 s) →
  moderate, i.e. a rise then recovery. (A blink *raises* `eyeBlink*`, so the plan's
  "骤降-恢复" is a rise-then-recovery in these units.)
- Finer text was also measured (8 levels, no merge): MAE 0.0054, max 0.129, rotation
  0.29° / 1.25°, velocity MAE 0.098, at 2.3× the text (3.78 segments/channel/window).
  Not adopted: it changes the vocabulary people read. It remains the option if this
  error level is still too high.

## Known limitations

- **Velocity is not fully recovered, and cannot be from this text.** Default-decode
  blendshape velocity MAE is 0.126 against a mean original speed of 0.158. But
  smoothing the *original* itself already gives 0.070: about half of the raw 30 Hz
  velocity is frame-to-frame tracker jitter, which no compact description reproduces.
  Between 0.126 and that floor is real motion inside segments. Only more text (finer
  levels or shorter segments, see above) recovers it; post-processing cannot. Use
  L1-decoded tensors for velocity supervision only with that in mind.
- The one-level merge guarantee is in level space. With the default decode the worst
  frame is 0.183 off (segment-edge frames of fast transitions); with the level step
  it was 0.528.
- One clip, one speaker, 10 windows. The edges were chosen for meaning, not tuned to
  this clip, but nothing here is a population statistic.
- `channel_loss_normalizer()` still divides blendshape by 52, including `_neutral`.
  L1 does not depend on it and it was deliberately left unchanged: it is part of the
  training contract with the modelling team and needs their agreement.
