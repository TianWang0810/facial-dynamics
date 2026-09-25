# Decision: one JSON file defines the 63-dim feature layout, code derives from it

## Context

The 63-dimensional per-frame vector is consumed by four places that must agree
exactly: data extraction in the Geometry layer, model input / ground truth
assembly, the decoder's output heads, and the loss. A disagreement about where
the gaze channel starts does not crash anything -- it silently trains a model
against misaligned targets, and the symptom appears much later as "the model
learned head pose badly".

Two teams work on either side of this boundary, in separate repositories.

## Decision

`schemas/geometry.schema.json` is the single authority. It carries, per channel:
slice bounds, dimension, source track, the geometry.parquet column it comes from,
its quality column, normalisation rule, decoder head activation and
post-processing, projection width, and loss weights.

`src/schema/feature_vector.py` **loads** that file at import and derives every
constant from it. No Python file in this repo writes a literal index. The
alternative -- a dataclass in code plus a JSON copy for the other team -- was
rejected: two artefacts describing one contract is precisely the drift this is
meant to prevent, and nothing would have flagged the moment they diverged.

The layout is validated on import: slices must be contiguous, in order,
non-overlapping, matching their declared widths, and summing to `total_dim`. A
malformed edit fails at import rather than producing a subtly wrong tensor.

`assemble()` and `split()` are exact inverses over the same spec, so all four
consumers can round-trip through one definition.

## Input space and target space are declared separately

Each channel carries `input_normalization` and `target_normalization` as distinct
fields, because they are not always the same transform and a single
`normalization` field made that ambiguous.

Blendshape is the case that motivated the split. The optional `[0,1] -> [-1,1]`
map is a convenience for models preferring zero-centred inputs, and it applies to
the encoder input **only**. The reconstruction target stays in `[0,1]`, matching
the sigmoid head's range; had the flag silently rescaled targets too, the loss
would compare a sigmoid output against a `[-1,1]` target and could never reach
zero. The per-channel projection layer absorbs the input-side factor, which is
part of what that layer is for.

Gaze deliberately behaves the other way: its optional tanh compression applies to
input **and** target together. The gaze head is a tanh bounded at the per-axis
limit, so a target outside that range is unreachable by construction and would
leave an irreducible loss floor. Compressing both sides keeps every target
attainable. The asymmetry between the two channels is intentional and is stated
in both `scope` fields.

## Window length is a duration, not a frame count

`target_seconds` is the contract; the frame count is computed as
`round(target_seconds * grid_hz)` against the rate of the timeline actually being
cut. A fixed `default_window_frames` was removed: at 30 it means 1.0s in a 30fps
clip and 1.2s in a 25fps one, which inverts the intent of `target_seconds`.

`scripts/run_features.py` resamples every clip onto one grid before windowing, so
the rate it passes is `--target_hz` and every window in a run covers the same
duration in the same number of frames. That is why it emits a single `features`
tensor rather than per-length buckets: there is only ever one length. Buckets are
what the *other* order of operations would need -- windowing raw clips at their
own PTS-measured rates produces 25- and 30-frame windows that cannot stack, and
those must be grouped per length rather than forced to a common frame count,
which would silently vary the duration. `src/sequence/window.py` supports both
and chooses neither; the caller supplies the rate.

Either way the manifest records `grid.window_frames` beside `grid.target_hz`, so
the duration a tensor represents is read, never inferred from its shape.

`frames_for_seconds()` quantises before rounding and rounds half up. Both matter
in practice: Python's half-to-even would flip a window length on the parity of an
exact half, and a PTS-measured 25fps arrives as 24.999999999999979, putting two
clips of the same real rate on opposite sides of a rounding boundary and into
different buckets for no physical reason.

## The 6D rotation loss is computed before Gram-Schmidt

`loss_contract.rotation_6d_loss_space` states this explicitly, and
`rotation_loss_is_pre_orthonormalization()` exposes it so training code can assert
on it rather than rely on prose.

The loss reads the raw 6-vector the head emits, compared against the raw 6-vector
in the data. This follows Zhou et al. 2019, where the network regresses the 6D
representation directly. Inserting the orthonormalization into the loss path also
changes the gradient: Gram-Schmidt is a normalising projection, so it discards
precisely the component the regression is being scored on. Orthonormalize only
when a rotation matrix is physically needed downstream -- driving an avatar or a
robot.

## Consequences

- The modelling repo reads the same JSON rather than restating the layout. Head
  activations (`sigmoid` / `identity` / `identity` + Gram-Schmidt / `tanh`) and
  loss weights live there, so the decoder can be built from the schema.
- `tests/test_feature_schema.py` restates the index ranges once, deliberately, as
  a tripwire: editing the schema without updating the consumers fails the test.
- Adding a channel means editing the JSON and the assembly path, and bumping
  `schema_version`. Any produced `features.npz` records the version it was built
  with, and the manifest next to it records the full channel table, so an old
  tensor can always be interpreted.
- Velocity and acceleration are deliberately absent from the 63 dimensions: they
  are supervision signals computed by differencing at loss time, not inputs. The
  Dynamics layer (`src/dynamics/derive.py`) remains the reference implementation
  of that differencing.
- Per-dimension decoder scales are stored as arrays, not prose. `decoder_scale()`
  always returns a vector of the channel's width, so the gaze head's two limits
  (35 degrees pitch, 50 degrees yaw) cannot be collapsed into one scalar by a
  reader who skimmed the note.
- `headpose_translation` and `headpose_rotation_6d` share the single
  `conf_headpose` column, so it appears twice in the mask column list. That
  duplication is intentional -- both are halves of one decomposition of one matrix
  -- and carries a `columns_note` saying so, because it otherwise looks like a
  copy-paste error a reviewer would "fix".
