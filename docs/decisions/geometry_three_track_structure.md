# Decision: Geometry layer as three parallel tracks with a shared quality mask

## Context

The Geometry layer originally ran as a single function that pulled landmarks,
blendshapes and a head-pose matrix out of one MediaPipe call and wrote them to
parquet largely as they arrived. Head pose was stored as a raw 4x4 matrix, gaze
was not extracted at all, and quality was summarised as one label per clip.

That shape does not survive contact with what the Dynamics layer does to the
signal. Differentiating three times punishes any representation with a
discontinuity, any signal that mixes two physical causes, and any quality flag
too coarse to mask.

## Decision

Frames fan out to three tracks, driven from one shared decode loop
(`src/geometry/pipeline.py`), and merge into a per-frame quality mask.

### Track A -- landmarks + blendshapes, jointly (`src/geometry/landmarks.py`)

MediaPipe Face Landmarker emits 478 3D landmarks and 52 ARKit-named blendshape
coefficients from a single inference. The two are computed jointly inside the
model, so we do not build a "fit landmarks, then solve for blendshapes" two-step
of our own: it would add a second error surface to reproduce an output we
already have.

### Track B -- head pose, 6D rotation + normalised translation (`src/geometry/headpose.py`)

Nine dimensions: six for rotation, three for translation.

Rotation uses the 6D continuous representation (Zhou et al. 2019) -- the first
two columns of R, with the third recovered by Gram-Schmidt. Euler angles are
rejected for gimbal lock and for the wrap-around at +/-180 degrees; unit
quaternions are rejected for double cover, where q and -q denote the same
rotation so an arbitrary sign flip between adjacent frames looks like a violent
reorientation. Both failure modes are invisible in a static frame and disastrous
under differentiation: `tests/test_geometry_tracks.py` pins this down by showing
an Euler jump of ~2*pi at a wrap point where the 6D form moves by <1e-2.

Translation subtracts a per-clip reference (mean head position over tracked
frames), so the signal stops encoding how far the subject sat from the lens. The
raw translation is kept as `headpose_t_raw`, leaving the normalisation
invertible.

MediaPipe's matrix is a similarity transform, so the scale is divided out before
the block is read as a rotation; how far the rescaled block then sits from
orthonormal is track B's confidence signal.

### Track C -- gaze, independent of track A (`src/geometry/gaze.py`)

Two dimensions (pitch, yaw, radians), zero-meaned per clip, with optional tanh
compression into the physiological range. Gaze is explicitly NOT taken from the
`eyeLookUp/Down/In/Out` blendshape coefficients: those are a by-product of an
expression model, heavily discretised, and not accurate enough for a signal that
gets differentiated three times.

The estimator sits behind a `GazeBackend` interface with two implementations:
`ExternalModelBackend` (onnxruntime, the intended production path, needs
env-v0.2) and `IrisGeometryBackend` (default). The fallback measures where the
iris physically sits inside the eye opening using the iris landmarks 468-477 --
geometry, not the rejected blendshape route -- but its offset-to-angle mapping is
an uncalibrated linear gain, so it is a relative signal, not absolute gaze.

### Shared quality mask (`src/geometry/confidence.py`)

Every track emits its own confidence, and all four land in `geometry.parquet` as
per-frame columns: `conf_landmarks`, `conf_blendshapes`, `conf_headpose`,
`conf_gaze`.

Per frame and per channel, because a clip-level label hides that head pose can be
solid while gaze is unusable across a stretch of blinks; a consumer masks one
channel without discarding the rest.

Mean and variance, not success/failure, because the binary flag misses the frames
that tracked but tracked badly -- partially out of frame, tiny face,
near-degenerate pose fit. Those are precisely the frames that contaminate
training quietly, since nothing marks them as suspect. The report carries mean,
std, var, min, max and percentiles per channel plus an explicit
`n_low_but_tracked` count.

Which confidences are measured and which are proxies is recorded per channel in
`geometry_metadata.json`, because the two carry very different weight: track B's
is computed from the model's own output, while track A's is derived from framing
(how many landmarks fall inside the image, how large the face is) since MediaPipe
Face Landmarker exposes no per-landmark score.

## Consequences

- `src/geometry/extract.py` is replaced by `landmarks.py` + `headpose.py` +
  `gaze.py` + `pipeline.py`; the video is still decoded once per clip.
- `geometry.parquet` gains 9 head-pose, 2 gaze and 4 confidence columns. Existing
  `landmarks` / `blendshapes` / `eye_distance` columns are unchanged, so a reader
  of the old schema keeps working.
- Running gaze on a dedicated model requires opening env-v0.2. Until then the
  default backend keeps the pipeline inside frozen env-v0.1 and labels its own
  output as a fallback in the metadata.
