# Engineering Log: video_metadata.json does not retain per-frame PTS

Format: Symptom -> Diagnosis -> Root Cause -> Impact -> Interim Workaround ->
Resolution -> Future Work -> Generalizable Takeaway.

Status: RESOLVED. video_metadata.json now publishes the complete per-frame PTS
array. A workaround (re-decoding the mp4 behind a --clips_dir argument) was
committed first and then replaced; it is kept in this record because the reason
it was discarded is the useful part.

Recorded while implementing the Dynamics layer (src/dynamics/derive.py).

## Issue: the Dynamics layer cannot get real per-frame timestamps from Observation output

Symptom: The Dynamics layer differentiates a Geometry sequence with respect to
time, and design doc section 1 requires that time come from real PTS rather than
frame_idx / fps. Its intended inputs were geometry.parquet plus the matching
video_metadata.json. But video_metadata.json carries only aggregate PTS
statistics -- first_pts_sec, last_pts_sec, duration_from_pts_sec,
n_frames_decoded -- and no per-frame list, so there is nothing to differentiate
against.

Diagnosis: src/observation/pts.py does decode and compute the full per-frame PTS
array (pts_list_sec). It uses that array to derive monotonicity, max/min frame
gap and the quality label, then returns only the summary; the array itself goes
out of scope when the function returns. src/observation/merge.py, which builds
video_metadata.json, therefore never had access to it in the first place.

Root Cause: merge.py persists only the summary portion of what pts.py computes.
This was a reasonable Observation-layer decision in isolation -- the per-frame
array is roughly 1600 floats per clip and Observation itself only needs the
aggregates -- but it was made without a downstream consumer that needed
per-frame timing. Dynamics is the first such consumer.

Impact (as assessed at the time of discovery): Latent, not active. Every clip in
the validation sample happens to be constant-frame-rate, so
min_frame_gap == max_frame_gap and
reconstructing timestamps as frame_idx / fps would coincidentally give the right
answer. The defect only bites on a genuine variable-frame-rate or re-encoded
clip, where video_metadata.json alone is insufficient to compute Dynamics
correctly and a constant-fps reconstruction is silently wrong rather than
loudly broken.

How wrong is quantified in tests/test_dynamics_derive.py
(test_dropped_frame_vs_constant_fps). Six frames of motion at a constant
2.0 units/s, with one dropped frame widening the interval from 0.04s to 0.12s:

    PTS-based    : [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]   correct
    constant-fps : [2.0, 2.0, 4.0, 4.0, 2.0, 2.0]   100% error across the gap

A dropped frame does not produce a small numerical error; it fabricates a
doubling of facial velocity that never happened. Keep this case in the test
suite -- it is the most direct evidence for the section 1 rule.

Interim Workaround (commit decf995, since replaced): read_stream_timing() and
extract_pts() were added to src/observation/pts.py to expose the per-frame array
pts.py already computed, and scripts/run_dynamics.py took a third argument,
--clips_dir, re-decoding each source mp4 to recover real timestamps.

This worked but was wrong in two ways. It cost a full extra decode pass per clip,
and more importantly it broke the property that Observation's published output is
self-contained: the Dynamics layer now depended on the raw clips staying
available, which contradicts the raw/derived separation in design doc section
5.3. It also left design doc section 1.1 -- which lists per-frame PTS as an
Observation deliverable -- unsatisfied while appearing to be addressed.

Resolution: video_metadata.json now carries the full per-frame array as a
pts_sec field. analyze_video() returns pts_sec, merge.py publishes it, and
run_dynamics.py reads it directly; --clips_dir and the now-dead extract_pts()
were both removed. No mp4 is re-decoded anywhere in the Dynamics path.

The decision was to fix this immediately rather than defer it to observation-v0.2.
Three reasons, in order of weight:

  1. Per-frame PTS is a design doc section 1.1 deliverable. This was never a new
     feature request, it was an unimplemented part of the Observation layer's
     specified output. Deferring it would have meant versioning around a known
     gap in the spec.
  2. The "versions are append-only, never edited" rule (section 6.4) protects
     *published* dataset versions. No such version exists yet: Observation output
     still lives in ~/facial-dynamics/data/, and the versioned
     datasets/observation/v0.1/ layout of section 5.3 is still [TARGET]. There is
     no observation-v0.1 artifact whose immutability this could violate.
  3. The change cost is currently near zero -- 4 validated clips, one user, no
     downstream consumer besides the Dynamics layer being written alongside it.
     That cost only ever grows.

Verified by an A/B against the workaround: the pre-change run_dynamics.py was run
from a git worktree at decf995 with --clips_dir, the current version from the
working tree, over the same four clips. The two JSON reports are byte-identical
(matching md5). Separately, the stored pts_sec was compared value-by-value with a
fresh decode of each mp4 under exact float equality -- 1625 values, all identical
-- confirming that publishing PTS through JSON is lossless.

tests/test_observation_schema.py (formerly test_pts_refactor_equivalence.py)
guards the result. It loads analyze_video() out of git at the pinned baseline and
asserts that every one of the 12 baseline fields is still accounted for -- 5 under
the same name, 6 through a recorded rename, and `issues` under the same name with
a changed representation -- and that their values are unchanged. Every field
appearing since the baseline is whitelisted by name with the decision behind it,
so an unwhitelisted addition or a silent disappearance still fails. It further
asserts that pts_sec is internally consistent with the aggregates derived from it
and never leaks into quality.json, which must stay a lightweight summary.
Checking video_metadata.json alone would have been insufficient for the first
claim: that file carries 6 of the 12 fields, so is_monotonic, max_frame_gap,
min_frame_gap, issues and quality_label would have gone unchecked.

A later note on that guard. When the renames landed, only the value comparison
was taught the rename map; the key-set assertion above it still compared the raw
baseline keys against the same-name list alone, so every rename read as both a
disappearance and an unexplained addition and the guard raised on any clip. It
was fixed by matching renames on both sides -- a rename is one decision, not a
removal plus an addition. The underlying claim was never in doubt once checked by
hand: all 11 carried-over fields compare equal. The lesson is narrower and worth
keeping: a guard that is not in whatever the routine "run the tests" command
happens to be will rot without anyone noticing. This one takes an --input_dir and
so sits outside the no-argument test scripts, which is exactly why it went stale.

Future Work -- migrate per-frame PTS into geometry.parquet at scale:

JSON is the right container for 4 clips and the wrong one for 1244 hours. Storing
pts_sec costs ~20 bytes per frame as indented JSON text; measured on the current
sample, video_metadata.json grew from 1,316 to 34,196 bytes (26x) for 1625
frames. Extrapolated to the full TalkVid corpus at 25fps that is roughly 2.3 GB,
and the real problem is not disk but access: JSON must be parsed in its entirety
to read a single clip's metadata, so per-clip reads degrade linearly with corpus
size.

  Trigger: when parsing video_metadata.json becomes a measurable cost in the
  processing pipeline -- concretely, when the file exceeds the low hundreds of MB,
  or when per-clip metadata reads start showing up in profiling. Not before; do
  not pre-optimise this for the current sample size.

  Plan: move the per-frame array into geometry.parquet, which already has exactly
  one row per frame, as a pts_sec column. Binary float64 storage costs ~8 bytes
  per frame instead of ~20, columnar layout allows reading timestamps without
  touching the landmark/blendshape columns, and row-group filtering allows lazy
  per-clip access instead of whole-file parsing. video_metadata.json then reverts
  to aggregates only.

  Note: at that point the per-frame timing has effectively become the timeline.json
  deliverable already assigned to member C in design doc section 4, so the two
  should be reconciled rather than implemented twice.

  This is a storage-layout migration, not a correctness change. The rule it must
  preserve is the one this whole entry is about: real PTS, never frame_idx / fps.

Generalizable Takeaway: A layer that computes a rich intermediate and publishes
only a summary silently constrains every future consumer downstream of it. The
cost is invisible until a consumer appears, and then surfaces as an
architectural workaround rather than a local fix. When a layer discards
something it already computed, that is worth recording as a deliberate decision
with its reasoning -- and when the sample data cannot distinguish a correct
implementation from an incorrect one (here, constant-frame-rate clips hiding
the whole problem), the validating test has to be synthetic and adversarial on
purpose, because real-data validation will pass either way.

A second takeaway from how this was resolved: a workaround that satisfies the
immediate consumer can still leave the specification unmet, and it is worth
re-asking whether the "proper" fix is actually expensive before deferring it on
principle. Here the version-bump rule was invoked against a dataset version that
did not exist yet, and the real cost of fixing it properly was one re-run over
four clips. Deferral rules are meant to stop churn in published artifacts, not to
protect unimplemented parts of the spec.
