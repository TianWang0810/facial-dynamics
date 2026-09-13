# Engineering Log: video_metadata.json does not retain per-frame PTS

Format: Symptom -> Diagnosis -> Root Cause -> Impact -> Fix -> Open Question ->
Generalizable Takeaway.

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

Impact: Currently latent, not active. Every clip in the validation sample
happens to be constant-frame-rate, so min_frame_gap == max_frame_gap and
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

Fix (a workaround, not a repair): Added read_stream_timing() and extract_pts()
to src/observation/pts.py, exposing the per-frame PTS array that pts.py already
computed. analyze_video() was refactored to call read_stream_timing() so its
behaviour is unchanged. scripts/run_dynamics.py consequently requires a third
argument, --clips_dir, and re-decodes the source mp4 to recover real timestamps;
video_metadata.json is still used, for clip_id -> filename resolution and a
frame-count cross-check.

The claim that the refactor is behaviour-preserving is backed by
tests/test_pts_refactor_equivalence.py, which loads the pre-refactor
analyze_video() out of git at the pinned parent commit, runs it and the current
version over the same clips, and asserts exact equality on all 12 returned
fields. Comparing only video_metadata.json would have been insufficient: that
file carries 6 of the 12 fields, so is_monotonic, max_frame_gap, min_frame_gap,
issues and quality_label would have gone unchecked. Result at the time of
writing: 5 clips x 12 fields = 60 exact-equality comparisons, all identical.

Open Question: This is a workaround. Re-decoding the mp4 costs a full extra
decode pass per clip and, more importantly, breaks the property that
Observation's published output is self-contained -- Dynamics now depends on the
raw clips remaining available, which conflicts with the raw/derived separation
in design doc section 5.3. Making video_metadata.json self-contained means
adding a per-frame PTS array to it, which is a schema change to a published
Observation artifact and therefore triggers observation-v0.2 under the
"versions are append-only, never edited" rule (section 6.4). Deferred
deliberately: it should be batched with any other Observation schema changes
rather than spending a version bump on this alone.

Generalizable Takeaway: A layer that computes a rich intermediate and publishes
only a summary silently constrains every future consumer downstream of it. The
cost is invisible until a consumer appears, and then surfaces as an
architectural workaround rather than a local fix. When a layer discards
something it already computed, that is worth recording as a deliberate decision
with its reasoning -- and when the sample data cannot distinguish a correct
implementation from an incorrect one (here, constant-frame-rate clips hiding
the whole problem), the validating test has to be synthetic and adversarial on
purpose, because real-data validation will pass either way.
