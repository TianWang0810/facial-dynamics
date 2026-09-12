# Engineering Log: Local Validation on macOS

Format: Symptom -> Diagnosis -> Root Cause -> Fix -> Generalizable Takeaway.

## Issue 1: TalkVid's top-level requirements.txt is unusable for this task

Symptom: The repo's requirements.txt lists deepspeed, bitsandbytes,
onnxruntime-gpu, torch+CUDA, and other heavy training-only dependencies.

Diagnosis: These packages belong to TalkVid's own model-training code
(under src/), unrelated to the download-and-quality-check task at hand.

Fix: Ignored that file; installed a hand-picked minimal dependency set
based on what download_clips.py actually imports plus what our own
validation scripts need.

Takeaway: Do not assume a repo's root requirements.txt applies to every
subtask within it, especially when the repo bundles both a training
pipeline and standalone utility scripts.

## Issue 2: PyAV and mediapipe conflict on macOS (duplicate libavdevice)

Symptom: import av, mediapipe together triggers an objc runtime warning
about duplicate class definitions from two different bundled libavdevice
versions.

Diagnosis: Both libraries vendor their own copy of ffmpeg-related dylibs
at different versions, loaded into the same process.

Fix: Never import both libraries in the same Python process; PTS
extraction and face detection run as separate script invocations.

Takeaway: When two libraries each vendor a conflicting native dependency,
process-level isolation is often cheaper than resolving the version
conflict directly.

## Issue 3: Hugging Face repo branch information was outdated

Symptom: git clone -b bench failed with "Remote branch bench not found".

Diagnosis: Used git ls-remote --heads directly against the repository
instead of trusting a previously found (stale) reference to a bench
branch; only main existed.

Takeaway: Prefer verifying directly against the live source over relying
on secondhand information that may be outdated.

## Issue 4: Naive sampling of the first N records hit the same video

Symptom: The first download attempt with --limit 5 failed 100% of the
time within one second.

Diagnosis: All 5 failed segments referenced the same video ID -- records
in the metadata JSON are grouped by source video and not shuffled.

Fix: Deduplicated by Video Link when constructing the sample input,
picking one segment per distinct video.

Takeaway: Understand how a dataset is internally ordered/grouped before
doing a naive "take the first N" sample; naive slicing can silently
introduce bias.

## Issue 5 (most significant): Download hangs indefinitely (40+ minutes)

Symptom: Download progress stalled at 80% for over 40 minutes with no
error.

Diagnosis: Checked CPU time vs. wall-clock time for the stuck ffmpeg
process (ps aux) -- only ~3 seconds of CPU time had been consumed despite
40+ minutes of wall time, indicating the process was blocked on network
I/O, not computation. Inspecting the process's full command line revealed
it was streaming from a googlevideo.com m3u8 (HLS) playlist URL via a
VisionOS client identity.

First attempted fix: switched to the android client
(--extractor-args "youtube:player_client=android"), which stopped the
hang -- but also silently degraded video quality to 360p, because that
client identity only exposed a single low-resolution format (a YouTube
"SABR-only" server-side experiment affecting some client types).

Deeper investigation: compared --list-formats output across default(web)/
ios/tv/android client identities. The default (web) client exposed the
full format list including a 1080p, https-protocol, progressive-download
format (format id 137). The actual root cause was that yt-dlp's default
"best" format-selection string preferred an HLS (m3u8) stream at the same
resolution over the https direct-download stream -- HLS streams are
chunk-based and prone to indefinite stalls under imperfect network
conditions, independent of resolution.

Fix: constrained the format-selection string to explicitly require
[protocol^=https], excluding all m3u8/HLS formats. This resolved both the
hang and the quality degradation with a single change, and made the
earlier client-switching workaround unnecessary.

Takeaway: A single root cause (protocol choice) manifested as two
seemingly separate symptoms (hang, quality loss). Comparing CPU time to
wall-clock time is a general technique for distinguishing I/O/network
blocking from genuine compute cost. Do not accept a fix that removes the
symptom without also explaining why the symptom occurred; two changes
made independently (client switch, then protocol constraint) turned out
to be redundant once the true root cause was found.

## Issue 6: ~30% of sampled video links are dead

Symptom: 2 of 6 sampled videos returned "Video unavailable".

Diagnosis: Verified independently with yt-dlp --simulate against each
URL directly; confirmed as genuine upstream unavailability (link rot),
not a script or network bug.

Takeaway: For any dataset built from external platform URLs, expect and
plan for a non-trivial natural attrition rate; large-scale runs should
over-request relative to the desired final sample count, and failure
logs should be retained for later per-subgroup attrition analysis.

## Issue 7: Dataset-provided quality scores do not substitute for
Observation-layer quality control

Symptom / Diagnosis: TalkVid's metadata includes dover_scores,
cotracker_ratio, and head_detail as clip-level scalar quality signals, but
provides no per-frame PTS, no per-frame occlusion/blur/lighting signal,
and the declared resolution/fps reflects the source video, not the
downloaded/re-encoded clip.

Fix: Treated TalkVid's contribution as limited to "which raw videos are
worth downloading" (URL + timestamps + a coarse prior on quality); all
Observation-layer deliverables (real per-frame PTS, PTS completeness,
frame-level face-detection quality, clip-level usable/low-quality/reject
label) are independently computed from the downloaded video files.

Takeaway: A dataset's own curation metrics answer a different question
(what to include in the dataset) than a downstream pipeline's own quality
control (what to trust for training this particular model); the two
should not be conflated.
