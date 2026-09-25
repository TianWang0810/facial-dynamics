# Engineering Log: TalkVid download with audio (2026-09-23, AICR HPC)

Format: Symptom -> Diagnosis -> Root Cause -> Fix -> Generalizable Takeaway.

## Issue 1: the validation clip had no audio stream

Symptom: Adding an audio stage, the only validation clip
(`data/talkvid_bench/age/…`) had one stream, h264 video.

Diagnosis: Its format was YouTube itag 137, a video-only DASH format.

Root Cause: The Issue 5 fix in `local_validation_day1.md` constrained the
format selection to `[protocol^=https]` to avoid stalling HLS streams, applied
as a single selector. The best https format at 1080p is video-only, so audio
was silently dropped.

Fix: `scripts/fetch_talkvid.py` selects video and audio separately, each
restricted to https, and merges them:
`bestvideo[ext=mp4][protocol^=https][height<=1080]+bestaudio[ext=m4a][protocol^=https]`.
All 207 clips downloaded with it carry AAC 44.1 kHz stereo.

Takeaway: A constraint added to fix one symptom can silently remove a property
nobody was checking yet. Assert the properties you need (here "has an audio
stream") at acquisition time, not when the first consumer appears.

## Issue 2: 5 of 16 clips rejected for `large_frame_gap`

Symptom: The pilot kept 11/16 clips.

Diagnosis: In every rejected clip the only large gap was after frame n-2. The
final frame sits 2–3 steps after its predecessor.

Root Cause: A cut made without re-encoding (`--download-sections`) ends with a
stretched final frame. This is an artefact of the cut, not of the source.

Fix: Observation reports it under its own code `tail_frame_gap`. Features keeps
the clip and excludes that one frame (recorded as `tail_frame_excluded`).
Interior gaps still reject. Pilot: 16/16 kept.

Takeaway: A "reject on any issue" policy is only as good as its issue
taxonomy. One code covering both "one cut artefact at the end" and "a hole in
the middle" forces a choice between losing 30% of the data and accepting real
defects.

## Issue 3: Dynamics skipped clips whose id contains a dot

Symptom: `run_dynamics.py` printed "no entry for <clip> in video_metadata.json"
for clips named `…__614.881_634.467`.

Root Cause: The metadata was keyed by `os.path.splitext(m["file"])[0]`, but
`file` is already the extension-less clip id. `splitext` was a no-op for the
earlier ids and truncated `.467` from these.

Fix: Key by `m["file"]` directly.

Takeaway: `splitext` on something that is not a filename is a latent bug that
waits for the first identifier with a dot in it.

## Issue 4: some clips saved under a URL-shaped nested path

Symptom: Files appeared under `<person>/https:/youtu.be/<id>?si=…__start_end.mp4`.

Root Cause: The video id was taken as `link.split("v=")[-1]`. TalkVid also has
`https://youtu.be/<id>?si=…` links, which have no `v=`, so the whole URL
(including slashes) became the filename.

Fix: Parse the 11-character id with a regex covering `watch?v=`, `youtu.be/`
and `shorts/`. The existing files were migrated
(`/scratch/$USER/talkvid/migrate_names.py`).

## Issue 5: YouTube bot check after ~200 downloads

Symptom: After about 200 successful downloads in ~5 minutes, 4 jobs × 6
workers, every request failed with "Sign in to confirm you're not a bot"
(509 of 765 attempts). Shards 4–7 got nothing.

Diagnosis: All compute nodes egress through one address, so parallelism across
nodes is parallelism from one IP.

Fix: The job array was cancelled at once. The download was not resumed with any
technique for evading the check. Options (throttled resume, the user's own
cookies, another source) are a decision for the project owner.

Takeaway: Size the concurrency of scraping jobs to the egress IP, not to the
number of nodes. The measured budget here is on the order of 200 requests in
5 minutes.

## Issue 6 (my own mistake): a false "1 s A/V misalignment"

Symptom: A duration probe reported one clip's video 0.99 s longer than its
audio. A ±3 s lip/audio cross-correlation seemed to confirm it (peak at
−1.07 s). I briefly recorded this as a measured misalignment in the audio spec.

Diagnosis: The probe computed duration as `last_pts − first_pts + 1/average_rate`.
The container's declared average rate for that clip was wrong, adding 1 s. From
the real PTS it is 8.33 s video vs 8.32 s audio. The ±3 s correlation peak was
spurious: limited to ±0.5 s, the clip peaks at +120 ms like the others.

Fix: The claim was removed from `schemas/audio_features.json` and
`src/audio/features.py`. The duration gate stays as a safeguard. The stage
computes duration from real PTS, never from a declared rate.

Takeaway: The project's rule "time from real PTS, never declared fps" applies to
throwaway diagnostic scripts too. Two independent-looking pieces of evidence
were both artefacts of the analysis: a declared rate, and a lag window wide
enough to find a periodic false peak.
