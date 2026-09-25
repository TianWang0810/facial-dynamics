# Decision: audio enters as a time-aligned conditioning sidecar (04_audio), not as tensor channels

## Context

Restoring natural detail after the L1 decode (speech articulation especially)
needs audio. Mouth micro-motion that is not synchronised with the sound looks
worse than none. The pipeline was video-only, and the first validation clip had
no audio stream at all (see `engineering_log/talkvid_audio_download.md`).

## Decision

**1. Audio is conditioning, not target.** It is not added to the 63-dim vector,
and `schemas/geometry.schema.json` is untouched (still 2.0.0). Adding it there
would make the decoder reconstruct audio, change `TOTAL_DIM` for the modelling
team, and put a 100 Hz signal into a 30 Hz tensor. Its parameters live in their
own authority file, `schemas/audio_features.json`, loaded and validated by
`src/audio/features.py` at import. This follows the same pattern as the feature
schema.

**2. A new stage `04_audio`, after Dynamics.** It needs the window times Features
produces, so it runs after Features. Numbering it 04 instead of inserting it
earlier keeps the canonical `00`–`03` paths of every existing run valid
(`pipeline_run_layout.md`). Output: `04_audio/audio.npz`, where row *i* is the
audio for row *i* of `features.npz`, with `clip_id` / `window_start` /
`window_time_sec` repeated so a reader can assert the alignment. Alongside it,
`audio_manifest.json`.

**3. Alignment is by real time only.** Video `pts_sec` and audio frame PTS are
both container-timeline seconds. Features now also writes `window_time_sec`, the
grid time of each window's first frame (an additive field). Audio samples are
placed by their own frame PTS and cut per window over
`[window_time_sec, + window_frames / target_hz)`. Nothing is aligned by position
and no common start is assumed. `test_alignment_is_by_time_not_by_position` pins
this.

**4. Representation: 80-bin log-mel at 100 Hz plus the 16 kHz waveform.** The
mel front end is the standard 25 ms / 10 ms / 80-mel one, computed in numpy, so
env-v0.1 stays frozen. The int16 waveform per window is kept so a pretrained
speech encoder (wav2vec2 / HuBERT, which needs torch and hence an env-v0.2) can
be run later without re-decoding the mp4s. Audio keeps its native 100 Hz rather
than being decimated to the 30 Hz video grid, because that would throw away
exactly the timing detail it is there to provide.

**5. Same "never repair" rules as video.**
- Audio PTS discontinuities are reported and left as holes.
- Uncovered, padded or discontinuous audio frames get `audio_valid = False`.
- A clip without audio is not an error: all its audio is marked invalid.
- A clip whose video and audio *durations* disagree by more than 0.1 s (after
  excluding a tail-gap frame) gets all its audio marked invalid
  (`av_duration_mismatch`). A content shift cannot be seen in PTS, and the offset
  is never guessed. On the 16-clip pilot this safeguard did not trip.

**6. A sync diagnostic, not a gate.** `audio_manifest.json → sync_check`
cross-correlates `jawOpen` with audio log energy over ±0.5 s. On the 16-clip
pilot the pooled peak is at **+110 ms** (audio later than mouth opening;
r = 0.137 vs 0.004 at lag 0), median per-clip +105 ms. With mismatched clip
pairs the correlation drops to |r| ≤ 0.04. A wider ±3 s search produced
spurious peaks at syllable-rhythm periods, which is why the range is limited.
The +110 ms is consistent with the mouth opening before the vowel energy peaks.
It is recorded as a measurement; whether a model should compensate for it is
the modelling team's call.

## Related changes

- **Observation** publishes per-stream timing from headers (`streams`: video and
  audio start, A/V start offset, audio format) and `relative_path` in
  `video_metadata.json`. `quality.json` gets `has_audio`.
- **Observation** reports a gap before the final frame only as
  `tail_frame_gap`. Features keeps such a clip and excludes that one frame
  (`tail_frame_excluded` in the manifest; `--keep_tail_gap_frame` to opt out).
  Every other timeline issue still rejects the clip. This is a cut artefact:
  5 of the first 16 TalkVid downloads had it and nothing else.
