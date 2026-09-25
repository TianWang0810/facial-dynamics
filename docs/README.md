# docs/ Directory Index

This directory contains non-code project knowledge -- engineering logs,
technical decisions and forward-looking plans -- kept separate from source code
so each evolves independently and remains easy to search.

## engineering_log/
Chronological problem to diagnosis to root-cause to fix records from each
validation round. One file per major validation session/environment.

- local_validation_day1.md -- macOS local validation (Observation and
  Geometry layers, 7 issues resolved)
- hpc_deployment.md -- AICR HPC deployment validation (2 issues resolved)
- talkvid_audio_download.md -- downloading TalkVid with audio: the video-only
  format, tail-gap rejections, two id bugs, the YouTube bot check, and a false
  A/V-misalignment finding of my own

## plans/
Forward-looking plans: what we intend to build next, and what has to be true for
it to work. Distinct from decisions/ (取舍已定) and engineering_log/ (故障已修)
-- a plan records intent, and the concrete trade-offs made while implementing it
should become their own decisions/ record.

- multilayer_semantic_representation.md -- the L1/L2/L3 human-readable
  representation layered on top of the 63-dim tensor: layer definitions, why L1
  and L2 are prerequisites for L3, open questions about the L3 axes, and the
  concrete L1 encode/decode spec with its acceptance criteria (L1 now implemented)
- l2_l3_rollout.md -- staged plan for L2 (action phrases) and L3 (communicative
  intent) on top of the implemented L1, with a measurable gate per step

## milestones/
Dated snapshots of what was achieved: numbers, data locations, open problems
and pending decisions. Reasons are not repeated here; they link to decisions/
and engineering_log/.

- 2026-09-23_l1_audio_talkvid.md -- L1 text layer, the 04_audio sidecar and
  the first multi-speaker TalkVid batch (206 clips); open problems and the
  decisions waiting on the owner

## decisions/
Standalone technical decision records (Architecture Decision Record style:
why we chose A over B), one file per decision so each remains easy to find
and update independently as the project evolves.

- mediapipe_version_pinning.md -- why mediapipe is pinned to 0.10.21
- observation_timeline_only.md -- why the Haar face check was removed and the
  visual verdict moved to the Geometry layer's MediaPipe tracking rate
- v0_training_data_contract.md -- what the pipeline guarantees for an
  autoencoder v0: timestamp policy, resampling, loss contract, limitations
- pipeline_run_layout.md -- why there is one entrypoint, how run directories
  are named, and what counts as a completed stage
- feature_vector_single_source_of_truth.md -- why schemas/geometry.schema.json
  is authoritative and code derives from it, never restates it
- geometry_three_track_structure.md -- why the Geometry layer fans out into
  three tracks, and why head pose uses a 6D rotation representation
- l1_text_codec.md -- the L1 atomic text layer: rules in schemas/l1_rules.json,
  level-based decode, bounded segment merging, rotation relative to a window
  reference, masks as header intervals; with the measured round-trip numbers
- audio_conditioning_sidecar.md -- why audio is a 04_audio sidecar aligned by
  real time (not tensor channels), its representation, validity rules and the
  measured lip/audio sync
