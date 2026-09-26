# facial-dynamics: semantic token representation of facial expressions

> Turn expression editing from tweaking parameters into editing meaning.

## Maintainers

| Name | GitHub |
|---|---|
| Shuran Zhao | [@shurandaa](https://github.com/shurandaa) |
| Tian Wang | [@TianWang0810](https://github.com/TianWang0810) |

## What we are building

Adjusting an expression today works the same way on almost any target: a 3D digital
human, an anime character, a game NPC or a robot face. Someone edits a list of
parameters that only specialists can read, such as `jawOpen = 0.37`,
`browInnerUp = 0.12` or "servo 7 to 42°", and every kind of target has its own
parameter set.

We want an expression representation that works for any target, that people can read,
and that they can edit directly:

```
expression on any target ──encode──▶ semantic tokens (readable, editable) ──decode──▶ expression parameters ──▶ any target
 real video / 3D / anime              "mouth moderately open for 0.3 s,
 games / robots                        then a strong blink"
```

It has to satisfy three properties:

1. **Interpretable**: the intermediate representation is segmented text a person can
   read, not a latent vector.
2. **Editable**: change the text directly, for example `slight` → `strong`, or a blink
   made 0.1 s longer. After decoding, only the matching channels and time spans change;
   everything else stays put (disentanglement).
3. **Invertible, with attributable error**: the encode → decode round-trip error is
   measured, and it can be broken down into how much information each layer loses.

The end goal is for 3D, anime, game and robot expressions all to be adjusted at the
semantic level. Edits land on one shared set of expression parameters, which are then
retargeted to the specific target.

## Architecture

```
            ┌───────────── target adapters (input) ─────────────┐
 real video ──▶ Observation → Geometry → Features                │  ← implemented here
 3D / games ──▶ direct mapping from ARKit-style blendshapes      │  ← planned
 anime / robots ──▶ native parameters → shared parameters        │  ← planned
            └───────────────────────┬──────────────────────────┘
                                    ▼
          shared expression parameters: 63 dims per frame (schemas/geometry.schema.json)
          52 blendshapes · head translation 3 · head rotation 6D · gaze 2
                                    ▼
   ┌────────────────── semantic layers (multi-level tokens) ──────────────────┐
   │ L1 atomic       per channel: intensity level + time segments  rules, deterministic │ ✅ implemented
   │ L2 action       blink / smile / nod … (with duration, intensity)  vocabulary + templates │ 📋 planned
   │ L3 intent       emotion · scenario · meaning · amplitude (parallel tracks)  generative model │ 📋 planned
   └──────────────────────────────────┬───────────────────────────────────────┘
                                      ▼  decode (every layer round-trips fully to L1)
          shared expression parameters ──▶ retargeting (3D / anime / games / robots) ← planned
```

Why three layers: the map from L3 to parameters is many-to-one, so on its own there is
no way to tell whether a decode is "right". L1 and L2 are auditable checkpoints in
between. L3 → L2 tests whether an intent maps to plausible actions, L2 → L1 tests
whether actions map to precise geometry, and L1 → parameters sets the lower bound on
quantization error. The disentanglement claim ("changing one semantic axis only affects
the channels it should") is tested along this chain of evidence. See
[`docs/plans/multilayer_semantic_representation.md`](docs/plans/multilayer_semantic_representation.md).

The shared parameters use the 52 ARKit-named blendshapes (MediaPipe emits the same
names), because 3D and game pipelines already use this naming widely. When those
targets are connected later, they can reuse the same semantics.

## What L1 looks like

A real encoding of one blink in a 1-second window (30 frames at 30 Hz; validation
clip, window @60):

```json
"eyeBlinkLeft": [
  {"t": [0.0, 0.6],    "level": "slight",   "value_hint": 0.1261},
  {"t": [0.6, 0.8333], "level": "strong",   "value_hint": 0.563},
  {"t": [0.8333, 1.0], "level": "moderate", "value_hint": 0.2758}
]
```

- The levels are fixed: `none / slight / moderate / strong`. Signed channels are
  written `+slight`, `-moderate`.
- `level` is what you edit. If you change the level, the decode follows the new level;
  `value_hint` only refines the intensity within the level.
- Head rotation is a rotation vector relative to the window's reference pose, with
  x/y/z quantized jointly. No Euler angles are used.
- Every threshold is written only in [`schemas/l1_rules.json`](schemas/l1_rules.json).
  Nothing is fitted to data, so the same window always yields the same text.

## Status

| Component | Status | Notes |
|---|---|---|
| Real video → 63-dim parameters | ✅ | Five-stage pipeline with quality masks and per-frame confidences |
| Audio conditioning sidecar | ✅ | 80-bin log-mel + waveform, row-aligned to video windows by real time (not part of the 63 dims) |
| L1 encode / decode / evaluation | ✅ | 57 text channels; all 7 structural checks pass |
| Multi-speaker data | 🟡 | TalkVid, 206 speakers, 6245 windows (~50 min); target ~1500 clips |
| L2 action phrases | 🟡 | 6 phrase types (blink, lids_lowered, smile, brow_raise, head_move, gaze_shift) on a track-level L1; lossless round trip; knowledge base + exemplar retrieval v1 — see [`docs/milestones/2026-09-26_detail_net_l2.md`](docs/milestones/2026-09-26_detail_net_l2.md) |
| L3 communicative intent | 📋 | First test axis independence with weak labels, then fix the axes; the generative model trains in the modelling repo |
| Input adapters and retargeting for other targets | 📋 | 3D / anime / games / robots |

L1 round-trip error on 206 clips (described frames only): blendshape MAE 0.0136, mean
head rotation error 0.92°, 2.0 segments per channel per window on average. Known issues
and pending decisions are in
[`docs/milestones/2026-09-23_l1_audio_talkvid.md`](docs/milestones/2026-09-23_l1_audio_talkvid.md).
The staged rollout plan, with a gate for each step, is in
[`docs/plans/l2_l3_rollout.md`](docs/plans/l2_l3_rollout.md).

## Repository scope

This repository produces **data, the semantic layers and evaluation**. It contains no
model training code. The L3 generative model and the detail network (L1 decode + audio →
original parameters) are trained in the modelling repository; this repository exports
their training pairs and the evaluation scripts. `schemas/geometry.schema.json` is a
cross-team contract with the modelling team.

## Quick start

The environment is locked for linux-64 (AICR HPC) only; see
`environment/env-v0.1/NOTES.md`. On the HPC, **never compute on the login node**; get a
compute node with `srun` first.

```bash
conda-lock install --name talkvid environment/env-v0.1/conda-lock.yml

# Video -> 63-dim parameters + audio sidecar (single entrypoint, resumable per stage)
python scripts/run_pipeline.py \
    --input_dir ../data/talkvid_bench \
    --model_path ../models/face_landmarker.task \
    --output_root ../runs

# Parameters -> L1 text -> parameters, plus a round-trip error report
python scripts/l1_encode.py --run_dir ../runs/<id> --output ../runs/<id>/l1/l1_text.json
python scripts/l1_decode.py --text ../runs/<id>/l1/l1_text.json --output ../runs/<id>/l1/l1_decoded.npz
python scripts/l1_report.py --run_dir ../runs/<id> --output_dir ../runs/<id>/l1 --text ../runs/<id>/l1/l1_text.json

# Tests (no video or model needed, < 1 s)
python -m pytest tests -q
```

## Layout

```
schemas/         cross-team contract and rules: geometry.schema.json (63-dim layout), l1_rules.json, audio_features.json
src/observation  timeline decoding and trust verdict (no pixels)
src/geometry     three tracks: landmarks+blendshapes, head pose (6D rotation), gaze
src/sequence     resampling, QC, windowing, training contract (masks and loss weights)
src/dynamics     velocity / acceleration validation from real PTS
src/audio        audio conditioning features
src/semantic     L1 semantic codec and evaluation
scripts/         per-stage CLIs, run_pipeline.py, l1_*.py, fetch_talkvid.py
docs/            decisions/ (why we chose what) · engineering_log/ (failures and root causes)
                 plans/ (forward-looking plans) · milestones/ (dated snapshots); index in docs/README.md
```

## Design principles (summary)

Each rule comes from a real silent data-corruption bug; the reasoning is recorded in
`docs/decisions/`.

- Time always comes from real PTS, never `frame_idx / fps`. A damaged timeline is
  reported, never repaired.
- Rotation uses the 6D representation and is interpolated on the manifold, never with
  Euler angles or element-wise operations.
- Gaze is a separate track and is never derived from the `eyeLook*` blendshapes.
- The four mask signals are stored separately and combined only at use. The semantic
  layer describes only frames with `loss_weight() > 0`.
- Every layout and threshold comes only from JSON; code never restates an index or a
  threshold.
