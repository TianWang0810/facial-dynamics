# Decision: the Observation layer validates time, not pixels

## Context

The Observation layer ran an OpenCV Haar cascade over every frame and folded its
face-detection rate into the clip verdict, alongside the PTS validation. On the
first real clip processed end to end, the two detectors disagreed completely:

| detector | source | detection rate | verdict |
|---|---|---|---|
| Haar cascade | Observation | 41.3% | `reject` |
| MediaPipe Face Landmarker | Geometry | 100% | `usable` |

MediaPipe tracked every one of the 121 frames and produced usable landmarks,
blendshapes and head pose throughout. The Haar verdict was simply wrong.

This was not a close call and not specific to one clip. `haarcascade_frontalface_default`
is a 2001-era frontal-face detector: it degrades with head rotation, non-frontal
pose, unusual lighting and high resolution -- exactly the conditions a talking-head
dataset consists of. It was chosen originally because the step "does not require a
fine-grained model", which turned out to be the wrong assumption: a gate that
decides whether a clip enters training does need to be right.

Worse, the verdict was inert. The Features layer reads the timeline verdict from
Observation but takes its detection rate from the Geometry layer's `detected`
column, so the clip above was kept despite `quality.json` saying `reject`. A
label that says "reject" while the pipeline proceeds is worse than no label: it
invites someone to trust it later.

## Decision

The Haar check is removed (`src/observation/quality.py` deleted). The Observation
layer answers exactly one question: **is this clip's time axis trustworthy?**

The visual verdict is the Geometry layer's MediaPipe tracking rate. That is the
same model that produces the features, so the gate and the data agree by
construction -- a frame that passes the gate is a frame the feature extractor
actually succeeded on. Two independent detectors disagreeing left no principled
way to choose, and the one being disagreed with was the weaker of the two.

`quality.json` now reports `timeline_label` (`usable` / `needs_review`) rather
than a blended `clip_quality_label`, and carries `visual_quality: null` with a
note pointing at `geometry_quality.json`. "needs_review" rather than "reject":
a timeline defect is a fact about the file, not a policy decision about whether
to keep it.

## Consequences

- Observation decodes each container **once, without pixels**. It was previously
  two passes, one PyAV for timing and one OpenCV for faces, with an RGB
  conversion per frame. On the test clip this took the stage from 23.2s to 0.4s,
  a 58x reduction, and the saving grows with resolution.
- The two QC axes stay separate and are applied in different places: the timeline
  check is a hard error raised by Observation; the detection-rate threshold is a
  configurable policy applied in Features against MediaPipe's output. Both are
  recorded with their own reason code, so one can be overridden without the other.
- OpenCV is no longer imported anywhere in the Observation layer.
- Clips that MediaPipe genuinely cannot track are still rejected -- by
  `--min_detection_rate`, on evidence from the model that matters.

## What this does NOT claim

MediaPipe's tracking rate is not ground truth about whether a face is present
either. It is the right gate because it measures the thing the pipeline depends
on -- whether features could be extracted -- not because it is a better face
detector in the abstract. A clip where MediaPipe tracks confidently but wrongly
would still pass, and nothing here detects that.
