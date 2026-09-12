# Decision: Pin mediapipe to 0.10.21

## Context

The Geometry layer requires a 3D facial landmark / blendshape extraction
tool. MediaPipe Face Landmarker was selected because it covers three of the
design doc's requirements in a single tool: 478 3D landmarks, 52 ARKit-
compatible blendshape coefficients, and head pose (via a transformation
matrix).

## Problem

Installing mediapipe via a plain pip install pulled in the newest release
(1.0.1 at the time), which crashes on macOS with a low-level error inside
the FaceDetector / FaceLandmarker graph. The crash stack trace includes
DrishtiMetalHelper and Service is unavailable, pointing to Apple's Metal
GPU-acceleration framework. Explicitly forcing a CPU delegate
(BaseOptions.Delegate.CPU) did not fully resolve the crash -- a downstream
calculator node (TensorsToDetectionsCalculator) still attempted to use
Metal internally on macOS.

## Investigation

A third-party project's changelog (PhotoCull) noted fixing "a mediapipe/
Metal crash in face scoring on Apple Silicon (mediapipe pinned less than
1.0)". Cross-checking mediapipe's release history confirmed 0.10.35/0.10.36
as the last stable releases before the 1.0.0 major version bump, which
happened very recently relative to this project's timeline. Multiple
community reports of successfully running Tasks-API-based detectors on
Mac (with CPU delegate) used 0.10.x releases specifically.

## Decision

Pin mediapipe==0.10.21 (a version already validated by the upstream
TalkVid repository's own requirements.txt, giving additional confidence)
across both macOS (local validation) and Linux (HPC), rather than
allowing the newest release to be installed by default.

## Outcome

With this pin, both the single-frame and full-video extraction pipelines
ran without any crashes on macOS. The same pin was reused on the HPC
(Linux) environment for consistency; no crash was observed there either
(GPU-unavailable conditions degrade gracefully to CPU on Linux instead of
crashing, unlike on macOS).

## Revisit Trigger

This decision should be revisited once the mediapipe 1.0.x line has been
in production for some time and the macOS Metal issue is confirmed fixed
upstream, or if a newer feature only available in 1.0.x becomes necessary
for the project.
