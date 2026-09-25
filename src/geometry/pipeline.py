"""
Shared decode loop driving the three geometry tracks.

Each decoded frame fans out to:
    track A  src/geometry/landmarks.py  478 landmarks + 52 blendshapes
    track B  src/geometry/headpose.py   6D rotation + 3D translation
    track C  src/geometry/gaze.py       pitch / yaw

Tracks A and B read two different fields of one MediaPipe result, so they share a
single forward pass per frame rather than running the model twice; track C runs
its own estimator. The video is decoded exactly once here, which is also why the
tracks are driven from a common loop instead of each opening the file itself.

Two of the three tracks need a clip-level reference that cannot be known until
every frame has been seen -- head-pose translation is centred on the clip mean
and gaze is zero-meaned. run_clip() therefore collects per-frame results first
and applies those normalisations in a second pass over the collected arrays, not
over the video.

Timing comes from the same decode as the pixels. Frames are read through
src/observation/decode.py, so each frame arrives with its own PTS attached and
the two cannot be misaligned -- previously pixels came from an OpenCV pass and
timing from a separate PyAV pass, related only by array position.

MediaPipe's VIDEO running mode requires a strictly increasing integer-millisecond
timestamp. That is a tracker constraint, not a property of the data, so the clock
handed to it is derived from the real PTS by tracker_timestamps_ms() and stored
in its own column. pts_sec keeps what the container reported, including NaN where
no timestamp existed.
"""
import mediapipe as mp
import numpy as np

from src.geometry import gaze as gaze_module
from src.geometry import headpose as headpose_module
from src.geometry import landmarks as landmarks_module
from src.geometry.canonicalization import canonicalize
from src.observation.decode import analyze_timeline, iter_frames, tracker_timestamps_ms

FaceLandmarker = mp.tasks.vision.FaceLandmarker


def run_clip(mp4_path: str, model_path: str, gaze_backend, apply_gaze_tanh: bool = False) -> dict:
    """Run all three tracks over one clip and return frame-aligned arrays.

    Every returned array is indexed by decode order and has one entry per decoded
    frame, including frames with no PTS and frames where tracking failed. Nothing
    is dropped, so source_frame_idx is a valid index into all of them.
    """
    # Two passes over the container: the first reads timing only, so the tracker
    # clock can be derived from the complete PTS array before inference starts.
    # Cheaper than it looks -- want_pixels=False skips RGB conversion, which is
    # the dominant cost -- and it keeps the clock a function of the whole timeline
    # rather than of whatever has been seen so far.
    timing_pts = [pts for _, pts, _ in iter_frames(mp4_path, want_pixels=False)]
    if not timing_pts:
        raise RuntimeError("No frames decoded from %s" % mp4_path)
    timeline = analyze_timeline(timing_pts, None)
    tracker_clock = tracker_timestamps_ms(timing_pts)

    frames_a, frames_b = [], []
    gaze_pitch, gaze_yaw, gaze_conf = [], [], []
    blendshape_names = None
    frame_idx = 0

    with FaceLandmarker.create_from_options(landmarks_module.build_options(model_path)) as landmarker:
        for decode_index, _pts_sec, frame_rgb in iter_frames(mp4_path, want_pixels=True):
            if decode_index >= tracker_clock.shape[0]:
                break
            timestamp_ms = int(tracker_clock[decode_index])
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb), timestamp_ms
            )

            if result.face_landmarks:
                track_a = landmarks_module.parse_frame(result.face_landmarks[0], result.face_blendshapes[0])
                track_b = headpose_module.parse_frame(result.facial_transformation_matrixes[0])
                if blendshape_names is None:
                    blendshape_names = landmarks_module.blendshape_names(result.face_blendshapes[0])
            else:
                track_a = landmarks_module.empty_frame()
                track_b = headpose_module.empty_frame()

            pitch, yaw, confidence = gaze_backend.estimate(
                frame_rgb,
                track_a["landmarks"] if track_a["detected"] else None,
                frame_rgb.shape,
            )

            frames_a.append(track_a)
            frames_b.append(track_b)
            gaze_pitch.append(pitch)
            gaze_yaw.append(yaw)
            gaze_conf.append(confidence)
            frame_idx += 1

    if frame_idx == 0:
        raise RuntimeError("No frames decoded from %s" % mp4_path)
    if frame_idx != len(timing_pts):
        raise RuntimeError(
            "decode mismatch in %s: timing pass saw %d frames, pixel pass saw %d"
            % (mp4_path, len(timing_pts), frame_idx)
        )

    detected = np.array([f["detected"] for f in frames_a], dtype=bool)
    raw_landmarks = np.stack([f["landmarks"] for f in frames_a])
    blendshapes = np.stack([f["blendshapes"] for f in frames_a])
    headpose_r6 = np.stack([f["headpose_r6"] for f in frames_b])
    headpose_t_raw = np.stack([f["headpose_t_raw"] for f in frames_b])

    # Second pass: the clip-level normalisations.
    canonical = canonicalize(raw_landmarks)
    headpose_t, translation_reference = headpose_module.normalise_translation(headpose_t_raw, detected)
    pitch, yaw, gaze_reference = gaze_module.normalise_gaze(
        np.array(gaze_pitch, dtype=np.float64),
        np.array(gaze_yaw, dtype=np.float64),
        np.array(gaze_conf, dtype=np.float64),
        apply_tanh=apply_gaze_tanh,
        unit=gaze_backend.unit,
    )

    return {
        "n_frames": frame_idx,
        "pts_sec": np.array(timing_pts, dtype=np.float64),
        "tracker_timestamp_ms": tracker_clock,
        "timeline": timeline,
        "detected": detected,
        "canonical_landmarks": canonical["canonical_landmarks"],
        "eye_distance": canonical["eye_distance"],
        "blendshapes": blendshapes,
        "headpose_r6": headpose_r6,
        "headpose_t": headpose_t,
        "headpose_t_raw": headpose_t_raw,
        "gaze_pitch": pitch,
        "gaze_yaw": yaw,
        "confidences": {
            "landmarks": np.array([f["conf_landmarks"] for f in frames_a], dtype=np.float64),
            "blendshapes": np.array([f["conf_blendshapes"] for f in frames_a], dtype=np.float64),
            "headpose": np.array([f["conf_headpose"] for f in frames_b], dtype=np.float64),
            "gaze": np.array(gaze_conf, dtype=np.float64),
        },
        "references": {
            "headpose_translation": translation_reference.tolist(),
            "gaze_pitch_yaw": gaze_reference.tolist(),
        },
        "blendshape_names": blendshape_names or [],
        "gaze_backend": gaze_backend.name,
        "gaze_unit": gaze_backend.unit,
        "gaze_is_calibrated": bool(gaze_backend.is_calibrated),
        "gaze_tanh_applied": bool(apply_gaze_tanh),
    }
