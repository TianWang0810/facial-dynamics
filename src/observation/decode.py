"""
Single decode path shared by the Observation and Geometry layers.

Every frame is decoded once, by PyAV, and carries its own presentation timestamp
with it. Previously timing came from a PyAV pass and pixels from a separate
OpenCV pass, and the two were related only by array position -- a correspondence
that is not verified by equal lengths, since the two decoders disagree about
what counts as a frame. Here a frame and its PTS are the same object, so there
is nothing to misalign.

Timestamp policy
----------------
A frame whose PTS the container does not provide is NOT dropped. Dropping shifts
every later frame's index by one while leaving the array length plausible, which
is precisely the silent corruption that is hardest to detect downstream. Such a
frame is kept with pts_sec = NaN and flagged, and the caller decides.

Duplicate and backward timestamps are likewise preserved as decoded and reported.
A stream whose PTS goes backwards is a damaged stream; rewriting it into
something monotonic would hide that, so the raw values stay and the issue is
recorded for the quality report.

MediaPipe's VIDEO running mode needs a strictly increasing integer millisecond
clock. That requirement belongs to the tracker, not to the data, so
tracker_timestamps_ms() derives a separate monotonic clock FROM the real PTS and
returns it alongside; pts_sec is never overwritten to satisfy it.
"""
import numpy as np

# PyAV is imported lazily inside the functions that decode. analyze_timeline() and
# tracker_timestamps_ms() are pure numpy over a PTS array, and the timestamp
# policy they implement is the part most worth testing; requiring a codec library
# to import the module would make those tests need an environment they do not use.


class FrameTimingIssue:
    """Issue codes recorded per clip. Strings, so they survive into JSON reports."""
    MISSING_PTS = "missing_pts"
    NON_MONOTONIC = "non_monotonic_pts"
    DUPLICATE_PTS = "duplicate_pts"
    LARGE_GAP = "large_frame_gap"
    # A large gap before the LAST frame only. Cutting a clip without re-encoding
    # (yt-dlp --download-sections) routinely leaves the final frame 2-3 steps
    # after its predecessor: 5 of the first 16 TalkVid downloads had exactly this
    # and nothing else. Reported under its own code so a policy can tell a cut
    # artefact on one frame from a gap inside the clip; still an issue, still
    # never repaired.
    TAIL_GAP = "tail_frame_gap"


def open_video(mp4_path: str):
    """Container plus the primary video stream and its static properties."""
    import av

    container = av.open(mp4_path)
    stream = container.streams.video[0]
    properties = {
        "time_base": stream.time_base,
        "fps_declared": float(stream.average_rate) if stream.average_rate else None,
        "width": stream.codec_context.width,
        "height": stream.codec_context.height,
    }
    return container, stream, properties


def iter_frames(mp4_path: str, want_pixels: bool = True):
    """Yield (decode_index, pts_sec, frame_rgb) for every decoded frame.

    decode_index counts frames in decode order starting at 0 and is never skipped,
    so it is a stable identifier for a frame within a clip even when its timestamp
    is missing. pts_sec is NaN for a frame the container gave no PTS. frame_rgb is
    None when want_pixels is False, which lets the timing-only pass avoid the
    RGB conversion cost.
    """
    container, stream, properties = open_video(mp4_path)
    time_base = properties["time_base"]
    try:
        for decode_index, frame in enumerate(container.decode(stream)):
            pts_sec = float("nan") if frame.pts is None else float(frame.pts * time_base)
            pixels = frame.to_ndarray(format="rgb24") if want_pixels else None
            yield decode_index, pts_sec, pixels
    finally:
        container.close()


def read_stream_timing(mp4_path: str) -> dict:
    """Container-level timing of the video and (optional) audio stream, from headers only.

    Both streams' PTS live on the one container timeline, so their start times
    are directly comparable; av_start_offset_sec = audio start - video start.
    A clip cut without re-encoding routinely starts its video on an earlier
    keyframe than its audio, so a non-zero offset is normal and is REPORTED, not
    corrected: downstream alignment uses each stream's own PTS, never position.
    Nothing is decoded here.
    """
    import av

    with av.open(mp4_path) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0] if container.streams.audio else None

        def seconds(stream, value):
            return None if value is None else float(value * stream.time_base)

        record = {
            "video_start_sec": seconds(video, video.start_time),
            "has_audio": audio is not None,
            "audio_codec": audio.codec_context.name if audio else None,
            "audio_sample_rate": int(audio.rate) if audio else None,
            "audio_channels": int(audio.codec_context.channels) if audio else None,
            "audio_start_sec": seconds(audio, audio.start_time) if audio else None,
            "audio_duration_sec": seconds(audio, audio.duration) if audio else None,
        }
    if record["has_audio"] and None not in (record["audio_start_sec"], record["video_start_sec"]):
        record["av_start_offset_sec"] = record["audio_start_sec"] - record["video_start_sec"]
    else:
        record["av_start_offset_sec"] = None
    return record


def read_timeline(mp4_path: str) -> dict:
    """Decode timing only and describe the clip's timeline.

    Returns the full per-frame pts_sec array -- including NaN for frames whose
    timestamp was missing, at their true positions -- plus the structured issues
    found. Nothing is repaired here.
    """
    container, stream, properties = open_video(mp4_path)
    time_base = properties["time_base"]
    pts_sec = []
    try:
        for frame in container.decode(stream):
            pts_sec.append(float("nan") if frame.pts is None else float(frame.pts * time_base))
    finally:
        container.close()

    timeline = analyze_timeline(pts_sec, properties["fps_declared"])
    timeline["streams"] = read_stream_timing(mp4_path)
    timeline.update({
        "resolution": "%dx%d" % (properties["width"], properties["height"]),
        "fps_declared": properties["fps_declared"],
        "width": properties["width"],
        "height": properties["height"],
    })
    return timeline


def analyze_timeline(pts_sec, fps_declared, gap_factor: float = 1.5) -> dict:
    """Classify a per-frame PTS array without modifying it.

    Pure, so it can be tested on synthetic timelines with no video involved.
    """
    pts = np.asarray(pts_sec, dtype=np.float64)
    n_frames = int(pts.shape[0])
    finite = np.isfinite(pts)
    n_missing = int((~finite).sum())

    issues = []
    if n_missing:
        issues.append({
            "code": FrameTimingIssue.MISSING_PTS,
            "n_frames": n_missing,
            "frame_indices": np.flatnonzero(~finite).tolist()[:50],
            "detail": "frames kept in place with pts_sec = NaN; dropping them would shift every later index",
        })

    # Comparisons only between consecutive frames that both have a timestamp.
    both = finite[:-1] & finite[1:] if n_frames > 1 else np.zeros(0, dtype=bool)
    steps = np.full(max(0, n_frames - 1), np.nan)
    if n_frames > 1:
        steps[both] = pts[1:][both] - pts[:-1][both]

    backward = np.flatnonzero(both & (steps < 0))
    duplicate = np.flatnonzero(both & (steps == 0))
    if backward.size:
        issues.append({
            "code": FrameTimingIssue.NON_MONOTONIC, "n_frames": int(backward.size),
            "frame_indices": backward.tolist()[:50],
            "detail": "PTS decreases; stream is damaged and the values are left as decoded",
        })
    if duplicate.size:
        issues.append({
            "code": FrameTimingIssue.DUPLICATE_PTS, "n_frames": int(duplicate.size),
            "frame_indices": duplicate.tolist()[:50],
            "detail": "consecutive frames share a timestamp",
        })

    finite_steps = steps[np.isfinite(steps)]
    median_step = float(np.median(finite_steps)) if finite_steps.size else None
    expected_step = (1.0 / fps_declared) if fps_declared else median_step
    large_gaps = []
    if expected_step and finite_steps.size:
        threshold = expected_step * gap_factor
        for index in np.flatnonzero(np.isfinite(steps) & (steps > threshold)):
            large_gaps.append({"after_frame": int(index), "gap_sec": float(steps[index])})
        interior = [g for g in large_gaps if g["after_frame"] != n_frames - 2]
        tail = [g for g in large_gaps if g["after_frame"] == n_frames - 2]
        if interior:
            issues.append({
                "code": FrameTimingIssue.LARGE_GAP, "n_frames": len(interior),
                "frame_indices": [g["after_frame"] for g in interior][:50],
                "detail": "gap exceeds %.4fs (%.1fx the expected step)" % (threshold, gap_factor),
            })
        if tail:
            issues.append({
                "code": FrameTimingIssue.TAIL_GAP, "n_frames": 1,
                "frame_indices": [tail[0]["after_frame"]],
                "detail": "gap of %.4fs before the final frame only; typical of a cut made without "
                          "re-encoding" % tail[0]["gap_sec"],
            })

    is_strictly_monotonic = bool(finite_steps.size and np.all(finite_steps > 0)) and n_missing == 0

    return {
        "pts_sec": [None if not np.isfinite(v) else float(v) for v in pts],
        "n_frames_decoded": n_frames,
        "n_missing_pts": n_missing,
        "is_strictly_monotonic": is_strictly_monotonic,
        "median_step_sec": median_step,
        "fps_from_pts": (1.0 / median_step) if median_step and median_step > 0 else None,
        "max_frame_gap_sec": float(finite_steps.max()) if finite_steps.size else None,
        "min_frame_gap_sec": float(finite_steps.min()) if finite_steps.size else None,
        "first_pts_sec": float(pts[finite][0]) if finite.any() else None,
        "last_pts_sec": float(pts[finite][-1]) if finite.any() else None,
        "duration_from_pts_sec": (float(pts[finite][-1] - pts[finite][0]) if finite.sum() > 1 else None),
        "large_gaps": large_gaps,
        "issues": issues,
    }


def tracker_timestamps_ms(pts_sec) -> np.ndarray:
    """Strictly increasing integer-millisecond clock derived from real PTS.

    MediaPipe's VIDEO mode rejects a non-increasing timestamp, but two frames
    3ms apart round to the same millisecond and a missing PTS has no millisecond
    at all. Those are tracker constraints, so they are resolved in a SEPARATE
    array: the returned clock is nudged forward where required, while pts_sec
    keeps the values the container actually reported.

    Frames with no PTS take the previous frame's clock plus one millisecond,
    which keeps the tracker advancing without inventing a timestamp for the data.
    """
    pts = np.asarray(pts_sec, dtype=np.float64)
    clock = np.empty(pts.shape[0], dtype=np.int64)
    previous = -1
    for index, value in enumerate(pts):
        candidate = previous + 1 if not np.isfinite(value) else int(round(value * 1000.0))
        if candidate <= previous:
            candidate = previous + 1
        clock[index] = candidate
        previous = candidate
    return clock
