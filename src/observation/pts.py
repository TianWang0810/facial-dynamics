"""
Extract real per-frame PTS (presentation timestamps) and check monotonicity/gaps/drops.
Reusable core logic; CLI entrypoint is in scripts/run_observation.py
"""
import av


def analyze_video(mp4_path: str) -> dict:
    container = av.open(mp4_path)
    video_stream = container.streams.video[0]
    time_base = video_stream.time_base
    fps_declared = float(video_stream.average_rate) if video_stream.average_rate else None
    width = video_stream.codec_context.width
    height = video_stream.codec_context.height

    pts_list_sec = []
    for frame in container.decode(video_stream):
        if frame.pts is None:
            continue
        pts_list_sec.append(float(frame.pts * time_base))
    container.close()

    n_frames = len(pts_list_sec)
    issues = []
    is_monotonic = all(pts_list_sec[i] < pts_list_sec[i + 1] for i in range(n_frames - 1))
    if not is_monotonic:
        issues.append("PTS is not strictly monotonically increasing")

    gaps = []
    if n_frames > 1 and fps_declared:
        expected_gap = 1.0 / fps_declared
        for i in range(n_frames - 1):
            gap = pts_list_sec[i + 1] - pts_list_sec[i]
            gaps.append(gap)
            if gap > expected_gap * 1.5:
                issues.append(f"Abnormally large gap between frame {i}->{i+1}: {gap:.4f}s (expected ~{expected_gap:.4f}s)")

    return {
        "file": mp4_path.split("/")[-1],
        "resolution": f"{width}x{height}",
        "fps_declared": fps_declared,
        "n_frames_decoded": n_frames,
        "duration_from_pts": pts_list_sec[-1] - pts_list_sec[0] if n_frames > 1 else None,
        "first_pts": pts_list_sec[0] if pts_list_sec else None,
        "last_pts": pts_list_sec[-1] if pts_list_sec else None,
        "is_monotonic": is_monotonic,
        "max_frame_gap": max(gaps) if gaps else None,
        "min_frame_gap": min(gaps) if gaps else None,
        "issues": issues,
        "quality_label": "usable" if not issues else "needs_review",
    }
