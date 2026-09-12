"""
逐帧提取真实 PTS，检查单调性/跳变/丢帧，并输出基础 video_metadata.json
用法: python check_pts.py <video_dir_containing_mp4_files>
"""
import av
import json
import sys
import os
import glob


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
        pts_sec = float(frame.pts * time_base)
        pts_list_sec.append(pts_sec)

    container.close()

    n_frames = len(pts_list_sec)
    issues = []

    is_monotonic = all(pts_list_sec[i] < pts_list_sec[i+1] for i in range(n_frames - 1))
    if not is_monotonic:
        issues.append("PTS不是严格单调递增")

    gaps = []
    if n_frames > 1 and fps_declared:
        expected_gap = 1.0 / fps_declared
        for i in range(n_frames - 1):
            gap = pts_list_sec[i+1] - pts_list_sec[i]
            gaps.append(gap)
            if gap > expected_gap * 1.5:
                issues.append(f"帧{i}->{i+1}间隔异常大: {gap:.4f}s (期望约{expected_gap:.4f}s)")

    result = {
        "file": os.path.basename(mp4_path),
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
    return result


def main():
    if len(sys.argv) < 2:
        print("用法: python check_pts.py <包含mp4文件的目录>")
        sys.exit(1)

    root_dir = sys.argv[1]
    mp4_files = glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True)

    if not mp4_files:
        print(f"在 {root_dir} 下没有找到 mp4 文件")
        sys.exit(1)

    all_results = []
    for mp4_path in sorted(mp4_files):
        print(f"处理: {mp4_path}")
        try:
            result = analyze_video(mp4_path)
            all_results.append(result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("-" * 60)
        except Exception as e:
            print(f"处理失败: {mp4_path} | 错误: {e}")
            all_results.append({"file": os.path.basename(mp4_path), "error": str(e)})

    out_path = os.path.join(root_dir, "video_metadata_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已写入: {out_path}")


if __name__ == "__main__":
    main()
