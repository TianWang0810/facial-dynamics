"""
汇总 check_pts.py 和 check_face.py 的结果,
按文档 1.1(视频类交付物) / 1.2(质量控制指标) 的字段要求,
生成 video_metadata.json 和 quality.json。
用法: python merge_observation_results.py <video_dir>
(video_dir 下需已存在 video_metadata_check.json 和 face_quality_check.json)
"""
import json
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("用法: python merge_observation_results.py <video_dir>")
        sys.exit(1)

    root_dir = sys.argv[1]
    pts_check_path = os.path.join(root_dir, "video_metadata_check.json")
    face_check_path = os.path.join(root_dir, "face_quality_check.json")

    with open(pts_check_path, "r", encoding="utf-8") as f:
        pts_results = json.load(f)
    with open(face_check_path, "r", encoding="utf-8") as f:
        face_results = json.load(f)

    face_by_file = {r["file"]: r for r in face_results}

    video_metadata_list = []
    quality_list = []

    for pts in pts_results:
        filename = pts["file"]
        face = face_by_file.get(filename, {})

        video_metadata_list.append({
            "file": filename,
            "resolution": pts.get("resolution"),
            "fps_declared": pts.get("fps_declared"),
            "n_frames_decoded": pts.get("n_frames_decoded"),
            "duration_from_pts_sec": pts.get("duration_from_pts"),
            "first_pts_sec": pts.get("first_pts"),
            "last_pts_sec": pts.get("last_pts"),
        })

        pts_issues = pts.get("issues", [])
        pts_ok = pts.get("is_monotonic", False) and len(pts_issues) == 0

        face_detection_rate = face.get("face_detection_rate", 0.0)
        clip_face_label = face.get("clip_quality_label", "reject")

        if pts_ok and clip_face_label == "usable":
            final_label = "usable"
        elif pts_ok and clip_face_label == "low-quality":
            final_label = "low-quality"
        else:
            final_label = "reject"

        quality_list.append({
            "file": filename,
            "pts_completeness": {
                "is_monotonic": pts.get("is_monotonic"),
                "max_frame_gap_sec": pts.get("max_frame_gap"),
                "min_frame_gap_sec": pts.get("min_frame_gap"),
                "issues": pts_issues,
            },
            "frame_level_availability": {
                "face_detection_rate": face_detection_rate,
                "avg_confidence": face.get("avg_confidence"),
                "min_confidence": face.get("min_confidence"),
            },
            "clip_quality_label": final_label,
        })

    video_metadata_out = os.path.join(root_dir, "video_metadata.json")
    quality_out = os.path.join(root_dir, "quality.json")

    with open(video_metadata_out, "w", encoding="utf-8") as f:
        json.dump(video_metadata_list, f, indent=2, ensure_ascii=False)
    with open(quality_out, "w", encoding="utf-8") as f:
        json.dump(quality_list, f, indent=2, ensure_ascii=False)

    print(f"已生成: {video_metadata_out}")
    print(f"已生成: {quality_out}")
    print(f"\n--- quality.json 摘要 ---")
    for q in quality_list:
        print(f"{q['file']}: {q['clip_quality_label']}")


if __name__ == "__main__":
    main()
