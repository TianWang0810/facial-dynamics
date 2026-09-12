"""
汇总4个视频的canonicalized geometry结果,
按文档2.4节产出: geometry.parquet / geometry_metadata.json / geometry_quality.json
用法: python summarize_geometry.py <geometry_test_dir> <output_dir>
"""
import numpy as np
import pandas as pd
import json
import sys
import os
import glob
from datetime import datetime

MEDIAPIPE_VERSION = "0.10.21"
CONFIDENCE_THRESHOLD = 0.0


def load_and_flatten(canonical_npz_path: str, clip_id: str) -> list:
    data = np.load(canonical_npz_path, allow_pickle=True)

    canonical_landmarks = data["canonical_landmarks"]
    blendshapes = data["blendshapes"]
    headpose = data["headpose"]
    detected = data["detected"]
    eye_distance = data["eye_distance"]

    n_frames = canonical_landmarks.shape[0]
    rows = []
    for i in range(n_frames):
        rows.append({
            "clip_id": clip_id,
            "frame_idx": i,
            "detected": bool(detected[i]),
            "eye_distance": float(eye_distance[i]) if not np.isnan(eye_distance[i]) else None,
            "landmarks": canonical_landmarks[i].flatten().tolist(),
            "blendshapes": blendshapes[i].tolist(),
            "headpose": headpose[i].flatten().tolist(),
        })
    return rows


def main():
    if len(sys.argv) < 3:
        print("用法: python summarize_geometry.py <geometry_test_dir> <output_dir>")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    canonical_files = sorted(glob.glob(os.path.join(in_dir, "*_canonical.npz")))
    if not canonical_files:
        print(f"在 {in_dir} 下没有找到 *_canonical.npz 文件")
        sys.exit(1)

    all_rows = []
    quality_per_clip = []

    for f in canonical_files:
        clip_id = os.path.basename(f).replace("_canonical.npz", "")
        print(f"处理: {clip_id}")

        rows = load_and_flatten(f, clip_id)
        all_rows.extend(rows)

        n_frames = len(rows)
        n_detected = sum(1 for r in rows if r["detected"])
        detection_rate = n_detected / n_frames if n_frames > 0 else 0.0
        tracking_failure_rate = 1.0 - detection_rate

        eye_dists = [r["eye_distance"] for r in rows if r["eye_distance"] is not None]
        eye_dist_mean = float(np.mean(eye_dists)) if eye_dists else None
        eye_dist_std = float(np.std(eye_dists)) if eye_dists else None

        if tracking_failure_rate <= 0.05:
            clip_label = "usable"
        elif tracking_failure_rate <= 0.3:
            clip_label = "low-quality"
        else:
            clip_label = "reject"

        quality_per_clip.append({
            "clip_id": clip_id,
            "n_frames": n_frames,
            "n_tracked": n_detected,
            "tracking_failure_rate": round(tracking_failure_rate, 4),
            "eye_distance_mean": eye_dist_mean,
            "eye_distance_std": eye_dist_std,
            "clip_quality_label": clip_label,
        })

    df = pd.DataFrame(all_rows)
    parquet_path = os.path.join(out_dir, "geometry.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"\n已生成: {parquet_path} ({len(df)} 行)")

    metadata = {
        "canonicalization_method": {
            "description": "以两眼外眼角欧氏距离为缩放基准,以两点中点为平移原点",
            "reference_landmarks": {"point_a": 33, "point_b": 263},
            "reference_source": "多篇独立mediapipe社区文档交叉确认(landmark索引一致,左右命名有分歧但不影响使用)",
        },
        "tracker": {
            "name": "MediaPipe Face Landmarker",
            "version": MEDIAPIPE_VERSION,
            "note": "明确锁定0.10.x版本,避开1.0.x在macOS上的Metal兼容性崩溃问题",
            "delegate": "CPU",
            "model_file": "face_landmarker.task (float16, v1)",
        },
        "output_fields": {
            "landmarks": "478 x 3 (canonical坐标,已按eye_distance归一化)",
            "blendshapes": "52维,ARKit兼容命名",
            "headpose": "4x4变换矩阵,展平为16维",
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = os.path.join(out_dir, "geometry_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"已生成: {metadata_path}")

    quality = {
        "per_clip": quality_per_clip,
        "note": "本次为小样本验证(4条),暂不做按人口子群体的统计,留待规模化处理阶段补充",
        "filtered_clips": [q["clip_id"] for q in quality_per_clip if q["clip_quality_label"] == "reject"],
    }
    quality_path = os.path.join(out_dir, "geometry_quality.json")
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2, ensure_ascii=False)
    print(f"已生成: {quality_path}")

    print(f"\n--- 摘要 ---")
    for q in quality_per_clip:
        print(f"{q['clip_id']}: 跟踪失败率={q['tracking_failure_rate']:.4f}, 标签={q['clip_quality_label']}")


if __name__ == "__main__":
    main()
