"""
Canonicalization: 身份归一化
方法: 以landmark 33和263(两眼外眼角,来源多篇独立资料交叉确认)的欧氏距离为缩放基准,
以两点中点为平移原点,将所有landmark坐标转换为归一化坐标。
用法: python canonicalize.py <input_npz> <output_npz>
"""
import numpy as np
import sys

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def canonicalize(landmarks: np.ndarray) -> dict:
    n_frames = landmarks.shape[0]
    canonical = np.full_like(landmarks, np.nan)
    eye_distances = np.full(n_frames, np.nan, dtype=np.float32)
    origins = np.full((n_frames, 3), np.nan, dtype=np.float32)

    for i in range(n_frames):
        frame_lm = landmarks[i]
        if np.isnan(frame_lm).any():
            continue

        p_left = frame_lm[LEFT_EYE_OUTER]
        p_right = frame_lm[RIGHT_EYE_OUTER]

        eye_dist = np.linalg.norm(p_left - p_right)
        origin = (p_left + p_right) / 2.0

        if eye_dist < 1e-6:
            continue

        canonical[i] = (frame_lm - origin) / eye_dist
        eye_distances[i] = eye_dist
        origins[i] = origin

    return {
        "canonical_landmarks": canonical,
        "eye_distance": eye_distances,
        "origin": origins,
        "reference_points": {"left": LEFT_EYE_OUTER, "right": RIGHT_EYE_OUTER},
    }


def main():
    if len(sys.argv) < 3:
        print("用法: python canonicalize.py <input_npz> <output_npz>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    data = np.load(in_path, allow_pickle=True)
    landmarks = data["landmarks"]

    result = canonicalize(landmarks)

    valid_frames = (~np.isnan(result["eye_distance"])).sum()
    total_frames = landmarks.shape[0]
    print(f"总帧数: {total_frames}")
    print(f"成功归一化帧数: {valid_frames} ({valid_frames/total_frames*100:.2f}%)")
    print(f"眼间距范围: {np.nanmin(result['eye_distance']):.4f} ~ {np.nanmax(result['eye_distance']):.4f}")
    print(f"眼间距均值/标准差: {np.nanmean(result['eye_distance']):.4f} / {np.nanstd(result['eye_distance']):.4f}")

    np.savez_compressed(
        out_path,
        canonical_landmarks=result["canonical_landmarks"],
        eye_distance=result["eye_distance"],
        origin=result["origin"],
        blendshapes=data["blendshapes"],
        headpose=data["headpose"],
        detected=data["detected"],
        blendshape_names=data["blendshape_names"],
        reference_points=np.array([LEFT_EYE_OUTER, RIGHT_EYE_OUTER]),
    )
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
