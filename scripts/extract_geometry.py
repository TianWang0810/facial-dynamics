"""
对整段视频逐帧跑Face Landmarker,提取3D landmarks + blendshape + head pose序列。
用法: python extract_geometry.py <mp4_path> <model_path> <output_npz_path>
"""
import cv2
import mediapipe as mp
import numpy as np
import sys
import os

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

BLENDSHAPE_NAMES = None


def extract_geometry(mp4_path: str, model_path: str) -> dict:
    global BLENDSHAPE_NAMES

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {mp4_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    frame_idx = 0
    last_timestamp_ms = -1

    all_detected = []
    all_landmarks = []
    all_blendshapes = []
    all_headpose = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                landmarks_arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

                bs = result.face_blendshapes[0]
                if BLENDSHAPE_NAMES is None:
                    BLENDSHAPE_NAMES = [c.category_name for c in bs]
                bs_arr = np.array([c.score for c in bs], dtype=np.float32)

                headpose_arr = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)

                all_detected.append(True)
                all_landmarks.append(landmarks_arr)
                all_blendshapes.append(bs_arr)
                all_headpose.append(headpose_arr)
            else:
                all_detected.append(False)
                all_landmarks.append(np.full((478, 3), np.nan, dtype=np.float32))
                all_blendshapes.append(np.full((52,), np.nan, dtype=np.float32))
                all_headpose.append(np.full((4, 4), np.nan, dtype=np.float32))

            frame_idx += 1

    cap.release()

    return {
        "n_frames": frame_idx,
        "detected": np.array(all_detected),
        "landmarks": np.stack(all_landmarks),
        "blendshapes": np.stack(all_blendshapes),
        "headpose": np.stack(all_headpose),
        "blendshape_names": np.array(BLENDSHAPE_NAMES) if BLENDSHAPE_NAMES else np.array([]),
    }


def main():
    if len(sys.argv) < 4:
        print("用法: python extract_geometry.py <mp4_path> <model_path> <output_npz_path>")
        sys.exit(1)

    mp4_path, model_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print(f"处理: {mp4_path}")
    data = extract_geometry(mp4_path, model_path)

    detection_rate = data["detected"].sum() / data["n_frames"] if data["n_frames"] > 0 else 0.0
    print(f"总帧数: {data['n_frames']}")
    print(f"Landmarker检测成功率: {detection_rate:.4f}")
    print(f"landmarks形状: {data['landmarks'].shape}")
    print(f"blendshapes形状: {data['blendshapes'].shape}")
    print(f"headpose形状: {data['headpose'].shape}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        n_frames=data["n_frames"],
        detected=data["detected"],
        landmarks=data["landmarks"],
        blendshapes=data["blendshapes"],
        headpose=data["headpose"],
        blendshape_names=data["blendshape_names"],
    )
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
