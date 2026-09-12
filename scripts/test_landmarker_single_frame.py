"""
单帧验证: 从一个已下载的视频里截一帧,跑MediaPipe Face Landmarker,
确认能正确输出3D landmarks + blendshape + head pose。
API来源: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
版本: mediapipe==0.10.21 (明确锁定,避开1.0.x在Mac上的Metal崩溃问题)
用法: python test_landmarker_single_frame.py <mp4_path> <model_path>
"""
import cv2
import mediapipe as mp
import sys
import json

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def main():
    if len(sys.argv) < 3:
        print("用法: python test_landmarker_single_frame.py <mp4_path> <model_path>")
        sys.exit(1)

    mp4_path = sys.argv[1]
    model_path = sys.argv[2]

    cap = cv2.VideoCapture(mp4_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("无法读取视频第一帧")
        sys.exit(1)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        print(f"检测到 {len(result.face_landmarks)} 张脸")

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            print(f"\n--- 3D Landmarks ---")
            print(f"关键点数量: {len(landmarks)}")
            print(f"第0个点: x={landmarks[0].x:.4f}, y={landmarks[0].y:.4f}, z={landmarks[0].z:.4f}")

        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]
            print(f"\n--- Blendshapes ---")
            print(f"系数数量: {len(blendshapes)}")
            for bs in blendshapes[:5]:
                print(f"  {bs.category_name}: {bs.score:.4f}")

        if result.facial_transformation_matrixes:
            matrix = result.facial_transformation_matrixes[0]
            print(f"\n--- Head Pose (变换矩阵) ---")
            print(f"矩阵形状: {matrix.shape if hasattr(matrix, 'shape') else len(matrix)}")
            print(matrix)


if __name__ == "__main__":
    main()
