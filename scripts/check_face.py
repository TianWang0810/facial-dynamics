"""
用 OpenCV 内置的 Haar级联人脸检测器，对视频每一帧做检测。
不依赖任何外部下载的模型文件(用OpenCV pip包自带的haarcascade xml)。
更换记录: mediapipe(Mac上Metal崩溃) -> OpenCV DNN/Caffe(OpenCV5.0移除Caffe支持) -> 本方案
用法: python check_face.py <video_dir_containing_mp4_files>
"""
import cv2
import json
import sys
import os
import glob


def analyze_video_faces(mp4_path: str) -> dict:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {mp4_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        face_detected = len(faces) > 0
        if face_detected:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_area_ratio = (largest_face[2] * largest_face[3]) / (width * height)
            confidence = min(1.0, face_area_ratio * 20)
        else:
            confidence = 0.0

        frame_results.append({
            "frame_idx": frame_idx,
            "face_detected": face_detected,
            "n_faces": len(faces),
            "confidence": float(confidence),
        })
        frame_idx += 1

    cap.release()

    n_checked = len(frame_results)
    n_detected = sum(1 for r in frame_results if r["face_detected"])
    detection_rate = n_detected / n_checked if n_checked > 0 else 0.0
    confidences = [r["confidence"] for r in frame_results if r["face_detected"]]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    min_confidence = min(confidences) if confidences else 0.0

    if detection_rate >= 0.95:
        clip_quality = "usable"
    elif detection_rate >= 0.7:
        clip_quality = "low-quality"
    else:
        clip_quality = "reject"

    return {
        "file": os.path.basename(mp4_path),
        "resolution": f"{width}x{height}",
        "total_frames_read": frame_idx,
        "frames_checked": n_checked,
        "face_detection_rate": round(detection_rate, 4),
        "avg_confidence": round(avg_confidence, 4),
        "min_confidence": round(min_confidence, 4),
        "clip_quality_label": clip_quality,
        "low_confidence_frames": [
            r for r in frame_results if not r["face_detected"]
        ][:20],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python check_face.py <包含mp4文件的目录>")
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
            result = analyze_video_faces(mp4_path)
            all_results.append(result)
            print(json.dumps(
                {k: v for k, v in result.items() if k != "low_confidence_frames"},
                indent=2, ensure_ascii=False
            ))
            print("-" * 60)
        except Exception as e:
            print(f"处理失败: {mp4_path} | 错误: {e}")
            all_results.append({"file": os.path.basename(mp4_path), "error": str(e)})

    out_path = os.path.join(root_dir, "face_quality_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已写入: {out_path}")


if __name__ == "__main__":
    main()
