"""
Per-frame face detection quality check using OpenCV's Haar Cascade detector.
Selection rationale: see docs/decisions/ (mediapipe has known Metal compatibility
issues on macOS; this step does not require a fine-grained model, so we use a
zero-external-dependency classical approach instead).
"""
import cv2
import os


def analyze_video_faces(mp4_path: str) -> dict:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_detected = len(faces) > 0
        if face_detected:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_area_ratio = (largest_face[2] * largest_face[3]) / (width * height)
            confidence = min(1.0, face_area_ratio * 20)
        else:
            confidence = 0.0
        frame_results.append({"frame_idx": frame_idx, "face_detected": face_detected, "n_faces": len(faces), "confidence": float(confidence)})
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
        "file": mp4_path.split("/")[-1], "resolution": f"{width}x{height}",
        "total_frames_read": frame_idx, "frames_checked": n_checked,
        "face_detection_rate": round(detection_rate, 4), "avg_confidence": round(avg_confidence, 4),
        "min_confidence": round(min_confidence, 4), "clip_quality_label": clip_quality,
        "low_confidence_frames": [r for r in frame_results if not r["face_detected"]][:20],
    }
