"""
Merge PTS and face-detection results into video_metadata.json and quality.json,
matching the schema defined in design doc sections 1.1/1.2.
"""


def merge_results(pts_results: list, face_results: list) -> tuple:
    face_by_file = {r["file"]: r for r in face_results}
    video_metadata_list = []
    quality_list = []

    for pts in pts_results:
        filename = pts["file"]
        face = face_by_file.get(filename, {})
        video_metadata_list.append({
            "file": filename, "resolution": pts.get("resolution"), "fps_declared": pts.get("fps_declared"),
            "n_frames_decoded": pts.get("n_frames_decoded"), "duration_from_pts_sec": pts.get("duration_from_pts"),
            "first_pts_sec": pts.get("first_pts"), "last_pts_sec": pts.get("last_pts"),
        })
        pts_issues = pts.get("issues", [])
        pts_ok = pts.get("is_monotonic", False) and len(pts_issues) == 0
        clip_face_label = face.get("clip_quality_label", "reject")

        if pts_ok and clip_face_label == "usable":
            final_label = "usable"
        elif pts_ok and clip_face_label == "low-quality":
            final_label = "low-quality"
        else:
            final_label = "reject"

        quality_list.append({
            "file": filename,
            "pts_completeness": {"is_monotonic": pts.get("is_monotonic"), "max_frame_gap_sec": pts.get("max_frame_gap"),
                                   "min_frame_gap_sec": pts.get("min_frame_gap"), "issues": pts_issues},
            "frame_level_availability": {"face_detection_rate": face.get("face_detection_rate", 0.0),
                                           "avg_confidence": face.get("avg_confidence"), "min_confidence": face.get("min_confidence")},
            "clip_quality_label": final_label,
        })
    return video_metadata_list, quality_list
