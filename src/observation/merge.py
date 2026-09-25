"""
Build video_metadata.json and quality.json from the Observation layer's timeline
analysis.

This layer answers exactly one question: is the clip's TIME AXIS trustworthy?
Whether a face is visible is a separate question, and one this layer is not
equipped to answer -- see docs/decisions/observation_timeline_only.md. The visual
verdict comes from the Geometry layer's MediaPipe tracking rate, which is
measured by the same model that produces the features.

video_metadata.json carries the complete per-frame PTS array (pts_sec) so that
downstream layers have real timestamps without re-decoding the source mp4; see
docs/engineering_log/video_metadata_pts_retention.md. quality.json deliberately
keeps only the aggregate statistics and must stay lightweight -- both records are
built field by field below, so the array never leaks into the quality report.
"""

# A clip whose timeline has no issues. Anything else needs a human decision
# before it is used, hence "needs_review" rather than "reject": a timeline defect
# is a fact about the file, not a policy verdict about whether to keep it.
TIMELINE_USABLE = "usable"
TIMELINE_NEEDS_REVIEW = "needs_review"


def merge_results(pts_results: list) -> tuple:
    video_metadata_list = []
    quality_list = []

    for pts in pts_results:
        filename = pts["file"]
        video_metadata_list.append({
            "file": filename, "resolution": pts.get("resolution"), "fps_declared": pts.get("fps_declared"),
            "fps_from_pts": pts.get("fps_from_pts"),
            "n_frames_decoded": pts.get("n_frames_decoded"),
            "n_missing_pts": pts.get("n_missing_pts"),
            "duration_from_pts_sec": pts.get("duration_from_pts_sec"),
            "first_pts_sec": pts.get("first_pts_sec"), "last_pts_sec": pts.get("last_pts_sec"),
            "pts_sec": pts.get("pts_sec", []),
            "relative_path": pts.get("relative_path"),
            "streams": pts.get("streams"),
        })

        pts_issues = pts.get("issues", [])
        pts_ok = pts.get("is_strictly_monotonic", False) and len(pts_issues) == 0
        reasons = ["timeline:%s" % issue["code"] for issue in pts_issues]
        if not pts.get("is_strictly_monotonic", False) and not pts_issues:
            reasons.append("timeline:not_strictly_monotonic")

        quality_list.append({
            "file": filename,
            "pts_completeness": {"is_strictly_monotonic": pts.get("is_strictly_monotonic"),
                                   "n_missing_pts": pts.get("n_missing_pts"),
                                   "max_frame_gap_sec": pts.get("max_frame_gap_sec"),
                                   "min_frame_gap_sec": pts.get("min_frame_gap_sec"),
                                   "large_gaps": pts.get("large_gaps", []),
                                   "issues": pts_issues},
            "timeline_label": TIMELINE_USABLE if pts_ok else TIMELINE_NEEDS_REVIEW,
            "reject_reasons": reasons,
            "has_audio": (pts.get("streams") or {}).get("has_audio"),
            "visual_quality": None,
            "visual_quality_note": "not assessed here; the Geometry layer's MediaPipe tracking "
                                   "rate is the visual verdict (geometry_quality.json)",
        })
    return video_metadata_list, quality_list
