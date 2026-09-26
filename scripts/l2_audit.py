"""
CLI: visual audit of L2 phrases -- eye-region crops from the source video before,
at, and after sampled phrases (or rejected candidates), tiled into one PNG.

Time mapping: a track frame k sits at container time time_sec + k / hz (the
features grid); the video frame shown is the one whose real PTS is nearest, from
video_metadata.json. The crop box comes from a fresh MediaPipe landmark pass on the
shown frame (eye corners 33 / 263), because the stored landmarks are canonicalised.

Usage:
    python scripts/l2_audit.py --run_dir ../runs/<id> --model_path ../models/face_landmarker.task \
        --kind blink --n 8 --output ../runs/<id>/l2/audit_blink.png
    python scripts/l2_audit.py ... --kind too_long      # rejected candidates, by reason
"""
import argparse
import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.pipeline.layout import STAGE_OBSERVATION, RunLayout  # noqa: E402
from src.semantic import l1, l2  # noqa: E402
from src.semantic.l1_io import add_input_arguments, load_inputs  # noqa: E402

MARGIN_SECONDS = 0.25
CROP_W, CROP_H = 320, 110
BANNER_H = 22


def read_frames(path: str, indices: list) -> dict:
    """Decoded frames by decode order, via PyAV like src/observation/decode.py (OpenCV here cannot decode AV1)."""
    import av

    wanted, out = set(indices), {}
    with av.open(path) as container:
        for k, frame in enumerate(container.decode(video=0)):
            if k in wanted:
                out[k] = frame.to_ndarray(format="bgr24")
            if k >= max(wanted):
                break
    return out


CROPS = {"eyes": (33, 263), "brows": (33, 263), "mouth": (61, 291), "face": None}


def crop_box(frame: np.ndarray, landmarker, mode: str):
    """(x0, y0, x1, y1, out_w, out_h) around MediaPipe landmarks, or None when no face is found."""
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    h, w = frame.shape[:2]
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    out_w, out_h = CROP_W, int(CROP_H * 2.5) if mode == "face" else CROP_H
    if CROPS[mode] is None:
        pts = np.array([[p.x * w, p.y * h] for p in lm])
        centre, half_w = (pts.min(0) + pts.max(0)) / 2, 0.75 * (pts[:, 0].max() - pts[:, 0].min())
    else:
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in CROPS[mode]])
        centre, dist = pts.mean(0), np.linalg.norm(pts[1] - pts[0])
        half_w = {"eyes": 0.8, "brows": 0.9, "mouth": 1.1}[mode] * dist
        if mode == "brows":
            centre = centre - np.array([0.0, 0.3 * dist])
    half_h = half_w * out_h / out_w
    return (int(max(0, centre[0] - half_w)), int(max(0, centre[1] - half_h)),
            int(min(w, centre[0] + half_w)), int(min(h, centre[1] + half_h)), out_w, out_h)


def apply_box(frame: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1, out_w, out_h = box
    return cv2.resize(frame[y0:y1, x0:x1], (out_w, out_h))


def describe(phrase: dict) -> str:
    p = phrase["params"]
    extra = [p[k] for k in ("symmetry", "dominant", "direction") if k in p]
    return "%s %s peak=%s drv=%s" % (phrase["type"], "".join(extra), p["peak"], p["driver"].replace("head_rotation", "rot"))


def span_stats(values: dict, rest: dict, t: list, hz: float) -> dict:
    """Eye/gaze/head numbers over a span, printed on each row so a visual label can be related to the text."""
    s, e = l1._frame(t[0], hz), l1._frame(t[1], hz)
    dev = lambda ch: values[ch][0][s:e, 0] - rest[ch]  # noqa: E731
    rot = values[l2.ROTATION_TEXT][0][s:e]
    return {"blink_max": float(max(dev("eyeBlinkLeft").max(), dev("eyeBlinkRight").max())),
            "blink_mean": float((dev("eyeBlinkLeft").mean() + dev("eyeBlinkRight").mean()) / 2),
            "lookdown": float((dev("eyeLookDownLeft").mean() + dev("eyeLookDownRight").mean()) / 2),
            "pitch": float(dev("gaze_pitch").mean()),
            "head_x": float(rot[:, 0].mean() - rest[l2.ROTATION_TEXT]["x"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_input_arguments(parser)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--kind", default="blink", help="a phrase type, or a rejection reason (too_long, too_short, ...)")
    parser.add_argument("--crop", choices=sorted(CROPS), default="eyes")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    npz, names, unit, _ = load_inputs(args)
    hz = float(npz["target_hz"])
    run_dir = os.path.abspath(args.run_dir)
    layout = RunLayout(root=os.path.dirname(run_dir), run_id=os.path.basename(run_dir))
    with open(layout.artifact(STAGE_OBSERVATION, "video_metadata.json"), encoding="utf-8") as handle:
        videos = {v["file"]: v for v in json.load(handle)}
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as handle:
        input_dir = json.load(handle)["input_dir"]

    candidates = []
    for track in l2.stitch_tracks(npz):
        text = l2.track_l1(track, hz, names, unit)
        l2_track, rejected = l2.encode_track(text, names, unit)
        values = l2.channel_values(text, names, unit)
        items = [dict(p, label=describe(p)) for p in l2_track["phrases"] if p["type"] == args.kind]
        items += [dict(r, label="rejected %s %.2fs" % (r["reason"], r["t"][1] - r["t"][0]))
                  for r in rejected if r["reason"] == args.kind]
        for item in items:
            item["stats"] = span_stats(values, l2_track["rest"], item["t"], hz)
        candidates += [(track, item) for item in items]
    if not candidates:
        raise SystemExit("nothing of kind %r" % args.kind)
    picks = np.random.default_rng(args.seed).choice(len(candidates), min(args.n, len(candidates)), replace=False)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.model_path, delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1)
    rows, log = [], []
    crop_blank = np.zeros((int(CROP_H * 2.5) if args.crop == "face" else CROP_H, CROP_W, 3), dtype=np.uint8)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for index in sorted(picks):
            track, item = candidates[index]
            s, e = item["t"]
            peak = s + item["params"]["onset"] if "params" in item else (s + e) / 2
            if "params" in item:
                item["stats"]["driver_dev"] = item["params"]["channel_dev"][item["params"]["driver"]]
            times = [track["time_sec"] + t for t in (s - MARGIN_SECONDS, peak, e + MARGIN_SECONDS)]
            video = videos[track["clip_id"]]
            pts = np.array([np.nan if p is None else p for p in video["pts_sec"]], dtype=np.float64)
            frames_idx = [int(np.nanargmin(np.abs(pts - t))) for t in times]
            frames = read_frames(os.path.join(input_dir, video["relative_path"]), frames_idx)
            # one box for all three frames (from the 'before' frame), so head motion stays visible
            box = crop_box(frames[frames_idx[0]], landmarker, args.crop) if frames_idx[0] in frames else None
            crops = [apply_box(frames[k], box) if (k in frames and box) else np.zeros_like(crop_blank) for k in frames_idx]
            row = np.concatenate(crops, axis=1)
            banner = np.zeros((BANNER_H, row.shape[1], 3), dtype=np.uint8)
            cv2.putText(banner, "#%d %s | %s t=%.2f-%.2f | %s" % (len(rows), item["label"], track["clip_id"][:24], s, e,
                                                               " ".join("%s=%+.2f" % kv for kv in item["stats"].items())),
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            log.append({"row": len(rows), "clip_id": track["clip_id"], "t": item["t"], "label": item["label"], **item["stats"]})
            rows.append(np.concatenate([banner, row]))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, np.concatenate(rows))
    with open(os.path.splitext(args.output)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)
    print("Written: %s (%d of %d %s)" % (args.output, len(rows), len(candidates), args.kind))


if __name__ == "__main__":
    main()
