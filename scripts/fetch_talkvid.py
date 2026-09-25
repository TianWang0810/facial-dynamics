"""
Sample and download TalkVid clips, WITH audio, into a speaker-grouped layout.

TalkVid distributes metadata only (HuggingFace FreedomIntelligence/TalkVid,
data/filtered_video_clips.json, CC BY-NC 4.0 -- non-commercial research use);
the clips themselves are cut from YouTube with yt-dlp. This script is the record
of which clips entered our data and how they were fetched, so it is deterministic
given --seed and the metadata file.

Two lessons from docs/engineering_log/local_validation_day1.md are built in:

    * records are grouped by source video, so sampling takes ONE clip per
      speaker (info."Person ID") after a seeded shuffle -- never "the first N";
    * HLS (m3u8) formats stall indefinitely, so both streams must be https.
      The earlier fix constrained only one selector and silently picked
      YouTube's video-only DASH format (137): the first validation clip has no
      audio stream at all. Here video and audio are selected SEPARATELY, each
      restricted to https, and merged.

Cuts use --download-sections without re-encoding, so a clip may start on the
keyframe before the requested time and its audio and video streams may start at
different container times. That is expected and is measured, not corrected, by
the Observation layer (per-stream start times, A/V offset).

Layout: <output>/<person_id>/<video_id>__<start>_<end>.mp4, so
run_pipeline.py --speaker_from parent_dir yields speaker-disjoint splits.

Usage (on a compute node, with the download env that provides yt-dlp + ffmpeg):
    python scripts/fetch_talkvid.py sample --metadata filtered_video_clips.json \
        --n 2200 --seed 0 --output selection.json
    python scripts/fetch_talkvid.py download --selection selection.json \
        --output /scratch/$USER/talkvid/clips --shard 0 --n_shards 20 --workers 4
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

FORMAT = ("bestvideo[ext=mp4][protocol^=https][height<=1080]"
          "+bestaudio[ext=m4a][protocol^=https]")
MAX_FPS = 30.5
ATTEMPTS = 3


def sample(args) -> None:
    with open(args.metadata, encoding="utf-8") as handle:
        records = json.load(handle)
    eligible = [r for r in records
                if args.min_seconds <= r["end-time"] - r["start-time"] <= args.max_seconds
                and float(r.get("fps") or 0) <= MAX_FPS
                and r.get("info", {}).get("Person ID")]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    chosen, seen = [], set()
    for record in eligible:
        person = record["info"]["Person ID"]
        if person in seen:
            continue
        seen.add(person)
        chosen.append({
            "person_id": person,
            "video_link": record["info"]["Video Link"],
            "clip_name": record["id"],
            "start": record["start-time"], "end": record["end-time"],
            "fps_declared": record.get("fps"),
            "language": record["info"].get("Language"),
            "gender": record["info"].get("Gender"),
            "age_group": record["info"].get("Age Group"),
            "ethnicity": record["info"].get("Ethnicity"),
            "dover_score": record.get("dover_scores"),
        })
        if len(chosen) >= args.n:
            break
    summary = {
        "source": os.path.abspath(args.metadata), "seed": args.seed, "n_requested": args.n,
        "n_records": len(records), "n_eligible": len(eligible), "n_selected": len(chosen),
        "filters": {"seconds": [args.min_seconds, args.max_seconds], "max_fps": MAX_FPS,
                    "one_clip_per": "info.Person ID"},
        "hours_selected": round(sum(c["end"] - c["start"] for c in chosen) / 3600, 2),
        "clips": chosen,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    print(f"selected {len(chosen)} clips ({summary['hours_selected']} h) from {len(eligible)} eligible")


def video_id(link: str) -> str:
    """YouTube id from a watch?v=, youtu.be/ or shorts/ link, query string dropped."""
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", link)
    if not match:
        raise ValueError("no YouTube video id in %r" % link)
    return match.group(1)


def _target(output: str, clip: dict) -> str:
    video_id_ = video_id(clip["video_link"])
    name = "%s__%.3f_%.3f.mp4" % (video_id_, clip["start"], clip["end"])
    return os.path.join(output, str(clip["person_id"]), name)


def _download(clip: dict, output: str, yt_dlp: str) -> dict:
    target = _target(output, clip)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return {"person_id": clip["person_id"], "status": "exists", "path": target}
    os.makedirs(os.path.dirname(target), exist_ok=True)
    command = [yt_dlp, "--quiet", "--no-warnings", "--no-playlist",
               "-f", FORMAT, "--merge-output-format", "mp4",
               "--download-sections", "*%.3f-%.3f" % (clip["start"], clip["end"]),
               "-o", target, clip["video_link"]]
    started = time.time()
    # "ffmpeg exited with code 8" is a transient stream error on YouTube's side:
    # the pilot's failures all succeeded on a plain retry. Dead links fail fast
    # and identically every attempt, so retrying costs them little.
    for attempt in range(1, ATTEMPTS + 1):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
            error = (completed.stderr.strip().splitlines() or ["?"])[-1][:300]
        except subprocess.TimeoutExpired:
            completed, error = None, "timeout after 600 s"
        ok = completed is not None and completed.returncode == 0 and os.path.exists(target) and os.path.getsize(target) > 0
        if ok or "Video unavailable" in error or "Private video" in error:
            break
        time.sleep(5 * attempt)
    return {"person_id": clip["person_id"], "status": "ok" if ok else "failed", "path": target,
            "attempts": attempt, "seconds": round(time.time() - started, 1), "error": None if ok else error}


def download(args) -> None:
    with open(args.selection, encoding="utf-8") as handle:
        clips = json.load(handle)["clips"]
    shard = clips[args.shard::args.n_shards]
    if args.limit:
        shard = shard[:args.limit]
    yt_dlp = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    os.makedirs(os.path.join(args.output, "_logs"), exist_ok=True)
    log_path = os.path.join(args.output, "_logs", "shard_%03d_of_%03d.jsonl" % (args.shard, args.n_shards))
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(log_path, "a", encoding="utf-8") as log:
        for result in pool.map(lambda c: _download(c, args.output, yt_dlp), shard):
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            print(result["status"], result["person_id"], result.get("error") or "")
    print(f"shard {args.shard}/{args.n_shards}: log {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--metadata", required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min_seconds", type=float, default=8.0)
    s.add_argument("--max_seconds", type=float, default=30.0)
    s.add_argument("--output", required=True)
    d = sub.add_parser("download")
    d.add_argument("--selection", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--shard", type=int, default=0)
    d.add_argument("--n_shards", type=int, default=1)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    sample(args) if args.command == "sample" else download(args)


if __name__ == "__main__":
    main()
