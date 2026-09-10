"""Rebuild the labelled eval clips from their sources, exactly as first cut.

`labels.json` carries, per clip, the source file, the trim offset and the duration.
Re-cutting with the same settings reproduces the clip the labels were made against,
so a reviewer can verify every reported number without trusting our copy of the
footage -- and without us redistributing 400 MB of someone else's dataset.

Re-encoding, not stream copy: a stream copy can only cut at keyframes, so the real
start drifts by up to a keyframe interval and every label silently shifts with it.

    python scripts/rebuild_eval_clips.py --labels /data/eval/labels.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from video_reasoning.decode import probe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="/data/eval/labels.json")
    ap.add_argument("--src", default="/data/meva-annotated")
    ap.add_argument("--out", default="/data/eval")
    ap.add_argument("--force", action="store_true",
                    help="Re-cut clips that already exist at the right duration.")
    args = ap.parse_args()

    entries = json.loads(Path(args.labels).read_text())
    src, out = Path(args.src), Path(args.out)
    built = kept = 0
    problems: list[str] = []

    for e in entries:
        dst, source = out / e["video"], src / e["source"]
        want = float(e["duration_s"])

        if dst.exists() and not args.force:
            got = probe(dst).duration_s
            if abs(got - want) < 0.5:
                print(f"  have  {e['video'][-40:]:<40} {got:6.1f}s")
                kept += 1
                continue

        if not source.exists():
            problems.append(f"{e['video']}: source {e['source']} not found under {src}")
            print(f"  MISS  {e['video'][-40:]:<40} needs {e['source']}")
            continue

        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(e["trimmed_from_s"]), "-i", str(source), "-t", str(want),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an", str(dst),
        ], check=True)
        got = probe(dst).duration_s
        # A clip whose duration drifted is a clip whose labels no longer line up.
        # Say so here rather than let it surface as a mysteriously poor tIoU.
        if abs(got - want) > 0.5:
            problems.append(f"{e['video']}: rebuilt to {got}s, labels assume {want}s")
        print(f"  built {e['video'][-40:]:<40} {got:6.1f}s  cut@{e['trimmed_from_s']:.1f}s")
        built += 1

    n_ev = sum(len(e["events"]) for e in entries)
    print(f"\n{len(entries)} clip(s): {kept} already correct, {built} rebuilt, "
          f"{n_ev} hand-labelled event(s)")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print("Eval set ready.  make eval")


if __name__ == "__main__":
    main()
