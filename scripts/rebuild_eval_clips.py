"""Rebuild the labelled eval clips from public sources — no AWS account, no CLI.

The clips are not in the repository: 400 MB of someone else's dataset does not
belong in a git history, and shipping it would ask a reviewer to trust our copy.
What IS in the repository is `labels.json`, and each entry carries the source file,
the trim offset and the duration — enough to rebuild the eval set exactly.

**Everything here runs in the container over plain HTTPS.** MEVA's bucket is public,
so `https://mevadata-public-01.s3.amazonaws.com/...` needs no credentials and no
`aws` binary on the host. An earlier version shelled out to `aws s3 cp`, which
quietly made the AWS CLI a prerequisite for reproducing a single number — against a
brief that asks for a run on the reviewer's machine without reading the code first.

Two path facts, both learned by having URLs 404:

1. Video filenames carry a release suffix the annotation names do not
   (`...G329.r13.avi` against `...G329.activities.yml`).
2. A clip crossing an hour boundary is filed under its **end** hour:
   `.../2018-03-15/15/2018-03-15.14-55-00.15-00-00.school.G421.r13.avi`

    python scripts/rebuild_eval_clips.py --labels /data/eval/labels.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from video_reasoning.decode import probe

BASE = "https://mevadata-public-01.s3.amazonaws.com"
DROP = "drops-123-r13"


def candidate_urls(fname: str, base: str = BASE, drop: str = DROP) -> list[str]:
    """Where a source clip might live. End hour first — see module docstring."""
    parts = fname.rsplit(".", 1)[0].split(".")
    day, h_start, h_end = parts[0], parts[1][:2], parts[2][:2]
    hours = [h_end] if h_end == h_start else [h_end, h_start]
    return [f"{base}/{drop}/{day}/{h}/{fname}" for h in hours]


def download(url: str, dst: Path, chunk: int = 1 << 20) -> int:
    """Stream to a .part file, then rename. Returns bytes written.

    Writing straight to the final path leaves a truncated file looking exactly like
    a complete one if the transfer dies — and the next run would trim it happily.
    """
    part = dst.with_suffix(dst.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        total, got = int(r.headers.get("Content-Length", 0)), 0
        with open(part, "wb") as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if total:
                    print(f"\r      {got / 1048576:6.0f} / {total / 1048576:.0f} MB"
                          f"  ({100 * got / total:3.0f}%)", end="", flush=True)
        print("\r" + " " * 44 + "\r", end="")
    # A short read is a failure even though nothing raised.
    if total and got != total:
        part.unlink(missing_ok=True)
        raise OSError(f"truncated: {got} of {total} bytes")
    part.rename(dst)
    return got


def fetch_source(fname: str, out: Path) -> Path | None:
    dst = out / fname
    if dst.exists():
        return dst
    for url in candidate_urls(fname):
        try:
            n = download(url, dst)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            continue
        print(f"      fetched {n / 1048576:.0f} MB")
        return dst
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="/data/eval/labels.json")
    ap.add_argument("--src", default="/data/meva-annotated")
    ap.add_argument("--out", default="/data/eval")
    ap.add_argument("--force", action="store_true",
                    help="Re-cut clips that already exist at the right duration.")
    ap.add_argument("--keep-sources", action="store_true",
                    help="Keep the 5-minute source files after trimming. Off by "
                         "default: ~110 MB each, and nothing reads them again once "
                         "the clip is cut.")
    args = ap.parse_args()

    entries = json.loads(Path(args.labels).read_text())
    src, out = Path(args.src), Path(args.out)
    built = kept = 0
    problems: list[str] = []

    for e in entries:
        dst, want = out / e["video"], float(e["duration_s"])

        if dst.exists() and not args.force:
            got = probe(dst).duration_s
            if abs(got - want) < 0.5:
                print(f"  have  {e['video'][-42:]:<42} {got:6.1f}s")
                kept += 1
                continue

        print(f"  build {e['video'][-42:]:<42} from {e['source'][-30:]}")
        source = fetch_source(e["source"], src)
        if source is None:
            problems.append(f"{e['video']}: could not fetch {e['source']}")
            print("      FAILED to fetch source")
            continue

        # Re-encode, never stream copy: a stream copy cuts only at keyframes, so
        # the real start drifts by up to a keyframe interval and every label
        # silently shifts with it.
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
        print(f"      {got:.1f}s, cut at {e['trimmed_from_s']:.1f}s")
        built += 1
        if not args.keep_sources:
            source.unlink(missing_ok=True)

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
