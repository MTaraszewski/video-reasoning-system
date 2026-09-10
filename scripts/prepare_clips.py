"""Trim source footage to the brief's clip length, and scaffold hand-labelling.

The brief asks for *"five to ten clips, each one to three minutes, with the events
you care about labelled by hand with start and end times"*. MEVA's clips are about
five minutes, so they need trimming — which is also the largest cost lever
available: at 4 fps with 12 s windows and 9 s stride, a 300 s clip is ~34 windows
*per description*, and trimming to 120 s cuts model calls by roughly 60 %.

Labelling is human work and this does not pretend otherwise. What it does is make
that work fast and consistent:

- trims to an exact duration, re-encoding so timestamps are honest rather than
  keyframe-approximate
- writes a **contact sheet** — a grid of frames with their times burned in — so a
  whole clip can be scanned at a glance instead of scrubbed
- emits a labels template with the clip already filled in, so only start, end and
  description remain

    python scripts/prepare_clips.py --src /data/meva --out /data/eval --seconds 120
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from video_reasoning.decode import overlay_timestamp, probe, sample_frames


def trim(src: Path, dst: Path, start_s: float, seconds: float) -> None:
    """Trim with re-encoding.

    Stream copy would be faster, but it can only cut at keyframes — so the real
    start drifts from the requested one by up to a keyframe interval. Every label
    would then be offset by an unknown amount, which is precisely the error the
    eval set exists to avoid.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s}", "-i", str(src), "-t", f"{seconds}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an", str(dst),
    ]
    subprocess.run(cmd, check=True)


def contact_sheet(video: Path, out: Path, every_s: float = 5.0,
                  cols: int = 6, thumb_w: int = 320) -> None:
    """A grid of timestamped frames, for scanning a clip without scrubbing it."""
    _, frames = sample_frames(video, fps=1.0 / every_s, max_side=thumb_w,
                              overlay=False)
    if not frames:
        return
    tiles = [overlay_timestamp(f.image, f.t, font_scale=0.09) for f in frames]
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/data/meva")
    ap.add_argument("--out", default="/data/eval")
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="Clip length. The brief wants 60-180.")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--sheet-every", type=float, default=5.0)
    ap.add_argument("--axis", default="real_fixed_camera")
    args = ap.parse_args()

    if not 60 <= args.seconds <= 180:
        print(f"warning: --seconds {args.seconds} is outside the brief's 1-3 minute "
              "range")

    src, out = Path(args.src), Path(args.out)
    sources = sorted([p for p in src.iterdir()
                      if p.suffix.lower() in (".mp4", ".avi")])[: args.limit]
    if not sources:
        print(f"no video files under {src}")
        return

    template: list[dict] = []
    for p in sources:
        meta = probe(p)
        if meta.duration_s < args.seconds:
            print(f"  skip {p.name}: {meta.duration_s}s shorter than requested "
                  f"{args.seconds}s")
            continue
        dst = out / f"{p.stem}.mp4"
        trim(p, dst, args.start, args.seconds)
        after = probe(dst)
        sheet = out / "sheets" / f"{p.stem}.jpg"
        contact_sheet(dst, sheet, every_s=args.sheet_every)
        print(f"  {dst.name:52s} {after.duration_s:6.1f}s  "
              f"{after.width}x{after.height}  sheet -> {sheet.name}")

        # A low-resolution source is kept as its own axis rather than pooled;
        # averaging it with 1080p clips would confound resolution with every
        # other difference between cameras.
        axis = "low_resolution" if after.width < 640 else args.axis
        template.append({
            "video": dst.name,
            "axis": axis,
            "duration_s": after.duration_s,
            "source": p.name,
            "events": [
                {"description": "TODO describe the event as a client would say it",
                 "start_s": 0.0, "end_s": 0.0}
            ],
        })

    labels = out / "labels.template.json"
    labels.write_text(json.dumps(template, indent=2))
    print(f"\nTemplate -> {labels}")
    print(f"Contact sheets -> {out / 'sheets'}")
    print("\nTo label: open each sheet, find the event, read the burned-in times,")
    print("fill in start_s/end_s, and phrase the description the way a CLIENT would")
    print('("a person enters through the door"), never as a dataset class name.')
    print("Then save as labels.json — the template is not read by the eval.")


if __name__ == "__main__":
    main()
