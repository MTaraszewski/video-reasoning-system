"""One dense contact sheet per candidate event, for confirming it by eye.

A whole-clip sheet at 5s intervals cannot confirm a 1-7s event: the event occupies
one or two tiles, and in wide fixed-camera footage the subject is a few dozen
pixels across. Reading a boundary to +/-1s off that is not possible, and pretending
otherwise would produce labels that LOOK hand-made while really being MEVA's
annotations passed through unchecked — the exact failure the labelling step exists
to prevent.

So each candidate event gets its own sheet: the declared window plus a few seconds
of lead-in and lead-out, sampled sub-second, at a size where a person is visible.

    python scripts/event_sheets.py --labels /data/eval/labels.template.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from video_reasoning.decode import overlay_timestamp, sample_frames

# How tall the actor should be ON THE SHEET, in pixels. Below roughly this, a
# human stops being able to judge what an actor is doing — G329's door was missed
# at ~86px and only settled by zooming into the source. Tiles are therefore sized
# from the actor's measured height in `screen.json`, not fixed: a 694px actor and
# a 232px actor need very different downscales to stay legible.
TARGET_ACTOR_PX = 150
MIN_TILE_W, MAX_TILE_W = 480, 1280


def tile_width(actor_h: float | None, source_w: int = 1920,
               source_h: int = 1080) -> int:
    """Tile width that renders `actor_h` at about TARGET_ACTOR_PX."""
    if not actor_h:
        return 720  # unknown actor size: err large, a big tile is never worse
    scale = TARGET_ACTOR_PX / float(actor_h)
    return max(MIN_TILE_W, min(MAX_TILE_W, round(source_w * scale)))


def actor_sizes(screen_path: Path) -> dict[tuple[str, str], int]:
    """(clip stem, MEVA activity) -> median actor height in source pixels."""
    if not screen_path.exists():
        return {}
    out: dict[tuple[str, str], int] = {}
    for row in json.loads(screen_path.read_text()):
        for ev in row.get("per_event", []):
            out[(row["clip"], ev["activity"])] = ev["median_actor_h"]
    return out


def event_sheet(video: Path, start_s: float, end_s: float, out: Path,
                pad_s: float = 4.0, fps: float = 2.0, cols: int = 6,
                width: int = 560) -> tuple[int, float, float]:
    """Render the window plus padding. Returns (frames, lo, hi)."""
    lo, hi = max(0.0, start_s - pad_s), end_s + pad_s
    _, frames = sample_frames(video, fps=fps, max_side=width, overlay=False,
                              start_s=lo, end_s=hi)
    if not frames:
        return 0, lo, hi

    tiles = []
    for f in frames:
        img = overlay_timestamp(f.image, f.t, font_scale=0.07)
        # Frames INSIDE the declared window get a full green BORDER, so lead-in
        # and event are distinguishable at a glance without reading every
        # timestamp. A border, not a top rule: a rule on one edge sits flush
        # against the neighbouring tile, so a marked row reads as an underline
        # of the row above it and every boundary judgement comes out shifted by
        # one row. A closed outline belongs to exactly one tile.
        # If the action visibly starts before or after the green run, the
        # candidate boundary is wrong and that is the whole point of looking.
        if start_s <= f.t <= end_s:
            ImageDraw.Draw(img).rectangle(
                [0, 0, img.size[0] - 1, img.size[1] - 1],
                outline=(40, 210, 90), width=6)
        tiles.append(img)

    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (18, 18, 18))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return len(tiles), lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="/data/eval/labels.template.json")
    ap.add_argument("--data-dir", default="/data/eval")
    ap.add_argument("--out", default="/data/eval/event-sheets")
    ap.add_argument("--pad", type=float, default=4.0,
                    help="Seconds of lead-in and lead-out around the window.")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=0,
                    help="Fixed tile width. 0 (default) sizes each sheet from the "
                         "actor's measured height so it renders ~150px tall.")
    ap.add_argument("--screen", default="/data/meva-index/screen.json")
    args = ap.parse_args()

    entries = json.loads(Path(args.labels).read_text())
    data_dir, out_dir = Path(args.data_dir), Path(args.out)
    sizes = actor_sizes(Path(args.screen))
    if sizes:
        print(f"actor sizes for {len(sizes)} event(s) from {Path(args.screen).name}\n")
    n = 0
    for entry in entries:
        video = data_dir / entry["video"]
        if not video.exists():
            print(f"  missing {video.name}")
            continue
        # A short, sortable id per event, so a correction can name the sheet it
        # came from: scene + camera + index.
        # Date AND camera. Two clips from one camera on different days produced
        # sheets called `G300.r13__00__...` apiece, distinguishable only by the
        # description that happened to differ -- unreadable, and one rename away
        # from silently overwriting each other.
        parts = entry["video"].split(".")
        day = parts[0][5:] if len(parts) > 1 else ""      # 03-13
        scene = f"{day}_{'.'.join(parts[-3:-1])}" if len(parts) >= 3 else entry["video"][:12]
        # screen.json keys on the source stem, which drops the release suffix.
        stem_key = entry["video"].rsplit(".", 1)[0]
        for suffix in (".r13", ""):
            if stem_key.endswith(suffix) and suffix:
                stem_key = stem_key[: -len(suffix)]
                break
        for i, ev in enumerate(entry.get("events", [])):
            slug = ev["description"][:34].replace(" ", "-")
            name = f"{scene}__{i:02d}__{slug}.jpg"
            actor_h = sizes.get((stem_key, ev.get("meva_activity", "")))
            w = args.width or tile_width(actor_h)
            # Fewer columns once tiles are large, so a sheet stays a shape a
            # screen can actually show.
            cols = 6 if w <= 640 else (5 if w <= 900 else 4)
            got, lo, hi = event_sheet(
                video, ev["start_s"], ev["end_s"], out_dir / name,
                pad_s=args.pad, fps=args.fps, cols=cols, width=w)
            n += 1
            ah = f"{actor_h}px" if actor_h else "actor ?"
            print(f"  {name[:52]:<52} {ev['start_s']:>6.1f}-{ev['end_s']:<6.1f}"
                  f" {got:>3}f  {ah:>8} -> tile {w}")

    print(f"\n{n} event sheet(s) -> {out_dir}")
    print("Green BORDER round a tile = that frame is inside the candidate window.")
    print("For each: is the event really there, and does the action line up with")
    print("the green run? Correct start_s/end_s, or delete the event entirely.")


if __name__ == "__main__":
    main()
