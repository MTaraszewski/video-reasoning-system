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
  a verdict remain

**It merges; it never overwrites.** An earlier version rewrote the template
wholesale on every run. That was harmless only until the first hand-labelling
session: after that, re-running it for any reason — one new clip, a changed trim
length — would silently discard every human judgement in the file, and `git status`
would report nothing wrong. Rows for clips already in the template are now carried
through untouched, and new clips are appended.

    python scripts/prepare_clips.py --src /data/meva-annotated --out /data/eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from video_reasoning.decode import overlay_timestamp, probe, sample_frames

# What a human has decided about a candidate. Nothing is ever deleted from the
# template: a rejection is a finding, and one that vanishes cannot be reported.
#   pending       nobody has looked yet
#   confirmed     event is there, boundaries checked -> the eval set
#   suspect       something is there but cannot be pinned -> reported separately
#   unlabellable  no human can confirm this from the footage -> no ground truth,
#                 used only to ask whether the model claims to see it anyway
STATUSES = ("pending", "confirmed", "suspect", "unlabellable")


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


def scaffold(events: list[dict]) -> list[dict]:
    """Give a plan's events the fields a human verdict needs, without opinions."""
    out = []
    for e in events:
        out.append({
            "description": e["description"],
            "start_s": e["start_s"],
            "end_s": e["end_s"],
            "meva_activity": e.get("meva_activity", ""),
            # Carried from the plan so the labeller sees the warnings it raised.
            "span_suspect": e.get("span_suspect", False),
            "phrasing_mapped": e.get("phrasing_mapped", True),
            "status": "pending",
            "confirmed_by_hand": False,
            # Separate from the times on purpose. MEVA's class name is mapped to
            # client language by this repo's PHRASING table, and that mapping can
            # assert things the camera never shows — it called a stairwell with no
            # visible door "a person enters through the door". The times can be
            # right while the words are wrong, so they are judged separately.
            "description_matches_footage": None,
            "note": "",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/data/meva-annotated")
    ap.add_argument("--out", default="/data/eval")
    ap.add_argument("--plan", default=None,
                    help="labels.candidate.json from meva-plan. When present, its "
                         "computed trim windows are used instead of --start.")
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="Clip length. The brief wants 60-180.")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--sheet-every", type=float, default=5.0)
    ap.add_argument("--axis", default="real_fixed_camera")
    ap.add_argument("--force", action="store_true",
                    help="Allow --retrim to discard existing hand judgements.")
    ap.add_argument("--retrim", action="store_true",
                    help="Re-encode clips that already exist. Off by default: a "
                         "re-trim with different settings would invalidate every "
                         "hand label already made against that clip.")
    args = ap.parse_args()

    if not 60 <= args.seconds <= 180:
        print(f"warning: --seconds {args.seconds} is outside the brief's 1-3 minute "
              "range")

    src, out = Path(args.src), Path(args.out)

    # A trim plan, if one exists, decides WHERE to cut. Cutting from zero would
    # silently drop events: one clip declares activity at 45s, 64s, 74s and 268s,
    # and a naive 120s window from the start loses the last one with no trace.
    plan_path = Path(args.plan) if args.plan else src / "labels.candidate.json"
    plan: dict[str, dict] = {}
    if plan_path.exists():
        for entry in json.loads(plan_path.read_text()):
            plan[entry["source"]] = entry
        print(f"using trim plan from {plan_path.name} "
              f"({len(plan)} clip(s), "
              f"{sum(len(e['events']) for e in plan.values())} located event(s))")
    else:
        print(f"no trim plan at {plan_path} — cutting from {args.start}s.\n"
              f"  Run `make meva-plan` first to place windows around real events.")

    # Anything already judged is authoritative and is never regenerated.
    labels = out / "labels.template.json"
    existing: dict[str, dict] = {}
    if labels.exists():
        for entry in json.loads(labels.read_text()):
            existing[entry["video"]] = entry
        judged = sum(1 for e in existing.values() for ev in e["events"]
                     if ev.get("status", "pending") != "pending")
        print(f"merging into {labels.name}: {len(existing)} clip(s) already present, "
              f"{judged} event(s) already judged")
    print()

    all_sources = sorted([p for p in src.iterdir()
                          if p.suffix.lower() in (".mp4", ".avi")])
    sources = all_sources[: args.limit]
    # Say so. Sorted order means a low limit drops the LAST names alphabetically,
    # which is exactly where freshly fetched clips land, and the run otherwise
    # looks like a clean success that simply found nothing new.
    if len(all_sources) > len(sources):
        print(f"warning: --limit {args.limit} truncates {len(all_sources)} source(s); "
              f"{len(all_sources) - len(sources)} not processed, starting with "
              f"{all_sources[len(sources)].name}")
    if not sources:
        print(f"no video files under {src}")
        return

    kept = added = 0
    for p in sources:
        dst = out / f"{p.stem}.mp4"
        prior = existing.get(dst.name)

        # --retrim re-encodes, which shifts every frame relative to the labels
        # already made against the old encode. Refusing here rather than warning:
        # a warning scrolls past, and the damage is silent and unrecoverable.
        if prior is not None and args.retrim and not args.force:
            n_j = sum(1 for ev in prior["events"]
                      if ev.get("status", "pending") != "pending")
            if n_j:
                print(f"  {dst.name[:44]:44s} REFUSED re-trim: {n_j} judged "
                      f"event(s) would be invalidated. --force to override")
                kept += 1
                continue

        if prior is not None and not args.retrim:
            n_j = sum(1 for ev in prior["events"]
                      if ev.get("status", "pending") != "pending")
            # Any human judgement makes the row authoritative: carried verbatim.
            if n_j:
                print(f"  {dst.name[:44]:44s} kept    "
                      f"{len(prior['events'])} event(s), {n_j} judged")
                kept += 1
                continue
            # Nothing judged yet, so nothing to lose -- refresh the events from
            # the plan so upstream fixes actually reach the labeller. Without
            # this, a clip scaffolded before a phrasing fix keeps the bad wording
            # forever, because "merge" would protect rows nobody had looked at.
            entry = plan.get(p.name)
            if entry:
                prior["events"] = scaffold(entry["events"])
                existing[dst.name] = prior
                print(f"  {dst.name[:44]:44s} refresh {len(prior['events'])} "
                      f"candidate(s), none judged")
                kept += 1
                continue
            print(f"  {dst.name[:44]:44s} kept    "
                  f"{len(prior['events'])} event(s), 0 judged (no plan entry)")
            kept += 1
            continue

        meta = probe(p)
        if meta.duration_s < args.seconds:
            print(f"  skip {p.name}: {meta.duration_s}s shorter than requested "
                  f"{args.seconds}s")
            continue
        entry = plan.get(p.name)
        start = entry["trim"]["start_s"] if entry else args.start
        trim(p, dst, start, args.seconds)
        after = probe(dst)
        contact_sheet(dst, out / "sheets" / f"{p.stem}.jpg", every_s=args.sheet_every)

        # A low-resolution source is kept as its own axis rather than pooled;
        # averaging it with 1080p clips would confound resolution with every
        # other difference between cameras.
        axis = "low_resolution" if after.width < 640 else args.axis
        events = scaffold(entry["events"]) if entry else []
        print(f"  {dst.name[:44]:44s} added   {after.duration_s:6.1f}s  "
              f"{after.width}x{after.height:<5} cut@{start:6.1f}s  "
              f"{len(events)} candidate(s)")
        existing[dst.name] = {
            "video": dst.name,
            "axis": axis,
            "duration_s": after.duration_s,
            "source": p.name,
            "trimmed_from_s": start,
            "events": events,
        }
        added += 1

    ordered = sorted(existing.values(), key=lambda e: e["video"])
    labels.write_text(json.dumps(ordered, indent=2))

    n_ev = sum(len(e["events"]) for e in ordered)
    by_status: dict[str, int] = {}
    for e in ordered:
        for ev in e["events"]:
            s = ev.get("status", "pending")
            by_status[s] = by_status.get(s, 0) + 1

    print(f"\n{len(ordered)} clip(s) ({kept} kept, {added} added), "
          f"{n_ev} candidate event(s)")
    print("  " + "  ".join(f"{k}: {v}" for k, v in sorted(by_status.items())))
    print(f"\nTemplate -> {labels}")
    print(f"Contact sheets -> {out / 'sheets'}   (per-event: `make event-sheets`)")
    print("\nTo label: open each event sheet, set status to one of "
          f"{'/'.join(STATUSES[1:])},")
    print("correct start_s/end_s, set description_matches_footage, and say why in")
    print("note. Then save as labels.json — the template is not read by the eval.")


if __name__ == "__main__":
    main()
