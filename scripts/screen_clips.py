"""Screen footage for activity before investing any effort in it.

Written after two rounds of wasted work. The first MEVA fetch downloaded 1.5 GB of
clips selected by filename duration and returned twelve camera views of a static
car park — two minutes each of nothing, with no event to label. The second returned
clips with the answers burned into the picture. Both were discovered only after
downloading, trimming, and rendering contact sheets by hand.

This asks one objective question first: **does anything happen in this clip?**

Method: the fraction of pixels changing between consecutive samples, beyond a noise
floor. No filenames, no annotations, no interpretation of what is in the frame.
Calibrated against clips whose content is known by construction:

    synthetic negative (nothing happens)      0.77 %
    static MEVA car park                      0.00 - 0.08 %
    synthetic box crossing (known event)      4.41 %
    MEVA enter-vehicle                        2.78 %
    MEVA embrace-interaction                 11.29 %

So roughly: below 0.5 % is static, above 1.5 % has real activity, and the band
between is worth looking at.

WHAT THIS CANNOT DO, measured against clips whose content is known exactly:

    box-crossing    4.41%  active   correct
    with-distractor 5.18%  active   correct
    long-event      1.07%  FAINT    wrong — a slow 27s traverse reads as marginal
    short-event     0.07%  STATIC   wrong — 0.15s events fall between samples
    negative        3.42%  ACTIVE   wrong — something moves, but no labelled event
    machine-stop    1.66%  active   misleading — "active" because the wheel spins
                                    OUTSIDE the event; the event is the stillness

So three blind spots, all real:

1. **Slow events read faint.** Motion is measured per-frame-pair, so a gradual
   change never produces a large single-step difference.
2. **Short events read static** at low sampling rates, exactly as they fall between
   the model's own frames. Raise --fps when hunting brief events.
3. **"Active" does not mean "contains the event you want."** A busy clip where your
   query never happens scores high. That is a NEGATIVE, and valuable, but the screen
   cannot tell the two apart.

It says only THAT something moves, never WHAT. Use it to discard footage that is
completely dead — which is what it caught, twice — not to decide what to label.

    make screen-clips DIR=data/meva-annotated
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from video_reasoning.decode import probe, sample_frames

STATIC = 0.5      # % of pixels moving, below which a clip is effectively still
ACTIVE = 1.5      # above which there is real activity
NOISE = 25        # per-pixel intensity change counted as motion


def measure(path: Path, fps: float, seconds: float) -> dict:
    meta = probe(path)
    _, frames = sample_frames(str(path), fps=fps, max_side=320, overlay=False,
                              end_s=min(seconds, meta.duration_s))
    if len(frames) < 2:
        return {"error": "too few frames"}
    arrs = [np.asarray(f.image.convert("L"), dtype=np.int16) for f in frames]
    moved = [float((np.abs(b - a) > NOISE).mean() * 100) for a, b in zip(arrs, arrs[1:])]
    # Peak matters more than mean: an event occupying ten seconds of a two-minute
    # clip barely moves the average, but it is exactly what we are looking for.
    peak = max(moved)
    return {
        "duration_s": meta.duration_s,
        "resolution": f"{meta.width}x{meta.height}",
        "peak_moved_pct": round(peak, 3),
        "mean_moved_pct": round(float(np.mean(moved)), 3),
        "active_fraction": round(sum(m > ACTIVE for m in moved) / len(moved), 3),
        "verdict": "static" if peak < STATIC else ("active" if peak > ACTIVE else "faint"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="How much of each clip to sample.")
    args = ap.parse_args()

    d = Path(args.dir)
    clips = sorted(p for p in d.iterdir()
                   if p.suffix.lower() in (".mp4", ".avi") and not p.name.startswith("sheet"))
    if not clips:
        print(f"no video files under {d}")
        return 1

    print(f"Screening {len(clips)} clip(s) in {d}  "
          f"(static < {STATIC}%, active > {ACTIVE}%)\n")
    print(f"  {'clip':<50} {'dur':>7} {'res':>10} {'peak':>7} {'busy':>6}  verdict")
    counts = {"static": 0, "faint": 0, "active": 0}
    for p in clips:
        m = measure(p, args.fps, args.seconds)
        if "error" in m:
            print(f"  {p.name[:50]:<50} {m['error']}")
            continue
        counts[m["verdict"]] += 1
        print(f"  {p.name[:50]:<50} {m['duration_s']:>6.0f}s {m['resolution']:>10} "
              f"{m['peak_moved_pct']:>6.2f}% {m['active_fraction']:>5.0%}  {m['verdict']}")

    print(f"\n  active {counts['active']}   faint {counts['faint']}   "
          f"static {counts['static']}")
    if counts["active"] == 0:
        print("\n  NOTHING HAPPENS IN ANY OF THESE. They are usable as negatives —")
        print("  clips where a query genuinely does not occur, which is how false")
        print("  positives get measured — but not as clips to label events in.")
        return 1
    print("\n  Label the 'active' clips. 'static' ones are negatives, and worth")
    print("  keeping: recall alone flatters a system that reports events everywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
