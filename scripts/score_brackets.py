#!/usr/bin/env python3
"""Score the change signal against the labelled events. No model, no GPU.

This is the decisive cheap test of the architecture's first stage: if a
labelled event does not fall inside any bracket, the observer is never shown it
and nothing downstream can recover it. Running this before spending GPU time is
the point -- the inherited z-score thresholds scored 5/15 here, and finding that
out cost thirty seconds rather than an hour of L4 time.

    python scripts/score_brackets.py                 # score the default config
    python scripts/score_brackets.py --sweep         # recall/coverage frontier

Two numbers matter, and they trade against each other:

    recall    fraction of labelled events some bracket overlaps. The ceiling on
              everything the pipeline can find.
    coverage  fraction of clip time brackets claim. The cost proxy, since the
              observer polls inside brackets and nowhere else.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eventfinder.config import Config, SignalConfig          # noqa: E402
from eventfinder.signal import brackets, change_score, coverage_of  # noqa: E402
import eventfinder.signal as signal_mod                       # noqa: E402

# An event this close to t=0 has almost no "before" to be different from: the
# eval clips were trimmed to place their event near the start, so these are
# reported separately rather than allowed to flatter the headline number.
BOUNDARY_S = 3.2


def load(labels_path: Path, clips_dir: Path):
    out = []
    for clip in json.loads(labels_path.read_text()):
        v = clips_dir / clip["video"]
        if v.exists():
            out.append((clip, str(v)))
        else:
            print(f"  skipped (not present): {clip['video']}", file=sys.stderr)
    return out


def cache_signals(clips, cfg: SignalConfig):
    """Decode once. Thresholds are pure post-processing over the same samples,
    so a sweep must not re-decode -- and must not be able to drift between
    settings either."""
    cache = {}
    for clip, path in clips:
        cache[clip["video"]] = change_score(path, fps=cfg.fps, thumb_px=cfg.thumb_px)
    signal_mod.change_score = lambda v, **k: cache[Path(v).name]
    return cache


def score(clips, cfg: SignalConfig):
    rows, cov, nb = [], [], []
    for clip, path in clips:
        bs = brackets(path, clip["duration_s"], cfg)
        cov.append(coverage_of(bs, clip["duration_s"]))
        nb.append(len(bs))
        for ev in clip["events"]:
            inside = [b for b in bs if b.end_s > ev["start_s"] and b.start_s < ev["end_s"]]
            rows.append(dict(
                video=clip["video"], desc=ev["description"],
                start=ev["start_s"], end=ev["end_s"],
                hit=bool(inside),
                full=any(b.start_s <= ev["start_s"] and b.end_s >= ev["end_s"] for b in bs),
                origins=sorted({b.origin for b in inside}),
                boundary=ev["start_s"] <= BOUNDARY_S,
                n_brackets=len(bs),
            ))
    return rows, sum(cov) / len(cov), sum(nb) / len(nb)


def report(rows, cov, nb, elapsed):
    by_video: dict[str, list] = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)

    for v, rs in by_video.items():
        print(f"\n  {v}   ({rs[0]['n_brackets']} brackets)")
        for r in rs:
            tag = "FULL" if r["full"] else ("part" if r["hit"] else "MISS")
            org = ",".join(r["origins"]) or "-"
            print(f"    {tag:4s} {r['start']:6.1f}-{r['end']:<6.1f} [{org:16s}] {r['desc']}")

    hit = sum(r["hit"] for r in rows)
    full = sum(r["full"] for r in rows)
    print(f"\n  bracket recall     {hit}/{len(rows)} = {hit / len(rows):.1%}")
    print(f"  fully contained    {full}/{len(rows)} = {full / len(rows):.1%}")
    print(f"  mean coverage      {cov:.1%}   ({nb:.1f} brackets per clip)")
    print(f"  signal wall time   {elapsed:.1f}s for {len(by_video)} clips")

    # The split that says whether the detector works or the sentinels are
    # carrying it. These are different systems with different failure modes.
    for name, want in (("at the clip boundary", True), ("elsewhere in the clip", False)):
        grp = [r for r in rows if r["boundary"] is want]
        if not grp:
            continue
        h = sum(r["hit"] for r in grp)
        ris = sum("rising" in r["origins"] or "falling" in r["origins"] for r in grp)
        print(f"  {name:22s} {h}/{len(grp)} = {h / len(grp):>4.0%}   "
              f"found by a detected change: {ris}/{len(grp)}")

    missed = [r for r in rows if not r["hit"]]
    if missed:
        print("\n  missed entirely:")
        for r in missed:
            where = "clip boundary" if r["boundary"] else "mid-clip"
            print(f"    {r['desc']}  ({r['start']:.1f}-{r['end']:.1f}s, {where})")


def sweep(clips):
    print(f"{'hi_pct':>6} {'every':>6} {'width':>6} {'pad':>4} | "
          f"{'recall':>10} {'full':>5} {'cov':>7} {'br/clip':>7}")
    best = []
    for hi, every, width, pad in itertools.product(
            (95.0, 97.0, 99.0), (20.0, 30.0, 45.0), (4.0, 8.0), (2.0, 4.0)):
        c = SignalConfig()
        c.hi_pct, c.lo_pct = hi, min(90.0, hi - 5)
        c.sentinel_every_s, c.sentinel_width_s, c.pad_s = every, width, pad
        rows, cov, nb = score(clips, c)
        hit = sum(r["hit"] for r in rows)
        full = sum(r["full"] for r in rows)
        best.append((hit / len(rows), cov, hi, every, width, pad))
        print(f"{hi:>6} {every:>6} {width:>6} {pad:>4} | {hit:>3}/{len(rows)}={hit / len(rows):>5.0%} "
              f"{full / len(rows):>5.0%} {cov:>7.1%} {nb:>7.1f}")
    print("\n  highest recall, then cheapest:")
    for r, cov, hi, every, width, pad in sorted(best, key=lambda x: (-x[0], x[1]))[:5]:
        print(f"    hi_pct={hi} sentinel_every_s={every} sentinel_width_s={width} "
              f"pad_s={pad}  ->  recall {r:.0%} at coverage {cov:.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/eval/labels.json")
    ap.add_argument("--clips", default="data/eval")
    ap.add_argument("--sweep", action="store_true", help="recall/coverage frontier")
    a = ap.parse_args()

    labels = Path(a.labels)
    if not labels.exists():
        print(f"no labels at {labels}. Build the evaluation set first (`make data`).",
              file=sys.stderr)
        return 2
    clips = load(labels, Path(a.clips))
    if not clips:
        print(f"no clips from {labels} are present under {a.clips}.", file=sys.stderr)
        return 2

    cfg = Config()
    t0 = time.perf_counter()
    cache_signals(clips, cfg.signal)
    elapsed = time.perf_counter() - t0

    if a.sweep:
        sweep(clips)
        return 0
    rows, cov, nb = score(clips, cfg.signal)
    report(rows, cov, nb, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
