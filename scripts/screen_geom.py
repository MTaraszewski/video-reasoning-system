"""Rank MEVA clips by how big the actor is during its declared events.

The first selection pass asked only "does the annotation declare an activity?".
Three of the six clips it chose showed that activity at 38-121 px — below what a
human can confirm from a frame, so below what we can honestly label, so below what
we can fairly score a model against. `.geom.yml` carries per-frame actor boxes and
answers the size question directly, for 5-70 KB per clip instead of 56-203 MB.

**This ranks; it does not reject.** A false positive costs ten seconds looking at
a contact sheet. A false negative is silent — the clip never appears, and nothing
records that it was dropped. We already came within one zoom of losing a good clip
(G329, 295 px) to an eyeball judgement, so every clip stays in the output and the
threshold only annotates.

Calibration is empirical, not guessed. Six clips were hand-judged from contact
sheets before this existed, and median actor height orders them exactly:

    G326  694 px   two events confirmed
    G329  295 px   confirmed (needed zooming)
    G331  267 px   suspect, hard to verify
    G301  121 px   all rejected
    G328   41 px   all rejected
    G336   38 px   all rejected

    python scripts/screen_geom.py --index data/meva-index
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

# `- { geom: { id1: 0, id0: 4, ts0: 2625, ts1: 87.5, g0: 561 223 838 844 , ... } }`
# ts0 is the frame index; g0 is x1 y1 x2 y2.
_GEOM = re.compile(r'ts0:\s*(\d+).*?g0:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')
_ACT = re.compile(r'act3:\s*\{\s*([A-Za-z_]+)\s*:')
_SPAN = re.compile(r'timespan:\s*\[\{\s*tsr0:\s*\[(\d+),\s*(\d+)\]')

# Bands, from the calibration above. Deliberately generous at the bottom: the
# point is to order candidates, not to refuse them.
BANDS = ((250, "good"), (120, "marginal"), (0, "too-small"))


def band(h: float) -> str:
    for lo, name in BANDS:
        if h >= lo:
            return name
    return "too-small"


def parse_geom(path: Path) -> dict[int, list[int]]:
    """frame -> list of actor box heights in that frame."""
    by_frame: dict[int, list[int]] = defaultdict(list)
    for line in path.read_text().splitlines():
        m = _GEOM.search(line)
        if m:
            f, x1, y1, x2, y2 = (int(g) for g in m.groups())
            by_frame[f].append(abs(y2 - y1))
    return by_frame


def parse_activities(path: Path) -> list[tuple[str, int, int]]:
    out = []
    for line in path.read_text().splitlines():
        if "act3:" not in line:
            continue
        a, sp = _ACT.search(line), _SPAN.search(line)
        if a and sp:
            out.append((a.group(1), int(sp.group(1)), int(sp.group(2))))
    return out


def screen(stem: str, act_path: Path, geom_path: Path) -> dict | None:
    acts = parse_activities(act_path)
    if not acts:
        return None
    by_frame = parse_geom(geom_path)
    if not by_frame:
        return None

    per_event = []
    for name, f0, f1 in acts:
        # Every actor box inside the event's frame span. Activity-to-actor
        # association exists in the annotation, but during the event the actors
        # present ARE the relevant ones, and the coarse version cannot silently
        # miss a large actor the way a mis-parsed id link could.
        hs = [h for f in range(f0, f1 + 1) for h in by_frame.get(f, ())]
        if not hs:
            continue
        per_event.append({
            "activity": name, "frames": [f0, f1],
            "median_actor_h": round(st.median(hs)),
            "max_actor_h": max(hs),
        })
    if not per_event:
        return None

    meds = [e["median_actor_h"] for e in per_event]
    best = max(meds)
    return {
        "clip": stem,
        "scene": stem.split(".")[-2] if "." in stem else "",
        "camera": stem.split(".")[-1],
        "events": len(per_event),
        "median_actor_h": round(st.median(meds)),
        # The BEST event decides the clip: one large, well-framed event is enough
        # to make a clip worth labelling, and averaging would hide it behind a
        # crowd of distant ones.
        "best_event_actor_h": best,
        "band": band(best),
        "activities": sorted({e["activity"] for e in per_event}),
        "per_event": per_event,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="data/meva-index")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    index = Path(args.index)
    rows = []
    for geom in sorted(index.rglob("*.geom.yml")):
        stem = geom.name[: -len(".geom.yml")]
        act = geom.with_name(f"{stem}.activities.yml")
        if not act.exists():
            continue
        r = screen(stem, act, geom)
        if r:
            rows.append(r)

    rows.sort(key=lambda r: r["best_event_actor_h"], reverse=True)

    print(f"{'band':<10}{'best h':>7}{'med h':>7}{'ev':>4}  {'camera':<7}{'scene':<10} activities")
    print("-" * 100)
    for r in rows[: args.top]:
        acts = ", ".join(a.replace("_", " ") for a in r["activities"][:4])
        print(f"{r['band']:<10}{r['best_event_actor_h']:>7}{r['median_actor_h']:>7}"
              f"{r['events']:>4}  {r['camera']:<7}{r['scene']:<10} {acts[:52]}")

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["band"]] += 1
    print(f"\n{len(rows)} clip(s) screened  |  " +
          "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print("Nothing is excluded — the band annotates, it does not reject.")

    out = Path(args.out) if args.out else index / "screen.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"Full ranking -> {out}")


if __name__ == "__main__":
    main()
