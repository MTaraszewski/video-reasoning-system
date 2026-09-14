#!/usr/bin/env python3
"""Recover an exchange corpus recorded before records carried their video.

Bracket ids are per-clip (b1, b2, ...) and an exchange log spans a whole
evaluation, so a corpus without a `video` field cannot be indexed: every clip's
b1 collides, and each clip silently replays another clip's replies. The first
GPU session's two corpora -- 644 paid-for calls -- were written that way.

They are recoverable because of how the evaluation runs: clips are processed
SEQUENTIALLY and only the tasks within a clip run concurrently. So records
arrive in per-clip blocks whose sizes the plan predicts exactly, scrambled
only inside each block.

Content matching alone is not enough -- 93 of 225 planned triples in the first
session are claimable by two clips, because sentinel brackets fall on the same
cadence regardless of what is in the footage. But block position plus content
is: assign each block to its clip, then require the block's MULTISET of
(bracket, subject, times) triples to equal that clip's planned multiset
exactly. If any block disagrees, nothing is written.

    python scripts/fix_exchanges.py out/ef/exchanges-per_bracket-video.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventfinder.config import Config, load          # noqa: E402
from eventfinder.pipeline import plan                 # noqa: E402


def descriptions_of(labels):
    seen = []
    for clip in labels:
        for ev in clip["events"]:
            if ev["description"] not in seen:
                seen.append(ev["description"])
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exchanges")
    ap.add_argument("--labels", default="data/eval/labels.json")
    ap.add_argument("--clips", default="data/eval")
    ap.add_argument("--config", default="eventfinder.yaml")
    ap.add_argument("--out", default=None, help="default: rewrite in place")
    ap.add_argument("--blocks", default=None,
                    help="comma-separated records per clip, e.g. 42,42,28,28,49,42,28,63. "
                         "Use when the code's grouping has changed since the corpus was "
                         "recorded, so re-planning no longer reproduces its task list. "
                         "Each block is then checked for internal consistency instead: "
                         "every bracket id in it must appear the same number of times, "
                         "once per subject group.")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.exchanges) if l.strip()]
    if all(r.get("video") for r in rows):
        print(f"  {a.exchanges}: already stamped, nothing to do")
        return 0

    mode = "per_bracket" if len(rows[0].get("times") or []) > 1 else "per_step"
    cfg = load(a.config) if Path(a.config).exists() else Config()
    cfg.observe.mode = mode
    labels = json.loads(Path(a.labels).read_text())
    descs = descriptions_of(labels)

    def triple(bracket_id, subject, times):
        return (bracket_id, subject, tuple(round(float(x), 3) for x in times))

    plans: dict[str, list[tuple]] = {}
    for clip in labels:
        video = Path(a.clips) / clip["video"]
        if not video.exists():
            print(f"  cannot verify without {video}", file=sys.stderr)
            return 2
        _, _, tasks, _ = plan(str(video), descs, cfg)
        plans[clip["video"]] = [triple(t.bracket.id, t.subject, t.stamp_times)
                                for t in tasks]

    from collections import Counter

    if a.blocks:
        sizes = [int(x) for x in a.blocks.split(",")]
        if len(sizes) != len(labels):
            print(f"  --blocks has {len(sizes)} entries for {len(labels)} clips",
                  file=sys.stderr)
            return 2
        if sum(sizes) != len(rows):
            print(f"  --blocks sums to {sum(sizes)}, corpus has {len(rows)} records",
                  file=sys.stderr)
            return 2
        i, report = 0, []
        for clip, n in zip(labels, sizes):
            block = rows[i:i + n]
            per_bracket = Counter(r["bracket_id"] for r in block)
            groups = set(per_bracket.values())
            if len(groups) != 1:
                print(f"  block for {clip['video']} is not internally consistent: "
                      f"bracket ids appear {sorted(groups)} times, expected one value "
                      "(one call per subject group per bracket)", file=sys.stderr)
                return 2
            # A bracket id must always carry the same requested times inside a clip.
            times_of = {}
            for r in block:
                key = r["bracket_id"]
                t = tuple(round(float(x), 3) for x in (r.get("times") or []))
                if times_of.setdefault(key, t) != t:
                    print(f"  block for {clip['video']}: bracket {key} has two different "
                          "time lists; the block boundary is wrong", file=sys.stderr)
                    return 2
            for r in block:
                r["video"] = clip["video"]
            report.append((clip["video"], n))
            i += n
        out = Path(a.out or a.exchanges)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  attributed {len(rows)} records from the supplied block sizes; every "
              "block was internally consistent (one call per subject group per bracket, "
              "one time list per bracket)")
        for v, n in report:
            print(f"    {n:>4} {v}")
        print(f"  -> {out}")
        return 0

    i, stamped, report = 0, 0, []
    for clip in labels:
        want = plans[clip["video"]]
        block = rows[i:i + len(want)]
        if len(block) < len(want):
            print(f"  corpus holds {len(rows)} records; {clip['video']} needs "
                  f"{len(want)} more and only {len(block)} remain. An interrupted "
                  "variant cannot be attributed past the point it stopped.",
                  file=sys.stderr)
            return 2
        got = Counter(triple(r["bracket_id"], r["subject"], r.get("times") or [])
                      for r in block)
        if got != Counter(want):
            missing = Counter(want) - got
            extra = got - Counter(want)
            print(f"  BLOCK MISMATCH at {clip['video']} (records {i}..{i + len(want)}):",
                  file=sys.stderr)
            for tr in list(missing)[:3]:
                print(f"    planned but absent: {tr[0]}/{tr[1]} {list(tr[2])[:3]}...",
                      file=sys.stderr)
            for tr in list(extra)[:3]:
                print(f"    present but unplanned: {tr[0]}/{tr[1]} {list(tr[2])[:3]}...",
                      file=sys.stderr)
            print("  refusing to write; this corpus was not produced by this config.",
                  file=sys.stderr)
            return 2
        for r in block:
            r["video"] = clip["video"]
        stamped += len(block)
        i += len(block)
        report.append((clip["video"], len(block)))

    if i != len(rows):
        print(f"  {len(rows) - i} trailing records could not be assigned; "
              "refusing to write.", file=sys.stderr)
        return 2

    out = Path(a.out or a.exchanges)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  attributed {stamped} records: every per-clip block's multiset of "
          "(bracket, subject, times) triples matched its plan exactly")
    for v, n in report:
        print(f"    {n:>4} {v}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
