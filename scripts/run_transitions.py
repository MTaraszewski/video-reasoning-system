"""Run caption-parse-derive over every hand-labelled event and score transitions.

Scoring is not interval tIoU. A state timeline answers "when was the door open";
the labels answer "when did the opening happen". Overlapping those compares two
different questions and understates the method -- state polling scored 0.26-0.55
on clips where the transition was within a second of the label.

The measure here is the **transition instant against the label's span**: did the
state change inside the window a human marked? Resolution equals the step size,
and a miss reports how far outside it landed.

Every caption is written to disk. Four negative results were reported in this
project before anyone read what the model actually said; when that finally
happened the diagnosis took one reading. Captions are the evidence, and they are
cheap to keep.

    python scripts/run_transitions.py --labels /data/eval/labels.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from reason_timeline import describe, match_text
from video_reasoning.config import load_config
from video_reasoning.decode import sample_frames


def timeline(client, model, cfg, video, states, t0, t1, step, span):
    """Captions and parsed states across a span. Returns rows of (t, state, text)."""
    rows, t = [], t0
    while t < t1:
        _, frames = sample_frames(video, fps=cfg.sampling.fps,
                                  max_side=cfg.sampling.frame_max_side,
                                  overlay=False, start_s=t, end_s=t + span)
        if not frames:
            break
        subject = states[0].replace("the ", "").split(" is ")[0]
        text, _, finish = describe(client, model, frames, subject)
        pr = match_text(text, states) if text else {}
        state = None
        if pr:
            state = states[1] if pr.get(states[1], 0) > pr.get(states[0], 0) else states[0]
        rows.append({"t": round(t, 3), "state": state, "text": text,
                     "truncated": finish == "length"})
        t += step
    return rows


def transitions(rows):
    """Instants where the parsed state changes, ignoring gaps of unknown state.

    An unparsed row is skipped rather than treated as a change: a description
    that says nothing about state is not evidence that the state moved.
    """
    out, prev, prev_t = [], None, None
    for r in rows:
        if r["state"] is None:
            continue
        if prev is not None and r["state"] != prev:
            out.append({"at": round((prev_t + r["t"]) / 2, 3),
                        "from": prev, "to": r["state"],
                        "bracket": [prev_t, r["t"]]})
        prev, prev_t = r["state"], r["t"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="/data/eval/labels.json")
    ap.add_argument("--data-dir", default="/data/eval")
    ap.add_argument("--states", default="states.json")
    ap.add_argument("--out", default="/out/transitions.json")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--span", type=float, default=2.0)
    ap.add_argument("--pad", type=float, default=6.0,
                    help="Seconds either side of the label to poll.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    client = OpenAI(base_url=cfg.model.base_url, api_key=cfg.model.api_key,
                    timeout=cfg.model.request_timeout_s)
    smap = {k: v for k, v in json.loads(Path(args.states).read_text()).items()
            if not k.startswith("_")}
    entries = json.loads(Path(args.labels).read_text())

    events = [(e, ev) for e in entries for ev in e["events"]]
    if args.limit:
        events = events[: args.limit]

    results, hits, scored, skipped = [], 0, 0, []
    for clip, ev in events:
        desc = ev["description"]
        tag = f"{clip['video'].split('.')[0][5:]}_{clip['video'].split('.')[-3]}"
        if desc not in smap:
            skipped.append((tag, desc))
            results.append({"clip": clip["video"], "description": desc,
                            "skipped": "no state pair"})
            continue

        states = smap[desc]
        ts, te = float(ev["start_s"]), float(ev["end_s"])
        lo = max(0.0, ts - args.pad)
        hi = min(float(clip["duration_s"]), te + args.pad)
        rows = timeline(client, cfg.model.name, cfg, str(Path(args.data_dir) / clip["video"]),
                        states, lo, hi, args.step, args.span)
        trs = transitions(rows)

        # A hit is any transition instant inside the labelled span. Direction is
        # recorded but not required: t=5 on G326 detected the change at the right
        # moment and reversed its sign, which is a different error from missing it.
        inside = [x for x in trs if ts <= x["at"] <= te]
        nearest = min((abs(x["at"] - ts) if x["at"] < ts else x["at"] - te
                       for x in trs), default=None)
        scored += 1
        if inside:
            hits += 1
        mark = "HIT " if inside else "miss"
        at = f"{inside[0]['at']:.1f}s" if inside else (
            f"nearest {nearest:.1f}s away" if nearest is not None else "no transition")
        print(f"  {mark} {tag:<12} {desc[:38]:<38} label {ts:>5.1f}-{te:<5.1f} {at}")
        results.append({"clip": clip["video"], "description": desc,
                        "states": states, "label": [ts, te],
                        "transitions": trs, "hit": bool(inside),
                        "nearest_s": nearest, "rows": rows})

    print(f"\n{hits}/{scored} events: a state transition inside the labelled span")
    if skipped:
        print(f"\n{len(skipped)} description(s) have no state pair — not a model "
              f"failure, a description that does not decompose into one:")
        for tag, d in skipped:
            print(f"    {tag:<12} {d}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"hits": hits, "scored": scored, "step_s": args.step, "span_s": args.span,
         "model": cfg.model.name, "results": results}, indent=2))
    print(f"\n-> {args.out}   (every caption kept: read them before re-running)")


if __name__ == "__main__":
    main()
