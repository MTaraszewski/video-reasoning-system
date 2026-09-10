"""Diff two transition runs: did the CAPTION change, or did the PARSE change?

The same command run twice gave different answers -- one event's transition moved
3 seconds and dropped out of its label, at temperature 0. Two very different causes
need two very different fixes:

  the model described the scene differently  -> sampling noise, fix by voting
  it described it the same and we read it differently -> our matcher is wrong

Averaging over the second would be averaging over our own bug, so the distinction
has to be settled before anything is built on top of it.

    python scripts/diff_runs.py /out/transitions.json /out/transitions-2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows_by_key(run: dict) -> dict:
    out = {}
    for r in run.get("results", []):
        if "rows" not in r:
            continue
        for row in r["rows"]:
            out[(r["clip"], r["description"], row["t"])] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--show", type=int, default=6, help="Examples to print.")
    args = ap.parse_args()

    A = rows_by_key(json.loads(Path(args.a).read_text()))
    B = rows_by_key(json.loads(Path(args.b).read_text()))
    keys = sorted(set(A) & set(B))
    if not keys:
        print("no overlapping polls between the two runs")
        return

    same_text = same_state = 0
    text_differs_state_same = text_differs_state_differs = 0
    text_same_state_differs = []
    examples = []

    for k in keys:
        a, b = A[k], B[k]
        t_eq = a["text"].strip() == b["text"].strip()
        s_eq = a["state"] == b["state"]
        same_text += t_eq
        same_state += s_eq
        if t_eq and not s_eq:
            text_same_state_differs.append((k, a, b))
        elif not t_eq and s_eq:
            text_differs_state_same += 1
        elif not t_eq and not s_eq:
            text_differs_state_differs += 1
            if len(examples) < args.show:
                examples.append((k, a, b))

    n = len(keys)
    print(f"{n} polls compared\n")
    print(f"  identical caption      {same_text:>4}/{n}  ({same_text/n:.0%})")
    print(f"  identical parsed state {same_state:>4}/{n}  ({same_state/n:.0%})")
    print()
    print(f"  caption differs, state same     {text_differs_state_same:>4}  "
          f"(harmless -- wording varies, meaning does not)")
    print(f"  caption differs, state differs  {text_differs_state_differs:>4}  "
          f"(SAMPLING noise -- fix by voting)")
    print(f"  caption same, state differs     {len(text_same_state_differs):>4}  "
          f"(PARSER bug -- voting would hide it)")

    if text_same_state_differs:
        print("\n=== parser disagreements on identical text ===")
        for (clip, desc, t), a, b in text_same_state_differs[: args.show]:
            print(f"\n  {desc[:40]} t={t}")
            print(f"    A: {a['state']}\n    B: {b['state']}")
            print(f"    text: {a['text'][:150]}")

    if examples:
        print("\n=== captions that changed the state ===")
        for (clip, desc, t), a, b in examples:
            print(f"\n  {desc[:40]} t={t}")
            print(f"    A [{a['state']}]: {a['text'][:130]}")
            print(f"    B [{b['state']}]: {b['text'][:130]}")

    print("\nVerdict: if most disagreement is 'caption differs, state differs', the "
          "model is sampling; majority voting over repeats fixes it. If it is "
          "'caption same, state differs', the matcher is at fault and voting "
          "would average over our own bug.")


if __name__ == "__main__":
    main()
