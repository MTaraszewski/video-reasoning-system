"""Build the positive-control set from MEVA's curated example clips.

These clips have MEVA's own annotations **burned into the picture** — a red box
around the actor, labelled with the activity name. That makes them useless as
evaluation data: the answer is written on the frame, and any score would measure
reading rather than seeing.

It also makes them uniquely useful as a **ceiling test**.

If the model cannot localise `Enter_Vehicle` on a clip where the words
"Enter_Vehicle" appear in a box around the person doing it, it will certainly fail
on clean footage. That separates failure modes which a clean-footage result cannot:

| Behaviour | What it means |
|---|---|
| Fails even with the answer on screen | Cannot read the frame or follow the task — the problem is prompting or vision, not event recognition |
| Succeeds with the label, fails without | Recognises *text*, not *events* — the interesting finding |
| Succeeds at both | The pipeline limits us, not the model |

A decisive negative costs seconds of GPU time here, versus a full evaluation.

Descriptions are phrased **the way a client would say them**, never as MEVA class
names. The system is open-vocabulary; labelling with `Enter_Vehicle` would test
string matching against a taxonomy, which is not the product.

    python scripts/make_positive_control.py --src /data/meva-examples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_reasoning.decode import probe

# MEVA activity -> how a client would ask for it. Two of these are effectively
# the brief's own examples: "a person enters through the door" and a vehicle
# reversing, which is the same event shape as "a forklift reverses".
PHRASING = {
    "abandon-package": "someone leaves a bag or package behind",
    "close-trunk": "a car boot is closed",
    "close-vehicle-door": "a car door closes",
    "embrace-interaction": "two people hug",
    "enter-through-structure": "a person enters through the door",
    "enter-vehicle": "a person gets into a car",
    "exit-through-structure": "a person comes out through the door",
    "exit-vehicle": "a person gets out of a car",
    "hand-interaction": "two people shake hands",
    "load-vehicle": "someone loads something into a vehicle",
    "open-facility-door": "a person opens a building door",
    "open-trunk": "a car boot is opened",
    "people-talking": "two people stand talking",
    "pickup-object": "someone picks an object up",
    "purchasing": "someone buys something",
    "riding": "a person rides past",
    "set-down-object": "someone puts an object down",
    "sit-down": "a person sits down",
    "stand-up": "a person stands up",
    "talk-on-phone": "a person talks on a phone",
    "text-on-phone": "a person looks at a phone",
    "transport-heavy-object": "two people carry something heavy",
    "unload-vehicle": "someone unloads something from a vehicle",
    "vehicle-dropsoff-person": "a vehicle drops someone off",
    "vehicle-picksup-person": "a vehicle picks someone up",
    "vehicle-reversing": "a vehicle reverses",
    "vehicle-starting": "a vehicle starts moving",
    "vehicle-stopping": "a vehicle stops moving",
    "vehicle-turning-left": "a vehicle turns left",
    "vehicle-turning-right": "a vehicle turns right",
    "vehicle-uturn": "a vehicle makes a u-turn",
}


def activity_of(name: str) -> str:
    a = name
    if a.startswith("ex"):
        a = a.split("-", 1)[1] if "-" in a else a
    return a.rsplit(".", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/data/meva-examples")
    ap.add_argument("--out", default=None, help="Default: <src>/labels.json")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="Seconds trimmed from each end of the assumed span.")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src / "labels.json"
    clips = sorted(p for p in src.glob("*.mp4") if not p.name.startswith("sheet"))
    if not clips:
        print(f"no clips under {src}")
        return

    entries = []
    for p in clips:
        act = activity_of(p.stem)
        desc = PHRASING.get(act)
        if desc is None:
            desc = act.replace("-", " ")
            print(f"  note: no client phrasing for {act!r}, using {desc!r}")
        meta = probe(p)
        # These clips are pre-cut AROUND the activity, so the event occupies
        # essentially the whole clip. That is coarse, and deliberately so: a
        # ceiling test asks whether the event is found AT ALL, not how precisely.
        # Precision belongs to the synthetic set, where truth is exact.
        start = round(min(args.margin, meta.duration_s / 4), 3)
        end = round(max(start + 0.1, meta.duration_s - args.margin), 3)
        entries.append({
            "video": p.name,
            "axis": "positive_control",
            "duration_s": meta.duration_s,
            "activity": act,
            "boundaries": "approximate — the clip is cut around the activity",
            "events": [{"description": desc, "start_s": start, "end_s": end}],
        })
        print(f"  {p.name:40s} {meta.duration_s:6.1f}s  {desc}")

    out.write_text(json.dumps(entries, indent=2))
    print(f"\n{len(entries)} positive-control clip(s) -> {out}")
    print("\nThese clips carry MEVA's annotations BURNED INTO THE PICTURE.")
    print("They are a ceiling test, never evaluation data, and the eval harness")
    print("excludes the positive_control axis from headline metrics.")


if __name__ == "__main__":
    main()
