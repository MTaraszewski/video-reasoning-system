"""Turn MEVA annotations into a trim plan and a labelling worksheet.

The brief requires events *"labelled by hand"*, and that is what happens — but
hand-labelling is searching plus judging, and only the judging needs a human.
MEVA already says where its activities are, so this does the searching:

1. parse `.activities.yml` into activity spans in seconds
2. choose a trim window that captures the most events in the brief's 1-3 minutes
3. emit a labels file with times pre-filled, phrased as a client would say them

The human then **confirms or corrects** against the contact sheet, rather than
scrubbing a five-minute clip hunting for something to label. The labels stay ours;
MEVA's annotations locate the footage and afterwards measure our labelling error.

**Why the trim window is computed, not fixed.** In one clip the declared events sit
at 45s, 64s, 74s — and 268s. Trimming naively from zero would silently discard the
last one, and nothing downstream would notice a labelled event had been cut away.

    python scripts/meva_labels.py --src /data/meva-annotated --seconds 120
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from video_reasoning.decode import probe

# MEVA activity -> how a client would ask for it. Never the class name: the system
# is open-vocabulary, and labelling with a taxonomy would test string matching.
# Three of these are effectively the brief's own examples.
PHRASING = {
    "Enter_Facility": "a person enters through the door",
    "Exit_Facility": "a person comes out through the door",
    "Open_Facility_Door": "a person opens a building door",
    "Close_Facility_Door": "a person closes a building door",
    "Enter_Vehicle": "a person gets into a vehicle",
    "Exit_Vehicle": "a person gets out of a vehicle",
    "Open_Vehicle_Door": "a vehicle door opens",
    "Close_Vehicle_Door": "a vehicle door closes",
    "Open_Trunk": "a vehicle boot is opened",
    "Close_Trunk": "a vehicle boot is closed",
    "Vehicle_Reversing": "a vehicle reverses",
    "Vehicle_Starting": "a vehicle starts moving",
    "Vehicle_Stopping": "a vehicle stops moving",
    "Vehicle_Turning_Left": "a vehicle turns left",
    "Vehicle_Turning_Right": "a vehicle turns right",
    "Vehicle_UTurn": "a vehicle makes a u-turn",
    "Vehicle_PicksUp_Person": "a vehicle picks someone up",
    "Vehicle_DropsOff_Person": "a vehicle drops someone off",
    "Transport_HeavyObject": "two people carry something heavy",
    "Object_Transfer": "someone hands an object to another person",
    "Pickup_Object": "a person picks an object up",
    "SetDown_Object": "a person puts an object down",
    "Talking": "two people stand talking",
    "Talk_On_Phone": "a person talks on a phone",
    "Text_On_Phone": "a person looks at a phone",
    "Read_Document": "a person reads a document",
    "Purchasing": "a person buys something",
    "Sitting_Down": "a person sits down",
    "Standing_Up": "a person stands up",
    "Riding": "a person rides past",
    "Abandon_Package": "someone leaves a bag behind",
    "Loading": "someone loads something into a vehicle",
    "Unloading": "someone unloads something from a vehicle",
    "Entering": "a person enters through the door",
    "Exiting": "a person comes out through the door",
}

_ACT = re.compile(r'act3:\s*\{\s*([A-Za-z_]+)\s*:', re.S)
_SPAN = re.compile(r'timespan:\s*\[\{\s*tsr0:\s*\[(\d+),\s*(\d+)\]', re.S)


def parse_activities(path: Path, fps: float) -> list[dict]:
    """Read one .activities.yml into activity spans, in seconds."""
    out: list[dict] = []
    for line in path.read_text().splitlines():
        if "act3:" not in line:
            continue
        a, sp = _ACT.search(line), _SPAN.search(line)
        if not (a and sp):
            continue
        f0, f1 = int(sp.group(1)), int(sp.group(2))
        out.append({
            "activity": a.group(1),
            "start_s": round(f0 / fps, 3),
            "end_s": round(f1 / fps, 3),
            "frames": [f0, f1],
        })
    return sorted(out, key=lambda e: e["start_s"])


def best_window(events: list[dict], length: float, duration: float) -> tuple[float, list[dict]]:
    """Pick the trim start capturing the most complete events.

    Candidate starts are the event starts themselves (minus a little lead-in), so
    a window always begins just before something happens rather than at an
    arbitrary offset. Only events lying WHOLLY inside count — a half-captured
    event would be labelled with a boundary the footage does not contain.
    """
    if not events:
        return 0.0, []
    best: tuple[float, list[dict]] = (0.0, [])
    for e in events:
        start = max(0.0, min(e["start_s"] - 3.0, max(0.0, duration - length)))
        inside = [x for x in events
                  if x["start_s"] >= start and x["end_s"] <= start + length]
        if len(inside) > len(best[1]):
            best = (round(start, 3), inside)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/data/meva-annotated")
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="Trim length. The brief wants 60-180.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.src)
    ann_dir = src / "annotations"
    if not ann_dir.exists():
        print(f"no annotations under {ann_dir}")
        return

    plan: list[dict] = []
    for ann in sorted(ann_dir.glob("*.activities.yml")):
        stem = ann.name.replace(".activities.yml", "")
        video = next((p for p in src.glob(f"{stem}*") if p.suffix in (".avi", ".mp4")), None)
        if video is None:
            print(f"  no video for {stem}")
            continue

        meta = probe(video)
        fps = meta.fps or 30.0
        events = parse_activities(ann, fps)
        if not events:
            print(f"  {video.name}: annotation declares no activity spans")
            continue

        start, inside = best_window(events, args.seconds, meta.duration_s)
        lost = [e for e in events if e not in inside]

        print(f"\n  {video.name}")
        print(f"    {meta.duration_s:.0f}s @ {fps:.0f}fps, {len(events)} declared event(s)")
        print(f"    trim {start:.1f}s -> {start + args.seconds:.1f}s "
              f"captures {len(inside)}, drops {len(lost)}")
        for e in events:
            mark = "keep" if e in inside else "DROP"
            print(f"      {mark}  {e['start_s']:>7.1f}-{e['end_s']:>7.1f}s  {e['activity']}")

        plan.append({
            "video": f"{stem}.mp4",
            "source": video.name,
            "axis": "real_fixed_camera",
            "trim": {"start_s": start, "length_s": args.seconds},
            "duration_s": args.seconds,
            "events": [
                {
                    "description": PHRASING.get(e["activity"],
                                                e["activity"].replace("_", " ").lower()),
                    # Times are relative to the TRIMMED clip, which is what the
                    # eval will actually open.
                    "start_s": round(e["start_s"] - start, 3),
                    "end_s": round(e["end_s"] - start, 3),
                    "meva_activity": e["activity"],
                    "confirmed_by_hand": False,
                }
                for e in inside
            ],
        })

    out = Path(args.out) if args.out else src / "labels.candidate.json"
    out.write_text(json.dumps(plan, indent=2))
    n = sum(len(p["events"]) for p in plan)
    print(f"\n{len(plan)} clip(s), {n} candidate event(s) -> {out}")
    print("\nThese are CANDIDATES, located by MEVA's annotations, not labels.")
    print("Open each contact sheet, confirm the event is really there and the")
    print("boundaries are right, correct them, set confirmed_by_hand: true, and")
    print("save as labels.json. Unconfirmed entries are not hand labels.")


if __name__ == "__main__":
    main()
