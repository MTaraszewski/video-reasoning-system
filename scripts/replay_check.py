"""Replay recorded model responses through the parser and report what changed.

The point of recording every exchange: parser work happens here, against real
model output, at no cost — instead of renting a GPU to find out whether a regex
worked.
"""
import json, pathlib, sys
from video_reasoning.backends.base import parse_events

dirs = [pathlib.Path(d) for d in (sys.argv[1:] or ["/data/_GPU_experiments/session1"])]
recs = [r for d in dirs for r in sorted(d.glob("*.json"))]
print(f"replaying {len(recs)} recorded call(s)\n")

ok = was_err = now_ok = still_err = 0
for r in recs:
    d = json.loads(r.read_text())
    events, err = parse_events(d["raw"])
    before_err = bool(d.get("error"))
    if before_err:
        was_err += 1
        if not err and events:
            now_ok += 1
        else:
            still_err += 1
    if not err:
        ok += 1

print(f"  parsed cleanly now : {ok}/{len(recs)}")
print(f"  failed before      : {was_err}")
print(f"    -> now recovered : {now_ok}")
print(f"    -> still failing : {still_err}")
print()
for r in recs:
    d = json.loads(r.read_text())
    events, err = parse_events(d["raw"])
    if d.get("error") and err:
        print(f"  STILL FAILING: {err}")
        print(f"    {d['raw'][:120].strip()}")
