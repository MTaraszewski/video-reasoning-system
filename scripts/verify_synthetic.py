"""Check that every synthetic clip actually shows what its label claims.

Written after `machine-stop.mp4` was found to be empty for eight seconds and then
show a MOVING box during the window labelled "the machine stops moving" — the
opposite of its own label. It had already passed through the pipeline, the probe
and the eval without complaint, because nothing in a pipeline can notice that
data disagrees with its ground truth.

Method: measure the fraction of pixels changing between consecutive samples, then
compare motion INSIDE each labelled event against motion OUTSIDE it. What counts
as correct depends on the axis, which is the point — an absence-of-motion clip
must show the opposite polarity to every other clip here.

    make verify-data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from video_reasoning.decode import sample_frames

MOVED_PX_THRESHOLD = 25       # per-pixel intensity change counted as motion
QUIET = 0.05                  # % of pixels moving that counts as "still"
ACTIVE = 0.30                 # % of pixels moving that counts as "moving"
BOUNDARY_S = 0.35             # samples spanning an event edge belong to neither side
                              # ...but never more than a quarter of the event, or
                              # a 0.15s event has no "inside" at all. The first
                              # version used a flat margin and reported the short
                              # clip as broken twice over.


def sample_rate_for(events: list[tuple[float, float]]) -> float:
    """Sample fast enough to SEE the shortest labelled event.

    The first version of this checker used a flat 4 fps and reported
    `short-event.mp4` as broken: its events are 0.15s and 0.20s long, so at
    0.25s spacing they fall between samples entirely. The clip was correct; the
    measurement was too coarse to observe it — which is exactly the failure mode
    that clip exists to represent, reproduced accidentally in the checker.
    """
    if not events:
        return 4.0
    shortest = min(e - s for s, e in events)
    # At least four samples across the shortest event, capped at source rates.
    return float(min(30.0, max(4.0, 4.0 / max(shortest, 0.05))))


def motion_trace(path: Path, fps: float) -> list[tuple[float, float]]:
    _, frames = sample_frames(str(path), fps=fps, max_side=320, overlay=False)
    arrs = [np.asarray(f.image.convert("L"), dtype=np.int16) for f in frames]
    return [
        (frames[i].t,
         float((np.abs(arrs[i] - arrs[i - 1]) > MOVED_PX_THRESHOLD).mean() * 100))
        for i in range(1, len(arrs))
    ]


def check(entry: dict, data_dir: Path) -> tuple[bool, str]:
    path = data_dir / entry["video"]
    if not path.exists():
        return False, "file missing"
    axis = entry.get("axis", "unknown")
    events = [(e["start_s"], e["end_s"]) for e in entry.get("events", [])]
    fps = sample_rate_for(events)
    trace = motion_trace(path, fps)
    if not trace:
        return False, "no frames"

    # A sample spanning an event edge compares a frame from each side, so it
    # belongs to NEITHER. An earlier version widened "inside" by a margin, which
    # pulled those transitions in and made the stopped window look like it moved.
    def zone(t: float) -> str:
        for s, e in events:
            margin = min(BOUNDARY_S, (e - s) / 4.0)
            if s + margin <= t <= e - margin:
                return "in"
            if s - BOUNDARY_S <= t <= e + BOUNDARY_S:
                return "edge"
        return "out"

    ins = [m for t, m in trace if zone(t) == "in"]
    out = [m for t, m in trace if zone(t) == "out"]
    mi = max(ins) if ins else 0.0
    mo = max(out) if out else 0.0

    if not ins and events and axis != "negative":
        return False, (f"no samples strictly inside any event at {fps:.0f}fps — "
                       "events shorter than the sampling interval")

    if axis == "absence_of_motion":
        # The event IS the stillness: quiet inside, active outside.
        if mi > QUIET:
            return False, f"moves during the 'stopped' window (max {mi:.2f}%)"
        if mo < ACTIVE:
            return False, f"never moves outside it (max {mo:.2f}%) — nothing stops"
        return True, (f"still inside {mi:.2f}%, moving outside {mo:.2f}% "
                      f"@{fps:.0f}fps")

    if axis == "negative":
        # No labelled event. A distractor may move; what matters is that no
        # interval looks like the event we ask about.
        return True, f"no labelled events; peak motion {mo:.2f}%"

    # Every other axis: something must happen during the event.
    if not events:
        return False, "axis expects events but none are labelled"
    if mi < ACTIVE:
        return False, f"nothing moves during the event (max {mi:.2f}%)"
    if mi < mo:
        return False, (f"more motion OUTSIDE the event ({mo:.2f}%) than inside "
                       f"({mi:.2f}%) — label may point at the wrong interval")
    return True, f"moving inside {mi:.2f}%, outside {mo:.2f}% @{fps:.0f}fps"


def check_labels(labels: list[dict], data_dir: Path) -> list[str]:
    """Validate labels.json itself, before trusting anything computed from it.

    Ground truth is the one thing nothing downstream can check. A typo here is
    silently absorbed into every metric.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for i, e in enumerate(labels):
        where = f"entry {i} ({e.get('video', '?')})"
        if "video" not in e:
            problems.append(f"{where}: no 'video' field"); continue
        if e["video"] in seen:
            problems.append(f"{where}: duplicate clip")
        seen.add(e["video"])
        if not (data_dir / e["video"]).exists():
            problems.append(f"{where}: file does not exist")
        if not e.get("axis"):
            problems.append(f"{where}: no axis — it cannot be reported separately")
        dur = e.get("duration_s")
        for j, ev in enumerate(e.get("events", [])):
            tag = f"{where} event {j}"
            d = ev.get("description", "")
            if not d.strip():
                problems.append(f"{tag}: empty description")
            if d.strip().lower() != d.strip() and "_" in d:
                problems.append(f"{tag}: {d!r} looks like a dataset class name, "
                                "not how a client would phrase it")
            s_, e_ = ev.get("start_s"), ev.get("end_s")
            if s_ is None or e_ is None:
                problems.append(f"{tag}: missing start_s/end_s"); continue
            if e_ <= s_:
                problems.append(f"{tag}: end_s {e_} not after start_s {s_}")
            if s_ < 0:
                problems.append(f"{tag}: negative start_s {s_}")
            if dur and e_ > dur + 1e-6:
                problems.append(f"{tag}: ends at {e_}s, past the clip's {dur}s")
    return problems


def main() -> int:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/synthetic")
    labels = json.loads((data_dir / "labels.json").read_text())

    print(f"labels.json — {len(labels)} clip(s), "
          f"{sum(len(e.get('events', [])) for e in labels)} labelled event(s)")
    problems = check_labels(labels, data_dir)
    for p in problems:
        print(f"  FAIL {p}")
    print("  ok   structure, times and phrasing" if not problems else "")
    print()

    print(f"{'clip':<22} {'axis':<19} {'verdict':<7} detail")
    failed = 0
    for entry in labels:
        ok, detail = check(entry, data_dir)
        if not ok:
            failed += 1
        print(f"  {entry['video']:<20} {entry.get('axis','?'):<19} "
              f"{'ok' if ok else 'FAIL':<7} {detail}")
    print()
    total = failed + len(problems)
    print("labels.json is sound and every clip shows what it claims" if not total
          else f"{failed} clip(s) disagree with their labels, "
               f"{len(problems)} problem(s) in labels.json")
    return total


if __name__ == "__main__":
    raise SystemExit(main())
