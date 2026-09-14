#!/usr/bin/env python3
"""Sweep the derivation thresholds against a recorded session. No GPU, no cost.

Observations are replayed ONCE per clip and held in memory; every configuration
then re-derives from the same observations. So the sweep measures derivation
alone -- the perception is identical across every row, which is the only way to
read a difference as a threshold effect rather than as noise.

    python scripts/sweep_derive.py out/ef/exchanges-per_bracket-video.jsonl

This is the half of the system that can be tuned for free, and it should be
exhausted before any GPU time is spent tuning the half that cannot.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventfinder.backends.replay import ReplayReasoner            # noqa: E402
from eventfinder.compile import RulesCompiler, compile_all        # noqa: E402
from eventfinder.config import Config, DeriveConfig, load          # noqa: E402
from eventfinder.derive import derive, rank                        # noqa: E402
from eventfinder.pipeline import _dedupe, plan                     # noqa: E402
from video_reasoning.metrics import CONTROL_AXES, aggregate        # noqa: E402


def collect(corpus: str, labels, cfg: Config, clips_dir: Path, strict: bool = True):
    """Replay every clip once. Returns {video: {probe_id: [Observation]}}."""
    descs = []
    for c in labels:
        for e in c["events"]:
            if e["description"] not in descs:
                descs.append(e["description"])
    r = ReplayReasoner(corpus, strict_prompt=strict)
    out, probes = {}, None
    for clip in labels:
        v = clip["video"]
        path = clips_dir / v
        if not path.exists():
            continue
        r.context = v
        probes, _, tasks, _ = plan(str(path), descs, cfg)
        by: dict[str, list] = {p.id: [] for p in probes if p.expressible}
        for t in tasks:
            obs = r.observe(None, t.stamp_times, t.subject, t.attributes, t.bracket.id)
            for pid in t.probe_ids:
                by[pid].extend(obs)
        out[v] = {pid: _dedupe(o)[0] for pid, o in by.items()}
    return out, probes, descs


def score(collected, probes, labels, dcfg: DeriveConfig, step_s: float):
    axes = {c["video"]: c.get("axis", "unknown") for c in labels}
    truths = [{"video": c["video"], "description": e["description"],
               "start_s": e["start_s"], "end_s": e["end_s"]}
              for c in labels for e in c["events"]
              if axes.get(c["video"]) not in CONTROL_AXES]
    preds = []
    for v, by in collected.items():
        if axes.get(v) in CONTROL_AXES:
            continue
        evs = []
        for p in probes:
            if p.expressible:
                evs.extend(derive(by.get(p.id, []), p, dcfg, step_s))
        for e in rank(evs):
            preds.append({"video": v, "description": e.description,
                          "start_s": e.start_s, "end_s": e.end_s,
                          "confidence": e.confidence})
    return aggregate(preds, truths), len(preds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--labels", default="data/eval/labels.json")
    ap.add_argument("--clips", default="data/eval")
    ap.add_argument("--config", default="eventfinder.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, e.g. --set signal.enabled=false. "
                         "Needed for a corpus recorded before exchanges carried their "
                         "plan_config: a replay re-plans to find bracket ids, and "
                         "re-planning under a different signal config makes every "
                         "lookup miss silently.")
    ap.add_argument("--stale-ok", action="store_true",
                    help="accept a corpus recorded under an older prompt; the "
                         "numbers then describe that prompt, not this one")
    a = ap.parse_args()

    cfg = load(a.config) if Path(a.config).exists() else Config()
    cfg.observe.mode = "per_bracket"
    for item in a.set:
        key, _, raw = item.partition("=")
        val = {"true": True, "false": False}.get(raw.lower(), raw)
        if isinstance(val, str):
            try:
                val = float(val) if "." in val else int(val)
            except ValueError:
                pass
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = getattr(node, k)
        setattr(node, leaf, val)
    cfg.check()
    print(f"  plan: signal.enabled={cfg.signal.enabled} "
          f"max_bracket_s={cfg.signal.max_bracket_s} step_s={cfg.observe.step_s}")
    labels = json.loads(Path(a.labels).read_text())
    print(f"  replaying {Path(a.corpus).name} once ...")
    collected, probes, _ = collect(a.corpus, labels, cfg, Path(a.clips),
                                   strict=not a.stale_ok)
    n_obs = sum(len(o) for by in collected.values() for o in by.values())
    print(f"  {len(collected)} clips, {n_obs} observations held in memory\n")

    grid = dict(
        min_run=[1, 2, 3],
        transition_hold_s=[0.0, 1.0],
        extend_by_motion_s=[2.0, 4.0, 8.0, 12.0],
    )
    base = cfg.derive.model_dump()
    rows = []
    print(f"{'min_run':>7} {'hold':>5} {'extend':>7} | {'preds':>5} {'tIoU':>6} "
          f"{'R@1.3':>6} {'R@1.5':>6} {'rec@.5':>7} {'prec@.5':>8} {'FP':>5}")
    for mr, hold, ext in itertools.product(*grid.values()):
        d = DeriveConfig(**{**base, "min_run": mr, "transition_hold_s": hold,
                            "extend_by_motion_s": ext})
        m, n = score(collected, probes, labels, d, cfg.observe.step_s)
        rows.append((m["mean_tIoU"], m["R@1_tIoU0.3"], n, mr, hold, ext, m))
        print(f"{mr:>7} {hold:>5} {ext:>7} | {n:>5} {m['mean_tIoU']:>6.3f} "
              f"{m['R@1_tIoU0.3']:>6.3f} {m['R@1_tIoU0.5']:>6.3f} "
              f"{m['recall@0.5']:>7.3f} {m['precision@0.5']:>8.3f} "
              f"{m['false_positive_rate']:>5.3f}")

    rows.sort(key=lambda r: (-r[0], -r[1], r[2]))
    print("\n  best mean tIoU, then R@1, then fewest predictions:")
    for t, r1, n, mr, hold, ext, _ in rows[:5]:
        print(f"    min_run={mr} transition_hold_s={hold} extend_by_motion_s={ext}"
              f"  ->  tIoU {t:.3f}  R@1>=0.3 {r1:.3f}  {n} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
