#!/usr/bin/env python3
"""Run a list of named experiments in order, and print one comparison table.

    python scripts/run_experiments.py                      # cost it, run nothing
    python scripts/run_experiments.py --go                  # run them
    python scripts/run_experiments.py --go --only a,c       # a subset
    python scripts/run_experiments.py --replay              # re-score recorded ones, no GPU

**Sequentially, on purpose.** There is one GPU. Running two experiments at once
would have them contend for the same server, so neither the latency figures nor
the throughput figures would describe either one -- and latency is a number this
project needs. Concurrency belongs inside a run (`observe.concurrency`), where
it is measured, not between runs where it is a confound.

Each experiment records its own corpus, so a session that dies partway keeps
everything it paid for, and every completed experiment can be re-scored offline
for free afterwards.

An experiment that already has a corpus is SKIPPED unless --force, because
paying twice for the same answer is the failure this whole record/replay
apparatus exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_events import (                                    # noqa: E402
    build_parser, descriptions_of, infer_from_corpus, one_run, show,
)
from eventfinder.config import Config, load                  # noqa: E402
from eventfinder.pipeline import plan                        # noqa: E402

GPU_HOURLY = 1.22249     # g6.2xlarge, verified
SEC_PER_CALL = 5.0       # measured: 4.68-5.45s across every run so far


def apply(cfg: Config, overrides: dict) -> Config:
    """Dotted overrides onto a config, validated by the same checks a run uses."""
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = getattr(node, key)
        if not hasattr(node, leaf):
            raise KeyError(f"no such config key: {dotted}")
        setattr(node, leaf, value)
    cfg.check()
    return cfg


def eval_args(**kw):
    """Every flag one_run reads, at its default, plus the overrides given.

    Taken from eval_events' own parser rather than hand-listed: a stand-in
    object silently loses each flag added later, which is how three experiments
    failed at once on an attribute that had existed for an hour.
    """
    a = build_parser().parse_args([])
    a.hourly = GPU_HOURLY
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def cost_of(exp: dict, labels, clips: Path, base: Config, descs) -> tuple[int, float]:
    cfg = apply(load_base(base), exp["set"])
    cfg.observe.mode = "per_bracket"
    calls = 0
    for clip in labels:
        v = clips / clip["video"]
        if v.exists():
            _, _, tasks, _ = plan(str(v), descs, cfg)
            calls += len(tasks)
    return calls, calls * SEC_PER_CALL / cfg.observe.concurrency / 3600 * GPU_HOURLY


def load_base(path) -> Config:
    return load(path) if Path(path).exists() else Config()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default="eval/experiments.yaml")
    ap.add_argument("--labels", default="data/eval/labels.json")
    ap.add_argument("--clips", default="data/eval")
    ap.add_argument("--config", default="eventfinder.yaml")
    ap.add_argument("--out", default="out/ef")
    ap.add_argument("--only", default=None, help="comma-separated name prefixes")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--go", action="store_true", help="actually run (default: cost only)")
    ap.add_argument("--replay", action="store_true",
                    help="re-score recorded corpora instead of calling the model")
    ap.add_argument("--force", action="store_true", help="re-run even if a corpus exists")
    ap.add_argument("--max-cost", type=float, default=2.00,
                    help="refuse the whole set above this estimate, in dollars")
    a = ap.parse_args()

    exps = yaml.safe_load(Path(a.experiments).read_text())
    if a.only:
        want = tuple(x.strip() for x in a.only.split(","))
        exps = [e for e in exps if e["name"].startswith(want)]
    if not exps:
        print("no experiments selected", file=sys.stderr)
        return 2

    labels = json.loads(Path(a.labels).read_text())
    all_descs = descriptions_of(labels)
    if a.max_clips:
        labels = labels[:a.max_clips]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = Path(a.clips)

    # --- cost the set before spending any of it --------------------------
    print(f"{'experiment':22s} {'calls':>7} {'est $':>7} {'est min':>8}  status")
    total_calls = total_cost = 0.0
    plans = []
    for e in exps:
        calls, cost = cost_of(e, labels, clips, a.config, all_descs)
        corpus = out_dir / f"exchanges-{e['name']}.jsonl"
        done = corpus.exists() and not a.force
        if not done:
            total_calls += calls
            total_cost += cost
        plans.append((e, calls, cost, corpus, done))
        print(f"{e['name']:22s} {calls:>7} {cost:>7.2f} "
              f"{calls * SEC_PER_CALL / 3 / 60:>7.0f}m  "
              f"{'recorded, will skip' if done else ('replay' if a.replay else 'to run')}")
    print(f"{'TOTAL (new work)':22s} {int(total_calls):>7} {total_cost:>7.2f} "
          f"{total_calls * SEC_PER_CALL / 3 / 60:>7.0f}m")

    if not a.go:
        print("\n  costing only. Add --go to run, or --replay --go to re-score recorded ones.")
        return 0
    if not a.replay and total_cost > a.max_cost:
        print(f"\n  refusing: estimated ${total_cost:.2f} exceeds --max-cost "
              f"${a.max_cost:.2f}", file=sys.stderr)
        return 2

    # --- run them, one at a time -----------------------------------------
    results = []
    for e, calls, cost, corpus, done in plans:
        if done and not a.replay:
            print(f"\n=== {e['name']}: already recorded, skipping ===")
            continue
        if a.replay and not corpus.exists():
            print(f"\n=== {e['name']}: no corpus, skipping ===")
            continue

        print(f"\n=== {e['name']} ===")
        print(f"  {' '.join(e['why'].split())}")
        print(f"  {e['set']}")
        cfg = apply(load_base(a.config), e["set"])
        cfg.observe.mode = "per_bracket"
        args = eval_args(replay=str(corpus) if a.replay else None,
                         stale_ok=a.replay)
        if a.replay:
            mode, media = infer_from_corpus(str(corpus))
            cfg.observe.mode = mode
        t0 = time.perf_counter()
        try:
            r = one_run(labels, clips, cfg, args, cfg.observe.mode,
                        getattr(cfg.model, "media", "video"), False, out_dir,
                        max_clips=a.max_clips, descs=all_descs)
        except Exception as ex:
            print(f"  FAILED: {type(ex).__name__}: {ex}", file=sys.stderr)
            results.append({"name": e["name"], "error": str(ex)})
            continue
        r["name"] = e["name"]
        r["set"] = e["set"]
        r["elapsed_s"] = round(time.perf_counter() - t0, 1)
        # The corpus is written by one_run under its variant tag; link it to the
        # experiment name so --replay and --force can find it next time.
        src = Path(r.get("exchanges", ""))
        if src.exists() and src != corpus:
            corpus.write_text(src.read_text())
            r["exchanges"] = str(corpus)
        results.append(r)
        show(r)

    report = out_dir / "experiments.json"
    report.write_text(json.dumps(results, indent=2))

    good = [r for r in results if "metrics" in r]
    if good:
        print(f"\n  {'experiment':22s} {'tIoU':>6} {'R@1.3':>6} {'rec@.5':>7} "
              f"{'prec@.5':>8} {'preds':>6} {'calls':>6} {'min':>5}")
        for r in good:
            m = r["metrics"]
            print(f"  {r['name']:22s} {m['mean_tIoU']:>6.3f} {m['R@1_tIoU0.3']:>6.3f} "
                  f"{m['recall@0.5']:>7.3f} {m['precision@0.5']:>8.3f} "
                  f"{m['n_predictions']:>6} {r['calls']:>6} {r['elapsed_s'] / 60:>5.1f}")
        print(f"\n  baseline to beat: main scores 0.292 on clips 1-4 / 8 truths")
    print(f"  -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
