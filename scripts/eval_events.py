#!/usr/bin/env python3
"""Run eventfinder over the labelled set and score it. This is the GPU session.

    # one configuration
    python scripts/eval_events.py --mode per_bracket --media video

    # the matrix: every question the first GPU session exists to answer
    python scripts/eval_events.py --matrix

    # re-score a recorded session without a GPU
    python scripts/eval_events.py --replay out/ef/exchanges-*.jsonl

**Each clip is asked EVERY description in the labelled set**, not only its own.
Asking a clip only about events it contains cannot measure a false positive, and
a precision number computed that way is meaningless. In this architecture that
costs almost nothing extra: probes sharing a subject and attribute list share a
call, so thirteen descriptions collapse to a handful of groups.

Metrics come from `video_reasoning.metrics`, unchanged. They enforce
per-(video, description) isolation in code, which matters because the same
phrase is asked of several clips by design -- pooling would let a prediction
from one clip satisfy a label in another and inflate every number.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventfinder.backends.replay import ReplayReasoner, summarise, write_exchanges  # noqa: E402
from eventfinder.backends.vllm import VLLMReasoner  # noqa: E402
from eventfinder.compile import RulesCompiler, compile_all  # noqa: E402
from eventfinder.config import Config, load  # noqa: E402
from eventfinder.pipeline import plan, run  # noqa: E402
from video_reasoning.metrics import CONTROL_AXES, aggregate, by_axis  # noqa: E402


def infer_from_corpus(path: str) -> tuple[str, str]:
    """Read the mode and media path back out of a recorded session.

    A replay MUST re-plan the run exactly as it was recorded: the observation
    lookup is keyed on (bracket_id, subject, mode), so replaying a per_bracket
    corpus in per_step mode asks for tasks that were never recorded and every
    lookup misses. That is silent -- it looks like a model that answered
    nothing -- so the mode is taken from the corpus and the flag is not
    trusted.

    `times` is the discriminator: one stamp per call is per_step, several is
    per_bracket.
    """
    import json as _json
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            e = _json.loads(line)
            mode = "per_bracket" if len(e.get("times") or []) > 1 else "per_step"
            return mode, e.get("media", "video")
    raise ValueError(f"{path} holds no exchanges")


def descriptions_of(labels: list[dict]) -> list[str]:
    seen: list[str] = []
    for clip in labels:
        for ev in clip["events"]:
            if ev["description"] not in seen:
                seen.append(ev["description"])
    return seen


def truths_of(labels: list[dict]) -> list[dict]:
    return [{"video": c["video"], "description": e["description"],
             "start_s": e["start_s"], "end_s": e["end_s"]}
            for c in labels for e in c["events"]]


def build_cfg(a, mode: str) -> Config:
    c = load(a.config) if Path(a.config).exists() else Config()
    c.observe.mode = mode
    if a.step_s:
        c.observe.step_s = a.step_s
    if a.concurrency:
        c.observe.concurrency = a.concurrency
    if a.crop:
        c.sampling.crop_to_motion = True
    if a.state_prompt:
        c.observe.state_prompt = a.state_prompt
    c.check()
    return c


def reasoner_for(cfg: Config, a, media: str):
    if a.replay:
        return ReplayReasoner(a.replay, strict_prompt=not a.stale_ok)
    return VLLMReasoner(
        cfg.model.base_url, cfg.model.name, api_key=cfg.model.api_key,
        temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens,
        tokens_per_record=cfg.model.tokens_per_record,
        state_prompt=cfg.observe.state_prompt,
        media=media, timeout_s=cfg.model.request_timeout_s,
        sample_fps=(1.0 / cfg.observe.step_s if cfg.observe.mode == "per_bracket"
                    else cfg.sampling.fps),
    )


def one_run(labels, clips_dir: Path, cfg: Config, a, mode: str, media: str,
            verify: bool, out_dir: Path, max_clips: int | None = None,
            descs: list[str] | None = None) -> dict:
    tag = f"{mode}-{media}{'-verify' if verify else ''}{'-crop' if a.crop else ''}{'-' + a.state_prompt if a.state_prompt else ''}"
    # The description set comes from the WHOLE labelled set and is passed in,
    # never derived from the truncated one. Deriving it here meant --max-clips 1
    # asked only that clip's own description: one subject, six calls, no false
    # positives measurable, and nothing asked about the subject the change under
    # test actually concerned.
    if max_clips:
        labels = labels[:max_clips]
        tag += f"-{len(labels)}clips"
    descs = descs or descriptions_of(labels)
    # A dry run plans and costs the work without touching the endpoint, so it
    # must not need one -- that is the point of being able to check the shape of
    # a session before starting the box.
    reasoner = None if a.dry_run else reasoner_for(cfg, a, media)
    ok, why = (True, "dry-run") if reasoner is None else reasoner.verify_model()
    if not ok and not a.replay:
        print(f"  model identity NOT verified: {why}", file=sys.stderr)
        if not a.allow_unverified:
            raise SystemExit("refusing to attribute results to a model that was not confirmed; "
                             "pass --allow-unverified to override")

    preds, docs, t0 = [], [], time.perf_counter()
    # Flushed after EVERY clip, not at the end of the variant. A run you are
    # meant to be able to stop must not throw away what it has already paid
    # for: interrupting the first session lost a clip's worth of replies that
    # had already been bought and could not be re-derived without the GPU.
    ex_path = out_dir / f"exchanges-{tag}.jsonl"
    for clip in labels:
        video = clips_dir / clip["video"]
        if not video.exists():
            print(f"  skipped (not present): {clip['video']}", file=sys.stderr)
            continue
        if a.dry_run:
            _, bs, tasks, info = plan(str(video), descs, cfg)
            print(f"  {clip['video'][:40]:40s} {len(bs):>3} brackets  {len(tasks):>4} calls")
            docs.append({"video": clip["video"], "calls": len(tasks), "brackets": len(bs)})
            continue
        doc = run(str(video), descs, cfg, reasoner, verify=verify)
        (out_dir / f"{tag}--{clip['video']}.json").write_text(doc.model_dump_json(indent=2))
        docs.append(json.loads(doc.model_dump_json()))
        for e in doc.events:
            preds.append({"video": clip["video"], "description": e.description,
                          "start_s": e.start_s, "end_s": e.end_s,
                          "confidence": e.confidence, "partial": e.partial})
        if not a.replay:
            write_exchanges(ex_path, reasoner.usage)
        # A replay runs on whatever machine you are sitting at, with no GPU and
        # no endpoint, so the hourly rate does not apply to it. Printing one
        # anyway is how a free re-score ends up quoted as having cost money.
        if a.replay:
            note = "  no GPU"
        else:
            note = f"  ${(time.perf_counter() - t0) / 3600 * float(a.hourly):.2f} so far"
        print(f"  {clip['video'][:40]:40s} {len(doc.events):>3} events  "
              f"{doc.run.calls:>4} calls  {doc.run.wall_time_s:>6.0f}s{note}")

    wall = time.perf_counter() - t0
    if a.dry_run:
        return {"variant": tag, "calls": sum(d["calls"] for d in docs), "dry_run": True}

    axes = {c["video"]: c.get("axis", "unknown") for c in labels}
    truths = truths_of(labels)
    # Control clips have the answer written on the frame. They are a ceiling
    # test, not a sample of the problem, and scoring them alongside real footage
    # would measure reading rather than seeing.
    scored = [t for t in truths if axes.get(t["video"]) not in CONTROL_AXES]
    scored_preds = [p for p in preds if axes.get(p["video"]) not in CONTROL_AXES]

    u = reasoner.usage
    result = {
        "variant": tag, "mode": mode, "media": media, "verify": verify,
        "model": reasoner.model, "result_class": "replay" if a.replay else "model",
        "model_verified": bool(ok),
        "clips": len(docs), "calls": u.calls, "failures": u.failures,
        "repairs": u.repairs, "wall_s": round(wall, 1),
        "mean_latency_s": u.mean_latency_s,
        "metrics": aggregate(scored_preds, scored),
        "by_axis": by_axis(scored_preds, scored, axes),
    }
    if not a.replay:
        write_exchanges(ex_path, u)
        result["exchanges"] = str(ex_path)
        result["latency_breakdown"] = summarise(ex_path)
    # The model's own verdict, counted but never acted on: this is the number
    # that decides whether `matches` should ever gate an emission.
    verdicts = [v for d in docs for v in d.get("coverage", {}).get("verdicts", [])]
    if verdicts:
        result["verdicts"] = {"n": len(verdicts),
                              "said_yes": sum(1 for v in verdicts if v["matches"])}
    return result


def show(r: dict) -> None:
    print(f"\n  {r['variant']}   [{r['result_class']}"
          f"{'' if r['model_verified'] else ', MODEL NOT VERIFIED'}]")
    m = r["metrics"]
    print(f"    mean tIoU {m['mean_tIoU']:.3f}   R@1>=0.3 {m['R@1_tIoU0.3']:.3f}   "
          f"R@1>=0.5 {m['R@1_tIoU0.5']:.3f}")
    print(f"    recall@0.5 {m['recall@0.5']:.3f}   precision@0.5 {m['precision@0.5']:.3f}   "
          f"FP rate {m['false_positive_rate']:.3f}")
    print(f"    {m['n_predictions']} predictions / {m['n_truths']} truths   "
          f"{r['calls']} calls ({r['failures']} failed, {r['repairs']} repaired)   "
          f"{r['wall_s']:.0f}s   mean call {r['mean_latency_s']:.2f}s")
    if r.get("verdicts"):
        v = r["verdicts"]
        print(f"    model said matches=true on {v['said_yes']}/{v['n']} calls "
              "(recorded, not acted on)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/eval/labels.json")
    ap.add_argument("--clips", default="data/eval")
    ap.add_argument("--config", default="eventfinder.yaml")
    ap.add_argument("--out", default="out/ef")
    ap.add_argument("--mode", default="per_step", choices=["per_step", "per_bracket"])
    ap.add_argument("--media", default="video", choices=["video", "frames"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--step-s", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--state-prompt", default=None,
                    choices=["generic", "examples", "strict"],
                    help="how the state vocabulary is offered to the model")
    ap.add_argument("--crop", action="store_true",
                    help="crop each bracket to where the motion is")
    ap.add_argument("--matrix", action="store_true",
                    help="every question the first GPU session exists to answer")
    ap.add_argument("--replay", default=None, help="score a recorded session, no GPU")
    ap.add_argument("--stale-ok", action="store_true",
                    help="accept a replay corpus recorded under another prompt")
    ap.add_argument("--hourly", default="1.22249",
                    help="GPU $/hour, for the running cost line")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="score only the first N clips (paired across variants)")
    ap.add_argument("--baseline-clips", type=int, default=3,
                    help="clips for the per_step baseline inside --matrix; it is a "
                         "comparison, not a new question, and costs ~9.5x per clip")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no calls")
    ap.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args()

    labels_path = Path(a.labels)
    if not labels_path.exists():
        print(f"no labels at {labels_path}; build the evaluation set first.", file=sys.stderr)
        return 2
    labels = json.loads(labels_path.read_text())
    # Every clip is asked every description in the set, whatever --max-clips
    # says: asking a clip only about events it contains cannot measure a false
    # positive, and a precision number computed that way is meaningless.
    all_descs = descriptions_of(labels)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.replay:
        corpus = Path(a.replay)
        if not corpus.exists():
            print(f"no corpus at {corpus}.", file=sys.stderr)
            here = Path(a.out)
            found = sorted(here.glob("exchanges-*.jsonl")) if here.is_dir() else []
            if found:
                print("  recorded sessions available here:", file=sys.stderr)
                for f in found:
                    print(f"    {f}", file=sys.stderr)
            else:
                print(f"  nothing under {here} either. Record one with a GPU run, "
                      "or copy one across.", file=sys.stderr)
            if not str(corpus).startswith("/"):
                print("  note: paths are resolved inside the container, where the "
                      "host's ./out is mounted at /out.", file=sys.stderr)
            return 2
        mode, media = infer_from_corpus(a.replay)
        if a.stale_ok:
            print("  --stale-ok: this corpus was recorded under an older prompt. The "
                  "numbers describe THAT prompt, and are evidence about derivation "
                  "only.", file=sys.stderr)
        if (a.mode, a.media) not in (("per_step", "video"), (mode, media)):
            print(f"  note: --mode/--media ignored for a replay; the corpus was "
                  f"recorded as {mode}/{media}", file=sys.stderr)
        a.mode, a.media, a.matrix = mode, media, False
        print(f"  replaying {Path(a.replay).name}: {mode} / {media}")

    # The matrix is ordered cheapest-first so a session that runs out of time
    # still answers the question that matters most.
    if a.matrix:
        # (mode, media, verify, clips). per_step costs ~9.5x per clip and is the
        # baseline the new mode is compared against, not a question of its own,
        # so it runs on a subset -- paired, so the comparison still holds.
        variants = [("per_bracket", "video", False, a.max_clips),
                    ("per_bracket", "frames", False, a.max_clips),
                    ("per_bracket", "video", True, a.max_clips),
                    ("per_step", "video", False,
                     min(a.baseline_clips, a.max_clips or a.baseline_clips))]
    else:
        variants = [(a.mode, a.media, a.verify, a.max_clips)]

    results = []
    for mode, media, verify, n_clips in variants:
        cfg = build_cfg(a, mode)
        print(f"\n=== {mode} / {media}{' / verify' if verify else ''}"
              f"{f' / first {n_clips} clips' if n_clips else ''} ===")
        try:
            r = one_run(labels, Path(a.clips), cfg, a, mode, media, verify, out_dir,
                        max_clips=n_clips, descs=all_descs)
        except Exception as e:
            print(f"  variant FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            results.append({"variant": f"{mode}-{media}", "error": str(e)})
            continue
        results.append(r)
        if not a.dry_run:
            show(r)

    report = out_dir / "report.json"
    report.write_text(json.dumps(results, indent=2))
    print(f"\n  -> {report}")

    good = [r for r in results if "metrics" in r]
    if len(good) > 1:
        print("\n  comparison")
        print(f"    {'variant':26s} {'tIoU':>6} {'R@1.3':>6} {'calls':>6} {'call s':>7} {'fail':>5}")
        for r in good:
            print(f"    {r['variant']:26s} {r['metrics']['mean_tIoU']:>6.3f} "
                  f"{r['metrics']['R@1_tIoU0.3']:>6.3f} {r['calls']:>6} "
                  f"{r['mean_latency_s']:>7.2f} {r['failures']:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
