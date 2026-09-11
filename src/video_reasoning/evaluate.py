"""Run the service over a hand-labelled set and report defensible metrics.

Two guards that matter more than the arithmetic:

**Stub results are refused.** A run produced by the deterministic stub cannot
become a number in a results table. Development convenience must not be able to
masquerade as evidence, and the only reliable way to ensure that is to make it
impossible rather than discouraged.

**Predictions carry their video.** Metrics group by `(video, description)`, so
every prediction must know which clip it came from or the isolation rule in
`metrics.py` cannot hold.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .backends.base import Backend
from .config import Config
from .core import find_events
from .errors import InvalidInput
from .metrics import CONTROL_AXES, aggregate, by_axis


def load_labels(path: str | Path) -> tuple[list[dict], dict[str, str]]:
    """Read the labels file into flat truths plus a video -> axis map.

    Format, one entry per clip:
        [{"video": "clip.mp4", "axis": "baseline", "duration_s": 20.0,
          "events": [{"description": "...", "start_s": 4.0, "end_s": 8.0}]}]

    A clip with an empty `events` list is a **negative**: the description is asked
    of it and should produce nothing. Those are what measure false positives, so
    they are kept rather than skipped.
    """
    path = Path(path)
    if not path.exists():
        # Name the command that builds THIS set. A single generic hint sent a
        # missing control set to `make data`, which does not build it, and the
        # reader then has to find that out by running the wrong thing.
        p = str(path)
        if "meva-examples" in p or "positive_control" in p:
            hint = ("the ceiling-test set is not here. Fetch MEVA's example "
                    "clips (scripts/fetch_meva.sh examples) or copy "
                    "data/meva-examples from a machine that has them")
        elif "eval" in p:
            hint = "run `make data-eval` to rebuild the labelled set from labels.json"
        else:
            hint = "run `make data-synthetic` to generate the synthetic set"
        raise InvalidInput(f"labels file not found: {path}", fix=hint)
    data = json.loads(path.read_text())
    truths: list[dict] = []
    axes: dict[str, str] = {}
    for item in data:
        video = item["video"]
        axes[video] = item.get("axis", "unknown")
        for e in item.get("events", []):
            truths.append({"video": video, "description": e["description"],
                           "start_s": e["start_s"], "end_s": e["end_s"]})
    return truths, axes


def queries_for(path: str | Path) -> dict[str, list[str]]:
    """Which descriptions to ask of each clip.

    Every description in the set is asked of every clip, not only of the clip it
    was labelled on. Otherwise the system is only ever asked questions whose
    answer is yes, and false positives go unmeasured.
    """
    # Descriptions are collected from the WHOLE labels file, before any --limit
    # narrows the clips. That is deliberate and looks like a bug: on a 4-clip
    # subsample the system is still asked all 13 descriptions, including ones
    # labelled only on clips 5-8. Those extra questions have no true answer on
    # this subset, which is exactly what makes them useful -- they are the
    # negatives. Collecting descriptions after the narrowing would delete most of
    # them and flatter every precision figure on a subsample.
    data = json.loads(Path(path).read_text())
    all_desc = sorted({e["description"] for item in data
                       for e in item.get("events", [])})
    return {item["video"]: all_desc for item in data}


def run_eval(
    labels_path: str | Path, data_dir: str | Path, cfg: Config, backend: Backend,
    *, prompt: str | None = None, progress: bool = True,
    gpu_hourly: float | None = None, limit: int | None = None,
    states_map: dict[str, tuple[str, str]] | None = None,
) -> dict:
    """Evaluate on every labelled clip and report metrics.

    `states_map` is what makes the `states` strategy scoreable. Without it every
    description routes to "not expressible" and the run scores zero -- which would
    read as a model result and is a wiring failure.
    """
    if getattr(backend, "is_stub", False):
        raise InvalidInput(
            "refusing to evaluate with the stub backend: its output is a "
            "deterministic pattern, not a measurement, and scoring it would "
            "produce a number that means nothing.",
            fix="serve a real model and use --backend vllm, or --backend replay "
                "against recorded responses",
        )

    truths, axes = load_labels(labels_path)

    # A labels file is either all control clips or none. Mixed files are refused
    # rather than silently handled: every description in a file is asked of every
    # clip in it, so mixing changes which questions real clips are asked and moves
    # the false-positive denominator. Results would then not be comparable between
    # runs, while looking perfectly normal.
    kinds = {a in CONTROL_AXES for a in axes.values()}
    if len(kinds) > 1:
        control = sorted(v for v, a in axes.items() if a in CONTROL_AXES)
        raise InvalidInput(
            f"{labels_path} mixes control clips with evaluation clips "
            f"({len(control)} control). Every description is asked of every clip, "
            "so mixing shifts the query set and the false-positive denominator.",
            fix="keep control clips in their own labels file and run them with "
                "`make eval-control`",
        )

    plan = queries_for(labels_path)
    # Subsampling exists so a change can be tried on one or two clips before
    # committing a metered GPU to the full cross-product. Clips are taken in
    # sorted order so the subset is reproducible rather than whatever the
    # filesystem returned.
    if limit:
        plan = dict(sorted(plan.items())[:limit])
        # Truths must be narrowed to match. Without this the events of every clip
        # NOT evaluated are still counted as ground truth with no predictions
        # against them, so they score as guaranteed misses and every recall and
        # false-positive figure on a subsample is wrong -- quietly, and in the
        # direction that makes a change look worse than it is.
        truths = [t for t in truths if t["video"] in plan]
    data_dir = Path(data_dir)

    preds: list[dict] = []
    per_video: dict[str, dict] = {}
    # The captions behind every state decision. Written to a sidecar rather than
    # inline: it is the evidence for each event, and it is also ~1.5 MB of prose
    # that would make the metric file unreadable.
    polls_by_video: dict[str, list[dict]] = {}
    t0 = time.time()
    total_video_s = 0.0
    total_calls = 0

    for i, (video, queries) in enumerate(sorted(plan.items()), 1):
        path = data_dir / video
        if progress:
            print(f"  [{i}/{len(plan)}] {video} x {len(queries)} description(s)")
        result = find_events(path, queries, cfg, backend, prompt=prompt,
                             progress=False, states_map=states_map)
        total_video_s += result.duration_s
        total_calls += result.run.model_calls
        if result.polls:
            polls_by_video[video] = result.polls
        for e in result.events:
            d = e.model_dump()
            d["video"] = video          # required by the isolation rule
            preds.append(d)
        per_video[video] = {
            "duration_s": result.duration_s,
            "events": len(result.events),
            "model_calls": result.run.model_calls,
            "elapsed_s": result.run.elapsed_s,
        }

    wall = time.time() - t0

    # Control axes are reported separately and NEVER averaged into the headline.
    # positive_control clips carry MEVA's annotations burned into the picture, so
    # the model can read the answer off the frame; including them would inflate
    # every number while looking entirely legitimate. Enforced here rather than
    # left to a convention, because a convention would eventually be forgotten.
    control_videos = {v for v, a in axes.items() if a in CONTROL_AXES}
    main_truths = [t for t in truths if t["video"] not in control_videos]
    main_preds = [p for p in preds if p["video"] not in control_videos]
    metrics = aggregate(main_preds, main_truths)

    # Coverage, reported as a first-class result rather than a footnote.
    #
    # The `states` strategy declines descriptions that do not decompose into a
    # binary state -- "someone hands an object to another person" is a relation
    # between two actors, and no prompt turns it into a property of one object.
    # There are two defensible numbers and they answer different questions, so
    # BOTH are reported and neither is called "the" score:
    #
    #   overall          every labelled event, a declined description counting as
    #                    a miss. This is what a client experiences.
    #   overall_routed   only the events whose description the strategy accepted.
    #                    This is how well the method works where it applies.
    #
    # Reporting only the second would flatter the system by scoring it solely on
    # the questions it chose to answer. Reporting only the first would hide that
    # the failure is a property of the request, not of the model.
    routed = set(states_map or {})
    if cfg.strategy == "states":
        r_truths = [t for t in main_truths if t["description"] in routed]
        r_preds = [p for p in main_preds if p["description"] in routed]
        coverage = {
            "strategy": "states",
            "descriptions_total": len({t["description"] for t in main_truths}),
            "descriptions_routed": len({t["description"] for t in main_truths}
                                       & routed),
            "events_total": len(main_truths),
            "events_routed": len(r_truths),
            "declined": sorted({t["description"] for t in main_truths} - routed),
        }
        routed_metrics = aggregate(r_preds, r_truths) if r_truths else None
    else:
        coverage, routed_metrics = {"strategy": cfg.strategy}, None

    out = {
        "labels": str(labels_path),
        "model": cfg.model.name,
        "backend": backend.describe(),
        "prompt": prompt or "overlay",
        "strategy": cfg.strategy,
        "sampling": {"fps": cfg.sampling.fps,
                     "window_s": cfg.windowing.window_s,
                     "stride_s": cfg.windowing.stride_s,
                     "step_s": cfg.states.step_s, "span_s": cfg.states.span_s,
                     "trigger": cfg.states.trigger,
                     "shared_caption": cfg.states.shared_caption},
        "coverage": coverage,
        "overall": metrics,
        "overall_routed": routed_metrics,
        "overall_excludes": sorted(CONTROL_AXES & set(axes.values())),
        "by_axis": by_axis(preds, truths, axes),
        "per_video": per_video,
        "cost": _cost(wall, total_video_s, total_calls, gpu_hourly),
        "predictions": preds,
    }
    if polls_by_video:
        out["polls"] = polls_by_video
    return out


def _cost(wall_s: float, video_s: float, calls: int,
          gpu_hourly: float | None) -> dict:
    """Latency and, only if a rate was given, money.

    An unset rate means cost is **not reported**, rather than reported wrongly.
    A plausible-looking dollar figure derived from a guessed hourly rate is worse
    than no figure at all.
    """
    video_min = max(1e-9, video_s / 60.0)
    out = {
        "wall_s": round(wall_s, 2),
        "video_minutes": round(video_min, 3),
        "model_calls": calls,
        "s_per_video_minute": round(wall_s / video_min, 2),
    }
    if gpu_hourly:
        out["usd_per_video_minute"] = round(
            (gpu_hourly / 3600.0) * (wall_s / video_min), 4)
        out["gpu_hourly_usd"] = gpu_hourly
    else:
        out["usd_per_video_minute"] = None
        out["note"] = "GPU_HOURLY unset — cost deliberately not reported"
    return out
