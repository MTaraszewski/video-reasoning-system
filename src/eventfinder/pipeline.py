"""compile -> signal -> observe -> derive, over a whole video.

The batch path. It shares every component with the live path in `stream.py`;
what differs is only where brackets come from and when results are emitted.
Keeping one implementation of observation and derivation is deliberate -- a
streaming result and a batch result that disagree would be impossible to debug.

Two call shapes, chosen by `observe.mode`, and the choice is the biggest open
question in the system:

  per_step     one model call per stamp time, frames spanning [t, t+span_s].
  per_bracket  one call for the whole bracket, one frame per stamp time, and
               the model returns a state for each.

Both are implemented over identical brackets and identical stamp times, so a
comparison isolates the request shape and nothing else.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .compile import RulesCompiler, compile_all
from .config import Config
from .decode import Frame, VideoInfo, probe as probe_video, sample_frames
from .derive import derive, rank
from .models import (
    Bracket,
    EventsDocument,
    Observation,
    Probe,
    Rejection,
    RunInfo,
)
from .signal import brackets as make_brackets, coverage_of


@dataclass
class _Task:
    """One model call: which frames, which times, about which subject."""

    bracket: Bracket
    subject: str
    attributes: list[str]
    stamp_times: list[float]
    probe_ids: list[str]
    start_s: float
    end_s: float


def _group(probes: list[Probe]) -> dict[tuple[str, tuple[str, ...]], list[Probe]]:
    """Probes that ask the same thing of the same subject share a call.

    On the labelled set nine descriptions reduce to a handful of subjects, and
    the frames are identical whichever probe is asking -- paying twice for the
    same perception is pure waste. Grouping is by (subject, attributes) rather
    than by subject alone so a probe never receives a field it did not ask for.
    """
    out: dict[tuple[str, tuple[str, ...]], list[Probe]] = {}
    for p in probes:
        if p.expressible:
            out.setdefault((p.subject, tuple(p.attributes)), []).append(p)
    return out


def plan(video: str, descriptions: list[str], cfg: Config,
         compiler=None) -> tuple[list[Probe], list[Bracket], list[_Task], VideoInfo]:
    """Everything the run will do, before a single model call is made.

    Exposed on its own so cost can be inspected -- and refused -- in advance
    rather than discovered in the bill.
    """
    info = probe_video(video)
    probes = compile_all(descriptions, compiler or RulesCompiler())
    bs = make_brackets(video, info.duration_s, cfg.signal)
    groups = _group(probes)

    tasks: list[_Task] = []
    for b in bs:
        times = _stamp_times(b, cfg)
        if not times:
            continue
        for (subject, attrs), ps in groups.items():
            ids = [p.id for p in ps]
            if cfg.observe.mode == "per_bracket":
                tasks.append(_Task(b, subject, list(attrs), times, ids, b.start_s, b.end_s))
            else:
                for t in times:
                    tasks.append(_Task(b, subject, list(attrs), [t], ids,
                                       t, min(t + cfg.observe.span_s, b.end_s)))
    return probes, bs, tasks, info


def _dedupe(obs: list[Observation]) -> tuple[list[Observation], int]:
    """One observation per timestamp per probe.

    Adjacent brackets share their boundary second, so that moment is observed
    twice. Two readings of the same instant that disagree would look to the
    derivation like a state flip and back -- a phantom transition manufactured
    by the bracketing, not seen in the video. Keep the usable one, then the more
    certain one, and count the collisions so the rate stays visible.
    """
    best: dict[float, Observation] = {}
    conflicts = 0
    for o in obs:
        t = round(o.t, 3)
        prev = best.get(t)
        if prev is None:
            best[t] = o
            continue
        conflicts += 1
        if (o.ok, o.certainty) > (prev.ok, prev.certainty):
            best[t] = o
    return sorted(best.values(), key=lambda o: o.t), conflicts


def _stamp_times(b: Bracket, cfg: Config) -> list[float]:
    step = cfg.observe.step_s
    out, t = [], b.start_s
    while t <= b.end_s + 1e-6:
        out.append(round(t, 3))
        t += step
    # A bracket shorter than one step still deserves one look at its start.
    return out[: cfg.sampling.max_frames_per_call] or [round(b.start_s, 3)]


def _frames_for(video: str, task: _Task, cfg: Config) -> list[Frame]:
    """Frames carrying their own timestamps.

    In `per_bracket` mode exactly one frame per stamp time, so the times the
    model is asked about and the frames it is shown are the same list. In
    `per_step` mode a short span at the sampling rate, which gives the model
    motion to look at within the moment.
    """
    o, s = cfg.observe, cfg.sampling
    if cfg.observe.mode == "per_bracket":
        fps = 1.0 / o.step_s
    else:
        fps = s.fps
    return sample_frames(
        video, fps=fps, max_side=s.frame_max_side,
        start_s=task.start_s, end_s=task.end_s, seek=s.seek,
        overlay=cfg.overlay.enabled, font_scale=cfg.overlay.font_scale,
        fmt=cfg.overlay.format, position=cfg.overlay.position,
        max_frames=s.max_frames_per_call,
    )


def run(video: str, descriptions: list[str], cfg: Config, reasoner,
        *, compiler=None, verify: bool = False,
        on_progress=None) -> EventsDocument:
    """Find the described events. Returns the document, whatever went wrong.

    A failed call is a recorded failure, not an exception: a run that observed
    nine brackets out of ten should say so and score what it has, rather than
    lose the nine.
    """
    t0 = time.perf_counter()
    probes, bs, tasks, info = plan(video, descriptions, cfg, compiler)
    expressible = [p for p in probes if p.expressible]

    if len(tasks) > cfg.limits.max_model_calls:
        raise ValueError(
            f"this run needs {len(tasks)} model calls, over the limit of "
            f"{cfg.limits.max_model_calls}. Raise limits.max_model_calls, shorten the "
            f"video, or set observe.mode=per_bracket "
            f"({'already set' if cfg.observe.mode == 'per_bracket' else 'currently per_step'})."
        )

    ok, verified_id = reasoner.verify_model()
    by_probe: dict[str, list[Observation]] = {p.id: [] for p in expressible}
    verdicts: list[dict] = []

    def do(task: _Task):
        frames = _frames_for(video, task, cfg)
        if not frames:
            return task, [], None
        if verify:
            # One description per call: `matches` is about a specific event, so
            # a grouped call cannot carry a shared verdict.
            desc = next((p.description for p in expressible if p.id == task.probe_ids[0]), "")
            v = reasoner.verify(frames, task.stamp_times, task.subject,
                                task.attributes, task.bracket.id, desc)
            return task, v.observations, v
        return task, reasoner.observe(frames, task.stamp_times, task.subject,
                                      task.attributes, task.bracket.id), None

    with ThreadPoolExecutor(max_workers=cfg.observe.concurrency) as pool:
        for i, (task, obs, v) in enumerate(pool.map(do, tasks), 1):
            for pid in task.probe_ids:
                by_probe[pid].extend(obs)
            if v is not None and v.matches is not None:
                # Recorded beside the derived answer, never gating it. Which one
                # is better is the measurement; until it is made, code decides.
                verdicts.append({"bracket_id": task.bracket.id, "probe_id": task.probe_ids[0],
                                 "matches": v.matches, "says": v.says,
                                 "model_confidence": v.confidence})
            if on_progress:
                on_progress(i, len(tasks), task)

    conflicts = 0
    events = []
    for p in expressible:
        obs, c = _dedupe(by_probe[p.id])
        conflicts += c
        by_probe[p.id] = obs
        events.extend(derive(obs, p, cfg.derive, cfg.observe.step_s))
    events = rank(events)

    u = reasoner.usage
    asked = sum(len(t.stamp_times) * len(t.probe_ids) for t in tasks)
    got = sum(sum(1 for o in v if o.ok) for v in by_probe.values())
    return EventsDocument(
        video={"path": str(video), "name": Path(video).name,
               "duration_s": info.duration_s, "width": info.width, "height": info.height},
        probes=probes,
        rejected=[Rejection(probe_id=p.id, description=p.description, reason=p.reason)
                  for p in probes if not p.expressible],
        events=events,
        brackets=bs,
        coverage={
            "clip_fraction_bracketed": coverage_of(bs, info.duration_s),
            "observations_asked": asked,
            "observations_returned": got,
            "failures": u.failures,
            "repairs": u.repairs,
            "duplicate_times": conflicts,
            "verdicts": verdicts,
        },
        run=RunInfo(
            model=reasoner.model, backend=reasoner.name,
            sample_fps=cfg.sampling.fps, observe_step_s=cfg.observe.step_s,
            observe_span_s=cfg.observe.span_s,
            signal=cfg.signal.model_dump(), derive=cfg.derive.model_dump(),
            calls=u.calls, tokens_in=u.tokens_in, tokens_out=u.tokens_out,
            wall_time_s=round(time.perf_counter() - t0, 3),
            model_verified=bool(ok),
            result_class="replay" if reasoner.name == "replay" else "model",
        ),
    )
