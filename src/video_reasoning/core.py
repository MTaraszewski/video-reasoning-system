"""`find_events` — the deliverable.

The brief asks for: *a video file and a plain-language description of what to look
for ... or a list of several such descriptions*, returning *a list of events, each
with a start time, an end time, the description it matched, and a confidence or
ranking signal, in a strict, documented JSON schema.*

That is exactly this call. Everything else in the package exists to serve it, and
the client sees none of it — no frames, no windows, no prompts, no model.
"""
from __future__ import annotations

import time
from pathlib import Path

from .backends.base import Backend, ExtractRequest
from .backends.prompts import DETECT, PromptVariant, get as get_prompt
from .config import Config
from .decode import probe, sample_frames
from .errors import BudgetExceeded, InvalidInput, UnprocessableMedia
from .merge import merge_all
from .motion import change_signal, peaks
from .schema import Event, FindEventsResult, RunInfo
from .states import (StatePoll, boundaries, parse_state, transitions)
from .windows import assign_frames, plan_cost, plan_windows


def validate_request(
    video: str | Path, queries: list[str], cfg: Config
) -> tuple[Path, list[str]]:
    """Stage 1 and 2 of validation: the request, then the media.

    Ordered cheapest first, so a bad request fails in milliseconds rather than
    after a GPU has been paid for.
    """
    path = Path(video)
    if not path.exists():
        raise InvalidInput(f"video not found: {path}", fix="check the path")
    if not path.is_file():
        raise InvalidInput(f"not a file: {path}")

    size = path.stat().st_size
    if size > cfg.limits.max_video_bytes:
        raise InvalidInput(
            f"{path.name} is {size / 1024**3:.2f} GiB, over the "
            f"{cfg.limits.max_video_bytes / 1024**3:.2f} GiB limit.",
            fix="trim the video, or raise limits.max_video_bytes",
        )

    if isinstance(queries, str):
        queries = [queries]
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise InvalidInput(
            "no descriptions given.",
            fix='pass at least one, e.g. -q "a person enters through the door"',
        )
    if len(queries) > cfg.limits.max_queries:
        raise InvalidInput(
            f"{len(queries)} descriptions, over the limit of "
            f"{cfg.limits.max_queries}."
        )
    for q in queries:
        if len(q) > cfg.limits.max_query_chars:
            raise InvalidInput(
                f"description is {len(q)} characters, over "
                f"{cfg.limits.max_query_chars}: {q[:60]}..."
            )
    # Duplicates are rejected rather than silently deduplicated: a repeat almost
    # always means the caller made a mistake, and hiding it hides the mistake.
    if len(set(queries)) != len(queries):
        dupes = {q for q in queries if queries.count(q) > 1}
        raise InvalidInput(f"duplicate descriptions: {sorted(dupes)}")

    meta = probe(path)
    if meta.duration_s < cfg.limits.min_duration_s:
        raise UnprocessableMedia(
            f"{path.name} is {meta.duration_s}s, under the minimum "
            f"{cfg.limits.min_duration_s}s.",
            fix="the file may be truncated",
        )
    if meta.duration_s > cfg.limits.max_duration_s:
        raise UnprocessableMedia(
            f"{path.name} is {meta.duration_s}s, over the maximum "
            f"{cfg.limits.max_duration_s}s.",
            fix="trim it, or raise limits.max_duration_s",
        )
    return path, queries


def check_budget(duration_s: float, n_queries: int, cfg: Config) -> dict:
    """Stage 4: refuse work too large to be an accident.

    Everything here is known from the inputs before a single model call, so an
    oversized job is a decision rather than something discovered an hour in.
    """
    cost = plan_cost(
        duration_s, cfg.windowing.window_s, cfg.windowing.stride_s,
        n_queries, cfg.sampling.fps,
    )
    if cost["model_calls"] > cfg.limits.max_model_calls:
        raise BudgetExceeded(
            f"this request needs {cost['model_calls']} model calls "
            f"({cost['windows']} windows x {n_queries} descriptions), over the "
            f"limit of {cfg.limits.max_model_calls}.",
            fix="shorten the video, use fewer descriptions, widen window_s/stride_s, "
                "or raise limits.max_model_calls if you mean it",
        )
    return cost


def _poll_states(
    video: Path, duration: float, states: tuple[str, str], config: Config,
    backend: Backend, progress: bool,
) -> list[StatePoll]:
    """Caption the clip on a grid, or only where something changed."""
    cfg = config.states
    if cfg.trigger:
        # Spend calls where the picture moved. The signal is cheap and its blind
        # spots are documented in motion.py -- a missed spike is a missed event
        # the uniform grid would have caught, which is why this is off by default.
        times = peaks(change_signal(str(video), fps=cfg.trigger_fps),
                      top_k=cfg.trigger_top_k, min_gap_s=cfg.trigger_min_gap_s)
    else:
        times = [t for t in _frange(0.0, duration - cfg.span_s, cfg.step_s)]

    polls: list[StatePoll] = []
    subject = states[0].replace("the ", "").split(" is ")[0]
    for i, t in enumerate(times):
        _, frames = sample_frames(
            video, fps=config.sampling.fps, max_side=config.sampling.frame_max_side,
            overlay=False, start_s=t, end_s=t + cfg.span_s,
        )
        if not frames:
            continue
        text, truncated = backend.caption(frames, subject)
        polls.append(StatePoll(t=round(t, 3), text=text,
                               state=parse_state(text, states),
                               truncated=truncated))
        if progress:
            print(f"  [{i + 1}/{len(times)}] t={t:.1f}s  {polls[-1].state}")
    return polls


def _frange(lo: float, hi: float, step: float) -> list[float]:
    out, t = [], lo
    while t <= hi + 1e-9:
        out.append(round(t, 3))
        t += step
    return out


def _events_from_states(
    query: str, polls: list[StatePoll], states: tuple[str, str], duration: float,
) -> list[Event]:
    """Turn a state timeline into client-facing events.

    The event is the interval spent in the target state: entered at a transition
    into it, left at the transition out, or open-ended if the span ends first.

    Confidence is the fraction of polls inside the interval that agree on the
    target state. That is a real measurement over data we already have, unlike
    the model's own stated confidence -- which came back as exactly 1.0 on 28 of
    81 predictions in Approach 1 and ranked nothing.
    """
    target = states[1]
    trs = transitions(polls)
    bnd = boundaries(polls, states)
    spans: list[tuple[float, float, bool]] = []   # start, end, partial

    open_at = 0.0 if bnd.partial_before else None
    for tr in trs:
        if tr.to_state == target and open_at is None:
            open_at = tr.at
        elif tr.from_state == target and open_at is not None:
            spans.append((open_at, tr.at, open_at == 0.0 and bnd.partial_before))
            open_at = None
    if open_at is not None:
        spans.append((open_at, duration, True))

    events: list[Event] = []
    for start, end, partial in spans:
        inside = [p for p in polls if start <= p.t <= end and p.state is not None]
        agree = (sum(p.state == target for p in inside) / len(inside)) if inside else 0.0
        events.append(Event(
            description=query, start_s=round(start, 3), end_s=round(min(end, duration), 3),
            confidence=round(agree, 4),
            evidence=next((p.text[:200] for p in inside if p.state == target), ""),
            partial=partial or end >= duration - 1e-6,
            source_windows=[],
        ))
    return events


def find_events(
    video: str | Path,
    queries: list[str] | str,
    config: Config,
    backend: Backend,
    *,
    prompt: PromptVariant | str | None = None,
    progress: bool = False,
    states_map: dict[str, tuple[str, str]] | None = None,
) -> FindEventsResult:
    """Find events matching plain-language descriptions in a video."""
    t0 = time.time()
    if isinstance(queries, str):
        queries = [queries]

    path, queries = validate_request(video, queries, config)
    meta = probe(path)

    if config.strategy == "states":
        return _find_events_states(path, queries, config, backend, meta,
                                   states_map or {}, progress, t0)

    check_budget(meta.duration_s, len(queries), config)

    variant = prompt if isinstance(prompt, PromptVariant) else get_prompt(prompt)

    duration, frames = sample_frames(
        path,
        fps=config.sampling.fps,
        max_side=config.sampling.frame_max_side,
        overlay=config.overlay.enabled,
        font_scale=config.overlay.font_scale,
        fmt=config.overlay.format,
        position=config.overlay.position,
    )

    windows = plan_windows(duration, config.windowing.window_s,
                           config.windowing.stride_s)
    assign_frames(windows, frames, config.sampling.max_frames_per_window)

    # Edges of the region each window actually observed. The merge uses these to
    # tell "the event ended here" from "this is where we stopped looking".
    observed_edges = {w.start_s for w in windows} | {w.end_s for w in windows}

    per_query: dict[str, list] = {q: [] for q in queries}
    calls = 0
    total = len(windows) * len(queries)

    for w in windows:
        for q in queries:
            req = ExtractRequest(
                window=w, query=q,
                system_prompt=variant.system,
                user_prompt=variant.user(q, w.start_s, w.end_s),
                video=path.name, prompt_variant=variant.name,
                # Present only when two-stage extraction is on. The backend asks
                # this first and localises only if the answer is yes.
                detect_prompt=(
                    (DETECT.system, DETECT.user(q, w.start_s, w.end_s))
                    if config.detect.enabled else None
                ),
            )
            res = backend.extract(req)
            calls += 1
            for ev in res.events:
                per_query[q].append((ev, w.index))
            if progress:
                note = f" [{res.error}]" if res.error else ""
                print(f"  [{calls}/{total}] window {w.index} "
                      f"({w.start_s:.1f}-{w.end_s:.1f}s) x {q!r}: "
                      f"{len(res.events)} hit(s){note}")

    events: list[Event] = merge_all(per_query, config.merge, observed_edges, duration)

    return FindEventsResult(
        video=path.name,
        duration_s=duration,
        queries=queries,
        events=events,
        run=RunInfo(
            model=config.model.name if not backend.is_stub else "stub",
            backend="stub" if backend.is_stub else "vllm",
            sample_fps=config.sampling.fps,
            window_s=config.windowing.window_s,
            stride_s=config.windowing.stride_s,
            windows=len(windows),
            model_calls=calls,
            elapsed_s=round(time.time() - t0, 3),
            stub=backend.is_stub,
        ),
    )


def _find_events_states(
    path: Path, queries: list[str], config: Config, backend: Backend, meta,
    states_map: dict[str, tuple[str, str]], progress: bool, t0: float,
) -> FindEventsResult:
    """Approach 3: caption, parse the state, derive events from transitions.

    A description with no state pair is NOT scored as "no events found". It is
    reported as unanswerable by this strategy, because the failure is a property
    of the request -- "someone hands an object to another person" is a relation
    between two actors, not a binary property of one object, and no prompt turns
    it into one. Five of fifteen hand-labelled descriptions are like that.
    """
    if not hasattr(backend, "caption"):
        raise InvalidInput(
            f"backend {backend.describe().get('backend', '?')} cannot caption, "
            f"which the 'states' strategy requires.",
            fix="use --backend vllm, or set strategy=windows",
        )

    events: list[Event] = []
    calls = 0
    unanswerable: list[str] = []
    for q in queries:
        states = states_map.get(q)
        if not states:
            unanswerable.append(q)
            if progress:
                print(f"  {q!r}: no state pair — not expressible as a state "
                      f"transition, skipped rather than reported as absent")
            continue
        if progress:
            print(f"  {q!r} -> {states[0]} / {states[1]}")
        polls = _poll_states(path, meta.duration_s, tuple(states), config,
                             backend, progress)
        calls += len(polls)
        events.extend(_events_from_states(q, polls, tuple(states),
                                          meta.duration_s))

    return FindEventsResult(
        video=path.name, duration_s=meta.duration_s, queries=queries,
        events=events,
        run=RunInfo(
            model=config.model.name if not backend.is_stub else "stub",
            backend="stub" if backend.is_stub else "vllm",
            sample_fps=config.sampling.fps,
            window_s=config.states.span_s,
            stride_s=config.states.step_s,
            windows=calls, model_calls=calls,
            elapsed_s=round(time.time() - t0, 3), stub=backend.is_stub,
        ),
    )
