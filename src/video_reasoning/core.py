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
from .states import (StatePoll, boundaries, edge, parse_state, transitions)
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


def _poll_times(duration: float, video: Path, cfg) -> list[float]:
    """Where to poll: a fixed grid, or the peaks of the change signal."""
    if cfg.trigger:
        # Spend calls where the picture moved. The signal is cheap and its blind
        # spots are documented in motion.py -- a missed spike is a missed event
        # the uniform grid would have caught, which is why this is off by default.
        return peaks(change_signal(str(video), fps=cfg.trigger_fps),
                     top_k=cfg.trigger_top_k, min_gap_s=cfg.trigger_min_gap_s)
    return [t for t in _frange(0.0, duration - cfg.span_s, cfg.step_s)]


def _subject(states: tuple[str, str]) -> str:
    return states[0].replace("the ", "").split(" is ")[0]


def _poll_states(
    video: Path, duration: float, groups: list[tuple[str, str]], config: Config,
    backend: Backend, progress: bool,
) -> tuple[dict[tuple[str, str], list[StatePoll]], int]:
    """Caption the clip and parse a state timeline for every group.

    Two modes, same timestamps and therefore the same resolution:

    - one call per subject per timestep (default), each asking about that subject
    - one call per timestep covering all subjects (`shared_caption`)

    Returns the timelines and the number of model calls made -- which is no longer
    `len(polls)` once one call answers several groups.
    """
    cfg = config.states
    times = _poll_times(duration, video, cfg)
    subjects = [_subject(g) for g in groups]
    shared = cfg.shared_caption and len(groups) > 1

    out: dict[tuple[str, str], list[StatePoll]] = {g: [] for g in groups}
    calls = 0
    for i, t in enumerate(times):
        _, frames = sample_frames(
            video, fps=config.sampling.fps, max_side=config.sampling.frame_max_side,
            overlay=False, start_s=t, end_s=t + cfg.span_s,
        )
        if not frames:
            continue
        if shared:
            texts, truncated = backend.caption_many(frames, subjects)
            calls += 1
            per_group = [(g, texts.get(s, ""), truncated)
                         for g, s in zip(groups, subjects)]
        else:
            per_group = []
            for g, s in zip(groups, subjects):
                text, truncated = backend.caption(frames, s)
                calls += 1
                per_group.append((g, text, truncated))
        for g, text, truncated in per_group:
            out[g].append(StatePoll(t=round(t, 3), text=text,
                                    state=parse_state(text, g),
                                    truncated=truncated))
        if progress:
            shown = "  ".join(f"{_subject(g)}={out[g][-1].state}" for g in groups)
            print(f"  [{i + 1}/{len(times)}] t={t:.1f}s  {shown}")
    return out, calls


def _frange(lo: float, hi: float, step: float) -> list[float]:
    out, t = [], lo
    while t <= hi + 1e-9:
        out.append(round(t, 3))
        t += step
    return out


def _coverage(times: list[float], start: float, end: float,
              span_s: float) -> float:
    """Fraction of the interval that a supporting poll actually observed.

    A poll at t saw `[t, t + span_s]`. Merging those windows and clipping to the
    interval gives the seconds of the reported event that were genuinely looked
    at, as opposed to inferred by carrying a state across a gap.

    Measured on admin.G326 under shared captioning: a "vehicle door opens" event
    was reported as 5.0-97.0s at confidence 1.0 on the strength of four polls --
    six observed seconds out of ninety-two. Coverage puts that at 0.065.
    """
    length = end - start
    if length <= 0:
        return 1.0
    spans = sorted((max(t, start), min(t + span_s, end)) for t in times)
    seen, hi = 0.0, start
    for lo, up in spans:
        if up <= hi:
            continue
        seen += up - max(lo, hi)
        hi = up
    return min(1.0, round(seen / length, 4))


def _events_from_states(
    query: str, polls: list[StatePoll], states: tuple[str, str], duration: float,
    *, max_bracket_s: float, span_s: float, step_s: float,
) -> list[Event]:
    """Turn a state timeline into client-facing events.

    The event is the interval spent in the target state: entered at a transition
    into it, left at the transition out, or open-ended if the span ends first.
    Each boundary is placed by `states.edge`, which interpolates inside a narrow
    bracket and refuses to inside a wide one.

    Confidence is `agreement x sharpness x coverage`. Each factor catches a
    failure that was actually observed, and none is a number the model asserts
    about itself -- Approach 1's merged confidence clustered on constants from its
    own merge code, 48 of 81 predictions carrying the ceiling or the ceiling times
    a default, and ranked nothing.

    - **agreement** -- the fraction of informative polls inside the interval that
      call it the target state.
    - **sharpness** -- how tightly the boundaries are pinned, `step_s` over the
      widest bracket interpolated across. A transition bracketed to one step is
      worth more than the same transition bracketed to thirty.
    - **coverage** -- how much of the interval was actually looked at: the union
      of the supporting polls' windows, over the interval's length. Without it, a
      92-second span resting on four polls scored 1.0, because every poll agreed
      and both edges had been truncated rather than interpolated. A single-poll
      blip scored 1.0 for the same reason from the other direction.
    """
    target = states[1]
    trs = transitions(polls)
    bnd = boundaries(polls, states)

    # start, end, partial, widest bracket actually interpolated across
    spans: list[tuple[float, float, bool, float]] = []
    start: float | None = 0.0 if bnd.partial_before else None
    partial = bnd.partial_before
    widest = 0.0

    for tr in trs:
        if tr.to_state == target and start is None:
            start, guessed = edge(tr, max_bracket_s=max_bracket_s,
                                  span_s=span_s, opening=True)
            partial, widest = guessed, (0.0 if guessed else tr.width)
        elif tr.from_state == target and start is not None:
            end, guessed = edge(tr, max_bracket_s=max_bracket_s,
                                span_s=span_s, opening=False)
            spans.append((start, end, partial or guessed,
                          max(widest, 0.0 if guessed else tr.width)))
            start, partial, widest = None, False, 0.0
    if start is not None:
        spans.append((start, duration, True, widest))

    events: list[Event] = []
    for start, end, partial, widest in spans:
        inside = [p for p in polls if start <= p.t <= end and p.state is not None]
        agree = (sum(p.state == target for p in inside) / len(inside)) if inside else 0.0
        # A truncated edge contributes no width here: the reported interval is
        # exactly the observed evidence, and what is unknown beyond it is carried
        # by `partial` rather than discounted twice.
        sharp = step_s / max(step_s, widest) if widest else 1.0
        cover = _coverage([p.t for p in inside if p.state == target],
                          start, end, span_s)
        events.append(Event(
            description=query, start_s=round(start, 3), end_s=round(min(end, duration), 3),
            confidence=round(agree * sharp * cover, 4),
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
    if config.states.shared_caption and not hasattr(backend, "caption_many"):
        raise InvalidInput(
            f"backend {backend.describe().get('backend', '?')} cannot caption "
            f"several subjects at once, which states.shared_caption requires.",
            fix="use --backend vllm, or set states.shared_caption=false",
        )

    events: list[Event] = []
    calls = 0
    unanswerable: list[str] = []

    # Poll once per distinct STATE SET, not once per description. Several
    # descriptions share a subject -- "opens a building door", "enters through the
    # door" and "comes out through the door" all reduce to closed/open on the same
    # door -- and polling each one separately sends identical frames with an
    # identical question and pays for the identical answer. On the labelled set
    # that is 9 descriptions over 4 subjects: a 2.3x saving with no change to
    # polling density, resolution, or any reported number.
    #
    # Keyed on the SET, so "sits down" (standing -> sitting) and "stands up"
    # (sitting -> standing) share one sweep: the poll asks what state the person
    # is in, and which direction counts as the event is decided afterwards, in
    # `_events_from_states`.
    by_states: dict[frozenset[str], list[str]] = {}
    for q in queries:
        states = states_map.get(q)
        if not states:
            unanswerable.append(q)
            if progress:
                print(f"  {q!r}: no state pair — not expressible as a state "
                      f"transition, skipped rather than reported as absent")
            continue
        by_states.setdefault(frozenset(states), []).append(q)

    groups = [tuple(states_map[qs[0]]) for qs in by_states.values()]
    if progress:
        for g, qs in zip(groups, by_states.values()):
            print(f"  {g[0]} / {g[1]}")
            for q in qs:
                print(f"    <- {q!r}")
        if config.states.shared_caption and len(groups) > 1:
            print(f"  shared caption: {len(groups)} subjects in one call per "
                  f"timestep")

    # Every description was unanswerable: there is nothing to look for, so do not
    # decode a single frame. Without this the grid is still walked, sampling
    # frames for no subject at all.
    # The work budget, which the windows path checks in `check_budget` and this
    # path did not check at all. It cannot be checked in `find_events` alongside
    # the other one: the cost here is subjects x polls, and how many subjects
    # there are is only known after routing has collapsed descriptions onto state
    # pairs. So it is checked here, at the first moment the number exists.
    planned = len(groups) * len(_poll_times(meta.duration_s, path, config.states))
    if planned > config.limits.max_model_calls:
        raise BudgetExceeded(
            f"this request needs {planned} model calls "
            f"({len(groups)} subject(s) x {planned // max(len(groups), 1)} polls), "
            f"over the limit of {config.limits.max_model_calls}.",
            fix="raise states.step_s to poll less often, ask fewer descriptions, "
                "shorten the video, or raise limits.max_model_calls if you mean it",
        )

    timelines, calls = ({}, 0) if not groups else _poll_states(
        path, meta.duration_s, groups, config, backend, progress)
    polls_out = [
        {"t": p.t, "subject": _subject(g), "state": p.state,
         "truncated": p.truncated, "text": p.text}
        for g in groups for p in timelines[g]
    ]
    for g, qs in zip(groups, by_states.values()):
        for q in qs:
            # Per description, because the TARGET state differs within a group:
            # for "sits down" it is sitting, for "stands up" it is standing.
            events.extend(_events_from_states(
                q, timelines[g], tuple(states_map[q]), meta.duration_s,
                max_bracket_s=config.states.carry_steps * config.states.step_s,
                span_s=config.states.span_s, step_s=config.states.step_s))

    return FindEventsResult(
        video=path.name, duration_s=meta.duration_s, queries=queries,
        events=events, polls=polls_out,
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
