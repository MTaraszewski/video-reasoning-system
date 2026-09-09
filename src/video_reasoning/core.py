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
from .backends.prompts import PromptVariant, get as get_prompt
from .config import Config
from .decode import probe, sample_frames
from .errors import BudgetExceeded, InvalidInput, UnprocessableMedia
from .merge import merge_all
from .schema import Event, FindEventsResult, RunInfo
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


def find_events(
    video: str | Path,
    queries: list[str] | str,
    config: Config,
    backend: Backend,
    *,
    prompt: PromptVariant | str | None = None,
    progress: bool = False,
) -> FindEventsResult:
    """Find events matching plain-language descriptions in a video."""
    t0 = time.time()
    if isinstance(queries, str):
        queries = [queries]

    path, queries = validate_request(video, queries, config)
    meta = probe(path)
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
