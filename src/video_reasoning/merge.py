"""Stitch per-window candidates into final events.

This is the part of the brief that says: *"how to merge or de-duplicate events
that straddle window boundaries, and how to report an event the model saw only
partially."*

Overlap guarantees duplicates — a real event is seen by every window covering it.
Three problems follow, and each is solved deliberately here.

**1. De-duplication must be bounded.** The obvious rule — merge a candidate into
the running span if it overlaps or nearly touches it — chains. The running span's
end keeps advancing, so on a busy video every candidate joins one ever-growing
event until a single detection covers the whole clip. `max_span_s` bounds it.

**2. Confidence must not saturate.** Agreement across independent windows is the
best cheap evidence available. But if agreement only ever raises confidence,
everything converges on 1.0 and the ranking signal the brief asks for stops
discriminating. Agreement raises the score toward a ceiling below 1, so ordering
survives.

**3. Partially-seen events must be reported as such.** After merging, an event can
still touch the edge of the region actually observed. Reporting a clipped span as
if it were exact is a quiet lie, so it is flagged instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import MergeConfig
from .schema import Event, WindowEvent


@dataclass
class Candidate:
    """One window's claim about one query."""

    start_s: float
    end_s: float
    confidence: float
    evidence: str
    window_index: int


def t_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Temporal intersection over union."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _agree(scores: list[float], ceiling: float = 0.97) -> float:
    """Combine per-window scores into one ranking signal.

    Noisy-OR pushed toward a ceiling below 1.0. Independent agreement raises the
    score, but never to certainty — and because the ceiling is shared, events with
    more agreement still order above events with less.
    """
    if not scores:
        return 0.0
    inv = 1.0
    for s in scores:
        inv *= (1.0 - max(0.0, min(1.0, s)))
    return round(ceiling * (1.0 - inv), 4)


def merge_query(
    cands: list[Candidate], cfg: MergeConfig, observed_edges: set[float],
    video_end_s: float,
) -> list[Event]:
    """Merge one query's candidates. Never merges across queries."""
    if not cands:
        return []

    cands = sorted(cands, key=lambda c: (c.start_s, c.end_s))
    groups: list[list[Candidate]] = [[cands[0]]]

    for c in cands[1:]:
        cur = groups[-1]
        cur_start = min(x.start_s for x in cur)
        cur_end = max(x.end_s for x in cur)

        overlaps = t_iou((cur_start, cur_end), (c.start_s, c.end_s)) >= cfg.iou
        adjacent = (c.start_s - cur_end) <= cfg.gap_s
        # The bound that stops runaway chaining. Without it a dense candidate
        # stream collapses into one event spanning the whole video.
        within_bound = (max(cur_end, c.end_s) - cur_start) <= cfg.max_span_s

        if (overlaps or adjacent) and within_bound:
            cur.append(c)
        else:
            groups.append([c])

    events: list[Event] = []
    for g in groups:
        start = min(c.start_s for c in g)
        end = max(c.end_s for c in g)
        windows = sorted({c.window_index for c in g})
        best = max(g, key=lambda c: c.confidence)

        events.append(
            Event(
                description="",
                start_s=round(start, 3),
                end_s=round(end, 3),
                confidence=_agree([c.confidence for c in g]),
                evidence=best.evidence,
                partial=_is_partial(start, end, windows, observed_edges, video_end_s),
                source_windows=windows,
            )
        )
    return events


def _is_partial(
    start: float, end: float, windows: list[int], edges: set[float],
    video_end_s: float, tol: float = 0.25,
) -> bool:
    """Was this event only partly observed?

    True when a boundary sits on the edge of the observed region and no second
    window confirmed it — meaning the true extent is unknown rather than measured.
    An event touching the video's own start or end is genuinely truncated: the
    footage does not exist either side.
    """
    if start <= tol or end >= video_end_s - tol:
        return True
    if len(windows) > 1:
        # Confirmed by independent windows, so the boundary was observed, not
        # merely where looking stopped.
        return False
    return any(abs(start - e) < tol or abs(end - e) < tol for e in edges)


def merge_all(
    per_query: dict[str, list[tuple[WindowEvent, int]]],
    cfg: MergeConfig,
    observed_edges: set[float],
    video_end_s: float,
) -> list[Event]:
    """Merge every query's candidates. Queries never merge into each other.

    Keeping them separate matters: two different descriptions matching at the same
    moment are two findings, and collapsing them would silently drop one.
    """
    out: list[Event] = []
    for query, pairs in per_query.items():
        cands = [
            Candidate(e.start_s, e.end_s, e.confidence, e.evidence, widx)
            for e, widx in pairs
        ]
        for ev in merge_query(cands, cfg, observed_edges, video_end_s):
            ev.description = query
            out.append(ev)
    out.sort(key=lambda e: (e.start_s, e.end_s))
    return out
