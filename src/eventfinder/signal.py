"""The cheap change detector that decides where the model looks at all.

The model is the only expensive thing in the pipeline. Deciding where to spend
it with pixels rather than with tokens is the difference between a run that
scales with clip length and one that does not.

The previous engine's attempt at this scored **tIoU 0.000 against 0.406** for
uniform polling on the same clip -- not because the change signal was wrong,
but because it picked the top-k most active moments and a minimum gap then
forbade sampling t=4.0, where the event was. Two design choices here exist
because of that failure:

  hysteresis   open a bracket at `hi_pct`, close it at `lo_pct`. A bracket is a
               contiguous region, not a point, so a real change cannot be
               reduced to one instant that a spacing rule then rejects.
  sentinels    a look on a fixed cadence regardless of change. This bounds the
               blind spot instead of hoping it is not where the event is.

Thresholds are percentiles of the clip's own change distribution, so a busy
street and a still corridor get comparable treatment without needing a scale.
Robust z-scoring was tried first and failed: see `SignalConfig.hi_pct`.

A percentile gate is a *ranking*, not a detector -- it fires on some fraction of
any clip including an empty one. `min_pixel_delta` is the floor that bounds what
it invents; sentinels bound what it misses. Neither is optional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SignalConfig
from .decode import sample_frames
from .models import Bracket

# Per-pixel intensity change counted as motion rather than sensor noise.
NOISE = 25


@dataclass
class ChangePoint:
    t: float       # time of the LATER sample; change is attributed to its arrival
    moved: float   # fraction of pixels whose intensity changed
    rank: float = 0.0  # percentile rank of `moved` within this clip, 0..100


def change_score(video: str, *, fps: float = 2.0, thumb_px: int = 128,
                 start_s: float = 0.0, end_s: float | None = None,
                 seek: bool = True) -> list[ChangePoint]:
    """Fraction of pixels moving between consecutive samples.

    Deliberately tiny: this is a where-to-look signal and 128px costs almost
    nothing beside a model call. No overlay -- nothing reads these pixels but
    numpy.
    """
    frames = sample_frames(video, fps=fps, max_side=thumb_px, overlay=False,
                           start_s=start_s, end_s=end_s, seek=seek)
    if len(frames) < 2:
        return []
    arrs = [np.asarray(f.image.convert("L"), dtype=np.int16) for f in frames]
    pts = [
        ChangePoint(t=round(b.t, 3), moved=round(float((np.abs(y - x) > NOISE).mean()), 6))
        for (x, y), b in zip(zip(arrs, arrs[1:]), frames[1:])
    ]
    return _rank(pts)


def _rank(pts: list[ChangePoint]) -> list[ChangePoint]:
    """Percentile rank of each sample within the clip.

    Scale-free by construction, which is what the z-score failed to be here: on
    six of eight labelled clips the median change is exactly zero, so there is
    no robust scale to divide by.
    """
    v = np.array([p.moved for p in pts])
    for p in pts:
        p.rank = round(float((v < p.moved).mean() * 100), 2)
    return pts


def _hysteresis(pts: list[ChangePoint], cfg: SignalConfig) -> list[tuple[float, float, float]]:
    """Contiguous (start, end, peak_rank) regions above the threshold pair."""
    spans: list[tuple[float, float, float]] = []
    open_at: float | None = None
    peak = 0.0
    for p in pts:
        hot = p.rank >= cfg.hi_pct and p.moved >= cfg.min_pixel_delta
        warm = p.rank >= cfg.lo_pct and p.moved >= cfg.min_pixel_delta
        if open_at is None:
            if hot:
                open_at, peak, last = p.t, p.rank, p.t
        else:
            if warm:
                peak = max(peak, p.rank)
                last = p.t
            else:
                spans.append((open_at, last, peak))
                open_at = None
    if open_at is not None:
        spans.append((open_at, pts[-1].t, peak))
    return spans


def _merge(spans: list[tuple[float, float, float]], gap: float
           ) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for s, e, z in sorted(spans):
        if out and s - out[-1][1] <= gap:
            ps, pe, pz = out[-1]
            out[-1] = (ps, max(pe, e), max(pz, z))
        else:
            out.append((s, e, z))
    return out


def brackets(video: str, duration_s: float, cfg: SignalConfig) -> list[Bracket]:
    """Candidate ranges for the observer, in time order.

    With `cfg.enabled` false the whole clip is one manual bracket: correct,
    linear in clip length, and the baseline every saving is measured against.
    """
    if not cfg.enabled:
        return [Bracket(id="b1", start_s=0.0, end_s=duration_s, origin="manual")]

    pts = change_score(video, fps=cfg.fps, thumb_px=cfg.thumb_px)
    spans = _merge(_hysteresis(pts, cfg), cfg.merge_gap_s) if pts else []

    # Pad so the observer sees both sides of the change, not only its middle.
    padded = [(max(0.0, s - cfg.pad_s), min(duration_s, e + cfg.pad_s), z)
              for s, e, z in spans]
    padded = _merge(padded, cfg.merge_gap_s)

    out: list[Bracket] = []
    for s, e, z in padded:
        # Split rather than truncate. A long bracket is a long thing happening,
        # and truncating it would drop observed time on the floor while still
        # reporting the span as covered.
        n = max(1, int(np.ceil((e - s) / cfg.max_bracket_s)))
        w = (e - s) / n
        for i in range(n):
            out.append(Bracket(id="", start_s=round(s + i * w, 3),
                               end_s=round(s + (i + 1) * w, 3),
                               origin="rising", peak_score=round(z, 3)))

    out.extend(_sentinels(out, duration_s, cfg))
    out.sort(key=lambda b: b.start_s)
    for i, b in enumerate(out, 1):
        b.id = f"b{i}"
    return out


def _sentinels(covered: list[Bracket], duration_s: float, cfg: SignalConfig) -> list[Bracket]:
    """Fixed-cadence looks into time no bracket claims.

    Not "every N seconds" -- every N seconds *of unobserved time*. A clip whose
    change signal already covers a busy stretch pays nothing extra for it, and a
    clip where the signal found nothing still gets looked at.
    """
    gaps, prev = [], 0.0
    for b in sorted(covered, key=lambda b: b.start_s):
        if b.start_s > prev:
            gaps.append((prev, b.start_s))
        prev = max(prev, b.end_s)
    if prev < duration_s:
        gaps.append((prev, duration_s))

    out: list[Bracket] = []
    width = min(cfg.max_bracket_s, cfg.sentinel_width_s, cfg.sentinel_every_s)
    for lo, hi in gaps:
        n = int((hi - lo) // cfg.sentinel_every_s)
        for i in range(n + 1):
            # Centre of each cadence slot, clipped to the gap.
            c = lo + (i + 0.5) * cfg.sentinel_every_s
            if c >= hi:
                if i == 0 and hi - lo > 0:
                    c = (lo + hi) / 2   # a gap shorter than one slot still gets one look
                else:
                    break
            s = max(lo, c - width / 2)
            e = min(hi, c + width / 2)
            if e - s > 0:
                out.append(Bracket(id="", start_s=round(s, 3), end_s=round(e, 3),
                                   origin="sentinel"))
    return out


def coverage_of(bs: list[Bracket], duration_s: float) -> float:
    """Fraction of the clip any bracket claims. The cost proxy, before any call."""
    if duration_s <= 0:
        return 0.0
    seen, hi = 0.0, 0.0
    for b in sorted(bs, key=lambda b: b.start_s):
        if b.end_s <= hi:
            continue
        seen += b.end_s - max(b.start_s, hi)
        hi = b.end_s
    return round(min(1.0, seen / duration_s), 4)
