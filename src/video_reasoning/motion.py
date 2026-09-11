"""A cheap change signal, used to decide WHERE to spend a model call.

Polling a state on a fixed grid spends most of its calls confirming that nothing
changed. Measured on `admin.G326`: every poll agreed with its neighbours except at
t=4 and t=5, which is the transition itself. The stable stretches were 90% of the
cost and carried none of the information.

This computes the same primitive `scripts/screen_clips.py` uses for clip
selection — the fraction of pixels whose intensity moves between consecutive
samples — and exposes it as a trigger.

**Its blind spots are known and measured**, because the clip screening documented
them: a slow 27s traverse scored 1.07%, a 0.15s event scored 0.07% at 2fps, and a
negative clip scored 3.42%. Small distant actors move a fraction of a percent. So
this narrows where to look; it does not decide what happened, and a trigger that
misses an event costs a detection the uniform grid would have found. That trade is
the thing to measure, not to assume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decode import sample_frames

# Per-pixel intensity change counted as motion rather than sensor noise. Same
# value the clip screening uses, so the two agree about what "moving" means.
NOISE = 25


@dataclass
class ChangePoint:
    """How much moved between one sample and the next."""

    t: float          # time of the LATER sample — change is attributed to its arrival
    moved_pct: float  # percentage of pixels whose intensity changed


def change_signal(video: str, *, fps: float = 2.0, start_s: float = 0.0,
                  end_s: float | None = None, max_side: int = 320
                  ) -> list[ChangePoint]:
    """Fraction of pixels moving between consecutive samples, over a span.

    Deliberately low resolution: this is a where-to-look signal, and 320px is
    enough for that while costing almost nothing next to a model call.
    """
    _, frames = sample_frames(video, fps=fps, max_side=max_side, overlay=False,
                              start_s=start_s, end_s=end_s)
    if len(frames) < 2:
        return []
    arrs = [np.asarray(f.image.convert("L"), dtype=np.int16) for f in frames]
    return [
        ChangePoint(t=round(b.t, 3),
                    moved_pct=round(float((np.abs(y - x) > NOISE).mean() * 100), 4))
        for (x, y), b in zip(zip(arrs, arrs[1:]), frames[1:])
    ]


def peaks(signal: list[ChangePoint], *, top_k: int | None = None,
          min_pct: float = 0.0, min_gap_s: float = 2.0) -> list[float]:
    """Times worth spending a model call on, strongest first, then in time order.

    `min_gap_s` stops a single event producing a cluster of adjacent calls that
    all describe the same moment. `top_k` bounds the cost directly, which matters
    more than a threshold: a threshold chosen on one clip does not transfer, but
    "the 12 most active moments" is a budget and behaves the same everywhere.
    """
    picked: list[ChangePoint] = []
    for c in sorted(signal, key=lambda c: -c.moved_pct):
        if c.moved_pct < min_pct:
            break
        if any(abs(c.t - p.t) < min_gap_s for p in picked):
            continue
        picked.append(c)
        if top_k and len(picked) >= top_k:
            break
    return sorted(p.t for p in picked)
