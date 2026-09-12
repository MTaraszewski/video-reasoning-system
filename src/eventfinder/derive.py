"""Turn observations into timed, scored events. In code, never in the model.

The model reports what it saw at times printed on frames. Everything about
*when an event happened* is decided here, from those reports. That division is
the whole architecture: perception is the model's job, temporal reasoning is
not.

**How a span is chosen.** For a transition kind, the event runs from the last
observation in the old state to the first observation in the new one. That is
not an estimate -- it is the bracket the evidence actually supports, and its
width is the polling interval. Nothing is interpolated to a midpoint; doing so
in the previous engine turned a 3-second door opening into a 42-second event at
full confidence, because two polls 82 seconds apart were averaged.

When that bracket is wider than `max_interp_gap_s` the span is anchored to the
observed evidence and marked `partial`: we know the subject was in the new
state at that time, and we do not know when it changed. An honest partial beats
a precise fiction.

**Confidence** is `agreement x sharpness x coverage`, published as three
separate numbers. One opaque score cannot be argued with; these can. Low with
high agreement and low coverage is a sampling problem; the same score with full
coverage and low agreement is a perception problem, and they call for different
fixes.

**Incremental by construction.** `Deriver.step(now)` yields events that have
just become witnessed -- a transition is believed once an observation after it
exists. The same code path serves a batch run and a live stream; there is no
second implementation to drift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import DeriveConfig
from .models import Event, Observation, Probe, Signals

# Facing -> the unit vector the subject's front points along, in image
# coordinates (x right, y down). `toward` the camera reads as moving down the
# frame, which holds for a camera looking along the ground and is the reason
# direction probes are the least trustworthy kind.
_FACING_VEC = {"left": (-1.0, 0.0), "right": (1.0, 0.0),
               "toward": (0.0, 1.0), "away": (0.0, -1.0)}


# --- state canonicalisation -------------------------------------------------

def canonical_state(raw: str | None, vocab: tuple[str, ...] | list[str]) -> str | None:
    """Match the model's free text against the probe's vocabulary, in code.

    Longest candidate first, then LAST mention wins: "the door was closed and is
    now open" is about a door that is open. Measured on the previous engine,
    this string match parsed 85% of door polls, while asking the model to pick
    from a supplied list scored 0.00 once option order was averaged out -- the
    choice was driven by the order the options appeared in, not by the video.
    """
    if not raw:
        return None
    low = raw.lower()
    best: tuple[float, int, str] | None = None
    for word in vocab:
        for m in re.finditer(rf"\b{re.escape(word.lower())}\b", low):
            # Rank by where the mention ENDS, then by length. Ending position
            # gives last-mention-wins; length breaks the tie when one phrase
            # contains another, so "partially open" beats the "open" inside it
            # rather than losing to it on start position.
            key = (m.end(), len(word), word)
            if best is None or key > best:
                best = key
    return best[2] if best else None


def _dimension(o: Observation, probe: Probe) -> str | None:
    """The single token this probe compares across time."""
    if not o.ok:
        return None
    kind = probe.kind
    if kind == "presence":
        return "present" if o.present else "absent"
    if kind == "cessation":
        return o.motion if o.motion != "unknown" else None
    if kind == "state":
        if not o.present:
            return None
        return canonical_state(o.raw_state or o.state, probe.states or ())
    if kind == "relation":
        if not o.present:
            return None
        val = (o.relations or {}).get(probe.relation_key or "", "")
        want = probe.relation_value
        if want == "none":
            return "none" if not val else "held"
        if want == "any":
            return "held" if val else "none"
        return val or "none"        # "change": the value itself is the token
    return None


def _transition_pair(probe: Probe) -> tuple[str, str] | None:
    """(before, after) tokens, or None for run-shaped kinds."""
    if probe.kind in ("state", "presence") and probe.states:
        return probe.states[0], probe.states[1]
    if probe.kind == "cessation":
        return "moving", "stationary"
    if probe.kind == "relation":
        if probe.relation_value == "none":
            return "held", "none"
        if probe.relation_value == "any":
            return "none", "held"
        return None                  # "change": any value change counts
    return None


# --- signals ----------------------------------------------------------------

def _agreement(obs: list[Observation], probe: Probe, want_before: str,
               want_after: str, lo: float, split: float, hi: float) -> float:
    """Fraction of parsed observations across the event that back the reading.

    Counted over a NEIGHBOURHOOD of the transition, not over the two runs. The
    runs are homogeneous by construction, so scoring them against their own
    token would always return 1.0 and measure nothing -- the window has to reach
    past them to see dissent. A reading of "open" in the middle of the closed
    stretch is evidence against this transition, and it should show.

    `split` is the last observation of the old state: everything up to it should
    say `want_before`, everything after should say `want_after`.

    Unparsed observations are excluded rather than counted against: their cost
    is already carried by `coverage`, and charging them twice would make a
    sparse run look like a contradicted one.
    """
    n = agree = 0
    for o in obs:
        if not (lo - 1e-6 <= o.t <= hi + 1e-6):
            continue
        tok = _dimension(o, probe)
        if tok is None:
            continue
        n += 1
        agree += tok == (want_before if o.t <= split + 1e-6 else want_after)
    return round(agree / n, 4) if n else 0.0


def _sharpness(gap_s: float, step_s: float) -> float:
    """1.0 when the transition sits between adjacent polls, falling as the
    bracket widens. This is the factor that says how much of the reported span
    is evidence and how much is the interval between looks."""
    if gap_s <= 0:
        return 1.0
    return round(min(1.0, step_s / gap_s), 4)


def _coverage(obs: list[Observation], start: float, end: float, step_s: float) -> float:
    """Fraction of the event's span that a usable observation actually covers.

    Each observation speaks for the interval [t, t+step_s) and no longer. Union
    those, intersect with the span, divide. Counting observations instead would
    score a span whose middle was never seen as fully covered, which is exactly
    the case coverage exists to catch.
    """
    span = end - start
    if span <= 0:
        return 1.0
    seen, hi = 0.0, start
    for o in sorted((o for o in obs if o.ok), key=lambda o: o.t):
        a, b = max(o.t, start), min(o.t + step_s, end)
        if b <= hi:
            continue
        seen += b - max(a, hi)
        hi = b
    return round(min(1.0, max(0.0, seen) / span), 4)


def _mean_certainty(obs: list[Observation]) -> float:
    good = [o.certainty for o in obs if o.ok]
    return round(sum(good) / len(good), 4) if good else 0.0


# --- the derivation ---------------------------------------------------------

@dataclass
class _Run:
    token: str
    obs: list[Observation] = field(default_factory=list)

    @property
    def first(self) -> float:
        return self.obs[0].t

    @property
    def last(self) -> float:
        return self.obs[-1].t


def _runs(obs: list[Observation], probe: Probe) -> list[_Run]:
    """Consecutive observations carrying the same token.

    Observations that failed or parsed to nothing do NOT break a run -- they
    are absence of evidence, not a change of state. Treating them as a break
    would let a single unparsed reply split one event into two.
    """
    out: list[_Run] = []
    for o in obs:
        tok = _dimension(o, probe)
        if tok is None:
            continue
        if out and out[-1].token == tok:
            out[-1].obs.append(o)
        else:
            out.append(_Run(token=tok, obs=[o]))
    return out


def derive(obs: list[Observation], probe: Probe, cfg: DeriveConfig,
           step_s: float, *, now: float | None = None) -> list[Event]:
    """Every event this probe's observations support, in time order.

    `now` bounds what counts as witnessed: a transition is believed only once
    an observation after it exists and has held for `transition_hold_s`. In a
    live stream that is what stops an event being emitted before its end is
    known; in a batch run it is simply the last observation time.
    """
    obs = sorted(obs, key=lambda o: o.t)
    if not obs:
        return []
    now = obs[-1].t if now is None else now
    if probe.kind == "direction":
        return _derive_direction(obs, probe, cfg, step_s, now)

    runs = _runs(obs, probe)
    pair = _transition_pair(probe)
    events: list[Event] = []

    for a, b in zip(runs, runs[1:]):
        if pair is None:                      # relation "change": any change counts
            if a.token == b.token:
                continue
            want_before, want_after = a.token, b.token
        else:
            want_before, want_after = pair
            if (a.token, b.token) != pair:
                continue
        # Both sides must be believed, not just seen once.
        if len(a.obs) < cfg.min_run or len(b.obs) < cfg.min_run:
            continue
        # The new state has to hold before the transition is witnessed.
        if b.last - b.first < cfg.transition_hold_s or now < b.first + cfg.transition_hold_s:
            continue

        gap = b.first - a.last
        partial = "none"
        start, end = a.last, b.first
        if gap > cfg.max_interp_gap_s:
            # Too wide to claim the event filled it. Anchor to the evidence and
            # say the start is unknown rather than inventing a duration.
            start, end = round(b.first - step_s, 3), b.first
            partial = "start_unknown"

        window = a.obs[-cfg.min_run:] + b.obs[:cfg.min_run]
        sig = Signals(
            agreement=_agreement(obs, probe, want_before, want_after,
                                 a.first - cfg.min_run * step_s, a.last,
                                 b.last + cfg.min_run * step_s),
            sharpness=_sharpness(gap, step_s),
            coverage=_coverage(obs, start, end, step_s),
            mean_certainty=_mean_certainty(window),
        )
        events.append(_event(probe, start, end, partial, sig, window, cfg))

    return [e for e in events if e.confidence >= cfg.min_confidence]


def _derive_direction(obs: list[Observation], probe: Probe, cfg: DeriveConfig,
                      step_s: float, now: float) -> list[Event]:
    """Displacement against facing, over consecutive observations.

    The riskiest kind by a distance: it needs position AND facing to be right at
    the same time, from a model that was never measured on either. Kept honest
    by refusing to score any pair where facing is unknown, rather than assuming
    a heading.
    """
    want_back = probe.direction == "backward"
    qualifying: list[Observation] = []
    events: list[Event] = []
    usable = [o for o in obs if o.ok and o.present and o.position and o.facing in _FACING_VEC]

    for p, q in zip(usable, usable[1:]):
        fx, fy = _FACING_VEC[p.facing]
        dx, dy = q.position[0] - p.position[0], q.position[1] - p.position[1]
        dot = dx * fx + dy * fy
        moving_back = dot < 0
        if moving_back == want_back and abs(dot) > 1e-3:
            if not qualifying:
                qualifying = [p]
            qualifying.append(q)
        elif qualifying:
            events.append(_run_event(qualifying, probe, cfg, step_s, obs))
            qualifying = []
    if qualifying and now >= qualifying[-1].t:
        events.append(_run_event(qualifying, probe, cfg, step_s, obs))

    return [e for e in events
            if e and len(e.evidence) >= cfg.min_run and e.confidence >= cfg.min_confidence]


def _run_event(run: list[Observation], probe: Probe, cfg: DeriveConfig,
               step_s: float, all_obs: list[Observation]) -> Event:
    start, end = run[0].t, run[-1].t
    sig = Signals(
        agreement=round(len(run) / max(1, len(run)), 4),
        sharpness=_sharpness(step_s, step_s),
        coverage=_coverage(all_obs, start, end, step_s),
        mean_certainty=_mean_certainty(run),
    )
    return _event(probe, start, end, "none", sig, run, cfg)


def _event(probe: Probe, start: float, end: float, partial: str, sig: Signals,
           evidence: list[Observation], cfg: DeriveConfig) -> Event:
    conf = sig.agreement * sig.sharpness * sig.coverage
    if cfg.certainty_weight:
        # Off by default. Whether a self-reported number carries information is
        # a measurement, not an assumption, and this is the knob that makes it
        # one.
        w = cfg.certainty_weight
        conf = conf * (1 - w) + conf * sig.mean_certainty * w
    return Event(
        id="", probe_id=probe.id, description=probe.description,
        start_s=round(start, 3), end_s=round(end, 3),
        confidence=round(conf, 4), partial=partial, signals=sig,
        evidence=[o.t for o in evidence],
        bracket_ids=sorted({o.bracket_id for o in evidence if o.bracket_id}),
    )


# --- incremental ------------------------------------------------------------

class Deriver:
    """Rolling buffer per probe, for the live path.

    Holds `window_s` of observations and yields each event once. The same
    `derive()` runs over the buffer as over a whole clip, so a streaming result
    and a batch result cannot diverge through having two implementations.
    """

    def __init__(self, probes: list[Probe], cfg: DeriveConfig, step_s: float,
                 window_s: float = 30.0):
        self.probes = [p for p in probes if p.expressible]
        self.cfg, self.step_s, self.window_s = cfg, step_s, window_s
        self.buf: dict[str, list[Observation]] = {p.id: [] for p in self.probes}
        self.emitted: set[tuple[str, float]] = set()
        self.n_emitted = 0

    def add(self, probe_id: str, obs: list[Observation]) -> None:
        if probe_id in self.buf:
            self.buf[probe_id].extend(obs)

    def step(self, now: float) -> list[Event]:
        """Events that have just become witnessed. Never the same one twice."""
        out: list[Event] = []
        for p in self.probes:
            buf = self.buf[p.id]
            if not buf:
                continue
            # Drop what has aged out, but only after the events it supported
            # have been emitted -- hence trimming AFTER deriving.
            for e in derive(buf, p, self.cfg, self.step_s, now=now):
                key = (p.id, e.start_s)
                if key in self.emitted:
                    continue
                self.emitted.add(key)
                self.n_emitted += 1
                e.id = f"e{self.n_emitted}"
                out.append(e)
            cutoff = now - self.window_s
            self.buf[p.id] = [o for o in buf if o.t >= cutoff]
        return sorted(out, key=lambda e: e.start_s)


def rank(events: list[Event]) -> list[Event]:
    """Confidence order, with ids and ranks assigned. The output contract wants
    a ranked list, and ranking by anything the model said about itself would
    put an unmeasured number in charge of the answer."""
    out = sorted(events, key=lambda e: (-e.confidence, e.start_s))
    for i, e in enumerate(out, 1):
        e.rank = i
        if not e.id:
            e.id = f"e{i}"
    return out
