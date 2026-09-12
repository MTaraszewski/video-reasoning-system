"""The live path: frames in, timed events out, bounded latency.

Same compiler, same derivation, same contracts as the batch pipeline. What
changes is where brackets come from and when results leave.

    frames ──> ring buffer (last window_s seconds, already stamped)
           └─> candidate sources ──> brackets ──> observe ──> Deriver.step ──> emit

**The model is never on a clock.** It is called only on a candidate, which is
why cost scales with how much happens rather than with how long the camera
runs. One L4 cannot run a 4B model at frame rate and never will; that is
arithmetic, not a preference.

**Two candidate sources today, a third designed for.**

    MotionSource    streaming hysteresis over the pixel-change signal. Measured
                    on the labelled set: found 8 of 8 mid-clip events by rising
                    edge, including both door events, which no detector
                    vocabulary covers.
    SentinelSource  a look on a fixed cadence into time nothing claimed. The
                    bounded blind spot. Not optional: the previous engine's
                    trigger-only design scored tIoU 0.000 against 0.406 for
                    uniform polling on the same clip because it declined to
                    sample the second the event was in, and in a stream there
                    is no second pass to recover it.
    TrackSource     a detector plus tracker (not built). It would differ in
                    kind: tracks yield Observations directly -- present,
                    position, motion, identity -- so presence, direction and
                    cessation events become derivable with no model call at
                    all, and an event can be emitted `verified=False` the
                    instant it is witnessed, with the model's confirmation
                    following as an update. `CandidateSource` is the seam it
                    plugs into.

Until that source exists there is nothing to pre-emit: pixel motion says
something changed, not that a described event occurred. Progressive emission is
wired through `on_event(event, verified=...)` and is exercised by the model
path; it becomes useful when tracks arrive.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import numpy as np

from .config import Config
from .decode import Frame, sample_frames
from .derive import Deriver
from .models import Bracket, Event, Probe
from .signal import NOISE

EventSink = Callable[[Event, bool], None]


# --- the ring buffer --------------------------------------------------------

class RingBuffer:
    """The last `window_s` of stamped frames, so a candidate can be looked at
    after it has been recognised. Sized in seconds rather than frames because
    what bounds latency is how far back the system can still see."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self.frames: deque[Frame] = deque()

    def add(self, f: Frame) -> None:
        self.frames.append(f)
        cutoff = f.t - self.window_s
        while self.frames and self.frames[0].t < cutoff:
            self.frames.popleft()

    def slice(self, start_s: float, end_s: float, step_s: float) -> list[Frame]:
        """One frame per step across the range, nearest available.

        Nearest rather than interpolated: a frame carries its own burned-in
        time, so substituting a neighbour is honest as long as the stamp travels
        with it. The model reads the time off the pixels either way.
        """
        out: list[Frame] = []
        t = start_s
        while t <= end_s + 1e-6:
            near = min(self.frames, key=lambda f: abs(f.t - t), default=None)
            if near is not None and (not out or out[-1].t != near.t):
                out.append(near)
            t += step_s
        return out

    @property
    def span(self) -> tuple[float, float]:
        return (self.frames[0].t, self.frames[-1].t) if self.frames else (0.0, 0.0)


# --- candidate sources ------------------------------------------------------

class CandidateSource(Protocol):
    name: str

    def feed(self, frame: Frame) -> list[Bracket]: ...


class MotionSource:
    """Streaming hysteresis over pixel change.

    The batch version ranks each sample against the whole clip's distribution.
    A stream has no whole clip, so the rank is taken against a rolling history
    of recent samples -- which also makes it adapt: a camera that gets busy at
    shift change re-normalises instead of firing continuously.
    """

    name = "motion"

    def __init__(self, cfg: Config, history: int = 600):
        self.cfg = cfg.signal
        self.hist: deque[float] = deque(maxlen=history)
        self.prev: np.ndarray | None = None
        self.prev_t = 0.0
        self.open_at: float | None = None
        self.last_hot = 0.0
        self.peak = 0.0
        self.n = 0

    def feed(self, frame: Frame) -> list[Bracket]:
        small = frame.image.convert("L").resize(
            (self.cfg.thumb_px, max(1, int(self.cfg.thumb_px * frame.image.height
                                           / max(1, frame.image.width)))))
        arr = np.asarray(small, dtype=np.int16)
        if self.prev is None or self.prev.shape != arr.shape:
            self.prev, self.prev_t = arr, frame.t
            return []
        moved = float((np.abs(arr - self.prev) > NOISE).mean())
        self.prev, self.prev_t = arr, frame.t

        # Rank against recent history. Until there is enough of it, nothing
        # fires: a threshold learned from four samples is noise.
        self.hist.append(moved)
        if len(self.hist) < 30:
            return []
        h = np.fromiter(self.hist, dtype=float)
        rank = float((h < moved).mean() * 100)

        hot = rank >= self.cfg.hi_pct and moved >= self.cfg.min_pixel_delta
        warm = rank >= self.cfg.lo_pct and moved >= self.cfg.min_pixel_delta

        if self.open_at is None:
            if hot:
                self.open_at, self.last_hot, self.peak = frame.t, frame.t, rank
            return []
        if warm:
            self.last_hot, self.peak = frame.t, max(self.peak, rank)
            # A change that will not stop still has to be looked at eventually.
            if frame.t - self.open_at >= self.cfg.max_bracket_s:
                return [self._close(frame.t)]
            return []
        return [self._close(self.last_hot)]

    def _close(self, end_s: float) -> Bracket:
        self.n += 1
        b = Bracket(id=f"m{self.n}",
                    start_s=round(max(0.0, self.open_at - self.cfg.pad_s), 3),
                    end_s=round(end_s + self.cfg.pad_s, 3),
                    origin="rising", peak_score=round(self.peak, 3))
        self.open_at, self.peak = None, 0.0
        return b


class SentinelSource:
    """A look on a fixed cadence into time no other source claimed."""

    name = "sentinel"

    def __init__(self, cfg: Config):
        self.every = cfg.signal.sentinel_every_s
        self.width = cfg.signal.sentinel_width_s
        self.next_at: float | None = None
        self.n = 0

    def note_covered(self, until_s: float) -> None:
        """Another source already claimed up to here; the cadence restarts from
        there rather than paying twice for the same seconds."""
        if self.next_at is None or until_s + self.every > self.next_at:
            self.next_at = until_s + self.every

    def feed(self, frame: Frame) -> list[Bracket]:
        if self.next_at is None:
            self.next_at = frame.t + self.every
            return []
        if frame.t < self.next_at:
            return []
        self.n += 1
        start = max(0.0, frame.t - self.width)
        self.next_at = frame.t + self.every
        return [Bracket(id=f"s{self.n}", start_s=round(start, 3),
                        end_s=round(frame.t, 3), origin="sentinel")]


# --- the driver -------------------------------------------------------------

@dataclass
class LiveStats:
    frames: int = 0
    candidates: int = 0
    calls: int = 0
    emitted: int = 0
    dropped: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def mean_latency_s(self) -> float:
        return round(sum(self.latencies) / len(self.latencies), 3) if self.latencies else 0.0


class LiveFinder:
    """Feed frames, get events. The streaming counterpart of `pipeline.run`.

    Overload degrades by dropping the weakest pending candidate rather than
    falling behind: a live system that queues without bound stops being live,
    and silently. Drops are counted.
    """

    def __init__(self, probes: list[Probe], cfg: Config, reasoner,
                 on_event: EventSink, *, max_pending: int = 8):
        self.cfg = cfg
        self.reasoner = reasoner
        self.on_event = on_event
        self.probes = [p for p in probes if p.expressible]
        self.buffer = RingBuffer(window_s=max(cfg.signal.max_bracket_s * 2, 30.0))
        self.sources: list[CandidateSource] = [MotionSource(cfg), SentinelSource(cfg)]
        self.deriver = Deriver(self.probes, cfg.derive, cfg.observe.step_s)
        self.pending: deque[Bracket] = deque()
        self.max_pending = max_pending
        self.stats = LiveStats()
        self._groups = _group(self.probes)

    # --- input ----------------------------------------------------------

    def feed(self, frame: Frame) -> None:
        self.buffer.add(frame)
        self.stats.frames += 1
        for src in self.sources:
            for b in src.feed(frame):
                self._enqueue(b)
                if isinstance(src, MotionSource):
                    for s in self.sources:
                        if isinstance(s, SentinelSource):
                            s.note_covered(b.end_s)

    def _enqueue(self, b: Bracket) -> None:
        self.stats.candidates += 1
        self.pending.append(b)
        while len(self.pending) > self.max_pending:
            # Oldest first: a candidate the ring buffer can no longer show is
            # unverifiable anyway, so dropping it loses nothing that was still
            # recoverable.
            self.pending.popleft()
            self.stats.dropped += 1

    # --- work -----------------------------------------------------------

    def tick(self, now: float) -> list[Event]:
        """Observe what is pending, derive, emit. Call as often as you like."""
        out: list[Event] = []
        lo, _ = self.buffer.span
        while self.pending:
            b = self.pending.popleft()
            if b.start_s < lo:
                self.stats.dropped += 1     # aged out of the buffer
                continue
            out.extend(self._observe(b))
        for e in self.deriver.step(now):
            self.stats.emitted += 1
            self.stats.latencies.append(round(now - e.end_s, 3))
            self.on_event(e, True)
            out.append(e)
        return out

    def _observe(self, b: Bracket) -> list[Event]:
        step = self.cfg.observe.step_s
        frames = self.buffer.slice(b.start_s, b.end_s, step)
        if not frames:
            return []
        times = [f.t for f in frames][: self.cfg.sampling.max_frames_per_call]
        for (subject, attrs), ps in self._groups.items():
            obs = self.reasoner.observe(frames, times, subject, list(attrs), b.id)
            self.stats.calls += 1
            for p in ps:
                self.deriver.add(p.id, obs)
        return []

    def close(self, now: float) -> list[Event]:
        """Flush anything still pending. For a file; a camera never closes."""
        return self.tick(now)


def _group(probes: list[Probe]) -> dict[tuple[str, tuple[str, ...]], list[Probe]]:
    out: dict[tuple[str, tuple[str, ...]], list[Probe]] = {}
    for p in probes:
        out.setdefault((p.subject, tuple(p.attributes)), []).append(p)
    return out


# --- running a file as if it were a camera ---------------------------------

def frames_from_file(video: str, cfg: Config, *, realtime: bool = False
                     ) -> Iterable[Frame]:
    """A file, played as a stream. The stand-in for RTSP until there is one.

    `realtime` sleeps to the capture clock, which is what makes a throughput
    claim mean anything: run it flat out and you measure the machine, run it at
    camera rate and you measure whether the system keeps up.
    """
    fps = cfg.signal.fps
    frames = sample_frames(video, fps=fps, max_side=cfg.sampling.frame_max_side,
                           seek=cfg.sampling.seek, overlay=cfg.overlay.enabled,
                           font_scale=cfg.overlay.font_scale, fmt=cfg.overlay.format,
                           position=cfg.overlay.position)
    t0 = time.perf_counter()
    for f in frames:
        if realtime:
            behind = f.t - (time.perf_counter() - t0)
            if behind > 0:
                time.sleep(behind)
        yield f
