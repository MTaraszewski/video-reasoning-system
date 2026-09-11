"""Split a long video into overlapping windows and assign frames to each.

This is the answer to the part of the brief that says videos may be longer than
the model can look at in one go.

Two properties matter more than the numbers:

**Overlap recovers boundary-straddling events.** With stride < window, an event
shorter than the overlap appears *whole* in at least one window, so the merge has
something complete to work with rather than two halves to guess at.

**Frames per call are capped.** Even at fixed fps, a long window would otherwise
grow the request without bound. The cap makes cost per call constant regardless of
window length or video duration — which is what keeps a two-hour video from
becoming a two-hour bill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .decode import Frame


@dataclass
class Window:
    index: int
    start_s: float
    end_s: float
    frames: list[Frame] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def clamp(self, t: float) -> float:
        """Force a time into this window's real span.

        Defends against the model reporting a timestamp for footage it never saw.
        A window must never be able to emit an event outside its own span.
        """
        return max(self.start_s, min(t, self.end_s))


def plan_windows(duration_s: float, window_s: float, stride_s: float) -> list[Window]:
    """Lay overlapping windows across the whole duration.

    Guarantees, because the merge depends on them:
      - every second of video is inside at least one window
      - windows are ordered and their indices are stable
      - the final window ends exactly at the video's end, never past it
    """
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    if duration_s <= 0:
        return []

    windows: list[Window] = []
    start = 0.0
    idx = 0
    while True:
        end = min(start + window_s, duration_s)
        windows.append(Window(index=idx, start_s=round(start, 3), end_s=round(end, 3)))
        idx += 1
        if end >= duration_s:
            break
        start += stride_s
    return windows


def assign_frames(
    windows: list[Window], frames: list[Frame], max_frames: int
) -> None:
    """Attach frames to windows, subsampling uniformly if a window exceeds the cap.

    Uniform rather than truncating: keeping the first N frames would silently make
    every window cover only its opening seconds, so events late in a window would
    vanish while everything still appeared to work.
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    for w in windows:
        inside = [f for f in frames if w.start_s - 1e-6 <= f.t <= w.end_s + 1e-6]
        if len(inside) > max_frames:
            step = len(inside) / float(max_frames)
            inside = [inside[min(int(i * step), len(inside) - 1)] for i in range(max_frames)]
        w.frames = inside


def plan_cost(duration_s: float, window_s: float, stride_s: float,
              n_queries: int, fps: float) -> dict:
    """What this request will cost, computed before any model call is made.

    Every term is known from the inputs alone, so an oversized job can be refused
    rather than discovered halfway through at GPU rates.
    """
    n_windows = len(plan_windows(duration_s, window_s, stride_s))
    return {
        "frames": int(duration_s * fps),
        "windows": n_windows,
        "queries": n_queries,
        "model_calls": n_windows * n_queries,
    }
