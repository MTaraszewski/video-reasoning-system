"""The model seam.

Everything above this line is arithmetic over `(start, end, score)` tuples and
does not care which model produced them. Everything below knows about prompts,
HTTP and response formats.

The seam exists for one identified reason, not for generality: the localisation
mechanism of the primary model is undocumented. If Cosmos3-Edge turns out not to
read burned-in timestamps, what changes is confined to a backend — prompt shape,
response parsing, possibly the whole localisation strategy — while windowing,
merging and scoring stay untouched.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from ..schema import WindowEvent, WindowResponse
from ..windows import Window


@dataclass
class ExtractRequest:
    """One (window, query) pair, ready to send."""

    window: Window
    query: str
    system_prompt: str
    user_prompt: str


@dataclass
class ExtractResult:
    """What came back, plus enough context to debug or replay it."""

    events: list[WindowEvent]
    raw: str = ""
    reasoning: str = ""
    latency_s: float = 0.0
    model: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)


class Backend(Protocol):
    """What every backend must provide."""

    name: str
    is_stub: bool

    def extract(self, req: ExtractRequest) -> ExtractResult: ...
    def describe(self) -> dict: ...


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_OBJECT = re.compile(r"\{.*\}", re.S)


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate a reasoning pass from the answer.

    Cosmos 3 is a reasoning model: vLLM's `--reasoning-parser qwen3` exists
    precisely because it emits a `<think>` block before answering. A parser that
    assumes bare JSON fails on every response.

    Returns (reasoning, answer). Either may be empty.
    """
    blocks = _THINK.findall(text or "")
    reasoning = "\n".join(b for b in blocks)
    answer = _THINK.sub("", text or "").strip()
    # An unterminated <think> means the response was truncated mid-reasoning:
    # there is no answer, and pretending otherwise invents data.
    if not blocks and "<think>" in (text or "").lower():
        return text or "", ""
    return reasoning.strip(), answer


def parse_events(text: str) -> tuple[list[WindowEvent], str | None]:
    """Pull events out of a model response, tolerating the ways it goes wrong.

    Handled, because all of them happen:
      - a `<think>` block before the answer
      - JSON wrapped in a ``` fence
      - prose around the JSON
      - a bare list instead of the documented object
      - truncation at max_tokens

    Returns (events, error). An empty list with no error means the model said
    nothing happened — a valid answer, not a failure.
    """
    _, answer = split_reasoning(text)
    if not answer.strip():
        return [], "empty response (possibly truncated during reasoning)"

    candidates: list[str] = []
    fenced = _FENCE.findall(answer)
    candidates.extend(fenced)
    candidates.append(answer)
    obj = _OBJECT.search(answer)
    if obj:
        candidates.append(obj.group(0))

    for raw in candidates:
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            data = {"events": data}
        if not isinstance(data, dict):
            continue
        try:
            return WindowResponse(**data).events, None
        except Exception:
            # Right shape, wrong field types — keep the rows that do validate
            # rather than discarding a whole window for one bad entry.
            good: list[WindowEvent] = []
            for item in data.get("events", []) or []:
                try:
                    good.append(WindowEvent(**item))
                except Exception:
                    continue
            if good:
                return good, None
    return [], "no parseable JSON in response"


def reconcile_times(
    events: list[WindowEvent], window: Window,
    *, tolerance_s: float = 1.0, frame_tolerance_s: float = 0.5,
) -> tuple[list[WindowEvent], dict]:
    """Reconcile reported times with what the window actually contained.

    Clamping alone is not enough, and is arguably worse than nothing on its own:
    a model reporting t=400s for a 12s window is unmistakably hallucinating, but
    *clamped* that becomes a confident event at the window edge — indistinguishable
    from a real detection, and no longer flagged as anything. Clamping turns a
    detectable error into an undetectable one.

    So two different responses to two different failures:

    **Small overshoot -> clamp.** Within `tolerance_s` of the window is consistent
    with rounding or a misread digit. The event is real, the boundary is fuzzy.

    **Wild value -> reject, and count it.** Beyond tolerance the model is not
    reporting something it saw. Dropping it is honest; clamping it is fabrication.

    **Reported near no frame we sent -> reject.** The strongest check available,
    and one only we can make: the exact timestamps burned into this window's frames
    are known. A time near none of them was never on screen to be read.

    Rejections are returned, not swallowed. A model that hallucinates often is a
    finding about the model, which is what the brief asks us to report.
    """
    stats = {"clamped": 0, "rejected_out_of_window": 0, "rejected_no_frame": 0}
    frame_times = [f.t for f in window.frames]
    out: list[WindowEvent] = []

    for e in events:
        s_, t_ = (e.start_s, e.end_s) if e.start_s <= e.end_s else (e.end_s, e.start_s)

        # Beyond plausible reading error: the model did not see this.
        if s_ > window.end_s + tolerance_s or t_ < window.start_s - tolerance_s:
            stats["rejected_out_of_window"] += 1
            continue

        # Every reported boundary must sit near a frame that was actually shown.
        if frame_times:
            near_start = min(abs(s_ - ft) for ft in frame_times)
            if near_start > frame_tolerance_s and not (
                window.start_s <= s_ <= window.end_s
            ):
                stats["rejected_no_frame"] += 1
                continue

        cs, ct = window.clamp(s_), window.clamp(t_)
        if abs(cs - s_) > 1e-6 or abs(ct - t_) > 1e-6:
            stats["clamped"] += 1
        if ct < cs:
            cs, ct = ct, cs

        out.append(WindowEvent(start_s=round(cs, 3), end_s=round(ct, 3),
                               confidence=e.confidence, evidence=e.evidence))
    return out, stats


def clamp_to_window(events: list[WindowEvent], window: Window) -> list[WindowEvent]:
    """Backwards-compatible wrapper. Prefer reconcile_times, which reports."""
    kept, _ = reconcile_times(events, window)
    return kept
