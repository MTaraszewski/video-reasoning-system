"""Caption a window, read the state out of the words, derive events from changes.

The alternative to asking the model *when* something happened. Approach 1 asked
that directly and could not establish whether the event was present at all: it
reported an event on 47% of pairs where one existed and 39% where none did.

Here the model is asked only what it sees, in words, and everything temporal is
done in code:

    caption  ->  parse  ->  transition

No forced choice, no logprobs, no threshold. Each of those was tried and each
produced a plausible-looking wrong answer: forced choice was dominated by
option-order bias, an order-averaged text classifier scored exactly 0.00 on every
description because it was choosing purely by position, and a baseline threshold
had to be retuned per clip.

Library form of what `scripts/reason_timeline.py` and `run_transitions.py` probe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Conversational opener the model prefixes to almost every answer. It is not a
# description, it consumes the token budget before the content arrives, and it was
# what the parser read when this was scored by a model.
_PREAMBLE = re.compile(
    r"^\s*(got it|okay|ok|sure|alright|let'?s|let me|first,?)\b[^.]*[.:]\s*", re.I)

# Words that name both options without asserting either: "its state (open/closed)
# does not change" reads as a confident "closed" under last-mention-wins.
_ENUM = re.compile(r"[a-z]+\s*/\s*[a-z]+")

_STOP = {"the", "is", "a", "an", "in", "of", "at", "on", "and", "it", "its"}


@dataclass
class StatePoll:
    """One timestep: what the model said, and what state that means."""

    t: float
    text: str
    state: str | None      # None when the description asserts no state
    truncated: bool = False


@dataclass
class Transition:
    """A change between consecutive known states."""

    at: float              # midpoint of the bracket — see `edge` before trusting it
    from_state: str
    to_state: str
    bracket: tuple[float, float]

    @property
    def width(self) -> float:
        """How much time the change could have happened in. The honest error bar."""
        lo, hi = self.bracket
        return hi - lo


@dataclass
class Boundaries:
    """State at the edges of the observed span.

    This is how an event seen only partially gets reported. If the span opens
    already in the target state, the change predates it; if it closes still in
    that state, the return is unobserved. Approach 1 could only infer partiality
    from a merged span touching a window edge.
    """

    first: str | None = None
    last: str | None = None
    partial_before: bool = False
    partial_after: bool = False


def clean(text: str) -> str:
    """Strip the reasoning block and the conversational opener."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    prev = None
    while prev != text:
        prev = text
        text = _PREAMBLE.sub("", text, count=1).strip()
    return text


def parse_state(text: str, states: tuple[str, str]) -> str | None:
    """Read which state a description asserts. Deterministic, not a model call.

    A model was tried for this and failed completely: order-averaged, it scored
    exactly 0.00 on every description, which is the signature of answering by
    position and never reading the text. String matching cannot acquire a
    preference for whichever option came last.

    Last mention wins, because these descriptions reason before concluding --
    "closed in the initial frames and then opens" is a transition and the later
    word is the answer.
    """
    a, b = states
    wa = [w for w in re.findall(r"[a-z]+", a.lower()) if w not in _STOP]
    wb = [w for w in re.findall(r"[a-z]+", b.lower()) if w not in _STOP]
    only_a = [w for w in wa if w not in wb]
    only_b = [w for w in wb if w not in wa]

    low = text.lower()
    for m in _ENUM.finditer(low):
        low = low[: m.start()] + " " * (m.end() - m.start()) + low[m.end():]

    pos_a = max((low.rfind(w) for w in only_a), default=-1)
    pos_b = max((low.rfind(w) for w in only_b), default=-1)
    if pos_a < 0 and pos_b < 0:
        return None
    return a if pos_a > pos_b else b


def transitions(polls: list[StatePoll]) -> list[Transition]:
    """Changes between consecutive KNOWN states.

    A poll with no state is skipped rather than treated as a change: a
    description that says nothing about state is not evidence the state moved.
    """
    out: list[Transition] = []
    prev: StatePoll | None = None
    for p in polls:
        if p.state is None:
            continue
        if prev is not None and p.state != prev.state:
            out.append(Transition(at=round((prev.t + p.t) / 2, 3),
                                  from_state=prev.state, to_state=p.state,
                                  bracket=(prev.t, p.t)))
        prev = p
    return out


def edge(tr: Transition, *, max_bracket_s: float, span_s: float,
         opening: bool) -> tuple[float, bool]:
    """Place a transition, refusing to interpolate across an unobserved gap.

    A bracket's midpoint is a fair estimate when the bracket is one step wide:
    the change happened somewhere in there and the middle is as good a guess as
    any. It is not fair when the bracket is 82 seconds wide, which is what
    triggered polling produces. Measured on admin.G326: a door seen open at
    7.5s and closed at 90.0s was reported open until 48.75s -- an interval forty
    times longer than the event, carrying confidence 1.0. Nothing was wrong with
    the polls. The derivation assumed a uniform grid and kept that assumption
    after the grid was removed.

    Past `max_bracket_s` we stop interpolating and report what was observed
    instead. A poll at t saw frames spanning t +/- span_s/2, so evidence for its
    state ends there; after that the state is unknown, which is not the same as
    unchanged. The caller marks such a span partial.

    Returns the time and whether the bracket was too wide to interpolate.
    """
    lo, hi = tr.bracket
    if tr.width <= max_bracket_s:
        return tr.at, False
    half = span_s / 2
    return (round(hi - half, 3), True) if opening else (round(lo + half, 3), True)


def boundaries(polls: list[StatePoll], states: tuple[str, str]) -> Boundaries:
    known = [p for p in polls if p.state is not None]
    if not known:
        return Boundaries()
    first, last = known[0].state, known[-1].state
    return Boundaries(first=first, last=last,
                      partial_before=first == states[1],
                      partial_after=last == states[1])
