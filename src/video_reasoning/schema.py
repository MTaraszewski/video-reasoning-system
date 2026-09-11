"""The client contract.

`FindEventsResult` is the product surface. Everything else in this package exists
to produce it. It is validated before it leaves the process, so a malformed
response is our bug and never the client's problem.

Field names, units and shape are fixed. Changing them is a breaking change.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class WindowEvent(BaseModel):
    """One candidate from a single (window, query) model call.

    Internal. Times are absolute video seconds — the model reads them off the
    burned-in overlay, so no per-window remapping is ever needed.
    """

    start_s: float = Field(..., description="Event start, absolute seconds.")
    end_s: float = Field(..., description="Event end, absolute seconds.")
    # Optional, because real output often omits it or echoes the schema's own
    # range back as a value. An event with times but no confidence is still an
    # event; discarding it would lose a real detection over a missing field.
    # The default is deliberately middling — it asserts nothing.
    confidence: float | None = Field(0.5, ge=0.0, le=1.0)
    evidence: str = Field("", description="One short line of visual justification.")

    @field_validator("confidence", mode="before")
    @classmethod
    def _default_confidence(cls, v):
        return 0.5 if v is None else v

    @model_validator(mode="after")
    def _ordered(self) -> "WindowEvent":
        if self.end_s < self.start_s:
            self.start_s, self.end_s = self.end_s, self.start_s
        return self


class WindowResponse(BaseModel):
    """Exactly what a single model call must return. Kept tiny for reliability."""

    events: list[WindowEvent] = Field(default_factory=list)


class Event(BaseModel):
    """A merged, client-facing event."""

    description: str = Field(..., description="Which query this matched.")
    start_s: float = Field(..., description="Absolute seconds from video start.")
    end_s: float = Field(..., description="Absolute seconds from video start.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Heuristic ranking signal, NOT a calibrated probability. Good for "
            "sorting and thresholding; not a probability. How it is computed "
            "depends on the strategy that produced the event. `windows` combines "
            "the model's self-report with agreement across overlapping windows. "
            "`states` uses no self-report at all: it is agreement across polls, "
            "times how tightly the boundaries are bracketed, times how much of "
            "the interval was actually observed."
        ),
    )
    evidence: str = Field("", description="What the model claimed to see.")
    partial: bool = Field(
        False,
        description=(
            "True if the event touches the edge of the observed region and its "
            "true extent is therefore unknown. Reported rather than silently "
            "clipped."
        ),
    )
    source_windows: list[int] = Field(
        default_factory=list,
        description=(
            "Provenance: which windows produced it. Always empty for the `states` "
            "strategy, which has no windows -- its provenance is the poll timeline "
            "in `FindEventsResult.polls`."
        ),
    )

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class RunInfo(BaseModel):
    """How this result was produced. Present so a number can be traced back."""

    model: str
    backend: Literal["vllm", "stub"] = "vllm"
    sample_fps: float
    window_s: float
    stride_s: float
    windows: int
    model_calls: int
    elapsed_s: float
    stub: bool = Field(
        False,
        description=(
            "True if produced by the deterministic stub rather than a real model. "
            "Stub results are never valid evidence and the eval harness refuses "
            "to score them."
        ),
    )


class FindEventsResult(BaseModel):
    """The public output."""

    video: str
    duration_s: float
    queries: list[str]
    events: list[Event] = Field(default_factory=list)
    run: RunInfo

    # The state timeline behind the events, when the `states` strategy produced
    # them. Every poll: the time, the subject asked about, the caption the model
    # returned, and the state parsed from it.
    #
    # Kept because it IS the reasoning. Each state decision is made by reading
    # that text, so without it an event is an assertion rather than a trace, and
    # the run's most interesting output -- the model visibly working out whether a
    # door is open -- was being generated, used, and then discarded. Only 200
    # characters of one poll survived, as an event's `evidence`.
    polls: list[dict] | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "FindEventsResult":
        # Ordered by start time — a documented guarantee.
        self.events.sort(key=lambda e: (e.start_s, e.end_s))
        # Nothing may be reported outside the video's real duration.
        for e in self.events:
            if e.start_s < 0 or e.end_s > self.duration_s + 1e-6:
                raise ValueError(
                    f"event {e.start_s:.3f}-{e.end_s:.3f}s falls outside the "
                    f"video's {self.duration_s:.3f}s duration"
                )
        return self
