"""Data contracts. Everything the pipeline produces is one of these.

    Probe        a plain-language description compiled into something code checks
    Bracket      a candidate time range the change signal wants examined
    Observation  one timestamped structured record the observer produced
    Event        one derived, scored, timed span in the output document

These four are the spine. The point of naming them is traceability: a wrong
result must be attributable to a wrong probe, a missing bracket, or a bad
observation -- never to a mystery inside one large function. That was the
concrete failure of the previous engine, where "the model said 1.0 and was
wrong" was the whole diagnosis available.

Keep them small, keep them serialisable, and keep the raw model output next to
the parsed value wherever code does the parsing.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"

ProbeKind = Literal["state", "presence", "direction", "relation", "cessation"]
GateMode = Literal["rising", "falling", "any"]
ResultClass = Literal["model", "synthetic_oracle", "replay"]


class Probe(BaseModel):
    """A description compiled into a check code can run over observations.

    `kind` selects the derivation rule (see derive.py):

      state      the subject's `state` goes from states[0] to states[1]
      presence   the subject appears, or disappears if states=("present","absent")
      direction  the subject's displacement opposes (or follows) its facing
      relation   relations[relation_key] becomes relation_value
      cessation  motion goes moving -> stationary

    An inexpressible probe is not a failure to be hidden. It is the honest
    answer for "the alarm sounds" when no audio is decoded, and it is reported
    in the output document with its reason rather than silently scored as a
    miss.
    """

    id: str
    description: str
    expressible: bool
    reason: str = ""
    subject: str = ""
    kind: Optional[ProbeKind] = None
    states: Optional[tuple[str, str]] = None
    relation_key: Optional[str] = None
    relation_value: Optional[str] = None
    direction: Optional[Literal["backward", "forward"]] = None
    # Which observation fields this probe actually reads. The observer asks for
    # these and no more: every extra field is decode tokens spent per call, and
    # the previous engine's 80-second calls were runaway generation.
    attributes: list[str] = Field(default_factory=list)
    gate: GateMode = "rising"
    compiler: str = "unknown"  # "rules" | "llm" | "manual"

    @model_validator(mode="after")
    def _check(self) -> "Probe":
        if self.expressible:
            if not self.subject or self.kind is None:
                raise ValueError("expressible probe needs subject and kind")
            if self.kind in ("state", "presence") and not self.states:
                raise ValueError(f"{self.kind} probe needs states")
            if self.kind == "relation" and not (self.relation_key and self.relation_value):
                raise ValueError("relation probe needs relation_key and relation_value")
            if self.kind == "direction" and not self.direction:
                raise ValueError("direction probe needs direction")
        return self


class Bracket(BaseModel):
    """A time range the cheap signal thinks is worth a model call.

    `origin` is kept because it is diagnostic: an event found only inside
    `sentinel` brackets means the change signal missed it, which is a different
    repair from the model mis-describing a frame it was shown.
    """

    id: str
    start_s: float
    end_s: float
    origin: Literal["rising", "falling", "sentinel", "manual"]
    peak_score: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class Observation(BaseModel):
    """One structured look at one subject at one stamped time.

    `t` is always a time the pipeline chose and printed on the frame, never a
    time the model named. Measured on real footage: the model localises badly
    and asserts confidently, so it is never asked when -- only what.

    `raw_state` holds the model's own words and `state` the canonical token code
    matched from them. Both are kept because the parse is where meaning is lost:
    a model asked to *choose* between supplied state words scored 0.00 once the
    option order was averaged out, while free text plus string matching in code
    parsed 85% of door polls. Keeping the raw text makes a parse failure legible
    instead of indistinguishable from the subject being absent.
    """

    t: float
    subject: str
    bracket_id: str
    present: bool = False
    state: Optional[str] = None
    raw_state: Optional[str] = None
    position: Optional[tuple[float, float]] = None  # normalised (x, y), top-left origin
    facing: Optional[Literal["left", "right", "toward", "away", "unknown"]] = None
    motion: Literal["moving", "stationary", "unknown"] = "unknown"
    relations: dict[str, str] = Field(default_factory=dict)
    # The model's self-reported certainty. Recorded, and by default NOT folded
    # into the event confidence: across one labelled run the stated values took
    # three distinct levels over 48 predictions, so they could not rank anything.
    # `derive.certainty_weight` exists to make that a measurement rather than an
    # assumption.
    certainty: float = 0.5
    source: Literal["model", "oracle", "replay"] = "model"
    ok: bool = True          # False when the call failed or nothing parsed
    note: str = ""           # why, when ok is False


class Signals(BaseModel):
    """The components of confidence, published separately.

    A single opaque number cannot be argued with. These four can: a low score
    with high agreement and low coverage is a sampling problem, the same score
    with full coverage and low agreement is a perception problem, and they call
    for different fixes.
    """

    agreement: float      # fraction of observations in the run that agree
    sharpness: float      # how cleanly the transition is bracketed in time
    coverage: float       # fraction of the span actually observed
    mean_certainty: float  # mean self-reported certainty, reported not multiplied


class Event(BaseModel):
    id: str
    probe_id: str
    description: str
    start_s: float
    end_s: float
    confidence: float
    rank: int = 0
    # `start_unknown` / `end_unknown` mean the boundary was not bracketed
    # tightly enough to interpolate. Interpolating anyway across an 82-second
    # gap produced a 42-second span for a 3-second door, at full confidence.
    # An honest partial beats a precise fiction.
    partial: Literal["none", "start_unknown", "end_unknown", "both"] = "none"
    signals: Signals
    evidence: list[float] = Field(default_factory=list)   # observation times used
    bracket_ids: list[str] = Field(default_factory=list)


class Rejection(BaseModel):
    probe_id: str
    description: str
    reason: str


class RunInfo(BaseModel):
    model: str
    backend: str
    sample_fps: float
    observe_step_s: float
    observe_span_s: float
    signal: dict = Field(default_factory=dict)
    derive: dict = Field(default_factory=dict)
    prompt_version: str = ""
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time_s: float = 0.0
    # True only when the serving endpoint was asked what it is running and
    # answered with the expected id. An unverified run is not a model result.
    model_verified: bool = False
    # Oracle and replay numbers are never printed under a model heading. The
    # tag travels with the document so the evaluator cannot lose it.
    result_class: ResultClass = "model"


class EventsDocument(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video: dict
    probes: list[Probe]
    rejected: list[Rejection]
    events: list[Event]
    brackets: list[Bracket]
    # failures: calls that errored or parsed to nothing. repairs: gaps the
    # pipeline re-observed. Both are counters rather than log lines so a run
    # that quietly degraded is visible in the artifact.
    coverage: dict = Field(default_factory=dict)
    run: RunInfo
