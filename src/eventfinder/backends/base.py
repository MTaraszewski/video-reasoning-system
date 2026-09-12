"""The observer interface, its prompts, and its response schemas.

One idea runs through all of it: **the model is asked what it sees at times it
can read off the frame, never when something happened.** The schema enforces
that rather than the prompt asking politely -- `t` is an enum over the exact
stamp list, so a reply cannot reference a time that was never shown.

Two call shapes:

  observe   what is the subject's state at each listed time? The pipeline
            derives events from the answers. This is the primary path.
  verify    the same, plus a top-level `matches` for whether the clip shows the
            description. Recorded, NOT trusted: whether the model's own verdict
            beats deriving from the states is an open measurement, and until it
            is made, `matches` never gates an emission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..models import Observation

PROMPT_VERSION = "obs-v1"

# Must match OverlayConfig.format. The prompt tells the model what the burned-in
# stamp looks like; a mismatch here means it is told to read something that is
# not on the frame.
STAMP_EXAMPLE = "t=12.500s"


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    repairs: int = 0
    failures: int = 0
    latency_s: list[float] = field(default_factory=list)
    # Every exchange, verbatim. This is what makes a GPU session reusable: one
    # real run becomes a replay corpus that derivation can be iterated against
    # for free, carrying the model's true error structure rather than an
    # invented noise model.
    exchanges: list[dict] = field(default_factory=list)

    @property
    def mean_latency_s(self) -> float:
        return round(sum(self.latency_s) / len(self.latency_s), 3) if self.latency_s else 0.0


@dataclass
class Verdict:
    """One `verify` reply: the model's own yes/no beside the observations."""

    observations: list[Observation]
    matches: bool | None = None
    says: str = ""
    confidence: float | None = None


class Reasoner(Protocol):
    name: str
    model: str
    usage: Usage

    def verify_model(self) -> tuple[bool, str]: ...

    def observe(self, clip_path: str, stamp_times: list[float], subject: str,
                attributes: list[str], bracket_id: str) -> list[Observation]: ...

    def verify(self, clip_path: str, stamp_times: list[float], subject: str,
               attributes: list[str], bracket_id: str, description: str) -> Verdict: ...


# --- schemas ---------------------------------------------------------------

def _record_props(attributes: list[str], stamp_times: list[float]) -> dict:
    props: dict = {
        # An enum, not a number. The model cannot name a time it was not shown,
        # which takes localisation out of its job by construction rather than by
        # instruction.
        "t": {"type": "number", "enum": [round(float(t), 3) for t in stamp_times]},
        "present": {"type": "boolean"},
        "certainty": {"type": "number", "minimum": 0, "maximum": 1},
    }
    if "state" in attributes:
        # Free text, matched in code. Offering a closed list to pick from is a
        # different task from describing, and which one it does better is
        # measured per probe rather than assumed.
        props["state"] = {"type": "string", "maxLength": 40}
    if "position" in attributes:
        props["position"] = {"type": "array", "minItems": 2, "maxItems": 2,
                             "items": {"type": "number", "minimum": 0, "maximum": 1}}
    if "facing" in attributes:
        props["facing"] = {"type": "string",
                           "enum": ["left", "right", "toward", "away", "unknown"]}
    if "motion" in attributes:
        props["motion"] = {"type": "string", "enum": ["moving", "stationary", "unknown"]}
    if "relations" in attributes:
        props["relations"] = {"type": "object", "additionalProperties": {"type": "string"}}
    return props


def observation_schema(stamp_times: list[float], attributes: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array", "minItems": 1, "maxItems": len(stamp_times),
                "items": {"type": "object",
                          "properties": _record_props(attributes, stamp_times),
                          "required": ["t", "present", "certainty"],
                          "additionalProperties": False},
            }
        },
        "required": ["observations"],
        "additionalProperties": False,
    }


def verify_schema(stamp_times: list[float], attributes: list[str]) -> dict:
    s = observation_schema(stamp_times, attributes)
    s["properties"]["matches"] = {"type": "boolean"}
    s["properties"]["says"] = {"type": "string", "maxLength": 120}
    s["properties"]["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
    s["required"] = ["matches", "observations"]
    return s


# --- prompts ---------------------------------------------------------------

_FIELD_HELP = {
    "state": "state: one short word for the subject's condition at that moment "
             "(open, closed, standing, sitting, moving, stationary, ...)",
    "position": "position: [x, y] of the subject's centre as fractions of the frame, "
                "x to the right, y down",
    "facing": "facing: which way the subject's front points -- left, right, toward, away, unknown",
    "motion": "motion: moving or stationary, compared with the previous listed time",
    "relations": "relations: key -> value pairs such as held_by: person, inside: vehicle",
}

_SYSTEM = (
    "You are an observer. Every frame of this clip has its absolute time burned into the "
    f"bottom-left corner, written like {STAMP_EXAMPLE}. "
    "For each time you are asked about, report what you SEE about the named subject at that "
    "moment. Describe states, not events. Do not say when something happened, do not guess "
    "what occurred between the times listed, and do not summarise the clip. "
    "If the subject is not visible at a time, set present=false and omit the other fields. "
    "Reply with JSON only."
)


def observer_prompt(subject: str, attributes: list[str],
                    stamp_times: list[float]) -> tuple[str, str]:
    fields = [_FIELD_HELP[a] for a in attributes if a in _FIELD_HELP]
    times = ", ".join(f"{t:.3f}" for t in stamp_times)
    user = (f"Subject: {subject}\n"
            f"Report at exactly these times, as printed on the frames: {times}\n"
            "Per time: present (true/false); certainty (0..1)"
            + ("; " + "; ".join(fields) if fields else ""))
    return _SYSTEM, user


def verify_prompt(subject: str, attributes: list[str], stamp_times: list[float],
                  description: str) -> tuple[str, str]:
    """The observation task, plus the model's own verdict.

    The verdict is asked for after the per-time observations, and the schema
    requires both. Whether it beats deriving from the states is the thing being
    measured; nothing downstream acts on it until that measurement exists.
    """
    system, user = observer_prompt(subject, attributes, stamp_times)
    system = system.replace(
        "Reply with JSON only.",
        "Then say whether this clip actually shows the event you are given. Answer false if "
        "it does not, including when the subject is absent or nothing changes. Reply with "
        "JSON only.")
    user = (f"Event: {description}\n" + user +
            "\nThen: matches (true/false) -- does the clip show that event; "
            "says -- one short line describing what you actually saw; confidence (0..1).")
    return system, user
