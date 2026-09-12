"""Compile a plain-language description into a Probe.

Two compilers behind one interface:

  RulesCompiler  deterministic, offline, no model. Covers the verbs the labelled
                 set actually uses. This is the default, and the oracle path
                 depends on it being model-free.
  LLMCompiler    one short *text-only* call returning a probe as JSON under a
                 schema, with the rules compiler as fallback. Falling back is
                 recorded in `reason`, never silent.

The compiled probe is echoed into the output document. That is the point: a
wrong result becomes traceable to a wrong probe rather than to a mystery. The
previous engine had no such artefact -- when it emitted "Empty hallway with a
closed door and no visible people" as evidence *for* a person entering, there
was nothing to inspect between the description and the answer.

Rule order is load-bearing. The table is matched top-down and the first hit
wins, so every specific rule must precede the general one it would otherwise be
swallowed by. `test_compile.py` pins the order against the labelled set.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .models import Probe

# --- subjects --------------------------------------------------------------

# Multi-word first: `vehicle door` must win over `door`, and `door` must win
# over `vehicle`. Measured cost of getting this wrong -- a shared caption that
# conflated a car door with a building door invented a 92-second false positive
# on a 3-second event.
_SUBJECT_WORDS = [
    "vehicle door", "car door", "building door", "trunk", "hatch",
    "forklift", "vehicle", "truck", "car", "van", "bus", "bicycle",
    "door", "gate", "boot", "machine", "conveyor", "robot", "pallet", "bag", "box", "package",
    "person", "worker", "man", "woman", "someone", "somebody", "dog", "light", "arm",
]

_SUBJECT_ALIASES = {
    "someone": "person", "somebody": "person", "worker": "person",
    "man": "person", "woman": "person",
    "car door": "vehicle door", "building door": "door",
    "truck": "vehicle", "van": "vehicle", "bus": "vehicle", "car": "vehicle",
}


def subject_of(desc: str) -> str:
    low = desc.lower()
    for w in _SUBJECT_WORDS:
        if re.search(rf"\b{re.escape(w)}s?\b", low):
            return _SUBJECT_ALIASES.get(w, w)
    m = re.search(r"\b(?:a|an|the)\s+([a-z]+)", low)
    return m.group(1) if m else "object"


# --- refusals --------------------------------------------------------------

_NOT_EXPRESSIBLE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(alarm|beep|sound|noise|says?|shouts?|siren|speaks?|honks?)\b", re.I),
     "audio-only cue; no audio track is decoded"),
    (re.compile(r"\b(second|third|next|another|first) time\b", re.I),
     "requires counting across the whole video; the observer sees short windows only"),
    (re.compile(r"\b(intends?|wants?|thinks?|decides?|angry|happy|suspicious)\b", re.I),
     "internal state, not an observable"),
    # Not a visual state at any resolution a fixed overhead camera offers. The
    # observable proxy -- a person at a counter, an object changing hands -- is
    # a different event that happens without a purchase and vice versa. Two of
    # the fifteen labelled events are this, and rejecting them costs real
    # recall; asserting them would cost the right to be believed on the rest.
    # A manual probe can express the proxy explicitly if that trade is wanted.
    (re.compile(r"\b(buys?|bought|purchas|pays?|paid|sells?)\b", re.I),
     "a commercial transaction is not a visual state; the observable proxy "
     "(a person at a counter, an object changing hands) is a different event"),
]

# --- rules -----------------------------------------------------------------
#
# `subject` in a spec overrides the word scan, for the cases where the grammar
# says which noun the probe is about and the scan would pick the other one.

_RULES: list[tuple[re.Pattern, dict]] = [
    # -- doors, most specific first ----------------------------------------
    (re.compile(r"\b(vehicle|car|truck|van)\s+(door|boot|trunk|hatch|tailgate)\b.*\b(open)", re.I),
     dict(subject="vehicle door", kind="state", states=("closed", "open"), gate="rising")),
    (re.compile(r"\b(vehicle|car|truck|van)\s+(door|boot|trunk|hatch|tailgate)\b.*\b(clos|shut)", re.I),
     dict(subject="vehicle door", kind="state", states=("open", "closed"), gate="rising")),
    (re.compile(r"\b(door|gate)\b.*\b(open)|\bopens?\b.*\b(door|gate)\b", re.I),
     dict(subject="door", kind="state", states=("closed", "open"), gate="rising")),
    (re.compile(r"\b(door|gate)\b.*\b(clos|shut)|\b(clos|shut)\w*\b.*\b(door|gate)\b", re.I),
     dict(subject="door", kind="state", states=("open", "closed"), gate="rising")),

    # -- going in: the subject leaves view ---------------------------------
    # ASSUMPTION, and the one most likely to be wrong. On MEVA's outdoor fixed
    # cameras `Enter_Facility` is a person walking *out of frame* through a
    # doorway, so entering is a disappearance. The same words on an indoor
    # camera pointed at the inside of that door mean the opposite. Encoded this
    # way because it is what the labelled footage shows; settled by watching one
    # clip, not by argument.
    (re.compile(r"\b(enters?|walks? in(to)?|goes? in(to)?|gets? in(to)?|climbs? in(to)?|boards?)\b"
                r".*\b(door|gate|building|facility|vehicle|car|truck|van|bus|room)\b", re.I),
     dict(subject="person", kind="presence", states=("present", "absent"), gate="rising")),

    # -- abandonment, before the `leaves` = exits rule steals it -----------
    (re.compile(r"\b(leaves?|left|abandons?|forgets?)\b.*\b(behind)\b"
                r"|\b(leaves?|abandons?)\b\s+(a|an|the)\s+\w+(\s+or\s+\w+)?\s*$", re.I),
     dict(kind="relation", relation_key="held_by", relation_value="none", gate="rising",
          reason="approximated: scored as the object ceasing to be held, which does not "
                 "require that it stays there after the person leaves")),

    # -- coming out: the subject enters view -------------------------------
    (re.compile(r"\b(comes?|steps?|walks?|gets?|climbs?)\s+out\b|\b(exits?|leaves?|departs?|emerges?|alights?)\b", re.I),
     dict(subject="person", kind="presence", states=("absent", "present"), gate="rising")),

    # -- appearing in frame, no portal named -------------------------------
    (re.compile(r"\b(enters?|appears?|comes? into)\b.*\b(from|view|frame|scene|left|right|screen)\b", re.I),
     dict(kind="presence", states=("absent", "present"), gate="rising")),

    # -- motion ------------------------------------------------------------
    (re.compile(r"\b(revers|backs? up|backs? away|backs? out)", re.I),
     dict(kind="direction", direction="backward", gate="rising")),
    (re.compile(r"\b(stops? mov|stops?\b|halts?|comes? to (a )?(rest|stop)|stands? still|parks?)", re.I),
     dict(kind="cessation", gate="falling")),
    (re.compile(r"\b(starts? mov|begins? mov|drives? off|moves? off|pulls? away|sets? off)", re.I),
     dict(kind="state", states=("stationary", "moving"), gate="rising")),

    # -- posture -----------------------------------------------------------
    (re.compile(r"\b(sits? down|sits? on|takes? a seat)", re.I),
     dict(subject="person", kind="state", states=("standing", "sitting"), gate="rising")),
    (re.compile(r"\b(stands? up|rises?|gets? up)", re.I),
     dict(subject="person", kind="state", states=("sitting", "standing"), gate="rising")),

    # -- objects changing hands --------------------------------------------
    (re.compile(r"\b(hands?|hand(s|ed)|pass(es|ed)?|gives?|delivers?)\b.*\bto\b", re.I),
     dict(kind="relation", relation_key="held_by", relation_value="change", gate="rising")),
    (re.compile(r"\b(picks? up|lifts?|grabs?|takes? hold)", re.I),
     dict(kind="relation", relation_key="held_by", relation_value="any", gate="rising")),
    (re.compile(r"\b(puts? down|drops?|places?|sets? down|releases?)\b(?!\s+\w+\s+off)", re.I),
     dict(kind="relation", relation_key="held_by", relation_value="none", gate="rising")),

    # -- compound, approximated (the approximation is stated in `reason`) ----
    (re.compile(r"\b(drops?|lets?)\s+(someone|somebody|a person|them|him|her)\s+off\b|\bdrop[- ]?off\b", re.I),
     dict(subject="person", kind="presence", states=("absent", "present"), gate="rising",
          reason="approximated: scored as the person appearing, which ignores the vehicle "
                 "stopping and so will also fire for a person who simply walks into view")),
    (re.compile(r"\bcrosses?\b.*\b(scene|frame|view|road|street)\b", re.I),
     dict(kind="presence", states=("absent", "present"), gate="rising",
          reason="approximated: scored as the crossing's start; the exit on the far side "
                 "is a second transition this probe does not require")),
]

# Which observation fields each kind reads. The observer asks for these and no
# more -- every unused field is decode tokens per call.
_ATTRS = {
    "state":     ["present", "state"],
    "presence":  ["present"],
    "direction": ["present", "position", "facing", "motion"],
    "relation":  ["present", "relations"],
    "cessation": ["present", "motion"],
}

# The vocabulary the observer is offered per kind, so that what the model is
# asked to say and what code matches cannot drift apart.
STATE_VOCAB = {
    "door":         ["open", "closed", "partially open"],
    "vehicle door": ["open", "closed", "partially open"],
    "person":       ["standing", "sitting", "walking", "crouching", "lying"],
}


class RulesCompiler:
    name = "rules"

    def compile(self, pid: str, desc: str) -> Probe:
        for pat, why in _NOT_EXPRESSIBLE:
            if pat.search(desc):
                return Probe(id=pid, description=desc, expressible=False,
                             reason=why, compiler=self.name)
        for pat, spec in _RULES:
            if pat.search(desc):
                spec = dict(spec)
                kind = spec["kind"]
                subject = spec.pop("subject", None) or self._subject(desc, kind)
                return Probe(id=pid, description=desc, expressible=True,
                             subject=subject, attributes=_ATTRS[kind],
                             compiler=self.name, **spec)
        return Probe(id=pid, description=desc, expressible=False, compiler=self.name,
                     reason="no rule matched; add a rule, supply a manual probe, "
                            "or enable the LLM compiler")

    @staticmethod
    def _subject(desc: str, kind: str) -> str:
        # A relation probe is about the thing whose `held_by` changes, not about
        # whoever moves it: in "someone hands an object to another person" the
        # subject is the object. The word scan would pick "person" and the probe
        # would then watch the wrong noun.
        if kind == "relation":
            m = re.search(r"\b(?:a|an|the)\s+([a-z]+)\s+(?:to|up|down|over)\b", desc.lower())
            if m:
                return m.group(1)
        return subject_of(desc)


# --- LLM-backed ------------------------------------------------------------

PROBE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "expressible": {"type": "boolean"},
        "reason": {"type": "string"},
        "subject": {"type": "string"},
        "kind": {"type": "string", "enum": ["state", "presence", "direction", "relation", "cessation"]},
        "states": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
        "relation_key": {"type": "string"},
        "relation_value": {"type": "string"},
        "direction": {"type": "string", "enum": ["backward", "forward"]},
        "gate": {"type": "string", "enum": ["rising", "falling", "any"]},
    },
    "required": ["expressible", "reason", "subject", "kind", "gate"],
    "additionalProperties": False,
}

COMPILER_PROMPT = """You turn an event description into a probe that code checks against
per-second observations of ONE subject. Each observation records: present (bool),
state (one short word), position (x,y in 0..1), facing (left/right/toward/away),
motion (moving/stationary), relations (e.g. held_by: person).

Choose exactly one kind:
- state: the subject's state word changes from states[0] to states[1]
- presence: the subject appears (states ["absent","present"]) or disappears (["present","absent"])
- direction: the subject moves backward (opposite to facing) or forward
- relation: relations[relation_key] becomes relation_value
- cessation: motion goes moving -> stationary (set gate to "falling")

The camera is fixed and outdoors. A person who goes in through a door LEAVES the
view, so entering a building is ["present","absent"]; coming out is the reverse.

Set expressible=false with a plain reason when the event needs audio, counting
across the whole video, intent, or a commercial transaction.
Reply with JSON only."""


class TextModel(Protocol):
    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


class LLMCompiler:
    """One text-only call per description, with the rules table as a floor.

    A failure here must never look like a refusal by the model: a fallback
    appends its cause to `reason`, so "inexpressible" and "the compiler broke"
    stay distinguishable in the output document.
    """

    name = "llm"

    def __init__(self, text_model: TextModel, fallback: RulesCompiler | None = None):
        self.tm = text_model
        self.fallback = fallback or RulesCompiler()

    def compile(self, pid: str, desc: str) -> Probe:
        try:
            raw = self.tm.complete_json(COMPILER_PROMPT, f"Description: {desc}", PROBE_JSON_SCHEMA)
            if not raw.get("expressible", False):
                return Probe(id=pid, description=desc, expressible=False,
                             reason=raw.get("reason", "") or "model declined without a reason",
                             compiler=self.name)
            kind = raw["kind"]
            states = tuple(raw["states"]) if raw.get("states") else None
            if kind == "presence" and not states:
                states = ("absent", "present")
            if kind == "cessation":
                raw["gate"] = "falling"
            return Probe(
                id=pid, description=desc, expressible=True,
                subject=(raw.get("subject") or "").lower(), kind=kind, states=states,
                relation_key=raw.get("relation_key"), relation_value=raw.get("relation_value"),
                direction=raw.get("direction"), gate=raw.get("gate", "rising"),
                attributes=_ATTRS[kind], reason=raw.get("reason", ""), compiler=self.name,
            )
        except Exception as e:
            p = self.fallback.compile(pid, desc)
            p.reason = f"{p.reason} [llm compiler failed: {type(e).__name__}: {e}]".strip()
            return p


# --- manual overrides ------------------------------------------------------

def load_manual_probes(path: str) -> dict[str, dict]:
    """Description -> probe spec. The escape hatch for anything the rules and
    the model both get wrong, and the only sanctioned way to express a proxy
    for something the compiler refuses."""
    with open(path) as f:
        return json.load(f)


def compile_all(descs: list[str], compiler, manual: dict[str, dict] | None = None) -> list[Probe]:
    out: list[Probe] = []
    for i, d in enumerate(descs):
        pid = f"p{i + 1}"
        if manual and d in manual:
            spec = dict(manual[d])
            kind = spec.get("kind")
            out.append(Probe(id=pid, description=d, expressible=True,
                             attributes=_ATTRS.get(kind, []), compiler="manual", **spec))
        else:
            out.append(compiler.compile(pid, d))
    return out
