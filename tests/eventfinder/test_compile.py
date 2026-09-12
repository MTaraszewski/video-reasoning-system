"""Phase 1: every description in the labelled set compiles to the right probe.

This is the table's regression test. The rules are matched top-down and the
first hit wins, so adding a rule can silently steal a description from a more
specific one further down -- these cases pin the order.

No GPU, no video, no model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eventfinder.compile import (
    LLMCompiler,
    RulesCompiler,
    compile_all,
    subject_of,
)

C = RulesCompiler()


def p(desc: str):
    return C.compile("p1", desc)


# --- the labelled set, description by description -------------------------
#
# (description, subject, kind, states | direction | relation)

EXPECTED = [
    # Going in is a DISAPPEARANCE. On MEVA's outdoor fixed cameras a person who
    # enters a facility walks out of frame through the doorway. Getting this
    # backwards makes every entry probe watch for the opposite transition.
    ("a person enters through the door", "person", "presence", ("present", "absent")),
    ("a person gets into a vehicle",     "person", "presence", ("present", "absent")),
    ("a person comes out through the door", "person", "presence", ("absent", "present")),
    ("a person gets out of a vehicle",   "person", "presence", ("absent", "present")),

    # The two doors must not collapse into one subject.
    ("a person opens a building door",   "door",         "state", ("closed", "open")),
    ("a vehicle door opens",             "vehicle door", "state", ("closed", "open")),

    ("a person sits down",               "person",  "state",     ("standing", "sitting")),
    ("a person stands up",               "person",  "state",     ("sitting", "standing")),
    ("a vehicle starts moving",          "vehicle", "state",     ("stationary", "moving")),
    ("a vehicle stops moving",           "vehicle", "cessation", None),
    ("a vehicle reverses",               "vehicle", "direction", "backward"),

    # The probe is about the object that changes hands, not about either person.
    ("someone hands an object to another person", "object", "relation", ("held_by", "change")),

    # Synthetic set.
    ("a red box enters from the left",   "box",     "presence",  ("absent", "present")),
    ("the machine stops moving",         "machine", "cessation", None),
]


@pytest.mark.parametrize("desc,subject,kind,detail", EXPECTED, ids=[e[0] for e in EXPECTED])
def test_labelled_descriptions_compile(desc, subject, kind, detail):
    pr = p(desc)
    assert pr.expressible, pr.reason
    assert (pr.subject, pr.kind) == (subject, kind)
    if kind in ("state", "presence"):
        assert pr.states == detail
    elif kind == "direction":
        assert pr.direction == detail
    elif kind == "relation":
        assert (pr.relation_key, pr.relation_value) == detail


def test_cessation_gates_on_falling():
    assert p("a vehicle stops moving").gate == "falling"


def test_attributes_are_minimal_per_kind():
    """Every field the observer is asked for costs decode tokens on every call."""
    assert p("a person enters through the door").attributes == ["present"]
    assert p("a vehicle door opens").attributes == ["present", "state"]
    assert "position" in p("a vehicle reverses").attributes


# --- refusals, which are answers ------------------------------------------

def test_purchase_is_refused_with_a_reason():
    """Two of fifteen labelled events. Refused on purpose, and it is recorded why."""
    pr = p("a person buys something")
    assert not pr.expressible
    assert "transaction" in pr.reason


@pytest.mark.parametrize("desc,frag", [
    ("the alarm sounds", "audio"),
    ("a person enters the second time", "counting"),
    ("a person looks suspicious", "internal state"),
])
def test_inexpressible_categories(desc, frag):
    pr = p(desc)
    assert not pr.expressible and frag in pr.reason


def test_unknown_verb_refuses_rather_than_guessing():
    """`turns left` has no rule. Saying so beats compiling it into something else."""
    pr = p("a vehicle turns left")
    assert not pr.expressible and "no rule matched" in pr.reason


# --- approximations are declared ------------------------------------------

@pytest.mark.parametrize("desc", [
    "a vehicle drops someone off",
    "a red box crosses the scene slowly",
])
def test_compound_events_declare_their_approximation(desc):
    pr = p(desc)
    assert pr.expressible and "approximated" in pr.reason


def test_drop_off_is_not_stolen_by_the_put_down_rule():
    """`drops someone off` and `drops the box` are different events."""
    assert p("a vehicle drops someone off").kind == "presence"
    assert p("a person drops the box").kind == "relation"


# --- subject scan ---------------------------------------------------------

@pytest.mark.parametrize("desc,want", [
    ("a car door opens", "vehicle door"),   # multi-word beats `door` and `car`
    ("a truck reverses", "vehicle"),        # aliased to one canonical noun
    ("someone stands up", "person"),
    ("a worker sits down", "person"),
])
def test_subject_aliasing(desc, want):
    assert p(desc).subject == want


def test_bare_subject_scan_falls_back_to_the_first_noun():
    assert subject_of("the widget wobbles") == "widget"


# --- compile_all ----------------------------------------------------------

def test_manual_probe_overrides_the_rules():
    manual = {"a person buys something": dict(
        subject="object", kind="relation", relation_key="held_by",
        relation_value="change", gate="rising")}
    out = compile_all(["a person buys something"], C, manual)
    assert out[0].expressible and out[0].compiler == "manual"


def test_ids_are_positional_and_stable():
    out = compile_all(["a vehicle reverses", "the alarm sounds"], C)
    assert [x.id for x in out] == ["p1", "p2"]
    assert out[1].expressible is False


# --- LLM compiler ---------------------------------------------------------

class _Broken:
    def complete_json(self, system, user, schema):
        raise RuntimeError("no route to host")


class _Fake:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, system, user, schema):
        return self.payload


def test_llm_failure_falls_back_and_says_so():
    """A broken compiler must not be mistakable for the model declining."""
    pr = LLMCompiler(_Broken()).compile("p1", "a vehicle reverses")
    assert pr.expressible and pr.kind == "direction"
    assert "llm compiler failed" in pr.reason


def test_llm_refusal_is_kept_as_a_refusal():
    pr = LLMCompiler(_Fake({"expressible": False, "reason": "needs audio"})).compile("p1", "x")
    assert not pr.expressible and pr.reason == "needs audio" and pr.compiler == "llm"


def test_llm_cessation_is_forced_to_falling():
    pr = LLMCompiler(_Fake({
        "expressible": True, "reason": "", "subject": "vehicle",
        "kind": "cessation", "gate": "rising",
    })).compile("p1", "a vehicle stops")
    assert pr.gate == "falling"


# --- the whole labelled corpus --------------------------------------------

def test_every_labelled_description_is_handled_without_crashing():
    """Coverage figure over the real corpus, not a sample of it."""
    descs = set()
    for f in Path("data").rglob("labels*.json"):
        for clip in json.loads(f.read_text()):
            for ev in clip["events"]:
                descs.add(ev["description"])
    probes = compile_all(sorted(descs), C)
    expressible = [x for x in probes if x.expressible]
    refused = [x for x in probes if not x.expressible]
    assert all(x.reason for x in refused), "every refusal carries a reason"
    # 16 real + 3 synthetic descriptions; the refusals are purchases and turns.
    assert len(expressible) >= 14, [x.description for x in refused]


# --- rules the corpus scan caught stealing from each other ----------------

def test_leaves_behind_is_abandonment_not_exiting():
    """`leaves` matched the exit rule and compiled abandonment into a person
    appearing -- right verb, wrong event, wrong subject, and no error."""
    pr = p("someone leaves a bag or package behind")
    assert (pr.subject, pr.kind) == ("bag", "relation")
    assert (pr.relation_key, pr.relation_value) == ("held_by", "none")
    assert "approximated" in pr.reason


def test_boot_and_trunk_are_door_like_states():
    pr = p("a car boot is closed")
    assert (pr.subject, pr.kind, pr.states) == ("vehicle door", "state", ("open", "closed"))


def test_two_people_hug_is_refused_for_want_of_a_proximity_relation():
    """Honest gap: no relation in the contract expresses two subjects touching."""
    assert not p("two people hug").expressible
