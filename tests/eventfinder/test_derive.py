"""Phase 3: events are derived correctly from observations. No GPU, no model.

Every test here constructs the observations directly, which is the point: the
derivation is a pure function of what was reported, so its correctness can be
settled without a video or a server. If these are wrong, better perception
cannot save the system.
"""
from __future__ import annotations

import pytest

from eventfinder.config import DeriveConfig
from eventfinder.compile import RulesCompiler
from eventfinder.derive import Deriver, canonical_state, derive, rank
from eventfinder.models import Observation, Probe

C = RulesCompiler()
CFG = DeriveConfig()
STEP = 1.0


def ob(t, subject="door", **kw):
    kw.setdefault("bracket_id", "b1")
    return Observation(t=t, subject=subject, **kw)


def door(t, state, present=True, certainty=0.8):
    return ob(t, present=present, state=state, raw_state=state, certainty=certainty)


DOOR = C.compile("p1", "a person opens a building door")     # state closed -> open
ENTER = C.compile("p2", "a person enters through the door")  # presence present -> absent
STOP = C.compile("p3", "a vehicle stops moving")             # cessation
HANDOVER = C.compile("p4", "someone hands an object to another person")
REVERSE = C.compile("p5", "a vehicle reverses")


# --- state canonicalisation -------------------------------------------------

def test_last_mention_wins():
    """"the door was closed and is now open" is about a door that is open."""
    assert canonical_state("the door was closed and is now open", ("closed", "open")) == "open"
    assert canonical_state("open earlier, closed now", ("closed", "open")) == "closed"


def test_a_containing_phrase_beats_the_word_inside_it():
    """"partially open" must not parse as "open" just because the shorter word
    starts later in the string."""
    assert canonical_state("partially open", ("open", "closed", "partially open")) == "partially open"
    assert canonical_state("the door is open", ("open", "partially open")) == "open"


def test_no_vocabulary_word_parses_to_nothing():
    """Not to a guess. An unparsed reply must stay distinguishable from a state."""
    assert canonical_state("I can see a corridor", ("open", "closed")) is None
    assert canonical_state(None, ("open", "closed")) is None


def test_substrings_do_not_match():
    assert canonical_state("the door is unopened", ("open",)) is None


# --- span selection ---------------------------------------------------------

def test_span_runs_from_last_old_state_to_first_new_state():
    """Not the midpoint. The bracket the evidence supports, and nothing wider."""
    obs = [door(1, "closed"), door(2, "closed"), door(3, "closed"),
           door(4, "open"), door(5, "open"), door(6, "open")]
    e, = derive(obs, DOOR, CFG, STEP)
    assert (e.start_s, e.end_s) == (3.0, 4.0)
    assert e.partial == "none"


def test_a_wide_bracket_is_not_claimed_as_a_long_event():
    """Interpolating across an 82-second gap produced a 42-second span for a
    3-second door opening, at confidence 1.0. This is that bug's regression."""
    obs = [door(1, "closed"), door(2, "closed"), door(84, "open"), door(85, "open")]
    e, = derive(obs, DOOR, CFG, STEP)
    assert e.partial == "start_unknown"
    assert e.end_s - e.start_s <= CFG.max_interp_gap_s
    assert e.end_s == 84.0
    assert e.signals.sharpness < 0.02      # and the score says why


def test_sharpness_falls_as_the_bracket_widens():
    tight = derive([door(1, "closed"), door(2, "closed"), door(3, "open"), door(4, "open")],
                   DOOR, CFG, STEP)[0]
    loose = derive([door(1, "closed"), door(2, "closed"), door(5, "open"), door(6, "open")],
                   DOOR, CFG, STEP)[0]
    assert tight.signals.sharpness == 1.0 > loose.signals.sharpness


# --- guards -----------------------------------------------------------------

def test_a_single_misparse_does_not_make_an_event():
    """min_run: one stray reading is noise, not a door opening."""
    obs = [door(1, "closed"), door(2, "closed"), door(3, "open"),
           door(4, "closed"), door(5, "closed")]
    assert derive(obs, DOOR, CFG, STEP) == []


def test_a_transition_that_does_not_hold_is_not_witnessed():
    obs = [door(1, "closed"), door(2, "closed"), door(3, "open"), door(3.2, "open")]
    assert derive(obs, DOOR, CFG, STEP) == []          # held 0.2s, needs 1.0s


def test_failed_observations_do_not_split_a_run():
    """An unparsed reply is absence of evidence, not a change of state. Treating
    it as a break would turn one event into two."""
    obs = [door(1, "closed"), door(2, "closed"),
           ob(3, ok=False, note="no JSON"),
           door(4, "open"), door(5, "open")]
    e, = derive(obs, DOOR, CFG, STEP)
    assert (e.start_s, e.end_s) == (2.0, 4.0)


def test_the_reverse_transition_is_not_the_event():
    """A door closing is not a door opening."""
    obs = [door(1, "open"), door(2, "open"), door(3, "closed"), door(4, "closed")]
    assert derive(obs, DOOR, CFG, STEP) == []


def test_absent_subject_contributes_no_state():
    obs = [door(1, "closed"), door(2, "closed"),
           door(3, "open", present=False), door(4, "open", present=False)]
    assert derive(obs, DOOR, CFG, STEP) == []


# --- the other probe kinds --------------------------------------------------

def test_presence_entering_is_a_disappearance():
    """On an outdoor fixed camera, going in through a door means leaving view."""
    obs = [ob(t, "person", present=True) for t in (1, 2)] + \
          [ob(t, "person", present=False) for t in (3, 4)]
    e, = derive(obs, ENTER, CFG, STEP)
    assert (e.start_s, e.end_s) == (2.0, 3.0)


def test_cessation_reads_motion_not_state():
    obs = [ob(t, "vehicle", present=True, motion="moving") for t in (1, 2)] + \
          [ob(t, "vehicle", present=True, motion="stationary") for t in (3, 4)]
    e, = derive(obs, STOP, CFG, STEP)
    assert (e.start_s, e.end_s) == (2.0, 3.0)


def test_unknown_motion_is_not_stationary():
    obs = [ob(t, "vehicle", present=True, motion="moving") for t in (1, 2)] + \
          [ob(t, "vehicle", present=True, motion="unknown") for t in (3, 4)]
    assert derive(obs, STOP, CFG, STEP) == []


def test_relation_change_of_holder():
    obs = [ob(t, "object", present=True, relations={"held_by": "person a"}) for t in (1, 2)] + \
          [ob(t, "object", present=True, relations={"held_by": "person b"}) for t in (3, 4)]
    e, = derive(obs, HANDOVER, CFG, STEP)
    assert (e.start_s, e.end_s) == (2.0, 3.0)


def test_relation_unchanged_is_not_a_handover():
    obs = [ob(t, "object", present=True, relations={"held_by": "person a"}) for t in (1, 2, 3, 4)]
    assert derive(obs, HANDOVER, CFG, STEP) == []


# --- direction, the riskiest kind ------------------------------------------

def _moving(times, xs, facing="right"):
    return [ob(t, "vehicle", present=True, position=(x, 0.5), facing=facing)
            for t, x in zip(times, xs)]


def test_reversing_is_displacement_against_facing():
    e, = derive(_moving([1, 2, 3, 4], [0.8, 0.7, 0.6, 0.5]), REVERSE, CFG, STEP)
    assert (e.start_s, e.end_s) == (1.0, 4.0)


def test_driving_forward_is_not_reversing():
    assert derive(_moving([1, 2, 3, 4], [0.2, 0.3, 0.4, 0.5]), REVERSE, CFG, STEP) == []


def test_unknown_facing_is_refused_rather_than_assumed():
    """Direction needs position AND facing to be right at the same time. With no
    heading there is no answer, and guessing one manufactures events."""
    assert derive(_moving([1, 2, 3, 4], [0.8, 0.7, 0.6, 0.5], facing="unknown"),
                  REVERSE, CFG, STEP) == []


# --- confidence -------------------------------------------------------------

def test_a_dissenting_observation_lowers_agreement():
    """Agreement is counted over every parsed observation in the span, not over
    the runs -- runs are homogeneous by construction, so scoring them against
    their own token would always return 1.0 and measure nothing."""
    clean = derive([door(t, "closed") for t in (1, 2, 3)] + [door(t, "open") for t in (4, 5, 6)],
                   DOOR, CFG, STEP)[0]
    noisy = derive([door(1, "closed"), door(2, "open"), door(3, "closed"), door(4, "closed")] +
                   [door(t, "open") for t in (5, 6)], DOOR, CFG, STEP)[-1]
    assert clean.signals.agreement == 1.0
    assert noisy.signals.agreement < 1.0


def test_unparsed_observations_are_not_charged_to_agreement():
    """Their cost is coverage's; charging them twice would make a sparse run
    look like a contradicted one."""
    obs = [door(1, "closed"), door(2, "closed"), ob(3, ok=False),
           door(4, "open"), door(5, "open")]
    e, = derive(obs, DOOR, CFG, STEP)
    assert e.signals.agreement == 1.0 and e.signals.coverage < 1.0


def test_confidence_is_the_product_of_its_published_parts():
    e, = derive([door(t, "closed") for t in (1, 2, 3)] + [door(t, "open") for t in (4, 5, 6)],
                DOOR, CFG, STEP)
    s = e.signals
    assert e.confidence == pytest.approx(s.agreement * s.sharpness * s.coverage, abs=1e-4)


def test_self_reported_certainty_is_recorded_but_not_folded_in():
    low = derive([door(t, "closed", certainty=0.1) for t in (1, 2, 3)] +
                 [door(t, "open", certainty=0.1) for t in (4, 5, 6)], DOOR, CFG, STEP)[0]
    high = derive([door(t, "closed", certainty=0.99) for t in (1, 2, 3)] +
                  [door(t, "open", certainty=0.99) for t in (4, 5, 6)], DOOR, CFG, STEP)[0]
    assert low.confidence == high.confidence
    assert low.signals.mean_certainty < high.signals.mean_certainty


def test_certainty_weight_makes_it_an_experiment():
    cfg = DeriveConfig(certainty_weight=0.5)
    low = derive([door(t, "closed", certainty=0.1) for t in (1, 2, 3)] +
                 [door(t, "open", certainty=0.1) for t in (4, 5, 6)], DOOR, cfg, STEP)[0]
    high = derive([door(t, "closed", certainty=0.99) for t in (1, 2, 3)] +
                  [door(t, "open", certainty=0.99) for t in (4, 5, 6)], DOOR, cfg, STEP)[0]
    assert low.confidence < high.confidence


def test_min_confidence_filters():
    obs = [door(1, "closed"), door(2, "closed"), door(40, "open"), door(41, "open")]
    assert derive(obs, DOOR, DeriveConfig(min_confidence=0.5), STEP) == []
    assert derive(obs, DOOR, DeriveConfig(min_confidence=0.0), STEP) != []


# --- ranking ----------------------------------------------------------------

def test_rank_orders_by_confidence_and_assigns_ids():
    a = derive([door(t, "closed") for t in (1, 2, 3)] + [door(t, "open") for t in (4, 5, 6)],
               DOOR, CFG, STEP)[0]
    b = derive([door(1, "closed"), door(2, "closed"), door(9, "open"), door(10, "open")],
               DOOR, CFG, STEP)[0]
    out = rank([b, a])
    assert out[0] is a and out[0].rank == 1 and out[0].id == "e1"


# --- incremental ------------------------------------------------------------

def test_deriver_emits_each_event_once():
    d = Deriver([DOOR], CFG, STEP)
    d.add(DOOR.id, [door(1, "closed"), door(2, "closed"), door(3, "open"), door(4, "open")])
    first = d.step(now=4.0)
    again = d.step(now=5.0)
    assert len(first) == 1 and again == []


def test_deriver_waits_until_the_transition_is_witnessed():
    """The live path's latency floor: an event is not emitted before an
    observation after it has held."""
    d = Deriver([DOOR], CFG, STEP)
    d.add(DOOR.id, [door(1, "closed"), door(2, "closed"), door(3, "open")])
    assert d.step(now=3.0) == []
    d.add(DOOR.id, [door(4, "open")])
    assert len(d.step(now=4.0)) == 1


def test_deriver_ages_observations_out():
    d = Deriver([DOOR], CFG, STEP, window_s=5.0)
    d.add(DOOR.id, [door(t, "closed") for t in (1, 2, 3)])
    d.step(now=20.0)
    assert d.buf[DOOR.id] == []


def test_deriver_ignores_inexpressible_probes():
    refused = C.compile("p9", "the alarm sounds")
    d = Deriver([refused], CFG, STEP)
    assert d.probes == [] and d.step(now=1.0) == []
