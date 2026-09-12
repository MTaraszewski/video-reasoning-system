"""Phase 0: the contracts hold, and the config refuses what cannot work.

No GPU, no video, no model. If these fail, nothing downstream is worth running.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from eventfinder import config as cfg_mod
from eventfinder.models import (
    Bracket,
    Event,
    EventsDocument,
    Observation,
    Probe,
    Rejection,
    RunInfo,
    Signals,
)


# --- probes ---------------------------------------------------------------

def test_inexpressible_probe_needs_no_subject():
    """A refusal is a first-class answer, not a malformed probe."""
    p = Probe(id="p1", description="the alarm sounds", expressible=False,
              reason="audio-only cue; no audio track is decoded")
    assert p.kind is None and p.reason


@pytest.mark.parametrize("kind,extra", [
    ("state", {}),                      # missing states
    ("presence", {}),                   # missing states
    ("relation", {}),                   # missing relation_key/value
    ("direction", {}),                  # missing direction
])
def test_expressible_probe_rejects_missing_fields(kind, extra):
    with pytest.raises(ValidationError):
        Probe(id="p1", description="d", expressible=True, subject="door",
              kind=kind, **extra)


def test_expressible_state_probe_round_trips():
    p = Probe(id="p1", description="the door opens", expressible=True,
              subject="door", kind="state", states=("closed", "open"),
              attributes=["present", "state"], compiler="rules")
    assert Probe.model_validate_json(p.model_dump_json()) == p


# --- observations ---------------------------------------------------------

def test_observation_keeps_raw_and_parsed_state():
    """The parse is where meaning is lost; both sides of it stay in the record."""
    o = Observation(t=4.0, subject="door", bracket_id="b1", present=True,
                    raw_state="the door appears to be standing open",
                    state="open", certainty=0.7)
    assert o.raw_state != o.state and o.ok


def test_failed_observation_is_distinguishable_from_absence():
    """`not present` and `never parsed` must not collapse to the same record."""
    absent = Observation(t=1.0, subject="person", bracket_id="b1", present=False)
    failed = Observation(t=1.0, subject="person", bracket_id="b1", ok=False,
                         note="no JSON in response")
    assert absent.ok and not failed.ok


# --- document -------------------------------------------------------------

def _doc(result_class="model") -> EventsDocument:
    return EventsDocument(
        video={"path": "clip.mp4", "duration_s": 120.0},
        probes=[Probe(id="p1", description="the door opens", expressible=True,
                      subject="door", kind="state", states=("closed", "open"))],
        rejected=[Rejection(probe_id="p2", description="the alarm sounds",
                            reason="audio-only cue")],
        events=[Event(id="e1", probe_id="p1", description="the door opens",
                      start_s=12.0, end_s=15.5, confidence=0.71, rank=1,
                      signals=Signals(agreement=0.9, sharpness=0.8, coverage=1.0,
                                      mean_certainty=0.7),
                      evidence=[12.0, 13.0], bracket_ids=["b1"])],
        brackets=[Bracket(id="b1", start_s=10.0, end_s=18.0, origin="rising",
                          peak_score=7.2)],
        coverage={"failures": 0, "repairs": 0},
        run=RunInfo(model="nvidia/Cosmos3-Edge", backend="vllm", sample_fps=4.0,
                    observe_step_s=1.0, observe_span_s=2.0,
                    result_class=result_class),
    )


def test_document_round_trips_through_json():
    d = _doc()
    assert EventsDocument.model_validate_json(d.model_dump_json()) == d


def test_result_class_travels_with_the_document():
    """An oracle number must never be printable under a model heading."""
    assert _doc("synthetic_oracle").run.result_class == "synthetic_oracle"
    assert _doc().run.model_verified is False  # nothing is verified until asked


def test_bracket_duration():
    assert Bracket(id="b1", start_s=10.0, end_s=18.0, origin="sentinel").duration_s == 8.0


# --- config ---------------------------------------------------------------

def test_defaults_are_coherent():
    cfg_mod.Config().check()


def test_span_shorter_than_sampling_interval_is_refused():
    """Would report no events without ever showing the model a frame."""
    c = cfg_mod.Config()
    c.sampling.fps, c.observe.span_s = 2.0, 0.25
    with pytest.raises(cfg_mod.ConfigError, match="no frames"):
        c.check()


def test_step_larger_than_span_is_refused():
    c = cfg_mod.Config()
    c.observe.step_s, c.observe.span_s = 4.0, 2.0
    with pytest.raises(cfg_mod.ConfigError, match="no poll observes"):
        c.check()


def test_hysteresis_without_width_is_refused():
    c = cfg_mod.Config()
    c.signal.lo_pct = c.signal.hi_pct
    with pytest.raises(cfg_mod.ConfigError, match="hysteresis"):
        c.check()


def test_interp_gap_below_step_is_refused():
    """Otherwise adjacent polls count as too far apart and everything is partial."""
    c = cfg_mod.Config()
    c.derive.max_interp_gap_s, c.observe.step_s = 0.5, 1.0
    with pytest.raises(cfg_mod.ConfigError, match="partial"):
        c.check()


def test_measured_bad_settings_warn_rather_than_fail():
    c = cfg_mod.Config()
    c.observe.shared_caption = True
    c.sampling.seek = False
    c.derive.certainty_weight = 0.5
    w = " ".join(c.warnings())
    c.check()  # still coherent
    assert "85%" in w and "quadratic" in w and "ranked nothing" in w
