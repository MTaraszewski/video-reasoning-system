"""The whole pipeline, cold: real decode, real signal, real HTTP, no GPU.

This exercises the client code path that will run against the box -- the same
request builder, the same parser, the same derivation -- against a server that
checks the request shape and returns a canned reply. What it proves is that the
plumbing is right. What it cannot prove is anything about the model, and the
canned replies are never scored as results.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from eventfinder.backends.replay import ReplayReasoner, summarise, write_exchanges  # noqa: E402
from eventfinder.backends.vllm import VLLMReasoner  # noqa: E402
from eventfinder.config import Config  # noqa: E402
from eventfinder.models import EventsDocument  # noqa: E402
from eventfinder.pipeline import plan, run  # noqa: E402
from fake_vllm import serve  # noqa: E402

CLIP = Path("data/eval/2018-03-07.16-50-01.16-55-01.admin.G326.r13.mp4")
DESCS = ["a person opens a building door", "the alarm sounds"]
pytestmark = pytest.mark.skipif(not CLIP.exists(), reason="labelled clip not present")


@pytest.fixture(scope="module")
def server():
    srv, checker = serve(0, "nvidia/Cosmos3-Edge")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", checker
    srv.shutdown()


def cfg_for(mode: str) -> Config:
    c = Config()
    c.observe.mode = mode
    c.observe.concurrency = 2
    # Two brackets' worth of work is enough to exercise every path.
    c.signal.sentinel_every_s = 60.0
    return c


def reasoner_for(url: str, media: str) -> VLLMReasoner:
    return VLLMReasoner(url, "nvidia/Cosmos3-Edge", media=media, max_tokens=512)


# --- planning -------------------------------------------------------------

def test_plan_costs_less_per_bracket_than_per_step():
    """The 8x question, visible before a single call is made."""
    _, bs, step_tasks, _ = plan(str(CLIP), DESCS, cfg_for("per_step"))
    _, _, bracket_tasks, _ = plan(str(CLIP), DESCS, cfg_for("per_bracket"))
    assert len(bracket_tasks) == len(bs)
    assert len(step_tasks) > 4 * len(bracket_tasks)


def test_inexpressible_descriptions_cost_nothing():
    _, _, tasks, _ = plan(str(CLIP), DESCS, cfg_for("per_bracket"))
    assert all("p2" not in t.probe_ids for t in tasks)


# --- the request shape ----------------------------------------------------

@pytest.mark.parametrize("media", ["video", "frames"])
def test_both_media_paths_are_accepted_by_the_checker(server, media):
    """The unresolved CONFLICT: this repo's earlier engine sent frame lists and
    produced results; the design note says only the video path preserves
    temporal merging. Both are built, so one GPU session settles it."""
    url, checker = server
    before = len(checker.problems)
    doc = run(str(CLIP), DESCS[:1], cfg_for("per_bracket"), reasoner_for(url, media))
    assert checker.problems[before:] == []
    assert doc.run.calls > 0 and doc.run.model_verified


def test_uncapped_max_tokens_is_rejected(server):
    """80-second generations under max_tokens=4096 are what this guards."""
    url, _ = server
    r = VLLMReasoner(url, "nvidia/Cosmos3-Edge", media="video", max_tokens=4096)
    doc = run(str(CLIP), DESCS[:1], cfg_for("per_bracket"), r)
    assert doc.run.calls == 0 and r.usage.failures > 0
    assert doc.events == []          # and it degrades to no events, not to a crash


def test_a_wrong_model_id_is_not_silently_accepted(server):
    url, _ = server
    r = VLLMReasoner(url, "nvidia/Cosmos3-Nano", media="video", max_tokens=512)
    ok, why = r.verify_model()
    assert not ok and "Cosmos3-Edge" in why


# --- end to end -----------------------------------------------------------

@pytest.mark.parametrize("mode", ["per_step", "per_bracket"])
def test_pipeline_produces_a_valid_document(server, mode):
    url, _ = server
    doc = run(str(CLIP), DESCS, cfg_for(mode), reasoner_for(url, "video"))
    EventsDocument.model_validate_json(doc.model_dump_json())   # the contract holds
    assert doc.rejected and "audio" in doc.rejected[0].reason
    assert doc.coverage["observations_returned"] > 0
    assert doc.run.result_class == "model"
    for e in doc.events:
        assert 0.0 <= e.confidence <= 1.0
        assert e.start_s <= e.end_s <= doc.video["duration_s"] + 1e-6
        assert e.rank >= 1


def test_events_are_derived_from_the_canned_state_flip(server):
    """The fake flips closed -> open at each bracket's midpoint, so a correct
    pipeline finds one transition per bracket that is long enough to hold
    `min_run` observations on each side -- and none from the ones that are not.
    A bracket too short to witness a transition must produce nothing rather than
    something weak."""
    url, _ = server
    cfg = cfg_for("per_bracket")
    doc = run(str(CLIP), DESCS[:1], cfg, reasoner_for(url, "video"))
    long_enough = [b for b in doc.brackets
                   if b.duration_s >= 2 * cfg.derive.min_run * cfg.observe.step_s]
    assert 0 < len(doc.events) <= len(long_enough)
    assert all(e.partial == "none" for e in doc.events)


def test_shared_bracket_boundaries_do_not_manufacture_a_transition(server):
    """Adjacent brackets observe their shared second twice. Two readings of one
    instant must not reach the derivation as a flip and back."""
    url, _ = server
    doc = run(str(CLIP), DESCS[:1], cfg_for("per_bracket"), reasoner_for(url, "video"))
    assert doc.coverage["duplicate_times"] > 0, "expected shared boundaries in this clip"
    for e in doc.events:
        assert len(set(e.evidence)) == len(e.evidence)


def test_verify_mode_records_the_verdict_without_acting_on_it(server):
    """`matches` is recorded beside the derived answer, never gating it."""
    url, _ = server
    doc = run(str(CLIP), DESCS[:1], cfg_for("per_bracket"), reasoner_for(url, "video"),
              verify=True)
    v = doc.coverage["verdicts"]
    assert v and all(x["matches"] is True for x in v)
    # The fake says matches=true on every call, including brackets too short to
    # witness a transition. Derivation still declines those -- which is the
    # point: the model's verdict is recorded, and code decides.
    assert 0 < len(doc.events) < len(v)


# --- recording and replay -------------------------------------------------

def test_a_session_records_a_replay_corpus(server, tmp_path):
    """One GPU session becomes reusable: the same document must come back from
    replay, with the result class changed so it can never pass as a model run."""
    url, _ = server
    r = reasoner_for(url, "video")
    live = run(str(CLIP), DESCS[:1], cfg_for("per_bracket"), r)
    path = write_exchanges(tmp_path / "exchanges.jsonl", r.usage)
    assert sum(1 for _ in open(path)) == r.usage.calls

    rp = ReplayReasoner(path)
    again = run(str(CLIP), DESCS[:1], cfg_for("per_bracket"), rp)
    assert [(e.start_s, e.end_s) for e in again.events] == \
           [(e.start_s, e.end_s) for e in live.events]
    assert again.run.result_class == "replay"

    s = summarise(path)
    assert "observe/video" in s and s["observe/video"]["calls"] == r.usage.calls
    assert s["observe/video"]["truncated"] == 0


def test_a_stale_replay_corpus_is_refused(tmp_path):
    """A corpus recorded under a different prompt answers a different question."""
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps({"bracket_id": "b1", "subject": "door", "times": [1.0],
                             "mode": "observe", "reply": "{}", "prompt_version": "obs-v0"}) + "\n")
    with pytest.raises(ValueError, match="prompt version"):
        ReplayReasoner(p)
    ReplayReasoner(p, strict_prompt=False)      # opt in, and own the caveat
