"""The live path, driven by a real clip against the fake server. No GPU.

What this pins is the streaming-specific behaviour -- the ring buffer, the
rolling threshold, sentinel cadence, drop policy, and the fact that a streamed
run and a batch run over the same footage use the same derivation.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from eventfinder.backends.vllm import VLLMReasoner  # noqa: E402
from eventfinder.compile import RulesCompiler, compile_all  # noqa: E402
from eventfinder.config import Config  # noqa: E402
from eventfinder.decode import Frame  # noqa: E402
from eventfinder.models import Bracket  # noqa: E402
from eventfinder.stream import (  # noqa: E402
    LiveFinder,
    MotionSource,
    RingBuffer,
    SentinelSource,
    frames_from_file,
)
from fake_vllm import serve  # noqa: E402
from PIL import Image  # noqa: E402

CLIP = Path("data/eval/2018-03-07.16-50-01.16-55-01.admin.G326.r13.mp4")
pytestmark = pytest.mark.skipif(not CLIP.exists(), reason="labelled clip not present")


@pytest.fixture(scope="module")
def server():
    srv, checker = serve(0, "nvidia/Cosmos3-Edge")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", checker
    srv.shutdown()


def frame(t, val=0):
    return Frame(t=t, image=Image.new("RGB", (64, 36), (val, val, val)))


# --- ring buffer ----------------------------------------------------------

def test_buffer_drops_frames_older_than_the_window():
    b = RingBuffer(window_s=5.0)
    for t in range(20):
        b.add(frame(float(t)))
    lo, hi = b.span
    assert hi == 19.0 and lo >= 14.0


def test_slice_returns_one_frame_per_step_without_repeats():
    b = RingBuffer(window_s=30.0)
    for t in range(0, 20):
        b.add(frame(float(t)))
    fr = b.slice(5.0, 9.0, step_s=1.0)
    assert [f.t for f in fr] == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_slice_of_an_empty_buffer_is_empty():
    assert RingBuffer(window_s=5.0).slice(0.0, 4.0, 1.0) == []


# --- motion source --------------------------------------------------------

def test_motion_needs_history_before_it_fires():
    """A threshold learned from four samples is noise."""
    src = MotionSource(Config())
    out = []
    for i in range(10):
        out += src.feed(frame(float(i) / 2, val=i * 20 % 255))
    assert out == []


def test_motion_opens_and_closes_a_bracket_around_a_change():
    src = MotionSource(Config())
    for i in range(60):
        src.feed(frame(i * 0.5, val=10))         # quiet history
    fired = []
    for i in range(60, 66):
        fired += src.feed(frame(i * 0.5, val=200))   # sudden change
    for i in range(66, 90):
        fired += src.feed(frame(i * 0.5, val=200))   # quiet again at the new level
    assert fired, "a large sustained change must produce a bracket"
    b = fired[0]
    assert b.origin == "rising" and b.end_s > b.start_s


def test_motion_closes_a_bracket_that_will_not_stop():
    """A change that never settles still has to be looked at."""
    cfg = Config()
    cfg.signal.max_bracket_s = 4.0
    src = MotionSource(cfg)
    for i in range(60):
        src.feed(frame(i * 0.5, val=10))
    fired = []
    for i in range(60, 120):
        fired += src.feed(frame(i * 0.5, val=(i * 37) % 255))
    assert fired and all(b.duration_s <= cfg.signal.max_bracket_s + 2 * cfg.signal.pad_s
                         for b in fired)


# --- sentinels ------------------------------------------------------------

def test_sentinel_fires_on_cadence():
    cfg = Config()
    cfg.signal.sentinel_every_s = 10.0
    src = SentinelSource(cfg)
    fired = [b for t in range(0, 60) for b in src.feed(frame(float(t)))]
    assert 4 <= len(fired) <= 6
    assert all(b.origin == "sentinel" for b in fired)


def test_covered_time_defers_the_next_sentinel():
    """Every N seconds of UNOBSERVED time, not every N seconds."""
    cfg = Config()
    cfg.signal.sentinel_every_s = 10.0
    src = SentinelSource(cfg)
    src.feed(frame(0.0))
    src.note_covered(30.0)
    assert [b for t in range(1, 35) for b in src.feed(frame(float(t)))] == []


# --- the driver -----------------------------------------------------------

def _finder(url, cfg=None, sink=None):
    cfg = cfg or Config()
    cfg.observe.concurrency = 1
    probes = compile_all(["a person opens a building door"], RulesCompiler())
    r = VLLMReasoner(url, "nvidia/Cosmos3-Edge", media="video", max_tokens=512,
                     sample_fps=1.0 / cfg.observe.step_s)
    return LiveFinder(probes, cfg, r, sink or (lambda e, v: None)), cfg


def test_a_clip_streamed_produces_events_and_bounded_latency(server):
    url, _ = server
    seen = []
    finder, cfg = _finder(url, sink=lambda e, v: seen.append((e, v)))
    for f in frames_from_file(str(CLIP), cfg):
        finder.feed(f)
        if finder.pending:
            finder.tick(f.t)
    finder.close(now=120.0)

    assert finder.stats.frames > 200
    assert finder.stats.candidates > 0
    assert seen, "streaming a clip with events in it must emit something"
    for e, verified in seen:
        assert verified is True
        assert 0.0 <= e.confidence <= 1.0
        assert e.start_s <= e.end_s
    # Emission is bounded by the witnessing window, not by clip length.
    assert finder.stats.mean_latency_s < 30.0


def test_the_model_is_called_only_on_candidates(server):
    """The whole economic argument: cost tracks events, not duration."""
    url, _ = server
    finder, cfg = _finder(url)
    for f in frames_from_file(str(CLIP), cfg):
        finder.feed(f)
        finder.tick(f.t)
    assert finder.stats.calls > 0
    assert finder.stats.calls < finder.stats.frames / 10


def test_overload_drops_the_oldest_candidate_rather_than_falling_behind(server):
    """A live system that queues without bound stops being live, silently."""
    url, _ = server
    finder, _ = _finder(url)
    finder.max_pending = 2
    for i in range(10):
        finder._enqueue(Bracket(id=f"b{i}", start_s=float(i), end_s=float(i) + 1,
                                origin="rising"))
    assert len(finder.pending) == 2
    assert finder.stats.dropped == 8
    assert finder.pending[0].id == "b8"          # the newest survive


def test_a_candidate_older_than_the_buffer_is_dropped_not_guessed(server):
    url, _ = server
    finder, _ = _finder(url)
    for t in range(100, 140):
        finder.buffer.add(frame(float(t)))
    finder._enqueue(Bracket(id="old", start_s=1.0, end_s=3.0, origin="rising"))
    finder.tick(now=140.0)
    assert finder.stats.dropped == 1 and finder.stats.calls == 0


def test_inexpressible_probes_never_reach_the_model(server):
    url, _ = server
    cfg = Config()
    probes = compile_all(["the alarm sounds"], RulesCompiler())
    r = VLLMReasoner(url, "nvidia/Cosmos3-Edge", media="video", max_tokens=512)
    finder = LiveFinder(probes, cfg, r, lambda e, v: None)
    finder._enqueue(Bracket(id="b1", start_s=0.0, end_s=4.0, origin="rising"))
    for t in range(0, 6):
        finder.buffer.add(frame(float(t)))
    finder.tick(now=6.0)
    assert finder.stats.calls == 0
