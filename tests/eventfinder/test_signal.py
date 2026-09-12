"""Phase 2: bracketing behaves as specified, on signals we construct.

Unit tests over synthetic change signals -- no video, no model. The companion
measurement on real footage is `scripts/score_brackets.py`, which is where the
thresholds came from; these pin the mechanics those thresholds rely on.
"""
from __future__ import annotations

import pytest

from eventfinder.config import Config, SignalConfig
from eventfinder.signal import (
    ChangePoint,
    _hysteresis,
    _merge,
    _rank,
    _sentinels,
    brackets,
    coverage_of,
)
from eventfinder.models import Bracket


def sig(values, step=0.5):
    """A change signal from raw moved-fractions, ranked as the real one is."""
    return _rank([ChangePoint(t=round(i * step, 3), moved=v) for i, v in enumerate(values)])


def cfg(**kw) -> SignalConfig:
    c = SignalConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# --- ranking --------------------------------------------------------------

def test_rank_survives_a_mostly_zero_signal():
    """The failure that made the inherited z-gate useless: on six of eight real
    clips the median change is exactly 0.0, the MAD collapses, and the scale
    becomes a standard deviation the event itself inflates."""
    pts = sig([0.0] * 20 + [0.5])
    assert pts[-1].rank > 95.0
    assert all(p.rank == 0.0 for p in pts[:20])


def test_rank_is_scale_free_across_clips():
    """A busy clip and a still clip must gate the same way."""
    still = sig([0.0] * 20 + [0.02])
    busy = sig([0.30] * 20 + [0.90])
    assert still[-1].rank == busy[-1].rank


# --- hysteresis -----------------------------------------------------------

def test_a_bracket_is_a_region_not_an_instant():
    """The previous engine reduced a change to its single peak, then a minimum
    gap rule rejected the sample the event was actually in -- tIoU 0.000 against
    0.406 for uniform polling on the same clip."""
    pts = sig([0.0] * 10 + [0.5, 0.4, 0.35, 0.3] + [0.0] * 10)
    spans = _hysteresis(pts, cfg(hi_pct=90.0, lo_pct=70.0, min_pixel_delta=0.0))
    assert len(spans) == 1
    s, e, _ = spans[0]
    assert e > s, "a sustained change must produce a span, not a point"


def test_hysteresis_does_not_fragment_a_decaying_change():
    """Single-threshold gating chops one slow event into several."""
    pts = sig([0.0] * 5 + [0.9, 0.5, 0.85, 0.45, 0.8] + [0.0] * 5)
    hyst = _hysteresis(pts, cfg(hi_pct=85.0, lo_pct=50.0, min_pixel_delta=0.0))
    single = _hysteresis(pts, cfg(hi_pct=85.0, lo_pct=85.0, min_pixel_delta=0.0))
    assert len(hyst) == 1 and len(single) > 1


def test_min_pixel_delta_floors_a_dead_clip():
    """A percentile gate is a ranking, not a detector: it fires on some fraction
    of any clip, including one where nothing happens."""
    # Tiny sensor-noise wobble with one clear maximum: the gate fires on rank
    # alone, and only the absolute floor can refuse it.
    pts = sig([0.0001] * 39 + [0.0009])
    assert _hysteresis(pts, cfg(hi_pct=95.0, lo_pct=90.0, min_pixel_delta=0.002)) == []
    assert _hysteresis(pts, cfg(hi_pct=95.0, lo_pct=90.0, min_pixel_delta=0.0)) != []


def test_unclosed_bracket_is_closed_at_the_last_sample():
    pts = sig([0.0] * 10 + [0.5] * 5)
    spans = _hysteresis(pts, cfg(hi_pct=60.0, lo_pct=50.0, min_pixel_delta=0.0))
    assert spans and spans[-1][1] == pts[-1].t


# --- merging and splitting ------------------------------------------------

def test_near_spans_merge_and_keep_the_higher_peak():
    assert _merge([(1.0, 2.0, 5.0), (2.5, 3.0, 9.0)], gap=1.0) == [(1.0, 3.0, 9.0)]


def test_distant_spans_stay_separate():
    assert len(_merge([(1.0, 2.0, 5.0), (20.0, 21.0, 9.0)], gap=1.0)) == 2


# --- sentinels ------------------------------------------------------------

def test_sentinels_fill_only_unclaimed_time():
    """Not every N seconds -- every N seconds of UNOBSERVED time. A clip the
    detector already covers pays nothing extra for it."""
    covered = [Bracket(id="b1", start_s=0.0, end_s=100.0, origin="rising")]
    assert _sentinels(covered, 100.0, cfg()) == []


def test_sentinels_cover_a_clip_the_detector_found_nothing_in():
    s = _sentinels([], 120.0, cfg(sentinel_every_s=45.0, sentinel_width_s=8.0))
    assert len(s) >= 2 and all(b.origin == "sentinel" for b in s)
    assert all(0.0 <= b.start_s < b.end_s <= 120.0 for b in s)


def test_a_short_gap_still_gets_one_look():
    """Otherwise a gap shorter than one cadence slot is never observed at all."""
    assert len(_sentinels([], 10.0, cfg(sentinel_every_s=45.0, sentinel_width_s=8.0))) == 1


def test_sentinel_width_is_a_glance_not_a_span():
    """Deriving width from the cadence made each sentinel 22.5 s wide and put
    mean coverage at 66% of the clip, which left the detector doing nothing."""
    wide = _sentinels([], 120.0, cfg(sentinel_width_s=22.5))
    narrow = _sentinels([], 120.0, cfg(sentinel_width_s=8.0))
    assert coverage_of(wide, 120.0) > 2 * coverage_of(narrow, 120.0)


# --- whole-clip behaviour -------------------------------------------------

def test_disabled_signal_is_the_whole_clip():
    """The linear-cost baseline every saving is measured against."""
    bs = brackets("unused.mp4", 120.0, cfg(enabled=False))
    assert len(bs) == 1 and bs[0].origin == "manual"
    assert coverage_of(bs, 120.0) == 1.0


def test_coverage_ignores_overlap():
    bs = [Bracket(id="b1", start_s=0.0, end_s=10.0, origin="rising"),
          Bracket(id="b2", start_s=5.0, end_s=15.0, origin="rising")]
    assert coverage_of(bs, 100.0) == 0.15


def test_coverage_of_nothing_is_zero():
    assert coverage_of([], 100.0) == 0.0


# --- config ---------------------------------------------------------------

def test_percentile_hysteresis_needs_width():
    c = Config()
    c.signal.lo_pct = c.signal.hi_pct
    with pytest.raises(Exception, match="hysteresis"):
        c.check()


def test_defaults_are_the_swept_operating_point():
    """97/90, 45 s cadence, 8 s glance, 4 s pad: 93% bracket recall at 41%
    coverage on the labelled set. Changing these should be an experiment."""
    s = SignalConfig()
    assert (s.hi_pct, s.lo_pct) == (97.0, 90.0)
    assert (s.sentinel_every_s, s.sentinel_width_s, s.pad_s) == (45.0, 8.0, 4.0)
