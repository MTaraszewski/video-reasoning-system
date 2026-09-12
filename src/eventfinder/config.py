"""Configuration, with the provenance of every default written next to it.

Three tiers of default, and the difference matters:

  MEASURED    a value this repo established on real footage. The source run is
              named. Changing it should be a deliberate experiment.
  INHERITED   carried from the design note that this architecture follows.
              Plausible, never tested here. These are the first things to sweep.
  STRUCTURAL  forced by arithmetic or by an API limit, not chosen.

`check()` refuses configurations that describe a system which cannot work. Each
check below corresponds to a way the previous engine produced confident wrong
answers rather than an error.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG = Path(os.getenv("EF_CONFIG", "eventfinder.yaml"))


class ConfigError(ValueError):
    def __init__(self, problem: str, fix: str = ""):
        self.problem, self.fix = problem, fix
        super().__init__(f"{problem}\n  fix: {fix}" if fix else problem)


class ModelConfig(BaseModel):
    # STRUCTURAL. Exact repo id; a single wrong character serves nothing.
    name: str = "nvidia/Cosmos3-Edge"
    base_url: str = "http://vllm:8000/v1"
    api_key: str = "EMPTY"
    # MEASURED. Five repeats at temperature 0 were bit-identical, so sampling is
    # not a source of run-to-run variance here. Restarting the server, or
    # changing the request history within one server session, does flip a
    # borderline poll -- batching and prefix-cache state, not the sampler.
    temperature: float = 0.0
    # MEASURED. The observer emits one small JSON object. The previous engine
    # left this at 4096 and paid 80 seconds for runaway generations that carried
    # no extra information.
    max_tokens: int = 192
    request_timeout_s: float = 120.0
    # STRUCTURAL. A run that never confirmed what it was talking to is not a
    # model result; RunInfo.model_verified records the answer.
    verify_identity: bool = True


class SamplingConfig(BaseModel):
    # MEASURED. 4 fps on 2-second spans is 8 frames per call, which the model
    # accepted without truncation at 640px.
    fps: float = Field(4.0, gt=0)
    frame_max_side: int = Field(640, gt=0)
    # STRUCTURAL. A hard ceiling so a long span cannot silently become a
    # thousand-frame request.
    max_frames_per_call: int = Field(48, gt=0)
    # MEASURED, and the single biggest cost bug carried forward. Decoding by
    # walking from frame zero is quadratic in clip length: 0.26 s to reach t=0
    # and 3.69 s to reach t=118 on the same clip, 235 s per 2-minute clip in
    # total. Seeking to the span keeps it flat. Off only as an escape hatch for
    # containers whose demuxer mis-seeks.
    seek: bool = True


class OverlayConfig(BaseModel):
    """The timestamp burned into each frame.

    MEASURED, and load-bearing: this is what lets the model be asked *what* and
    never *when*. The frame carries its own time, so a returned time is a
    transcription rather than an estimate.
    """

    enabled: bool = True
    format: str = "t={:.3f}s"
    position: str = "bottom-left"
    font_scale: float = Field(0.045, gt=0, le=0.5)


class SignalConfig(BaseModel):
    """The cheap change detector that decides where the model looks at all.

    Every value here is INHERITED from the design note and untested on real
    footage. This is deliberately the first thing to measure, because it is free
    to sweep: brackets can be scored against the labelled events with no model
    call at all. If the labelled events do not fall inside brackets, nothing
    downstream can recover them.
    """

    enabled: bool = True
    # Frames per second at which the change score itself is computed -- far
    # cheaper than the observer's sampling, since it only needs pixel deltas.
    fps: float = Field(2.0, gt=0)
    # Long edge of the greyscale thumbnail the delta is computed on.
    thumb_px: int = Field(128, gt=0)
    # Hysteresis, as percentiles of the clip's OWN change distribution. Open a
    # bracket above hi_pct, close it below lo_pct.
    #
    # MEASURED, and a correction to the inherited design, which gated on robust
    # z-units. On this footage the median change is exactly 0.0 on six of eight
    # clips, so the MAD collapses, the z-scale falls back to a standard
    # deviation the event itself inflates, and `max z` ranges from 6.1 on one
    # clip to 359.9 on another -- not comparable, which was the whole point of
    # standardising. At hi_z=6.0 the detector contributed nothing: bracket
    # recall was 5/15 and every hit came from a sentinel.
    #
    # A percentile gate needs no scale. The labelled events sit at a median
    # percentile rank of 0.97 within their clips, and 11 of 15 are above p95 --
    # so the ranking was always good enough and only the threshold was wrong.
    hi_pct: float = Field(97.0, gt=0, lt=100)
    lo_pct: float = Field(90.0, gt=0, lt=100)
    # Floor on the raw moving-pixel fraction, so a still clip cannot manufacture
    # brackets out of its own sensor noise -- a percentile gate always fires on
    # some fraction of any clip, including an empty one. MEASURED: the labelled
    # events run from 0.0085 to 0.11, so the inherited 0.04 rejected half of
    # them on its own.
    min_pixel_delta: float = Field(0.002, ge=0)
    # Padding around a detected change, so the observer sees both sides of it.
    pad_s: float = Field(4.0, ge=0)
    # A look taken on a fixed cadence regardless of change. This is the guard
    # against the signal's blind spots: the previous engine's trigger scored
    # tIoU 0.000 against 0.406 for uniform polling on the same clip, entirely
    # because it declined to sample a time it judged uninteresting. Sentinels
    # bound that failure instead of hoping it does not happen.
    sentinel_every_s: float = Field(45.0, gt=0)
    # How WIDE a sentinel look is. A sentinel is a glance, not a span: derived
    # from `sentinel_every_s` it was 22.5 s wide, which alone put mean coverage
    # at 66% of the clip and made the change detector nearly pointless.
    sentinel_width_s: float = Field(8.0, gt=0)
    # Merge brackets closer than this, rather than paying twice to look at the
    # same moment from two sides.
    merge_gap_s: float = Field(1.0, ge=0)
    max_bracket_s: float = Field(30.0, gt=0)


class ObserveConfig(BaseModel):
    """How densely the model looks inside a bracket."""

    # MEASURED. This IS the boundary resolution -- a transition is located no
    # more precisely than the interval between polls, which is why the evaluator
    # scores with a tolerance of one step.
    step_s: float = Field(1.0, gt=0)
    # MEASURED. Seconds of frames shown per poll. Sweeping this to 1.0 did not
    # move tIoU at all, against a prediction that it would worsen it.
    span_s: float = Field(2.0, gt=0)
    # MEASURED. The observer ran strictly serially: vLLM reported one running
    # request and 1.8% KV cache use throughout an 8,871-second evaluation, so
    # two to three times the throughput was available and unused.
    concurrency: int = Field(3, gt=0)
    # MEASURED. One caption per timestep covering every subject, instead of one
    # sweep per subject. Tried and it FAILED: door parse rate fell 85% -> 29%,
    # `person` parsed 0 of 119 polls, per-call latency rose to 10.2 s, and the
    # model conflated `door` with `car door` into a 92-second false positive.
    # Retained as a switch because the saving is real (9 descriptions collapse
    # to 4 subjects, and sharing collapses those to 1) and a better prompt may
    # recover it -- but it is not the default, and the failure is why.
    shared_caption: bool = False


class DeriveConfig(BaseModel):
    """Turning observations into timed events, in code.

    The model never holds state and never decides that an event occurred. That
    is the whole point of the architecture, and it is why these knobs exist in
    code where they can be swept without a GPU.
    """

    # INHERITED. Consecutive agreeing observations required before a run counts.
    # min_run=1 admits any single mis-parse as an event.
    min_run: int = Field(2, gt=0)
    # INHERITED. A transition must hold for this long to be believed.
    transition_hold_s: float = Field(1.0, ge=0)
    # INHERITED. Extend an event's end while the subject is still moving.
    extend_by_motion_s: float = Field(4.0, ge=0)
    # MEASURED, and the fix for the worst failure the previous engine produced.
    # A transition bracketed by two polls further apart than this is NOT
    # interpolated to their midpoint; the event is emitted at the observed edge
    # and marked partial. Interpolating across an 82-second gap produced a
    # 42-second span for a 3-second door opening, at confidence 1.0.
    max_interp_gap_s: float = Field(4.0, gt=0)
    # MEASURED. Weight of the model's self-reported certainty in the confidence
    # product. Zero by default: over one labelled run the stated certainty took
    # three distinct values across 48 predictions, so it could not rank them.
    # It is still recorded in Signals.mean_certainty; this knob makes folding it
    # in an experiment rather than an assumption.
    certainty_weight: float = Field(0.0, ge=0.0, le=1.0)
    # MEASURED. Sweeping the emitted confidence on the labelled set cut 153
    # predictions to 35 at >=0.8 with no loss of recall, R@1 or mean tIoU
    # (0.292 -> 0.288); at >=0.9 it broke (0.130). Default 0.0 so the evaluator
    # sees the full distribution and the sweep stays reproducible.
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)


class LimitsConfig(BaseModel):
    max_video_bytes: int = 2 * 1024**3
    max_duration_s: float = 1800.0
    min_duration_s: float = 0.5
    max_queries: int = 20
    max_query_chars: int = 300
    # STRUCTURAL. A per-request ceiling, not a per-run one. The previous engine
    # spent 1,904 calls on a four-clip evaluation without ever checking this on
    # the path that was actually running.
    max_model_calls: int = 1000


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    observe: ObserveConfig = Field(default_factory=ObserveConfig)
    derive: DeriveConfig = Field(default_factory=DeriveConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    def check(self) -> None:
        s, o, sm = self.signal, self.observe, self.sampling
        interval = 1.0 / sm.fps

        # Every poll would contain zero frames, every poll would be skipped, and
        # the run would report no events -- which reads as "the model found
        # nothing" and is actually "we never showed it anything".
        if o.span_s < interval:
            raise ConfigError(
                f"observe.span_s ({o.span_s}s) is shorter than one sampling interval "
                f"({interval:.3f}s at {sm.fps} fps), so every poll would contain no "
                "frames and the run would report no events without looking at the video.",
                f"raise observe.span_s above {interval:.3f}s, or raise sampling.fps",
            )
        if o.step_s > o.span_s:
            raise ConfigError(
                f"observe.step_s ({o.step_s}s) exceeds observe.span_s ({o.span_s}s), "
                f"leaving {o.step_s - o.span_s:.2f}s between polls that no poll observes.",
                "set observe.step_s <= observe.span_s",
            )
        if s.lo_pct >= s.hi_pct:
            raise ConfigError(
                f"signal.lo_pct ({s.lo_pct}) is not below signal.hi_pct ({s.hi_pct}), so "
                "the hysteresis has no width and a bracket closes on the sample it opens.",
                "set signal.lo_pct strictly below signal.hi_pct",
            )
        if s.max_bracket_s < o.span_s:
            raise ConfigError(
                f"signal.max_bracket_s ({s.max_bracket_s}s) is shorter than one observe "
                f"span ({o.span_s}s), so a full-width bracket cannot hold a single poll.",
                "raise signal.max_bracket_s, or lower observe.span_s",
            )
        if self.derive.max_interp_gap_s < o.step_s:
            raise ConfigError(
                f"derive.max_interp_gap_s ({self.derive.max_interp_gap_s}s) is below "
                f"observe.step_s ({o.step_s}s), so adjacent polls count as too far apart "
                "and every event would be marked partial.",
                f"raise derive.max_interp_gap_s to at least observe.step_s ({o.step_s}s)",
            )

    def warnings(self) -> list[str]:
        """Coherent, but measured to behave badly. Said out loud, not enforced."""
        out: list[str] = []
        if self.observe.shared_caption:
            out.append(
                "observe.shared_caption is on. Measured: door parse rate fell 85% -> 29%, "
                "`person` parsed 0 of 119 polls, and the model conflated two subjects into "
                "a 92-second false positive."
            )
        if not self.signal.enabled:
            out.append(
                "signal.enabled is off: every step in the clip becomes a poll. Correct, and "
                "the cost is linear in clip length."
            )
        if self.derive.certainty_weight > 0:
            out.append(
                f"derive.certainty_weight is {self.derive.certainty_weight}. Self-reported "
                "certainty took three distinct values over 48 predictions and ranked nothing."
            )
        if not self.sampling.seek:
            out.append(
                "sampling.seek is off: decode cost becomes quadratic in clip length "
                "(0.26 s at t=0 vs 3.69 s at t=118 on the same clip)."
            )
        return out


def load(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    cfg = Config.model_validate(raw)
    cfg.check()
    return cfg
