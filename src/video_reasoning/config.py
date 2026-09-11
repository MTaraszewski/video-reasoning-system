"""Configuration loading and coherence checks.

Values live in config.yaml with their rationale. This module loads them, applies
overrides, and refuses configurations that cannot work — the checks that would
otherwise fail silently and produce plausible-looking wrong answers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .errors import Misconfigured

DEFAULT_CONFIG = Path(os.getenv("VRS_CONFIG", "config.yaml"))


class ModelConfig(BaseModel):
    name: str = "nvidia/Cosmos3-Edge"
    base_url: str = "http://vllm:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout_s: float = 300.0


class SamplingConfig(BaseModel):
    fps: float = Field(4.0, gt=0)
    frame_max_side: int = Field(640, gt=0)
    max_frames_per_window: int = Field(48, gt=0)


class WindowingConfig(BaseModel):
    window_s: float = Field(12.0, gt=0)
    stride_s: float = Field(9.0, gt=0)

    @property
    def overlap_s(self) -> float:
        return max(0.0, self.window_s - self.stride_s)


class OverlayConfig(BaseModel):
    enabled: bool = True
    format: str = "t={:.3f}s"
    position: str = "bottom-left"
    font_scale: float = Field(0.045, gt=0, le=0.5)


class StatesConfig(BaseModel):
    """The `states` strategy: caption, parse, derive.

    Approach 1 asks the model when an event happened and cannot establish whether
    it happened at all. This asks only what the scene is, repeatedly, and does the
    temporal reasoning in code. Kept behind a switch so both remain runnable and
    a back-to-back comparison is one flag apart.
    """

    # Seconds between polls. This IS the boundary resolution: a transition can be
    # located no more precisely than the interval, which is why the eval scores
    # with a tolerance of one step.
    step_s: float = Field(1.0, gt=0)
    # Seconds of frames shown per poll.
    span_s: float = Field(2.0, gt=0)

    # Spend calls where something changed rather than on a fixed grid. Measured on
    # admin.G326: every poll agreed with its neighbours except at the transition,
    # so the uniform grid spent most of its budget confirming stillness. The
    # trigger's blind spots are real and documented in motion.py -- this is a
    # measurable trade, not a free win.
    trigger: bool = False
    trigger_top_k: int = Field(12, gt=0)

    # One caption per timestep covering every subject, instead of one sweep per
    # subject. The frames are identical whichever subject is asked about, so the
    # per-subject sweeps pay repeatedly for the same perception: on the labelled
    # set, 9 expressible descriptions reduce to 4 subjects, and sharing collapses
    # those to 1 -- 8,568 calls to 952 at identical polling density.
    #
    # OFF by default because it trades a measured risk for that saving: a single
    # caption covering four subjects may mention none of them clearly, and
    # `split_by_subject` reports a missing line as no answer rather than guessing.
    # Whether parse rates hold is measurable, and until measured the cheaper path
    # is not the default one.
    shared_caption: bool = False

    # How far a state may be carried across unobserved time, in steps. Uniform
    # polling never exercises this -- consecutive polls are one step apart. The
    # trigger leaves gaps of a minute or more between polls, and interpolating a
    # transition into the middle of one produced a 42-second event for a
    # 3-second door on admin.G326. Past this, `states.edge` reports observed
    # evidence and marks the span partial instead of guessing.
    carry_steps: float = Field(3.0, gt=0)
    trigger_min_gap_s: float = Field(2.0, ge=0)
    trigger_fps: float = Field(2.0, gt=0)


class DetectConfig(BaseModel):
    """Two-stage extraction: ask IF, then ask WHEN.

    The single-stage prompt asks the model to find moments matching a
    description, which presupposes the description applies. Measured on real
    footage, 7 of 81 emitted events carried evidence that denied the event --
    "Empty hallway with a closed door and no visible people", stated with the
    model's own confidence of 1.0. Nothing in the pipeline invented those; the
    model produced them under a prompt that made agreeing easier than declining.

    Splitting the question fixes what we can measure, not just what we ask.
    Detection and localisation become separately scoreable, so "it never said
    present" and "it said present in the wrong place" stop being the same number.
    """

    enabled: bool = False
    # From the yes/no token logprobs, not from a number the model states. 28 of
    # 81 predictions came back at exactly 1.0 -- a stated confidence carries no
    # information, so it cannot rank anything.
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    # A constrained yes/no needs very few tokens, and the cap forecloses the
    # runaway generations that cost 80s per call in the eval. Not 1: the model
    # may emit a leading token before the answer, which at max_tokens=1 truncates
    # to a preamble ("Got") and parses as nothing.
    max_tokens: int = Field(4, gt=0)


class MergeConfig(BaseModel):
    iou: float = Field(0.3, ge=0.0, le=1.0)
    gap_s: float = Field(0.5, ge=0.0)
    max_span_s: float = Field(60.0, gt=0)


class LimitsConfig(BaseModel):
    max_video_bytes: int = 2 * 1024**3
    max_duration_s: float = 1800.0
    min_duration_s: float = 0.5
    max_queries: int = 20
    max_query_chars: int = 300
    max_model_calls: int = 1000


class Config(BaseModel):
    # Which engine find_events runs. `windows` is Approach 1 -- ask the model when
    # the event happened, merge across overlapping windows. `states` is Approach 3
    # -- caption, parse the state, derive the event from transitions. Both stay
    # available so a comparison is one flag apart rather than a branch apart.
    strategy: str = Field("windows", pattern="^(windows|states)$")
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    states: StatesConfig = Field(default_factory=StatesConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    def check_coherence(self) -> None:
        """Reject configurations that describe a system which cannot work.

        These are not style preferences. Each one, left unchecked, produces
        results that look reasonable and are wrong.
        """
        w = self.windowing
        if w.stride_s > w.window_s:
            raise Misconfigured(
                f"stride_s ({w.stride_s}s) is greater than window_s ({w.window_s}s), "
                f"which leaves {w.stride_s - w.window_s:.1f}s gaps that no window "
                "ever observes.",
                fix="set stride_s <= window_s; stride_s < window_s gives overlap, "
                    "which is what recovers events straddling a boundary",
            )

        # A window shorter than one sampling interval can contain no frames.
        interval = 1.0 / self.sampling.fps
        if w.window_s < interval:
            raise Misconfigured(
                f"window_s ({w.window_s}s) is shorter than one sampling interval "
                f"({interval:.3f}s at {self.sampling.fps} fps), so some windows "
                "would contain no frames at all.",
                fix=f"raise window_s above {interval:.3f}s, or raise sampling.fps",
            )

        # The states strategy has its own arithmetic and none of the checks above
        # touch it. Left unchecked, a span shorter than one sampling interval
        # yields zero frames on every poll, every poll is skipped, and the run
        # reports no events -- which reads as "the model found nothing" and is
        # actually "we never showed it anything".
        if self.strategy == "states":
            st = self.states
            if st.span_s < interval:
                raise Misconfigured(
                    f"states.span_s ({st.span_s}s) is shorter than one sampling "
                    f"interval ({interval:.3f}s at {self.sampling.fps} fps), so "
                    "every poll would contain no frames and the run would report "
                    "no events without ever looking at the video.",
                    fix=f"raise states.span_s above {interval:.3f}s, or raise "
                        "sampling.fps",
                )
            if st.step_s > st.span_s:
                raise Misconfigured(
                    f"states.step_s ({st.step_s}s) is greater than states.span_s "
                    f"({st.span_s}s), which leaves {st.step_s - st.span_s:.1f}s "
                    "between polls that no poll ever observes.",
                    fix="set step_s <= span_s so consecutive polls at least meet",
                )

        if self.merge.max_span_s < w.window_s:
            raise Misconfigured(
                f"merge.max_span_s ({self.merge.max_span_s}s) is smaller than "
                f"window_s ({w.window_s}s), so a single window's own event could "
                "exceed the merge limit.",
                fix="set merge.max_span_s >= window_s",
            )

    def warnings(self) -> list[str]:
        """Legal but probably-unintended settings. Surfaced, not enforced."""
        out: list[str] = []
        w = self.windowing
        if w.overlap_s == 0:
            out.append(
                f"stride_s == window_s ({w.window_s}s): zero overlap. An event "
                "straddling a boundary will be seen only in fragments by each "
                "window, and the merge cannot recover it."
            )
        if self.sampling.fps < 2:
            out.append(
                f"sampling.fps is {self.sampling.fps}: boundary precision cannot "
                f"beat {1.0 / self.sampling.fps:.2f}s before any model error."
            )
        if self.strategy == "states":
            st = self.states
            # No warning for span_s > step_s, which is the DEFAULT. The overlap
            # ambiguity it creates is real and documented in DESIGN.md, but a
            # warning that fires on every single run of the shipped configuration
            # teaches people to ignore warnings, which costs more than it saves.
            if st.trigger:
                out.append(
                    "states.trigger is on. Measured on admin.G326 it used 12 calls "
                    "against 119 and LOST the event (tIoU 0.000 against 0.406), "
                    "because trigger_min_gap_s forbids two polls closer than "
                    f"{st.trigger_min_gap_s}s even at the strongest peak."
                )
            if st.shared_caption:
                out.append(
                    "states.shared_caption is on. Measured on admin.G326 it cut "
                    "the door parse rate from 85% to 29%, never parsed 'person' in "
                    "119 calls, and ran 2.6x slower per call."
                )

        if not self.overlay.enabled:
            out.append(
                "overlay.enabled is false: frames carry no timestamp, so a model "
                "that localises by reading them has nothing to read."
            )
        return out


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load config.yaml, apply dotted overrides, then check coherence.

    Overrides use dotted keys so CLI flags map onto nested settings:
        load_config(**{"sampling.fps": 8})
    """
    path = Path(path) if path else DEFAULT_CONFIG
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}

    for dotted, value in overrides.items():
        if value is None:
            continue
        section, _, key = dotted.partition(".")
        if not key:
            data[dotted] = value
        else:
            data.setdefault(section, {})[key] = value

    cfg = Config(**data)
    cfg.check_coherence()
    return cfg
