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
    # One token is all a yes/no needs, and it forecloses the runaway generations
    # that cost 80s per call in the eval.
    max_tokens: int = Field(1, gt=0)


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
    max_model_calls: int = 500


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
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
