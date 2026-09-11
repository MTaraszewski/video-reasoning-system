"""Model backends, behind one seam.

    stub    deterministic, no GPU, no network. Proves the pipeline. Never evidence.
    vllm    the real model, served by stock vLLM.
    replay  real recorded responses, replayed with no GPU.

`replay` is why one GPU session goes a long way: record real exchanges once, then
do all parser, prompt and merge work on a laptop against genuine model output.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from .base import Backend, ExtractRequest, ExtractResult
from .stub import StubBackend

__all__ = ["Backend", "ExtractRequest", "ExtractResult", "StubBackend", "make_backend"]


def make_backend(cfg: Config, kind: str = "auto", *,
                 record_dir: str | Path | None = None,
                 replay_dir: str | Path | None = None) -> Backend:
    """Choose a backend. `auto` picks vllm unless a stub is explicitly allowed."""
    import os

    if kind == "replay" or replay_dir:
        from .replay import ReplayBackend
        return ReplayBackend(replay_dir or "fixtures/responses")
    if kind == "stub":
        return StubBackend()
    if kind == "auto" and os.getenv("ALLOW_NO_GPU"):
        return StubBackend()

    from .vllm import VLLMBackend
    return VLLMBackend(
        model=cfg.model.name, base_url=cfg.model.base_url,
        api_key=cfg.model.api_key, temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens, timeout_s=cfg.model.request_timeout_s,
        record_dir=record_dir,
        detect_enabled=cfg.detect.enabled,
        detect_threshold=cfg.detect.threshold,
        detect_max_tokens=cfg.detect.max_tokens,
    )
