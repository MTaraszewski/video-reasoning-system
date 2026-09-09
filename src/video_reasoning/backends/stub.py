"""Deterministic backend. No GPU, no network.

It exists to prove the pipeline runs — decode, window, extract, merge, schema —
on any machine. It is not a simulation of the model and its output means nothing
about the model's ability.

Two guards keep that honest: every result is stamped `is_stub`, and the evaluation
harness refuses to compute a metric from stub-generated results. Convenience for
development must not be able to masquerade as evidence.
"""
from __future__ import annotations

import hashlib

from ..schema import WindowEvent
from .base import ExtractRequest, ExtractResult


class StubBackend:
    name = "stub"
    is_stub = True

    def __init__(self, seed: str = "vrs") -> None:
        self.seed = seed

    def _hash(self, req: ExtractRequest) -> int:
        key = f"{self.seed}:{req.window.index}:{req.query}"
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)

    def extract(self, req: ExtractRequest) -> ExtractResult:
        h = self._hash(req)
        w = req.window

        # Deterministically silent for roughly half of (window, query) pairs, so
        # the merge sees a realistic mix of hits, gaps and duplicates rather than
        # a uniform stream.
        if h % 100 < 45:
            return ExtractResult(events=[], raw='{"events": []}', model=self.name)

        span = w.duration_s
        start = w.start_s + span * ((h >> 8) % 40) / 100.0
        end = min(w.end_s, start + span * (0.15 + ((h >> 16) % 35) / 100.0))
        conf = 0.4 + ((h >> 24) % 55) / 100.0

        ev = WindowEvent(
            start_s=round(start, 3), end_s=round(end, 3),
            confidence=round(conf, 3),
            evidence=f"[STUB] deterministic pattern for {req.query!r}",
        )
        return ExtractResult(
            events=[ev],
            raw=f'{{"events": [{{"start_s": {ev.start_s}, "end_s": {ev.end_s}}}]}}',
            model=self.name,
        )

    def describe(self) -> dict:
        return {"backend": self.name, "stub": True, "seed": self.seed}
