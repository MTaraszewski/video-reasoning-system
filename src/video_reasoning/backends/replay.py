"""Replay recorded model responses. No GPU, no network, real model output.

The point: a GPU costs money per hour, and iterating on a JSON parser does not
need one. Record real exchanges during a GPU session, then develop against them
indefinitely on a laptop — with the model's genuine quirks intact, including the
malformed responses that are exactly what the parser has to survive.

It also makes failures permanent. A strange response at 2am becomes a fixture
rather than a story.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..errors import BackendUnavailable
from .base import (ExtractRequest, ExtractResult, parse_events,
                   reconcile_times, split_reasoning)


class ReplayBackend:
    name = "replay"
    is_stub = False   # the responses are real, so results are real

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        if not self.dir.exists():
            raise BackendUnavailable(
                f"no recordings at {self.dir}",
                fix="record a session first: make run RECORD=1, on the GPU box",
            )
        # Keyed on all four identifying fields. Older recordings predate the
        # video/prompt_variant fields; they still replay, but only when the clip
        # and prompt are unambiguous, and `ambiguous` reports how many are not.
        self._by_key: dict[tuple, list[dict]] = defaultdict(list)
        self._n = 0
        self.ambiguous = 0
        for f in sorted(self.dir.glob("*.json")):
            rec = json.loads(f.read_text())
            key = (rec.get("video", ""), rec.get("prompt_variant", ""),
                   rec["window"]["index"], rec["query"])
            if not rec.get("video") or not rec.get("prompt_variant"):
                self.ambiguous += 1
            self._by_key[key].append(rec)
            self._n += 1
        if not self._n:
            raise BackendUnavailable(f"{self.dir} contains no recordings")

    def extract(self, req: ExtractRequest) -> ExtractResult:
        recs = self._by_key.get(
            (req.video, req.prompt_variant, req.window.index, req.query))
        if not recs:
            # Fall back to the pre-video/variant key, so older recordings remain
            # usable for parser work even though they cannot distinguish a sweep.
            recs = self._by_key.get(("", "", req.window.index, req.query))
        if not recs:
            return ExtractResult(
                events=[], model="replay",
                error=f"no recording for window {req.window.index} x {req.query!r}",
            )
        rec = recs[0]
        raw = rec.get("raw", "")
        reasoning = rec.get("reasoning") or split_reasoning(raw)[0]
        events, err = parse_events(raw)
        events, recon = reconcile_times(events, req.window)
        return ExtractResult(
            events=events, raw=raw, reasoning=reasoning,
            latency_s=rec.get("latency_s", 0.0),
            model=rec.get("model", "replay"),
            error=err or rec.get("error"),
            meta={"replayed_from": str(self.dir), **recon},
        )

    def describe(self) -> dict:
        return {"backend": self.name, "stub": False,
                "recordings": self._n, "dir": str(self.dir),
                "ambiguous_recordings": self.ambiguous}
