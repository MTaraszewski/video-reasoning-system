"""The real backend: an OpenAI-compatible vLLM endpoint.

Cosmos 3's Reasoner is served natively by stock vLLM as
`Cosmos3EdgeForConditionalGeneration`. We use only that tower — video in, text out
— so vLLM-Omni and the `--omni` flag, which serve the diffusion Generator, are out
of scope.

Every exchange can be recorded to disk. One GPU session then produces fixtures
that make all later parser and prompt work runnable on a laptop, instead of at
GPU rates.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..decode import frame_to_data_url
from ..errors import BackendUnavailable
from .base import ExtractRequest, ExtractResult, clamp_to_window, parse_events, split_reasoning


class VLLMBackend:
    name = "vllm"
    is_stub = False

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_s: float = 300.0,
        record_dir: str | Path | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable("openai client not installed") from e
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.record_dir = Path(record_dir) if record_dir else None
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)

    # -- health ------------------------------------------------------------

    def served_models(self) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception as e:
            raise BackendUnavailable(
                f"cannot reach the model endpoint at {self.base_url}: {e}",
                fix="start it with `make serve-bg`, or point BASE_URL elsewhere",
            ) from e

    def check(self) -> None:
        """Confirm the endpoint serves the model we think it does.

        A mismatch is worse than an outage: every result would be attributed to
        the wrong model, and wrong attribution looks like data.
        """
        served = self.served_models()
        if self.model not in served:
            raise BackendUnavailable(
                f"endpoint is serving {served or '<nothing>'}, but MODEL is "
                f"{self.model!r}. Results would be attributed to the wrong model.",
                fix=f"serve the right model, or set MODEL to one of {served}",
            )

    # -- extraction --------------------------------------------------------

    def _messages(self, req: ExtractRequest) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": req.user_prompt}]
        for fr in req.window.frames:
            content.append(
                {"type": "image_url",
                 "image_url": {"url": frame_to_data_url(fr.image)}}
            )
        return [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": content},
        ]

    def extract(self, req: ExtractRequest) -> ExtractResult:
        if not req.window.frames:
            return ExtractResult(events=[], model=self.model,
                                 error="window contained no frames")

        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(req),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            return ExtractResult(events=[], model=self.model,
                                 latency_s=time.time() - t0, error=str(e))

        latency = time.time() - t0
        msg = resp.choices[0].message
        text = msg.content or ""

        # vLLM's --reasoning-parser puts the think block in its own field. Without
        # that flag it stays inline. Handle both rather than depending on serving
        # flags we do not control.
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if not reasoning:
            reasoning, text = split_reasoning(text)

        events, err = parse_events(text if reasoning else (msg.content or ""))
        events = clamp_to_window(events, req.window)

        result = ExtractResult(
            events=events, raw=msg.content or "", reasoning=reasoning,
            latency_s=round(latency, 3), model=self.model, error=err,
            meta={"finish_reason": resp.choices[0].finish_reason,
                  "frames": len(req.window.frames)},
        )
        self._record(req, result)
        return result

    def _record(self, req: ExtractRequest, res: ExtractResult) -> None:
        """Persist the exchange so it can be replayed without a GPU."""
        if not self.record_dir:
            return
        rec = {
            "model": self.model,
            "query": req.query,
            "window": {"index": req.window.index,
                       "start_s": req.window.start_s, "end_s": req.window.end_s,
                       "frames": len(req.window.frames)},
            "system_prompt": req.system_prompt,
            "user_prompt": req.user_prompt,
            "raw": res.raw,
            "reasoning": res.reasoning,
            "latency_s": res.latency_s,
            "error": res.error,
            "meta": res.meta,
        }
        name = f"w{req.window.index:04d}-{uuid.uuid4().hex[:8]}.json"
        (self.record_dir / name).write_text(json.dumps(rec, indent=2))

    def describe(self) -> dict:
        return {"backend": self.name, "stub": False, "model": self.model,
                "base_url": self.base_url,
                "recording": str(self.record_dir) if self.record_dir else None}
