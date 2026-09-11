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
import math
import time
from dataclasses import replace
import uuid
from pathlib import Path

from ..decode import frame_to_data_url
from ..errors import BackendUnavailable
from ..states import clean, split_by_subject
from .base import (ExtractRequest, ExtractResult, parse_events,
                   reconcile_times, split_reasoning)


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
        detect_enabled: bool = False,
        detect_threshold: float = 0.5,
        detect_max_tokens: int = 1,
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
        self.detect_enabled = detect_enabled
        self.detect_threshold = detect_threshold
        self.detect_max_tokens = detect_max_tokens
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

    def detect(self, req: ExtractRequest) -> tuple[float, float, str]:
        """Stage A — is the described action visible? Returns (p_yes, latency, raw).

        `guided_choice` constrains the output to exactly "yes" or "no", so the
        model cannot rationalise its way to a positive, and `logprobs` gives the
        probability behind that single token. That probability is a real signal,
        unlike a confidence the model states about itself -- which came back as
        exactly 1.0 on 28 of 81 predictions in the measured run.

        A refusal to answer, or an endpoint that cannot do guided decoding, is
        treated as "not present": the alternative is inventing a positive, and
        that is the failure mode this stage exists to remove.
        """
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(req),
                temperature=0.0,
                max_tokens=max(4, self.detect_max_tokens),
                logprobs=True,
                top_logprobs=8,
                # vLLM removed the guided_* parameters in v0.12.0. On 0.29 an
                # extra_body key it does not recognise is silently ignored, so
                # this stage was never actually constrained -- the model
                # free-generated and we read logprobs off whatever it produced.
                # https://docs.vllm.ai/en/latest/features/structured_outputs.html
                extra_body={"structured_outputs": {"choice": ["yes", "no"]}},
            )
        except Exception as e:
            return 0.0, round(time.time() - t0, 3), f"detect failed: {e}"

        latency = round(time.time() - t0, 3)
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()

        # Prefer the logprob. Softmax over just the yes/no alternatives, because
        # guided decoding has already excluded everything else.
        try:
            top = choice.logprobs.content[0].top_logprobs
            lp = {t.token.strip().lower(): t.logprob for t in top}
            if "yes" in lp or "no" in lp:
                y = math.exp(lp.get("yes", -60.0))
                n = math.exp(lp.get("no", -60.0))
                if y + n > 0:
                    return y / (y + n), latency, raw
        except Exception:
            pass
        # No logprobs available: fall back to the word, but never to a confident
        # value -- a coarse 0.6 says "yes, weakly", not "certain".
        return (0.6 if raw.lower().startswith("y") else 0.0), latency, raw

    def caption(self, frames: list, subject: str,
                max_tokens: int = 400) -> tuple[str, bool]:
        """Ask what state the subject is in. Free-form, unconstrained.

        Asking for a DESCRIPTION returns appearance -- colour, handle, frame --
        which is accurate and useless for deciding open versus closed. Asking for
        the state returns "The door is closed in all frames" and "The door is
        closed in the initial frames and then opens in the later frames", which is
        what the parser needs.

        Returns (text, truncated). Truncation matters because the conclusion comes
        last: a cut-off description keeps the setup and loses the answer.
        """
        content: list[dict] = [{"type": "text", "text":
                                f"Look at {subject} in these frames.\n\n"
                                f"What state is it in, and does that state change "
                                f"across the frames? Answer in one or two "
                                f"sentences, saying only what is visible. Begin "
                                f"with the state."}]
        for fr in frames:
            content.append({"type": "image_url",
                            "image_url": {"url": frame_to_data_url(fr.image)}})
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "system", "content":
                       "You describe what is visible in video frames, briefly and "
                       "literally."},
                      {"role": "user", "content": content}],
        )
        msg = r.choices[0].message
        text = msg.content or ""
        # The server concatenates the reasoning block with the answer, so without
        # this the parser reads the model thinking aloud rather than concluding.
        return clean(text), r.choices[0].finish_reason == "length"

    def caption_many(self, frames: list, subjects: list[str],
                     max_tokens: int | None = None) -> tuple[dict[str, str], bool]:
        """One call, several subjects, one line each.

        The frames are identical whichever subject is asked about, so polling per
        description pays repeatedly for the same perception. On the labelled set
        the nine expressible descriptions reduce to four subjects, and a shared
        caption collapses those four sweeps into one.

        The format is load-bearing, not cosmetic: `split_by_subject` parses each
        line in isolation, so a run-on paragraph mentioning two doors would make
        both answers unreliable. Asking for labelled lines is what keeps the
        attribution honest, and a missing line is reported as no answer rather
        than guessed from a neighbour's.
        """
        items = "\n".join(f"- {s}" for s in subjects)
        content: list[dict] = [{"type": "text", "text":
                                f"Look at these frames.\n\n"
                                f"For each item below, say what state it is in and "
                                f"whether that state changes across the frames. "
                                f"Write ONE line per item, beginning with the "
                                f"item's name and a colon. If an item is not "
                                f"visible, say so on its line.\n\n{items}"}]
        for fr in frames:
            content.append({"type": "image_url",
                            "image_url": {"url": frame_to_data_url(fr.image)}})
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.0,
            max_tokens=max_tokens or (120 * len(subjects) + 200),
            messages=[{"role": "system", "content":
                       "You describe what is visible in video frames, briefly and "
                       "literally."},
                      {"role": "user", "content": content}],
        )
        text = clean(r.choices[0].message.content or "")
        return (split_by_subject(text, subjects),
                r.choices[0].finish_reason == "length")

    def extract(self, req: ExtractRequest) -> ExtractResult:
        if not req.window.frames:
            return ExtractResult(events=[], model=self.model,
                                 error="window contained no frames")

        p_yes = None
        detect_latency = 0.0
        if self.detect_enabled and req.detect_prompt is not None:
            probe_req = replace(req, system_prompt=req.detect_prompt[0],
                                user_prompt=req.detect_prompt[1])
            p_yes, detect_latency, detect_raw = self.detect(probe_req)
            if p_yes < self.detect_threshold:
                # Said no. Emit nothing -- and record that it was ASKED, so a
                # true negative is distinguishable from a window never examined.
                return ExtractResult(
                    events=[], model=self.model, latency_s=detect_latency,
                    raw=detect_raw,
                    meta={"stage": "detect", "p_present": round(p_yes, 4),
                          "detected": False, "frames": len(req.window.frames)},
                )

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
        events, recon = reconcile_times(events, req.window)

        # Stage A's probability is the ranking signal, replacing whatever the
        # model may have stated about itself.
        if p_yes is not None:
            for e in events:
                e.confidence = round(p_yes, 4)

        result = ExtractResult(
            events=events, raw=msg.content or "", reasoning=reasoning,
            latency_s=round(latency + detect_latency, 3), model=self.model,
            error=err,
            meta={"finish_reason": resp.choices[0].finish_reason,
                  "frames": len(req.window.frames),
                  **({"p_present": round(p_yes, 4), "detected": True}
                     if p_yes is not None else {}),
                  **recon},
        )
        self._record(req, result)
        return result

    def _record(self, req: ExtractRequest, res: ExtractResult) -> None:
        """Persist the exchange so it can be replayed without a GPU."""
        if not self.record_dir:
            return
        rec = {
            "model": self.model,
            "video": req.video,
            "prompt_variant": req.prompt_variant,
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
