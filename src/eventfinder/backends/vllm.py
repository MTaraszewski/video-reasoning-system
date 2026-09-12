"""OpenAI-compatible backend for a vLLM server running Cosmos 3 Edge.

Request-shape choices, and their evidence status:

**Media path -- CONFLICT, unresolved, and measured in session 1.**
  This repo's earlier engine sent each window as a LIST OF IMAGES
  (`image_url` per frame) and produced real results on an L4.
  The design note this architecture follows asserts the opposite: that a
  Qwen3-VL-lineage processor applies temporal merging and timestamp alignment
  only on the `video_url` path, and that image lists silently disable both --
  but that note was never run against a GPU.
  Both paths are implemented here, switchable, over the SAME Frame objects, so
  the comparison is one flag apart and isolates the request shape.

**Thinking off** (`chat_template_kwargs.enable_thinking=false`) -- INHERITED.
  A per-time observation needs no reasoning trace, and traces multiply output
  length. Untested here.

**Structured output** (`structured_outputs.json`) -- INHERITED. The older
  `guided_json` is reportedly ignored by recent servers without an error, which
  would degrade silently into free text.

**max_tokens capped** -- MEASURED as necessary, size not yet tuned. The earlier
  engine ran uncapped at 4096 and paid up to 80 s for single generations that
  carried no extra information.

**Identity checked before any result is attributed to a model name.**
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
import time
from pathlib import Path

from openai import OpenAI

from ..decode import Frame, write_clip
from ..models import Observation
from .base import (
    PROMPT_VERSION,
    Usage,
    Verdict,
    observation_schema,
    observer_prompt,
    verify_prompt,
    verify_schema,
)


class VLLMReasoner:
    name = "vllm"

    def __init__(self, base_url: str, model: str, *, api_key: str = "EMPTY",
                 temperature: float = 0.0, max_tokens: int = 192,
                 media: str = "video", timeout_s: float = 120.0,
                 sample_fps: float = 4.0, record: bool = True,
                 jpeg_quality: int = 85, tokens_per_record: int = 48):
        if media not in ("video", "frames"):
            raise ValueError(f"media must be 'video' or 'frames', not {media!r}")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokens_per_record = tokens_per_record
        self.media = media
        self.sample_fps = sample_fps
        self.record = record
        self.jpeg_quality = jpeg_quality
        self.usage = Usage()

    # --- identity ---------------------------------------------------------

    def verify_model(self) -> tuple[bool, str]:
        """A run that never asked what it was talking to is not a model result."""
        try:
            ids = [m.id for m in self.client.models.list().data]
        except Exception as e:
            return False, f"models.list failed: {e}"
        if self.model in ids:
            return True, self.model
        return False, f"served={ids} configured={self.model}"

    # --- text-only, for the compiler --------------------------------------

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0, max_tokens=300,
            extra_body={"chat_template_kwargs": {"enable_thinking": False},
                        "structured_outputs": {"json": schema}},
        )
        self._count(r)
        return json.loads(_strip(r.choices[0].message.content or ""))

    # --- observation ------------------------------------------------------

    def observe(self, frames: list[Frame], stamp_times: list[float], subject: str,
                attributes: list[str], bracket_id: str) -> list[Observation]:
        system, user = observer_prompt(subject, attributes, stamp_times)
        schema = observation_schema(stamp_times, attributes)
        data, raw = self._call(frames, system, user, schema, stamp_times,
                               bracket_id, subject, "observe")
        if data is None:
            return []
        obs, repaired = _parse(data, stamp_times, subject, bracket_id, raw)
        self.usage.repairs += int(repaired)
        return obs

    def verify(self, frames: list[Frame], stamp_times: list[float], subject: str,
               attributes: list[str], bracket_id: str, description: str) -> Verdict:
        system, user = verify_prompt(subject, attributes, stamp_times, description)
        schema = verify_schema(stamp_times, attributes)
        data, raw = self._call(frames, system, user, schema, stamp_times,
                               bracket_id, subject, "verify", description)
        if data is None:
            return Verdict(observations=[])
        obs, repaired = _parse(data, stamp_times, subject, bracket_id, raw)
        self.usage.repairs += int(repaired)
        return Verdict(observations=obs, matches=data.get("matches"),
                       says=str(data.get("says", ""))[:200],
                       confidence=data.get("confidence"))

    # --- the call itself --------------------------------------------------

    def _call(self, frames, system, user, schema, stamp_times, bracket_id,
              subject, mode, description=""):
        content = self._media(frames) + [{"type": "text", "text": user}]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
        extra = {"chat_template_kwargs": {"enable_thinking": False},
                 "structured_outputs": {"json": schema}}
        if self.media == "video":
            extra["media_io_kwargs"] = {"video": {"num_frames": len(frames)}}

        budget = self._budget(len(stamp_times))
        t0 = time.perf_counter()
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                max_tokens=budget, extra_body=extra)
        except Exception as e:
            dt = time.perf_counter() - t0
            self.usage.failures += 1
            self.usage.latency_s.append(dt)
            self._rec(bracket_id, subject, stamp_times, mode, description, None,
                      f"{type(e).__name__}: {e}", dt, None)
            return None, ""
        dt = time.perf_counter() - t0
        self.usage.latency_s.append(dt)
        self._count(r)
        raw = _strip(r.choices[0].message.content or "")
        finish = getattr(r.choices[0], "finish_reason", None)
        data, repaired = _load(raw)
        if data is None:
            self.usage.failures += 1
        self._rec(bracket_id, subject, stamp_times, mode, description, raw, None, dt, finish)
        if repaired:
            self.usage.repairs += 1
        return data, raw

    def _budget(self, n_times: int) -> int:
        """Enough room for one record per time asked about, and no more.

        Truncation here is invisible: the repair path salvages what arrived, so
        a starved call returns fewer observations rather than an error, and the
        loss is indistinguishable from the model declining to answer.
        """
        return max(self.max_tokens, self.tokens_per_record * n_times + 64)

    def _media(self, frames: list[Frame]) -> list[dict]:
        """The switchable half of the CONFLICT above. Same frames either way."""
        if self.media == "frames":
            return [{"type": "image_url",
                     "image_url": {"url": _jpeg_url(f, self.jpeg_quality)}} for f in frames]
        with tempfile.TemporaryDirectory() as d:
            p = write_clip(frames, Path(d) / "clip.mp4", fps=self.sample_fps)
            url = "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()
        return [{"type": "video_url", "video_url": {"url": url}}]

    # --- bookkeeping ------------------------------------------------------

    def _count(self, r) -> None:
        self.usage.calls += 1
        u = getattr(r, "usage", None)
        if u:
            self.usage.tokens_in += u.prompt_tokens or 0
            self.usage.tokens_out += u.completion_tokens or 0

    def _rec(self, bracket_id, subject, times, mode, description, reply, error,
             dt, finish) -> None:
        """One line per exchange. A GPU session that records is a GPU session
        that only has to happen once."""
        if not self.record:
            return
        self.usage.exchanges.append({
            "bracket_id": bracket_id, "subject": subject, "times": times,
            "mode": mode, "description": description, "reply": reply, "error": error,
            "latency_s": round(dt, 3), "finish_reason": finish, "model": self.model,
            "media": self.media, "max_tokens": self._budget(len(times)),
            "prompt_version": PROMPT_VERSION,
        })


# --- parsing ---------------------------------------------------------------

def _jpeg_url(f: Frame, quality: int) -> str:
    buf = io.BytesIO()
    f.image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _strip(s: str) -> str:
    if "<think>" in s and "</think>" in s:
        s = s.split("</think>", 1)[1]
    return s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _load(text: str) -> tuple[dict | None, bool]:
    """Parse, with one cheap repair for a truncated generation.

    A reply cut off by max_tokens is the expected failure of capping it, and
    losing the whole call to that is worse than salvaging the records that did
    arrive. The repair is counted so the rate stays visible.
    """
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    fixed = text.replace(",]", "]").replace(",}", "}")
    if not fixed.rstrip().endswith("}"):
        cut = fixed.rfind("}")
        if cut == -1:
            return None, True
        fixed = fixed[:cut + 1] + "]}"
    try:
        return json.loads(fixed), True
    except Exception:
        return None, True


def _parse(data: dict, stamp_times: list[float], subject: str, bracket_id: str,
           raw: str) -> tuple[list[Observation], bool]:
    """Records for times that were actually shown, in time order.

    The schema already constrains `t` to the stamp list, but a server without
    structured-output support would fail open into free text -- so it is
    enforced here too. Silent acceptance of an invented time would put the model
    back in charge of localisation, which is the one thing the design forbids.
    """
    allowed = {round(float(t), 3) for t in stamp_times}
    out: list[Observation] = []
    seen: set[float] = set()
    repaired = False
    for rec in (data.get("observations") or []):
        try:
            t = round(float(rec["t"]), 3)
        except Exception:
            repaired = True
            continue
        if t not in allowed or t in seen:
            repaired = True
            continue
        seen.add(t)
        try:
            pos = rec.get("position")
            state = rec.get("state")
            out.append(Observation(
                t=t, subject=subject, bracket_id=bracket_id,
                present=bool(rec.get("present", False)),
                # Canonicalised downstream by the probe's vocabulary; the raw
                # text is kept so a parse failure stays legible.
                state=(str(state).lower().strip() or None) if state else None,
                raw_state=(str(state) if state else None),
                position=(float(pos[0]), float(pos[1])) if pos else None,
                facing=rec.get("facing"), motion=rec.get("motion", "unknown"),
                relations={str(k).lower(): str(v).lower()
                           for k, v in (rec.get("relations") or {}).items()},
                certainty=float(rec.get("certainty", 0.5)), source="model",
            ))
        except Exception:
            repaired = True
    # Times that were asked for and never came back are absences of evidence,
    # not evidence of absence -- recorded so coverage can see them.
    for t in sorted(allowed - seen):
        out.append(Observation(t=t, subject=subject, bracket_id=bracket_id,
                               ok=False, note="no record returned for this time",
                               source="model"))
    out.sort(key=lambda o: o.t)
    return out, repaired
