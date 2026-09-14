"""Replay recorded exchanges from a real run. No GPU, no server, no cost.

A GPU session that records is a session that only has to happen once. Every
call the vLLM backend makes is written to JSONL; this reads them back and
re-parses them, so derivation, confidence and scoring can be changed and
re-measured for free -- against the model's ACTUAL errors, in their actual
correlation structure, rather than against an invented noise model whose
parameters nobody can defend choosing.

Results are tagged `replay` and the evaluator never prints them under a model
heading. A replay is evidence about derivation, not about the model: change the
prompt and the corpus is stale, which is why the prompt version travels in every
record.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..models import Observation
from .base import PROMPT_VERSION, Usage, Verdict
from .vllm import _load, _parse


class ReplayReasoner:
    name = "replay"

    def __init__(self, exchanges_path: str | Path, *, strict_prompt: bool = True):
        self.model = "replay"
        self.usage = Usage()
        self.context = ""
        self.by_key: dict[tuple[str, str, str, str], list[dict]] = {}
        self.no_video = 0
        self.versions: set[str] = set()
        self.media_kinds: set[str] = set()
        n = 0
        with open(exchanges_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                vid = e.get("video", "")
                if not vid:
                    self.no_video += 1
                key = (vid, e["bracket_id"], e["subject"], e.get("mode", "observe"))
                self.by_key.setdefault(key, []).append(e)
                self.model = e.get("model", self.model)
                self.versions.add(e.get("prompt_version", "?"))
                self.media_kinds.add(e.get("media", "?"))
                n += 1
        self.n_records = n
        if self.no_video:
            raise ValueError(
                f"{self.no_video} of {n} records in {exchanges_path} carry no `video` "
                "field. Bracket ids are per-clip (b1, b2, ...), so without it every "
                "clip's b1 collides and each clip silently replays another clip's "
                "replies. Run scripts/fix_exchanges.py to recover the corpus.")
        # A corpus recorded under a different prompt answers a different
        # question. Failing loudly beats silently scoring stale replies.
        if strict_prompt and self.versions - {PROMPT_VERSION}:
            raise ValueError(
                f"this corpus was recorded under prompt version(s) "
                f"{sorted(self.versions)}; the code is now {PROMPT_VERSION!r}. "
                "Its replies answer the OLD prompt, so scoring them as if they "
                "answered the new one would misattribute the difference.\n"
                "  To measure a change to DERIVATION (prompt-independent), it is "
                "fine to re-score them:\n"
                "      make ef-replay EXCHANGES=... STALE=1\n"
                "      python scripts/eval_events.py --replay ... --stale-ok\n"
                "  To make a claim about the MODEL under the new prompt, record "
                "again with a GPU run.")

    def verify_model(self) -> tuple[bool, str]:
        return True, f"replay of {self.model} ({self.n_records} exchanges, media={sorted(self.media_kinds)})"

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raise RuntimeError("replay has no text model; use the rules compiler")

    def _get(self, bracket_id: str, subject: str, mode: str,
             attributes: list[str] | None = None):
        self.usage.calls += 1
        e = self._pick(self.by_key.get((self.context, bracket_id, subject, mode)) or [],
                       attributes or [])
        if e is None:
            self.usage.failures += 1
            return None
        self.usage.latency_s.append(float(e.get("latency_s", 0.0)))
        if e.get("error") or not e.get("reply"):
            self.usage.failures += 1
            return None
        data, repaired = _load(e["reply"])
        self.usage.repairs += int(repaired)
        if data is None:
            self.usage.failures += 1
        return data

    @staticmethod
    def _pick(cands: list[dict], attributes: list[str]) -> dict | None:
        """Choose among records sharing one key.

        A corpus recorded before calls were grouped by subject holds several
        records per (video, bracket, subject) -- one per attribute set. The
        wanted one is whichever was asked for the most fields, since the
        smaller sets are subsets of it and its reply therefore contains
        everything the others would have returned. Nothing is inferred: a
        corpus that records `attributes` is matched exactly.
        """
        if not cands:
            return None
        exact = [c for c in cands if c.get("attributes") == list(attributes)]
        if exact:
            return exact[0]
        return max(cands, key=lambda c: len(c.get("attributes") or []) or _fields(c))

    def observe(self, frames, stamp_times: list[float], subject: str,
                attributes: list[str], bracket_id: str,
                state_vocab: list[str] | None = None,
                constrain_state: bool = False) -> list[Observation]:
        data = self._get(bracket_id, subject, "observe", attributes)
        if data is None:
            return []
        obs, rep = _parse(data, stamp_times, subject, bracket_id, "")
        self.usage.repairs += int(rep)
        for o in obs:
            o.source = "replay"
        return obs

    def verify(self, frames, stamp_times: list[float], subject: str,
               attributes: list[str], bracket_id: str, description: str,
               state_vocab: list[str] | None = None,
               constrain_state: bool = False) -> Verdict:
        data = self._get(bracket_id, subject, "verify", attributes)
        if data is None:
            return Verdict(observations=[])
        obs, rep = _parse(data, stamp_times, subject, bracket_id, "")
        self.usage.repairs += int(rep)
        for o in obs:
            o.source = "replay"
        return Verdict(observations=obs, matches=data.get("matches"),
                       says=str(data.get("says", ""))[:200],
                       confidence=data.get("confidence"))


def _fields(e: dict) -> int:
    """How many distinct observation fields a recorded reply carries -- the
    stand-in for `attributes` on a corpus that predates it."""
    try:
        d = json.loads(e.get("reply") or "{}")
        return max((len(o) for o in d.get("observations", [])), default=0)
    except Exception:
        return 0


def write_exchanges(path: str | Path, usage: Usage) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in usage.exchanges:
            f.write(json.dumps(e) + "\n")
    return path


def summarise(exchanges_path: str | Path) -> dict:
    """What a recorded session actually cost and how often it failed.

    The headline numbers for session 1 come from here: latency by mode and by
    media path, and the truncation rate that says whether max_tokens is set
    anywhere near right.
    """
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(exchanges_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for e in rows:
        by[(e.get("mode", "observe"), e.get("media", "?"))].append(e)
    out = {}
    for (mode, media), es in sorted(by.items()):
        lat = sorted(e.get("latency_s", 0.0) for e in es)
        out[f"{mode}/{media}"] = {
            "calls": len(es),
            "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
            "p50_latency_s": lat[len(lat) // 2] if lat else 0.0,
            "p95_latency_s": lat[int(len(lat) * 0.95)] if lat else 0.0,
            "errors": sum(1 for e in es if e.get("error")),
            "truncated": sum(1 for e in es if e.get("finish_reason") == "length"),
            "mean_reply_chars": round(sum(len(e.get("reply") or "") for e in es) / len(es), 1),
        }
    return out
