"""A fake vLLM endpoint.

The stub backend proves the pipeline. It does not prove the code that talks to a
model: HTTP, authentication, `<think>` parsing, timestamp clamping, recording,
timeouts, and the model-identity check. All of that sits between us and the GPU,
and none of it needs a GPU to test.

So this speaks the OpenAI-compatible subset vLLM serves, and can be told to
return the responses that actually break parsers:

    python scripts/fake_vllm.py --model nvidia/Cosmos3-Edge --scenario think
    python scripts/fake_vllm.py --scenario truncated
    python scripts/fake_vllm.py --scenario hallucinate

Scenarios exist because a happy-path fake proves almost nothing. The interesting
question is what happens when the model returns a refusal, stops mid-reasoning at
the token limit, or reports a timestamp for footage it never saw.

Standard library only, so it runs anywhere without touching the lockfile.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

SCENARIOS = (
    "think",        # reasoning block then JSON — what Cosmos 3 actually does
    "plain",        # bare JSON, no reasoning
    "fenced",       # JSON inside a ```json fence
    "prose",        # JSON buried in commentary
    "empty",        # a valid "nothing happened" answer
    "truncated",    # stops mid-<think>, as at max_tokens
    "refusal",      # natural language, no JSON at all
    "hallucinate",  # timestamps outside the window it was shown
    "malformed",    # JSON-ish but wrong types
    "slow",         # valid, but slow enough to exercise timeouts
    "error500",     # server error
    "mixed",        # a different scenario per call, deterministic by index
)

_WINDOW = re.compile(r"t=([\d.]+)s to t=([\d.]+)s")


class Handler(BaseHTTPRequestHandler):
    model = "nvidia/Cosmos3-Edge"
    scenario = "think"
    delay = 0.0
    calls = 0

    def log_message(self, fmt, *args):  # quieter output
        print(f"  fake-vllm: {fmt % args}")

    # -- helpers ----------------------------------------------------------

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _window_from(prompt: str) -> tuple[float, float]:
        """Read the window span out of the prompt we generated."""
        m = _WINDOW.search(prompt or "")
        return (float(m.group(1)), float(m.group(2))) if m else (0.0, 12.0)

    def _body(self, start: float, end: float, scenario: str, n: int) -> str:
        span = max(0.1, end - start)
        rng = random.Random(f"{start}:{end}:{n}")
        s = round(start + span * rng.uniform(0.1, 0.5), 3)
        e = round(min(end, s + span * rng.uniform(0.1, 0.4)), 3)
        conf = round(rng.uniform(0.35, 0.95), 2)
        events = (f'{{"start_s": {s}, "end_s": {e}, "confidence": {conf}, '
                  f'"evidence": "observed in frames"}}')
        payload = f'{{"events": [{events}]}}'
        think = ("<think>\nI look at the printed timestamps on each frame. The "
                 f"segment runs from t={start}s to t={end}s. The described event "
                 f"appears to begin near t={s}s.\n</think>\n")

        if scenario == "think":
            return think + payload
        if scenario == "plain":
            return payload
        if scenario == "fenced":
            return think + f"```json\n{payload}\n```"
        if scenario == "prose":
            return (think + "Here is what I found in this segment:\n" + payload +
                    "\nLet me know if you need more detail.")
        if scenario == "empty":
            return think + '{"events": []}'
        if scenario == "truncated":
            # Hit the token limit mid-reasoning: no answer was ever produced.
            return ("<think>\nI examine the frames one by one. The timestamp on "
                    "the first frame reads t=" + str(start) + "s and I am trying "
                    "to determine whether the described")
        if scenario == "refusal":
            return "I'm not able to determine when that happens from these frames."
        if scenario == "hallucinate":
            # Times well outside the window it was shown. Clamping must catch this,
            # or a window emits events for footage it never received.
            return think + (f'{{"events": [{{"start_s": {round(end + 120, 3)}, '
                            f'"end_s": {round(end + 140, 3)}, "confidence": 0.9, '
                            f'"evidence": "outside the window entirely"}}]}}')
        if scenario == "malformed":
            return think + ('{"events": [{"start_s": "twelve", "end_s": null, '
                            '"confidence": 5, "evidence": 42}]}')
        return think + payload

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": self.model, "object": "model", "owned_by": "fake"}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return

        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        Handler.calls += 1
        n = Handler.calls

        scenario = self.scenario
        if scenario == "mixed":
            # Deterministic rotation, so a run is reproducible while still
            # exercising every failure mode in one pass.
            pool = ["think", "empty", "fenced", "truncated", "hallucinate",
                    "prose", "refusal", "malformed"]
            scenario = pool[n % len(pool)]

        if scenario == "error500":
            self._json(500, {"error": {"message": "simulated server error"}})
            return
        if scenario == "slow":
            time.sleep(self.delay or 5.0)

        # Verify the request looks like what we intend to send a real model.
        msgs = req.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), {})
        content = user.get("content", [])
        text = ""
        images = 0
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text += part.get("text", "")
                elif part.get("type") == "image_url":
                    images += 1
        else:
            text = str(content)

        start, end = self._window_from(text)
        body = self._body(start, end, scenario, n)
        print(f"  call {n}: window {start}-{end}s, {images} image(s), "
              f"scenario={scenario}")

        self._json(200, {
            "id": f"chatcmpl-fake-{n}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", self.model),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": body},
                "finish_reason": "length" if scenario == "truncated" else "stop",
            }],
            "usage": {"prompt_tokens": 100 + images * 200,
                      "completion_tokens": len(body) // 4,
                      "total_tokens": 100 + images * 200 + len(body) // 4},
        })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="nvidia/Cosmos3-Edge")
    ap.add_argument("--scenario", default="think", choices=SCENARIOS)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()

    Handler.model = args.model
    Handler.scenario = args.scenario
    Handler.delay = args.delay

    print(f"fake vLLM on http://{args.host}:{args.port}/v1")
    print(f"  model    {args.model}")
    print(f"  scenario {args.scenario}")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
