#!/usr/bin/env python3
"""A fake vLLM endpoint that CHECKS the request and returns a canned reply.

Not a simulator of the model -- a checker of everything around it. The failures
this catches are the ones that cost a GPU session to discover:

  * media sent as the wrong content type, or not at all
  * `structured_outputs` missing, so the server would fail open into free text
  * `enable_thinking` left on, multiplying output length
  * `max_tokens` uncapped, which is how single generations reached 80 s
  * a `t` enum that does not match the times named in the prompt, which would
    let the model answer about moments it was never shown

Any of those and it returns HTTP 400 with the reason, so `make ef-smoke` fails
loudly on the laptop instead of quietly on the box.

    python scripts/fake_vllm.py --port 8000 [--model nvidia/Cosmos3-Edge]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_TOKENS_CAP = 1024
STATE_FLIP = {"closed": "open", "open": "closed",
              "standing": "sitting", "sitting": "standing",
              "stationary": "moving", "moving": "stationary"}


class Checker:
    def __init__(self, model: str):
        self.model = model
        self.seen = 0
        self.problems: list[str] = []

    def check(self, body: dict) -> str | None:
        msgs = body.get("messages") or []
        if len(msgs) < 2:
            return "expected a system and a user message"
        content = msgs[-1].get("content")
        if not isinstance(content, list):
            return "user content must be a list of parts (media + text), not a bare string"
        kinds = [c.get("type") for c in content]
        if not ({"video_url", "image_url"} & set(kinds)):
            return f"no media part in the request; parts were {kinds}"
        if "text" not in kinds:
            return "no text part in the request"

        so = body.get("structured_outputs") or {}
        if "json" not in so:
            return ("structured_outputs.json missing. A server without it fails OPEN "
                    "into free text rather than erroring, so this must never be optional.")
        ctk = body.get("chat_template_kwargs") or {}
        if ctk.get("enable_thinking") is not False:
            return "chat_template_kwargs.enable_thinking must be explicitly false"
        mt = body.get("max_tokens")
        if not mt or mt > MAX_TOKENS_CAP:
            return f"max_tokens={mt}; must be set and <= {MAX_TOKENS_CAP}"

        schema = so["json"]
        enum = _enum(schema)
        text = next(c["text"] for c in content if c.get("type") == "text")
        named = [round(float(x), 3) for x in re.findall(r"\d+\.\d{3}", text)]
        if enum and named and set(named) - set(enum):
            return (f"prompt names times {sorted(set(named) - set(enum))} that the schema's "
                    "`t` enum does not allow; the model could answer about a moment it was "
                    "never shown")
        return None

    def reply(self, body: dict) -> dict:
        """A well-formed answer with one state flip in the middle, so a pipeline
        run produces exactly one derivable transition per subject."""
        schema = (body.get("structured_outputs") or {})["json"]
        times = _enum(schema) or [0.0]
        props = schema["properties"]["observations"]["items"]["properties"]
        wants_state = "state" in props
        half = len(times) / 2
        obs = []
        for i, t in enumerate(times):
            rec = {"t": t, "present": True, "certainty": 0.8}
            if wants_state:
                rec["state"] = "closed" if i < half else "open"
            if "motion" in props:
                rec["motion"] = "moving" if i < half else "stationary"
            if "position" in props:
                rec["position"] = [round(0.8 - 0.05 * i, 3), 0.5]
            if "facing" in props:
                rec["facing"] = "right"
            if "relations" in props:
                rec["relations"] = {"held_by": "person a" if i < half else "person b"}
            obs.append(rec)
        out: dict = {"observations": obs}
        if "matches" in schema.get("properties", {}):
            out["matches"] = True
            out["says"] = "fake server: canned reply"
            out["confidence"] = 0.9
        return out


def _enum(schema: dict) -> list[float]:
    try:
        return schema["properties"]["observations"]["items"]["properties"]["t"]["enum"]
    except Exception:
        return []


def make_handler(checker: Checker):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code: int, payload: dict):
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list",
                                 "data": [{"id": checker.model, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            problem = checker.check(body)
            if problem:
                checker.problems.append(problem)
                print(f"  REQUEST REJECTED: {problem}", file=sys.stderr, flush=True)
                self._send(400, {"error": {"message": problem, "type": "request_shape"}})
                return
            checker.seen += 1
            content = json.dumps(checker.reply(body))
            self._send(200, {
                "id": f"fake-{checker.seen}", "object": "chat.completion",
                "model": checker.model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": len(content) // 4,
                          "total_tokens": 100 + len(content) // 4},
            })
    return H


def serve(port: int, model: str) -> tuple[ThreadingHTTPServer, Checker]:
    """Threading, not the plain HTTPServer: the client sends concurrent
    keep-alive requests, and a single-threaded server deadlocks on the second
    one rather than erroring."""
    c = Checker(model)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(c))
    srv.daemon_threads = True
    return srv, c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="nvidia/Cosmos3-Edge")
    a = ap.parse_args()
    srv, _ = serve(a.port, a.model)
    print(f"fake vLLM on http://127.0.0.1:{a.port}/v1  model={a.model}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
