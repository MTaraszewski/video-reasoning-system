"""Let the model reason, then classify its own reasoning.

Every probe so far capped generation at 1-4 tokens and read a logprob. That uses a
reasoning model as a one-token classifier, which is the one mode it was not built
for -- and all three framings tried so far returned something close to chance.

Here it describes what it sees first, freely, and a second constrained call scores
that description. If the perception is there but the single-token head cannot
express it, this recovers it. If the descriptions are themselves wrong or generic,
that is a cleaner negative than a flat probability: we get to read what it thought
it saw.

    python scripts/reason_timeline.py \
      --video /data/eval/...admin.G326.r13.mp4 \
      --states "the door is closed" "the door is open" \
      --truth 3.0 5.733 --end 20
"""
from __future__ import annotations

import argparse
import math
import re
import textwrap

from openai import OpenAI

from video_reasoning.config import load_config
from video_reasoning.decode import frame_to_data_url, sample_frames


PREAMBLE = re.compile(
    r'^\s*(got it|okay|ok|sure|alright|let\'s|let me|first,?)\b[^.]*[.:]\s*',
    re.I)


def after_think(text: str) -> str:
    """Keep only what follows the reasoning block.

    The server concatenates the model's thinking and its answer, so the scored
    text was the model reasoning aloud rather than its conclusion -- visible as
    a stray "</think>" mid-string in several descriptions.
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def strip_preamble(text: str) -> str:
    """Drop the conversational opener.

    Every response begins "Got it, let's look at the images." That is not a
    description, it consumes the token budget before the content arrives, and it
    is what the text classifier was scoring -- G423 returned +0.96 for "sitting"
    on text whose visible content said "standing".
    """
    prev = None
    while prev != text:
        prev = text
        text = PREAMBLE.sub("", text, count=1)
    return text.strip()


def describe(client, model, frames, subject, max_tokens=400):
    """Free-form: what is the subject doing? No constraint, no forced choice."""
    # Ask for the STATE, not for a description. "Describe the door" returns
    # colour, handle and frame -- accurate, and useless for deciding open versus
    # closed. When the answer happened to mention state the classifier was right
    # (t=2 closed, t=4 opening, t=6 open); when it described appearance the
    # classifier had nothing to read and returned noise, scoring +0.91 for "open"
    # on text saying "closed in most frames".
    content = [{"type": "text", "text":
                f"Look at {subject} in these frames.\n\n"
                f"What state is it in, and does that state change across the "
                f"frames? Answer in one or two sentences, saying only what is "
                f"visible. Begin with the state."}]
    for f in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": frame_to_data_url(f.image)}})
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content":
                   "You describe what is visible in video frames, briefly and "
                   "literally."},
                  {"role": "user", "content": content}],
    )
    msg = r.choices[0].message
    text = strip_preamble(after_think((msg.content or "").strip()))
    think = (getattr(msg, "reasoning_content", None) or "").strip()
    # A truncated description is worse than a short one: the conclusion tends to
    # come last, so cutting it off leaves the setup and drops the answer.
    return text, think, r.choices[0].finish_reason


def score_once(client, model, text, states):
    """One classification of a description. Text only — no images."""
    letters = [chr(97 + i) for i in range(len(states))]
    opts = "\n".join(f"({l}) {s}" for l, s in zip(letters, states))
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=4, logprobs=True, top_logprobs=10,
        extra_body={"structured_outputs": {"choice": letters}},
        messages=[{"role": "system", "content":
                   "You read a description and pick which statement it supports. "
                   "Answer with a single letter."},
                  {"role": "user", "content":
                   f"Description:\n{text}\n\nWhich does this support?\n{opts}\n\n"
                   f"Answer with one letter."}],
    )
    try:
        lp = {t.token.strip().lower(): t.logprob
              for t in r.choices[0].logprobs.content[0].top_logprobs}
        ps = {l: math.exp(lp[l]) for l in letters if l in lp}
        tot = sum(ps.values())
        if not tot:
            return {}
        return {states[letters.index(l)]: v / tot for l, v in ps.items()}
    except Exception:
        return {}


def score_text(client, model, text, states):
    """Ask in BOTH option orders and average.

    The image polling was order-averaged and this was not, so the text classifier
    inherited the same position bias in full: it returned +0.86 to +1.00 for
    "open" on descriptions reading "The door is closed in all frames". The
    descriptions were correct at every timestep and the scorer was answering "(b)"
    whichever statement sat second.
    """
    out: dict[str, list[float]] = {s: [] for s in states}
    for order in (list(states), list(reversed(states))):
        for k, v in score_once(client, model, text, order).items():
            out[k].append(v)
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--states", nargs=2, required=True)
    ap.add_argument("--subject", default=None,
                    help="What to describe. Defaults to the noun in state one.")
    ap.add_argument("--truth", nargs=2, type=float, metavar=("START", "END"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=20.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--span", type=float, default=2.0)
    ap.add_argument("--max-side", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.max_side:
        cfg.sampling.frame_max_side = args.max_side
    client = OpenAI(base_url=cfg.model.base_url, api_key=cfg.model.api_key,
                    timeout=cfg.model.request_timeout_s)
    subject = args.subject or args.states[0].replace("the ", "").split(" is ")[0]
    a, b = args.states

    print(f"clip    {args.video.rsplit('/', 1)[-1]}")
    print(f"subject {subject!r}")
    if args.truth:
        print(f"truth   {args.truth[0]:.1f}-{args.truth[1]:.1f}s")
    print()

    t = args.start
    while t < args.end:
        _, frames = sample_frames(args.video, fps=cfg.sampling.fps,
                                  max_side=cfg.sampling.frame_max_side,
                                  overlay=False, start_s=t, end_s=t + args.span)
        if not frames:
            break
        text, think, finish = describe(client, cfg.model.name, frames, subject)
        pr = score_text(client, cfg.model.name, text, args.states) if text else {}
        score = pr.get(b, float("nan")) - pr.get(a, float("nan"))
        inside = "*" if args.truth and args.truth[0] <= t <= args.truth[1] else " "
        cut = "  [TRUNCATED]" if finish == "length" else ""
        print(f" {inside} t={t:>5.1f}s  {score:+.2f}{cut}")
        for line in textwrap.wrap(text or "(empty)", 92)[:4]:
            print(f"              {line}")
        t += args.step

    print("\n* = inside the labelled event window.")
    print("Read the DESCRIPTIONS, not just the scores. If they are generic or "
          "wrong, the limit is perception and no scoring scheme recovers it.")


if __name__ == "__main__":
    main()
