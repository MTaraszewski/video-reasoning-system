"""Poll the scene's STATE over time and print the timeline.

Approach 1 asked the model to find and time an event; it could do neither. This
asks a much easier question, repeatedly: which of these fixed options describes
what you see? Events then come from where the answer changes -- so the timestamp
is ours, from the sampling grid, not something the model has to report.

Throwaway probe, not pipeline code. It exists to answer one question before any
of that gets built: can the model classify a persistent state at all?

    python scripts/state_timeline.py \
      --video /data/eval/2018-03-07.16-50-01.16-55-01.admin.G326.r13.mp4 \
      --states "the door is closed" "the door is open" \
      --truth 3.0 5.733 --end 20
"""
from __future__ import annotations

import argparse
import math

from openai import OpenAI

from video_reasoning.config import load_config
from video_reasoning.decode import frame_to_data_url, sample_frames


def classify(client, model, frames, states, debug=False):
    """Ask which state fits. Returns (index, probability)."""
    opts = "\n".join(f"({chr(97 + i)}) {s}" for i, s in enumerate(states))
    letters = [chr(97 + i) for i in range(len(states))]
    content = [{"type": "text",
                "text": f"Which of these describes what you see?\n\n{opts}\n\n"
                        f"Answer with one letter."}]
    for f in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": frame_to_data_url(f.image)}})
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=4, logprobs=True, top_logprobs=10,
        # vLLM removed guided_choice in v0.12.0; on 0.29 it is silently ignored
        # and the model free-generates. That is what produced raw='Got'.
        # https://docs.vllm.ai/en/latest/features/structured_outputs.html
        extra_body={"structured_outputs": {"choice": letters}},
        messages=[{"role": "system",
                   "content": "You classify what is visible in video frames. "
                              "Answer with a single letter."},
                  {"role": "user", "content": content}],
    )
    msg = r.choices[0].message
    raw = msg.content or ""
    if debug:
        print(f"    [raw={raw[:60]!r} finish={r.choices[0].finish_reason} "
              f"reasoning={(getattr(msg, 'reasoning_content', None) or '')[:40]!r}]")
    tok = raw.strip().lower()[:1]
    # A response that is not one of the offered letters must not silently become
    # the first option -- that is what made every poll read "(a) closed".
    idx = letters.index(tok) if tok in letters else None
    # Probability of the chosen letter against the others.
    try:
        lp = {t.token.strip().lower(): t.logprob
              for t in r.choices[0].logprobs.content[0].top_logprobs}
        tot = sum(math.exp(lp[l]) for l in letters if l in lp)
        p = math.exp(lp.get(tok, -60.0)) / tot if tot else 0.0
    except Exception:
        p = float("nan")
    return idx, p, raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--truth", nargs=2, type=float, metavar=("START", "END"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=20.0)
    ap.add_argument("--step", type=float, default=1.0, help="Seconds between polls.")
    ap.add_argument("--span", type=float, default=2.0, help="Seconds of frames per poll.")
    args = ap.parse_args()

    cfg = load_config()
    client = OpenAI(base_url=cfg.model.base_url, api_key=cfg.model.api_key,
                    timeout=cfg.model.request_timeout_s)

    print(f"clip   {args.video.rsplit('/', 1)[-1]}")
    for i, s in enumerate(args.states):
        print(f"  ({chr(97 + i)}) {s}")
    if args.truth:
        print(f"truth  {args.truth[0]:.1f}-{args.truth[1]:.1f}s")
    print()

    prev = None
    t = args.start
    while t < args.end:
        _, frames = sample_frames(args.video, fps=cfg.sampling.fps,
                                  max_side=cfg.sampling.frame_max_side,
                                  overlay=False, start_s=t, end_s=t + args.span)
        if not frames:
            break
        idx, p, raw = classify(client, cfg.model.name, frames, args.states,
                               debug=(t == args.start))
        inside = "*" if args.truth and args.truth[0] <= t <= args.truth[1] else " "
        if idx is None:
            print(f" {inside} t={t:>5.1f}s  UNPARSED  raw={raw[:40]!r}")
            prev = None
            t += args.step
            continue
        mark = "  <- CHANGE" if prev is not None and idx != prev else ""
        print(f" {inside} t={t:>5.1f}s  ({chr(97 + idx)}) {args.states[idx][:38]:<38} "
              f"p={p:.2f}{mark}")
        prev = idx
        t += args.step

    print("\n* = inside the labelled event window.")
    print("A clean CHANGE near the truth means state polling works and the design "
          "is worth building. A flat or oscillating column means it is not.")


if __name__ == "__main__":
    main()
