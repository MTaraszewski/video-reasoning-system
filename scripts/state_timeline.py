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
from PIL import Image

from video_reasoning.config import load_config
from video_reasoning.decode import frame_to_data_url, sample_frames


def crop_frames(frames, box, max_side):
    """Crop at NATIVE resolution, then resize the crop back up to max_side.

    Order matters and getting it wrong looks like a negative result. Cropping a
    frame that has already been downscaled to 640px gains nothing: the subject
    occupies the same pixels it did, in a smaller image with the context removed.
    Measured -- that collapsed the score range from 0.12 to 0.03.

    Cropping the full-resolution frame and resizing the crop up is what actually
    enlarges the subject: a 295px actor in a 0.45-height crop of a 1080p frame
    comes back at ~246px instead of ~98px.
    """
    if not box:
        return frames
    x0, y0, x1, y1 = box
    for f in frames:
        w, h = f.image.size
        c = f.image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        scale = max_side / max(c.size)
        if scale != 1.0:
            c = c.resize((max(1, round(c.width * scale)),
                          max(1, round(c.height * scale))), Image.LANCZOS)
        f.image = c
    return frames


def ask_once(client, model, frames, states, debug=False):
    """One classification. Returns {state: probability} over ALL states.

    The probability of every option is read, not just the winner. Which option
    wins turned out to be nearly worthless -- the model answered "(b)" whatever
    (b) said -- but the distribution over options still carries signal.
    """
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
        print(f"    [raw={raw[:40]!r} finish={r.choices[0].finish_reason}]")
    try:
        lp = {t.token.strip().lower(): t.logprob
              for t in r.choices[0].logprobs.content[0].top_logprobs}
    except Exception:
        return {}
    probs = {l: math.exp(lp[l]) for l in letters if l in lp}
    tot = sum(probs.values())
    if not tot:
        return {}
    return {states[letters.index(l)]: v / tot for l, v in probs.items()}


def poll(client, model, frames, states, debug=False, repeat=1):
    """Ask in BOTH option orders and average — cancels option-order bias.

    Measured on one clip: with (a) closed / (b) open the model answered "b" at
    every timestep; with the labels swapped it answered "b" again. It was picking
    the last option, not reading the scene, and the probability curve rose at the
    same moment in both runs -- so the apparent signal was positional.

    Position bias is symmetric under reversal, so averaging the two orders
    cancels it and leaves whatever perception is underneath.
    """
    out: dict[str, list[float]] = {s: [] for s in states}
    for _ in range(max(1, repeat)):
        for order in (list(states), list(reversed(states))):
            got = ask_once(client, model, frames, order, debug=debug)
            debug = False
            for k, v in got.items():
                out[k].append(v)
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--truth", nargs=2, type=float, metavar=("START", "END"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=20.0)
    ap.add_argument("--step", type=float, default=1.0, help="Seconds between polls.")
    ap.add_argument("--span", type=float, default=2.0, help="Seconds of frames per poll.")
    ap.add_argument("--crop", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                    help="Region of interest as FRACTIONS of the frame, e.g. "
                         "0.3 0.1 0.7 0.6. Chosen once by a human looking at the "
                         "scene -- never derived from per-frame annotations, "
                         "which would be leakage.")
    ap.add_argument("--max-side", type=int, default=None,
                    help="Frame size sent to the model. Defaults to config "
                         "(640, Cosmos3-Edge's robot-control resolution). The "
                         "encoder budget is 24,300 tokens and we are nowhere near "
                         "it, so raising this is untested headroom.")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Polls per timestep, averaged. Reduces variance at N x cost.")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="How far above baseline the score must rise to count as "
                         "the second state.")
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="Seconds the score must stay above the line. Rejects "
                         "single-poll spikes.")
    args = ap.parse_args()

    cfg = load_config()
    if args.max_side:
        cfg.sampling.frame_max_side = args.max_side
    client = OpenAI(base_url=cfg.model.base_url, api_key=cfg.model.api_key,
                    timeout=cfg.model.request_timeout_s)

    print(f"clip   {args.video.rsplit('/', 1)[-1]}")
    for i, s in enumerate(args.states):
        print(f"  ({chr(97 + i)}) {s}")
    if args.truth:
        print(f"truth  {args.truth[0]:.1f}-{args.truth[1]:.1f}s")
    print()

    # The reported score is P(last state) - P(first state), order-averaged. One
    # signed number per timestep: negative means the first state, positive the
    # second, and the event is where it crosses zero.
    a, b = args.states[0], args.states[-1]
    print(f"frame  {cfg.sampling.frame_max_side}px max side, {args.span:.0f}s per poll")
    if args.crop:
        print(f"crop   {args.crop}  (fractional, applied at native resolution, "
              f"then resized to {cfg.sampling.frame_max_side}px)")
    if args.repeat > 1:
        print(f"repeat {args.repeat} polls per timestep, averaged")
    print(f"score = P({b[:24]}) - P({a[:24]}), averaged over both option orders\n")
    print(f"     {'t':>7}  {'P(' + a[:18] + ')':>24}  {'P(' + b[:18] + ')':>24}  score")

    series = []
    t = args.start
    while t < args.end:
        # With a crop, decode at native resolution so the crop has pixels to
        # work with; without one, downscale during decode as usual.
        _, frames = sample_frames(
            args.video, fps=cfg.sampling.fps,
            max_side=(4096 if args.crop else cfg.sampling.frame_max_side),
            overlay=False, start_s=t, end_s=t + args.span)
        if not frames:
            break
        frames = crop_frames(frames, args.crop, cfg.sampling.frame_max_side)
        pr = poll(client, cfg.model.name, frames, args.states,
                  debug=(t == args.start), repeat=args.repeat)
        pa, pb = pr.get(a, float("nan")), pr.get(b, float("nan"))
        score = pb - pa
        inside = "*" if args.truth and args.truth[0] <= t <= args.truth[1] else " "
        bar = "#" * int(abs(score) * 20)
        print(f" {inside} t={t:>5.1f}s  {pa:>24.2f}  {pb:>24.2f}  {score:+.2f} {bar}")
        series.append((t, score))
        t += args.step

    print("\n* = inside the labelled event window.")
    if not series:
        return

    # Zero is the wrong line. Debiasing removes the OPTION-ORDER bias, but a
    # residual preference for one state remains -- on the first clip the resting
    # level was -0.21, not 0, so a real excursion never crossed zero. What marks
    # the event is departure from the clip's own baseline.
    vals = sorted(s for _, s in series)
    baseline = vals[len(vals) // 2]          # median: the scene's usual state
    line = baseline + args.margin
    print(f"baseline {baseline:+.2f} (median)   threshold {line:+.2f} "
          f"(+{args.margin:.2f})   dwell {args.dwell:.0f}s")

    # Intervals above the line, kept only if they last long enough. A single poll
    # above threshold is noise; a state that persists is a state.
    runs, cur = [], None
    for t, sc in series:
        if sc >= line and cur is None:
            cur = t
        elif sc < line and cur is not None:
            runs.append((cur, t))
            cur = None
    if cur is not None:
        runs.append((cur, series[-1][0] + args.step))
    runs = [(a, b) for a, b in runs if b - a >= args.dwell]

    print()
    if not runs:
        rng = vals[-1] - vals[0]
        print(f"no interval above baseline for {args.dwell:.0f}s. score range "
              f"{rng:.2f} — if that is small the model is not distinguishing "
              f"these states at all.")
        return

    for a, b in runs:
        line_out = f"detected  {a:.1f}-{b:.1f}s  ({args.states[-1]})"
        if args.truth:
            ts, te = args.truth
            inter = max(0.0, min(b, te) - max(a, ts))
            union = max(b, te) - min(a, ts)
            line_out += f"   tIoU {inter / union:.2f} vs label {ts:.1f}-{te:.1f}s"
        print(line_out)

    if args.truth:
        print()
        print("A state interval answers 'when was it open', while the hand label "
              "answers 'when did the opening happen'. A late end is expected and "
              "is a difference in question, not an error.")


if __name__ == "__main__":
    main()
