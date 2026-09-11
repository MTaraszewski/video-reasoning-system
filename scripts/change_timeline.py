"""Ask whether the scene CHANGED between two moments, instead of what it is.

State polling stalled for a measurable reason: every score sat near 0.5. The model
was being asked an absolute question -- "is the door open?" -- which needs a
calibrated notion of what open looks like for this door, in this light, at this
distance. It does not have one, so it hovers at chance and the answer comes out of
option-order bias.

A comparison needs no such notion. Show frames from before and after and ask which
half contains the change. Each poll carries its own reference, so there is no
baseline to estimate and no threshold to tune -- the two failure modes that made
the state version fragile.

The event is then where the change is reported, not where a level crosses a line.

    python scripts/change_timeline.py \
      --video /data/eval/...admin.G326.r13.mp4 \
      --change "the door opens" --truth 3.0 5.733 --end 20
"""
from __future__ import annotations

import argparse
import math

from openai import OpenAI

from video_reasoning.config import load_config
from video_reasoning.decode import frame_to_data_url, sample_frames


def ask(client, model, before, after, change, swap, debug=False):
    """P(change happened) from one A/B comparison.

    `swap` presents the halves in the opposite order and inverts the reading, so
    the pair of calls cancels the position bias that dominated state polling --
    there, the model answered "(b)" whichever label sat second.
    """
    first, second = (after, before) if swap else (before, after)
    letters = ["a", "b"]
    content = [{"type": "text", "text":
                f"You are shown two short sequences of frames from the same fixed "
                f"camera, in order: sequence (a) then sequence (b).\n\n"
                f"In which sequence does this happen: \"{change}\"?\n\n"
                f"(a) in the first sequence\n(b) in the second sequence\n\n"
                f"Answer with one letter."}]
    for tag, fs in (("a", first), ("b", second)):
        content.append({"type": "text", "text": f"sequence ({tag}):"})
        for f in fs:
            content.append({"type": "image_url",
                            "image_url": {"url": frame_to_data_url(f.image)}})
    r = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=4, logprobs=True, top_logprobs=10,
        extra_body={"structured_outputs": {"choice": letters}},
        messages=[{"role": "system", "content":
                   "You compare two sequences of video frames and say which one "
                   "contains a described change. Answer with a single letter."},
                  {"role": "user", "content": content}],
    )
    if debug:
        print(f"    [raw={(r.choices[0].message.content or '')[:20]!r} "
              f"finish={r.choices[0].finish_reason}]")
    try:
        lp = {t.token.strip().lower(): t.logprob
              for t in r.choices[0].logprobs.content[0].top_logprobs}
        pa, pb = math.exp(lp.get("a", -60.0)), math.exp(lp.get("b", -60.0))
        if pa + pb <= 0:
            return float("nan")
        p_second = pb / (pa + pb)
    except Exception:
        return float("nan")
    # Without a swap, "second" is `after`. With a swap it is `before`, so the
    # probability that the change is in the LATER window is one minus it.
    return 1.0 - p_second if swap else p_second


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--change", required=True,
                    help='The change, as a client would say it: "the door opens".')
    ap.add_argument("--truth", nargs=2, type=float, metavar=("START", "END"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=20.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--span", type=float, default=2.0,
                    help="Seconds of frames in each half of the comparison.")
    ap.add_argument("--max-side", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.max_side:
        cfg.sampling.frame_max_side = args.max_side
    client = OpenAI(base_url=cfg.model.base_url, api_key=cfg.model.api_key,
                    timeout=cfg.model.request_timeout_s)

    print(f"clip   {args.video.rsplit('/', 1)[-1]}")
    print(f"change {args.change!r}")
    if args.truth:
        print(f"truth  {args.truth[0]:.1f}-{args.truth[1]:.1f}s")
    print(f"frame  {cfg.sampling.frame_max_side}px, {args.span:.0f}s per half\n")
    print(f"     {'t':>7}  P(change is in the later half)")

    def frames_at(t0):
        _, fs = sample_frames(args.video, fps=cfg.sampling.fps,
                              max_side=cfg.sampling.frame_max_side,
                              overlay=False, start_s=t0, end_s=t0 + args.span)
        return fs

    series, t = [], args.start
    while t + 2 * args.span <= args.end:
        before, after = frames_at(t), frames_at(t + args.span)
        if not before or not after:
            break
        ps = [ask(client, cfg.model.name, before, after, args.change, sw,
                  debug=(t == args.start and not sw)) for sw in (False, True)]
        ps = [p for p in ps if p == p]
        if not ps:
            t += args.step
            continue
        p = sum(ps) / len(ps)
        # The change, if any, sits at the boundary between the two halves.
        edge = t + args.span
        inside = "*" if args.truth and args.truth[0] <= edge <= args.truth[1] else " "
        print(f" {inside} t={edge:>5.1f}s  {p:>5.2f}  {'#' * int(p * 30)}")
        series.append((edge, p))
        t += args.step

    if not series:
        return
    best_t, best_p = max(series, key=lambda x: x[1])
    print(f"\n* = the boundary falls inside the labelled event.")
    print(f"peak at t={best_t:.1f}s (p={best_p:.2f})")
    if args.truth:
        ts, te = args.truth
        hit = ts <= best_t <= te
        print(f"label {ts:.1f}-{te:.1f}s — peak is {'INSIDE' if hit else 'outside'}")
    vals = sorted(p for _, p in series)
    print(f"range {vals[-1] - vals[0]:.2f} — a flat column means the model is not "
          f"seeing the change at all, whatever it answers.")


if __name__ == "__main__":
    main()
