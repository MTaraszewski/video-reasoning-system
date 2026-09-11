"""Does the reported timestamp follow the EVENT, or the WINDOW?

The two-stage eval produced a pattern that decides how every other number should
be read: 11 of 12 predictions came from window 0 and all landed at 10.0-12.0s --
the final two seconds of that window -- whatever the question asked. The one
prediction from another window (81-93s) landed at 88-93s. Again the tail.

Two explanations fit that, and they have opposite consequences:

  A. The model reads the burned-in timestamps and localises the event. The
     clustering is a coincidence of these clips.
  B. The model reports the end of the frame sequence it was shown. The absolute
     value then tracks whichever window it was given, and carries no information
     about the event at all.

If B, the timestamp overlay -- the mechanism this whole system rests on -- is not
grounding events for this model, and that explains the near-zero scores far better
than "it cannot detect events".

The test: ask the SAME question about the SAME event, in windows placed at
different offsets. Under A the answer stays near the event. Under B it slides
with the window.

    python scripts/window_offset_test.py \
      --video /data/eval/2018-03-07.16-50-01.16-55-01.admin.G326.r13.mp4 \
      --query "a person opens a building door" --truth 3.0 5.733
"""
from __future__ import annotations

import argparse

from video_reasoning.backends import make_backend
from video_reasoning.backends.base import ExtractRequest
from video_reasoning.backends.prompts import get as get_prompt
from video_reasoning.config import load_config
from video_reasoning.decode import sample_frames
from video_reasoning.windows import Window


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--truth", nargs=2, type=float, required=True,
                    metavar=("START", "END"))
    ap.add_argument("--offsets", default="0,3,9,15",
                    help="Window start times, comma separated.")
    ap.add_argument("--window-s", type=float, default=12.0)
    ap.add_argument("--prompt", default="overlay")
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    cfg = load_config()
    be = make_backend(cfg, args.backend)
    if hasattr(be, "check"):
        be.check()
    variant = get_prompt(args.prompt)
    ts, te = args.truth

    print(f"clip   {args.video.rsplit('/', 1)[-1]}")
    print(f"query  {args.query!r}")
    print(f"truth  {ts:.1f}-{te:.1f}s\n")
    print(f"{'window':<14}{'contains':<10}{'reported':<26}{'near end?':<10}evidence")
    print("-" * 104)

    rows = []
    for off in [float(x) for x in args.offsets.split(",") if x.strip()]:
        lo, hi = off, off + args.window_s
        _, frames = sample_frames(
            args.video, fps=cfg.sampling.fps,
            max_side=cfg.sampling.frame_max_side, overlay=cfg.overlay.enabled,
            font_scale=cfg.overlay.font_scale, fmt=cfg.overlay.format,
            position=cfg.overlay.position, start_s=lo, end_s=hi,
        )
        if len(frames) > cfg.sampling.max_frames_per_window:
            step = len(frames) / cfg.sampling.max_frames_per_window
            frames = [frames[min(int(i * step), len(frames) - 1)]
                      for i in range(cfg.sampling.max_frames_per_window)]
        w = Window(index=0, start_s=lo, end_s=hi, frames=frames)
        res = be.extract(ExtractRequest(
            window=w, query=args.query,
            system_prompt=variant.system,
            user_prompt=variant.user(args.query, lo, hi),
            video=args.video.rsplit("/", 1)[-1], prompt_variant=variant.name,
        ))
        # Does the window even contain the event? Offsets are chosen so some do
        # and some do not: under explanation A the answer should disappear when
        # the event leaves the window, under B it should keep arriving.
        contains = "yes" if (ts < hi and te > lo) else "no"
        rep = ", ".join(f"{e.start_s:.1f}-{e.end_s:.1f}" for e in res.events) or "-"
        # "Near end" = the reported start sits in the last quarter of the window.
        near = "-"
        if res.events:
            frac = (res.events[0].start_s - lo) / max(1e-6, hi - lo)
            near = "YES" if frac >= 0.75 else f"no ({frac:.0%})"
        ev = (res.events[0].evidence[:34] if res.events else (res.error or ""))
        print(f"{lo:>5.1f}-{hi:<7.1f}{contains:<10}{rep:<26}{near:<10}{ev}")
        rows.append((lo, hi, contains, [(e.start_s, e.end_s) for e in res.events]))

    # The verdict, stated rather than left to the reader.
    answered = [r for r in rows if r[3]]
    print()
    if not answered:
        print("No window produced an answer — inconclusive; the model declined "
              "throughout, which is the behaviour seen on 13 of 15 real-clip "
              "probe cases.")
        return
    # A slide needs at least two windows at DIFFERENT offsets. With one answer
    # the ratio is 0/0, and the 1e-6 guard below turned that into a confident
    # "0.00 -- tracks the EVENT" on a single data point. Refusing to conclude is
    # the whole job of this script; a statistic computed from one sample is not
    # a weaker conclusion, it is no conclusion.
    offsets = [r[0] for r in answered]
    if len(answered) < 2 or max(offsets) - min(offsets) < 1e-6:
        lo, hi, _, evs = answered[0]
        frac = (evs[0][0] - lo) / max(1e-6, hi - lo)
        print(f"Only {len(answered)} window answered — no slide can be computed.")
        print(f"  that one answer sat {frac:.0%} into its window "
              f"({evs[0][0]:.1f}-{evs[0][1]:.1f}s of {lo:.1f}-{hi:.1f}s)")
        print("  Suggestive, not conclusive. Gather more answers: more offsets, "
              "more queries, or a clip the model answers on more often.")
        return

    starts = [r[3][0][0] for r in answered]
    slide = (max(starts) - min(starts)) / (max(offsets) - min(offsets))
    print(f"reported start moved {max(starts) - min(starts):.1f}s across "
          f"{max(offsets) - min(offsets):.1f}s of window movement  "
          f"(slide = {slide:.2f})")
    if slide > 0.7:
        print("  -> tracks the WINDOW, not the event. The burned-in timestamp is "
              "not grounding this model: it reports where it was looking, not "
              "when the event happened.")
    elif slide < 0.3:
        print("  -> tracks the EVENT. The overlay mechanism works; the near-zero "
              "eval scores are a detection problem, not a grounding one.")
    else:
        print("  -> mixed. Neither explanation is clean on this sample; widen "
              "--offsets or try a second clip before concluding.")


if __name__ == "__main__":
    main()
