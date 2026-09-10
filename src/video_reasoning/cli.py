"""Command-line interface — the primary way to use the service.

Kept thin on purpose: parse arguments, call `find_events`, print JSON. Anything
with real logic belongs in the library, where it can be used without a terminal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from . import __version__
from .errors import VideoReasoningError

app = typer.Typer(
    add_completion=False,
    help="Find events in a video from a plain-language description.",
)
console = Console()
err_console = Console(stderr=True)


def _fail(e: VideoReasoningError) -> None:
    """One place where errors become output, so every failure looks the same."""
    err_console.print(f"[red]{type(e).__name__}[/] {e.message}")
    if e.fix:
        err_console.print(f"  [yellow]fix[/] {e.fix}")
    raise typer.Exit(e.exit_code)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"video-reasoning-system {__version__}")


@app.command()
def run(
    video: str = typer.Argument(..., help="Path to the video file."),
    query: list[str] = typer.Option(
        None, "-q", "--query",
        help='Plain-language description. Repeat for several, e.g. -q "a door opens".',
    ),
    queries: str = typer.Option(
        None, "--queries",
        help=('Several descriptions in one string, separated by ";". Exists because '
              'make and shell word-splitting mangle multi-word arguments.'),
    ),
    out: str = typer.Option(None, "-o", "--out", help="Write JSON here (default stdout)."),
    config_path: str = typer.Option(None, "-c", "--config", help="Config YAML."),
    backend: str = typer.Option("auto", help="auto | vllm | stub | replay."),
    model: str = typer.Option(None, help="Override the model id."),
    base_url: str = typer.Option(None, help="Override the endpoint."),
    fps: float = typer.Option(None, help="Override sampling fps."),
    window_s: float = typer.Option(None, help="Override window length."),
    stride_s: float = typer.Option(None, help="Override stride."),
    prompt: str = typer.Option(None, help="Prompt variant: overlay | native | terse."),
    record: str = typer.Option(None, help="Record every model exchange to this dir."),
    replay: str = typer.Option(None, help="Replay recorded exchanges from this dir."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress."),
) -> None:
    """Find events in a video matching one or more descriptions."""
    from .backends import make_backend
    from .config import load_config
    from .core import find_events
    from .errors import InvalidInput

    # Splitting happens here, not in make: `foreach` splits on whitespace, so a
    # multi-word description silently became one query per word — producing valid
    # JSON for the wrong question.
    wanted = list(query or [])
    if queries:
        wanted += [q.strip() for q in queries.split(";") if q.strip()]

    try:
        if not wanted:
            raise InvalidInput(
                "no descriptions given.",
                fix='use -q "a person enters through the door", or '
                    '--queries "first;second"',
            )
        cfg = load_config(
            config_path,
            **{
                "model.name": model,
                "model.base_url": base_url,
                "sampling.fps": fps,
                "windowing.window_s": window_s,
                "windowing.stride_s": stride_s,
            },
        )
        for w in cfg.warnings():
            err_console.print(f"[yellow]warning[/] {w}")

        be = make_backend(cfg, backend, record_dir=record, replay_dir=replay)

        # Confirm the endpoint serves the model we think it does, before spending
        # anything. A mismatch would attribute every result to the wrong model,
        # which is worse than an error because it looks like data.
        if hasattr(be, "check"):
            be.check()

        if be.is_stub:
            err_console.print(
                "[yellow]STUB BACKEND[/] results are synthetic and cannot be used "
                "as evidence; the eval harness will refuse to score them."
            )

        result = find_events(video, wanted, cfg, be,
                             prompt=prompt, progress=not quiet)
    except VideoReasoningError as e:
        _fail(e)

    text = json.dumps(result.model_dump(), indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
        err_console.print(
            f"\n{len(result.events)} event(s) from {result.run.model_calls} model "
            f"call(s) in {result.run.elapsed_s}s -> {out}"
        )
    else:
        print(text)


@app.command()
def frames(
    video: str = typer.Argument(..., help="Video to sample."),
    out: str = typer.Option("/out/frames", "-o", "--out", help="Directory for PNGs."),
    fps: float = typer.Option(None, help="Override sampling fps."),
    max_side: int = typer.Option(None, help="Override longest side."),
    limit: int = typer.Option(12, help="Stop after this many frames."),
    sweep: bool = typer.Option(False, help="Render one frame at several font scales."),
) -> None:
    """Dump sampled frames with burned-in timestamps, so you can LOOK at them.

    The point is legibility: Cosmos3-Edge sees 640x360, and if the timestamp is a
    smear at that size the localisation mechanism cannot work. Cheaper to find out
    here than on a metered GPU.
    """
    from .config import load_config
    from .decode import (has_scalable_font, probe as probe_video,
                         render_scale_sweep, sample_frames)

    try:
        cfg = load_config(**{"sampling.fps": fps, "sampling.frame_max_side": max_side})
        for w in cfg.warnings():
            err_console.print(f"[yellow]warning[/] {w}")
        if not has_scalable_font():
            err_console.print(
                "[red]FAIL[/] no scalable font found; the overlay would be "
                "unreadable after downscaling. Install fonts-dejavu-core."
            )
            raise typer.Exit(1)

        meta = probe_video(video)
        console.print(f"[bold]{video}[/]")
        console.print(f"  source   {meta.width}x{meta.height} @ {meta.fps} fps, "
                      f"{meta.duration_s}s")
        console.print(f"  sampling {cfg.sampling.fps} fps -> max side "
                      f"{cfg.sampling.frame_max_side}px")

        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)

        if sweep:
            for scale, img in render_scale_sweep(
                video, t=meta.duration_s / 2,
                max_side=cfg.sampling.frame_max_side,
                scales=[0.02, 0.03, 0.045, 0.06, 0.09],
            ):
                path = out_dir / f"scale-{scale:.3f}.png"
                img.save(path)
                console.print(f"  {img.size[0]}x{img.size[1]}  "
                              f"font_scale={scale:<6} -> {path}")
            console.print("\n[bold]Check the timestamp is readable at the size "
                          "the model receives.[/]")
            return

        duration, fs = sample_frames(
            video, fps=cfg.sampling.fps, max_side=cfg.sampling.frame_max_side,
            overlay=cfg.overlay.enabled, font_scale=cfg.overlay.font_scale,
            fmt=cfg.overlay.format, position=cfg.overlay.position,
            end_s=(limit / cfg.sampling.fps) if limit else None,
        )
        for f in fs[:limit]:
            f.image.save(out_dir / f"t{f.t:09.3f}.png")
        console.print(f"  wrote {min(limit, len(fs))} frame(s) to {out_dir}")
        console.print(f"  full sampling would give "
                      f"{int(duration * cfg.sampling.fps)} frames for {duration}s")
    except VideoReasoningError as e:
        _fail(e)


@app.command()
def plan(
    video: str = typer.Argument(..., help="Video to plan for."),
    query: list[str] = typer.Option(None, "-q", "--query", help="Descriptions."),
    queries: str = typer.Option(None, "--queries", help='Several, separated by ";".'),
    config_path: str = typer.Option(None, "-c", "--config"),
    fps: float = typer.Option(None, help="Override sampling fps."),
    window_s: float = typer.Option(None, help="Override window length."),
    stride_s: float = typer.Option(None, help="Override stride."),
) -> None:
    """Show what a run would cost, without running it or touching a model."""
    from .config import load_config
    from .core import check_budget, validate_request
    from .decode import probe as probe_video

    try:
        cfg = load_config(config_path, **{
            "sampling.fps": fps,
            "windowing.window_s": window_s,
            "windowing.stride_s": stride_s,
        })
        wanted = list(query or [])
        if queries:
            wanted += [q.strip() for q in queries.split(";") if q.strip()]
        path, wanted = validate_request(video, wanted, cfg)
        meta = probe_video(path)
        cost = check_budget(meta.duration_s, len(wanted), cfg)
    except VideoReasoningError as e:
        _fail(e)

    console.print(f"[bold]{path.name}[/]  {meta.duration_s}s  "
                  f"{meta.width}x{meta.height} @ {meta.fps} fps")
    console.print(f"  sampling      {cfg.sampling.fps} fps -> {cost['frames']} frames")
    console.print(f"  windows       {cost['windows']} "
                  f"({cfg.windowing.window_s}s window, {cfg.windowing.stride_s}s "
                  f"stride, {cfg.windowing.overlap_s}s overlap)")
    console.print(f"  descriptions  {cost['queries']}")
    console.print(f"  [bold]model calls   {cost['model_calls']}[/]  "
                  f"(limit {cfg.limits.max_model_calls})")


@app.command()
def probe(
    labels: str = typer.Option("/data/synthetic/labels.json", help="Labels with EXACT ground truth."),
    data_dir: str = typer.Option("/data/synthetic", help="Where the clips live."),
    out: str = typer.Option("/out/probe.json", "-o", "--out"),
    backend: str = typer.Option("auto", help="auto | vllm | stub | replay."),
    base_url: str = typer.Option(None, help="Override the endpoint."),
    model: str = typer.Option(None, help="Override the model id."),
    prompts: str = typer.Option("overlay,native,terse", help="Prompt variants to sweep."),
    fps: str = typer.Option("4", help="Sampling rates to sweep, comma separated."),
    excerpt_s: float = typer.Option(
        None, help="Decode only this many seconds around each event, instead of "
                   "the whole clip. Holds sampling density constant when clips "
                   "differ in length."),
    record: str = typer.Option(None, help="Record every exchange, for later replay."),
    replay: str = typer.Option(None, help="Replay recorded exchanges."),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Characterise the model: can it ground events in time, and how precisely.

    Runs against clips whose ground truth is exact by construction, as a single
    window per clip — so windowing and merging are removed from the measurement
    and whatever error remains belongs to the model.
    """
    from .backends import make_backend
    from .config import load_config
    from .probe import load_cases, run_probe, verdict

    try:
        cfg = load_config(**{"model.name": model, "model.base_url": base_url})
        be = make_backend(cfg, backend, record_dir=record, replay_dir=replay)
        if hasattr(be, "check"):
            be.check()

        cases = load_cases(labels, data_dir, excerpt_s=excerpt_s)
        if not cases:
            err_console.print(f"[red]FAIL[/] no labelled cases in {labels}")
            raise typer.Exit(1)

        console.print(f"[bold]Capability probe[/]  {len(cases)} case(s)")
        summary = run_probe(
            cases, be, cfg,
            prompts=[p.strip() for p in prompts.split(",") if p.strip()],
            fps_values=[float(f) for f in fps.split(",") if f.strip()],
            progress=not quiet,
        )
    except VideoReasoningError as e:
        _fail(e)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2))

    console.print("\n[bold]Results by prompt and sampling rate[/]")
    console.print(f"  {'combination':<22} {'emit':>5} {'median':>8} {'p90':>8} "
                  f"{'rej':>4} {'clamp':>6} {'frames':>7}")
    for key, s in summary["by_prompt_fps"].items():
        med = f"{s['median_abs_error_s']}s" if s["median_abs_error_s"] is not None else "—"
        p90 = f"{s['p90_abs_error_s']}s" if s["p90_abs_error_s"] is not None else "—"
        console.print(f"  {key:<22} {s['emitted_times']:>5.0%} {med:>8} {p90:>8} "
                      f"{s['rejected_times']:>4} {s['clamped_times']:>6} "
                      f"{s['mean_frames_sent']:>7.0f}")

    console.print("\n[bold]Boundary error by failure axis[/]")
    for axis, s in summary["by_axis"].items():
        med = f"{s['median_abs_error_s']}s" if s["median_abs_error_s"] is not None else "—"
        console.print(f"  {axis:<22} {med:>8}  ({s['boundaries']} boundaries)")

    console.print("\n[bold]Verdict[/]")
    for line in verdict(summary):
        console.print(f"  {line}")
    console.print(f"\n-> {out}")


@app.command()
def evaluate(
    labels: str = typer.Option("/data/synthetic/labels.json", help="Hand-labelled set."),
    data_dir: str = typer.Option("/data/synthetic", help="Where the clips live."),
    out: str = typer.Option("/out/eval.json", "-o", "--out"),
    backend: str = typer.Option("auto", help="auto | vllm | replay (NOT stub)."),
    base_url: str = typer.Option(None, help="Override the endpoint."),
    model: str = typer.Option(None, help="Override the model id."),
    prompt: str = typer.Option(None, help="Prompt variant."),
    fps: float = typer.Option(None, help="Override sampling fps."),
    window_s: float = typer.Option(None, help="Override window length."),
    stride_s: float = typer.Option(None, help="Override stride."),
    gpu_hourly: float = typer.Option(None, help="Instance $/hr, for cost per video-minute."),
    replay: str = typer.Option(None, help="Replay recorded exchanges."),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Run the labelled set and report defensible temporal metrics."""
    from .backends import make_backend
    from .config import load_config
    from .evaluate import run_eval

    try:
        cfg = load_config(**{
            "model.name": model, "model.base_url": base_url,
            "sampling.fps": fps,
            "windowing.window_s": window_s, "windowing.stride_s": stride_s,
        })
        be = make_backend(cfg, backend, replay_dir=replay)
        if hasattr(be, "check"):
            be.check()
        res = run_eval(labels, data_dir, cfg, be, prompt=prompt,
                       progress=not quiet, gpu_hourly=gpu_hourly)
    except VideoReasoningError as e:
        _fail(e)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))

    o = res["overall"]
    console.print(f"\n[bold]Overall[/]  {res['model']}  "
                  f"prompt={res['prompt']}  fps={res['sampling']['fps']}")
    console.print(f"  predictions {o['n_predictions']:<5} truths {o['n_truths']}")
    for thr in (0.3, 0.5, 0.7):
        console.print(f"  R@1 tIoU>={thr}      {o[f'R@1_tIoU{thr}']:.3f}")
    console.print(f"  mean tIoU          {o['mean_tIoU']:.3f}")
    console.print(f"  precision@0.5      {o['precision@0.5']:.3f}")
    console.print(f"  recall@0.5         {o['recall@0.5']:.3f}")
    console.print(f"  false positive rt  {o['false_positive_rate']:.3f}")
    mre = o["mean_relative_error"]
    console.print(f"  mean rel. error    "
                  f"{mre if mre is None else f'{mre:.3f}'}"
                  "   (NVIDIA target <0.30)")

    console.print("\n[bold]By failure axis[/]  — where does it break?")
    console.print(f"  {'axis':<20} {'R@1_0.5':>8} {'mIoU':>7} {'recall':>7} {'FP rate':>8}")
    for axis, m in res["by_axis"].items():
        console.print(f"  {axis:<20} {m['R@1_tIoU0.5']:>8.3f} {m['mean_tIoU']:>7.3f} "
                      f"{m['recall@0.5']:>7.3f} {m['false_positive_rate']:>8.3f}")

    c = res["cost"]
    console.print(f"\n[bold]Cost[/]  {c['model_calls']} calls, "
                  f"{c['s_per_video_minute']}s per video-minute")
    if c["usd_per_video_minute"] is not None:
        console.print(f"  ${c['usd_per_video_minute']}/video-minute "
                      f"at ${c['gpu_hourly_usd']}/hr")
    else:
        console.print(f"  [dim]{c['note']}[/]")
    console.print(f"\n-> {out}")


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
