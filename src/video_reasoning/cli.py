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
    video: str = typer.Option(None, help="Clip with known ground truth."),
) -> None:
    """Characterise the model: can it ground events in time, and how precisely."""
    err_console.print(
        "[yellow]probe[/] not implemented yet — lands in increment 3, on the GPU box."
    )
    raise typer.Exit(1)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
