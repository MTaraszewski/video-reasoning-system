"""Command-line interface — the primary way to use the service.

Increment 2a: the command surface exists and is discoverable, but the pipeline
behind it is not built yet. Each unimplemented command says so and exits
non-zero rather than pretending to work.
"""
from __future__ import annotations

import sys

import typer
from pathlib import Path

from rich.console import Console

from . import __version__

app = typer.Typer(
    add_completion=False,
    help="Find events in a video from a plain-language description.",
)
console = Console()

_NOT_YET = "not implemented yet — see PLAN.md §7 for the increment that lands it"


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"video-reasoning-system {__version__}")


@app.command()
def run(
    video: str = typer.Argument(..., help="Path to the video file."),
    query: list[str] = typer.Option(
        ..., "-q", "--query", help="Event description. Repeat for several."
    ),
    out: str = typer.Option(None, "-o", "--out", help="Write JSON here."),
) -> None:
    """Find events in a video."""
    console.print(f"[yellow]run[/] {_NOT_YET} (increment 2g)")
    console.print(f"  video   : {video}")
    console.print(f"  queries : {list(query)}")
    console.print(f"  out     : {out or '<stdout>'}")
    raise typer.Exit(1)


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

    cfg = load_config(**{"sampling.fps": fps, "sampling.frame_max_side": max_side})
    for w in cfg.warnings():
        console.print(f"[yellow]warning[/] {w}")
    if not has_scalable_font():
        console.print("[red]FAIL[/] no scalable font found; the overlay would be "
                      "unreadable after downscaling. Install fonts-dejavu-core.")
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
        scales = [0.02, 0.03, 0.045, 0.06, 0.09]
        for scale, img in render_scale_sweep(
            video, t=meta.duration_s / 2, max_side=cfg.sampling.frame_max_side,
            scales=scales,
        ):
            path = out_dir / f"scale-{scale:.3f}.png"
            img.save(path)
            console.print(f"  {img.size[0]}x{img.size[1]}  font_scale={scale:<6} "
                          f"-> {path}")
        console.print("\n[bold]Open these and check the timestamp is readable "
                      "at the size the model receives.[/]")
        return

    duration, fs = sample_frames(
        video, fps=cfg.sampling.fps, max_side=cfg.sampling.frame_max_side,
        overlay=cfg.overlay.enabled, font_scale=cfg.overlay.font_scale,
        fmt=cfg.overlay.format, position=cfg.overlay.position,
        end_s=(limit / cfg.sampling.fps) if limit else None,
    )
    for f in fs[:limit]:
        path = out_dir / f"t{f.t:09.3f}.png"
        f.image.save(path)
    console.print(f"  wrote {min(limit, len(fs))} frame(s) to {out_dir}")
    console.print(f"  total sampled would be "
                  f"{int(duration * cfg.sampling.fps)} frames for {duration}s")


@app.command()
def probe(
    video: str = typer.Option(None, help="Clip with known ground truth."),
) -> None:
    """Characterise the model: can it ground events in time, and how precisely."""
    console.print(f"[yellow]probe[/] {_NOT_YET} (increment 3)")
    raise typer.Exit(1)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
