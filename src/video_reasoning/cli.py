"""Command-line interface — the primary way to use the service.

Increment 2a: the command surface exists and is discoverable, but the pipeline
behind it is not built yet. Each unimplemented command says so and exits
non-zero rather than pretending to work.
"""
from __future__ import annotations

import sys

import typer
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
