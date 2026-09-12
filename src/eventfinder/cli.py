"""Command line for eventfinder.

    eventfinder plan   clip.mp4 -d "..."      what a run would cost, no calls
    eventfinder run    clip.mp4 -d "..."      batch: find events, write JSON
    eventfinder live   clip.mp4 -d "..."      stream a file as if it were a camera
    eventfinder replay exchanges.jsonl        re-derive from a recorded session

`plan` exists because cost should be inspectable before it is incurred rather
than discovered afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from .backends.replay import ReplayReasoner, summarise, write_exchanges
from .backends.vllm import VLLMReasoner
from .compile import RulesCompiler, compile_all
from .config import Config, load
from .pipeline import plan as plan_run, run as batch_run
from .stream import LiveFinder, frames_from_file

app = typer.Typer(add_completion=False, help=__doc__)


def _cfg(config: str | None, mode: str | None, media: str | None) -> Config:
    c = load(config) if config else Config()
    if mode:
        c.observe.mode = mode
    c.check()
    for w in c.warnings():
        typer.secho(f"  warning: {w}", fg="yellow", err=True)
    return c


def _reasoner(c: Config, media: str):
    return VLLMReasoner(c.model.base_url, c.model.name, api_key=c.model.api_key,
                        temperature=c.model.temperature, max_tokens=c.model.max_tokens,
                        media=media, timeout_s=c.model.request_timeout_s,
                        sample_fps=1.0 / c.observe.step_s if c.observe.mode == "per_bracket"
                        else c.sampling.fps)


@app.command()
def plan(video: str, description: list[str] = typer.Option(..., "-d", "--description"),
         config: str = typer.Option(None), mode: str = typer.Option(None)):
    """What the run would do, before a single model call."""
    c = _cfg(config, mode, None)
    probes, brackets, tasks, info = plan_run(video, description, c)
    typer.echo(f"video       {Path(video).name}  {info.duration_s:.1f}s  {info.width}x{info.height}")
    for p in probes:
        if p.expressible:
            typer.echo(f"  probe     {p.id} {p.subject:13s} {p.kind:9s} {p.states or p.direction or ''}")
        else:
            typer.secho(f"  REFUSED   {p.id} {p.description}: {p.reason}", fg="yellow")
    covered = sum(b.duration_s for b in brackets)
    typer.echo(f"brackets    {len(brackets)}  covering {covered:.0f}s "
               f"({covered / max(info.duration_s, 1e-9):.0%} of the clip)")
    typer.echo(f"calls       {len(tasks)}  (observe.mode={c.observe.mode}, "
               f"limit {c.limits.max_model_calls})")
    if len(tasks) > c.limits.max_model_calls:
        typer.secho("  over the call limit; this run would be refused", fg="red")


@app.command()
def run(video: str, description: list[str] = typer.Option(..., "-d", "--description"),
        out: str = typer.Option("out/events.json"), config: str = typer.Option(None),
        mode: str = typer.Option(None), media: str = typer.Option("video"),
        verify: bool = typer.Option(False, help="also ask the model for its own verdict"),
        record: str = typer.Option(None, help="write the exchange log here for replay")):
    """Find the described events and write the document."""
    c = _cfg(config, mode, media)
    r = _reasoner(c, media)
    ok, why = r.verify_model()
    if not ok:
        typer.secho(f"  model identity NOT verified: {why}", fg="red", err=True)
    with typer.progressbar(length=100, label="observing") as bar:
        state = {"n": 0}

        def prog(i, total, _task):
            want = int(100 * i / total)
            bar.update(want - state["n"])
            state["n"] = want

        doc = batch_run(video, description, c, r, verify=verify, on_progress=prog)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(doc.model_dump_json(indent=2))
    if record:
        typer.echo(f"  recorded    {write_exchanges(record, r.usage)}")
    typer.echo(f"  {len(doc.events)} events, {doc.run.calls} calls, "
               f"{doc.run.wall_time_s:.0f}s, mean latency {r.usage.mean_latency_s:.2f}s")
    for e in doc.events[:10]:
        typer.echo(f"    {e.confidence:.2f}  {e.start_s:7.2f}-{e.end_s:<7.2f} "
                   f"{'[' + e.partial + '] ' if e.partial != 'none' else ''}{e.description}")
    typer.echo(f"  -> {out}")


@app.command()
def live(video: str, description: list[str] = typer.Option(..., "-d", "--description"),
         config: str = typer.Option(None), media: str = typer.Option("video"),
         realtime: bool = typer.Option(False, help="play at the capture clock")):
    """Stream a file as if it were a camera."""
    c = _cfg(config, None, media)
    c.observe.mode = "per_bracket"
    probes = compile_all(description, RulesCompiler())
    r = _reasoner(c, media)

    def emit(e, verified):
        typer.echo(f"  {'VERIFIED' if verified else 'candidate'}  "
                   f"{e.start_s:7.2f}-{e.end_s:<7.2f} conf {e.confidence:.2f}  {e.description}")

    f = LiveFinder(probes, c, r, emit)
    last = 0.0
    for fr in frames_from_file(video, c, realtime=realtime):
        f.feed(fr)
        f.tick(fr.t)
        last = fr.t
    f.close(now=last)
    s = f.stats
    typer.echo(f"  {s.frames} frames, {s.candidates} candidates, {s.calls} calls, "
               f"{s.emitted} events, {s.dropped} dropped, "
               f"mean emit latency {s.mean_latency_s:.2f}s")


@app.command()
def replay(exchanges: str, video: str = typer.Option(...),
           description: list[str] = typer.Option(..., "-d", "--description"),
           out: str = typer.Option("out/replay.json"), config: str = typer.Option(None),
           mode: str = typer.Option("per_bracket")):
    """Re-derive from a recorded session. No GPU, no cost."""
    c = _cfg(config, mode, None)
    r = ReplayReasoner(exchanges)
    doc = batch_run(video, description, c, r)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(doc.model_dump_json(indent=2))
    typer.echo(json.dumps(summarise(exchanges), indent=2))
    typer.echo(f"  {len(doc.events)} events (result_class={doc.run.result_class}) -> {out}")


@app.command()
def schema(out: str = typer.Option("eventfinder.schema.json")):
    """Write the machine-readable output contract."""
    from .models import EventsDocument
    Path(out).write_text(json.dumps(EventsDocument.model_json_schema(), indent=2))
    typer.echo(f"  -> {out}")


if __name__ == "__main__":
    app()
