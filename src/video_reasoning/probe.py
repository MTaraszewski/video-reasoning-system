"""The capability probe — characterise the model before trusting the pipeline.

The brief asks us to *"understand what it actually guarantees"* and to report
*"where the limits of their abilities are"*. It also asserts that Cosmos 3 Edge
"can localise events with timestamps **when prompted correctly**". That assertion
is treated here as a hypothesis with a stated test, not a specification.

Four questions, in order of how much they decide:

1. Does the model report times at all, in a form we can parse?
2. **By which mechanism** — reading timestamps burned onto frames, or something
   else? The brief does not say, and the Edge model card does not mention
   timestamps at all.
3. At what **precision floor** — the smallest boundary error achievable before
   any windowing is layered on?
4. How does that degrade with event duration, sampling rate and overlay size?

Question 3 is the one that makes every later number interpretable. Without it,
a mediocre tIoU is unattributable: model error and pipeline error cannot be told
apart. So the probe deliberately runs **one window covering the whole clip**
wherever the clip fits — no windowing, no merging, nothing between the model and
the measurement.

It runs on synthetic clips because their ground truth is exact by construction.
Measuring a precision floor against hand labels that are themselves fuzzy by
±0.3 s would measure the labeller.
"""
from __future__ import annotations

import json
import statistics as st
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import Backend, ExtractRequest
from .backends.prompts import VARIANTS, PromptVariant
from .config import Config
from .decode import probe as probe_video, sample_frames
from .windows import Window


@dataclass
class ProbeCase:
    """One clip with exactly-known ground truth."""

    video: str
    description: str
    events: list[tuple[float, float]]
    axis: str = "baseline"
    duration_s: float = 0.0


@dataclass
class ProbeObservation:
    """What the model did on one (case, prompt, fps) combination."""

    case: ProbeCase
    prompt: str
    fps: float
    reported: list[tuple[float, float]]
    parse_error: str | None
    rejected: int
    clamped: int
    latency_s: float
    frames_sent: int
    raw: str = ""

    @property
    def emitted(self) -> bool:
        return bool(self.reported)


def load_cases(labels_path: str | Path, data_dir: str | Path) -> list[ProbeCase]:
    """Read the synthetic labels file into probe cases."""
    labels_path, data_dir = Path(labels_path), Path(data_dir)
    cases: list[ProbeCase] = []
    for item in json.loads(labels_path.read_text()):
        events = [(e["start_s"], e["end_s"]) for e in item.get("events", [])]
        if not events:
            continue  # negatives are for false-positive rate, not boundary error
        cases.append(
            ProbeCase(
                video=str(data_dir / item["video"]),
                description=item["events"][0]["description"],
                events=events,
                axis=item.get("axis", "baseline"),
                duration_s=item.get("duration_s", 0.0),
            )
        )
    return cases


def observe(
    case: ProbeCase, backend: Backend, cfg: Config, variant: PromptVariant, fps: float
) -> ProbeObservation:
    """Run one case, one prompt, one sampling rate — as a single window.

    A single window is the point: it removes windowing and merging from the
    measurement, so whatever error remains belongs to the model.
    """
    meta = probe_video(case.video)
    duration, frames = sample_frames(
        case.video, fps=fps, max_side=cfg.sampling.frame_max_side,
        overlay=cfg.overlay.enabled, font_scale=cfg.overlay.font_scale,
        fmt=cfg.overlay.format, position=cfg.overlay.position,
    )
    # Cap frames so a long clip cannot blow the request; the cap is itself a
    # measured limit once the GPU session establishes it.
    if len(frames) > cfg.sampling.max_frames_per_window:
        step = len(frames) / cfg.sampling.max_frames_per_window
        frames = [frames[min(int(i * step), len(frames) - 1)]
                  for i in range(cfg.sampling.max_frames_per_window)]

    window = Window(index=0, start_s=0.0, end_s=duration, frames=frames)
    req = ExtractRequest(
        window=window, query=case.description,
        system_prompt=variant.system,
        user_prompt=variant.user(case.description, 0.0, duration),
    )
    t0 = time.time()
    res = backend.extract(req)
    return ProbeObservation(
        case=case, prompt=variant.name, fps=fps,
        reported=[(e.start_s, e.end_s) for e in res.events],
        parse_error=res.error,
        rejected=int(res.meta.get("rejected_out_of_window", 0))
                 + int(res.meta.get("rejected_no_frame", 0)),
        clamped=int(res.meta.get("clamped", 0)),
        latency_s=res.latency_s or round(time.time() - t0, 3),
        frames_sent=len(frames),
        raw=res.raw[:400],
    )


def boundary_errors(obs: ProbeObservation) -> list[float]:
    """Absolute error, in seconds, of each true boundary against its best match.

    Each ground-truth event is paired with the reported event whose start is
    nearest, then start and end errors are taken from that pair. Pairing on start
    rather than on overlap is deliberate: a model that reports the right moment
    with the wrong duration should score as a boundary error, not as a miss.
    """
    if not obs.reported:
        return []
    errs: list[float] = []
    for (ts, te) in obs.case.events:
        rs, re_ = min(obs.reported, key=lambda r: abs(r[0] - ts))
        errs.extend([abs(rs - ts), abs(re_ - te)])
    return errs


def summarise(observations: list[ProbeObservation]) -> dict:
    """Aggregate into the four questions, per prompt variant and sampling rate."""
    out: dict = {"by_prompt_fps": {}, "by_axis": {}}

    groups: dict[tuple[str, float], list[ProbeObservation]] = {}
    for o in observations:
        groups.setdefault((o.prompt, o.fps), []).append(o)

    for (prompt, fps), obs in sorted(groups.items()):
        errs = [e for o in obs for e in boundary_errors(o)]
        n = len(obs)
        out["by_prompt_fps"][f"{prompt}@{fps}fps"] = {
            "cases": n,
            "emitted_times": round(sum(o.emitted for o in obs) / n, 3) if n else 0.0,
            "parse_failures": sum(1 for o in obs if o.parse_error),
            "rejected_times": sum(o.rejected for o in obs),
            "clamped_times": sum(o.clamped for o in obs),
            "median_abs_error_s": round(st.median(errs), 3) if errs else None,
            "mean_abs_error_s": round(st.fmean(errs), 3) if errs else None,
            "p90_abs_error_s": (round(sorted(errs)[int(len(errs) * 0.9)], 3)
                                if len(errs) >= 10 else None),
            "mean_latency_s": round(st.fmean([o.latency_s for o in obs]), 3),
            "mean_frames_sent": round(st.fmean([o.frames_sent for o in obs]), 1),
        }

    by_axis: dict[str, list[float]] = {}
    for o in observations:
        by_axis.setdefault(o.case.axis, []).extend(boundary_errors(o))
    for axis, errs in sorted(by_axis.items()):
        out["by_axis"][axis] = {
            "boundaries": len(errs),
            "median_abs_error_s": round(st.median(errs), 3) if errs else None,
        }
    return out


def run_probe(
    cases: list[ProbeCase], backend: Backend, cfg: Config,
    prompts: list[str] | None = None, fps_values: list[float] | None = None,
    progress: bool = True,
) -> dict:
    """Sweep prompt variants and sampling rates over every case."""
    prompts = prompts or list(VARIANTS)
    fps_values = fps_values or [cfg.sampling.fps]

    observations: list[ProbeObservation] = []
    total = len(cases) * len(prompts) * len(fps_values)
    i = 0
    for fps in fps_values:
        for name in prompts:
            variant = VARIANTS[name]
            for case in cases:
                i += 1
                o = observe(case, backend, cfg, variant, fps)
                observations.append(o)
                if progress:
                    errs = boundary_errors(o)
                    med = f"{st.median(errs):.3f}s" if errs else "—"
                    print(f"  [{i}/{total}] {Path(case.video).name:22s} "
                          f"{name:8s} {fps}fps  "
                          f"{len(o.reported)} reported, err {med}"
                          + (f"  [{o.parse_error}]" if o.parse_error else ""))

    summary = summarise(observations)
    summary["backend"] = backend.describe()
    summary["stub"] = getattr(backend, "is_stub", False)
    summary["observations"] = [
        {
            "video": Path(o.case.video).name, "axis": o.case.axis,
            "description": o.case.description, "prompt": o.prompt, "fps": o.fps,
            "truth": o.case.events, "reported": o.reported,
            "abs_errors_s": [round(e, 3) for e in boundary_errors(o)],
            "parse_error": o.parse_error, "rejected": o.rejected,
            "clamped": o.clamped, "latency_s": o.latency_s,
            "frames_sent": o.frames_sent,
        }
        for o in observations
    ]
    return summary


def verdict(summary: dict) -> list[str]:
    """Plain-language findings. A negative result is a finding, not a failure."""
    lines: list[str] = []
    if summary.get("stub"):
        lines.append(
            "STUB BACKEND — these numbers describe a deterministic pattern "
            "generator, not a model. They prove the probe runs; they say nothing "
            "about any model's ability."
        )
        return lines

    best = None
    for key, s in summary["by_prompt_fps"].items():
        if s["median_abs_error_s"] is None:
            continue
        if best is None or s["median_abs_error_s"] < best[1]["median_abs_error_s"]:
            best = (key, s)

    if best is None:
        lines.append(
            "The model reported no usable times under any prompt or sampling rate. "
            "That is a finding: this model cannot ground events in time by the "
            "means tested, and the fallback backend should be measured next."
        )
        return lines

    key, s = best
    lines.append(f"Best combination: {key}")
    lines.append(f"  precision floor (median absolute boundary error): "
                 f"{s['median_abs_error_s']}s")
    if s["p90_abs_error_s"] is not None:
        lines.append(f"  p90 absolute boundary error: {s['p90_abs_error_s']}s")
    lines.append(f"  times emitted on {s['emitted_times']:.0%} of cases; "
                 f"{s['parse_failures']} parse failure(s)")
    if s["rejected_times"]:
        lines.append(
            f"  {s['rejected_times']} reported time(s) REJECTED as outside the "
            "footage shown — the model reported moments it could not have seen."
        )
    lines.append("")
    lines.append("Every later metric should be read against this floor: boundary "
                 "error below it belongs to the model, not to the pipeline.")
    return lines
