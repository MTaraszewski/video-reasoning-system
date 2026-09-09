"""Generate synthetic clips with EXACT ground truth.

This is an instrument, not a fallback. It is the only source where the labels
carry zero error, which is what makes it possible to measure the model's
*precision floor* — the best boundary accuracy achievable before any windowing
is layered on. Every result from real footage is read against that number,
because without it model error and pipeline error cannot be told apart.

Each scene targets one failure axis. Descriptions are written the way a client
would phrase them, not as class names: the system is open-vocabulary, and
labelling with a taxonomy would test the wrong thing.

    python scripts/make_synthetic.py --out data/synthetic
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageDraw

W, H, FPS = 640, 360, 25
BG = (238, 238, 238)
FLOOR = (205, 205, 205)


@dataclass
class Scene:
    name: str
    duration_s: float
    description: str          # how a client would say it
    events: list[tuple[float, float]]
    axis: str                 # which failure axis this probes
    distractor: bool = False
    extra: dict = field(default_factory=dict)


def _base(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([0, H - 40, W, H], fill=FLOOR)


def _box(draw: ImageDraw.ImageDraw, x: float, w: int, h: int, colour) -> None:
    draw.rectangle([x, H - 40 - h, x + w, H - 40], fill=colour)


def render(scene: Scene, t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    _base(d)

    for (start, end) in scene.events:
        if not (start <= t <= end):
            continue
        frac = (t - start) / max(1e-6, end - start)

        if scene.axis == "machine_stop":
            # A wheel spins, then stops. The EVENT is the stillness.
            continue
        size = scene.extra.get("size", 60)
        x = -size + frac * (W + size)
        _box(d, x, size, size + 20, (215, 35, 35))

    if scene.axis == "machine_stop":
        # Spinning marker outside the event window; frozen inside it.
        stopped = any(s <= t <= e for s, e in scene.events)
        angle = 0.0 if stopped else (t * 4.0)
        cx, cy, r = W // 2, H // 2 - 20, 46
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(70, 70, 70), width=4)
        d.line(
            [cx, cy, cx + r * math.cos(angle), cy + r * math.sin(angle)],
            fill=(30, 30, 30), width=6,
        )

    if scene.distractor:
        # A similar object that never performs the event. Measures false positives.
        dx = (t * 40) % (W + 50) - 50
        _box(d, dx, 40, 40, (60, 110, 205))

    return img


def write_clip(scene: Scene, out_dir: Path) -> Path:
    path = out_dir / f"{scene.name}.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.time_base = Fraction(1, FPS)
    stream.options = {"crf": "20", "preset": "veryfast"}

    for i in range(int(scene.duration_s * FPS)):
        t = i / FPS
        frame = av.VideoFrame.from_image(render(scene, t))
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


# Each scene isolates one axis. Durations stay short here so the generator runs
# in seconds; the eval-set versions are 1-3 min, set by --scale.
SCENES = [
    Scene("box-crossing", 20.0, "a red box enters from the left",
          [(4.0, 8.0), (13.0, 16.5)], axis="baseline", extra={"size": 60}),

    Scene("short-event", 20.0, "a red box enters from the left",
          [(6.0, 6.2), (12.0, 12.15)], axis="short",
          extra={"size": 60}),

    Scene("long-event", 40.0, "a red box crosses the scene slowly",
          [(5.0, 32.0)], axis="long", extra={"size": 60}),

    Scene("machine-stop", 24.0, "the machine stops moving",
          [(9.0, 15.0)], axis="absence_of_motion"),

    Scene("with-distractor", 20.0, "a red box enters from the left",
          [(7.0, 11.0)], axis="distractor", distractor=True,
          extra={"size": 60}),

    Scene("negative", 15.0, "a red box enters from the left",
          [], axis="negative", distractor=True),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every duration and event time (e.g. 5 -> 1-3 min clips)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for base in SCENES:
        s = args.scale
        scene = Scene(
            name=base.name, duration_s=base.duration_s * s,
            description=base.description,
            events=[(a * s, b * s) for a, b in base.events],
            axis=base.axis, distractor=base.distractor, extra=base.extra,
        )
        path = write_clip(scene, out_dir)
        labels.append({
            "video": path.name,
            "axis": scene.axis,
            "duration_s": round(scene.duration_s, 3),
            "events": [
                {"description": scene.description,
                 "start_s": round(a, 3), "end_s": round(b, 3)}
                for a, b in scene.events
            ],
        })
        n = len(scene.events)
        print(f"  {path.name:20s} {scene.duration_s:6.1f}s  "
              f"{n} event(s)  axis={scene.axis}")

    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2))
    print(f"\nGround truth is exact by construction -> {labels_path}")


if __name__ == "__main__":
    main()
