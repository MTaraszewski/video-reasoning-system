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
    distractor: str | None = None   # "colour" | "direction" | None
    extra: dict = field(default_factory=dict)


def _base(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([0, H - 40, W, H], fill=FLOOR)


def _box(draw: ImageDraw.ImageDraw, x: float, w: int, h: int, colour) -> None:
    draw.rectangle([x, H - 40 - h, x + w, H - 40], fill=colour)


# The axis name is used by BOTH the scene definitions and the renderer, so it is
# named once here. An earlier version had render() branch on "machine_stop" while
# the scene declared "absence_of_motion": the branch never fired, the machine was
# never drawn, and the clip silently became another moving-box clip. The axis the
# brief explicitly names — "the machine stops moving" — was never being tested.
ABSENCE_OF_MOTION = "absence_of_motion"


def render(scene: Scene, t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    _base(d)

    if scene.axis == ABSENCE_OF_MOTION:
        # A machine that runs, then STOPS. The event is the stillness, so the
        # element must be plainly moving outside the window and plainly frozen
        # inside it — the opposite polarity to every other scene here.
        stopped = any(s <= t <= e for s, e in scene.events)
        angle = 0.0 if stopped else (t * 2.2)
        cx, cy, r = W // 2, H // 2 - 10, 70
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(60, 60, 60), width=5)
        # Three spokes, so rotation is unambiguous rather than a symmetric blur.
        for k in range(3):
            a = angle + k * (2 * math.pi / 3)
            d.line([cx, cy, cx + r * math.cos(a), cy + r * math.sin(a)],
                   fill=(200, 40, 40) if k == 0 else (40, 40, 40), width=9)
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(30, 30, 30))
        return img

    for (start, end) in scene.events:
        if not (start <= t <= end):
            continue
        frac = (t - start) / max(1e-6, end - start)
        size = scene.extra.get("size", 60)
        x = -size + frac * (W + size)
        _box(d, x, size, size + 20, (215, 35, 35))

    if scene.distractor == "colour":
        # Differs from the query only in COLOUR: does the model check what the
        # object IS, or only that something moved?
        dx = (t * 40) % (W + 50) - 50
        _box(d, dx, 40, 40, (60, 110, 205))

    elif scene.distractor == "direction":
        # Differs only in DIRECTION. Same colour, same size, entering from the
        # RIGHT — so the query "a red box enters from the LEFT" is false, and the
        # only way to know that is to read the directional clause rather than
        # spotting a red box. A colour-only distractor cannot test this.
        size = 60
        x = W - (t * 90) % (W + size)
        _box(d, x, size, size + 20, (215, 35, 35))

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
          [(9.0, 15.0)], axis=ABSENCE_OF_MOTION),

    Scene("with-distractor", 20.0, "a red box enters from the left",
          [(7.0, 11.0)], axis="distractor", distractor="colour",
          extra={"size": 60}),

    # A hard negative: the query is false, but a box of the SAME colour and size
    # is moving. Only the direction differs. A model that answers "yes" here is
    # matching objects, not descriptions.
    Scene("negative", 15.0, "a red box enters from the left",
          [], axis="negative", distractor="direction"),
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
