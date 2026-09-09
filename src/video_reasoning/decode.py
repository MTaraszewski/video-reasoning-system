"""Decode a video, sample frames, and burn an absolute timestamp onto each.

The overlay is the mechanism the whole system rests on. NVIDIA states it plainly
for Cosmos Reason 2: *"Our AI model recognizes timestamps added at the bottom of
each frame for accurate temporal localization."* The model has no clock — it reads
the time off the picture.

Two decisions here carry weight:

**Absolute, video-relative time.** We burn `t=73.250s`, never `t=1.250s into
window 6`. So a timestamp the model reports is directly usable, no per-window
remapping exists, and an entire class of off-by-one-window bugs cannot occur.

**Legibility is a variable, not a given.** Cosmos3-Edge runs at robot-control
resolution 640x360. Text that is obvious at 1080p can be a smear once downscaled
to that, and if the model cannot read the timestamp the design does not work. So
`overlay.font_scale` is tunable and `render_scale_sweep` exists to compare
settings by eye before any GPU time is spent.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from .errors import UnprocessableMedia

_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


@dataclass
class Frame:
    """One sampled frame: absolute time, and the image the model will see."""

    t: float
    image: Image.Image


@dataclass
class Probe:
    """Cheap metadata, read without decoding the whole file."""

    duration_s: float
    width: int
    height: int
    fps: float
    n_streams: int


def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A real scalable font, or fail loudly.

    PIL's built-in bitmap font is fixed at ~11px and does not scale. At 640x360
    that is unreadable after downscaling — silently falling back to it would
    disable the mechanism the design depends on while appearing to work.
    """
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


def has_scalable_font() -> bool:
    """True if a real TrueType font is available. See _load_font."""
    return any(Path(p).exists() for p in _FONT_PATHS)


def probe(path: str | Path) -> Probe:
    """Read metadata only. Runs in milliseconds, before any decode is attempted."""
    path = Path(path)
    try:
        container = av.open(str(path))
    except Exception as e:
        raise UnprocessableMedia(
            f"cannot open {path.name!r} as a video: {e}",
            fix="check the file is a video and not truncated",
        ) from e

    with container:
        if not container.streams.video:
            raise UnprocessableMedia(
                f"{path.name!r} contains no video stream.",
                fix="supply a video file, not audio or an image",
            )
        st = container.streams.video[0]
        tb = float(st.time_base) if st.time_base else None

        duration = None
        if st.duration and tb:
            duration = st.duration * tb
        elif container.duration:
            duration = container.duration / av.time_base

        fps = float(st.average_rate) if st.average_rate else 0.0
        return Probe(
            duration_s=round(float(duration or 0.0), 3),
            width=st.codec_context.width,
            height=st.codec_context.height,
            fps=round(fps, 3),
            n_streams=len(container.streams.video),
        )


def overlay_timestamp(
    img: Image.Image, t: float, *, font_scale: float = 0.045,
    fmt: str = "t={:.3f}s", position: str = "bottom-left",
) -> Image.Image:
    """Burn `t=SS.SSSs` onto the frame on a high-contrast background."""
    img = img.convert("RGB")
    w, h = img.size
    px = max(10, int(h * font_scale))
    font = _load_font(px)
    label = fmt.format(t)

    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), label, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = max(2, px // 4)

    if position == "bottom-left":
        x0, y0 = 0, h - th - 2 * pad
    elif position == "top-left":
        x0, y0 = 0, 0
    else:
        x0, y0 = 0, h - th - 2 * pad

    # Opaque background, not translucent: the model has to read this after
    # downscaling and JPEG compression, and contrast is what survives both.
    draw.rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad], fill=(0, 0, 0))
    draw.text((x0 + pad, y0 + pad - box[1]), label, fill=(255, 255, 255), font=font)
    return img


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale >= 1.0:
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def sample_frames(
    path: str | Path,
    fps: float,
    max_side: int,
    *,
    overlay: bool = True,
    font_scale: float = 0.045,
    fmt: str = "t={:.3f}s",
    position: str = "bottom-left",
    start_s: float = 0.0,
    end_s: float | None = None,
) -> tuple[float, list[Frame]]:
    """Sample at ~`fps`, resize, and burn the absolute timestamp on each frame.

    Every packet is decoded, but `to_image()` — the expensive part — runs only for
    frames actually kept. At 4 fps from 30 fps source that is ~7x less conversion
    work than converting everything and discarding most of it.
    """
    path = Path(path)
    try:
        container = av.open(str(path))
    except Exception as e:
        raise UnprocessableMedia(f"cannot open {path.name!r}: {e}") from e

    frames: list[Frame] = []
    step = 1.0 / fps
    next_t = start_s
    last_t = 0.0

    with container:
        if not container.streams.video:
            raise UnprocessableMedia(f"{path.name!r} has no video stream.")
        st = container.streams.video[0]
        st.thread_type = "AUTO"
        tb = float(st.time_base) if st.time_base else None

        for frame in container.decode(video=0):
            t = float(frame.pts * tb) if (frame.pts is not None and tb) else last_t
            last_t = t
            if end_s is not None and t > end_s:
                break
            if t + 1e-6 < next_t:
                continue

            img = _resize(frame.to_image(), max_side)
            if overlay:
                img = overlay_timestamp(
                    img, t, font_scale=font_scale, fmt=fmt, position=position
                )
            frames.append(Frame(t=round(t, 3), image=img))

            # Catch up past any sample points a gap skipped, so variable frame
            # rates and seeks do not produce a burst of near-identical frames.
            next_t += step
            while t + 1e-6 >= next_t:
                next_t += step

    duration = probe(path).duration_s or last_t
    return round(float(duration), 3), frames


def frame_to_data_url(img: Image.Image, quality: int = 85) -> str:
    """Encode for the chat-completions image_url field."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_scale_sweep(
    path: str | Path, t: float, max_side: int, scales: list[float]
) -> list[tuple[float, Image.Image]]:
    """Render one frame at several overlay font scales, for judging legibility.

    Whether the model can read the timestamp at the resolution it actually
    receives is not something to assume. This makes it a thing you look at,
    on a laptop, before renting anything.
    """
    _, frames = sample_frames(
        path, fps=1000.0, max_side=max_side, overlay=False,
        start_s=max(0.0, t - 0.05), end_s=t + 0.05,
    )
    if not frames:
        raise UnprocessableMedia(f"no frame found near t={t}s")
    base = frames[0].image
    return [
        (s, overlay_timestamp(base.copy(), frames[0].t, font_scale=s)) for s in scales
    ]
