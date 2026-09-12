"""Frame sampling, with the absolute time burned into the pixels.

Two things here are load-bearing.

**Seeking.** The previous decoder walked packets from frame zero on every call
and skipped forward to the span it wanted, which makes cost quadratic in clip
length: measured 0.26 s to reach t=0 and 3.69 s to reach t=118 on the same
two-minute clip, 235 s per clip across a full poll sweep. Seeking to the
keyframe before the span makes it flat. `seek=False` stays as an escape hatch
for containers whose demuxer lands in the wrong place.

**The overlay.** Each frame carries its own timestamp as pixels. This is what
lets the model be asked *what it sees* and never *when something happened*: a
time in the response is a transcription of something visible, not an estimate.
Measured on real footage, the model localises badly and asserts confidently,
so the architecture never gives it the chance.
"""
from __future__ import annotations

import io
import base64
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)

# Seek lands on the keyframe at or before the target. Back off a little further
# so the first wanted frame is never before the first decodable one.
_SEEK_BACKOFF_S = 1.0


class MediaError(RuntimeError):
    pass


@dataclass
class Frame:
    t: float
    image: Image.Image


@dataclass
class VideoInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    n_streams: int


def probe(path: str | Path) -> VideoInfo:
    """Metadata without decoding the file."""
    path = Path(path)
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                raise MediaError(f"{path.name!r} has no video stream")
            st = c.streams.video[0]
            dur = float(c.duration / av.time_base) if c.duration else (
                float(st.duration * st.time_base) if st.duration and st.time_base else 0.0)
            return VideoInfo(
                duration_s=round(dur, 3), width=st.codec_context.width,
                height=st.codec_context.height,
                fps=float(st.average_rate) if st.average_rate else 0.0,
                n_streams=len(c.streams),
            )
    except MediaError:
        raise
    except Exception as e:
        raise MediaError(f"cannot open {path.name!r}: {e}") from e


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    """A real scalable font, or fail loudly.

    PIL's bitmap default is ~11px at any size. A timestamp the model cannot read
    is worse than no timestamp: the pipeline still believes the frame is
    self-describing, and every returned time is then a guess.
    """
    for p in _FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, px)
            except Exception:
                continue
    raise MediaError(
        "no scalable font found, so the burned-in timestamp would be unreadably "
        "small. Install fonts-dejavu, or set overlay.enabled=false and accept "
        "that the model can no longer report a time it can see."
    )


def has_scalable_font() -> bool:
    try:
        _load_font(20)
        return True
    except MediaError:
        return False


def overlay_timestamp(img: Image.Image, t: float, *, font_scale: float = 0.045,
                      fmt: str = "t={:.3f}s", position: str = "bottom-left") -> Image.Image:
    img = img.convert("RGB")
    px = max(12, int(img.height * font_scale))
    font = _load_font(px)
    text = fmt.format(t)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    w, h, pad = x1 - x0, y1 - y0, max(3, px // 4)
    x = pad if "left" in position else img.width - w - pad * 2
    y = img.height - h - pad * 2 if "bottom" in position else pad
    # Solid plate behind it: a timestamp over a bright sky is not readable, and
    # an unreadable timestamp fails silently.
    d.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=(0, 0, 0))
    d.text((x - x0, y - y0), text, font=font, fill=(255, 255, 255))
    return img


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / max(w, h)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)


def sample_frames(
    path: str | Path,
    *,
    fps: float,
    max_side: int,
    start_s: float = 0.0,
    end_s: float | None = None,
    overlay: bool = True,
    font_scale: float = 0.045,
    fmt: str = "t={:.3f}s",
    position: str = "bottom-left",
    seek: bool = True,
    max_frames: int | None = None,
) -> list[Frame]:
    """Sample at ~`fps` over [start_s, end_s], resized, timestamped.

    `to_image()` is the expensive part and runs only for frames actually kept.
    """
    path = Path(path)
    try:
        container = av.open(str(path))
    except Exception as e:
        raise MediaError(f"cannot open {path.name!r}: {e}") from e

    frames: list[Frame] = []
    step = 1.0 / fps
    next_t = start_s

    with container:
        if not container.streams.video:
            raise MediaError(f"{path.name!r} has no video stream")
        st = container.streams.video[0]
        st.thread_type = "AUTO"
        tb = Fraction(st.time_base) if st.time_base else None

        if seek and start_s > 0 and tb:
            target = max(0.0, start_s - _SEEK_BACKOFF_S)
            try:
                container.seek(int(target / tb), stream=st, any_frame=False, backward=True)
            except Exception:
                # A demuxer that cannot seek is a slow path, not a failure.
                container.seek(0)

        last_t = start_s
        for frame in container.decode(video=0):
            t = float(frame.pts * tb) if (frame.pts is not None and tb) else last_t
            last_t = t
            if end_s is not None and t > end_s + 1e-6:
                break
            if t + 1e-6 < next_t:
                continue

            img = _resize(frame.to_image(), max_side)
            if overlay:
                img = overlay_timestamp(img, t, font_scale=font_scale, fmt=fmt,
                                        position=position)
            frames.append(Frame(t=round(t, 3), image=img))
            if max_frames and len(frames) >= max_frames:
                break

            # Catch up past sample points a gap skipped, so a variable frame
            # rate does not produce a burst of near-identical frames.
            next_t += step
            while t + 1e-6 >= next_t:
                next_t += step

    return frames


def frame_to_data_url(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
