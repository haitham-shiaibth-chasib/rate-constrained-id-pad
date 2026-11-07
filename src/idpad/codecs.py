"""Encode images under a hard byte budget with JPEG, WebP or AVIF (Pillow >= 11.3).

The search finds the highest quality whose encoded size is <= budget. If even the
lowest quality is too large, the image is downscaled step by step. Every result
reports the exact payload size, so all comparisons are made at equal (measured) bytes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from PIL import Image, features

CODECS = ("jpeg", "webp", "avif")
SUBSAMPLINGS = ("4:2:0", "4:4:4")


@dataclass(frozen=True)
class Encoded:
    data: bytes
    codec: str
    quality: int
    scale: float
    subsampling: str
    size: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def decode(self, original_size: tuple[int, int] | None = None) -> Image.Image:
        """Decode and, if it was downscaled, upsample back to the original size."""
        im = Image.open(io.BytesIO(self.data)).convert("RGB")
        if original_size is not None and im.size != tuple(original_size):
            im = im.resize(original_size, Image.BICUBIC)
        return im


def available_codecs() -> list[str]:
    out = ["jpeg"]
    if features.check("webp"):
        out.append("webp")
    if features.check("avif"):
        out.append("avif")
    return out


def encode(im: Image.Image, codec: str, quality: int, subsampling: str = "4:2:0") -> bytes:
    buf = io.BytesIO()
    im = im.convert("RGB")
    if codec == "jpeg":
        ss = {"4:4:4": 0, "4:2:0": 2}[subsampling]
        im.save(buf, "JPEG", quality=int(quality), subsampling=ss, optimize=True)
    elif codec == "webp":
        # Lossy WebP is always 4:2:0; ``subsampling`` is ignored (recorded for bookkeeping).
        im.save(buf, "WEBP", quality=int(quality), method=4)
    elif codec == "avif":
        im.save(buf, "AVIF", quality=int(quality), subsampling=subsampling, speed=6)
    else:
        raise ValueError(f"unknown codec {codec}")
    return buf.getvalue()


def encode_to_budget(
    im: Image.Image,
    codec: str,
    budget_bytes: int,
    subsampling: str = "4:2:0",
    q_min: int = 1,
    q_max: int = 95,
    min_side: int = 32,
    scale_step: float = 0.8,
) -> Encoded | None:
    """Largest quality (then largest scale) that fits ``budget_bytes``. None if impossible."""
    im = im.convert("RGB")
    scale = 1.0
    while True:
        w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
        if min(w, h) < min_side:
            return None
        cur = im if scale == 1.0 else im.resize((w, h), Image.LANCZOS)
        lo_bytes = encode(cur, codec, q_min, subsampling)
        if len(lo_bytes) <= budget_bytes:
            lo, hi, best_q, best = q_min, q_max, q_min, lo_bytes
            while lo <= hi:
                mid = (lo + hi) // 2
                b = encode(cur, codec, mid, subsampling)
                if len(b) <= budget_bytes:
                    best_q, best, lo = mid, b, mid + 1
                else:
                    hi = mid - 1
            return Encoded(best, codec, best_q, scale, subsampling, (w, h))
        scale *= scale_step
