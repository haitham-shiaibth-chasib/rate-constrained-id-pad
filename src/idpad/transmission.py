"""How a client spends an uplink byte budget on one verification attempt.

A transmission plan decides (i) how many frames K of the capture are sent,
(ii) the chroma subsampling, and (iii) optionally a forensic region of interest
that is sent as a second, higher-quality layer on top of a cheap base layer.

Strategies (all measured at equal total bytes):

* ``uniform``  : K frames, budget split evenly, 4:2:0.
* ``chroma``   : as uniform but 4:4:4, keeping colour detail (holograms are chromatic).
* ``roi``      : forensic-aware two-layer coding. A base layer of the whole frame plus a
                 high-quality layer of the forensic region, which is learned from the
                 *training* bona fide clips (where colour changes most over time).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

from .codecs import encode_to_budget

Box = tuple[float, float, float, float]  # left, top, right, bottom as fractions of width/height


@dataclass(frozen=True)
class Plan:
    strategy: str = "uniform"     # uniform | chroma | roi
    codec: str = "jpeg"
    k_frames: int = 1
    roi_share: float = 0.6        # fraction of each frame's bytes spent on the ROI layer
    roi_subsampling: str = "4:4:4"

    @property
    def subsampling(self) -> str:
        return "4:4:4" if self.strategy == "chroma" else "4:2:0"

    def tag(self) -> str:
        t = f"{self.strategy}-{self.codec}-K{self.k_frames}"
        if self.strategy == "roi":
            t += f"-r{self.roi_share:.2f}"
        return t

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Sent:
    frames: list[Image.Image]     # what the server reconstructs
    nbytes: int                   # exact payload
    feasible: bool


def _encode_frame(im: Image.Image, plan: Plan, budget: int, roi: Box | None) -> tuple[Image.Image, int] | None:
    if plan.strategy != "roi" or roi is None:
        enc = encode_to_budget(im, plan.codec, budget, plan.subsampling)
        if enc is None:
            return None
        return enc.decode(im.size), enc.nbytes

    w, h = im.size
    px = (int(roi[0] * w), int(roi[1] * h), max(int(roi[2] * w), int(roi[0] * w) + 8), max(int(roi[3] * h), int(roi[1] * h) + 8))
    px = (px[0], px[1], min(px[2], w), min(px[3], h))
    roi_budget = int(budget * plan.roi_share)
    base_budget = budget - roi_budget
    base = encode_to_budget(im, plan.codec, base_budget, "4:2:0")
    patch_im = im.crop(px)
    patch = encode_to_budget(patch_im, plan.codec, roi_budget, plan.roi_subsampling)
    if base is None or patch is None:
        return None
    out = base.decode(im.size)
    out.paste(patch.decode(patch_im.size), px[:2])
    return out, base.nbytes + patch.nbytes


def transmit(frames: list[Image.Image], plan: Plan, budget_bytes: int | None, roi: Box | None = None) -> Sent:
    """Simulate sending ``plan.k_frames`` evenly spaced frames under a total budget.

    ``budget_bytes=None`` means the reference condition (no further compression).
    """
    k = max(1, min(plan.k_frames, len(frames)))
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    chosen = [frames[i] for i in idx]
    if budget_bytes is None:
        return Sent(chosen, 0, True)
    per = budget_bytes // k
    out, total = [], 0
    for im in chosen:
        r = _encode_frame(im, plan, per, roi)
        if r is None:
            return Sent([], 0, False)
        out.append(r[0])
        total += r[1]
    return Sent(out, total, True)


def forensic_prior(clips_frames: list[list[np.ndarray]], size: tuple[int, int]) -> np.ndarray:
    """Average per-pixel temporal chroma variability over bona fide training clips.

    Each element of ``clips_frames`` is a list of HxWx3 uint8 RGB arrays (same clip).
    Returns an HxW map in [0, 1].
    """
    from .features import chroma_dynamics_map

    acc = np.zeros(size[::-1], dtype=np.float64)
    n = 0
    for frames in clips_frames:
        if len(frames) < 2:
            continue
        m = chroma_dynamics_map(frames, out_size=size)
        acc += m
        n += 1
    if n == 0:
        return acc
    acc /= n
    return acc / (acc.max() + 1e-12)


def prior_to_box(prior: np.ndarray, area_fraction: float = 0.25, min_side: int = 16) -> Box:
    """Smallest axis-aligned box that holds the top ``area_fraction`` most active pixels."""
    h, w = prior.shape
    thr = np.quantile(prior, 1 - area_fraction)
    ys, xs = np.nonzero(prior >= thr)
    if len(xs) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    # trim 2% outliers on each side so isolated pixels do not inflate the box
    x0, x1 = np.quantile(xs, [0.02, 0.98]).astype(int)
    y0, y1 = np.quantile(ys, [0.02, 0.98]).astype(int)
    x1, y1 = max(x1 + 1, x0 + min_side), max(y1 + 1, y0 + min_side)
    return (x0 / w, y0 / h, min(x1, w) / w, min(y1, h) / h)
