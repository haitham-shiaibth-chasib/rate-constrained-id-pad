"""Clip-level forensic features.

Optically variable devices change colour with the viewing angle, so a genuine
hologram shows chroma that varies *over time* at fixed document positions, while
print and screen copies stay static. ``chroma_dynamics_map`` measures that per pixel.
``clip_descriptor`` adds per-frame colour/texture statistics so that single-frame
transmission (K = 1) can still be scored.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

WORK_SIZE = (160, 112)  # (width, height) analysis grid; ROIs are resized to it


def _as_array(im: Image.Image | np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    im = im.convert("RGB")
    if size is not None and im.size != tuple(size):
        im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _cbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return cb, cr


def chroma_dynamics_map(frames: list, out_size: tuple[int, int] = WORK_SIZE) -> np.ndarray:
    """Per-pixel temporal standard deviation of chroma (Cb, Cr). HxW, >= 0."""
    arr = np.stack([_as_array(f, out_size) for f in frames])  # T,H,W,3
    cb, cr = _cbcr(arr)
    return np.sqrt(cb.var(axis=0) + cr.var(axis=0))


def dynamics_score(frames: list, top_fraction: float = 0.05) -> float:
    """Training-free bona fide score: mean of the most chroma-dynamic pixels.

    Returns 0 for a single frame (no temporal information).
    """
    if len(frames) < 2:
        return 0.0
    m = chroma_dynamics_map(frames).ravel()
    k = max(1, int(len(m) * top_fraction))
    return float(np.partition(m, -k)[-k:].mean())


def _frame_stats(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    lap = np.abs(4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform")
    lbp_hist = np.bincount(lbp.astype(int).ravel(), minlength=10)[:10] / lbp.size
    cb, cr = _cbcr(rgb)
    spec = np.abs(np.fft.rfft2(gray - gray.mean()))
    hf = spec[spec.shape[0] // 4:, spec.shape[1] // 2:].sum() / (spec.sum() + 1e-9)
    return np.concatenate([
        [sat.mean(), sat.std(), np.quantile(sat, 0.95), colorfulness, lap.mean(), lap.std(),
         cb.std(), cr.std(), hf],
        lbp_hist,
    ]).astype(np.float32)


N_FRAME_STATS = 19
N_DYNAMICS = 5


def clip_descriptor(frames: list) -> np.ndarray:
    """Fixed-length descriptor: mean & std of frame stats + chroma-dynamics quantiles."""
    arrs = [_as_array(f, WORK_SIZE) for f in frames]
    stats = np.stack([_frame_stats(a) for a in arrs])
    mean, std = stats.mean(0), stats.std(0)
    if len(arrs) >= 2:
        m = chroma_dynamics_map(arrs_to_images(arrs)).ravel()
        k = max(1, int(len(m) * 0.05))
        dyn = np.array([m.mean(), *np.quantile(m, [0.5, 0.9, 0.99]), np.partition(m, -k)[-k:].mean()])
    else:
        dyn = np.zeros(N_DYNAMICS)
    return np.concatenate([mean, std, dyn]).astype(np.float32)


def arrs_to_images(arrs: list[np.ndarray]) -> list[np.ndarray]:
    return [(a * 255).clip(0, 255).astype(np.uint8) for a in arrs]
