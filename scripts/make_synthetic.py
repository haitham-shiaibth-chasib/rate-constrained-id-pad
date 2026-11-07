"""Tiny synthetic stand-in for MIDV-Holo used ONLY to test the code path.

Results on this data mean nothing scientifically. Layout and split files mimic
Zenodo 17079529 (cropped ROIs) and the public k-fold splits.
"""
import argparse
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 240, 160


def card(rng, ident):
    base = np.full((H, W, 3), 225, np.uint8)
    base[..., 0] -= 10 * (ident % 3)
    im = Image.fromarray(base)
    d = ImageDraw.Draw(im)
    for y in range(0, H, 6):  # guilloche-like lines
        d.line([(0, y), (W, y + 8 * np.sin(y / 9))], fill=(200, 205, 215), width=1)
    d.ellipse([20, 30, 90, 120], fill=tuple(int(v) for v in rng.integers(90, 160, 3)))  # face
    for i in range(4):
        d.rectangle([110, 30 + 22 * i, 110 + rng.integers(60, 110), 40 + 22 * i], fill=(60, 60, 70))
    return im


def holo(im, t, dynamic=True):
    arr = np.asarray(im).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((xx - 150) ** 2 / 50 ** 2 + (yy - 110) ** 2 / 35 ** 2) < 1
    hue = ((xx + yy) / 120.0 + (t / 8.0 if dynamic else 0.0)) % 1.0
    rgb = np.stack(np.vectorize(lambda h: colorsys.hsv_to_rgb(h, 0.6, 0.95))(hue), -1) * 255
    arr[mask] = 0.5 * arr[mask] + 0.5 * rgb[mask]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def jitter(im, rng):
    arr = np.asarray(im).astype(np.float32) * rng.uniform(0.9, 1.1) + rng.normal(0, 3, (H, W, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--frames", type=int, default=12)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(a.out) / "cropped"
    names = [f"ID/id{m:02d}_{i:02d}_{c:02d}" for m in range(1, 4) for i in range(1, 6) for c in range(1, 3)]
    kinds = {"bona": None, "copy_without_holo": "none", "pseudo_holo_copy": "static",
             "photo_holo_copy": "photo", "photo_replacement": "dynamic_swap"}
    for n in names:
        ident = int(n.split("_")[1])
        c = card(rng, ident)
        snap = holo(c, 3)
        for kind, mode in kinds.items():
            d = root / ("origins" if kind == "bona" else f"fraud/{kind}") / n
            d.mkdir(parents=True, exist_ok=True)
            for t in range(a.frames):
                if mode is None:
                    im = holo(c, t)
                elif mode == "none":
                    im = c
                elif mode == "static":
                    im = holo(c, 0, dynamic=False)
                elif mode == "photo":
                    im = snap.filter(ImageFilter.GaussianBlur(0.8))
                else:
                    im = holo(card(np.random.default_rng(ident + 99), ident), t)
                jitter(im, rng).save(d / f"img_{t + 1:04d}.jpg", quality=92)
    # 2 folds split by identity
    split = Path(a.out) / "splits"
    for k in range(2):
        test_ids, val_ids = {1 + 2 * k, 2 + 2 * k}, {5}
        parts = {"train": [], "val": [], "test": []}
        for n in names:
            i = int(n.split("_")[1])
            parts["test" if i in test_ids else "val" if i in val_ids else "train"].append(n)
        sd = split / f"k{k}" / "simple"; sd.mkdir(parents=True, exist_ok=True)
        for p, v in parts.items():
            (sd / f"{p}.txt").write_text("\n".join(v) + "\n")
        for kind in list(kinds)[1:]:
            fd = split / f"k{k}" / "fraud" / kind; fd.mkdir(parents=True, exist_ok=True)
            (fd / "train.txt").write_text("\n".join(f"{n}/list.lst" for n in parts["train"] + parts["val"]) + "\n")
            (fd / "test.txt").write_text("\n".join(f"{n}/list.lst" for n in parts["test"]) + "\n")
    print("synthetic data written to", a.out)


if __name__ == "__main__":
    main()
