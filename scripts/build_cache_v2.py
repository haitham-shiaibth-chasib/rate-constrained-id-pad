"""Encode every sampled frame on the (codec, scale, quality) grid and store payload sizes
and MobileNetV3 embeddings of the decoded frames. One .npz per clip; safe to resume.

The v2 analysis (`idpad.study_v2`) then evaluates any budget / K / policy by lookup.
"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from idpad import data as D
from idpad.cnn import frame_embeddings

CODECS = ("jpeg", "webp")
SCALES = (1.0, 0.5, 0.25)
QUALITIES = (5, 10, 20, 30, 45, 60, 75, 90)
N_FRAMES = 16


def encode_decode(im: Image.Image, codec: str, scale: float, q: int) -> tuple[int, Image.Image]:
    w, h = im.size
    src = im if scale == 1.0 else im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    if codec == "jpeg":
        src.save(buf, "JPEG", quality=q, subsampling=2, optimize=True)
    else:
        src.save(buf, "WEBP", quality=q, method=4)
    n = buf.tell()
    dec = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if dec.size != (w, h):
        dec = dec.resize((w, h), Image.BICUBIC)
    return n, dec


def clips_to_cache(root: Path, attacks) -> list[D.Clip]:
    out = []
    for d in sorted((root / "origins").glob("*/*")):
        out.append(D.Clip(f"{d.parent.name}/{d.name}", d, 0, "bona_fide", D.list_frames(d)))
    for a in attacks:
        for d in sorted((root / "fraud" / a).glob("*/*")):
            out.append(D.Clip(f"{d.parent.name}/{d.name}", d, 1, a, D.list_frames(d)))
    return [c for c in out if len(c.frames) >= 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cropped_root", default="data/midvdynattack/cropped")
    ap.add_argument("--out", default="data/cache_v2")
    ap.add_argument("--attacks", nargs="*", default=list(D.MIDV_HOLO_ATTACKS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--codecs", nargs="*", default=list(CODECS))
    ap.add_argument("--scales", type=float, nargs="*", default=list(SCALES))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    clips = clips_to_cache(Path(a.cropped_root), a.attacks)
    print(f"{len(clips)} clips", flush=True)
    t0 = time.time()
    for i, c in enumerate(clips):
        f = out / c.attack_type / f"{c.name.replace('/', '__')}.npz"
        res = dict(np.load(f)) if f.exists() else {}
        todo = [(cd, s) for cd in a.codecs for s in a.scales if f"{cd}_{s}_emb" not in res]
        if res and not todo:
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        frames = [Image.open(p).convert("RGB") for p in D.sample_frames(c, N_FRAMES)]
        if "ref_emb" not in res:
            res["ref_emb"] = frame_embeddings(frames).astype(np.float16)
            res["ref_bytes"] = np.array([p.stat().st_size for p in D.sample_frames(c, N_FRAMES)])
        for codec, s in todo:
            sizes, decs = [], []
            for q in QUALITIES:
                row = [encode_decode(im, codec, s, q) for im in frames]
                sizes.append([n for n, _ in row])
                decs.extend(d for _, d in row)
            emb = frame_embeddings(decs).reshape(len(QUALITIES), len(frames), -1)
            res[f"{codec}_{s}_bytes"] = np.array(sizes, dtype=np.int32)
            res[f"{codec}_{s}_emb"] = emb.astype(np.float16)
        np.savez_compressed(f, **res)
        if i % 10 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{len(clips)} {el / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
