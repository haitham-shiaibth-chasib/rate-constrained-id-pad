"""Video-codec baseline: code the 16 sampled frames as one H.264 stream instead of 16 stills.

For every clip, K and scale, the frames are encoded with libx264 over a CRF grid, SEI
units are stripped (a deployed encoder would not send them), the stream is decoded,
and MobileNetV3 embeddings of the decoded frames are stored with the stream size.
The analysis then picks, per budget, the best CRF that fits, exactly as for stills.

Usage: python scripts/build_cache_h264.py --out data/cache_h264
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from idpad import data as D
from idpad.cnn import frame_embeddings
from idpad.study_v2 import nested

CRFS = (24, 30, 36, 42, 48)  # the first 149 clips were also coded at 18 and 51; the analysis drops those levels
KS = (16,)
SCALES = (0.5, 0.25)
N_FRAMES = 16


def _even(x: int) -> int:
    return max(16, x - (x % 2))


def h264_roundtrip(frames: list[Image.Image], scale: float, crf: int) -> tuple[int, list[Image.Image]]:
    w, h = frames[0].size
    sw, sh = _even(round(w * scale)), _even(round(h * scale))
    raw = b"".join(f.resize((sw, sh), Image.LANCZOS).tobytes() for f in frames)
    enc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sw}x{sh}", "-r", "10",
         "-i", "pipe:", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-g", "1000", "-bsf:v", "filter_units=remove_types=6", "-f", "h264", "pipe:"],
        input=raw, capture_output=True, check=True).stdout
    dec = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "h264", "-i", "pipe:", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"],
        input=enc, capture_output=True, check=True).stdout
    n = sw * sh * 3
    out = [Image.frombytes("RGB", (sw, sh), dec[i * n:(i + 1) * n]).resize((w, h), Image.BICUBIC)
           for i in range(len(frames))]
    return len(enc), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cropped_root", default="data/midvdynattack/cropped")
    ap.add_argument("--out", default="data/cache_h264")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attacks", nargs="*", default=list(D.MIDV_HOLO_ATTACKS))
    ap.add_argument("--no_bona_fide", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from build_cache_v2 import clips_to_cache  # same clip list and frame sampling as the stills cache

    clips = clips_to_cache(Path(a.cropped_root), a.attacks)
    if a.no_bona_fide:
        clips = [c for c in clips if c.label == 1]
    if a.limit:
        clips = clips[:a.limit]
    print(f"{len(clips)} clips", flush=True)
    t0 = time.time()
    for i, c in enumerate(clips):
        f = Path(a.out) / c.attack_type / f"{c.name.replace('/', '__')}.npz"
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        frames = [Image.open(p).convert("RGB") for p in D.sample_frames(c, N_FRAMES)]
        res = {}
        for k in KS:
            sel = [frames[j] for j in nested(k)]
            for s in SCALES:
                sizes, decs = [], []
                for crf in CRFS:
                    n, d = h264_roundtrip(sel, s, crf)
                    sizes.append(n)
                    decs.extend(d)
                emb = frame_embeddings(decs).reshape(len(CRFS), k, -1)
                res[f"h264_k{k}_{s}_bytes"] = np.array(sizes, dtype=np.int32)
                res[f"h264_k{k}_{s}_emb"] = emb.astype(np.float16)
        np.savez_compressed(f, **res)
        if i % 10 == 0:
            print(f"{i + 1}/{len(clips)} {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
