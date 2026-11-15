"""Table 13: computational cost of one attempt on the client and the server.

Times are wall-clock medians and 95th percentiles over repeated runs, one thread, on the machine
that runs the script (the paper: a 2-vCPU Intel Xeon at 2.1 GHz). Frames are synthetic
357 x 333 RGB images whose smooth texture shifts between frames; at JPEG quality 95 one frame
takes about 37 kB, against about 30 kB for the MIDV-Holo crops. Encoding runs in-process
(Pillow for WebP, PyAV/libx264 for H.264, the settings of Sec. 4.4), so no process start-up is
included; the start-up of an ffmpeg process is measured separately. The client searches the
quality or rate-factor grid by bisection (at most 3 encodes for 8 qualities or 5 rate factors).
The detector runs with random weights, which leaves its cost unchanged.

Run from the repository root:  python scripts/cost_benchmark.py --reps 50
"""
import argparse, io, json, platform, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, "src")
from idpad.network import Link

W, H = 357, 333
GRID = [2000, 4000, 8000, 16000, 32000, 64000]
QUALITIES = [5, 10, 20, 30, 45, 60, 75, 90]
CRFS = [24, 30, 36, 42, 48]


def frames(n=16, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(H // 8 + 4, W // 8 + 4, 3))
    base = np.asarray(Image.fromarray(((base - base.min()) / np.ptp(base) * 255).astype(np.uint8))
                      .resize((W + 32, H + 32), Image.BICUBIC), float)
    return [Image.fromarray(np.clip(base[k:k + H, 2 * k:2 * k + W] * (0.8 + 0.2 * np.cos(k / 3)), 0, 255).astype(np.uint8))
            for k in range(n)]


def timeit(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        a = time.perf_counter(); fn(); t.append(time.perf_counter() - a)
    t = np.array(t) * 1000
    return round(float(np.median(t)), 3), round(float(np.percentile(t, 95)), 3)


def webp(fr, k, scale, q):
    out = []
    for im in fr[::16 // k][:k]:
        b = io.BytesIO(); im.resize((round(W * scale), round(H * scale)), Image.BILINEAR).save(b, "WEBP", quality=q, method=4)
        out.append(b.getvalue())
    return out


def h264(fr, scale, crf):
    import av
    sw, sh = 2 * round(W * scale / 2), 2 * round(H * scale / 2)
    buf = io.BytesIO()
    with av.open(buf, "w", format="h264") as c:
        s = c.add_stream("libx264", rate=10)
        s.width, s.height, s.pix_fmt = sw, sh, "yuv420p"
        s.options = {"preset": "medium", "crf": str(crf), "g": "1000"}
        s.time_base = Fraction(1, 10)
        for im in fr:
            for p in s.encode(av.VideoFrame.from_image(im.resize((sw, sh), Image.BILINEAR))):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def bisect(grid, size_of, budget, higher_is_bigger=True):
    """Best grid value whose size fits; grid sorted so that size grows with index."""
    lo, hi, best, n = 0, len(grid) - 1, None, 0
    while lo <= hi:
        mid = (lo + hi) // 2; n += 1
        if size_of(grid[mid]) <= budget:
            best, lo = grid[mid], mid + 1
        else:
            hi = mid - 1
    return best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/paper/table11_cost.json")
    a = ap.parse_args()
    import torch, timm, av
    torch.set_num_threads(a.threads)
    fr = frames()
    res, info = {}, {}
    link = Link("x", 960, 300)
    res["Client: budget choice, Eq. (2)"] = timeit(lambda: link.budget_for_deadline(1.5, grid=GRID), a.reps * 20)
    # deployed 4 kB plans: H.264 stream of 16 frames at scale 0.25 (CA policy) and 2 or 4 WebP stills (adaptive policy)
    crfs_desc = sorted(CRFS, reverse=True)                     # size grows as the rate factor falls
    res["Client: H.264, 16 frames, scale 0.25, one encode"] = timeit(lambda: h264(fr, 0.25, 36), a.reps)
    res["Client: H.264, 16 frames, scale 0.25, rate-factor search for 4 kB"] = timeit(
        lambda: bisect(crfs_desc, lambda c: len(h264(fr, 0.25, c)), 4000), a.reps)
    res["Client: 2 WebP stills, scale 0.5, quality search for 4 kB"] = timeit(
        lambda: bisect(QUALITIES, lambda q: sum(map(len, webp(fr, 2, 0.5, q))), 4000), a.reps)
    res["Client: 4 WebP stills, scale 0.25, quality search for 4 kB"] = timeit(
        lambda: bisect(QUALITIES, lambda q: sum(map(len, webp(fr, 4, 0.25, q))), 4000), a.reps)
    info["h264_crf_found_4kB"], info["h264_encodes"] = bisect(crfs_desc, lambda c: len(h264(fr, 0.25, c)), 4000)
    info["webp2_q_found_4kB"], _ = bisect(QUALITIES, lambda q: sum(map(len, webp(fr, 2, 0.5, q))), 4000)
    info["webp4_q_found_4kB"], _ = bisect(QUALITIES, lambda q: sum(map(len, webp(fr, 4, 0.25, q))), 4000)
    t = []
    for _ in range(20):
        s0 = time.perf_counter(); subprocess.run(["ffmpeg", "-v", "error", "-version"], capture_output=True); t.append(time.perf_counter() - s0)
    info["ffmpeg_process_start_ms_median"] = round(float(np.median(t)) * 1000, 1)

    m = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=0).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    w = np.random.default_rng(0).normal(size=3840).astype(np.float32)
    stream = h264(fr, 0.25, info["h264_crf_found_4kB"] or 48)

    def decode_h264(data):
        with av.open(io.BytesIO(data), format="h264") as c:
            return [f.to_image().resize((W, H), Image.BILINEAR) for f in c.decode(video=0)]

    def score(sel):
        x = torch.from_numpy(np.stack([np.asarray(im.resize((224, 224), Image.BILINEAR), np.float32) / 255 for im in sel])).permute(0, 3, 1, 2)
        with torch.no_grad():
            e = m((x - mean) / std).numpy()
        k = len(sel)
        z = np.concatenate([e.mean(0), e.std(0), np.abs(np.diff(e, axis=0)).mean(0) if k > 1 else np.zeros(1280, np.float32)])
        return 1 / (1 + np.exp(-(z @ w)))
    stills = webp(fr, 2, 0.5, info["webp2_q_found_4kB"] or 5)
    res["Server: decode H.264 stream, embed and score 16 frames"] = timeit(lambda: score(decode_h264(stream)), a.reps)
    res["Server: decode, embed and score 2 WebP stills"] = timeit(
        lambda: score([Image.open(io.BytesIO(b)).convert("RGB").resize((W, H), Image.BILINEAR) for b in stills]), a.reps)
    info.update({"machine": platform.machine(), "threads": a.threads, "torch": torch.__version__, "pyav": av.__version__, "reps": a.reps})
    Path(a.out).write_text(json.dumps({"info": info, "ms_median_p95": res}, indent=1))
    for k, v in res.items():
        print(f"{k:72s} median {v[0]:9.3f} ms   p95 {v[1]:9.3f} ms")
    print(info)


if __name__ == "__main__":
    main()
