"""Describe the cropped MIDV-Holo / DynAttack data before any experiment.

Prints clip counts per fold and partition, checks identity leakage, and measures
frame size and the JPEG size of the source frames (the starting point of every
rate-detection curve). Writes results/dataset_report.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from idpad import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cropped_root", default="data/midvdynattack/cropped")
    ap.add_argument("--split_root", default="data/splits_kfold_s0")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/dataset_report.json")
    a = ap.parse_args()
    root = Path(a.cropped_root)
    report = {"folders": {}, "folds": {}}
    for d in sorted([root / "origins"] + sorted((root / "fraud").glob("*"))):
        clips = [p for p in d.glob("*/*") if p.is_dir()]
        frames = [len(D.list_frames(c)) for c in clips]
        sample = [f for c in clips[:20] for f in D.list_frames(c)[:1]]
        sizes = [Image.open(f).size for f in sample]
        bytes_ = [f.stat().st_size for f in sample]
        report["folders"][str(d.relative_to(root))] = dict(
            clips=len(clips), frames_mean=float(np.mean(frames)) if frames else 0,
            frame_size_median=[int(v) for v in np.median(np.array(sizes), 0)] if sizes else None,
            source_jpeg_bytes_median=float(np.median(bytes_)) if bytes_ else None)
    for k in a.folds:
        tr = D.load_fold(root, a.split_root, k, "train", D.MIDV_HOLO_ATTACKS[:3])
        va = D.load_fold(root, a.split_root, k, "val", D.MIDV_HOLO_ATTACKS[:3])
        te = D.load_fold(root, a.split_root, k, "test", D.MIDV_HOLO_ATTACKS)
        report["folds"][k] = dict(train=D.describe(tr), val=D.describe(va), test=D.describe(te),
                                  leakage={kk: sorted(v) for kk, v in D.check_leakage(tr, va, te).items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
