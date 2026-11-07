"""Pilot study: detection error versus uplink payload, per transmission plan.

For each fold: (1) load train/val/test clips and verify there is no identity leakage,
(2) learn the forensic ROI from training bona fide clips, (3) train detectors on the
reference (un-recompressed) frames, (4) transmit every val/test clip under each plan and
budget and score it. Plan selection for the "adaptive" rows uses validation only.

Usage:  python -m idpad.pilot --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from PIL import Image

from . import data as D
from .detectors import DescriptorDetector, DynamicsDetector
from .features import N_DYNAMICS, N_FRAME_STATS, WORK_SIZE, clip_descriptor, dynamics_score
from .metrics import bootstrap_ci, bpcer_at_apcer, eer
from .transmission import Plan, forensic_prior, prior_to_box, transmit


def load_clip_frames(clip: D.Clip, n: int) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in D.sample_frames(clip, n)]


def make_plans(cfg: dict) -> list[Plan]:
    plans = []
    for codec, strat, k in product(cfg["codecs"], cfg["strategies"], cfg["k_frames"]):
        if strat == "chroma" and codec == "webp":
            continue  # lossy WebP has no 4:4:4 mode
        plans.append(Plan(strat, codec, k, cfg.get("roi_share", 0.6)))
    return plans


def _process(frames: list[Image.Image], plan: Plan, budget, roi) -> tuple[np.ndarray, float, int, bool]:
    sent = transmit(frames, plan, budget, roi)
    if not sent.feasible:
        return np.full(2 * N_FRAME_STATS + N_DYNAMICS, np.nan, np.float32), np.nan, 0, False
    return clip_descriptor(sent.frames), dynamics_score(sent.frames), sent.nbytes, True


def evaluate_set(frames_list, labels, species, plans, budgets, rois, desc_models, n_jobs):
    """Return per (plan, budget) arrays of scores for both detectors."""
    jobs = [(pi, bi, ci) for pi in range(len(plans)) for bi in range(len(budgets)) for ci in range(len(frames_list))]
    out = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(_process)(frames_list[ci], plans[pi], budgets[bi], rois[ci]) for pi, bi, ci in jobs
    )
    res = {}
    for (pi, bi, ci), (desc, dyn, nb, ok) in zip(jobs, out):
        r = res.setdefault((pi, bi), {"desc": [], "dyn": [], "bytes": [], "ok": []})
        r["desc"].append(desc); r["dyn"].append(dyn); r["bytes"].append(nb); r["ok"].append(ok)
    scored = {}
    for (pi, bi), r in res.items():
        ok = np.array(r["ok"])
        Dm = np.stack(r["desc"])
        s_desc = np.full(len(ok), np.nan)
        if ok.any():
            s_desc[ok] = desc_models[plans[pi].k_frames].score(None, Dm[ok])
        s_dyn = -np.array(r["dyn"], dtype=float)
        scored[(pi, bi)] = {"descriptor_lr": s_desc, "dynamics": s_dyn,
                            "bytes": np.array(r["bytes"]), "ok": ok}
    return scored


def summarise(labels, s, ok):
    m = ok & ~np.isnan(s)
    if m.sum() < 4 or len(np.unique(labels[m])) < 2:
        return dict(eer=np.nan, eer_lo=np.nan, eer_hi=np.nan, bpcer_at_apcer5=np.nan)
    lo, hi = bootstrap_ci(labels[m], s[m], n=300)
    return dict(eer=eer(labels[m], s[m]), eer_lo=lo, eer_hi=hi, bpcer_at_apcer5=bpcer_at_apcer(labels[m], s[m], 0.05))


def run_fold(cfg: dict, fold: int, plans: list[Plan], log) -> list[dict]:
    root, splits = cfg["cropped_root"], cfg["split_root"]
    tr = D.load_fold(root, splits, fold, "train", cfg["train_attacks"])
    va = D.load_fold(root, splits, fold, "val", cfg["train_attacks"])
    te = D.load_fold(root, splits, fold, "test", cfg["test_attacks"])
    leak = D.check_leakage(tr, va, te)
    log(f"fold {fold}: train {D.describe(tr)} | val {D.describe(va)} | test {D.describe(te)}")
    if leak:
        raise RuntimeError(f"identity leakage in fold {fold}: {leak}")
    if not tr or not te:
        raise RuntimeError(f"fold {fold}: empty partition, check paths")

    n = cfg["pool_frames"]
    load = lambda clips: Parallel(n_jobs=cfg["n_jobs"])(delayed(load_clip_frames)(c, n) for c in clips)
    Ftr, Fva, Fte = load(tr), load(va), load(te)

    # forensic ROI per document type, learned from training bona fide clips only
    rois_by_type = {}
    for dtype in {c.name.split("/")[0] for c in tr}:
        clips = [f for c, f in zip(tr, Ftr) if c.label == 0 and c.name.startswith(dtype)]
        prior = forensic_prior([[np.asarray(x) for x in f] for f in clips], WORK_SIZE)
        rois_by_type[dtype] = prior_to_box(prior, cfg.get("roi_area_fraction", 0.25))
    log(f"fold {fold}: forensic ROI {json.dumps(rois_by_type)}")
    roi_of = lambda clips: [rois_by_type.get(c.name.split("/")[0]) for c in clips]

    ytr = np.array([c.label for c in tr])
    desc_models = {}
    for k in sorted({p.k_frames for p in plans}):
        ref = [transmit(f, Plan("uniform", "jpeg", k), None).frames for f in Ftr]
        Dtr = np.stack(Parallel(n_jobs=cfg["n_jobs"])(delayed(clip_descriptor)(x) for x in ref))
        desc_models[k] = DescriptorDetector().fit(None, ytr, Dtr)

    budgets = cfg["budgets"]
    rows = []
    sets = {"val": (va, Fva), "test": (te, Fte)}
    results = {}
    for name, (clips, frames) in sets.items():
        if not clips:
            continue
        t0 = time.time()
        results[name] = evaluate_set(frames, None, None, plans, budgets, roi_of(clips), desc_models, cfg["n_jobs"])
        log(f"fold {fold}: scored {name} in {time.time() - t0:.0f}s")
        y = np.array([c.label for c in clips])
        for (pi, bi), r in results[name].items():
            for det in ("descriptor_lr", "dynamics"):
                rows.append(dict(fold=fold, split=name, plan=plans[pi].tag(), **plans[pi].as_dict(),
                                 budget=budgets[bi] if budgets[bi] is not None else -1,
                                 detector=det, feasible=float(r["ok"].mean()),
                                 mean_bytes=float(r["bytes"][r["ok"]].mean()) if r["ok"].any() else np.nan,
                                 n_clips=len(y), **summarise(y, r[det], r["ok"])))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--folds", type=int, nargs="*")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.folds:
        cfg["folds"] = args.folds
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    logf = (out / "log.txt").open("a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    plans = make_plans(cfg)
    log(f"{len(plans)} plans x {len(cfg['budgets'])} budgets")
    all_rows = []
    for fold in cfg["folds"]:
        all_rows += run_fold(cfg, fold, plans, log)
        pd.DataFrame(all_rows).to_csv(out / "results.csv", index=False)
    log(f"saved {out / 'results.csv'}")
    from .gate import evaluate_gate
    verdict = evaluate_gate(pd.DataFrame(all_rows), cfg["gate"])
    (out / "gate.json").write_text(json.dumps(verdict, indent=2))
    log(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
