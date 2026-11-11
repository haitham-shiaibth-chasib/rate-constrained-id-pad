"""v3 analysis: the v2 study with corrected metrics, uncertainty, and the ablations
asked for in review.

What it adds to ``study_v2``:
  * per-clip scores are kept, so every policy gets a clip-bootstrap 95 % CI and every
    comparison a paired bootstrap test (Holm-corrected across budgets);
  * ISO/IEC 30107-3 reporting: BPCER20, and APCER of the worst attack species together
    with BPCER at a threshold fixed on validation;
  * frame layout ablation: K frames spread over the tilt ("spread") or the first K of the
    16 samples ("window", covering only the first K/16 of the tilt);
  * compression-aware training: the server model for a plan is trained on training clips
    sent with that same plan ("ca"), against the clean-trained model ("clean");
  * an H.264 baseline (the 16 frames coded as one stream), when ``data/cache_h264`` exists;
  * unseen MIDV-DynAttack species with plans chosen on MIDV-Holo validation.

Every plan choice uses validation data only. Usage:
    python -m idpad.study_v3 --out results/v3 [--h264 data/cache_h264] [--unseen]
"""

from __future__ import annotations

import argparse
import json
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from .cnn import clip_embedding
from .metrics import bpcer_at_apcer, eer, threshold_at_eer

KS = (1, 2, 4, 8, 16)
STILL_CODECS = ("jpeg", "webp")
SCALES = (1.0, 0.5, 0.25)
BUDGETS = (64000, 32000, 16000, 8000, 4000, 2000)
H264_KS, H264_SCALES = (16,), (0.5, 0.25)
SEED = 0


def frame_idx(k: int, layout: str, n: int = 16) -> np.ndarray:
    if layout == "spread":
        return np.arange(0, n, n // k)[:k]
    if layout == "window":
        return np.arange(k)
    raise ValueError(layout)


def _load(cache: Path, clips, h264: Path | None):
    out = []
    for c in clips:
        name = f"{c.name.replace('/', '__')}.npz"
        e = dict(np.load(cache / c.attack_type / name))
        if h264 is not None and (h264 / c.attack_type / name).exists():
            e.update(np.load(h264 / c.attack_type / name))
        out.append(e)
    return out


def plan_vector(e: dict, k: int, layout: str, codec: str, scale: float, budget: int):
    """Clip vector and bytes for a plan, or None when the plan does not fit this clip.

    codec 'ref' is the source frames (JPEG q~95 as distributed); the best quality level
    that fits the budget is used for still codecs and H.264 alike.
    """
    idx = frame_idx(k, layout)
    if codec == "ref":
        return clip_embedding(e["ref_emb"][idx].astype(np.float32)), int(e["ref_bytes"][idx].sum())
    if codec == "h264":
        key = f"h264_k{k}_{scale}"
        if layout != "spread" or f"{key}_bytes" not in e:
            return None
        sizes, emb = e[f"{key}_bytes"], e[f"{key}_emb"]
        if len(sizes) == 7:  # early cache files also hold CRF 18 and 51; keep the common grid 24-48
            sizes, emb = sizes[1:6], emb[1:6]
        ok = np.nonzero(sizes <= budget)[0]
        if len(ok) == 0:
            return None
        qi = ok.min()  # CRF grid is ordered from best to worst quality
        return clip_embedding(emb[qi].astype(np.float32)), int(sizes[qi])
    key = f"{codec}_{scale}"
    if f"{key}_bytes" not in e:
        return None
    sizes = e[f"{key}_bytes"][:, idx].sum(1)
    ok = np.nonzero(sizes <= budget)[0]
    if len(ok) == 0:
        return None
    qi = ok.max()  # quality grid is ordered from worst to best
    return clip_embedding(e[f"{key}_emb"][qi, idx].astype(np.float32)), int(sizes[qi])


def _vectors(entries, plan):
    vecs, nb, ok = [], [], []
    for e in entries:
        r = plan_vector(e, *plan)
        ok.append(r is not None)
        if r is not None:
            vecs.append(r[0]); nb.append(r[1])
    return vecs, np.array(nb), np.array(ok)


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))


def all_plans(h264: bool):
    plans = []
    for layout in ("spread", "window"):
        for k in KS:
            if layout == "window" and k in (1, 16):
                continue  # identical to spread
            plans.append((k, layout, "ref", 1.0, -1))
            for codec, scale, b in product(STILL_CODECS, SCALES, BUDGETS):
                plans.append((k, layout, codec, scale, b))
    if h264:
        for k, scale, b in product(H264_KS, H264_SCALES, BUDGETS):
            plans.append((k, "spread", "h264", scale, b))
    return plans


def run_fold(cache, root, splits, fold, h264_dir=None, ca=True):
    tr = D.load_fold(root, splits, fold, "train", D.MIDV_HOLO_ATTACKS[:3])
    va = D.load_fold(root, splits, fold, "val", D.MIDV_HOLO_ATTACKS[:3])
    te = D.load_fold(root, splits, fold, "test", D.MIDV_HOLO_ATTACKS)
    assert not D.check_leakage(tr, va, te), "identity leakage"
    Etr, Eva, Ete = (_load(cache, x, h264_dir) for x in (tr, va, te))
    ytr, yva, yte = (np.array([c.label for c in x]) for x in (tr, va, te))
    clean = {}
    for k in KS:
        for layout in ("spread", "window"):
            idx = frame_idx(k, layout)
            X = np.stack([clip_embedding(e["ref_emb"][idx].astype(np.float32)) for e in Etr])
            clean[(k, layout)] = _model().fit(X, ytr)
    rows, scores = [], {}
    for plan in all_plans(h264_dir is not None):
        k, layout, codec, scale, b = plan
        trains = ["clean"] + (["ca"] if ca and codec != "ref" else [])
        for train in trains:
            if train == "clean":
                m = clean[(k, layout)]
            else:
                v, _, ok = _vectors(Etr, plan)
                if ok.sum() < 20 or len(np.unique(ytr[ok])) < 2:
                    continue
                m = _model().fit(np.stack(v), ytr[ok])
            for split, E, y in (("val", Eva, yva), ("test", Ete, yte)):
                v, nb, ok = _vectors(E, plan)
                feas = float(ok.mean())
                row = dict(fold=fold, split=split, train=train, k=k, layout=layout, codec=codec, scale=scale,
                           budget=b, feasible=feas, mean_bytes=float(nb.mean()) if len(nb) else np.nan,
                           max_bytes=float(nb.max()) if len(nb) else np.nan, eer=np.nan)
                if feas == 1.0:
                    s = m.predict_proba(np.stack(v))[:, 1]
                    row["eer"] = eer(y, s)
                    scores[(fold, split, train) + plan] = s.astype(np.float32)
                rows.append(row)
    meta = dict(y_val=yva, y_test=yte, species_test=np.array([c.attack_type for c in te]),
                names_test=[c.name for c in te])
    return rows, scores, meta


POLICIES = {
    # name: (train, layouts, codecs, ks)
    "photo": ("clean", ("spread",), STILL_CODECS, (1,)),
    "video16": ("clean", ("spread",), STILL_CODECS, (16,)),
    "adaptive": ("clean", ("spread",), STILL_CODECS, KS),
    "adaptive_jpeg": ("clean", ("spread",), ("jpeg",), KS),
    "adaptive_webp": ("clean", ("spread",), ("webp",), KS),
    "photo_webp": ("clean", ("spread",), ("webp",), (1,)),
    "video16_webp": ("clean", ("spread",), ("webp",), (16,)),
    "adaptive_window": ("clean", ("window",), STILL_CODECS, (2, 4, 8)),
    "photo_ca": ("ca", ("spread",), STILL_CODECS, (1,)),
    "video16_ca": ("ca", ("spread",), STILL_CODECS, (16,)),
    "adaptive_ca": ("ca", ("spread",), STILL_CODECS, KS),
    "video16_h264": ("clean", ("spread",), ("h264",), (16,)),
    "video16_h264_ca": ("ca", ("spread",), ("h264",), (16,)),
}


def select(df: pd.DataFrame) -> pd.DataFrame:
    """For each fold, budget and policy: the plan with the lowest validation EER
    (ties: fewer mean bytes, then fewer frames). The reference budget (-1) uses 'ref'."""
    out = []
    # Deployability is a size check and uses no labels: a plan must fit every validation
    # and test clip. Plan quality is judged on validation EER only.
    keys = ["fold", "train", "k", "layout", "codec", "scale", "budget"]
    fits_test = df[(df.split == "test") & (df.feasible == 1.0)][keys]
    val = df[df.split == "val"].dropna(subset=["eer"]).merge(fits_test, on=keys)
    for (fold, b), g in val.groupby(["fold", "budget"]):
        for name, (train, layouts, codecs, ks) in POLICIES.items():
            cod = ("ref",) if b == -1 else codecs
            if b == -1 and name not in ("photo", "video16"):
                continue  # the source frames are one plan per K; listed once
            v = g[(g.train == train) & g.layout.isin(layouts) & g.codec.isin(cod) & g.k.isin(ks)]
            if v.empty:
                out.append(dict(fold=fold, budget=b, policy=name, feasible=False)); continue
            best = v.sort_values(["eer", "mean_bytes", "k"]).iloc[0]
            out.append(dict(fold=fold, budget=b, policy=name, feasible=True, train=train, k=int(best.k),
                            layout=best.layout, codec=best.codec, scale=float(best.scale), val_eer=best.eer))
    return pd.DataFrame(out)


def key_of(r) -> tuple:
    return (int(r.fold), r.train, int(r.k), r.layout, r.codec, float(r.scale), int(r.budget))


def evaluate(sel, df, scores, metas):
    rows = []
    test = df[df.split == "test"].set_index(["fold", "train", "k", "layout", "codec", "scale", "budget"])
    for r in sel.itertuples():
        if not r.feasible:
            rows.append(dict(fold=r.fold, budget=r.budget, policy=r.policy)); continue
        f, tr, k, lay, cod, sc, b = key_of(r)
        st = scores[(f, "test", tr, k, lay, cod, sc, b)]
        sv = scores[(f, "val", tr, k, lay, cod, sc, b)]
        m = metas[f]
        y, sp = m["y_test"], m["species_test"]
        thr = threshold_at_eer(m["y_val"], sv)
        apcer_sp = {s: float((st[(y == 1) & (sp == s)] < thr).mean()) for s in np.unique(sp[y == 1])}
        t = test.loc[(f, tr, k, lay, cod, sc, b)]
        rows.append(dict(fold=f, budget=b, policy=r.policy, k=k, layout=lay, codec=cod, scale=sc,
                         mean_bytes=t.mean_bytes, max_bytes=t.max_bytes, eer=eer(y, st),
                         bpcer20=bpcer_at_apcer(y, st, 0.05), bpcer_at_thr=float((st[y == 0] >= thr).mean()),
                         apcer_worst=max(apcer_sp.values()), apcer_worst_species=max(apcer_sp, key=apcer_sp.get),
                         **{f"apcer_{s}": v for s, v in apcer_sp.items()}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def _boot_idx(y, rng):
    i0, i1 = np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]
    return np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]


def boot_cell(per_fold, B=2000, seed=SEED):
    """per_fold: list of (y, s). CI of the mean-over-folds EER by resampling clips within folds."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        acc = []
        for y, s in per_fold:
            i = _boot_idx(y, rng)
            acc.append(eer(y[i], s[i]))
        vals.append(np.mean(acc))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def boot_diff(per_fold, B=2000, seed=SEED):
    """per_fold: list of (y, s_base, s_new) on the same clips. Paired bootstrap of
    mean-over-folds EER(base) - EER(new); returns mean, 95 % CI, one-sided p (new not better)."""
    rng = np.random.default_rng(seed)
    obs = np.mean([eer(y, a) - eer(y, b) for y, a, b in per_fold])
    vals = []
    for _ in range(B):
        acc = []
        for y, a, b in per_fold:
            i = _boot_idx(y, rng)
            acc.append(eer(y[i], a[i]) - eer(y[i], b[i]))
        vals.append(np.mean(acc))
    vals = np.array(vals)
    p = float((np.sum(vals <= 0) + 1) / (B + 1))
    return float(obs), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), p


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


COMPARISONS = [  # (family, baseline, new)
    ("adaptive_vs_photo", "photo", "adaptive"),
    ("adaptive_vs_video16", "video16", "adaptive"),
    ("webp_vs_jpeg", "adaptive_jpeg", "adaptive_webp"),
    ("spread_vs_window", "adaptive_window", "adaptive"),
    ("ca_vs_clean", "adaptive", "adaptive_ca"),
    ("h264_vs_adaptive", "adaptive", "video16_h264"),
    ("h264_vs_stills_k16", "video16", "video16_h264"),
    ("h264ca_vs_stillsca_k16", "video16_ca", "video16_h264_ca"),
    ("ca_vs_clean_h264", "video16_h264", "video16_h264_ca"),
    ("h264ca_vs_adaptive", "adaptive", "video16_h264_ca"),
    ("ca_vs_clean_k16", "video16", "video16_ca"),
    ("ca_vs_clean_k1", "photo", "photo_ca"),
]


def _scores_for(sel, scores, metas, policy, budget):
    s = sel[(sel.policy == policy) & (sel.budget == budget)]
    if s.empty or not s.feasible.all() or len(s) < len(metas):
        return None
    out = {}
    for r in s.itertuples():
        f, tr, k, lay, cod, sc, b = key_of(r)
        out[f] = scores[(f, "test", tr, k, lay, cod, sc, b)]
    return out


def statistics(sel, scores, metas, B=2000):
    cells, comps = [], []
    for (pol, b), _ in sel.groupby(["policy", "budget"]):
        sc = _scores_for(sel, scores, metas, pol, b)
        if sc is None:
            continue
        lo, hi = boot_cell([(metas[f]["y_test"], sc[f]) for f in sorted(sc)], B)
        cells.append(dict(policy=pol, budget=b, eer_lo=lo, eer_hi=hi))
    for fam, base, new in COMPARISONS:
        fam_rows = []
        for b in sorted(sel.budget.unique()):
            if b == -1:
                continue
            a, n = _scores_for(sel, scores, metas, base, b), _scores_for(sel, scores, metas, new, b)
            if a is None or n is None:
                continue
            folds = sorted(a)
            diffs = [eer(metas[f]["y_test"], a[f]) - eer(metas[f]["y_test"], n[f]) for f in folds]
            d, lo, hi, p = boot_diff([(metas[f]["y_test"], a[f], n[f]) for f in folds], B)
            fam_rows.append(dict(family=fam, baseline=base, new=new, budget=b, diff=d, ci_lo=lo, ci_hi=hi, p=p,
                                 folds_better=int(np.sum(np.array(diffs) > 0)), folds_tied=int(np.sum(np.array(diffs) == 0))))
        if fam_rows:
            for r, pa in zip(fam_rows, holm([r["p"] for r in fam_rows])):
                r["p_holm"] = pa
            comps += fam_rows
    return pd.DataFrame(cells), pd.DataFrame(comps)


# ---------------------------------------------------------------- unseen attacks

def run_unseen(cache, root, splits, fold, sel_fold, h264_dir=None):
    """MIDV-DynAttack species (test identities, never trained on). Plans come from MIDV-Holo
    validation; only WebP and the source frames exist in the DynAttack cache."""
    tr = D.load_fold(root, splits, fold, "train", D.MIDV_HOLO_ATTACKS[:3])
    va = D.load_fold(root, splits, fold, "val", D.MIDV_HOLO_ATTACKS[:3])
    te = D.load_fold(root, splits, fold, "test", D.DYNATTACK_ATTACKS)
    Etr, Eva, Ete = (_load(cache, x, h264_dir) for x in (tr, va, te))
    ytr, yva, yte = (np.array([c.label for c in x]) for x in (tr, va, te))
    sp = np.array([c.attack_type for c in te])
    rows, scores = [], {}
    keep = {"photo_webp": "photo", "video16_webp": "video16", "adaptive_webp": "adaptive", "video16_h264": "video16_h264",
            "video16_h264_ca": "video16_h264_ca"}
    for r in sel_fold.itertuples():
        if not r.feasible or not (r.policy in keep or (r.budget == -1 and r.policy in ("photo", "video16"))):
            continue
        plan = (int(r.k), r.layout, r.codec, float(r.scale), int(r.budget))
        if getattr(r, "train", "clean") == "ca":
            v, _, ok = _vectors(Etr, plan)
            m = _model().fit(np.stack(v), ytr[ok])
        else:
            idx = frame_idx(plan[0], plan[1])
            X = np.stack([clip_embedding(e["ref_emb"][idx].astype(np.float32)) for e in Etr])
            m = _model().fit(X, ytr)
        v, nb, ok = _vectors(Ete, plan)
        if ok.mean() < 1:
            continue
        s = m.predict_proba(np.stack(v))[:, 1]
        vv, _, okv = _vectors(Eva, plan)
        thr = threshold_at_eer(yva[okv], m.predict_proba(np.stack(vv))[:, 1])
        apc = {x: float((s[(yte == 1) & (sp == x)] < thr).mean()) for x in np.unique(sp[yte == 1])}
        pol = keep.get(r.policy, r.policy)
        scores[(fold, pol, int(r.budget))] = s
        rows.append(dict(fold=fold, budget=int(r.budget), policy=pol, k=plan[0], scale=plan[3],
                         mean_bytes=float(nb.mean()), eer=eer(yte, s), bpcer20=bpcer_at_apcer(yte, s, 0.05),
                         apcer_worst=max(apc.values()), apcer_worst_species=max(apc, key=apc.get),
                         **{f"eer_{x}": eer(yte[(yte == 0) | (sp == x)], s[(yte == 0) | (sp == x)]) for x in np.unique(sp[yte == 1])}))
    return rows, scores, yte


def unseen_all(a, sel, out, h264):
    ur, us, ys = [], {}, {}
    for f in a.folds:
        r, s, y = run_unseen(Path(a.cache), a.cropped_root, a.split_root, f, sel[sel.fold == f], h264)
        ur += r; us.update(s); ys[f] = y
    pd.DataFrame(ur).to_csv(out / "unseen.csv", index=False)
    uc = []
    for base, new in (("photo", "adaptive"), ("video16", "adaptive"), ("adaptive", "video16_h264"),
                      ("photo", "video16_h264"), ("adaptive", "video16_h264_ca"), ("video16_h264", "video16_h264_ca")):
        for b in sorted({k[2] for k in us if k[2] > 0}):
            folds = [f for f in a.folds if (f, base, b) in us and (f, new, b) in us]
            if len(folds) < len(a.folds):
                continue
            d, lo, hi, p = boot_diff([(ys[f], us[(f, base, b)], us[(f, new, b)]) for f in folds], a.boot)
            uc.append(dict(baseline=base, new=new, budget=b, diff=d, ci_lo=lo, ci_hi=hi, p=p))
    uc = pd.DataFrame(uc)
    for (base, new), m in uc.groupby(["baseline", "new"]).groups.items():
        uc.loc[m, "p_holm"] = holm(uc.loc[m, "p"].tolist())
    uc.to_csv(out / "unseen_comparisons.csv", index=False)
    print("unseen done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache_v2")
    ap.add_argument("--h264", default=None)
    ap.add_argument("--cropped_root", default="data/midvdynattack/cropped")
    ap.add_argument("--split_root", default="data/splits_kfold_s0")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/v3")
    ap.add_argument("--no_ca", action="store_true")
    ap.add_argument("--unseen", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--stats_only", action="store_true", help="recompute statistics from saved scores")
    ap.add_argument("--unseen_only", action="store_true", help="rerun only the unseen-attack test from saved plans")
    a = ap.parse_args(argv)
    if a.unseen_only:
        out = Path(a.out)
        unseen_all(a, pd.read_csv(out / "selection.csv"), out, Path(a.h264) if a.h264 else None)
        return
    if a.stats_only:
        out = Path(a.out)
        with open(out / "scores.pkl", "rb") as fh:
            d = pickle.load(fh)
        sel = pd.read_csv(out / "selection.csv")
        cells, comps = statistics(sel, d["scores"], d["metas"], a.boot)
        cells.to_csv(out / "policy_ci.csv", index=False)
        comps.to_csv(out / "comparisons.csv", index=False)
        print(comps.to_string())
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    h264 = Path(a.h264) if a.h264 else None
    rows, scores, metas = [], {}, {}
    for f in a.folds:
        r, s, m = run_fold(Path(a.cache), a.cropped_root, a.split_root, f, h264, ca=not a.no_ca)
        rows += r; scores.update(s); metas[f] = m
        print(f"fold {f} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "configs.csv", index=False)
    sel = select(df)
    sel.to_csv(out / "selection.csv", index=False)
    pol = evaluate(sel, df, scores, metas)
    pol.to_csv(out / "policies.csv", index=False)
    with open(out / "scores.pkl", "wb") as fh:
        pickle.dump(dict(scores=scores, metas=metas), fh)
    cells, comps = statistics(sel, scores, metas, a.boot)
    cells.to_csv(out / "policy_ci.csv", index=False)
    comps.to_csv(out / "comparisons.csv", index=False)
    print(comps.to_string(), flush=True)
    if a.unseen:
        unseen_all(a, sel, out, h264)
    (out / "run.json").write_text(json.dumps(dict(folds=a.folds, h264=bool(h264), ca=not a.no_ca, boot=a.boot,
                                                  seed=SEED), indent=2))


if __name__ == "__main__":
    main()
