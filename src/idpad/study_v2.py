"""v2 analysis: budget allocation between frames, quality, resolution and codec.

Reads the per-clip cache from ``scripts/build_cache_v2.py`` and evaluates, per fold:
every (K, codec, scale) configuration at every budget, and the three policies of
``docs/protocol_v2.md`` (photo, video16, adaptive), with selection on validation only.

Usage: python -m idpad.study_v2 --cache data/cache_v2 --out results/v2
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from .cnn import clip_embedding
from .metrics import bootstrap_ci, bpcer_at_apcer, eer

KS = (1, 2, 4, 8, 16)
CODECS = ("jpeg", "webp")
SCALES = (1.0, 0.5)
BUDGETS = (64000, 32000, 16000, 8000, 4000, 2000)


def nested(k: int, n: int = 16) -> np.ndarray:
    """Indices of K frames out of n, nested across K (1 ⊂ 2 ⊂ 4 ⊂ 8 ⊂ 16)."""
    return np.arange(0, n, n // k)[:k]


def load_cache(cache: Path, clips: list[D.Clip]) -> list[dict]:
    out = []
    for c in clips:
        f = cache / c.attack_type / f"{c.name.replace('/', '__')}.npz"
        out.append(dict(np.load(f)))
    return out


def transmit_lookup(entry: dict, k: int, codec: str, scale: float, budget: int | None):
    """Return (clip vector, total bytes) for the best common quality that fits, or None."""
    idx = nested(k)
    if budget is None:
        return clip_embedding(entry["ref_emb"][idx].astype(np.float32)), int(entry["ref_bytes"][idx].sum())
    sizes = entry[f"{codec}_{scale}_bytes"][:, idx].sum(1)  # per quality level
    ok = np.nonzero(sizes <= budget)[0]
    if len(ok) == 0:
        return None
    qi = ok.max()
    return clip_embedding(entry[f"{codec}_{scale}_emb"][qi, idx].astype(np.float32)), int(sizes[qi])


def fit(entries, y, k):
    X = np.stack([clip_embedding(e["ref_emb"][nested(k)].astype(np.float32)) for e in entries])
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)).fit(X, y)


def score_config(model, entries, k, codec, scale, budget):
    vecs, nbytes, ok = [], [], []
    for e in entries:
        r = transmit_lookup(e, k, codec, scale, budget)
        ok.append(r is not None)
        if r is not None:
            vecs.append(r[0]); nbytes.append(r[1])
    ok = np.array(ok)
    s = np.full(len(entries), np.nan)
    if ok.any():
        s[ok] = model.predict_proba(np.stack(vecs))[:, 1]
    return s, ok, (float(np.mean(nbytes)) if nbytes else np.nan)


def metrics_row(y, s, ok):
    if ok.mean() < 1.0:  # a plan must work for every clip to be deployable
        return dict(eer=np.nan, eer_lo=np.nan, eer_hi=np.nan, bpcer_at_apcer5=np.nan)
    lo, hi = bootstrap_ci(y, s, n=500)
    return dict(eer=eer(y, s), eer_lo=lo, eer_hi=hi, bpcer_at_apcer5=bpcer_at_apcer(y, s, 0.05))


def run_fold(cache, root, splits, fold):
    tr = D.load_fold(root, splits, fold, "train", D.MIDV_HOLO_ATTACKS[:3])
    va = D.load_fold(root, splits, fold, "val", D.MIDV_HOLO_ATTACKS[:3])
    te = D.load_fold(root, splits, fold, "test", D.MIDV_HOLO_ATTACKS)
    assert not D.check_leakage(tr, va, te), "identity leakage"
    Etr, Eva, Ete = (load_cache(cache, x) for x in (tr, va, te))
    ytr, yva, yte = (np.array([c.label for c in x]) for x in (tr, va, te))
    species = np.array([c.attack_type for c in te])
    rows = []
    for k in KS:
        model = fit(Etr, ytr, k)
        for split, E, y in (("val", Eva, yva), ("test", Ete, yte)):
            s, ok, nb = score_config(model, E, k, None, None, None)
            rows.append(dict(fold=fold, split=split, k=k, codec="ref", scale=1.0, budget=-1, mean_bytes=nb,
                             feasible=ok.mean(), **metrics_row(y, s, ok)))
            for codec, scale, b in product(CODECS, SCALES, BUDGETS):
                s, ok, nb = score_config(model, E, k, codec, scale, b)
                row = dict(fold=fold, split=split, k=k, codec=codec, scale=scale, budget=b, mean_bytes=nb,
                           feasible=ok.mean(), **metrics_row(y, s, ok))
                if split == "test" and ok.all():
                    for sp in np.unique(species[y == 1]):
                        m = (y == 0) | (species == sp)
                        row[f"eer_{sp}"] = eer(y[m], s[m])
                rows.append(row)
    return rows


def policies(df: pd.DataFrame) -> pd.DataFrame:
    """photo / video16 / adaptive, each selected on validation EER, read on test."""
    out = []
    for (fold, b), g in df[df.budget > 0].groupby(["fold", "budget"]):
        val, test = g[g.split == "val"].dropna(subset=["eer"]), g[g.split == "test"]
        for name, cond in (("photo", val.k == 1), ("video16", val.k == 16), ("adaptive", val.k > 0)):
            v = val[cond]
            if v.empty:
                out.append(dict(fold=fold, budget=b, policy=name, eer=np.nan)); continue
            best = v.sort_values(["eer", "mean_bytes"]).iloc[0]
            t = test[(test.k == best.k) & (test.codec == best.codec) & (test.scale == best.scale)].iloc[0]
            out.append(dict(fold=fold, budget=b, policy=name, k=int(best.k), codec=best.codec, scale=best.scale,
                            eer=t.eer, eer_lo=t.eer_lo, eer_hi=t.eer_hi, bpcer_at_apcer5=t.bpcer_at_apcer5,
                            mean_bytes=t.mean_bytes))
    return pd.DataFrame(out)


def decision(pol: pd.DataFrame, min_gain=0.2, min_budgets=3, min_folds=4) -> dict:
    res = {}
    for b, g in pol.groupby("budget"):
        p = g.pivot(index="fold", columns="policy", values="eer")
        def rel(base):
            r = (p[base] - p["adaptive"]) / p[base]
            return r.where(p[base].notna(), np.inf)  # infeasible baseline counts as beaten
        both = (rel("photo") >= min_gain) & (rel("video16") >= min_gain)
        res[int(b)] = dict(folds_beating_both=int(both.sum()), passes=bool(both.sum() >= min_folds),
                           mean_eer={k: (float(v) if np.isfinite(v) else None) for k, v in p.mean().items()})
    n = sum(v["passes"] for v in res.values())
    return dict(per_budget=res, budgets_passing=n, decision="GO" if n >= min_budgets else "NO-GO")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache_v2")
    ap.add_argument("--cropped_root", default="data/midvdynattack/cropped")
    ap.add_argument("--split_root", default="data/splits_kfold_s0")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/v2")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in a.folds:
        rows += run_fold(Path(a.cache), a.cropped_root, a.split_root, f)
        print(f"fold {f} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "configs.csv", index=False)
    pol = policies(df); pol.to_csv(out / "policies.csv", index=False)
    dec = decision(pol)
    (out / "decision.json").write_text(json.dumps(dec, indent=2))
    print(json.dumps(dec, indent=2))


if __name__ == "__main__":
    main()
