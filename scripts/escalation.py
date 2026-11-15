"""Table 10: two-stage escalation policy evaluated from saved per-clip scores (known attacks only).

A first attempt is sent with the plan the deadline rule selects at budget B1. If its score
lies close to its deployment threshold, the server asks for a second attempt with the plan
selected at budget B2, and the second decision replaces the first. The uncertainty band is
set on validation so that a fraction e of validation attempts is escalated. Scores of the two
models are made comparable by subtracting each plan's own threshold.

Run from the repository root: python scripts/escalation.py --boot 10000
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
try:
    import idpad.study_v3 as S
except ImportError:
    import types
    _c = types.ModuleType("idpad.cnn"); _c.clip_embedding = None; sys.modules["idpad.cnn"] = _c
    import idpad.study_v3 as S
from idpad.metrics import eer, threshold_at_eer, bpcer_at_apcer
from idpad.network import load_links
from idpad.data import _CLIP_RE
from idpad.plan_table import rule_table

CANDIDATES = ["photo", "adaptive", "video16", "video16_h264", "video16_h264_ca"]


_T = {}


def rule_key(sel, f, b):
    """Plan of the plan table at budget b (idpad.plan_table: documented tie-break, cumulative)."""
    if not _T:
        _T.update(rule_table(sel))
    r = _T[(f, b)]
    return S.key_of(r)[1:]          # (train, k, layout, codec, scale, budget of the encoding)


def identity(n):
    m = _CLIP_RE.match(n); return f"{m['dtype']}/{m['model']}_{m['ident']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/v3"); ap.add_argument("--out", default="results/paper")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    d = pickle.load(open(Path(a.res) / "scores.pkl", "rb")); sc, M = d["scores"], d["metas"]
    sel = pd.read_csv(Path(a.res) / "selection.csv")
    gprs = [l for l in load_links("configs/links.yaml") if l.name == "gprs_2g"][0]
    rows, store = [], {}
    for b1, b2 in [(2000, 16000), (4000, 16000), (4000, 32000)]:
        for e in [0.0, 0.1, 0.2, 0.3, 1.0]:
            per = []
            for f in sorted(M):
                k1, k2 = rule_key(sel, f, b1), rule_key(sel, f, b2)
                yv, yt = M[f]["y_val"], M[f]["y_test"]
                s1v, s1t = sc[(f, "val") + k1], sc[(f, "test") + k1]
                s2v, s2t = sc[(f, "val") + k2], sc[(f, "test") + k2]
                t1, t2 = threshold_at_eer(yv, s1v), threshold_at_eer(yv, s2v)
                if e >= 1.0:
                    esc = np.ones(len(yt), bool)
                elif e <= 0.0:
                    esc = np.zeros(len(yt), bool)
                else:
                    dlt = np.quantile(np.abs(s1v - t1), e)
                    esc = np.abs(s1t - t1) <= dlt
                m = np.where(esc, s2t - t2, s1t - t1)
                per.append((yt, m, s1t - t1, s2t - t2, esc))
                rows.append(dict(b1=b1, b2=b2, e=e, fold=f, esc_rate=esc.mean(),
                                 esc_bona=esc[yt == 0].mean(), esc_attack=esc[yt == 1].mean(),
                                 eer=eer(yt, m), bpcer20=bpcer_at_apcer(yt, m, 0.05),
                                 apcer=float((m[yt == 1] < 0).mean()), bpcer=float((m[yt == 0] >= 0).mean())))
            store[(b1, b2, e)] = per
    df = pd.DataFrame(rows)
    T1 = {b: gprs.upload_seconds_tcp(b) for b in (2000, 4000, 16000, 32000)}
    df["hter"] = (df.apcer + df.bpcer) / 2          # error at the deployed thresholds (m = 0)
    g = df.groupby(["b1", "b2", "e"]).mean(numeric_only=True).reset_index()
    g["gprs_mean_s"] = [T1[r.b1] * (r.e < 1) + T1[r.b2] * r.esc_rate if r.e < 1 else T1[r.b2] for r in g.itertuples()]
    # identity-level paired bootstrap: two-stage policy vs first attempt alone and vs second alone
    groups = {f: None for f in M}
    for f in M:
        ids = np.array([identity(n) for n in M[f]["names_test"]]); groups[f] = [np.nonzero(ids == u)[0] for u in np.unique(ids)]
    rng = np.random.default_rng(0); comp = []
    for b1, b2 in [(2000, 16000), (4000, 16000), (4000, 32000)]:
        for e in (0.1, 0.2, 0.3):
            P = store[(b1, b2, e)]
            obs1 = np.mean([eer(y, m1) - eer(y, m) for y, m, m1, m2, _ in P])
            obs2 = np.mean([eer(y, m) - eer(y, m2) for y, m, m1, m2, _ in P])
            v1, v2 = [], []
            for _ in range(a.boot):
                acc1, acc2 = [], []
                for f, (y, m, m1, m2, _) in zip(sorted(M), P):
                    gg = groups[f]; i = np.concatenate([gg[j] for j in rng.integers(0, len(gg), len(gg))])
                    acc1.append(eer(y[i], m1[i]) - eer(y[i], m[i])); acc2.append(eer(y[i], m[i]) - eer(y[i], m2[i]))
                v1.append(np.mean(acc1)); v2.append(np.mean(acc2))
            v1, v2 = np.array(v1), np.array(v2)
            comp.append(dict(b1=b1, b2=b2, e=e, gain_vs_first=obs1, gain_lo=np.quantile(v1, .025), gain_hi=np.quantile(v1, .975),
                             p_gain=(1 + np.sum(v1 <= 0)) / (a.boot + 1),
                             gap_to_second=obs2, gap_lo=np.quantile(v2, .025), gap_hi=np.quantile(v2, .975)))
    c = pd.DataFrame(comp)
    o = np.argsort(c.p_gain.values); adj = np.empty(len(c)); run = 0.0
    for r, i in enumerate(o):                       # Holm over the nine settings
        run = max(run, min(1.0, (len(c) - r) * c.p_gain.values[i])); adj[i] = run
    c["p_holm"] = adj
    g.to_csv(Path(a.out) / "table7_escalation.csv", index=False); c.to_csv(Path(a.out) / "table7_escalation_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print((g.assign(**{k: g[k] * 100 for k in ["esc_rate", "esc_bona", "esc_attack", "eer", "bpcer20", "apcer", "bpcer", "hter"]})).round(1).to_string(index=False))
    print((c.assign(**{k: c[k] * 100 for k in ["gain_vs_first", "gain_lo", "gain_hi", "gap_to_second", "gap_lo", "gap_hi"]})).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
