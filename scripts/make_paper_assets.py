"""Build every table and figure of the paper from results/v3 (one command, no manual steps).

    python scripts/make_paper_assets.py --res results/v3 --out results/paper

Writes tables as CSV and Markdown (table1.csv ... table6.csv, tables.md) and figures as
PNG (fig2_k_vs_budget.png, fig3_policies.png, fig4_unseen.png). Figure 1 is drawn by
scripts/fig_pipeline.py.
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idpad.metrics import bpcer_at_apcer, eer, threshold_at_eer
from idpad.network import load_links
from idpad.study_v3 import boot_cell

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]
KB = {b: f"{b // 1000}" for b in BUDGETS}


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def table1(res, S, M):
    """Source frames (no re-encoding), by K: EER, CI, BPCER20, worst-species APCER."""
    cfg = pd.read_csv(res / "configs.csv")
    rows = []
    for k in (1, 2, 4, 8, 16):
        per, e, b20, aw, bt, nb = [], [], [], [], [], []
        for f in sorted(M):
            key = (f, "test", "clean", k, "spread", "ref", 1.0, -1)
            st, sv = S[key], S[(f, "val", "clean", k, "spread", "ref", 1.0, -1)]
            y, sp = M[f]["y_test"], M[f]["species_test"]
            per.append((y, st)); e.append(eer(y, st)); b20.append(bpcer_at_apcer(y, st, 0.05))
            t = threshold_at_eer(M[f]["y_val"], sv)
            aw.append(max(float((st[(y == 1) & (sp == s)] < t).mean()) for s in np.unique(sp[y == 1])))
            bt.append(float((st[y == 0] >= t).mean()))
        lo, hi = boot_cell(per)
        c = cfg[(cfg.split == "test") & (cfg.train == "clean") & (cfg.k == k) & (cfg.layout == "spread") & (cfg.codec == "ref")]
        rows.append(dict(K=k, EER=f"{pct(np.mean(e))} ± {pct(np.std(e, ddof=1))}", CI95=f"{pct(lo)}–{pct(hi)}",
                         BPCER20=pct(np.mean(b20)), APCER_worst=pct(np.mean(aw)), BPCER_at_thr=pct(np.mean(bt)),
                         payload_kB=f"{c.mean_bytes.mean() / 1000:.0f}"))
    return pd.DataFrame(rows)


LAB = {"photo": "Photo (K = 1)", "adaptive": "Adaptive stills", "video16": "Video16 stills",
       "video16_h264": "H.264, K = 16", "video16_h264_ca": "H.264, K = 16, CA"}


def policy_table(res, policies):
    """Two rows per budget: EER (95 % CI) and BPCER20, both in %."""
    p = pd.read_csv(res / "policies.csv")
    ci = pd.read_csv(res / "policy_ci.csv")
    rows = []
    for b in BUDGETS:
        r1 = {"Budget (kB)": KB[b], "Metric": "EER (95 % CI)"}
        r2 = {"Budget (kB)": "", "Metric": "BPCER20"}
        for pol in policies:
            g = p[(p.policy == pol) & (p.budget == b)].dropna(subset=["eer"])
            c = ci[(ci.policy == pol) & (ci.budget == b)]
            if len(g) < 5 or c.empty:
                r1[LAB[pol]] = r2[LAB[pol]] = "\u2013"; continue
            r1[LAB[pol]] = f"{pct(g.eer.mean())} ({pct(c.eer_lo.iloc[0])}\u2013{pct(c.eer_hi.iloc[0])})"
            r2[LAB[pol]] = pct(g.bpcer20.mean())
        rows += [r1, r2]
    return pd.DataFrame(rows)


def comparisons(res, families):
    c = pd.read_csv(res / "comparisons.csv")
    c = c[c.family.isin(families)].copy()
    c["Budget (kB)"] = (c.budget // 1000).astype(int).astype(str)
    c["ΔEER (pp)"] = c.apply(lambda r: f"{100 * r['diff']:+.1f} ({100 * r.ci_lo:+.1f} to {100 * r.ci_hi:+.1f})", axis=1)
    c["p (Holm)"] = c.p_holm.map(lambda x: "<0.01" if x < 0.01 else f"{x:.2f}")
    c["Folds better"] = c.folds_better.astype(str) + "/5"
    return c[["family", "Budget (kB)", "ΔEER (pp)", "p (Holm)", "Folds better"]]


NAMES = {"adaptive_vs_photo": "Photo → adaptive", "adaptive_vs_video16": "Video16 → adaptive",
         "webp_vs_jpeg": "JPEG → WebP (adaptive)", "spread_vs_window": "Window → spread frames",
         "ca_vs_clean": "Clean → CA training, adaptive", "ca_vs_clean_k16": "Clean → CA training, K = 16",
         "ca_vs_clean_k1": "Clean → CA training, K = 1", "h264_vs_adaptive": "Adaptive stills → H.264",
         "h264_vs_stills_k16": "Video16 stills → H.264", "ca_vs_clean_h264": "H.264 → H.264 with CA training",
         "h264ca_vs_adaptive": "Adaptive stills → H.264 with CA training"}


def comparisons_wide(res, families):
    c = pd.read_csv(res / "comparisons.csv")
    rows = []
    for fam in families:
        g = c[c.family == fam].set_index("budget")
        if g.empty:
            continue
        r1 = {"Comparison": NAMES[fam], "Statistic": "ΔEER, pp (95 % CI)"}
        r2 = {"Comparison": "", "Statistic": "p Holm (folds better)"}
        for b in BUDGETS:
            if b in g.index:
                x = g.loc[b]
                r1[KB[b]] = f"{100 * x['diff']:+.1f} ({100 * x.ci_lo:+.1f}, {100 * x.ci_hi:+.1f})"
                r2[KB[b]] = ("<0.01" if x.p_holm < 0.01 else f"{x.p_holm:.2f}") + f" ({int(x.folds_better)}/5)"
            else:
                r1[KB[b]] = r2[KB[b]] = "–"
        rows += [r1, r2]
    return pd.DataFrame(rows)


def unseen_table(res):
    u = pd.read_csv(res / "unseen.csv")
    rows = []
    for b in [-1, *BUDGETS]:
        r = {"Budget (kB)": "source" if b == -1 else KB[b]}
        for pol in ("photo", "adaptive", "video16", "video16_h264", "video16_h264_ca"):
            g = u[(u.policy == pol) & (u.budget == b)]
            r[pol] = pct(g.eer.mean()) if len(g) == 5 else "–"
            if len(g) == 5:
                r[pol + " APCER_worst"] = pct(g.apcer_worst.mean())
        rows.append(r)
    sp = sorted(c[4:] for c in u.columns if c.startswith("eer_"))
    per = []
    for s in sp:
        r = {"Species": s.replace("_", " ")}
        for pol, b in (("photo", -1), ("video16", -1), ("adaptive", 16000), ("video16", 16000), ("video16_h264", 16000)):
            g = u[(u.policy == pol) & (u.budget == b)]
            r[f"{pol} {'source' if b == -1 else KB[b] + ' kB'}"] = pct(g[f"eer_{s}"].mean()) if len(g) == 5 else "–"
        per.append(r)
    return pd.DataFrame(rows), pd.DataFrame(per)


def time_table(links_yaml, res):
    p = pd.read_csv(res / "policies.csv")
    links = load_links(links_yaml)
    src16 = p[(p.policy == "video16") & (p.budget == -1)].max_bytes.max()
    rows = []
    for b in [*BUDGETS, int(src16)]:
        r = {"Payload": f"{b // 1000} kB" if b in BUDGETS else f"{b / 1000:.0f} kB (16 source frames, largest clip)"}
        for l in links:
            t = l.upload_seconds_tcp(b)
            r[l.name] = f"{t:.2f}" if t < 10 else f"{t:.0f}"
        rows.append(r)
    return pd.DataFrame(rows)


def fig_k(res, out):
    cfg = pd.read_csv(res / "configs.csv")
    c = cfg[(cfg.train == "clean") & (cfg.layout == "spread") & cfg.codec.isin(["jpeg", "webp"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=300)
    marks = {1: "o", 2: "s", 4: "^", 8: "D", 16: "v"}
    for k in (1, 2, 4, 8, 16):
        xs, ys = [], []
        for b in BUDGETS:
            vals = []
            for f in range(5):
                v = c[(c.split == "val") & (c.k == k) & (c.budget == b) & (c.fold == f)].dropna(subset=["eer"])
                ok = c[(c.split == "test") & (c.k == k) & (c.budget == b) & (c.fold == f) & (c.feasible == 1.0)]
                v = v.merge(ok[["codec", "scale"]], on=["codec", "scale"])
                if v.empty:
                    break
                best = v.sort_values(["eer", "mean_bytes"]).iloc[0]
                t = c[(c.split == "test") & (c.k == k) & (c.budget == b) & (c.fold == f) & (c.codec == best.codec) & (c.scale == best.scale)]
                vals.append(t.eer.iloc[0])
            if len(vals) == 5:
                xs.append(b / 1000); ys.append(100 * np.mean(vals))
        ax.plot(xs, ys, marker=marks[k], ms=4, lw=1.2, label=f"K = {k}")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("Uplink budget per attempt (kB)"); ax.set_ylabel("Test EER (%)")
    ax.legend(frameon=False, ncol=1, fontsize=9); ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_pol(res, out, policies, labels, unseen=False):
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=300)
    if unseen:
        u = pd.read_csv(res / "unseen.csv")
        for pol, lab, m in zip(policies, labels, "osv^D"):
            g = u[u.policy == pol].groupby("budget").eer.agg(["mean", "count"])
            g = g[(g.index > 0) & (g["count"] == 5)]
            ax.plot(g.index / 1000, 100 * g["mean"], marker=m, ms=4, lw=1.2, label=lab)
    else:
        p = pd.read_csv(res / "policies.csv"); ci = pd.read_csv(res / "policy_ci.csv")
        for pol, lab, m in zip(policies, labels, "osv^D"):
            g = p[p.policy == pol].groupby("budget").eer.agg(["mean", "count"])
            g = g[(g.index > 0) & (g["count"] == 5)]
            c = ci[ci.policy == pol].set_index("budget").loc[g.index]
            x = g.index / 1000
            ax.errorbar(x, 100 * g["mean"], yerr=[100 * (g["mean"] - c.eer_lo), 100 * (c.eer_hi - g["mean"])],
                        marker=m, ms=4, lw=1.2, capsize=2.5, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("Uplink budget per attempt (kB)"); ax.set_ylabel("Test EER (%)")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/v3")
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--links", default="configs/links.yaml")
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(res / "scores.pkl", "rb") as fh:
        d = pickle.load(fh)
    S, M = d["scores"], d["metas"]
    has_h264 = pd.read_csv(res / "policies.csv").query("policy == 'video16_h264'").eer.notna().any()
    t = {}
    t["table1"] = table1(res, S, M)
    pols = ["photo", "adaptive", "video16"] + (["video16_h264", "video16_h264_ca"] if has_h264 else [])
    t["table2"] = policy_table(res, pols)
    t["table3"] = comparisons_wide(res, ["adaptive_vs_photo", "adaptive_vs_video16", "webp_vs_jpeg"])
    t["table4"] = comparisons_wide(res, ["h264_vs_adaptive", "h264_vs_stills_k16", "ca_vs_clean_h264", "h264ca_vs_adaptive"])
    t["table4b"] = comparisons_wide(res, ["spread_vs_window", "ca_vs_clean", "ca_vs_clean_k16", "ca_vs_clean_k1"])
    t["table5"], t["table5b"] = unseen_table(res)
    t["table6"] = time_table(a.links, res)
    with open(out / "tables.md", "w") as fh:
        for name, df in t.items():
            df.to_csv(out / f"{name}.csv", index=False)
            fh.write(f"## {name}\n\n{df.to_markdown(index=False)}\n\n")
    fig_k(res, out / "fig2_k_vs_budget.png")
    labs = [LAB[x] for x in pols]
    fig_pol(res, out / "fig3_policies.png", pols, labs)
    fig_pol(res, out / "fig4_unseen.png", pols, labs, unseen=True)
    print(open(out / "tables.md").read())


if __name__ == "__main__":
    main()
