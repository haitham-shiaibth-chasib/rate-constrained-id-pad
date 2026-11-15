"""Tables 8 and 9 of the paper: delivery regimes and deadline-driven configuration on GPRS.

Table 8: minimum uplink rate at which a budget meets a deadline (Eq. 8).
Table 9: for each GPRS deadline, the largest budget that fits (Eq. 2), the EER of each
policy at that budget, the plan of the plan table (idpad.plan_table: lowest validation EER across
{photo, adaptive, video16, video16_h264, video16_h264_ca}, documented tie-break, cumulative) (the rule), its EER on known
and unseen attacks, and its BPCER at the deployment threshold (manual-review share).

Run from the repository root:  python scripts/deadline_tables.py
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "src")
from idpad.network import Link, load_links
from idpad.plan_table import rule_table

RES, OUT = Path("results/v3"), Path("results/paper")
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]
CANDIDATES = ["photo", "adaptive", "video16", "video16_h264", "video16_h264_ca"]


def min_rate_kbps(nbytes, rtt_ms, deadline):
    lo, hi = 0.1, 1e6
    if Link("x", hi, rtt_ms).upload_seconds_tcp(nbytes) > deadline:
        return None
    for _ in range(80):
        mid = (lo * hi) ** 0.5
        lo, hi = (lo, mid) if Link("x", mid, rtt_ms).upload_seconds_tcp(nbytes) <= deadline else (mid, hi)
    return round(hi, 1)


def regime_table():
    cols = [(83, 3), (83, 10), (684, 3), (684, 10), (1000, 10), (1000, 30)]
    rows = []
    for b in (4000, 16000, 64000):
        r = {"Budget (kB)": b // 1000}
        for rtt, dl in cols:
            v = min_rate_kbps(b, rtt, dl)
            r[f"RTT {rtt} ms, D = {dl} s"] = "–" if v is None else v
        rows.append(r)
    return pd.DataFrame(rows)


def gprs_table(links_yaml="configs/links.yaml"):
    gprs = [l for l in load_links(links_yaml) if l.name == "gprs_2g"][0]
    sel = pd.read_csv(RES / "selection.csv"); pol = pd.read_csv(RES / "policies.csv")
    uns = pd.read_csv(RES / "unseen.csv")
    mean = pol.groupby(["policy", "budget"]).eer.mean() * 100
    rows = []
    T = rule_table(sel)
    for dl, b in [(6, 2000), (10, 4000), (15, 8000), (25, 16000), (40, 32000), (75, 64000)]:
        assert gprs.budget_for_deadline(dl, grid=BUDGETS) == b
        kn, un, rv = [], [], []
        for f in sorted(sel.fold.unique()):
            r0 = T[(f, b)]; p, sb = r0.policy, r0.source_budget
            r = pol[(pol.fold == f) & (pol.budget == sb) & (pol.policy == p)].iloc[0]
            kn.append(r.eer); rv.append(r.bpcer_at_thr)
            u = uns[(uns.fold == f) & (uns.budget == sb) & (uns.policy == p)].eer
            un.append(u.iloc[0] if len(u) else None)
        rows.append({"Deadline (s)": dl, "Budget (kB)": b // 1000, "Time (s)": round(gprs.upload_seconds_tcp(b), 1),
                     "Photo": round(mean[("photo", b)], 1), "Adaptive stills": round(mean[("adaptive", b)], 1),
                     "H.264 CA": round(mean[("video16_h264_ca", b)], 1),
                     "Rule, known": round(100 * sum(kn) / len(kn), 1),
                     "Rule, unseen": "–" if None in un else round(100 * sum(un) / len(un), 1),
                     "Rule, unseen fold range": "–" if None in un else f"{100*min(un):.1f}-{100*max(un):.1f}",
                     "Review": round(100 * sum(rv) / len(rv), 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t5, t6 = regime_table(), gprs_table()
    t5.to_csv(OUT / "table5_regimes.csv", index=False); t6.to_csv(OUT / "table6_gprs.csv", index=False)
    print(t5.to_string(index=False)); print(t6.to_string(index=False))
