"""Section 5.9, Tables 11 and 12: the deadline rule against budget rules from published adaptive-streaming work,
on three real cellular uplink traces.

An attempt starts at every whole second of each trace (after a 20 s warm-up, leaving 30 s at the
end). Each rule sees past one-second capacities (5 s for the harmonic mean, 10 s for the RobustMPC
error term, 20 s for FESTIVE with its original window) and the RTT, plans for the deadline minus the
measured processing overhead (Table 13), picks a budget from
the grid, and the upload is replayed on the trace (idpad.traces.deliver). An attempt is late
when its delivery time exceeds the deadline. A budget is scored with the known-attack EER of
the plan the rule selects on validation at that budget (Table 9, 'Rule, known'; plan table of idpad.plan_table). 'On time and
correct' = (1 - late) x (1 - EER). Uncertainty: moving-block bootstrap (30 s blocks, resampled
within each trace).

Run from the repository root:
  bash scripts/get_traces.sh && python scripts/trace_comparison.py --boot 10000
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "src")
from idpad.traces import Trace, deliver, estimates, choose, harmonic, GRID
from idpad.plan_table import rule_table

TRACES = ["Verizon-EVDO-driving.up", "TMobile-UMTS-driving.up", "ATT-LTE-driving.up"]
POLICIES = ["fixed4", "fixed64", "festive", "festive20", "robustmpc", "linear_setup", "eq8", "eq8_robust", "oracle"]
# 300 ms with a 1 s deadline is left out: three round trips plus the measured processing time exceed it.
SETTINGS = [(100, 1.0), (100, 1.5), (100, 2.0), (100, 3.0), (300, 1.5), (300, 2.0), (300, 3.0)]
PAIRS = [("eq8", "festive"), ("eq8_robust", "robustmpc"), ("eq8_robust", "linear_setup"), ("eq8_robust", "fixed4")]
CANDIDATES = ["photo", "adaptive", "video16", "video16_h264", "video16_h264_ca"]


def rule_eer(res):
    """Known-attack test EER of the plan-table plan at each budget, mean of five folds (Table 9)."""
    sel, pol = pd.read_csv(res / "selection.csv"), pd.read_csv(res / "policies.csv")
    T = rule_table(sel); out = {}
    for b in GRID:
        v = [pol[(pol.fold == f) & (pol.budget == T[(f, b)].source_budget) & (pol.policy == T[(f, b)].policy)].iloc[0].eer
             for f in sorted(sel.fold.unique())]
        out[b] = float(np.mean(v))
    return out


def simulate(trace_dir, overhead):
    """Every rule plans for deadline - overhead (encoding + server time) and the attempt takes
    overhead + its replayed network time."""
    rows = []
    for name in TRACES:
        tr = Trace(str(Path(trace_dir) / name))
        for s in range(21, len(tr.kbps) - 30):
            h, hr = estimates(tr, s); h20 = harmonic(tr.kbps[s - 20:s])
            for rtt, dl in SETTINGS:
                net = dl - overhead
                t = {b: overhead + deliver(tr, s * 1000, b, rtt) for b in GRID}
                for p in POLICIES:
                    if p == "oracle":
                        ok = [b for b in GRID if t[b] <= dl]
                        b = max(ok) if ok else GRID[0]
                    else:
                        b = choose(p, h, hr, net, rtt, h20=h20)
                    rows.append((name, s, rtt, dl, p, b, t[b]))
    return pd.DataFrame(rows, columns=["trace", "start", "rtt_ms", "deadline", "policy", "budget", "time"])


def block_boot(df, pairs, n_boot, block=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for (rtt, dl), g in df.groupby(["rtt_ms", "deadline"]):
        w = g.pivot_table(index=["trace", "start"], columns="policy", values=["late", "good"])
        for a, b in pairs:
            dl_late = (w["late"][a] - w["late"][b]).groupby(level=0)
            dl_good = (w["good"][a] - w["good"][b]).groupby(level=0)
            reps_late, reps_good = np.zeros(n_boot), np.zeros(n_boot)
            ntot = 0
            for tr in w.index.get_level_values(0).unique():
                xl, xg = dl_late.get_group(tr).to_numpy(), dl_good.get_group(tr).to_numpy()
                n = len(xl); nb = int(np.ceil(n / block)); ntot += nb * block
                cl = np.concatenate([[0], np.cumsum(np.concatenate([xl, xl[:block]]))])
                cg = np.concatenate([[0], np.cumsum(np.concatenate([xg, xg[:block]]))])
                st = rng.integers(0, n, size=(n_boot, nb))
                reps_late += (cl[st + block] - cl[st]).sum(1); reps_good += (cg[st + block] - cg[st]).sum(1)
            reps_late, reps_good = 100 * reps_late / ntot, 100 * reps_good / ntot
            pt_l, pt_g = 100 * (w["late"][a] - w["late"][b]).mean(), 100 * (w["good"][a] - w["good"][b]).mean()
            out.append({"RTT (ms)": rtt, "Deadline (s)": dl, "Comparison": f"{a} - {b}",
                        "Late diff (pp)": round(pt_l, 1),
                        "Late 95% CI": f"[{np.percentile(reps_late, 2.5):.1f}, {np.percentile(reps_late, 97.5):.1f}]",
                        "Good diff (pp)": round(pt_g, 1),
                        "Good 95% CI": f"[{np.percentile(reps_good, 2.5):.1f}, {np.percentile(reps_good, 97.5):.1f}]"})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces"); ap.add_argument("--res", default="results/v3")
    ap.add_argument("--out", default="results/paper"); ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--cost", default="results/paper/table11_cost.json",
                    help="Table 13 timings; overhead = p95 of the 4 kB H.264 rate-factor search + p95 server time")
    ap.add_argument("--overhead", type=float, default=None, help="override the overhead in seconds")
    a = ap.parse_args()
    if a.overhead is None:
        import json
        c = json.load(open(a.cost))["ms_median_p95"]
        a.overhead = round((c["Client: H.264, 16 frames, scale 0.25, rate-factor search for 4 kB"][1]
                            + c["Server: decode H.264 stream, embed and score 16 frames"][1]) / 1000, 3)
    print("overhead (s):", a.overhead)
    eer = rule_eer(Path(a.res))
    df = simulate(a.traces, a.overhead)
    df["late"] = (df.time > df.deadline).astype(float)
    df["eer"] = df.budget.map(eer)
    df["good"] = (1 - df.late) * (1 - df.eer)
    out = Path(a.out)
    df.to_csv(out / "trace_attempts.csv.gz", index=False)
    summ = df.groupby(["rtt_ms", "deadline", "policy"]).agg(late=("late", "mean"), eer=("eer", "mean"),
                                                             good=("good", "mean"), kB=("budget", "mean"))
    summ[["late", "eer", "good"]] *= 100; summ["kB"] /= 1000
    summ = summ.round(1).reset_index()
    late = summ.pivot_table(index=["rtt_ms", "deadline"], columns="policy", values="late")[POLICIES].reset_index()
    good = summ.pivot_table(index=["rtt_ms", "deadline"], columns="policy", values="good")[POLICIES].reset_index()
    late.to_csv(out / "table8_late.csv", index=False); good.to_csv(out / "table8_good.csv", index=False)
    summ.to_csv(out / "table8_summary.csv", index=False)
    per_trace = (df.groupby(["trace", "rtt_ms", "deadline", "policy"]).late.mean() * 100).round(1).unstack()[POLICIES]
    per_trace.to_csv(out / "table8_late_by_trace.csv")
    tests = block_boot(df, PAIRS, a.boot)
    tests.to_csv(out / "table8_tests.csv", index=False)
    print("EER by budget:", {k // 1000: round(100 * v, 1) for k, v in eer.items()})
    print(late.to_string(index=False)); print(good.to_string(index=False)); print(tests.to_string(index=False))
    caps = {n: Trace(str(Path(a.traces) / n)).kbps for n in TRACES}
    for n, c in caps.items():
        print(n, f"{len(c)} s, mean {c.mean():.0f} kbit/s, median {np.median(c):.0f}, p10 {np.percentile(c, 10):.0f}, "
                 f"zero seconds {100 * (c == 0).mean():.1f}%")


if __name__ == "__main__":
    main()
