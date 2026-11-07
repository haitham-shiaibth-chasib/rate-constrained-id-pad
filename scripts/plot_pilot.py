"""EER versus mean payload (bytes) per strategy, averaged over folds (test split)."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/pilot/results.csv")
    ap.add_argument("--detector", default="descriptor_lr")
    ap.add_argument("--out", default="results/pilot/eer_vs_bytes.png")
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    df = df[(df.split == "test") & (df.detector == a.detector) & (df.budget > 0) & (df.feasible == 1.0)]
    g = df.groupby(["strategy", "budget"]).agg(eer=("eer", "min"), nbytes=("mean_bytes", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, d in g.groupby("strategy"):
        d = d.sort_values("nbytes")
        ax.plot(d.nbytes / 1000, d.eer * 100, marker="o", label=s)
    ax.set_xscale("log"); ax.set_xlabel("Uplink payload per attempt (kB)"); ax.set_ylabel("EER (%)")
    ax.grid(alpha=.3); ax.legend(title="strategy (best plan per budget)")
    fig.tight_layout(); fig.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
