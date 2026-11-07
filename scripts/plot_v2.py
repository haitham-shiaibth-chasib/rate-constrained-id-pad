"""Figures for the v2 study: EER vs budget per K (best codec/scale) and per policy."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/v2")
    a = ap.parse_args()
    df = pd.read_csv(f"{a.dir}/configs.csv"); pol = pd.read_csv(f"{a.dir}/policies.csv")
    t = df[(df.split == "test") & (df.budget > 0)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, g in t.groupby("k"):
        m = g.groupby(["budget", "fold"]).eer.min().groupby("budget").mean().dropna()
        ax[0].plot(m.index / 1000, m * 100, marker="o", label=f"K = {k}")
    ax[0].set_title("Best codec/scale per K (test, mean of 5 folds)")
    for p, g in pol.groupby("policy"):
        m = g.groupby("budget").eer.mean().dropna()
        ax[1].plot(m.index / 1000, m * 100, marker="s", label=p)
    ax[1].set_title("Policies selected on validation (test)")
    for x in ax:
        x.set_xscale("log"); x.set_xticks([2, 4, 8, 16, 32, 64]); x.set_xticklabels(["2", "4", "8", "16", "32", "64"]); x.minorticks_off(); x.set_xlabel("Uplink payload per attempt (kB)"); x.set_ylabel("EER (%)")
        x.grid(alpha=.3); x.legend()
    fig.tight_layout(); fig.savefig(f"{a.dir}/v2_eer_vs_budget.png", dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
