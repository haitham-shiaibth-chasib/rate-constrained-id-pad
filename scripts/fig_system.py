"""Fig. 1: (a) the management loop of Sec. 3.3 in MAPE-K terms, (b) the data path of one attempt.

Run from the repository root:  python scripts/fig_system.py results/paper/fig1_pipeline.png
(panel b is drawn by scripts/fig_pipeline.py into results/paper/fig1_datapath.png first)
"""
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg

plt.rcParams.update({"font.family": "serif", "font.size": 8})


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015", fc=fc, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=8, weight="bold")
    ax.text(x + w / 2, y + h - 0.085, body, ha="center", va="top", fontsize=7, linespacing=1.25)


def arrow(ax, a, b, text=None, rad=0.0, off=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="center", fontsize=6.5,
                style="italic", bbox=dict(fc="white", ec="none", pad=0.5))


def main(out):
    fig = plt.figure(figsize=(7.0, 6.4), dpi=300)
    ax = fig.add_axes([0, 0.36, 1, 0.64]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.985, "(a)", fontsize=9, weight="bold", va="top")
    ax.text(0.24, 0.975, "Client (phone)", ha="center", fontsize=8.5, weight="bold", va="top")
    ax.text(0.76, 0.975, "Server (verification service)", ha="center", fontsize=8.5, weight="bold", va="top")
    c1, c2, c3 = "#e8f0fb", "#fdf1e3", "#eaf6ea"
    L, R, Wd = 0.03, 0.59, 0.38
    box(ax, L, 0.66, Wd, 0.25, "Monitor (per attempt)", "RTT from the TCP handshake;\nrecent goodput, robust estimate R̂;\nlog: budget, plan, delivery time,\ndeadline met, score", c1)
    box(ax, L, 0.36, Wd, 0.24, "Plan, online (label-free)", "B* = max{B : T(B; R̂, RTT) + T_enc ≤ D}\n(Eq. 2); smallest budget and a\nnotice if none fits", c1)
    box(ax, L, 0.06, Wd, 0.24, "Execute", "capture 16 frames; encode with\np*(B*) at the best quality or rate\nfactor that fits B*; HTTPS upload", c1)
    box(ax, R, 0.66, Wd, 0.25, "Knowledge", "versioned plan table\nB ↦ (p*(B), t(B), model version),\ncumulative, documented tie-break", c3)
    box(ax, R, 0.36, Wd, 0.24, "Plan, offline (labelled)", "on retraining or encoder change:\nre-select p*(B) and t(B) on\nvalidation data", c2)
    box(ax, R, 0.06, Wd, 0.24, "Analyse", "score s against t(B*): accept,\nreview, or |s − t| ≤ δ: ask for B₂;\nwatch miss rate and score drift", c2)
    arrow(ax, (L + Wd / 2, 0.66), (L + Wd / 2, 0.60)); arrow(ax, (L + Wd / 2, 0.36), (L + Wd / 2, 0.30))
    arrow(ax, (R + Wd / 2, 0.30), (R + Wd / 2, 0.36)); arrow(ax, (R + Wd / 2, 0.60), (R + Wd / 2, 0.66))
    ax.text(R + Wd / 2 + 0.01, 0.33, "review labels, drift", fontsize=6.5, style="italic", va="center", ha="left")
    ax.text(R + Wd / 2 + 0.01, 0.63, "new table version", fontsize=6.5, style="italic", va="center", ha="left")
    arrow(ax, (L + Wd, 0.21), (R, 0.21)); ax.text(0.5, 0.235, "attempt (B* bytes)", ha="center", fontsize=6.5, style="italic")
    arrow(ax, (R, 0.13), (L + Wd, 0.13)); ax.text(0.5, 0.155, "decision; request for B₂", ha="center", fontsize=6.5, style="italic")
    arrow(ax, (R, 0.80), (L + Wd, 0.80)); ax.text(0.5, 0.825, "table pushed", ha="center", fontsize=6.5, style="italic")
    ax2 = fig.add_axes([0.0, 0.0, 1, 0.34]); ax2.axis("off")
    ax2.imshow(mpimg.imread("results/paper/fig1_datapath.png"))
    ax2.text(0.0, 1.0, "(b)", transform=ax2.transAxes, fontsize=9, weight="bold", va="top")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/paper/fig1_pipeline.png")
