"""Figure 1: the verification pipeline studied in the paper."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = sys.argv[1] if len(sys.argv) > 1 else "results/paper/fig1_datapath.png"
import os; os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"], "font.size": 10})
fig, ax = plt.subplots(figsize=(6.8, 2.6), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, linespacing=1.3)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#333333"))

ax.text(16, 37.5, "Phone (client)", ha="center", fontsize=9.5, weight="bold")
ax.text(50, 37.5, "Uplink", ha="center", fontsize=9.5, weight="bold")
ax.text(84, 37.5, "Server", ha="center", fontsize=9.5, weight="bold")
box(3, 25, 26, 8, "Tilt video of the ID card\n(16 frames sampled)", "#e8eef6")
box(3, 13, 26, 8, "Keep K frames\nK = 1, 2, 4, 8, 16", "#e8eef6")
box(3, 1, 26, 8, "Encode: codec, scale, quality\ntotal size ≤ budget B", "#e8eef6")
box(37, 13, 26, 12, "B bytes per attempt\nrate, RTT, slow start\n→ upload time", "#f3f3f3")
box(71, 25, 26, 8, "Decode, MobileNetV3\nembedding per frame", "#eef6ea")
box(71, 13, 26, 8, "Clip vector: mean, std,\nmean |frame difference|", "#eef6ea")
box(71, 1, 26, 8, "Logistic regression\n→ attack score", "#eef6ea")
arrow(16, 25, 16, 21.6); arrow(16, 13, 16, 9.6)
arrow(29.6, 5, 36.6, 15); arrow(63.4, 22, 70.4, 28)
arrow(84, 25, 84, 21.6); arrow(84, 13, 84, 9.6)
ax.text(51.5, 5.5, "Plan = (K, codec, scale) chosen per\nbudget on validation data only", ha="center", fontsize=7.5, style="italic")
fig.savefig(out, bbox_inches="tight")
print(out)
