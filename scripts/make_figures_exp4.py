"""Experiment 2 (decayed) vs Experiment 4 (fixed) comparison figures. CPU only; run from the repo root:  python scripts/make_figures_exp4.py

  success_matrix_comparison.png : two 10x10 success-matrix heatmaps          (results/*.json, key c_matrix)
  final_sr_bar.png              : final-row success per task, both runs       (results/*.json)
  expert_norm_comparison.png    : norm_A of experts 0 and 1 per checkpoint    (analysis/exp3/expert_norms.csv, analysis/exp4/expert_norms_fixed.txt)
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT, DPI = "analysis/figures", 150
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": DPI})
N2 = "D-long10-dmpel-full-rho005-bias-ne10-original-s100"
N4 = "D-long10-dmpel-full-rho005-bias-ne10-FIXEDWD-original-s100"
RED, GREEN = "#C0392B", "#1E8449"


def cmat(name):
    c = json.load(open(f"results/{name}.json"))["c_matrix"]
    return np.array([[np.nan if v is None else v for v in row] for row in c], float)


C2, C4 = cmat(N2), cmat(N4)
assert C2.shape == C4.shape == (10, 10) and not np.isnan(C2[9]).any() and not np.isnan(C4[9]).any()

# ---- 1. success matrices ----
cm = LinearSegmentedColormap.from_list("w2g", ["#FFFFFF", "#00441B"])
cm.set_bad("#E6E6E6")
fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
for ax, C, name, label in ((axes[0], C2, N2, "Experiment 2: decayed experts"), (axes[1], C4, N4, "Experiment 4: fixed experts")):
    im = ax.imshow(np.ma.masked_invalid(C), cmap=cm, vmin=0, vmax=1)
    for t in range(10):
        for k in range(t + 1):
            ax.text(k, t, f"{C[t, k]:.1f}", ha="center", va="center", fontsize=9, color="white" if C[t, k] > 0.55 else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Evaluated task")
    ax.set_title(f"{name}\n{label}", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("Evaluated after training task")
fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="Success rate (10 rollouts per cell)")
fig.savefig(f"{OUT}/success_matrix_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---- 2. final success per task ----
fig, ax = plt.subplots(figsize=(10, 5.2))
x, w = np.arange(10), 0.38
b2 = ax.bar(x - w / 2, C2[9], w, color=RED, label="Experiment 2 (decayed)")
b4 = ax.bar(x + w / 2, C4[9], w, color=GREEN, label="Experiment 4 (fixed)")
for bars in (b2, b4):
    bars[1].set_edgecolor("black")
    bars[1].set_linewidth(2.2)
ax.set_xticks(x)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Task index (black outline = task 1)")
ax.set_ylabel("Final success rate (10 rollouts per bar)")
ax.set_title("Per-task final success: decayed vs fixed experts", fontweight="bold", loc="left")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/final_sr_bar.png")
plt.close(fig)

# ---- 3. expert norm_A ----
def csv_series(path):
    rows = list(csv.DictReader(open(path)))
    ck = next(k for k in rows[0] if "checkpoint" in k)
    return {e: sorted((int(r[ck]), float(r["norm_A"])) for r in rows if int(r["expert_id"]) == e) for e in (0, 1)}


def txt_series(path):
    pts = {0: [], 1: []}
    for line in open(path):
        f = line.split()
        if len(f) >= 6 and f[0].isdigit() and f[1].isdigit() and int(f[1]) in pts:
            pts[int(f[1])].append((int(f[0]), float(f[2])))
    return {e: sorted(v) for e, v in pts.items()}


dec, fix = csv_series("analysis/exp3/expert_norms.csv"), txt_series("analysis/exp4/expert_norms_fixed.txt")
assert len(dec[0]) == 10 and len(fix[0]) == 10 and len(dec[1]) == 9 and len(fix[1]) == 9
fig, ax = plt.subplots(figsize=(9.5, 5.4))
col = {0: "#0072B2", 1: "#D55E00"}
for e in (0, 1):
    ax.plot(*zip(*dec[e]), "--o", color=col[e], lw=2, ms=4, label=f"Expert {e}, decayed (Experiment 2)")
    ax.plot(*zip(*fix[e]), "-o", color=col[e], lw=2, ms=4, label=f"Expert {e}, fixed (Experiment 4)")
ax.set_xticks(range(10))
ax.set_xlabel("Task checkpoint (weights saved after training task k)")
ax.set_ylabel("norm_A (L2 norm of the expert's A matrices)")
ax.set_title("Expert norm decay eliminated by optimizer fix", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)  # below the axes: nothing is covered
fig.tight_layout()
fig.savefig(f"{OUT}/expert_norm_comparison.png", bbox_inches="tight")
plt.close(fig)

for f in ("success_matrix_comparison.png", "final_sr_bar.png", "expert_norm_comparison.png"):
    print("[done]", f, os.path.getsize(f"{OUT}/{f}") // 1024, "kB")
