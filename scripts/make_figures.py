"""Figures for the Experiment 3 mechanistic analysis (CPU only). Run from the repo root:  python scripts/make_figures.py

  expert_norm_decay.png/.gif : norm_A of experts 0 and 1 across the task checkpoints        (analysis/exp3/expert_norms.csv)
  restore_comparison.png/.gif: original final / control / restored success per task         (analysis/exp3/restore_test_results.json|.txt)
  router_drift.png, expert_trajectory.png: copies of the existing Part A / Part B figures (not regenerated)
All files go to analysis/figures/ at 150 DPI. The GIFs reveal the same chart step by step.
"""
import csv, io, json, os, re, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SRC, OUT, DPI = "analysis/exp3", "analysis/figures", 150
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "savefig.dpi": DPI})
BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#D55E00", "#009E73", "#8C8C8C", "#D62728"


def gif(frames, path, ms=550, hold_ms=2200):
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[ms] * (len(imgs) - 1) + [hold_ms], loop=0, optimize=True)


def render(draw, steps, png, gif_path):
    """draw(step) -> figure; the last step is the saved PNG, every step is a GIF frame."""
    frames = []
    for s in steps:
        fig = draw(s)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        frames.append(buf.getvalue())
        if s == steps[-1]:
            fig.savefig(png, dpi=DPI)
        plt.close(fig)
    gif(frames, gif_path)


# ---------- Plot 1: expert norm decay ----------
rows = list(csv.DictReader(open(f"{SRC}/expert_norms.csv")))
ck = next(k for k in rows[0] if "checkpoint" in k)
series = {e: sorted((int(r[ck]), float(r["norm_A"])) for r in rows if int(r["expert_id"]) == e) for e in (0, 1)}
colors = {0: BLUE, 1: ORANGE}


def draw1(upto):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for e, pts in series.items():
        x0, y0 = pts[0]
        ax.axhline(y0, color=colors[e], ls="--", lw=1.1, alpha=0.55)
        ax.text(9.55, y0, f"creation norm {y0:.2f}", color=colors[e], va="bottom", ha="right", fontsize=9)
        vis = [(x, y) for x, y in pts if x <= upto]
        ax.plot([p[0] for p in vis], [p[1] for p in vis], "-o", color=colors[e], lw=2.2, ms=5, mfc="white", label=f"Expert {e}")
        if x0 <= upto:
            ax.plot([x0], [y0], "o", color=colors[e], ms=11, zorder=5)
        if vis[-1][0] == 9:
            ax.annotate(f"{vis[-1][1] / y0:.2f}× creation", (9, vis[-1][1]), xytext=(9.1, vis[-1][1]), color=colors[e], fontsize=10,
                        va="center", fontweight="bold")
    ax.plot([], [], "o", color="0.3", ms=11, label="checkpoint where the expert was created")
    ax.set_xlim(-0.4, 10.5)
    ax.set_ylim(4.0, 9.4)
    ax.set_xticks(range(10))
    ax.set_xlabel("Task checkpoint (weights saved after training task k)")
    ax.set_ylabel("L2 norm of the expert's A matrices (norm_A)")
    ax.set_title("AdamW Weight Decay Silently Shrinks Frozen LoRA Experts", fontweight="bold", loc="left")
    ax.legend(loc="lower left", frameon=False)
    fig.text(0.01, 0.01, "DMPEL on LIBERO-Long, seed 100. AdamW lr 1e-4 (OneCycle), weight decay 0.1. Frozen experts get zero gradient but are still decayed.",
             fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


render(draw1, list(range(1, 10)), f"{OUT}/expert_norm_decay.png", f"{OUT}/expert_norm_decay.gif")

# ---------- Plot 2: restore test ----------
jp = f"{SRC}/restore_test_results.json"
if os.path.exists(jp):
    d = json.load(open(jp))
    orig, ctrl, rest = ([float(d[k][str(j)]) for j in range(10)] for k in ("original_final", "control", "restored"))
else:  # parse the text table
    tab = [list(map(float, re.findall(r"[-+]?\d+\.?\d*", l.split("|")[0] + "|" + "|".join(l.split("|")[1:4])))) for l in open(f"{SRC}/restore_test_results.txt")
           if re.match(r"\s*\d+ \|", l)]
    orig, ctrl, rest = ([r[i] for r in tab] for i in (1, 2, 3))
assert len(orig) == len(ctrl) == len(rest) == 10
conds = [("Original final (run's last row)", orig, GREY), ("Control (final model re-evaluated)", ctrl, BLUE), ("Restored (experts reset to creation-time weights)", rest, GREEN)]


def draw2(n):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x, w = np.arange(10), 0.27
    for i, (name, vals, col) in enumerate(conds[:n]):
        bars = ax.bar(x + (i - 1) * w, vals, w, color=col, label=name, edgecolor="white", linewidth=0.6)
        bars[1].set_edgecolor(RED)
        bars[1].set_linewidth(2.4)
    if n == 3:
        ax.annotate("1.0 (p=0.0007)", (1 + w, rest[1]), xytext=(1 + w + 0.55, 1.1), fontsize=10, fontweight="bold", color=RED, ha="left",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(x)
    ax.set_xlabel("Task index (LIBERO-Long order; red outline = task index 1)")
    ax.set_ylabel("Success rate (10 rollouts per bar)")
    ax.set_title("Restoring Expert Weights Recovers Forgotten Tasks", fontweight="bold", loc="left")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "DMPEL on LIBERO-Long, seed 100, final model after task 9. Same router in all conditions. p: Fisher exact test, restored vs control (10/10 vs 2/10).",
             fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


render(draw2, [1, 2, 3], f"{OUT}/restore_comparison.png", f"{OUT}/restore_comparison.gif")

# ---------- Plot 3: copies of existing figures (not regenerated) ----------
shutil.copyfile(f"{SRC}/part_a_task_drift.png", f"{OUT}/router_drift.png")
shutil.copyfile(f"{SRC}/part_b_trajectory_comparison.png", f"{OUT}/expert_trajectory.png")

print("\nChecklist:")
for f in ("expert_norm_decay.png", "restore_comparison.png", "router_drift.png", "expert_trajectory.png"):
    p = f"{OUT}/{f}"
    im = Image.open(p)
    print(f"  [{'done' if os.path.getsize(p) > 0 else 'MISSING'}] {f}  {im.size[0]}x{im.size[1]} px, {os.path.getsize(p) // 1024} kB")
for f in ("expert_norm_decay.gif", "restore_comparison.gif"):
    print(f"  [{'done' if os.path.exists(f'{OUT}/{f}') else 'MISSING'}] {f} ({os.path.getsize(f'{OUT}/{f}') // 1024} kB, {Image.open(f'{OUT}/{f}').n_frames} frames)")
