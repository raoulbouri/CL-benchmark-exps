"""Experiment 3: router analysis on the audit routing logs (CPU only, read-only).

Usage: python scripts/analyze_routing.py [--out analysis/exp3]

Inputs (written by scripts/patch_dmpel_audit.py): outputs/<run>/a*/seed_*/routing/eval####_task{J}_{sel|matrix}.npz
  topk_idx, topk_coeff : (calls * rows, 6 groups * topk)  the router's selected experts and weights per step per env
  call_loop, call_step, call_rows : which loop/step each call belongs to; rows = parallel envs (5)
  act_grip (Long only) : commanded gripper (+1 / -1) per step per env
File order: for training stage K, the 6 selection evaluations of task K, then one matrix evaluation per task J = 0..K.
Part A: does the router send old-task frames elsewhere as later tasks arrive (drift), and does that track the success drop?
Part B: does the router switch experts mid-episode more on Long than on Goal, and at gripper changes?
"""
import argparse
import glob
import json
import os
import re

import numpy as np

P = 10  # final pool size (one expert per task)
RUNS = {
    "long": "outputs/D-long10-dmpel-full-rho005-bias-ne10-original-s100",
    "goal": "outputs/D-goal10-dmpel-full-original-s100",
}
RNG = np.random.default_rng(0)


def run_files(run):
    base = sorted(glob.glob(f"{RUNS[run]}/a*/seed_*"))[-1]
    meta, K = [], None
    for f in sorted(glob.glob(f"{base}/routing/eval*.npz")):
        m = re.search(r"eval(\d+)_task(\d+)_(sel|matrix)\.npz", f)
        J, kind = int(m.group(2)), m.group(3)
        if kind == "sel":
            K = J
        meta.append(dict(path=f, K=K, J=J, kind=kind))
    return meta


def episodes(path):
    """One dict per episode (env row within a loop): idx/co (T, 6, k), grip (T,) or None.

    Only genuine rollout calls are used: 5 rows (one per env) and a step counter that continues the loop's chain
    1, 2, 3, ... The logs also hold eval-mode forward passes that are not rollouts (buffer pass, 32-64 rows, stale
    step counter); those are dropped. The commanded gripper (Long only) is matched by (loop, step).
    """
    z = np.load(path)
    z = {name: z[name] for name in z.files}  # NpzFile decompresses on every access: read each array once
    rows, loop, step = z["call_rows"], z["call_loop"], z["call_step"]
    off = np.concatenate([[0], np.cumsum(rows)])
    last, keep = {}, []
    for i in range(len(rows)):
        l = int(loop[i])
        if rows[i] == 5 and int(step[i]) == last.get(l, 0) + 1:
            keep.append(i)
            last[l] = int(step[i])
    k = z["topk_idx"].shape[1] // 6
    grip = {}
    if "act_grip" in z:
        ag = z["act_grip"].reshape(-1, 5).astype(np.float32)
        grip = {(int(l), int(s)): ag[i] for i, (l, s) in enumerate(zip(z["act_loop"], z["act_step"]))}
    out = []
    for l in sorted(set(int(loop[i]) for i in keep)):
        ci = [i for i in keep if int(loop[i]) == l]
        idx = np.stack([z["topk_idx"][off[i]:off[i + 1]] for i in ci]).reshape(len(ci), 5, 6, k).astype(int)
        co = np.stack([z["topk_coeff"][off[i]:off[i + 1]] for i in ci]).reshape(len(ci), 5, 6, k).astype(np.float32)
        g = np.stack([grip[(l, int(step[i]))] for i in ci]) if grip and all((l, int(step[i])) in grip for i in ci) else None
        for r in range(5):
            out.append(dict(idx=idx[:, r], co=co[:, r], grip=None if g is None else g[:, r]))
    return out


def dense(ep):
    """(T, 6, P): router weight of every expert; zero for experts outside the top-k."""
    d = np.zeros(ep["idx"].shape[:2] + (P,), np.float32)
    np.put_along_axis(d, ep["idx"], ep["co"], axis=-1)
    return d


def mean_vec(eps):
    return np.concatenate([dense(e).reshape(-1, 6 * P) for e in eps]).mean(0)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def top1(ep):
    """(T, 6) id of the expert with the largest weight in each group."""
    return np.take_along_axis(ep["idx"], ep["co"].argmax(-1)[..., None], -1)[..., 0]


def switches(ep):
    t = top1(ep)
    return (t[1:] != t[:-1]).any(-1).astype(float), (t[1:] != t[:-1]).mean(-1)  # any-group flag, fraction of groups


def rank(x):
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def spearman(a, b, perms=5000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rho = float(np.corrcoef(rank(a), rank(b))[0, 1])
    null = np.array([np.corrcoef(rank(a), rank(RNG.permutation(b)))[0, 1] for _ in range(perms)])
    return rho, float((np.abs(null) >= abs(rho)).mean())


def part_a(meta, C, out):
    mat = {(m["K"], m["J"]): m for m in meta if m["kind"] == "matrix"}
    vec, hij, own = {}, {}, {}
    for (K, J), m in mat.items():
        eps = episodes(m["path"])
        vec[(K, J)] = mean_vec(eps)
        slots = np.concatenate([e["idx"].reshape(-1) for e in eps])
        hij[(K, J)] = float((slots > J).mean())  # share of selections that go to experts created after task J
        own[(K, J)] = float((slots == J).mean())
    # noise floor: two random halves of the episodes of one evaluation (conservative: halves are noisier than full samples)
    floor = []
    for (K, J), m in mat.items():
        if K == 0:
            continue
        eps = episodes(m["path"])
        perm = RNG.permutation(len(eps))
        h = len(eps) // 2
        floor.append(cos(mean_vec([eps[i] for i in perm[:h]]), mean_vec([eps[i] for i in perm[h:]])))
    cells = [(K, J) for (K, J) in mat if K > J]
    cs = np.array([cos(vec[(J, J)], vec[(K, J)]) for K, J in cells])
    drop = np.array([C[J][J] - C[K][J] for K, J in cells])
    rho, p = spearman(1 - cs, drop)
    print("\n=== Part A: routing drift of old tasks ===")
    print(f"noise floor (split-half cosine, same weights): median {np.median(floor):.3f}, 5th pct {np.percentile(floor, 5):.3f}, n={len(floor)}")
    for J in (1, 0):
        print(f"\ntask index {J} ({'forgotten: 1.0 -> 0.1' if J == 1 else 'control: stable'}); baseline = evaluation right after learning it")
        print(f"{'after task':>10} {'success':>8} {'cos to baseline':>16} {'share->newer experts':>21} {'share->own expert':>18}")
        for K in range(J, 10):
            c = 1.0 if K == J else cos(vec[(J, J)], vec[(K, J)])
            print(f"{K:>10} {C[K][J]:>8.1f} {c:>16.3f} {hij[(K, J)]:>21.3f} {own[(K, J)]:>18.3f}")
    print(f"\nacross all {len(cells)} off-diagonal cells: Spearman(routing drift = 1-cos, success drop) = {rho:.3f}, permutation p = {p:.4f}")
    print("per-task minimum cosine to baseline:", {J: round(min(cos(vec[(J, J)], vec[(K, J)]) for K in range(J + 1, 10)), 3) for J in range(9)})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for J in range(9):
            ks = list(range(J, 10))
            ax.plot(ks, [1.0 if K == J else cos(vec[(J, J)], vec[(K, J)]) for K in ks], color="0.8", lw=1)
        ks = list(range(1, 10))
        ax.plot(ks, [1.0 if K == 1 else cos(vec[(1, 1)], vec[(K, 1)]) for K in ks], "o-", color="C3", label="task index 1 (forgotten)")
        ks = list(range(0, 10))
        ax.plot(ks, [1.0 if K == 0 else cos(vec[(0, 0)], vec[(K, 0)]) for K in ks], "s-", color="C0", label="task index 0 (stable)")
        ax.axhline(np.percentile(floor, 5), color="k", ls=":", label="noise floor (5th pct, split-half)")
        ax.set_xlabel("after training task K")
        ax.set_ylabel("cosine to baseline routing")
        ax.set_title("Router drift of earlier tasks (Long, grey = other tasks)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out}/part_a_task_drift.png", dpi=150)
    except Exception as e:  # plotting must never hide the numbers
        print("plot skipped:", e)
    return dict(floor_median=float(np.median(floor)), floor_p5=float(np.percentile(floor, 5)), spearman=rho, p=p, n_cells=len(cells))


def episode_metrics(eps, window=None):
    sw_any, sw_frac, var = [], [], []
    for e in eps:
        a, f = switches(e)
        d = dense(e).reshape(-1, 6 * P)
        if window:
            a, f, d = a[:window], f[:window], d[:window]
        sw_any.append(a.mean())
        sw_frac.append(f.mean())
        var.append(((d - d.mean(0)) ** 2).sum(1).mean())
    return np.array(sw_any), np.array(sw_frac), np.array(var)


def boot_ratio(x, y, reps=2000):
    r = [np.mean(RNG.choice(x, len(x))) / max(np.mean(RNG.choice(y, len(y))), 1e-12) for _ in range(reps)]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def part_b(metas, out):
    res = {}
    for run in ("long", "goal"):
        per = {}
        for m in metas[run]:
            if m["kind"] == "matrix" and m["K"] == m["J"] and m["K"] >= 1:  # just learned, router has >= 2 experts
                per[m["K"]] = episodes(m["path"])
        res[run] = per
    print("\n=== Part B: mid-episode routing, Long vs Goal (just-learned evaluations, tasks 1..9) ===")
    summary = {}
    cache = {(r, K, w): episode_metrics(res[r][K], w) for r in res for K in range(1, 10) for w in (None, 200)}
    for name, window in (("all logged steps", None), ("first 200 steps", 200)):
        agg = {r: [np.concatenate([cache[(r, K, window)][i] for K in range(1, 10)]) for i in range(3)] for r in res}
        print(f"\nwindow: {name}  (episodes: long {len(agg['long'][0])}, goal {len(agg['goal'][0])})")
        for i, label in enumerate(["expert-switch rate (any group)", "expert-switch rate (mean over groups)", "coefficient variance over time"]):
            lo, hi = boot_ratio(agg["long"][i], agg["goal"][i])
            print(f"  {label:<40} long {agg['long'][i].mean():.4f}  goal {agg['goal'][i].mean():.4f}  ratio {agg['long'][i].mean() / agg['goal'][i].mean():.2f}x  (95% CI {lo:.2f}-{hi:.2f})")
            summary[f"{name}|{label}"] = dict(long=float(agg["long"][i].mean()), goal=float(agg["goal"][i].mean()), ci=[lo, hi])
    print("\nper-task switch rate (any group, all steps): task: long / goal")
    for K in range(1, 10):
        l, g = cache[("long", K, None)][0].mean(), cache[("goal", K, None)][0].mean()
        print(f"  {K}: {l:.4f} / {g:.4f}  ({l / max(g, 1e-9):.2f}x)")
    # Long only: are switches concentrated around gripper changes?
    print("\nLong: expert switches within +-5 steps of a gripper open/close command vs elsewhere (all matrix evals with pool >= 2)")
    eps_sw, eps_mask, n_flips = [], [], 0
    for m in metas["long"]:
        if m["kind"] != "matrix" or m["K"] < 1:
            continue
        for e in episodes(m["path"]):
            sw, _ = switches(e)
            flips = np.where(np.abs(np.diff(e["grip"])) > 1)[0]
            if len(flips) == 0 or len(sw) < 20:
                continue
            n_flips += len(flips)
            mask = np.zeros(len(sw), bool)
            for f in flips:
                mask[max(0, f - 5):min(len(sw), f + 6)] = True
            eps_sw.append(sw)
            eps_mask.append(mask)

    def pooled(masks):
        n_near = sum(s[m].sum() for s, m in zip(eps_sw, masks))
        c_near = sum(m.sum() for m in masks)
        n_far = sum(s[~m].sum() for s, m in zip(eps_sw, masks))
        c_far = sum((~m).sum() for m in masks)
        return n_near / c_near, n_far / c_far

    near, far = pooled(eps_mask)
    null = np.array([(lambda r: r[0] / r[1])(pooled([np.roll(m, RNG.integers(1, len(m))) for m in eps_mask])) for _ in range(300)])
    obs = near / far
    print(f"  episodes with gripper changes: {len(eps_sw)}, changes: {n_flips}; switch prob near {near:.4f} vs far {far:.4f}; ratio {obs:.2f}x; "
          f"circular-shift null {null.mean():.2f}x (95th pct {np.percentile(null, 95):.2f}x)")
    summary["gripper_enrichment"] = dict(ratio=float(obs), near=float(near), far=float(far), null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)), n_flips=int(n_flips))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        for ax, run in zip(axes, ("long", "goal")):
            e = res[run][3][0]
            ax.imshow(top1(e).T, aspect="auto", cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
            ax.set_yticks(range(6))
            ax.set_yticklabels(["img", "txt", "extra", "fusion", "tem", "head"], fontsize=7)
            ax.set_title(f"{run.capitalize()}, task index 3, just learned, episode 0: top-1 expert per group (colour = expert id)", fontsize=9)
            if e["grip"] is not None:
                for f in np.where(np.abs(np.diff(e["grip"])) > 1)[0]:
                    ax.axvline(f, color="k", ls="--", lw=1)
        axes[1].set_xlabel("timestep (dashed = gripper open/close command; the Goal run has no gripper log)")
        fig.tight_layout()
        fig.savefig(f"{out}/part_b_trajectory_comparison.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/exp3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    metas = {r: run_files(r) for r in RUNS}
    for r in metas:
        assert len(metas[r]) == 115, (r, len(metas[r]))
    C = json.load(open("results/D-long10-dmpel-full-rho005-bias-ne10-original-s100.json"))["c_matrix"]
    C = [[(0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else v) for v in row] + [0.0] * (10 - len(row)) for row in C]
    summary = dict(part_a=part_a(metas["long"], C, a.out), part_b=part_b(metas, a.out))
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    print(f"\nsaved figures and summary.json in {a.out}")
