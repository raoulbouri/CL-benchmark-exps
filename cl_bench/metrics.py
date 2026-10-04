"""Continual-learning metrics from a success matrix and per-task learning curves.

c[t][k] = success on task k after training through task t (row = training stage), as in LIBERO's
S_conf_mat. Rows are filled for t < n_done; entries with k > t are unused.

FWT_k uses the learning curve of task k (success at each selection evaluation), held at its peak
after the best evaluation (LIBERO definition). DMPEL's own logged S_fwd is NOT used: its
cumulated_counter only increments on improvement, which inflates it (can exceed 1).
"""
import numpy as np


def fwt_from_curve(curve):
    s = np.asarray(curve, dtype=float)
    if s.size == 0:
        return float("nan")
    best = int(np.argmax(s))  # first epoch reaching the peak
    s = s.copy()
    s[best:] = s[best]
    return float(s.mean())


def compute(c, curves, n_done=None):
    """Return a dict of metrics over the first n_done tasks."""
    c = np.asarray(c, dtype=float)
    n = int(n_done if n_done is not None else len(curves))
    if n == 0:
        return {"n_tasks_scored": 0}
    fwd = np.array([fwt_from_curve(curves[k]) for k in range(n)])
    diag = np.array([c[k, k] for k in range(n)])
    nbt_k = [np.mean([c[k, k] - c[t, k] for t in range(k + 1, n)]) for k in range(n - 1)]
    nnbt_k = [np.mean([(c[k, k] - c[t, k]) / c[k, k] for t in range(k + 1, n)])
              for k in range(n - 1) if c[k, k] > 0]
    auc_k = [(fwd[k] + sum(c[t, k] for t in range(k + 1, n))) / (1 + n - 1 - k) for k in range(n)]
    return {
        "n_tasks_scored": n,
        "fwt": float(fwd.mean()),
        "nbt": float(np.mean(nbt_k)) if nbt_k else float("nan"),
        "nbt_normalized": float(np.mean(nnbt_k)) if nnbt_k else float("nan"),
        "auc": float(np.mean(auc_k)),
        "final_success": float(c[n - 1, :n].mean()),
        "mean_just_learned_success": float(diag.mean()),
        "best_minus_last_epoch": float(np.mean([np.max(curves[k]) - curves[k][-1] for k in range(n)])),
    }


def parametric_bootstrap(c, curves, n_episodes, n_boot=2000, seed=0, n_done=None):
    """95% intervals from resampling each success cell as Binomial(n_episodes, p).

    DMPEL stores per-cell success rates, not per-episode outcomes, so episode-level
    resampling is parametric. Seed-level resampling happens when aggregating runs.
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(c, dtype=float)
    n = int(n_done if n_done is not None else len(curves))  # never infer from non-zero rows (all-zero rows are valid)
    out = {k: [] for k in ("fwt", "nbt", "auc", "final_success")}
    for _ in range(n_boot):
        cb = rng.binomial(n_episodes, np.clip(np.nan_to_num(c), 0, 1)) / n_episodes
        cur = [rng.binomial(n_episodes, np.clip(np.asarray(curves[k], float), 0, 1)) / n_episodes for k in range(n)]
        m = compute(cb, cur, n)
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}


def aggregate(metric_dicts, keys=("fwt", "nbt", "auc", "final_success")):
    """Mean, SD and seed-bootstrap 95% interval across runs (one dict per seed)."""
    rng = np.random.default_rng(0)
    res = {}
    for k in keys:
        v = np.array([m[k] for m in metric_dicts], dtype=float)
        boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(2000)] if len(v) > 1 else [v.mean()]
        res[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n": int(len(v))}
    return res
