"""Export a DMPEL-codebase run (DMPEL, ER, Seq, EWC...) to the results schema and W&B.

Called repeatedly while a run is in progress (after each task) and once at the end. Only rows not
yet sent are logged to W&B, so re-running the exporter is safe.
"""
import os
import re

import numpy as np

from . import accounting, metrics, results

TASK_ROW_RE = re.compile(r"\[Task\s+(\d+) succ\.\]")


def _tasks_done(run_dir, log_path):
    """Number of tasks whose end-of-task success row exists. Only DMPEL's "[Task k succ.]" lines
    (printed after the matrix row is saved) count; learning-curve files exist earlier and must not."""
    if log_path and os.path.exists(log_path):
        rows = TASK_ROW_RE.findall(open(log_path, errors="ignore").read())
        return max(int(r) for r in rows) + 1 if rows else 0
    return 0


def build_record(spec, run_dir, log_path, status, extra=None):
    import torch
    res_path = os.path.join(run_dir, "result.pt")
    n_done = _tasks_done(run_dir, log_path)
    rec = results.load(results.run_name(spec)) or {}
    rec.update({"name": results.run_name(spec), "group": results.group_name(spec), "spec": spec,
                "status": status, "run_dir": run_dir, "log_path": log_path, "tasks_done": n_done})
    if not os.path.exists(res_path) or n_done == 0:
        for k in ("c_matrix", "curves", "metrics", "accounting", "timings", "bootstrap_episode_ci95", "dmpel_logged_s_fwd", "n_tasks"):
            rec.pop(k, None)  # never carry results over from a previous attempt
        if extra:
            rec.update(extra)
        return rec
    r = torch.load(res_path, map_location="cpu")
    S = np.asarray(r["S_conf_mat"], dtype=float)
    curves = []
    for k in range(n_done):
        f = os.path.join(run_dir, f"task{k}_auc.log")
        curves.append(np.asarray(torch.load(f)["success"], float).tolist() if os.path.exists(f) else [])
    c = [[(float(S[t, k]) if (t < n_done and k <= t) else None) for k in range(S.shape[1])] for t in range(S.shape[0])]
    rec["n_tasks"] = int(S.shape[0])
    rec["c_matrix"] = c
    rec["curves"] = curves
    rec["dmpel_logged_s_fwd"] = np.asarray(r["S_fwd"], float)[:n_done].tolist()  # kept for the FWT-bug record
    rec["metrics"] = metrics.compute(np.nan_to_num(np.array(c, dtype=float)), curves, n_done)
    rec["accounting"] = {"per_task": [accounting.dmpel_memory(run_dir, k) for k in range(n_done)]}
    rec["timings"] = accounting.dmpel_timings(log_path)
    if status == "complete":
        ep = spec.get("eval_episodes", {}).get("matrix", 20)
        rec["bootstrap_episode_ci95"] = metrics.parametric_bootstrap(np.nan_to_num(np.array(c, float)), curves, ep, n_done=n_done)
    if extra:
        rec.update(extra)
    return rec


def log_new_tasks(wb, rec, state):
    """Log rows for tasks not yet sent. `state` is a dict persisted in the record."""
    if not rec.get("c_matrix") or not rec.get("curves"):
        return state
    sent = state.get("tasks_logged", 0)
    for k in range(sent, rec.get("tasks_done", 0)):
        row = rec["c_matrix"][k][: k + 1]
        payload = {"task": k, "just_learned_success": row[k], "mean_seen_success": float(np.mean(row)),
                   "best_minus_last_epoch": float(np.max(rec["curves"][k]) - rec["curves"][k][-1]) if rec["curves"][k] else None}
        acc = rec["accounting"]["per_task"][k]
        payload.update({f"memory/{a}": b for a, b in acc.items() if b is not None})
        t = rec.get("timings", {})
        if k < len(t.get("per_task_train_min", [])):
            payload["time/train_min"] = t["per_task_train_min"][k]
            payload["time/eval_success_min"] = t["per_task_eval_success_min"][k]
        sub = metrics.compute(np.nan_to_num(np.array(rec["c_matrix"], float)), rec["curves"], k + 1)
        payload.update({f"so_far/{a}": b for a, b in sub.items() if isinstance(b, float)})
        wb.log(payload, step=k)
        state["tasks_logged"] = k + 1
    return state


def finalize_wandb(wb, rec, result_path):
    import wandb
    m = rec.get("metrics", {})
    wb.summary.update({f"final/{k}": v for k, v in m.items()})
    if rec.get("bootstrap_episode_ci95"):
        wb.summary.update({f"final/{k}_ci95": v for k, v in rec["bootstrap_episode_ci95"].items()})
    n = rec.get("tasks_done", 0)
    if n:
        cols = ["after_task"] + [f"task_{k}" for k in range(rec["n_tasks"])]
        wb.log({"success_matrix": wandb.Table(columns=cols, data=[[t] + rec["c_matrix"][t] for t in range(n)])})
    art = wandb.Artifact(name=f"result-{rec['name']}", type="result")
    art.add_file(str(result_path))
    wb.log_artifact(art)
