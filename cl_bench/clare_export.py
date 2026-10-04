"""Export a CLARE-codebase run (CLARE, ER, Seq in learnsyslab/clare) to the results schema.

CLARE runs one process per task (stage). Stage k trains on task k and, at its end, evaluates every
task 0..k; that evaluation is row k of the success matrix. There is no selection curve, so the
"curve" for task k is the single just-learned value c[k][k] (FWT then equals mean c[k][k]).
"""
import json
import os
import re
import struct

import numpy as np

from . import metrics, results

# CLARE prints e.g. "success_Libero_10_Task_0:85.0" (pc_success, a percentage, shown as "success")
PC_RE = re.compile(r"(?<![A-Za-z])(?:pc_)?success_(?:Libero_[A-Za-z0-9]+_)?Task_(\d+)\s*[:=]\s*([\d.]+)")


def parse_stage_log(log_path):
    """Return {task_id: success in [0,1]} from the last evaluation printed in a stage log."""
    if not os.path.exists(log_path):
        return {}
    found = {}
    for m in PC_RE.finditer(open(log_path, errors="ignore").read()):
        found[int(m.group(1))] = float(m.group(2)) / 100.0  # pc_success is a percentage
    return found


def safetensors_params(path):
    """Parameter count from a .safetensors header without importing torch or safetensors."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    total = 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        total += int(np.prod(v["shape"])) if v["shape"] else 1
    return total


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) if os.path.isdir(path) else 0


def stage_memory(stage_dir, method):
    """Bytes kept after a stage. CLARE: adapter (incl. discriminators) grows; ER: replay is the
    raw past datasets (counted by the budget logic in Phase 4, reported here as 0 = full data)."""
    ckpt = os.path.join(stage_dir, "checkpoints", "last")
    adapter = os.path.join(ckpt, "adapter")
    out = {"adapter_bytes": dir_bytes(adapter)}
    st = [os.path.join(r, f) for r, _, fs in os.walk(adapter) for f in fs if f.endswith(".safetensors")] if os.path.isdir(adapter) else []
    out["adapter_params"] = sum(safetensors_params(p) for p in st) if st else None
    if method == "er":
        out["note"] = "ER replays full past-task datasets (budgeted replay is Phase 4)"
    return out


def build_record(spec, out_dir, n_done, status, stage_minutes=None, extra=None):
    name = results.run_name(spec)
    rec = results.load(name) or {}
    n = spec.get("n_tasks") or 10
    rec.update({"name": name, "group": results.group_name(spec), "spec": spec, "status": status,
                "run_dir": str(out_dir), "tasks_done": n_done, "n_tasks": n})
    c = [[None] * n for _ in range(n)]
    for t in range(n_done):
        row = parse_stage_log(os.path.join(out_dir, f"task_{t}.log"))
        for k in range(t + 1):
            c[t][k] = row.get(k)
    curves = [[c[k][k]] if c[k][k] is not None else [] for k in range(n_done)]
    rec["c_matrix"], rec["curves"] = c, curves
    if n_done and all(c[t][k] is not None for t in range(n_done) for k in range(t + 1)):
        rec["metrics"] = metrics.compute(np.nan_to_num(np.array(c, dtype=float)), curves, n_done)
    rec["accounting"] = {"per_task": [stage_memory(os.path.join(out_dir, f"task_{t}"), spec["method"]) for t in range(n_done)]}
    rec["timings"] = {"per_task_wall_min": stage_minutes or []}
    if status == "complete" and rec.get("metrics"):
        ep = spec.get("eval_episodes", {}).get("matrix", 100)
        rec["bootstrap_episode_ci95"] = metrics.parametric_bootstrap(np.nan_to_num(np.array(c, float)), curves, ep, n_done=n_done)
    if extra:
        rec.update(extra)
    return rec
