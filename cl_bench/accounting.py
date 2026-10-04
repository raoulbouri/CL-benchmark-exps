"""Memory and compute accounting.

M_total = M_replay + M_growth + M_aux, measured as bytes written to disk in native dtype
(PI_REVIEW.md section 5). Phase 0 covers DMPEL; other methods add their own patterns.
"""
import glob
import os
import re

# Files a method keeps across tasks, by category. Model checkpoints of the current task are not memory.
DMPEL_PATTERNS = {
    "replay": ["task*_moe_attn_recall_query.pth", "task*_moe_attn_recall_attn.pth"],
    "aux": ["task*_expert_stats.pth"],
}


def bytes_of(run_dir, patterns, upto_task=None):
    total = 0
    for p in patterns:
        for f in glob.glob(os.path.join(run_dir, p)):
            m = re.search(r"task(\d+)_", os.path.basename(f))
            if upto_task is not None and m and int(m.group(1)) > upto_task:
                continue
            total += os.path.getsize(f)
    return total


def param_count(state_dict_path):
    import torch  # local import: only needed where torch exists
    sd = torch.load(state_dict_path, map_location="cpu")
    sd = sd[0] if isinstance(sd, (list, tuple)) else sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    return int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))


def dmpel_memory(run_dir, task):
    """Memory kept after finishing `task`. Replay stores of tasks < task are what future tasks use;
    we report the store including the current task (what is on disk) and the past-only part."""
    ckpt = os.path.join(run_dir, f"task{task}_model.pth")
    growth_params = param_count(ckpt) if os.path.exists(ckpt) else None
    growth_bytes = os.path.getsize(ckpt) if os.path.exists(ckpt) else None
    return {
        "replay_bytes_on_disk": bytes_of(run_dir, DMPEL_PATTERNS["replay"], task),
        "replay_bytes_past_tasks": bytes_of(run_dir, DMPEL_PATTERNS["replay"], task - 1) if task > 0 else 0,
        "aux_bytes": bytes_of(run_dir, DMPEL_PATTERNS["aux"], task),
        "growth_params": growth_params,
        "growth_bytes": growth_bytes,  # learnable-only checkpoint = expert library + router
    }


TIME_RE = re.compile(r"train time \(min\) ([\d.]+) eval loss time ([\d.]+) eval success time ([\d.]+)")
SEL_RE = re.compile(r"Epoch:\s+\d+ \| succ: .* \| time: ([\d.]+)")
EPOCH_RE = re.compile(r"# Batch: \d+ \| Epoch:\s+\d+ \| train loss: .* \| time: ([\d.]+) \| Memory utilization: ([\d.]+) GB")


def dmpel_timings(log_path):
    """Per-task minutes from DMPEL's stdout log: train (incl. selection evals), eval loss, eval success."""
    if not log_path or not os.path.exists(log_path):
        return {}
    txt = open(log_path, errors="ignore").read()
    rows = [tuple(map(float, m)) for m in TIME_RE.findall(txt)]
    sel = [float(x) for x in SEL_RE.findall(txt)]
    mem = [float(m[1]) for m in EPOCH_RE.findall(txt)]
    return {
        "per_task_train_min": [r[0] for r in rows],
        "per_task_eval_loss_min": [r[1] for r in rows],
        "per_task_eval_success_min": [r[2] for r in rows],
        "selection_eval_min_total": float(sum(sel)),
        "gpu_hours_logged": float(sum(sum(r) for r in rows) / 60.0),
        "peak_train_mem_gb": max(mem) if mem else None,
    }
