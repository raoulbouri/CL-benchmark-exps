"""Read-only, CPU-only audit of a DMPEL run directory (answers Experiment 1's four questions).

Usage: python scripts/audit_dmpel_ckpt.py <run_dir> [--md]

Q1 rho            from the cfg stored in each task checkpoint (and the saved buffer audit file, if present)
Q2 router state   task{k}_moe_router.pth present/shapes; router keys inside task{k}_model.pth
Q3 routing logs   routing/*.npz (eval-time per-step router output); task{k}_buffer_audit.pth (per-frame, all frames)
Q4 checkpoint     size, tensor count, bytes by key group, expert/LoRA count per task (does the pool grow?)
"""
import argparse
import collections
import glob
import os
import re
import sys

import torch


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def mb(n):
    return f"{n / 2**20:.1f}"


def group(key):
    for g in ("moe_router", "router", "lora", "expert", "pool", "adapter", "head", "norm"):
        if g in key.lower():
            return g
    return "other"


def get_rho(cfg):
    try:
        return cfg.lifelong.moe_attn_recall_sample_ratio
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md", action="store_true", help="print a markdown table for docs/DECISIONS.md")
    a = ap.parse_args()
    d = a.run_dir
    tasks = sorted(int(re.search(r"task(\d+)_model", f).group(1)) for f in glob.glob(f"{d}/task*_model.pth"))
    if not tasks:
        sys.exit(f"no task*_model.pth in {d}")
    rows, rhos = [], set()
    for k in tasks:
        f = f"{d}/task{k}_model.pth"
        ck = load(f)
        sd = ck["state_dict"]
        by = collections.Counter()
        for key, v in sd.items():
            by[group(key)] += v.numel() * v.element_size()
        rho = get_rho(ck.get("cfg"))
        rhos.add(rho)
        router_sep = f"{d}/task{k}_moe_router.pth"
        router_keys = sum("router" in key for key in sd)
        sep = os.path.exists(router_sep)
        sep_shapes = ""
        if sep:
            r = load(router_sep)["state_dict"]
            sep_shapes = ";".join(f"{n}:{tuple(v.shape)}" for n, v in list(r.items())[:2]) + "..."
        buf = f"{d}/task{k}_buffer_audit.pth"
        buf_info = ""
        if os.path.exists(buf):
            b = load(buf)
            buf_info = f"kept {len(b['kept_indices'])}/{b['total_sample_num']} (rho={b['rho']})"
        rec = f"{d}/task{k}_moe_attn_recall_query.pth"
        rows.append(dict(task=k, size=os.path.getsize(f), n_tensors=len(sd), bytes=sum(by.values()), by=by, rho=rho,
                         router_in_model=router_keys, router_file=sep, router_shapes=sep_shapes, buffer=buf_info,
                         recall_mb=os.path.getsize(rec) if os.path.exists(rec) else None))
    routing = sorted(glob.glob(f"{d}/routing/*.npz"))
    print(f"run dir: {d}\ntasks with checkpoints: {tasks}\n")
    print("Q1 rho:", sorted(map(str, rhos)), "(1.0 = every frame stored; 0.05 = the paper's 5%)")
    print("\nQ4 checkpoint contents (learnable parameters only):")
    print(f"{'task':>4} {'file MB':>8} {'tensors':>8} {'param MB':>9}  bytes by group (MB)")
    for r in rows:
        print(f"{r['task']:>4} {mb(r['size']):>8} {r['n_tensors']:>8} {mb(r['bytes']):>9}  " +
              ", ".join(f"{g}={mb(v)}" for g, v in r["by"].most_common(5)))
    # the expert pool is stacked into a few big tensors, so growth shows in bytes, not in the tensor count
    pool = [r["by"].get("pool", 0) for r in rows]
    print("expert-pool bytes grow with task index:", pool == sorted(pool) and len(set(pool)) > 1, [mb(x) for x in pool])
    print("\nQ2 router state:")
    for r in rows:
        print(f"  task {r['task']}: router keys in model file={r['router_in_model']}, separate file={r['router_file']} {r['router_shapes']}")
    print("\nQ3 logs:")
    print(f"  eval-time routing files: {len(routing)}" + (f" (e.g. {os.path.basename(routing[0])})" if routing else " (none: CLB_AUDIT=1 was not set or the audit patch was not applied)"))
    for r in rows:
        if r["buffer"] or r["recall_mb"] is not None:
            print(f"  task {r['task']}: buffer audit {r['buffer'] or 'absent'}; coefficient file {mb(r['recall_mb']) if r['recall_mb'] else 'absent'} MB")
    if a.md:
        print("\n| task | file MB | tensors | rho | router file | buffer |\n|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['task']} | {mb(r['size'])} | {r['n_tensors']} | {r['rho']} | {r['router_file']} | {r['buffer']} |")


if __name__ == "__main__":
    main()
