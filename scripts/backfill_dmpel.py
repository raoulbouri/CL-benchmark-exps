"""Phase 0: backfill the finished DMPEL LIBERO-Goal replication runs (seeds 100/200/300) into
results/ and W&B. These ran before the harness existed, with DMPEL's original protocol.

python scripts/backfill_dmpel.py [--no-wandb]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cl_bench import CL_ROOT, dmpel_export, results, wandb_util  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    for seed in (100, 200, 300):
        spec = {"track": "D", "suite": "libero_goal", "method": "dmpel", "budget": "default",
                "protocol": "original", "seed": seed, "phase": "p0", "codebase": "dmpel",
                "eval_episodes": {"selection": 20, "matrix": 20, "state_ids": "0-19 for both"},
                "source": "pre-harness replication (scripts/run_dmpel.sh in continual-learning)"}
        run_dir = str(CL_ROOT / f"outputs/dmpel/libero_goal/seed_{seed}/seed_{seed}")
        log = str(CL_ROOT / f"logs/dmpel/libero_goal_seed{seed}.log")
        rec = dmpel_export.build_record(spec, run_dir, log, "complete")
        path = results.write(rec)
        m = rec["metrics"]
        print(f"{rec['name']}: FWT {m['fwt']:.3f} NBT {m['nbt']:.3f} AUC {m['auc']:.3f} "
              f"final {m['final_success']:.3f} best-last {100*m['best_minus_last_epoch']:.1f} pts -> {path}")
        if args.no_wandb:
            continue
        wb, mode = wandb_util.init(rec["name"], config=spec, group=rec["group"],
                                   tags=["p0", "backfill", "original"])
        state = dmpel_export.log_new_tasks(wb, rec, {})
        rec["wandb"] = {"id": wb.id, "mode": mode, "url": getattr(wb, "url", None), "tasks_logged": state["tasks_logged"]}
        results.write(rec)
        dmpel_export.finalize_wandb(wb, rec, path)
        wb.finish()


if __name__ == "__main__":
    main()
