"""Restore test: does restoring every expert to its creation-time weights recover forgotten tasks? (evaluation only)

Run from the DMPEL clone, GPU 1:
  cd $CL_ROOT/third_party/DMPEL && CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 WANDB_MODE=disabled \
    python /mnt/data/users/bbouri/CL-benchmark/scripts/restore_experts_eval.py [--dry] [--tasks 0,1,...]

Conditions (same model, same 10-episode `original` protocol, same initial states 0..9):
  control  : the final model (task9_model.pth) unchanged, re-evaluated now (separates sampling noise from the effect)
  restored : final model with, for every expert k, its A/B/bias rows replaced by those in task{k}_model.pth
No training. Checkpoints are only read (sha256 checked before and after). The router is the final router in both conditions.
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import torch
from easydict import EasyDict

RUN = "/mnt/data/users/bbouri/CL-benchmark/outputs/D-long10-dmpel-full-rho005-bias-ne10-original-s100/a0"
CK = f"{RUN}/seed_100"
RESULT = "/mnt/data/users/bbouri/CL-benchmark/results/D-long10-dmpel-full-rho005-bias-ne10-original-s100.json"
OUT = "/mnt/data/users/bbouri/CL-benchmark/analysis/exp3"
POOL = ("A_pool", "A_q_pool", "A_v_pool", "B_pool", "B_q_pool", "B_v_pool", "bias_pool")

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true", help="build and load everything, print norms, skip rollouts")
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--procs", type=int, default=5)


def main():
    args = ap.parse_args()
    tasks = [int(t) for t in args.tasks.split(",")]

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    from libero.lifelong.metric import evaluate_one_task_success
    from libero.lifelong.utils import control_seed, safe_device, torch_load_model, get_task_embs


    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


    files = sorted(glob.glob(f"{CK}/task*_model.pth"))
    before = {f: sha(f) for f in files}

    cfg = EasyDict(json.load(open(f"{CK}/config.json")))
    cfg.eval.num_procs = args.procs  # the original run used 5 evaluation workers
    cfg.use_ddp = False
    cfg.experiment_dir = "/mnt/data/users/bbouri/CL-benchmark/outputs/restore_test_scratch"  # never the original run directory
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    assert cfg.eval.n_eval == 10 and cfg.eval.get("protocol", "original") == "original" and cfg.policy.tune_bias is True
    control_seed(cfg.seed)

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_tasks = benchmark.n_tasks
    # the first dataset call also initialises robomimic's observation-key table, exactly as main.py does
    get_dataset(dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(0)), obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True, seq_len=cfg.data.seq_len, demos=range(cfg.data.n_demos_per_task))
    descriptions = [benchmark.get_task(i).language for i in range(n_tasks)]
    benchmark.set_task_embs(get_task_embs(cfg, descriptions))

    algo = get_algo_class(cfg.lifelong.algo)(n_tasks, cfg)
    sd0 = torch_load_model(cfg.pretrain_model_path)[0]
    algo.policy.load_state_dict(sd0, strict=False)
    algo.policy.init_moe_policy()
    algo = safe_device(algo, cfg.device)
    for _ in range(n_tasks):  # grow the expert pool and router to 10 experts, as start-of-task does in training
        algo.policy.add_new_and_freeze_previous(cfg.policy.ll_expert_per_task)

    final = torch_load_model(f"{CK}/task9_model.pth", map_location="cpu")[0]
    ck = {k: torch_load_model(f"{CK}/task{k}_model.pth", map_location="cpu")[0] for k in range(10)}
    pool_names = [n for n in final if n.split(".")[-1] in POOL]
    restored = {n: v.clone() for n, v in final.items()}
    for k in range(10):
        for n in pool_names:
            restored[n][k] = ck[k][n][k]


    def norms(sd, e):
        return {kind: float(sum((v[e].float() ** 2).sum() for n, v in sd.items() if n.split(".")[-1] in kinds) ** 0.5)
                for kind, kinds in (("A", POOL[:3]), ("B", POOL[3:6]), ("bias", POOL[6:]))}


    print("pool tensors:", len(pool_names))
    for e in (0, 1, 2):
        print(f"expert {e} norms  final {norms(final, e)}  restored {norms(restored, e)}")
    changed = [e for e in range(10) if any(not torch.equal(final[n][e], restored[n][e]) for n in pool_names)]
    print("experts whose weights differ between final and restored:", changed, "(expert 9 must be unchanged)")
    assert 9 not in changed and 0 in changed

    hist = json.load(open(RESULT))["c_matrix"]
    orig_final = {j: hist[9][j] for j in range(10)}


    def load(sd):
        msg = algo.policy.load_state_dict({k: v.to(cfg.device) for k, v in sd.items()}, strict=False)
        assert not msg.unexpected_keys, msg.unexpected_keys[:5]
        bad = [k for k in msg.missing_keys if k.split(".")[-1] in POOL or "moe_router" in k]
        assert not bad, bad[:5]  # only frozen backbone weights (loaded from the pretrained model) may be absent
        return len(msg.missing_keys)


    res = {"control": {}, "restored": {}}
    if not args.dry:
        for cond, sd in (("control", final), ("restored", restored)):
            print(f"\n=== condition: {cond} (missing frozen keys: {load(sd)}) ===", flush=True)
            for j in tasks:
                t0 = time.time()
                res[cond][j] = evaluate_one_task_success(cfg=cfg, algo=algo, task=benchmark.get_task(j), task_emb=benchmark.get_task_emb(j),
                                                         task_id=j, sim_states=None, task_str="matrix")
                print(f"[{cond}] task {j}: {res[cond][j]:.1f}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        print("dry run: load check", load(final), load(restored), "- no rollouts")

    after = {f: sha(f) for f in files}
    assert before == after, "a checkpoint changed"
    print("checkpoints unchanged (sha256 of", len(files), "files identical before and after)")
    if args.dry:
        sys.exit(0)

    lines = ["Restore test: DMPEL LIBERO-Long, seed 100, final model (task 9) with every expert restored to its creation-time weights.",
             "10 rollouts per task, `original` protocol, initial states 0..9, evaluation workers: %d. Router = final router in both conditions." % args.procs,
             "'Original final SR' = last row of the run's success matrix. 'Control' = the unchanged final model re-evaluated in this script.", "",
             f"{'Task':>4} | {'Original final SR':>17} | {'Control re-eval':>15} | {'Restored SR':>11} | {'Change vs original':>18} | {'Change vs control':>17}"]
    for j in tasks:
        o, c, r = orig_final[j], res["control"][j], res["restored"][j]
        lines.append(f"{j:>4} | {o:>17.1f} | {c:>15.1f} | {r:>11.1f} | {r - o:>+18.1f} | {r - c:>+17.1f}")
    mean = lambda d: float(np.mean([d[j] for j in tasks]))
    lines.append(f"{'mean':>4} | {mean(orig_final):>17.3f} | {mean(res['control']):>15.3f} | {mean(res['restored']):>11.3f} | "
                 f"{mean(res['restored']) - mean(orig_final):>+18.3f} | {mean(res['restored']) - mean(res['control']):>+17.3f}")
    if 1 in tasks:
        r1 = res["restored"][1]
        lines += ["", f"Key question: task index 1 restored SR = {r1:.1f} (original final {orig_final[1]:.1f}, control {res['control'][1]:.1f}); "
                  f"recovers above 0.4: {'yes' if r1 > 0.4 else 'no'}. With 10 rollouts one task's SR moves in steps of 0.1 (Wilson 95% interval of 4/10 is 0.17-0.69)."]
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/restore_test_results.txt", "w").write("\n".join(lines) + "\n")
    json.dump({"original_final": orig_final, **res}, open(f"{OUT}/restore_test_results.json", "w"), indent=1)
    print("\n" + "\n".join(lines))
    try:  # W&B is the official result log
        sys.path.insert(0, "/mnt/data/users/bbouri/CL-benchmark")
        from cl_bench import wandb_util
        run, mode = wandb_util.init("E3-restore-test-long-s100", {"experiment": "restore frozen experts", "n_eval": 10, "procs": args.procs}, "E3", ["exp3", "restore"], job_type="eval")
        import wandb
        run.log({"table": wandb.Table(columns=["task", "original_final", "control", "restored"], data=[[j, orig_final[j], res["control"][j], res["restored"][j]] for j in tasks])})
        run.finish()
        print("W&B:", mode)
    except Exception as e:
        print("W&B logging skipped:", type(e).__name__, e)


if __name__ == "__main__":  # evaluation spawns worker processes that re-import this file
    main()
