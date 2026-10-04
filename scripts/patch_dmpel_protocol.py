"""Idempotent patches to the DMPEL codebase needed by CL-benchmark (Phase 1).

1. max_tasks: cap the number of tasks (micro-streams for smoke tests). Default null = all tasks.
2. eval.protocol: 'original' (unchanged DMPEL behaviour) or 'heldout' (selection on initial states
   eval.select_state_ids, success matrix on eval.test_state_ids; see docs/PROTOCOL.md).

python scripts/patch_dmpel_protocol.py   (then the diff is saved to configs/patches/)
"""
import os
import pathlib

CL_ROOT = os.environ.get("CL_ROOT", "/usr1/home/bbouri/continual-learning")
D = pathlib.Path(CL_ROOT + "/third_party/DMPEL/libero")


def patch(path, old, new, marker):
    s = path.read_text()
    if marker in s:
        print(f"already patched: {path.name} ({marker})")
        return
    assert s.count(old) == 1, f"anchor not unique in {path}: {old[:60]!r}"
    path.write_text(s.replace(old, new))
    print(f"patched: {path.name} ({marker})")


# 1. config defaults
patch(D / "configs/config.yaml", "use_ddp: false", "use_ddp: false\nmax_tasks: null # CLB: cap number of tasks (smoke tests)", "max_tasks:")
patch(D / "configs/eval/default.yaml", "save_sim_states: false",
      "save_sim_states: false\n# CLB protocol switch (docs/PROTOCOL.md in CL-benchmark)\nprotocol: original\n"
      "select_state_ids: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]\n"
      "test_state_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]",
      "protocol:")

# 2. task cap
patch(D / "lifelong/main.py", "    n_manip_tasks = benchmark.n_tasks\n",
      "    n_manip_tasks = benchmark.n_tasks\n    if cfg.get(\"max_tasks\"):  # CLB: micro-streams\n"
      "        n_manip_tasks = min(n_manip_tasks, int(cfg.max_tasks))\n", "CLB: micro-streams")

# 3. evaluation protocol in evaluate_one_task_success
m = D / "lifelong/metric.py"
patch(m, "        algo.eval()\n        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1\n        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num\n\n        # initiate evaluation envs",
      "        algo.eval()\n"
      "        # CLB protocol: selection evals (task_str == '') and matrix evals use disjoint initial states\n"
      "        state_ids = None\n"
      "        if cfg.eval.get(\"protocol\", \"original\") == \"heldout\":\n"
      "            state_ids = list(cfg.eval.select_state_ids if task_str == \"\" else cfg.eval.test_state_ids)\n"
      "        n_eval = len(state_ids) if state_ids is not None else cfg.eval.n_eval\n"
      "        env_num = min(cfg.eval.num_procs, n_eval) if cfg.eval.use_mp else 1\n"
      "        eval_loop_num = (n_eval + env_num - 1) // env_num\n\n        # initiate evaluation envs",
      "CLB protocol: selection")
patch(m, "        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1\n        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num\n\n        # Try to handle",
      "        env_num = min(cfg.eval.num_procs, n_eval) if cfg.eval.use_mp else 1\n        eval_loop_num = (n_eval + env_num - 1) // env_num\n\n        # Try to handle",
      "eval_loop_num = (n_eval + env_num - 1) // env_num\n\n        # Try")
patch(m, "            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]\n",
      "            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]\n"
      "            if state_ids is not None:  # CLB heldout\n"
      "                indices = np.asarray(state_ids)[np.arange(i * env_num, (i + 1) * env_num) % len(state_ids)]\n",
      "CLB heldout")
s = m.read_text()
if "if i * env_num + k < n_eval and sim_states" not in s:
    s = s.replace("if i * env_num + k < cfg.eval.n_eval and sim_states", "if i * env_num + k < n_eval and sim_states")
    s = s.replace("                if i * env_num + k < cfg.eval.n_eval:\n                    num_success", "                if i * env_num + k < n_eval:\n                    num_success")
    s = s.replace("        success_rate = num_success / cfg.eval.n_eval\n", "        success_rate = num_success / n_eval\n")
    m.write_text(s)
    print("patched: metric.py (n_eval uses)")
body = m.read_text().split("def evaluate_one_task_success")[1].split("def evaluate_success")[0]
assert body.count("cfg.eval.n_eval") == 1 and "else cfg.eval.n_eval" in body, "unexpected cfg.eval.n_eval use in evaluate_one_task_success"
print("ok")
