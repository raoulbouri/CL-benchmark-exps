"""Staged rendering check on one GPU: 1 env, then N workers. Reports GPU memory per stage. Run with a small N (xulab: 5)."""
import multiprocessing, os, subprocess, sys, time

def smi():
    r = subprocess.run(["timeout", "15", "nvidia-smi", "-i", os.environ.get("CHK_GPU", "1"), "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return r.stdout.strip() or "nvidia-smi: no answer"

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
    b = benchmark.get_benchmark_dict()["libero_10"](); t = b.get_task(0)
    args = {"bddl_file_name": os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file), "camera_heights": 128, "camera_widths": 128}
    print("baseline GPU (MiB, C):", smi(), flush=True)
    env = OffScreenRenderEnv(**args); env.seed(0); env.reset(); obs = env.set_init_state(b.get_task_init_states(0)[0])
    for _ in range(5): obs, *_ = env.step([0.0] * 7)
    print("1 env OK, image mean", float(np.asarray(obs["agentview_image"]).mean()), "| GPU:", smi(), flush=True); env.close(); time.sleep(3)
    n = int(sys.argv[1])
    venv = SubprocVectorEnv([lambda: OffScreenRenderEnv(**args) for _ in range(n)]); venv.reset()
    o = venv.set_init_state(b.get_task_init_states(0)[:n])
    for _ in range(5): o, *_ = venv.step(np.zeros((n, 7)))
    time.sleep(3); print(f"{n} workers OK | GPU:", smi(), flush=True); venv.close(); time.sleep(3)
    print("after close | GPU:", smi(), flush=True); print("RENDER CHECK OK")
