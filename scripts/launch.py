"""Launch one run from a YAML spec, export results after every task, and finalise W&B.

python scripts/launch.py configs/runs/p1/<spec>.yaml --gpu 0 [--poll 120]

Runs in the `libero` env (harness); training runs in its codebase's env (`dmpel` or `clare`).
Exit codes: 0 = complete or already complete, 1 = failed or interrupted, 2 = pre-flight refused.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cl_bench import CL_ROOT, OUTPUTS, clare_export, dmpel_export, results, wandb_util  # noqa: E402

MIN_FREE_GB = 150
EVAL_EPISODES = {"original": {"selection": 20, "matrix": 20}, "heldout": {"selection": 10, "matrix": 40}}
EVAL_EPISODES_CLARE = {"original": {"selection": 0, "matrix": 100}, "heldout": {"selection": 0, "matrix": 40}}
HELDOUT_TEST_IDS = list(range(40))
DMPEL_DIR = CL_ROOT / "third_party/DMPEL"
DMPEL_CKPT = CL_ROOT / "checkpoints/dmpel_pretrain/multitask_model_ep10.pth"
CLIP_LOCAL = CL_ROOT / "checkpoints/hf/clip-vit-base-patch16"
CLARE_DIR = CL_ROOT / "third_party/clare"
CLARE_CKPT = CL_ROOT / "checkpoints/clare/dit_flow_mt_libero_90_pretrain"
CLARE_TASK_PREFIX = {"libero_10": "Libero_10_Task_"}

STOP = {"flag": False, "proc": None}


class GpuSampler:
    """Every `period` s, append per-process GPU memory to <out_dir>/gpu_mem.csv, tagged with how many
    evaluations the run has logged so far (DMPEL: 'evaluate task' lines). Used for peak VRAM and leak diagnosis."""

    def __init__(self, out_dir, period=20):
        import threading
        self.path, self.period, self.peak = out_dir / "gpu_mem.csv", period, {}
        self.out_dir = out_dir
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _evals(self):
        n = 0
        for f in self.out_dir.glob("*.log"):
            try:
                n += open(f, errors="ignore").read().count("evaluate task")
            except OSError:
                pass
        return n

    def _run(self):
        uuid2idx = {}
        try:
            for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines():
                i, u = [x.strip() for x in line.split(",")]
                uuid2idx[u] = int(i)
        except Exception:
            pass
        with open(self.path, "a") as f:
            f.write("time,evals_so_far,gpu,pid,used_mib,gpu_total_used_mib\n")
            while not self._stop.is_set():
                try:
                    tot = {}
                    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True).splitlines():
                        i, m = [x.strip() for x in line.split(",")]
                        tot[int(i)] = int(m)
                        self.peak[int(i)] = max(self.peak.get(int(i), 0), int(m))
                    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True).splitlines()
                    ev, now = self._evals(), int(time.time())
                    for line in apps:
                        u, pid, mem = [x.strip() for x in line.split(",")]
                        g = uuid2idx.get(u, -1)
                        f.write(f"{now},{ev},{g},{pid},{mem},{tot.get(g, '')}\n")
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.period)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)
        return {f"gpu{k}_peak_mib": v for k, v in self.peak.items()}


def _on_signal(signum, _frame):
    STOP["flag"] = True
    p = STOP["proc"]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)  # whole job incl. simulator subprocesses
        except ProcessLookupError:
            pass


def preflight(gpus):
    disk = os.environ.get("CLB_DISK_PATH", "/usr1")
    free_gb = shutil.disk_usage(disk).free / 1e9
    if free_gb < MIN_FREE_GB:
        return f"only {free_gb:.0f} GB free on {disk} (< {MIN_FREE_GB})"
    root_min = float(os.environ.get("CLB_ROOT_MIN_GB", "0"))  # machines whose system disk is nearly full (xulab)
    if root_min and shutil.disk_usage("/").free / 1e9 < root_min:
        return f"system disk / has less than {root_min:.0f} GB free"
    for gpu in gpus:
        used = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"], text=True).strip()
        if int(used) > 2000:
            return f"GPU {gpu} busy ({used} MiB in use)"
    return None


def gpus_for(spec, gpu):
    n = int(spec.get("dmpel", {}).get("ddp_gpus", 1)) if spec["codebase"] == "dmpel" else 1
    return list(range(n)) if n > 1 else [gpu]


def run_job(cmd, log_path, poll, on_poll=None):
    """Run a bash command in its own process group; call on_poll() every `poll` seconds."""
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        STOP["proc"] = proc
        while proc.poll() is None:
            for _ in range(poll):
                if proc.poll() is not None:
                    break
                time.sleep(1)
            if on_poll and proc.poll() is None:
                try:
                    on_poll()
                except Exception as e:  # never let a logging/export bug kill or orphan a run
                    print(f"[clb] export during run failed (continuing): {type(e).__name__}: {e}", flush=True)
    STOP["proc"] = None
    return proc.returncode


# ---------------- DMPEL codebase ----------------
def dmpel_command(spec, out_dir, gpu):
    d = spec["dmpel"]
    args = [f"seed={spec['seed']}", f"benchmark_name={spec['suite']}", f"policy={d['policy']}",
            f"lifelong={d['lifelong']}", f"exp={out_dir}", f"pretrain_model_path={d.get('pretrain', DMPEL_CKPT)}",
            f"policy.language_encoder.network_kwargs.model_name={CLIP_LOCAL}", f"eval.protocol={spec['protocol']}"]
    if spec.get("n_tasks"):
        args.append(f"max_tasks={spec['n_tasks']}")
    args += d.get("overrides", [])
    n = int(d.get("ddp_gpus", 1))
    xenv = "".join(f"{k}={v} " for k, v in d.get("env", {}).items())  # per-spec environment, e.g. CLB_AUDIT=1
    if n > 1:  # authors' setup: torchrun DDP; train.batch_size is per GPU; evaluation runs on rank 0 (GPU 0)
        args.append("use_ddp=true")
        py = " ".join(f"'{a}'" for a in args)
        devs = ",".join(str(i) for i in range(n))
        return (f"source {CL_ROOT}/env_dmpel.sh && export CUDA_VISIBLE_DEVICES={devs} MUJOCO_EGL_DEVICE_ID=0 {xenv}"
                f"WANDB_MODE=disabled && cd {DMPEL_DIR} && exec torchrun --standalone --nproc_per_node={n} libero/lifelong/main.py {py}")
    py = " ".join(f"'{a}'" for a in args)
    return (f"source {CL_ROOT}/env_dmpel.sh && export CUDA_VISIBLE_DEVICES={gpu} MUJOCO_EGL_DEVICE_ID={gpu} {xenv}"
            f"WANDB_MODE=disabled && cd {DMPEL_DIR} && exec python libero/lifelong/main.py {py}")


def run_dmpel(spec, out_dir, gpu, poll, wb, mode, extra):
    log_path = out_dir / "train.log"
    run_dir = str(out_dir / f"seed_{spec['seed']}")
    os.makedirs(run_dir, exist_ok=True)  # DDP: both ranks call create_experiment_dir; pre-create to avoid a mkdir race
    cmd = dmpel_command(spec, out_dir, gpu)
    (out_dir / "command.sh").write_text(cmd + "\n")
    state = {"tasks_logged": 0}

    def export(status):
        nonlocal state
        rec = dmpel_export.build_record(spec, run_dir, str(log_path), status, extra)
        state = dmpel_export.log_new_tasks(wb, rec, state)
        rec["wandb"] = {"id": wb.id, "mode": mode, "url": getattr(wb, "url", None), **state}
        return rec, results.write(rec)

    results.write(dmpel_export.build_record(spec, run_dir, str(log_path), "running", extra))
    rc = run_job(cmd, log_path, poll, on_poll=lambda: export("running"))
    finished = "finished learning" in log_path.read_text(errors="ignore")
    status = "complete" if (rc == 0 and finished) else ("interrupted" if STOP["flag"] else "failed")
    extra.update({"returncode": rc})
    return status, export


# ---------------- CLARE codebase ----------------
def clare_dataset_root(repo_id):
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HOME", str(CL_ROOT / "cache/hf"))
    # CLARE's LeRobot fork reads dataset format v2.1 (CODEBASE_VERSION); the repos' main branch is v3.0
    return snapshot_download(repo_id, repo_type="dataset", revision="v2.1")


def clare_stage_command(spec, k, out_dir, gpu):
    c = spec["clare"]
    suite = spec["suite"]
    prefix = CLARE_TASK_PREFIX[suite]
    tasks = ",".join(f"{prefix}{i}" for i in range(k + 1))
    repo = f"continuallearning/{suite}_image_task_{k}"
    stage_dir = out_dir / f"task_{k}"
    S, D = c["steps"], c.get("disc_steps", 2000)
    ep = spec["eval_episodes"]["matrix"]
    eval_bs = ep if spec["protocol"] == "heldout" else c.get("eval_batch", 50)
    common = [f"--seed={spec['seed']}", f"--job_name={results.run_name(spec)}_task_{k}", f"--output_dir={stage_dir}",
              f"--dataset.repo_id={repo}", f"--dataset.root={clare_dataset_root(repo)}", "--policy.push_to_hub=false",
              f"--batch_size={c.get('batch_size', 32)}", f"--num_workers={c.get('num_workers', 8)}", f"--steps={S}",
              "--env.type=libero", f"--env.benchmark={suite}", f"--env.task={tasks}",
              f"--eval.batch_size={eval_bs}", f"--eval.n_episodes={ep}", "--eval.max_episodes_rendered=0",
              f"--save_freq={S}", "--log_freq=100", "--wandb.enable=false"]
    if spec["method"] == "clare":
        prev = f"--peft_weight_path={out_dir}/task_{k-1}/checkpoints/last/adapter" if k > 0 else ""
        script = "clare.py"
        args = common + [f"--policy.path={CLARE_CKPT}", "--eval_freq=200000",
                         "--peft_cfg_path=./peft_lsy/peft_config/clare_dit_flow_encoder_adapter", prev,
                         f"--expand_threshold={c.get('expand_threshold', 1.0)}",
                         "--detect_distribution_shift_steps=200", "--detect_distribution_shift_batch_size=32",
                         f"--detect_distribution_shift_num_workers={c.get('num_workers', 8)}", "--detect_distribution_shift_log_freq=10",
                         f"--train_discriminators_steps={D}", "--train_discriminators_batch_size=32",
                         f"--train_discriminators_num_workers={c.get('num_workers', 8)}", "--train_discriminators_log_freq=50",
                         f"--train_discriminators_eval_freq={D}", f"--train_discriminators_save_freq={D}"]
    elif spec["method"] == "er":
        policy = CLARE_CKPT if k == 0 else f"{out_dir}/task_{k-1}/checkpoints/last/pretrained_model"
        rk = max(k - 1, 0)  # stage 0 has no past task: replay its own data (documented in DECISIONS.md)
        rrepo = f"continuallearning/{suite}_image_task_{rk}"
        script = "er.py"
        args = common + [f"--policy.path={policy}", f"--eval_freq={S}",
                         f"--replay_dataset.repo_id={rrepo}", f"--replay_dataset.root={clare_dataset_root(rrepo)}",
                         f"--replay_batch_size={c.get('replay_batch_size', 8)}"]
    else:
        raise NotImplementedError(spec["method"])
    env = (f"source {CL_ROOT}/miniforge3/etc/profile.d/conda.sh && conda activate clare && "
           f"export CUDA_VISIBLE_DEVICES={gpu} MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID={gpu} PYOPENGL_PLATFORM=egl "
           f"HF_HOME={CL_ROOT}/cache/hf HF_LEROBOT_HOME={CL_ROOT}/cache/lerobot WANDB_MODE=disabled")
    if spec["protocol"] == "heldout":
        env += f" CLB_EVAL_STATE_IDS={','.join(map(str, HELDOUT_TEST_IDS))}"
    a = " ".join(f"'{x}'" for x in args if x)
    return f"{env} && cd {CLARE_DIR} && exec python ./lerobot_lsy/src/lerobot/scripts/{script} {a}"


def run_clare(spec, out_dir, gpu, poll, wb, mode, extra):
    n = spec.get("n_tasks") or 10
    minutes, state, rc = [], {"tasks_logged": 0}, 0

    def export(status, n_done):
        nonlocal state
        rec = clare_export.build_record(spec, out_dir, n_done, status, minutes, extra)
        if rec.get("metrics") or status != "running":
            state = dmpel_export.log_new_tasks(wb, rec, state) if rec.get("c_matrix") and rec.get("metrics") else state
        rec["wandb"] = {"id": wb.id, "mode": mode, "url": getattr(wb, "url", None), **state}
        return rec, results.write(rec)

    done = 0
    for k in range(n):
        cmd = clare_stage_command(spec, k, out_dir, gpu)
        (out_dir / f"command_task_{k}.sh").write_text(cmd + "\n")
        t0 = time.time()
        rc = run_job(cmd, out_dir / f"task_{k}.log", poll)
        minutes.append((time.time() - t0) / 60)
        row = clare_export.parse_stage_log(str(out_dir / f"task_{k}.log"))
        if rc != 0 or STOP["flag"] or not all(i in row for i in range(k + 1)):
            break
        done = k + 1
        export("running", done)
    status = "complete" if done == n else ("interrupted" if STOP["flag"] else "failed")
    extra.update({"returncode": rc})
    return status, (lambda st: export(st, done))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--poll", type=int, default=120)
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.spec))
    table = EVAL_EPISODES_CLARE if spec["codebase"] == "clare" else EVAL_EPISODES
    spec.setdefault("eval_episodes", table[spec["protocol"]])
    name = results.run_name(spec)
    prev = results.load(name)
    if prev and prev.get("status") == "complete":
        print(f"[clb] {name}: already complete, skipping", flush=True)
        return 0
    why = preflight(gpus_for(spec, a.gpu))
    if why:
        print(f"[clb] {name}: pre-flight refused: {why}", flush=True)
        return 2
    attempt = (prev or {}).get("attempt", -1) + 1
    out_dir = OUTPUTS / name / f"a{attempt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ("returncode", "wall_hours", "wandb", "bootstrap_episode_ci95"):
        if prev:
            prev.pop(k, None)
    extra = {"attempt": attempt, "spec_path": os.path.abspath(a.spec), "gpu_index": a.gpu,
             "provenance": results.provenance(), "started_at": time.time()}
    wb, mode = wandb_util.init(name, config=spec, group=results.group_name(spec),
                               tags=[spec.get("phase", "p?"), spec["protocol"], spec["method"], spec["codebase"]], attempt=attempt)
    print(f"[clb] {name}: attempt {attempt}, W&B {mode} {getattr(wb, 'url', '')}, out {out_dir}", flush=True)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    t0 = time.time()
    runner = run_clare if spec["codebase"] == "clare" else run_dmpel
    sampler = GpuSampler(out_dir, period=float(os.environ.get("CLB_GPU_SAMPLE_S", "20"))).start()
    try:
        status, export = runner(spec, out_dir, a.gpu, a.poll, wb, mode, extra)
    except BaseException:
        _on_signal(signal.SIGTERM, None)  # kill the training job's process group before dying
        raise
    finally:
        extra["gpu_peak_mib"] = sampler.stop()
    extra["wall_hours"] = (time.time() - t0) / 3600
    extra["gpus_used"] = gpus_for(spec, a.gpu)
    rec, path = export(status)
    if rec.get("metrics"):
        dmpel_export.finalize_wandb(wb, rec, path)
    wb.summary.update({"status": status, "wall_hours": extra["wall_hours"], "returncode": extra.get("returncode")})
    wb.finish(exit_code=0 if status == "complete" else 1)
    print(f"[clb] {name}: {status} (rc={extra.get('returncode')}, {extra['wall_hours']:.2f} h)", flush=True)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
