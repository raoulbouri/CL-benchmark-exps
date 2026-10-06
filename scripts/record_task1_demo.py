"""Portfolio GIF: LIBERO-Long task 1, decayed (Experiment 2) vs fixed (Experiment 4) final models, side by side. Evaluation only.

Run from the DMPEL clone on one GPU (see /mnt/data/users/bbouri/tmp/launch_demo.sh):
  python scripts/record_task1_demo.py [--gif-only]
Records 5 rollouts per model on task 1 (initial states 0-4, one environment, no worker processes), saving every 128x128 frame;
picks one episode per model; writes analysis/figures/task1_demo.gif and the chosen episodes' frames to analysis/figures/raw_frames/.
--gif-only rebuilds the GIF from the frames already on disk (no GPU).
"""
import argparse, glob, json, math, os, shutil, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = "/mnt/data/users/bbouri/CL-benchmark"
DEC = f"{R}/outputs/D-long10-dmpel-full-rho005-bias-ne10-original-s100/a0/seed_100"
FIXRUN = f"{R}/outputs/D-long10-dmpel-full-rho005-bias-ne10-FIXEDWD-original-s100"
TMP = f"{R}/outputs/task1_demo_frames"
FIG = f"{R}/analysis/figures"
RAW = f"{FIG}/raw_frames"
REPORT = "/mnt/data/users/bbouri/logs/task1_demo_report.txt"
POOL = ("A_pool", "A_q_pool", "A_v_pool", "B_pool", "B_q_pool", "B_v_pool", "bias_pool")
TASK, N_EP, RED, GREEN = 1, 5, (214, 39, 40), (30, 160, 80)
LABELS = {"decayed": "Decayed experts  SR 0.1", "fixed": "Optimizer fix    SR 0.9"}


def fixed_dir():
    ok = [d for d in sorted(glob.glob(f"{FIXRUN}/a*/seed_100")) if os.path.exists(f"{d}/task9_model.pth") and len(glob.glob(f"{d}/task*_model.pth")) == 10]
    assert ok, "no complete Experiment 4 attempt folder"
    return ok[-1]


def frame_of(obs):
    return np.ascontiguousarray(obs[0]["agentview_image"][::-1, ::-1])  # LIBERO images arrive rotated 180 degrees


def record():
    import torch
    from easydict import EasyDict
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    from libero.lifelong.utils import control_seed, safe_device, get_task_embs

    cfg = EasyDict(json.load(open(f"{DEC}/config.json")))
    cfg.use_ddp = False
    cfg.experiment_dir = f"{R}/outputs/restore_test_scratch"
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    control_seed(cfg.seed)
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_tasks = benchmark.n_tasks
    get_dataset(dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(0)), obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True, seq_len=cfg.data.seq_len, demos=range(cfg.data.n_demos_per_task))
    benchmark.set_task_embs(get_task_embs(cfg, [benchmark.get_task(i).language for i in range(n_tasks)]))
    task = benchmark.get_task(TASK)
    assert task.language == "put both the cream cheese box and the butter in the basket", task.language
    algo = get_algo_class(cfg.lifelong.algo)(n_tasks, cfg)
    algo.policy.load_state_dict(torch.load(cfg.pretrain_model_path, map_location="cpu", weights_only=False)["state_dict"], strict=False)
    algo.policy.init_moe_policy()
    algo = safe_device(algo, cfg.device)
    for _ in range(n_tasks):
        algo.policy.add_new_and_freeze_previous(cfg.policy.ll_expert_per_task)
    fdir = fixed_dir()
    print("decayed model:", f"{DEC}/task9_model.pth", "\nfixed model:  ", f"{fdir}/task9_model.pth", flush=True)
    states = {"decayed": f"{DEC}/task9_model.pth", "fixed": f"{fdir}/task9_model.pth"}
    env = DummyVectorEnv([lambda: OffScreenRenderEnv(bddl_file_name=os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
                                                     camera_heights=cfg.data.img_h, camera_widths=cfg.data.img_w)])
    init_states = torch.load(os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file))
    task_emb = benchmark.get_task_emb(TASK)
    shutil.rmtree(TMP, ignore_errors=True)
    summary = {"fixed_dir": fdir}
    for ci, (cond, path) in enumerate(states.items()):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        msg = algo.policy.load_state_dict({k: v.to(cfg.device) for k, v in sd.items()}, strict=False)
        assert not msg.unexpected_keys and not [k for k in msg.missing_keys if k.split(".")[-1] in POOL or "moe_router" in k]
        summary[cond] = []
        for r in range(N_EP):
            out = f"{TMP}/{cond}/ep{r}"
            os.makedirs(out, exist_ok=True)
            seed = 100 * ci + r
            torch.manual_seed(seed); np.random.seed(seed)
            algo.eval(); algo.policy.frozen_language_emb = None
            env.reset(); algo.reset()
            obs = env.set_init_state(init_states[r:r + 1])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros((1, 7)))
            t, success, t0 = 0, False, time.time()
            eef = [np.asarray(obs[0].get("robot0_eef_pos", np.zeros(3)), float)]
            Image.fromarray(frame_of(obs)).save(f"{out}/frame_{t:04d}.png")
            if cond == "decayed" and r == 0:
                f = frame_of(obs).astype(float)
                print(f"first-frame check: mean {f.mean():.1f}, std {f.std():.1f}", flush=True)
                if f.std() < 8 or f.mean() < 10:
                    print("BLANK FRAME: rendering is not working", flush=True)
                    sys.exit(2)
            while t < cfg.eval.max_steps:
                t += 1
                obs, _, done, _ = env.step(algo.policy.get_action(raw_obs_to_tensor_obs(obs, task_emb, cfg)))
                eef.append(np.asarray(obs[0].get("robot0_eef_pos", np.zeros(3)), float))
                Image.fromarray(frame_of(obs)).save(f"{out}/frame_{t:04d}.png")
                if done[0]:
                    success = True
                    break
            e = np.array(eef)[-150:]
            stall = float((np.linalg.norm(e[10:] - e[:-10], axis=1) < 0.01).mean()) if len(e) > 10 else 0.0  # fraction of the last 150 steps with the hand nearly still
            summary[cond].append({"episode": r, "init_state": r, "success": success, "steps": t, "seed": seed, "stall_last150": round(stall, 3)})
            print(f"[{cond}] ep{r} (initial state {r}): success={success} steps={t} stall={stall:.2f} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(summary, open(f"{TMP}/summary.json", "w"), indent=1)
    return summary


def font(size):
    import matplotlib.font_manager as fm
    return ImageFont.truetype(fm.findfont("DejaVu Sans:bold"), size)


def panel(frame, color, label, speed):
    im = Image.fromarray(frame).resize((320, 320), Image.NEAREST)  # nearest neighbour, no blur
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 319, 319], outline=color, width=3)
    f = font(14)
    d.text(((320 - d.textlength(label, font=f)) / 2, 8), label, font=f, fill="white", stroke_width=2, stroke_fill="black")
    f2, tag = font(10), f"{speed:.1f}x speed"
    d.text((312 - d.textlength(tag, font=f2), 300), tag, font=f2, fill="white", stroke_width=2, stroke_fill="black")
    return im


def load(cond, r):
    return [np.asarray(Image.open(f).convert("RGB")) for f in sorted(glob.glob(f"{TMP}/{cond}/ep{r}/frame_*.png"))]


def choose(summary):
    dec, fix = summary["decayed"], summary["fixed"]
    failed = [e for e in dec if not e["success"]]
    if failed:
        best = max(failed, key=lambda e: (e["stall_last150"], -e["episode"]))
        why_d = (f"failed; chosen as the most clearly failing: its hand was nearly still for {best['stall_last150']:.0%} of the last 150 steps"
                 if best["stall_last150"] > 0 else "failed; no failed episode stalled, so the first failed one was used")
        if all(not e["success"] for e in dec) and best["stall_last150"] == 0:
            best, why_d = dec[0], "all 5 failed and none stalled, so episode 0 was used"
    else:
        best = max(dec, key=lambda e: e["steps"])
        why_d = "NO decayed episode failed in 5; the longest episode was used, so the left panel does NOT show a failure"
    won = [e for e in fix if e["success"]]
    if won:
        bf, why_f = won[0], "first episode that succeeded"
    else:
        bf, why_f = fix[0], "none of 5 succeeded; episode 0 used (progress was not measured), so the right panel does NOT show a success"
    return best, why_d, bf, why_f


def build(summary):
    best, why_d, bf, why_f = choose(summary)
    fd, ff = load("decayed", best["episode"]), load("fixed", bf["episode"])
    out = f"{FIG}/task1_demo.gif"
    for fps, total in ((12, 96), (8, 64)):
        stride = math.ceil(max(len(fd), len(ff)) / total)  # same stride for both panels so the whole attempt fits
        speed = stride * fps / 20.0  # the control loop runs at 20 Hz
        def pick(fr):
            s = fr[::stride]
            return (s + [s[-1]] * (total - len(s)))[:total]  # freeze on the last frame
        sd, sf = pick(fd), pick(ff)
        imgs = []
        for a, b in zip(sd, sf):
            c = Image.new("RGB", (640, 320))
            c.paste(panel(a, RED, LABELS["decayed"], speed), (0, 0))
            c.paste(panel(b, GREEN, LABELS["fixed"], speed), (320, 0))
            imgs.append(c)
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(round(1000 / fps)), loop=0, optimize=True)
        size = os.path.getsize(out) / 1048576
        print(f"GIF at {fps} fps, {total} frames, stride {stride} ({speed:.1f}x speed): {size:.2f} MB", flush=True)
        if size <= 8.0:
            break
    os.makedirs(RAW, exist_ok=True)
    for cond, ep in (("decayed", best), ("fixed", bf)):
        dst = f"{RAW}/{cond}_ep{ep['episode']}_state{ep['init_state']}"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(f"{TMP}/{cond}/ep{ep['episode']}", dst)
    lines = ["Task 1 demo GIF report",
             f"Decayed panel: episode {best['episode']} (initial state {best['init_state']}), success={best['success']}, {best['steps']} steps. Why: {why_d}.",
             f"Fixed panel:   episode {bf['episode']} (initial state {bf['init_state']}), success={bf['success']}, {bf['steps']} steps. Why: {why_f}.",
             f"Fixed model succeeded in the chosen episode: {'YES' if bf['success'] else 'NO'}",
             f"GIF: {out}  ({size:.2f} MB, {fps} fps, {total} frames, {speed:.1f}x speed, 640x320)",
             "All episodes: " + json.dumps({c: [(e['episode'], e['success'], e['steps']) for e in summary[c]] for c in ('decayed', 'fixed')}),
             f"Raw frames of the two chosen episodes: {RAW}/"]
    open(REPORT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif-only", action="store_true")
    a = ap.parse_args()
    s = json.load(open(f"{TMP}/summary.json")) if a.gif_only else record()
    build(s)
