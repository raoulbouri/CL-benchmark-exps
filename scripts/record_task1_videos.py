"""Record rollouts of LIBERO-Long task index 1 for the decayed (control) and restored DMPEL models, then build comparison GIFs.

Run from the DMPEL clone, on one GPU:
  cd $CL_ROOT/third_party/DMPEL && CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
    python /mnt/data/users/bbouri/CL-benchmark/scripts/record_task1_videos.py [--gifs-only]

Models (same construction as restore_experts_eval.py; no training, checkpoints only read):
  decayed  = final task9_model.pth unchanged;  restored = every expert's A/B/bias rows taken from its own task{k}_model.pth
Each model: 3 rollouts from initial state 0, one environment (no worker processes), the original evaluation loop (5 settle steps,
max_steps from the run config). Frames are the policy's own 128x128 agentview observation, rotated 180 degrees to upright.
The first saved frame is checked for being non-blank before anything else runs.
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
from easydict import EasyDict
from PIL import Image, ImageDraw, ImageFont

CK = "/mnt/data/users/bbouri/CL-benchmark/outputs/D-long10-dmpel-full-rho005-bias-ne10-original-s100/a0/seed_100"
FRAMES = "/mnt/data/users/bbouri/CL-benchmark/outputs/videos_task1"
FIG = "/mnt/data/users/bbouri/CL-benchmark/analysis/figures"
POOL = ("A_pool", "A_q_pool", "A_v_pool", "B_pool", "B_q_pool", "B_v_pool", "bias_pool")
TASK, STATE, N_EP, RED, GREEN = 1, 0, 3, (214, 39, 40), (0, 158, 115)


def frame_of(obs):
    return np.ascontiguousarray(obs[0]["agentview_image"][::-1, ::-1])  # LIBERO images arrive rotated 180 degrees


def record():
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    from libero.lifelong.utils import control_seed, safe_device, torch_load_model, get_task_embs

    cfg = EasyDict(json.load(open(f"{CK}/config.json")))
    cfg.use_ddp = False
    cfg.experiment_dir = "/mnt/data/users/bbouri/CL-benchmark/outputs/restore_test_scratch"
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    control_seed(cfg.seed)
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_tasks = benchmark.n_tasks
    get_dataset(dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(0)), obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True, seq_len=cfg.data.seq_len, demos=range(cfg.data.n_demos_per_task))
    benchmark.set_task_embs(get_task_embs(cfg, [benchmark.get_task(i).language for i in range(n_tasks)]))
    task = benchmark.get_task(TASK)
    print("task:", TASK, task.language, flush=True)
    assert task.language == "put both the cream cheese box and the butter in the basket"

    algo = get_algo_class(cfg.lifelong.algo)(n_tasks, cfg)
    algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path)[0], strict=False)
    algo.policy.init_moe_policy()
    algo = safe_device(algo, cfg.device)
    for _ in range(n_tasks):
        algo.policy.add_new_and_freeze_previous(cfg.policy.ll_expert_per_task)
    final = torch_load_model(f"{CK}/task9_model.pth", map_location="cpu")[0]
    restored = {n: v.clone() for n, v in final.items()}
    for k in range(10):
        ck = torch_load_model(f"{CK}/task{k}_model.pth", map_location="cpu")[0]
        for n in restored:
            if n.split(".")[-1] in POOL:
                restored[n][k] = ck[n][k]

    env = DummyVectorEnv([lambda: OffScreenRenderEnv(bddl_file_name=os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
                                                     camera_heights=cfg.data.img_h, camera_widths=cfg.data.img_w)])
    init_states = torch.load(os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file))
    task_emb = benchmark.get_task_emb(TASK)
    summary = {}
    for cond, sd in (("decayed", final), ("restored", restored)):
        msg = algo.policy.load_state_dict({k: v.to(cfg.device) for k, v in sd.items()}, strict=False)
        assert not msg.unexpected_keys and not [k for k in msg.missing_keys if k.split(".")[-1] in POOL or "moe_router" in k]
        summary[cond] = []
        for r in range(N_EP):
            out = f"{FRAMES}/{cond}/ep{r}"
            os.makedirs(out, exist_ok=True)
            seed = 1000 * (cond == "restored") + r
            torch.manual_seed(seed); np.random.seed(seed)
            algo.eval(); algo.policy.frozen_language_emb = None
            env.reset(); algo.reset()
            obs = env.set_init_state(init_states[STATE:STATE + 1])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros((1, 7)))
            t, success, t0 = 0, False, time.time()
            Image.fromarray(frame_of(obs)).save(f"{out}/frame_{t:04d}.png")
            if cond == "decayed" and r == 0:  # blank-frame check before spending more GPU time
                f = frame_of(obs).astype(float)
                print(f"first-frame check: mean {f.mean():.1f}, std {f.std():.1f}, shape {f.shape}", flush=True)
                if f.std() < 8 or f.mean() < 10:
                    print("BLANK FRAME: rendering is not working (check MUJOCO_GL=egl / EGL device)", flush=True)
                    sys.exit(2)
            while t < cfg.eval.max_steps:
                t += 1
                actions = algo.policy.get_action(raw_obs_to_tensor_obs(obs, task_emb, cfg))
                obs, _, done, _ = env.step(actions)
                Image.fromarray(frame_of(obs)).save(f"{out}/frame_{t:04d}.png")
                if done[0]:
                    success = True
                    break
            summary[cond].append({"episode": r, "success": success, "steps": t, "seed": seed})
            print(f"[{cond}] ep{r}: success={success} steps={t} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(summary, open(f"{FRAMES}/summary.json", "w"), indent=1)


def font(size):
    import matplotlib.font_manager as fm
    return ImageFont.truetype(fm.findfont("DejaVu Sans:bold"), size)


def load_frames(cond, r):
    fs = sorted(glob.glob(f"{FRAMES}/{cond}/ep{r}/frame_*.png"))
    return [np.asarray(Image.open(f).convert("RGB")) for f in fs]


def panel(frame, color, speed, border=3):
    im = Image.fromarray(frame).resize((256, 256), Image.NEAREST)  # 128 -> 256, nearest neighbour
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 255, 255], outline=color, width=border)
    f = font(11)
    d.text((169, 238), f"{speed:g}x speed", font=f, fill=(0, 0, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((169, 238), f"{speed:g}x speed", font=f, fill=(255, 255, 255))
    return im


def banner(text, color, h=30):
    im = Image.new("RGB", (256, h), "white")
    d = ImageDraw.Draw(im)
    f = font(13)
    w = d.textlength(text, font=f)
    d.text(((256 - w) / 2, (h - 15) / 2), text, font=f, fill=color)
    return im


def save_gif(imgs, path, ms=100, hold=1200):
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[ms] * (len(imgs) - 1) + [hold], loop=0, optimize=True)


def gifs():
    s = json.load(open(f"{FRAMES}/summary.json"))
    print("decayed:", [(e["episode"], e["success"], e["steps"]) for e in s["decayed"]], "| restored:", [(e["episode"], e["success"], e["steps"]) for e in s["restored"]])
    failed = [e for e in s["decayed"] if not e["success"]]
    won = sorted([e for e in s["restored"] if e["success"]], key=lambda e: e["steps"])
    if not failed or not won:
        print("cannot build the comparison: need one failed decayed episode and one successful restored episode", "(no failure in the control set)" if not failed else "(no success in the restored set)")
        return
    ok_ctrl = [e["episode"] for e in s["decayed"] if e["success"]]
    if ok_ctrl:
        print("note: control episodes that succeeded:", ok_ctrl, "- the first failed episode is used")
    fe, we = failed[0], won[0]
    fr_d, fr_r = load_frames("decayed", fe["episode"]), load_frames("restored", we["episode"])
    stride = math.ceil(max(len(fr_d), len(fr_r)) / 80)  # same stride for both: the longer episode fits in <= 80 frames
    speed = stride / 2.0  # control runs at 20 Hz, GIF plays at 10 fps
    sd, sr = fr_d[::stride][:80], fr_r[::stride][:80]
    n = max(len(sd), len(sr))
    sd = sd + [sd[-1]] * (n - len(sd))
    sr = sr + [sr[-1]] * (n - len(sr))  # pad the shorter episode with its last frame
    ban_d, ban_r = banner("Decayed experts (SR 0.2)", RED), banner("Restored experts (SR 1.0)", GREEN)
    os.makedirs(FIG, exist_ok=True)
    both = []
    for a, b in zip(sd, sr):
        c = Image.new("RGB", (256 * 2 + 6, 256 + 30), "white")
        c.paste(ban_d, (0, 0)); c.paste(ban_r, (262, 0))
        c.paste(panel(a, RED, speed), (0, 30)); c.paste(panel(b, GREEN, speed), (262, 30))
        both.append(c)
    save_gif(both, f"{FIG}/task1_comparison.gif")
    save_gif([panel(f, RED, speed) for f in fr_d[::stride][:80]], f"{FIG}/task1_decayed.gif")
    save_gif([panel(f, GREEN, speed) for f in fr_r[::stride][:80]], f"{FIG}/task1_restored.gif")
    both[-1].save(f"{FIG}/task1_comparison_last_frame.png")
    print(f"used decayed ep{fe['episode']} ({len(fr_d)} frames, failed) and restored ep{we['episode']} ({len(fr_r)} frames, success); stride {stride} = {speed:g}x real time; {n} GIF frames = {n / 10:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gifs-only", action="store_true")
    a = ap.parse_args()
    if not a.gifs_only:
        record()
    gifs()
