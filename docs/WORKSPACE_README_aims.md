# continual-learning (LIBERO): workspace

> **Scope (updated 3 Oct 2026).** This directory is the shared workspace: conda environments, data, checkpoints, caches, third-party code and the original DMPEL replication. The benchmark itself (harness, run specs, results, plan and decision log) lives in `/usr1/home/bbouri/CL-benchmark`; start with its `docs/REPO_MAP.md`. The replication scripts below are still valid but predate the benchmark.

Lifelong robot-learning workspace on AIMS (`riddle`, 2× RTX 4090). Everything — conda, code, data, caches, outputs — lives under this directory.

```bash
source /usr1/home/bbouri/continual-learning/env.sh        # conda env `libero` (original LIBERO, py3.10) -- data checks
python scripts/verify_libero.py                             # end-to-end sanity check -> outputs/verify_report.json
source /usr1/home/bbouri/continual-learning/env_dmpel.sh  # conda env `dmpel` (DMPEL code, py3.8) -- training/eval
```

## Layout
| Path | Contents |
|---|---|
| `miniforge3/` | Project-local conda; envs `libero` (py3.10, torch 2.4.1+cu121; also runs the benchmark harness), `dmpel` (py3.8, torch 2.4.1+cu118, MuJoCo 2.3.7) and `clare` (py3.10, torch 2.6.0+cu124, MuJoCo 3.3.0, transformers 4.56.2) |
| `third_party/LIBERO/` | Official LIBERO repo (editable install in env `libero`) |
| `third_party/DMPEL/` | Official DMPEL code (github.com/HarryLui98/DMPEL, a LIBERO fork; editable install in env `dmpel`), with our patches (see `CL-benchmark/configs/patches/`) |
| `third_party/clare/` | CLARE (github.com/learnsyslab/clare: modified LeRobot + PEFT), with our evaluation patch; env `clare` |
| `third_party/gym-libero/` | Gymnasium wrapper of LIBERO that CLARE's environment factory imports |
| `third_party/openpi/` | π0 / π0.5 code, read for the architecture study; not used by benchmark runs |
| `checkpoints/` | `dmpel_pretrain/` (authors' LIBERO-90 model), `clare/` (authors' LIBERO-90 model), `hf/` (CLIP ViT-B/16 local copy) |
| `config/libero/`, `config/dmpel/` | LIBERO path configs (used via `LIBERO_CONFIG_PATH`; nothing in `~/.libero`); both point at the shared `data/libero` |
| `data/libero/<suite>/*.hdf5` | Demonstrations, 50 demos/task |
| `cache/{hf,torch,pip}` | Model weights (CLIP ViT-B/16) and package caches |
| `scripts/` | `setup_env.sh`, `download_data.sh`, `verify_libero.py`, `setup_dmpel_env.sh` |
| `outputs/`, `logs/` | Runs, W&B, verification artifacts, setup logs |

## Data
| Suite | Tasks | Size | Role |
|---|---|---|---|
| `libero_90` | 90 | 66.7 GB | LIBERO-100 part 1: pretraining (DMPEL) |
| `libero_10` (LIBERO-Long) | 10 | 13.7 GB | LIBERO-100 part 2: long-horizon lifelong suite |
| `libero_goal` | 10 | 6.4 GB | Lifelong suite |
| `libero_spatial` | 10 | 6.2 GB | Lifelong suite |
| `libero_object` | 10 | 7.4 GB | Lifelong suite |

Source: HF `yifengzhu-hf/LIBERO-datasets`. Per demo (`data/demo_i/`): `actions` (T×7: Δpos, Δrot, gripper), `states` (sim states for replay), `obs/{agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori, ee_states, gripper_states, joint_states}` (128×128 RGB), `rewards`, `dones`.

## DMPEL replication (completed; the first step of the project)
**DMPEL** (arXiv:2506.05985; code github.com/HarryLui98/DMPEL; LIBERO-90 checkpoint HF `leiyuheng/DMPEL`):
- Pretrain on LIBERO-90, then adapt sequentially on the Goal/Spatial/Object/Long 10-task suites. We start from the authors' pretrained checkpoint.
- Policy: CLIP ViT-B/16 vision and text encoders, FiLM fusion, a temporal transformer, and a GMM action head. Inputs are agentview, eye-in-hand and proprio.
- Continual learning: a progressive library of LoRA experts (rank 8 in CLIP, 16 elsewhere), a router MLP, and expert-coefficient replay.
- Hyperparameters: 10 epochs per task, batch size 32, AdamW at 1e-4, evaluation every 2 epochs with 20 episodes per task, seeds 100/200/300.
- Paper targets on LIBERO-Goal: FWT 0.68±0.03, BWT 0.00±0.01, AUC 0.78±0.02. The other suites' targets are in the paper's Fig. 3.

Run and evaluate:
- `scripts/run_dmpel.sh <bench> <seed> <gpu>` launches a run; logs go to `logs/dmpel/`, results to `outputs/dmpel/<bench>/seed_<s>/seed_<s>/`.
- `python scripts/compute_metrics.py <bench>` (in env `dmpel`) computes FWT/NBT/AUC.
- DMPEL's logged `S_fwd` in `result.pt` is inflated and can exceed 1. `algos/dmpel.py` increments `cumulated_counter` only when success improves, whereas LIBERO increments it at every evaluation. For this reason FWT is computed from the per-task `task{k}_auc.log` curves using the LIBERO definition.
- Compatibility changes to DMPEL:
  - `scripts/patch_dmpel_datasets.py` lets `SequenceDataset` accept `demos=`, which no released robomimic supports. It loads the same 50 demos per task.
  - The CLIP text encoder is loaded from `checkpoints/hf/clip-vit-base-patch16`, because transformers 4.21.1 can't follow the HF Hub's redirects.

**Result:** on LIBERO-Goal, 3 seeds, full schedule: FWT 0.704, NBT 0.000, AUC 0.806 (paper 0.68 / 0.00 / 0.78). The current project is the budget-matched continual-learning benchmark in `CL-benchmark/`; PHASER-style phase-aware replay is deferred.

## Environment gotchas
- LIBERO's torch 1.11/cu113 pin can't run on a 4090 (sm_89), so env `libero` uses torch 2.4.1+cu121 with `numpy<2`, `robosuite==1.4.0` and `mujoco==2.3.7`.
- Env `dmpel` follows DMPEL's `requirements.txt` (torch 2.4.1+cu118, robosuite 1.4.1, transformers 4.21.1), plus `mujoco==2.3.7`. Its `torchtext` pin names a version that doesn't exist; torchtext is unused, so it is skipped.
- Both LIBERO copies must be installed with `--config-settings editable_mode=compat`. Otherwise `import libero` fails because the top-level `libero/` has no `__init__.py`.
- Env `clare` needs MuJoCo 3.3.0 (2.3.7 breaks `gym_libero` rendering), transformers 4.56.2, diffusers 0.35.1, and the `v2.1` revision of CLARE's datasets (details: `CL-benchmark/docs/DECISIONS.md`).
- Only one DMPEL or CLARE job at a time should run evaluation on this shared machine (host RAM, GPU-context limits).
- Always source `env.sh` or `env_dmpel.sh` first. Without `LIBERO_CONFIG_PATH`, LIBERO prompts interactively and writes to `~/.libero`.
