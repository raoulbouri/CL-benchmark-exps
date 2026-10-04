# Running on xulab (host `autolab`, 2x RTX 5090, shared workstation), 3 Oct 2026

Everything lives under `/mnt/data/users/bbouri` (the system disk `/` has ~14 GB free and must not be written).

| Item | Value |
|---|---|
| Machine | Ubuntu 24.04, 32 cores, 249 GB RAM, driver 580.173; other users active (desktop sessions on GPU 0) |
| GPU used | **GPU 1 only** (idle, no display); GPU 0 drives a desktop and is left alone |
| Layout | `miniforge3/` (conda), `cl/` (third_party/DMPEL, checkpoints, data, config), `CL-benchmark/` (this harness), `cache/`, `tmp/`, `logs/` |
| Environment | `source /mnt/data/users/bbouri/cl/env_dmpel.sh` (env `dmpel`: py3.10, torch 2.7.1+cu128) |
| Setup / run | `scripts/xulab_setup.sh` (one time), `scripts/run_xulab.sh <gpu> <queue>` (run + guard) |

## Deviations from the AIMS / published setup (all forced by the hardware or by the shared machine)
- **PyTorch 2.7.1 (cu128) and Python 3.10**, because the RTX 5090 (sm_120) is unsupported by DMPEL's pinned torch 2.4.1 and Python 3.8. numpy 1.23.5 keeps DMPEL's deprecated `np.bool` working. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is needed because DMPEL checkpoints contain pickled config objects.
- `setuptools==69.5.1` (DMPEL imports `pkg_resources.packaging`).
- Single GPU (no DDP), 5 evaluation workers, 8 data-loader workers, nice 10: to be gentle on a shared workstation.
- Different GPU and memory than AIMS, so **cost numbers are not directly comparable** (recorded in each result's provenance).
- W&B: no key on this machine yet, so runs log offline (`WANDB_MODE` falls back automatically). To go online, put `WANDB_API_KEY=...` in `/mnt/data/users/bbouri/CL-benchmark/.env` (chmod 600), then `wandb sync` the offline runs.

## Safety measures (after two incidents on AIMS)
`scripts/guard.py` runs beside every job and stops OUR job (never anyone else's) on: GPU temperature > 85 C, GPU memory > 29 GB, system disk < 4 GB free, host RAM available < 30 GB, `nvidia-smi` not answering twice, or two or more of our processes stuck in uninterruptible sleep for three checks. GPU polling is every 60 s with hard deadlines. Reasons are written to `logs/guard.log`.

## Source of truth
This directory is a COPY of the harness from the Mac working copy, with no git history. The git history and all earlier results live on AIMS (currently unreachable). Reconcile when AIMS returns: results from here carry `host`/`gpu` in their provenance.

## Results from xulab

### D-long2-er-micro-original-s100: complete (3-4 Oct 2026, GPU 1, attempt 0)
DMPEL-codebase ER, 2-task LIBERO-Long micro-stream, original protocol. Single RTX 5090, micro-batch 4 (+4 replay) x 8 accumulation = 32 + 32 per optimizer step, 5 evaluation workers, 8 data-loader workers, nice 10, guard active (no stop events). W&B offline (no key on this machine).

| | |
|---|---|
| Success matrix c[stage][task] | stage 0: 0.60; stage 1: 0.50, 0.90 |
| Selection evaluations | task 0: 0.00 then 0.70; task 1: 0.00 then 0.95 |
| FWT / NBT / AUC / final success | 0.412 / 0.100 / 0.450 / 0.700 (2 tasks, short schedule: costing only, not a result) |
| Wall time | 0.81 h; train 16.7 and 23.3 min per task (incl. selection evals), end-of-task evaluation 1.4 and 2.6 min |
| GPU 1 memory | peak 15.4 GB (training with replay); evaluation peaks 10.0, 9.4 GB; no growth across evaluations |
| System impact | home 536 KB, nothing of ours on `/` (free space moved 14G to 13G from other users), 17 GB under `/mnt/data/users/bbouri` |

What this establishes:
- **DMPEL-codebase ER fits on ONE 32 GB GPU** with gradient accumulation. The two-GPU DDP setup (and its checkpoint race) is not needed. The AIMS failures were GPU-memory pressure from evaluation workers plus concurrency, not ER training itself (not proven on 24 GB 4090s).
- **Cost projection (estimate):** from 4.7 min per pass without replay and about 6.9 min per pass with replay, a full 10-task LIBERO-Long ER run (10 epochs) is about 59 + 9 x 83 min of training plus 55 task-evaluations at about 1.4 min, i.e. **about 15 GPU-hours on one 5090** (roughly the same as DMPEL). This replaces the earlier assumption that ER would cost double because it needs two GPUs.
- 5 evaluation workers give the same evaluation time per task (1.4 min) as 20 workers did on AIMS.
- The harness works unchanged on this machine and PyTorch 2.7.1 once paths come from the environment (fixes: `CLB_*` variables, `wandb` offline mode passed explicitly).

### Still to reconcile when AIMS returns
- Copy this result's JSON and the cost-model rows into the AIMS repository (`results/`, `docs/COST_MODEL.md`); the AIMS ER row there still shows failures.
- This copy of the harness has no git history.
