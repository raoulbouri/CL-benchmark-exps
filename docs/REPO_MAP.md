# Repository map: CL-benchmark and everything it depends on

Written 2026-10-03 from the actual file list on AIMS. It is a guide, not a source of results: for what was decided and why, read `DECISIONS.md`.

## 0. How to read this project (suggested order)

| Step | Read | Time | You learn |
|---|---|---|---|
| 1 | `PI_REVIEW.md` ("Status and amendments" first, then §0–2) | 20 min | What the project is, why it exists, what is approved, the four contributions (C1–C4) |
| 2 | `docs/PROTOCOL.md` | 5 min | The two evaluation protocols (`original`, `heldout`) and every metric formula |
| 3 | `docs/DECISIONS.md` | 30 min | Every protocol/scope change, bug, workaround and incident, dated. The most complete record |
| 4 | `PI_REVIEW.md` §3–5 | 15 min | What is public vs adapted vs built, the infrastructure standard, the phase plan and gates |
| 5 | `docs/COST_MODEL.md` | 5 min | Measured costs from the Phase 1 micro-runs |
| 6 | This file §2–4 | 20 min | Where each piece of code lives and what it does |
| 7 | `docs/HANDOFF.md` | 5 min | Current status, ground rules, open decisions, next steps (rewritten 3 Oct). A shorter version of this reading list |

Superseded or reference only: `PLAN.md` (the v2 review; `PI_REVIEW.md` governs), `docs/ISSUE_DRAFTS.md` and `docs/ISSUES_DMPEL_FOR_APPROVAL.md` (drafts; issues #6–#8 were posted on HarryLui98/DMPEL, issue 4 was not).

External papers, in the order they matter: LIBERO (arXiv 2306.03310) → DMPEL (2506.05985) → CLARE (2601.09512) → "Pretrained VLAs resist forgetting" (2603.03818) → MLR+IFA (2603.10929) → Zhou et al. "A Model or 603 Exemplars" (2205.13218) → Prabhu et al. "Computationally Budgeted CL" (2303.11165) → LIBERO-PRO (2510.03827), LIBERO-Plus.

## 1. Where things live

| Location | What | In git? |
|---|---|---|
| `/usr1/home/bbouri/CL-benchmark` (AIMS) | The benchmark harness, specs, results, docs. **Source of truth** | Yes (AIMS only, no remote) |
| `/usr1/home/bbouri/continual-learning` (AIMS) | Older workspace: conda envs, data, checkpoints, the cloned third-party code with our patches applied | No |
| `~/Developer/CL-benchmark-HANDOFF.md` (Mac) | Copy of the 2 Oct handoff | – |
| `~/.claude/jobs/…/tmp/clb` (Mac) | Working copy of the harness, rsynced to AIMS (never edit docs here) | No |
| W&B `rahulbouri16/cl-benchmark` | Mirror of metrics for every run | – |

## 2. The benchmark repository, file by file (`CL-benchmark/`)

### Top level
- `PI_REVIEW.md`: governing plan and phase gates. `PLAN.md`: earlier review, superseded.
- `.gitignore`: excludes `.env` (W&B key), `outputs/`, `wandb/`, checkpoints and caches. `.env` holds `WANDB_API_KEY`, never committed.

### `cl_bench/`: the library (no training code is imported here)
| File | Purpose |
|---|---|
| `__init__.py` | Paths (`ROOT`, `CL_ROOT`), W&B entity/project, `.env` loader |
| `metrics.py` | FWT, NBT, AUC, final success, normalised NBT; parametric bootstrap intervals; seed aggregation. Uses the LIBERO FWT definition, **never** DMPEL's logged `S_fwd` |
| `accounting.py` | Memory (replay + growth + aux bytes on disk), parameter counts, per-task timings parsed from DMPEL logs |
| `results.py` | Run-name convention, the results JSON schema, atomic writes, provenance (commit, GPU, host) |
| `wandb_util.py` | W&B init with deterministic ids and offline fallback; payloads kept small (free plan is 5 GB) |
| `dmpel_export.py` | Turns a DMPEL-codebase run directory + log into a results record and W&B rows, task by task |
| `clare_export.py` | Same for the CLARE codebase (one process per task; parses CLARE's logs; adapter size from `.safetensors` headers) |

### `scripts/`
| File | Purpose |
|---|---|
| `launch.py` | Runs one spec: builds the command for either codebase, runs it in its own process group, exports after every task, logs to W&B, samples GPU memory (`gpu_mem.csv`), handles kill signals, refuses to start if disk is low or a GPU is busy. Supports 2-GPU DDP |
| `queue.sh` | Runs a list of specs one after another on a GPU, skipping runs already `complete` |
| `chain_p1_final.sh` | One-off: ran the last Phase 1 jobs strictly one at a time |
| `patch_dmpel_protocol.py` | Adds `max_tasks` and `eval.protocol` (`original` / `heldout`) to DMPEL |
| `patch_dmpel_grad_accum.py` | Gradient accumulation (`train.grad_accum_steps`) for DMPEL's ER |
| `patch_dmpel_ddp_ckpt.py` | Fixes a two-process checkpoint race in DMPEL's `base.py` (found 3 Oct) |
| `patch_clare_protocol.py` | CLARE: held-out initial-state pinning, and success counting that works under Gymnasium ≥ 1.0 |
| `setup_clare_env.sh` | Builds the `clare` conda env and downloads its checkpoint and datasets |
| `backfill_dmpel.py` | Phase 0: loaded the three finished LIBERO-Goal replication runs into `results/` and W&B |
| `report_phase1.py` | Rewrites the results table in `docs/COST_MODEL.md` from the Phase 1 results; preserves the hand-written projection section |

All `patch_*.py` scripts are idempotent and edit third-party clones in `continual-learning/third_party/`. The resulting diffs are saved in `configs/patches/`.

### `configs/`
- `runs/p0/`, `runs/p1/`: one YAML per run (track, suite, method, protocol, seed, overrides). Naming: `{track}-{suite}{ntasks}-{method}-{budget}-{protocol}-s{seed}`; D = DMPEL base, C = CLARE base. Comments in each file record why settings changed.
- `patches/*.patch`: the changes applied to third-party code (5 files; reapply with the `patch_*.py` scripts).
- `third_party.lock`: commits of DMPEL, LIBERO, openpi at the time of use.

### `queues/`: lists of spec paths fed to `queue.sh` (one file per past batch; historical)
### `results/`: one JSON per run (success matrix, curves, metrics, accounting, timings, provenance). **Source of truth for numbers.** 10 files so far
### `tests/test_metrics.py`: reproduces the DMPEL Goal numbers and checks the metric definitions
### `logs/`: queue and setup logs (ignored by git since 3 Oct; still on disk)
### `outputs/` (not in git): per-attempt run directories: `train.log`, `command.sh`, `gpu_mem.csv`, checkpoints, per-task files
### `docs/`
`PROTOCOL.md`, `DECISIONS.md`, `COST_MODEL.md`, `HANDOFF.md`, `ISSUE_DRAFTS.md`, `ISSUES_DMPEL_FOR_APPROVAL.md`, and this file.

## 3. The third-party code (`continual-learning/third_party/`)

| Clone | What it is | Where the action is |
|---|---|---|
| `DMPEL/` (github.com/HarryLui98/DMPEL) | A fork of LIBERO's lifelong-learning code plus the DMPEL method | `libero/lifelong/main.py` (task loop), `algos/` (`base.py` Sequential, `er.py`, `ewc.py`, `packnet.py`, `dmpel.py`, …), `models/` (policies, CLIP encoders, LoRA adapters), `datasets.py`, `metric.py` (evaluation), `libero/configs/` (Hydra configs) |
| `clare/` (github.com/learnsyslab/clare) | CLARE on a modified LeRobot + PEFT | `lerobot_lsy/src/lerobot/scripts/clare.py` (the method), `er.py` (replay baseline), `eval_peft.py` (evaluation), `peft_lsy/` (the adapter implementation), `bash/clare/` (authors' run scripts) |
| `gym-libero/` | Gymnasium wrapper of LIBERO that CLARE's environment factory imports (not declared by CLARE) | `gym_libero/env.py` |
| `LIBERO/` | Original LIBERO simulator, tasks and metric definitions | used for the Track D data checks and reference renders |
| `openpi/` | π0 / π0.5 code, read for the architecture study; not used in benchmark runs | `src/openpi/models/` |

Patches applied on top of DMPEL: data-loading `demos` shim, protocol and task cap, gradient accumulation, DDP checkpoint fix. On top of CLARE: held-out pinning and the success-counting fix. Everything else is as released.

## 4. The workspace around it (`continual-learning/`)
| Path | What |
|---|---|
| `miniforge3/envs/` | `libero` (harness + data checks, py3.10), `dmpel` (py3.8, torch 2.4.1+cu118, MuJoCo 2.3.7), `clare` (py3.10, torch 2.6.0, MuJoCo 3.3.0) |
| `data/libero/` | 94 GB of LIBERO demonstrations: libero_90, libero_10 (Long), libero_goal, libero_spatial, libero_object, 50 demos per task, 128×128 |
| `checkpoints/` | `dmpel_pretrain` (authors' LIBERO-90 model), `clare` (authors' LIBERO-90 model), `hf/` (CLIP ViT-B/16 local copy) |
| `outputs/dmpel/libero_goal/` | The three full replication runs (seeds 100/200/300) |
| `logs/`, `scripts/`, `README.md`, `env.sh`, `env_dmpel.sh`, `config/` | Earlier setup and replication scripts; `env.sh` and `env_dmpel.sh` activate the environments |

## 5. How one run flows through the code
`configs/runs/*.yaml` → `queue.sh` → `launch.py` (pre-flight, W&B init, GPU sampler) → the codebase's own training command (DMPEL `main.py`, or CLARE `clare.py`/`er.py` once per task) → after each task, `dmpel_export.py`/`clare_export.py` read the logs and files → `metrics.py` + `accounting.py` → `results/{name}.json` (atomic write) + W&B rows → at the end, success-matrix table and result artifact to W&B.

## 6. Known gaps in this documentation
- No docstring-level API docs for `cl_bench/`; the module headers above are the only description.
- `continual-learning/README.md` was refreshed on 3 Oct but still describes the older replication scripts in `continual-learning/scripts/`, which the benchmark does not use.
- `PLAN.md` (v2 review) is kept for history and is superseded by `PI_REVIEW.md`.
