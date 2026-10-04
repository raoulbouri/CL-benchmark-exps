# CL-Benchmark: critical review and phased plan

> **Status (30 Sep 2026):** superseded by `PI_REVIEW.md` in this directory, which governs where the two disagree. Kept as the record of the v2 reasoning.

Budget-matched continual-learning benchmark for robot manipulation on LIBERO.
Reviewed 30 Sep 2026 against the public code, checkpoints, papers and our own runs on AIMS.
Supersedes the v1 plan published at https://claude.ai/artifact/7XiAhtRJR4zFMPbqiqVHzX (the "v1 plan" below).

Official result log: **Weights & Biases**. The entity is `rahulbouri16` and the project is `cl-benchmark`; conventions are in section 7. The API key lives in `.env` in this directory and must never be committed.

---

## 1. Reviewer's verdict

**Do not approve 300+ GPU-hours on the v1 plan as written.**

The motivation is sound, and stronger than v1 stated. But the design rests on four assumptions that turned out to be false when checked against the code:

1. **"One data source for every method" is not possible with the released CLARE checkpoint.** CLARE re-renders LIBERO at 256×256, drops no-op actions and keeps only successful replays (`lerobot_lsy/src/lerobot/scripts/util/regenerate_libero_hdf5_dataset_as_hf.py`). Its LIBERO-90 checkpoint was trained on that data. DMPEL trains on the original 128×128 HDF5 files.
2. **DMPEL's replay buffer size is not a config knob.** `lifelong=er` reads `n_memories: 1000` and then ignores it. `libero/lifelong/algos/er.py` hardcodes `TruncatedSequenceDataset(dataset, len(dataset)//5)`: the *first* 20% of each task's sequences, not a random sample.
3. **DMPEL's evaluation protocol is biased.** It selects the best epoch using the same first 20 initial states that it then reports on (`libero/lifelong/metric.py`, `indices = np.arange(i*env_num, (i+1)*env_num) % 50`). CLARE does no selection and cycles through all 50 initial states with 100 rollouts. Measured on our three Goal seeds, best-epoch success exceeds last-epoch success by **6.0, 9.0 and 10.0 points on average** (up to 25 on single tasks). Picking the maximum of six 20-episode estimates is expected to add about 1.3 standard errors, roughly 10 points, from noise alone.
4. **The compute estimate is a guess.** Nothing on LIBERO-Long has been timed on our GPUs. DMPEL's ER and EWC baselines fully fine-tune the whole model (CLIP included), and the authors ran them with 2-GPU DDP at batch 16 each. They may not fit on one 4090 at batch 32.

**What I approve now:** phases 0–2 in section 6, at most about 60 GPU-hours. They build the harness, fix the protocol and reproduce both reference methods. Phase 4 decides whether the full budget is justified, using real effect sizes.

---

## 2. Why the benchmark is needed (verified)

Checking the literature made the case stronger than v1 said.

- **CLARE's headline comparison mixes protocols.** CLARE (arXiv 2601.09512, Table III) reports its own numbers from 100 rollouts per task on a ~200M DiT base, beside DMPEL's and MLR's numbers copied from their papers ("For DMPEL and MLR, we report the authors' results"). Those were measured with 20 rollouts (DMPEL), different bases, and, for MLR, a different pretraining protocol (6 tasks, 10 demos per new task).
- **The same methods look very different across papers.** On LIBERO-Long, CLARE reports ER at AUC 60.5 / FWT 76.6 / NBT 22.7 and DMPEL at AUC 58 / FWT 55 / NBT 7 (DMPEL's own numbers). The VLA forgetting study (arXiv 2603.03818) reports π0 with raw replay of 1,000 frames per task at NBT −0.068 on LIBERO-10. Pretraining, data, evaluation and budgets all differ.
- **Protocol details alone move the numbers by several points.** Our DMPEL runs show:
  - best-epoch selection alone shifts just-learned success by 6–10 points;
  - DMPEL's logged forward-transfer value is inflated, because its counter only increments on improvement (can exceed 1);
  - DMPEL's coefficient-replay store measures 404 MB after 10 Goal tasks (float32, every frame), against 0.06 GB in the paper. See the caveat in section 3.
- **Memory decides the forgetting result.** π0 barely forgets at 20% raw replay (NBT ≈ 0) but not at 2% (NBT 0.15 on Goal, 0.23 on LIBERO-10). No paper puts compact-memory methods on the same byte axis.

**Revised thesis:** under one evaluation protocol, compare how each family of methods trades memory and compute for forgetting. Keep within-base comparisons separate from across-base ones.

---

## 3. Claim-by-claim audit of the v1 plan

| # | v1 claim | Verdict | Evidence | Required change |
|---|---|---|---|---|
| 1 | All methods use the original LIBERO HDF5 | **Wrong** | CLARE's regeneration script: 256×256, no-op filtered, success filtered; its checkpoint was trained on that | Two tracks, each on its native data (section 4). An optional data-matched control re-pretrains CLARE's base on our data (200k steps per `bash/clare/pretrain.sh`). |
| 2 | ER at 10 / 100 / 1,000 frames is a setting | **Wrong** | `er.py` hardcodes `len(dataset)//5` and truncates to the first sequences | Build a byte-budgeted, random, segment-level sampler for DMPEL's ER. CLARE's `er.py` selects replay *episodes*; add a budget-to-episodes mapping. |
| 3 | Model selection "as each codebase does it" is acceptable | **Wrong** | 6–10 point optimistic gap, measured | One protocol for everyone: final checkpoint, evaluated on all 50 initial states. Also report DMPEL with its original selection, to quantify the bias. |
| 4 | 20 episodes per cell, 50 for the final row | **Needs change** | DMPEL's 20 episodes always use initial states 0–19, not a sample of the 50 | Evaluate each cell once on each of the 50 initial states. Cost: about 2.5× DMPEL's current evaluation time. |
| 5 | CLARE trains 20,000 steps per task | **Unresolved** | Paper Table I says 10,000 for simulation tasks; `bash/clare/clare_libero_10.sh` uses 20,000 | Time both in phase 1. Ask the authors (GitHub issue) which produced Table III. |
| 6 | CLARE ≈ 1 h per task (RTX 5090) | Unverified on our GPUs | Paper text | Measure in phase 1. |
| 7 | Latent-ER can reuse DMPEL's checkpoint with frozen encoders | **Needs change** | DMPEL's frozen-encoder baseline (`seq_fpf.sh`) uses a separate `chunkonly_frozen` pretraining checkpoint that was not released | Freeze the released checkpoint's encoders and run frozen-encoder Seq, raw ER and latent-ER on that *same* variant, so latent vs raw is a clean comparison. Label it as our variant. |
| 8 | DMPEL stores 404 MB vs the paper's 0.06 GB | **Measurement correct, conclusion unverified** | 10 × `task*_moe_attn_recall_query.pth` = 395 MB, plus `*_attn.pth` 8.3 MB, plus 28.6 MB of experts | Before any public claim, read how the DMPEL paper computed 0.06 GB (per task? fp16? a sample ratio?). Then report both the default and a budget-matched setting (`moe_attn_recall_sample_ratio`, fp16). |
| 9 | DMPEL's logged FWT is buggy | **Correct** | `algos/dmpel.py` increments `cumulated_counter` inside the improvement branch; LIBERO increments at every evaluation | Open a GitHub issue on HarryLui98/DMPEL before blogging. Our metric script recomputes FWT from the curves. |
| 10 | `continuallearning/pi0_libero_90_pretrain` removes the π0 pretraining confound | **Not usable yet** | The HF page has no model card (only "4B params"), so its provenance is unknown | Drop from the plan unless the CLARE authors document it. |
| 11 | π0 with replay is runnable | **Correct, heavy** | `Continual-VLAs/continual-openpi` has `create_deterministic_buffer`, a replay loader, EWC and PackNet. It is JAX, uses OpenVLA-style LeRobot data, and LoRA needs more than 22.5 GB | Stays a stretch goal, gated by a timing pilot. |
| 12 | MLR is partly runnable | **Correct** | The repo releases only the lifelong stage for Goal and Object; no pretraining code | A reference run adds little because its protocol is incompatible. Keep optional. |
| 13 | ~330 GPU-hours for the core tier | **Unverified; likely low** | Only DMPEL Goal has been timed (9.1 h per run). Full fine-tuning ER and EWC need 2-GPU DDP or gradient accumulation | Time everything in phase 1 and re-budget at the phase 4 gate. |
| 14 | Three seeds capture variance | **Partly** | DMPEL's released checkpoint is a single pretraining seed (300); seeds vary only the continual stage | State it. Do not claim variance over pretraining. |
| 15 | Budget-matched comparison across all methods | **Overstated** | Only replay methods can be set to exact byte budgets; CLARE's memory follows its expansion threshold γ | Present Pareto frontiers. Exact matching only inside the replay family. Optionally sweep γ. |
| 16 | W&B as the logger | **Works** | Key verified (account `rahulbouri16`). `wandb.init` succeeds from the old `dmpel` env (wandb 0.13.1) and the `libero` env (0.30.0) | DMPEL has no W&B hooks, only TensorBoard. Add `sync_tensorboard=True` plus explicit logging of the success matrix. Delete the test project `cl-benchmark-keytest` created during verification. |

---

## 4. Revised design (v2)

### Two tracks, one protocol

| | Track D (DMPEL base) | Track C (CLARE base) |
|---|---|---|
| Base checkpoint | `leiyuheng/DMPEL` (LIBERO-90, CLIP ViT-B/16, 174M) | `continuallearning/dit_flow_mt_libero_90_pretrain[_new]` (DiT, ~200M) |
| Data | Original LIBERO HDF5, 128×128, 50 demos per task (on AIMS) | CLARE's regenerated LeRobot datasets (`continuallearning/libero_*_image_task_*`), 256×256 |
| Methods | Seq (full fine-tune), raw ER at byte budgets, DMPEL (default and budget-matched), frozen-encoder Seq, frozen-encoder raw ER, latent-ER, EWC (one seed) | SeqFFT, ER (CLARE repo), CLARE (default γ; γ sweep optional) |
| What it answers | Memory form (raw vs latent vs coefficient) inside one architecture | Exemplar-free vs replay inside one architecture |

**Primary claims are within a track.** Cross-track comparisons use the same metrics and axes, but are labelled as confounded by architecture and data. The optional data-matched control (phase 6) re-pretrains CLARE's base on Track D data to measure that confound.

### Shared protocol (both tracks)

- Suites: LIBERO-Long (primary), LIBERO-Goal (secondary); LIBERO default task order. One robustness run uses the VLA forgetting study's randomised order.
- Checkpoint rule: the final checkpoint of each task; no best-epoch selection. DMPEL is additionally run once with its original rule, to report the bias.
- Evaluation after each task: every seen task, once on each of the 50 initial states, with LIBERO's default episode limit.
- Seeds: 100, 200, 300 for core cells. Pretraining variance is out of scope and stated.
- Metrics: the success matrix `c[k,τ]` and learning curves, stored raw. FWT, NBT, AUC (LIBERO definitions, recomputed from curves), final success, mean just-learned success, normalised NBT.
- Memory: `M_total = M_replay + M_growth + M_aux`, in bytes written to disk in native dtype. Frames are counted as unique frames; history models replay whole 10-frame segments.
- Compute: GPU-hours split into train / eval / other, CPU-hours, peak VRAM, trainable parameters, FLOPs per training step, latency per action.
- Statistics: hierarchical bootstrap (seed → task → episode), paired by seed. "Tie" whenever the 95% interval of a difference includes zero.

### Hypotheses (pre-registered in phase 4, not before)

These are refined from v1 and frozen only after the phase 4 mini-benchmark, so they rest on real effect sizes:

- **H1 (memory form, Track D):** at equal bytes (at most 88 MB), latent-ER forgets less than raw ER on the same frozen-encoder variant.
- **H2 (storing nothing, Track C):** CLARE's NBT is within 0.03 of ER's, at higher parameter growth and compute per task.
- **H3 (compute):** CLARE's advantage in FWT shrinks when its training steps are cut to match DMPEL's GPU-hours.
- **H4 (protocol):** at least one published ordering reverses or becomes a tie under the shared protocol.
- **H5 (stretch):** π0 with raw ER at 88 MB forgets more than the best compact-memory Track D method at 88 MB.

---

## 5. Component inventory: what exists, what needs adapting, what to build

Effort is in focused working days for one person.

### Public and ready (verify install only)

| Component | Source | Notes |
|---|---|---|
| LIBERO simulator and demonstrations | `Lifelong-Robot-Learning/LIBERO` (MIT); HF `yifengzhu-hf/LIBERO-datasets` | Installed and verified on AIMS |
| DMPEL: code, LIBERO-90 checkpoint, Seq / ER / EWC / PackNet / TAIL scripts | `HarryLui98/DMPEL` (MIT); HF `leiyuheng/DMPEL` | Replicated on Goal (FWT 0.70, NBT −0.005, AUC 0.81 over 2 complete seeds) |
| CLARE: code, ER script, LIBERO eval, LIBERO-90 checkpoint, LeRobot datasets | `learnsyslab/clare` (no licence file); HF `continuallearning` | Not yet installed. Python 3.10, custom LeRobot + PEFT forks |
| π0 continual learning with replay, EWC, PackNet | `Continual-VLAs/continual-openpi` (fork of Apache-2.0 openpi) | JAX, Docker-oriented LIBERO eval; stretch only |
| W&B | account `rahulbouri16` | Verified from both envs |

### Minor adaptation (≤ 1 day each)

| Item | Where | Effort |
|---|---|---|
| W&B logging for DMPEL: `wandb.init(sync_tensorboard=True)`, success matrix and curves as tables, `result.pt` as an artifact | `third_party/DMPEL/libero/lifelong/main.py` via a wrapper | 0.5 |
| CLARE's W&B entity and project; map its outputs to our results schema | CLARE bash scripts; converter script | 0.5 |
| Evaluation protocol: final-checkpoint option and all-50-initial-states evaluation in DMPEL | `algos/*.py` selection logic, `metric.py` indices | 0.5–1 |
| Check CLARE's evaluation against the same protocol (it already cycles the 50 states) | `scripts/eval_libero.py` | 0.25 |
| DMPEL coefficient-store budget: `moe_attn_recall_sample_ratio`, fp16 storage | `algos/dmpel.py` | 0.25 |
| CLARE ER budget: map bytes to a number of replay episodes | `scripts/er.py`, `replay_dataset.episodes` | 0.5 |
| Metrics: final success, normalised NBT, bootstrap | extend `scripts/compute_metrics.py` | 1 |

### Build from scratch

| Item | Purpose | Effort |
|---|---|---|
| Benchmark harness: YAML run specs, one launcher that wraps each codebase, per-GPU queue (nohup, lock files, resume, idempotent run IDs), W&B run naming | Every run is reproducible and logged the same way | 2 |
| Results schema and aggregator (JSON per run, pulled into tables and figures) | One source of truth; W&B mirrors it | 1.5 |
| Memory and compute accounting tool (bytes on disk, parameter growth, GPU and CPU hours, peak VRAM, FLOPs) | The budget axes | 1 |
| Byte-budgeted random segment sampler for DMPEL's ER | H1 | 0.5–1 |
| Latent-ER for Track D: feature cache from frozen encoders, latent dataset, replay mixing, frozen-encoder policy config | H1 | 3–4 |
| Figure scripts (Pareto plots, heatmaps, growth curves) | Blog | 1.5 |
| Optional: pipeline to re-pretrain CLARE's base on Track D data | Data-matched control | 1 + GPU time |

**Total development before large runs: about 12–15 working days.** This, not GPU time, is the critical path.

---

## 6. Phased plan

Each phase builds on the previous one and ends with a gate. A phase is not started until the previous gate passes and is signed off.

### Phase 0: Infrastructure (no training GPU time; ≤ 2 GPU-h of smoke tests)

Goal: every later run is launched, logged and stored the same way.

1. Set up the repository in `/usr1/home/bbouri/CL-benchmark`: `git init`, a `.gitignore` that contains `.env`, `outputs/`, `wandb/` and caches.
2. Layout: `configs/` (YAML run specs), `cl_bench/` (launcher, accounting, metrics, schema), `scripts/` (queue, backfill, figures), `results/` (JSON per run, git-tracked), `docs/` (protocol and decision log), and this `PLAN.md`.
3. Record third-party commit hashes in `configs/third_party.lock`: DMPEL `b1abe28`, openpi `215abfb`, and CLARE/continual-openpi when cloned.
4. W&B conventions (section 7), in a helper that every wrapper calls.
5. Metrics library with unit tests: it must reproduce our DMPEL Goal numbers from the existing `result.pt` and `task*_auc.log` files.
6. Accounting tool: bytes on disk per stored artifact, parameter growth, timings.
7. **Backfill:** log the three existing DMPEL Goal seeds to W&B through the new schema. This tests the whole pipeline at zero GPU cost.

**Gate:** backfilled runs appear in W&B with success matrices, curves, config and accounting; unit tests pass; a new run can be launched and resumed from the queue.

### Phase 1: Instrument both codebases on a 2-task micro-stream (≈ 10–20 GPU-h)

Goal: prove every method runs through the harness, and time it.

1. Install CLARE (new conda env `clare`). Download its checkpoint and LIBERO-Long datasets. Resolve the `_new` checkpoint name.
2. Track D, LIBERO-Long tasks 0–1, 2 epochs: Seq, ER (existing 20% rule), DMPEL.
3. Track C, LIBERO-Long tasks 0–1, 1–2k steps: SeqFFT, ER, CLARE.
4. Implement the shared evaluation protocol (final checkpoint, 50 initial states) in both tracks. Check it against DMPEL's original rule on the micro-stream.
5. Measure per-step time, per-evaluation time, peak VRAM, and whether full-fine-tune ER fits on one 4090 at batch 32.
6. Open GitHub issues: CLARE (10k vs 20k steps; checkpoint name), DMPEL (FWT counter; how 0.06 GB was measured).

**Gate:** every method produces a valid results JSON and W&B run through the harness, with measured timings. Re-estimate all later phases from these timings.

### Phase 2: Reproduce the two reference methods on LIBERO-Long (≈ 40 GPU-h)

Goal: trust the baselines before comparing anything.

1. DMPEL on LIBERO-Long, 1 seed, with its **original** protocol. Target: the paper's AUC 58, FWT 55, NBT 7.
2. CLARE on LIBERO-Long, 1 seed, with its **original** protocol. Target: AUC 75.1, FWT 75.0, NBT 1.9.
3. The same two runs re-scored under the shared protocol. Save the curves so both rules can be scored from the same run where possible.

**Gate (go / no-go):** each method lands within about 5 AUC points of its paper. If not, debug before continuing. A failed reproduction is itself a finding, but it must be understood first.

### Phase 3: Build the budget knobs (development; ≈ 15 GPU-h of checks)

1. Byte-budgeted ER sampler for Track D, and byte-to-episode mapping for Track C.
2. DMPEL coefficient-store budget settings.
3. Latent-ER plus the frozen-encoder Seq and raw-ER baselines.
4. Check on a 3-task LIBERO-Long mini-stream (tasks 0–2), 1 seed. Accounting must equal bytes on disk. Sanity checks: ER-1000 ≥ ER-10 on NBT; latent-ER at B2 stores what it claims.

**Gate:** accounting tests pass and every budget variant runs end to end.

### Phase 4: Mini-benchmark on the 3-task stream (≈ 60–80 GPU-h)

Goal: run the whole experiment cheaply to get real effect sizes.

1. All core cells from both tracks, 3 seeds, on LIBERO-Long tasks 0–2.
2. Compute bootstrap intervals and effect sizes. Decide which cells can separate at 10 tasks and which will be ties.
3. Freeze the hypotheses and analysis plan in `docs/PREREGISTRATION.md` and tag the commit.

**Gate:** a costed list of full-length cells, justified by phase 4 effect sizes, approved before any 10-task sweeps.

### Phase 5: Full LIBERO-Long benchmark (≈ 250–350 GPU-h, re-estimated at the phase 4 gate)

Only the cells approved at phase 4, 3 seeds, 10 tasks, both tracks.

### Phase 6: Breadth and controls (gated individually)

- LIBERO-Goal for the core cells (Track D DMPEL already exists).
- Randomised task order (1 seed).
- EWC and DMPEL's original-protocol variant, for reference.
- Optional data-matched control: re-pretrain CLARE's base on Track D data.
- Stretch: π0 with raw ER through `continual-openpi`, after a 200-step timing pilot on one 4090, capped at 120 GPU-h.

### Phase 7: Analysis and write-up

Figures, hypothesis verdicts, reproduction report, released result JSONs, and the blog post.

| Phase | GPU-hours (estimate) | Cumulative |
|---|---|---|
| 0 | ≤ 2 | 2 |
| 1 | 10–20 | ≤ 22 |
| 2 | ≈ 40 | ≤ 62 |
| 3 | ≈ 15 | ≤ 77 |
| 4 | 60–80 | ≤ 157 |
| 5 | 250–350 | ≤ 507 |
| 6 | per item | – |

---

## 7. Logging and data conventions

- **W&B:** entity `rahulbouri16`, project `cl-benchmark`.
  - Run name: `{track}-{suite}-{method}-{budget}-s{seed}`, e.g. `D-long-er-B2-s100`.
  - `group`: `{track}-{suite}-{method}-{budget}`; `job_type`: `train` or `eval`.
  - Tags: phase (`p1`…`p7`), protocol (`shared` or `original`).
  - Config: the full resolved run spec plus third-party commit hashes and the harness commit.
  - Logged per task: training loss, the success curve, the new row of the success matrix, and accounting (bytes, parameters, timings).
  - Artifacts: `result.json` (success matrix, curves, accounting); DMPEL's `result.pt` and learning curves; the evaluation video sample CLARE already renders.
- **Local source of truth:** `results/{run_name}.json`, git-tracked. W&B mirrors it. If they disagree, the local JSON wins and the W&B run is re-logged.
- **Key handling:** `WANDB_API_KEY` is loaded from `.env` by the launcher. It never appears in configs, logs or commits.
- **Decision log:** `docs/DECISIONS.md` records every protocol change, with date and reason.
- **Disk:** `/usr1` is a shared disk at about 95% use. Per-task checkpoints are deleted after metrics and artifacts are uploaded, and the launcher refuses to start if less than 150 GB is free.

---

## 8. Risks

| Risk | Early sign | Response |
|---|---|---|
| CLARE or DMPEL does not reproduce on LIBERO-Long | Phase 2 misses by more than 5 AUC points | Stop and debug. Check data, protocol and steps; ask the authors. |
| Full fine-tune ER / EWC do not fit on one 4090 | Out-of-memory in phase 1 | Gradient accumulation to batch 32; otherwise 2-GPU DDP (halves parallelism; re-budget). |
| Evaluation cost with 50 initial states per cell | Phase 1 timings | Reduce to the diagonal plus final row at 50, other cells at 20 randomly drawn states. Record it in the decision log. |
| Latent-ER takes longer than 4 days to build | Phase 3 slips | Ship Track D without latent-ER first. H1 moves to a follow-up. |
| Methods tie everywhere | Phase 4 intervals | A valid result. It changes the blog's thesis to "under one protocol, the differences disappear". |
| Shared disk fills | Free space under 150 GB | The launcher refuses new runs; prune checkpoints. |

---

## 9. Sources

Papers
- DMPEL, arXiv 2506.05985
- CLARE, arXiv 2601.09512 (RA-L 2026)
- MLR+IFA, arXiv 2603.10929 (CVPR 2026)
- Pretrained VLAs resist forgetting, arXiv 2603.03818
- LiMoDE, arXiv 2606.26183 (context; no public code found)
- LIBERO, arXiv 2306.03310

Code
- https://github.com/HarryLui98/DMPEL (MIT)
- https://github.com/learnsyslab/clare (no licence file)
- https://github.com/yfqi/lifelong_mlr_ifa (MIT)
- https://github.com/Continual-VLAs/continual-openpi (openpi fork)
- https://github.com/Physical-Intelligence/openpi (Apache-2.0)
- https://github.com/Lifelong-Robot-Learning/LIBERO (MIT)
- https://github.com/huggingface/lerobot (Apache-2.0)

Checkpoints and data
- https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
- https://huggingface.co/leiyuheng/DMPEL
- https://huggingface.co/continuallearning (CLARE checkpoints and datasets; `pi0_libero_90_pretrain` has no model card)

Measurements on AIMS used in this review
- DMPEL LIBERO-Goal replication and timings: `/usr1/home/bbouri/continual-learning/logs/dmpel/`, `outputs/dmpel/libero_goal/`
- Best-vs-last epoch gap: computed from `task*_auc.log` for seeds 100, 200 and 300
