# CL-Benchmark: PI review of PLAN.md (v2) and approved phase plan

Reviewer role: PI. Reviewed 30 Sep 2026.
Inputs: `PLAN.md` (v2, written by the PhD student, reviewing the grad student's v1), the public code and checkpoints, the papers cited below, and our own runs on AIMS.

Where this document and `PLAN.md` disagree, **this document governs**. `PLAN.md` stays as the record of the v2 reasoning.

Official result log: **Weights & Biases**, entity `rahulbouri16`, project `cl-benchmark` (verified working from both conda envs on 30 Sep). The key is read from `.env` only.

---

## Status and amendments (updated 3 Oct 2026)

This section was added after the review as work progressed. **Where it differs from the sections below, it governs.** The dated detail for every item is in `docs/DECISIONS.md`; the file-level guide is `docs/REPO_MAP.md`.

### Where the phases stand

| Phase | Status | Notes |
|---|---|---|
| 0: system of record | **Complete, gate passed** | Harness, metrics tests (reproduce the DMPEL Goal numbers exactly), kill/skip behaviour, W&B backfill of the 3 Goal replication runs. One adaptation: resume is at run level, not task level (DMPEL cannot resume mid-stream) |
| 1: every method through one pipe | **5 of 6 micro-runs complete; ER (DMPEL codebase) in progress** (attempt 7 started 3 Oct 15:41 UTC) | About 8 of 15 approved GPU-hours used, including failed attempts. Cost model written; full-schedule projection added (`docs/COST_MODEL.md`) |
| 2 onwards | **Not started, not approved** | Waiting for the Phase 1 gate and the planning discussion |

### Phase 1 gate criteria

| Criterion | Status |
|---|---|
| DMPEL, held-out and original: valid results + W&B | Done |
| CLARE, held-out and original: valid results + W&B | Done (after fixes below) |
| CLARE-codebase ER: valid results + W&B | Done |
| DMPEL-codebase ER: valid results + W&B | **Open.** Attempts 0–5 failed (memory, then a driver incident); attempt 6 failed on a DDP checkpoint race (fixed); attempt 7 running |
| Cost model replaces estimates | Projection written from measured numbers; **ER row and peak VRAM for CLARE pending attempt 7 and the next CLARE run** |
| Held-out protocol works in both codebases | Done (verified on real runs) |

### Amendments to the plan above

1. **Component status corrections.**
   - CLARE's ER baseline is "minor adaptation", not "ready": CLARE releases no ER run script, so the harness builds the command.
   - CLARE's LIBERO environment is not in its repo; it needs `gym-libero`.
   - Track C requires MuJoCo 3.3.0 (2.3.7 breaks `gym_libero` rendering); Track D keeps 2.3.7.
2. **CLARE reproduction uses the paper's settings** (user decision): 10k adapter steps, batch 32, constant learning rate 1e-4, expansion threshold γ = 2.5, 2k discriminator steps. Not the released script's 20k steps and γ = 1.0.
3. **DMPEL-codebase ER runs on 2 GPUs with gradient accumulation** (user decision): 2 × 4 (+4 replay) × 4 steps = 32 + 32 per optimizer step, the authors' effective batch, with 10 evaluation workers instead of 20. The half-batch fallback spec exists but is not queued. **Budget consequence: an ER run occupies both GPUs, so its GPU-hours count double.**
4. **Evaluation cost is larger than the plan assumed.** CLARE evaluation, not training, dominates its run time (see `COST_MODEL.md`). The ≈ 75 GPU-hour Phase 2 estimate still holds (≈ 68 projected), but the **Phase 5 cap of 250 GPU-hours is likely too small** once ER cells run on two GPUs each; this is a planning-discussion item.
5. **FWT is not comparable across codebases.** DMPEL has a per-epoch selection curve; CLARE has none, so its FWT equals the just-learned success. Cross-track comparisons should use AUC, NBT and final success, with this caveat stated (`docs/PROTOCOL.md`).
6. **Issue posting:** issues #6, #7 and #8 were posted on the DMPEL repository (FWT counter, ER buffer size, AMP gradient clipping). The installation issue and all CLARE questions were not posted (user decision). CLARE questions are therefore settled by our own checks.

### Incidents and process rules (details in DECISIONS.md)

- **GPU driver wedge on AIMS, 2 Oct 04:35 UTC.** Two jobs creating many GPU contexts at once, ours (a memory probe with 20 rendering workers) and another user's, deadlocked the NVIDIA driver; an admin restart cleared it. Rules adopted: no probes that open many GPU contexts; 10 evaluation workers; no launches within minutes of another user's GPU job. Cause is not proven (kernel logs need root).
- **Docs are edited on AIMS only.** A copy from the Mac overwrote three notes in `DECISIONS.md` (restored from git). Syncs from the Mac now exclude `docs/`.
- **Completion is not correctness.** Several runs completed with wrong results (CLARE's 0% success from broken rendering and a broken success counter). Each reference method must be validated against its paper in Phase 2 before any comparison.

### Open decisions

1. Phase 1 gate sign-off, after ER attempt 7 and the CLARE VRAM capture.
2. Whether benchmark runs should fix the AMP gradient-clipping behaviour of the ER/EWC/PackNet baselines (Phase 4 decision; the `original` protocol keeps the released behaviour).
3. The Phase 5 budget and cell list, given ER's cost (amendment 4).
4. Whether to put the harness repository on a private remote (it currently exists only on AIMS).
5. Deleting the W&B test project `cl-benchmark-keytest` (manual).

---

## 0. Decision

| | |
|---|---|
| Approved now | Phase 0 (infrastructure, no training) and Phase 1 (wrappers and smoke runs, ≤ 15 GPU-hours) |
| Approved at the Phase 1 gate | Phase 2 (reference reproductions, ≈ 75 GPU-hours), after real timings exist |
| Not approved yet | Everything from Phase 3 on. Each phase is approved separately at its gate. |
| GPU-hours committed today | **≤ 15** |

v2 was a large improvement on v1, and most of its code audit holds up: I re-verified the data mismatch, the ER buffer bug, the evaluation leak, the W&B setup and the CLARE copy-paste comparison. But v2 has five problems a PI cannot sign off on:

1. **It misstates its own novelty.** Memory-aligned comparison that counts model parameters (Zhou et al., ICLR 2023) and compute-budgeted continual learning (Prabhu et al., CVPR 2023) are established ideas in vision. v2 cites neither. The contribution is transferring them to robot manipulation, which is still worthwhile, but it must be framed that way.
2. **It ignores the known weakness of LIBERO's evaluation.** LIBERO-PRO and LIBERO-Plus show that LIBERO success at fixed initial states largely reflects memorised trajectories (above 90% collapsing to 0% under LIBERO-PRO; 95% to below 30% under LIBERO-Plus). A forgetting benchmark scored only that way measures retention of memorised trajectories. Reviewers of the blog will raise this immediately.
3. **Its two-track design gives up the most interesting question.** With CLARE and DMPEL on different data, CLARE vs DMPEL is confounded. Track C's within-track comparison (SeqFFT, ER, CLARE) is essentially what CLARE's own Table III already reports. The data-matched control is the only thing that makes the headline comparison valid, and v2 made it optional.
4. **Its gates are statistically weak.** A single-seed reproduction checked against "within 5 AUC points" can fail or pass by chance (DMPEL's own seed SD is about ±3 AUC). A 3-task mini-benchmark cannot provide effect sizes for 10-task forgetting, because forgetting accumulates with sequence length.
5. **It over-builds and under-protects the system of record.** 12–15 days for a general harness is too much for a portfolio project. Meanwhile the W&B free plan holds only **5 GB**, and v2 planned to upload artifacts and evaluation videos.

---

## 1. Is the experiment worth doing?

Yes, if it is scoped to the four contributions below. Each is tied to the phase that delivers it, so if the project stops early, it stops with something publishable.

| # | Contribution | Why it matters | Prior art it builds on | Delivered by |
|---|---|---|---|---|
| C1 | Reproduce DMPEL and CLARE on LIBERO-Long under one protocol | CLARE's Table III compares its own 100-rollout numbers with DMPEL and MLR numbers copied from their papers, under different protocols | CLARE (arXiv 2601.09512), DMPEL (arXiv 2506.05985) | Phase 2 |
| C2 | Measure evaluation-protocol bias in LIBERO continual learning | We already measured that DMPEL's best-epoch selection on its own test states adds 6–10 points of just-learned success (seeds 100/200/300, LIBERO-Goal) | LIBERO (arXiv 2306.03310) protocol; LIBERO-PRO, LIBERO-Plus critiques | Phases 0–2 (largely from data we have) |
| C3 | Memory- and compute-aligned comparison for manipulation: does "simple replay wins under budgets" transfer from vision to robots? | No robot continual-learning paper reports success against bytes (including model growth) and GPU-hours on one protocol | Zhou et al., "A Model or 603 Exemplars" (ICLR 2023); Prabhu et al., "Computationally Budgeted Continual Learning" (CVPR 2023, code `drimpossible/BudgetCL`); Liu et al. VLA forgetting study (arXiv 2603.03818) | Phases 3–5 |
| C4 | Robust retention: do retained skills survive perturbations, or only the memorised initial states? | Separates true retention from trajectory memorisation | LIBERO-Plus (CVPR 2026, `sylvestf/LIBERO-plus`), LIBERO-PRO (arXiv 2510.03827, MIT) | Phase 6 (evaluation only) |

**Out of scope, stated in the write-up:** generative or world-model replay (e.g. arXiv 2606.27374), RL-based continual fine-tuning (arXiv 2603.11653, 2602.10503), LiMoDE (no public code), real robots.

**What the blog can honestly claim if every phase succeeds:** on LIBERO-Long, under one protocol, here is how replay, coefficient replay and exemplar-free adapters trade bytes and GPU-hours for forgetting, and how much of the retained success survives perturbation. It cannot claim a new method or results beyond LIBERO.

---

## 2. Audit of PLAN.md (v2)

### Claims in v2 that hold (re-verified)

| v2 claim | Evidence I checked |
|---|---|
| CLARE trains on re-rendered 256×256, no-op-filtered, success-filtered data | `learnsyslab/clare`, `lerobot_lsy/src/lerobot/scripts/util/regenerate_libero_hdf5_dataset_as_hf.py` (`IMAGE_RESOLUTION = 256`, `is_noop`) |
| DMPEL's ER ignores `n_memories` and truncates to the first 20% | `third_party/DMPEL/libero/lifelong/algos/er.py`: `TruncatedSequenceDataset(dataset, len(dataset)//5)`; `TruncatedSequenceDataset` returns indices `0..buffer_size−1` |
| DMPEL selects and evaluates on the same initial states | `metric.py`: `indices = np.arange(i*env_num, (i+1)*env_num) % 50`; selection gap 6.0 / 9.0 / 10.0 points measured |
| CLARE's Table III copies DMPEL and MLR numbers | CLARE paper: "For DMPEL and MLR, we report the authors' results" |
| CLARE: 10k steps per task (paper) vs 20k (script) | Paper Table I vs `bash/clare/clare_libero_10.sh` (`STEPS=20000`) |
| `pi0_libero_90_pretrain` has no model card | HF page shows only "4B params" |
| W&B works from both envs | `wandb.init` succeeded with wandb 0.13.1 (`dmpel`) and `wandb.login` with 0.30.0 (`libero`), account `rahulbouri16` |

### Claims in v2 that need correction

| # | v2 statement | Problem | Correction |
|---|---|---|---|
| 1 | DMPEL's 0.06 GB storage is unverified | Now traced: the 0.06 GB is from DMPEL Table 7, **at the end of a 30-task sequence**, with no settings given. Our run with the repo defaults stores 404 MB for 10 tasks (about 40 MB per task, so about 1.2 GB at 30 tasks): a gap of about 20× | Ask the authors which setting produced Table 7 before publishing the discrepancy. Report both numbers in the meantime. |
| 2 | Shared protocol = final checkpoint, no selection | Removing selection penalises DMPEL, whose defaults were tuned with selection. Part of the 6–10 point gap is real overfitting across epochs, part is selection noise | **Held-out selection**: split the 50 initial states into 10 for selection and 40 for evaluation, and give every method the same selection budget (CLARE picks among its saved checkpoints). Report final-checkpoint and original-protocol scores alongside. |
| 3 | Data-matched CLARE base is optional (Phase 6) | Without it, the only cross-method question readers care about (exemplar-free CLARE vs coefficient-replay DMPEL) is confounded | Re-pretrain CLARE's base on our original HDF5 (`bash/clare/pretrain.sh`, 200k steps at batch 32) as its own gated phase (Phase 3). Fall back to two tracks only if it fails its gate. |
| 4 | Track C runs SeqFFT, ER and CLARE | SeqFFT on CLARE's base is already reported (NBT 74.7 on Long) and is not informative | Track C keeps CLARE and ER only. |
| 5 | Phase 2 gate: 1 seed within 5 AUC points | DMPEL's paper SD is about ±3 AUC, so one seed can miss by 5 by chance | 2 seeds per reference method. Pass if the 2-seed mean is within 2 reported SDs of the paper's mean **and** per-task curves show no systematic failure. |
| 6 | Phase 4: 3-task mini-benchmark with 3 seeds gives effect sizes | NBT on a 3-task stream averages over 2 tasks with at most 2 later tasks; forgetting grows with sequence length | Phase 4 becomes a 1-seed dress rehearsal (bugs, accounting, cost). Effect sizes come from Phase 2's full-length runs and published numbers. Cost falls from 60–80 to about 25 GPU-hours. |
| 7 | 12–15 development days, mostly a general harness | Over-built for the goal | Minimal harness (section 4): thin wrappers, a run spec, W&B logging, JSON results, metrics and accounting. About 5–7 days. Latent-ER (3–4 days) is deferred until C1–C3 do not depend on it. |
| 8 | Upload `result.pt`, curves and evaluation videos to W&B | W&B free plan: 5 GB of storage (weighted 30-day average) | Log metrics and small JSON tables only. No checkpoints or videos on W&B. Apply for the free W&B academic licence (200 GB) with the CMU address. Local `results/*.json` stays the source of truth. |
| 9 | Hierarchical bootstrap over seeds, tasks and episodes | Resampling tasks answers "would this generalise to other tasks", not "what happened on LIBERO-Long" | Primary intervals over seeds and episodes, with the 10 tasks fixed. Report task-resampled intervals as a secondary robustness view. |
| 10 | Many metrics and many cells, all reported as results | Multiple comparisons; no primary endpoint | **Primary endpoint:** AUC on LIBERO-Long under the shared protocol, plotted against total memory (bytes). Secondary: NBT, FWT, AUC against GPU-hours. Everything else is exploratory and labelled so. |
| 11 | Calendar not stated | Planning risk | Realistic estimate: 6–8 weeks to a draft blog (≈ 7 development days, 3–4 weeks of GPU time with gates, 1 week of analysis and writing). |

### Things v2 left out

- **Prior art** (Zhou 2023, Prabhu 2023, LIBERO-PRO, LIBERO-Plus) and the hypothesis it supplies: Prabhu found that under compute limits, simple uniform-sampling replay beats specialised continual-learning methods. That becomes H3 below.
- **Protection against data loss**, which you asked for explicitly:
  - every run writes local JSON at the end of each task, not only at the end of the run;
  - W&B runs in online mode, with `WANDB_MODE=offline` as the fallback and `wandb sync` afterwards;
  - the launcher refuses to start if W&B or disk checks fail.
- **Shared-machine risk.** AIMS is shared, and `/usr1` is at 95% use.
- **Licence status:** CLARE and LIBERO-Plus have no licence file. We can run them and report results, but should not redistribute modified copies of their code.

---

## 3. What exists, what to adapt, what to build

### Public and ready (install and verify only)

| Component | Source | Status on AIMS |
|---|---|---|
| LIBERO simulator and demonstrations (all 5 suites) | `Lifelong-Robot-Learning/LIBERO` (MIT), HF `yifengzhu-hf/LIBERO-datasets` | Installed and verified |
| DMPEL code, LIBERO-90 checkpoint, Seq / ER / EWC scripts | `HarryLui98/DMPEL` (MIT), HF `leiyuheng/DMPEL` | Installed; LIBERO-Goal replicated (FWT 0.70, NBT −0.005, AUC 0.81, 2 complete seeds) |
| CLARE code with ER baseline, LIBERO evaluation, LIBERO-90 checkpoint, LeRobot datasets, pretraining script | `learnsyslab/clare` (no licence file), HF `continuallearning` | Not installed |
| LIBERO-Plus perturbation suite | `sylvestf/LIBERO-plus` (no licence file) | Not installed |
| LIBERO-PRO | `Zijian007/LIBERO-PRO` (MIT) | Not installed; backup to LIBERO-Plus |
| π0 continual learning with replay buffers, EWC, PackNet | `Continual-VLAs/continual-openpi` | Not installed; stretch only |
| Budgeted continual-learning reference code (vision) | `drimpossible/BudgetCL` | Methodology reference only |
| W&B | account `rahulbouri16` | Verified |

### Minor adaptation (≤ 1 day each)

| Item | Effort (days) |
|---|---|
| DMPEL wrapper: W&B (`sync_tensorboard=True` plus success-matrix rows), JSON after every task | 0.5 |
| CLARE wrapper: W&B entity/project, map outputs to our JSON schema | 0.5 |
| Held-out selection protocol in DMPEL (`metric.py` indices, selection logic) and CLARE (checkpoint choice) | 1 |
| DMPEL coefficient-store budget: `moe_attn_recall_sample_ratio`, fp16 | 0.25 |
| CLARE ER budget: bytes → replay episodes (`replay_dataset.episodes`) | 0.5 |
| Metrics: final success, normalised NBT, bootstrap (extends our `compute_metrics.py`) | 1 |
| LIBERO-Plus evaluation of final checkpoints for both codebases (checking obs formats: 128 px for DMPEL, 256 px for CLARE) | 1–2 |

### Build from scratch

| Item | Effort (days) | Needed for |
|---|---|---|
| Minimal harness: YAML run spec → wrapper → per-GPU queue (we already have `scripts/run_dmpel.sh` and the nohup pattern), resume by run ID | 2 | All phases |
| Results JSON schema and aggregator | 1 | All phases |
| Accounting: bytes on disk per stored artifact, parameter growth, GPU and CPU hours, peak VRAM | 1 | C3 |
| Byte-budgeted random segment sampler for DMPEL's ER | 0.5–1 | C3 |
| Figure scripts | 1 | Write-up |
| Latent-ER on a frozen-encoder DMPEL variant (deferred) | 3–4 | Optional C3 extension |

Development before Phase 5: about **7 days**, plus 3–4 if latent-ER is added.

---

## 4. Infrastructure standard (Phase 0 builds this)

Directory: `/usr1/home/bbouri/CL-benchmark`

```
CL-benchmark/
├── .env                   # WANDB_API_KEY (never committed)
├── .gitignore             # .env, outputs/, wandb/, *.pth, caches
├── PLAN.md                # v2 record (PhD student)
├── PI_REVIEW.md           # this file (governs)
├── docs/
│   ├── PROTOCOL.md        # frozen evaluation protocol (versioned)
│   ├── PREREGISTRATION.md # hypotheses and analysis plan (tagged before Phase 5)
│   └── DECISIONS.md       # dated log of every protocol or scope change
├── configs/
│   ├── third_party.lock   # commit hashes: DMPEL b1abe28, openpi 215abfb, CLARE, LIBERO-Plus
│   └── runs/*.yaml        # one file per run: track, suite, method, budget, seed, protocol
├── cl_bench/              # wrappers, metrics, accounting, schema, wandb helper
├── scripts/               # queue, backfill, aggregate, figures
├── results/               # one JSON per run, git-tracked (the source of truth)
└── outputs/               # raw run outputs (not tracked; pruned after upload)
```

**W&B conventions**
- Project `cl-benchmark`, entity `rahulbouri16`.
- Run name `{track}-{suite}-{method}-{budget}-{protocol}-s{seed}`, for example `D-long-er-B2-heldout-s100`. Group = name without the seed. Tags: phase (`p0`…`p7`).
- Config: the run YAML, third-party commit hashes, harness commit, GPU model, hostname.
- Logged per task: training loss curve, selection-evaluation success, the new row of the success matrix, accounting snapshot.
- Summary at the end: FWT, NBT, AUC, final success, memory bytes, GPU-hours.
- Artifacts: only `result.json` (a few KB). Never checkpoints or videos.
- Quota: free plan 5 GB. Apply for the academic licence (200 GB) during Phase 0.

**Run rules**
- A run writes `results/{name}.json` after every task. A crash loses at most one task.
- W&B online by default. If the network or W&B fails, the wrapper switches to offline mode and records a `needs_sync` flag; `scripts/sync_offline.sh` uploads later.
- Before launch, the launcher checks: W&B login, free disk ≥ 150 GB, GPU idle, `third_party.lock` matches the checked-out commits.

---

## 5. Approved phase plan

Each phase answers one question, produces named deliverables, and ends at a gate. A phase's GPU budget is a cap; exceeding it requires a new approval.

### Phase 0: System of record (0 training GPU-hours, 4–5 days)

- **Question:** can we run, log and recover any experiment without losing data?
- **Work:**
  1. Repository, `.gitignore`, layout above.
  2. W&B helper and academic-licence application.
  3. JSON schema.
  4. Metrics library with unit tests.
  5. Accounting v1 (bytes on disk, parameter counts).
  6. `third_party.lock`.
  7. Write `docs/PROTOCOL.md` v0 (held-out selection, 10/40 split of initial states).
- **First real result (C2, zero GPU):** backfill the three finished DMPEL LIBERO-Goal seeds into W&B. Report the best-vs-last selection gap from their learning curves as the first entry in `results/`.
- **Gate:** unit tests reproduce our Goal metrics exactly (FWT 0.717 / 0.680, NBT −0.015 / 0.006, AUC 0.839 / 0.782 for seeds 100 / 200). The backfilled runs are visible in W&B. A deliberately killed run resumes from its last task.

### Phase 1: Every method through one pipe (≤ 15 GPU-hours, 3–4 days)

- **Question:** what does each method actually cost on our GPUs, and do all of them log correctly?
- **Work:**
  1. Install the CLARE env; resolve the `dit_flow_mt_libero_90_pretrain[_new]` naming.
  2. DMPEL wrapper and CLARE wrapper.
  3. 2-task LIBERO-Long micro-runs: DMPEL, DMPEL's ER, CLARE, CLARE's ER, at short schedules.
  4. Implement held-out selection in both codebases.
  5. Measure step time, evaluation time per 50 states, and peak VRAM. In particular, check whether DMPEL's full-fine-tune ER fits on one 4090 at batch 32.
  6. Open the GitHub issues: CLARE (10k vs 20k steps; checkpoint name); DMPEL (FWT counter; Table 7 storage setting).
- **Deliverable:** `docs/COST_MODEL.md` with measured per-step and per-evaluation costs, and re-estimated GPU-hours for Phases 2–5.
- **Gate:** all four methods produce valid `results/*.json` and W&B runs; the cost model replaces every estimate in this file.

### Phase 2: Trustworthy references (≈ 75 GPU-hours, revised at the Phase 1 gate)

- **Question (C1):** do DMPEL and CLARE reproduce on LIBERO-Long, and how much does the protocol change their numbers?
- **Work:**
  1. DMPEL, 2 seeds, original protocol. Targets from its paper: AUC 58 ± 3, FWT 55 ± 4, NBT 7 ± 1.
  2. CLARE, 2 seeds, original protocol. Targets: AUC 75.1 ± 1.3, FWT 75.0 ± 1.4, NBT 1.9 ± 0.4.
  3. Every run also saves the checkpoints and curves needed to re-score it under the held-out protocol without retraining.
- **Deliverables:** a reproduction report; first C2 table (original vs held-out vs final-checkpoint scores for both methods on the same runs).
- **Gate:** the 2-seed means fall within 2 reported SDs of each paper, and the per-task curves show no systematic failure. If either fails, debug before anything else is approved. A confirmed non-reproduction is reported, but only once understood.

### Phase 3: Remove the data confound (≈ 40 GPU-hours, estimate)

- **Question:** can CLARE's base model be retrained on the same data DMPEL uses, so the two can be compared directly?
- **Work:** re-pretrain CLARE's DiT base on our original LIBERO-90 HDF5 (`bash/clare/pretrain.sh`, 200k steps, batch 32). Evaluate its multitask success on a fixed 10-task subset of LIBERO-90, against the released checkpoint.
- **Deliverable:** a data-matched CLARE checkpoint and a short note on the data effect (256 px filtered vs 128 px raw).
- **Gate:** multitask success within 10 points of the released checkpoint on the subset. If it fails, keep two tracks, and every cross-track comparison is labelled as confounded.

### Phase 4: Budget knobs and dress rehearsal (dev + ≈ 25 GPU-hours)

- **Question:** do the budget settings and accounting work end to end, and are the bytes reported the bytes stored?
- **Work:**
  1. Byte-budgeted ER sampler (DMPEL), bytes-to-episodes for CLARE's ER.
  2. DMPEL coefficient-store budgets.
  3. Accounting checked against `du`.
  4. 3-task LIBERO-Long stream, 1 seed, every planned Phase 5 cell.
- **Deliverables:** passing accounting tests; final cell list and costs; `docs/PREREGISTRATION.md` committed and tagged, with hypotheses written from Phase 2 effect sizes and the literature (not from the 3-task stream):
  - **H1 (memory):** at budgets ≤ 88 MB, coefficient replay (DMPEL at a budget-matched setting) has higher AUC than raw ER on the same base.
  - **H2 (storing nothing):** data-matched CLARE matches raw ER at 885 MB on NBT (within 0.03), at higher parameter growth.
  - **H3 (compute, after Prabhu et al.):** at matched GPU-hours, uniform raw ER is not beaten by DMPEL or CLARE on AUC.
  - **H4 (protocol):** held-out selection changes at least one published ordering into a reversal or a tie.
- **Gate:** pre-registration tagged; Phase 5 cost re-approved.

### Phase 5: Core benchmark (cap 250 GPU-hours, final number set at the Phase 4 gate)

- **Question (C3):** how do the methods trade bytes and GPU-hours for forgetting on LIBERO-Long?
- **Cells (3 seeds each, held-out protocol, LIBERO-Long):**
  - raw ER at B1, B2 and B3 (DMPEL base);
  - DMPEL at default and budget-matched settings;
  - CLARE (data-matched if Phase 3 passed);
  - ER at B3 on CLARE's base;
  - sequential fine-tuning (lower bound);
  - multitask (upper bound, 1 seed).
- **Deliverable:** primary-endpoint figure (AUC against total bytes) with intervals; secondary figures (NBT, FWT, AUC against GPU-hours); hypothesis verdicts.

### Phase 6: Extensions (each approved separately)

- **C4, robust retention:** LIBERO-Plus evaluation of every Phase 5 final checkpoint, on a fixed subset of perturbation types (object layout, camera viewpoint, lighting). Evaluation only; cost set from Phase 1 timings.
- LIBERO-Goal for the core cells.
- Compute-capped CLARE (10k vs 20k steps, and matched to DMPEL's GPU-hours).
- Latent-ER on a frozen-encoder variant.
- π0 with replay via `continual-openpi`, after a 200-step timing pilot, capped at 120 GPU-hours.

### Phase 7: Analysis and write-up (≈ 1 week)

Figures, reproduction report, protocol-bias section, released JSONs, and the blog post, framed around C1–C4 and their prior art.

| Phase | Answers | GPU-hours | Cumulative | Approval |
|---|---|---|---|---|
| 0 | Can we log and recover everything? (plus C2 from existing data) | 0 | 0 | **Approved** |
| 1 | What does each method cost here? | ≤ 15 | ≤ 15 | **Approved** |
| 2 | C1: do the references reproduce? | ≈ 75 | ≈ 90 | At Phase 1 gate |
| 3 | Can the data confound be removed? | ≈ 40 | ≈ 130 | At Phase 2 gate |
| 4 | Do budgets and accounting work? | ≈ 25 | ≈ 155 | At Phase 3 gate |
| 5 | C3: bytes and GPU-hours vs forgetting | ≤ 250 | ≈ 405 | At Phase 4 gate |
| 6 | C4 and extensions | per item | – | Per item |

---

## 6. Risks I am accepting, and how they are bounded

| Risk | Bound |
|---|---|
| A reference method fails to reproduce | Phase 2 gate; spend ≤ 75 GPU-hours before knowing |
| Data-matched CLARE underperforms because of 128 px raw data | Phase 3 gate; fall back to labelled two-track results |
| LIBERO-Plus does not plug into one of the codebases | C4 is a separate Phase 6 item; LIBERO-PRO is the backup |
| W&B quota or outage | Local JSON is the source of truth; offline mode plus sync; academic licence |
| Shared disk fills | Launcher refuses below 150 GB free; prune `outputs/` after upload |
| Result is "no differences" | Pre-registered hypotheses; a tie under a fair protocol is a valid finding for C3 |

---

## 7. Sources

Papers
- Liu et al., LIBERO, arXiv 2306.03310
- Lei et al., DMPEL, arXiv 2506.05985 (Table 7: 0.06 GB at K = 30)
- Römer et al., CLARE, arXiv 2601.09512 (Table III)
- Yu et al., MLR+IFA, arXiv 2603.10929
- Liu et al., Pretrained VLAs are surprisingly resistant to forgetting, arXiv 2603.03818
- Zhou et al., A Model or 603 Exemplars, ICLR 2023, arXiv 2205.13218
- Prabhu et al., Computationally Budgeted Continual Learning: What Does Matter?, CVPR 2023, arXiv 2303.11165
- LIBERO-PRO, arXiv 2510.03827
- LIBERO-Plus, CVPR 2026
- Out of scope, cited for context: LiMoDE arXiv 2606.26183; World Action Models with generative replay arXiv 2606.27374; RL continual fine-tuning arXiv 2603.11653, 2602.10503

Code and data
- https://github.com/HarryLui98/DMPEL (MIT) and https://huggingface.co/leiyuheng/DMPEL
- https://github.com/learnsyslab/clare (no licence file) and https://huggingface.co/continuallearning
- https://github.com/sylvestf/LIBERO-plus (no licence file)
- https://github.com/Zijian007/LIBERO-PRO (MIT)
- https://github.com/Continual-VLAs/continual-openpi
- https://github.com/drimpossible/BudgetCL
- https://github.com/Lifelong-Robot-Learning/LIBERO (MIT) and https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
- W&B pricing (free plan 5 GB, academic 200 GB): https://wandb.ai/pricing

Measurements on AIMS used here
- DMPEL LIBERO-Goal runs: `/usr1/home/bbouri/continual-learning/outputs/dmpel/libero_goal/` and `logs/dmpel/`
- Selection gap: from `task*_auc.log`, seeds 100 / 200 / 300 (6.0 / 9.0 / 10.0 points; max 25)
- DMPEL coefficient store: 404 MB for 10 tasks (`*_moe_attn_recall_*.pth`), expert library 28.6 MB
