# CL-Benchmark handoff (updated 3 Oct 2026, replaces the 2 Oct version)

For a new Claude Code session: "read /usr1/home/bbouri/CL-benchmark/docs/HANDOFF.md, then docs/REPO_MAP.md, then continue".
A Mac copy is at `~/Developer/CL-benchmark-HANDOFF.md` (may lag; AIMS is the source of truth).

## Ground rules (from the user)
- Everything runs on the AIMS GPU machine (`ssh aims`; riddle, user bbouri, 2× RTX 4090, **shared with other users**). Nothing runs locally.
- Explain each stage and get approval before launching runs. Do not leave shells or agents polling runs, except when the user explicitly asks to monitor a specific run to completion (as for ER attempt 7). Queue jobs with nohup, check once, report.
- W&B is the official logger: entity `rahulbouri16`, project `cl-benchmark`. The key is in `/usr1/home/bbouri/CL-benchmark/.env`; never print or commit it. Free plan 5 GB: log metrics and small JSON only.
- Do not post more GitHub issues (stopped after DMPEL #6, #7, #8). Never post or publish anything external without asking.
- Be critical, not sycophantic; verify claims against code and papers. Edit docs on AIMS only (see below).

## Read first
`docs/REPO_MAP.md` (every file, what it does, how a run flows), then `PI_REVIEW.md` (the "Status and amendments" section governs), `docs/PROTOCOL.md`, `docs/DECISIONS.md` (dated log of every change), `docs/COST_MODEL.md`.

## Status
**Phase 0: complete, gate passed.**
**Phase 1: 5 of 6 micro-runs complete. DMPEL-codebase ER is the remaining one.**
- Complete: D-long2-dmpel (original, held-out), C-long2-clare (original, held-out), C-long2-er.
- **D-long2-er-micro-original-s100:** attempt 7 started 3 Oct 15:41 UTC (W&B run `354e500aae5a`; output `outputs/D-long2-er-micro-original-s100/a7/`; queue log `logs/queue_p1_er_ddp_d.log`).
  - Setup: 2 GPUs (torchrun DDP), micro-batch 4 (+4 replay) × 4 accumulation = 32 + 32 per optimizer step, `eval.num_procs=10`.
  - History: attempts 0–3 ran out of GPU memory; attempt 5 hit an evaluation OOM and then the driver incident; attempt 6 (10 workers) passed both task-0 evaluations and then crashed on a DDP checkpoint race in DMPEL's `base.py`, now patched (`configs/patches/dmpel_ddp_ckpt_race.patch`).
  - Watch for: end of task 0 (where attempt 6 died), then task 1 evaluations (earlier OOM point), and GPU 0 memory staying flat across evaluations (`gpu_mem.csv`).
  - Check with: `grep -E "^\[clb\]" logs/queue_p1_er_ddp_d.log`, then `python scripts/report_phase1.py` (libero env) to refresh the cost model (it preserves the hand-written projection).
- If it fails with another memory error: the user-approved fallback is half the published effective batch (`configs/runs/p1/D-long2-er-microhalf-original-s100.yaml`), which will not fix an evaluation-side OOM; reduce `eval.num_procs` further first and discuss with the user.
- To close Phase 1: ER valid run; CLARE peak VRAM (the launcher's GPU sampler records it on the next CLARE run); ER row in the cost model; user gate sign-off. A full-schedule projection is already written in `docs/COST_MODEL.md`.

## Decisions made by the user
- CLARE reproduction uses the paper's settings (10k steps, batch 32, constant LR 1e-4, γ = 2.5, 2k discriminator steps), not the script's 20k / γ = 1.0.
- DMPEL ER: 2 GPUs with gradient accumulation (32 + 32 per step); fallback is half the published batch.
- Issue posting stopped after #6–#8.

## Key facts to carry (all in DECISIONS.md)
- **Completion is not correctness.** CLARE runs "completed" with 0% success (broken rendering under MuJoCo 2.3.7, plus its success counter not counting under Gymnasium ≥ 1.0). Both fixed. DMPEL's logged FWT is inflated (>1). Validate against papers in Phase 2.
- **Track C (CLARE):** MuJoCo 3.3.0, 256 px regenerated data, `gym-libero` dependency, transformers 4.56.2, dataset revision `v2.1`, one CLARE job at a time (host RAM).
- **Track D (DMPEL):** MuJoCo 2.3.7, 128 px original HDF5. Our patches: data-loader shim, protocol + task cap, gradient accumulation, DDP checkpoint fix.
- **FWT is not comparable across tracks** (CLARE has no selection curve); use AUC, NBT, final success for cross-track comparisons.
- **Measured costs** (see COST_MODEL): DMPEL ≈ 15 GPU-h per 10-task LIBERO-Long run; CLARE ≈ 18 (original) or 11.5 (held-out), evaluation-dominated. Phase 5's 250 GPU-h cap is likely too small once ER cells cost double.
- **GPU incident (2 Oct 04:35 UTC):** two jobs creating many GPU contexts at once wedged the NVIDIA driver; an admin restart fixed it. Rules: no probes that open many GPU contexts; 10 evaluation workers; do not launch within minutes of another user's GPU job; if `nvidia-smi` hangs, stop and tell the user (do not retry; each call adds a stuck process).
- AIMS is shared (`sradhak2` runs GPU jobs). `/usr1` is about 96% full (291 GB free); the launcher refuses below 150 GB.

## Housekeeping
- The harness git repo exists only on AIMS (no remote). Offer a private remote to the user; do not create one unasked.
- `logs/` is ignored by git from 3 Oct.
- The W&B test project `cl-benchmark-keytest` can be deleted by the user.

## Open decisions for the user
1. Phase 1 gate sign-off (after ER attempt 7).
2. Whether benchmark runs fix the AMP gradient clipping in ER/EWC/PackNet baselines (Phase 4).
3. Phase 5 budget and cell list, given ER's cost.
4. Whether to add a private remote for the repository.

## Next steps
1. Check ER attempt 7; fix or report.
2. Refresh the cost model and bring Phase 1 to the user for sign-off.
3. Then pause for the full review and planning discussion the user asked for, before any Phase 2 work.
