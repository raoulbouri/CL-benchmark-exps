# Benchmark portfolio project vs the course roadmap: how to shape the proposal

Written 3 Oct 2026 for a proposal due in about 10 days, to become a course project and, if it works, a workshop paper.
Sources: our own runs and docs (`CL-benchmark/docs/DECISIONS.md`, `PI_REVIEW.md`, `COST_MODEL.md`), the roadmap (`course_and_paper_roadmap.md`), and the papers named in section 9. Evidence tags used throughout: **[measured]** by us in this project, **[paper]** as reported by the authors, **[code]** verified by reading the released code, **[estimate]** my own arithmetic or projection, **[unverified]** not checked.

---

## 1. Bottom line

1. **Do not finish the benchmark before the proposal.** It needs about 400+ GPU-hours (Phases 2–5), Phase 2 has not started, and our machines have been unreliable. It also answers a different question from the roadmap's.
2. **Do not adopt the roadmap exactly as written.** Its structure is good, but it has fixable problems: too little forgetting headroom, a buffer premise that is wrong at the released defaults, missing 2026 prior art, a biased evaluation protocol, and low statistical power.
3. **Write the proposal around the roadmap's research question, with an analysis-first framing, and use the benchmark work as the methodological backbone and the preliminary evidence.** The full benchmark becomes a stretch goal or future work.
4. **Run one instrumented DMPEL Goal pilot** on `xulab` (about 9–10 GPU-hours) so that the proposal includes a first look at the phase hypothesis, not only a plan.

What would change this recommendation is in section 8.

---

## 2. The two projects at a glance

| | **A. Benchmark portfolio project** | **B. Course roadmap (phase-aware DMPEL)** |
|---|---|---|
| Central question | Under equal memory (bytes) and compute (GPU-hours), which way of remembering old tasks works best for long-horizon manipulation: raw replay, latent replay, coefficient replay (DMPEL), or storing nothing (CLARE)? | Does phase structure help DMPEL's router memory and expert allocation, and in which regime? |
| Type of contribution | Measurement and reproducibility: one protocol, two budget axes, released result matrices | Method plus analysis: phase-floored coefficient buffer, phase-novelty gate, joint replay, and analyses beyond accuracy |
| Headline you could claim | "Under one protocol, here is how the methods trade bytes and GPU-hours for forgetting" | "Phase structure predicts expert reuse" (if A1 is strong), or "Do continual routers need phase information?" (if weak) |
| Novelty | Moderate. Memory-aligned and compute-budgeted comparison exist in vision (Zhou 2023, Prabhu 2023); transfer to manipulation is new | Low to moderate as a method (a composition of PHASER's floor, CLARE's gate, and DMPEL); the analysis can carry it |
| Compute | About 400+ GPU-hours for the full plan [estimate] | About 435 GPU-hours for the full plan, about 410 after the reproduction already done [estimate]; the offline analyses are nearly free |
| Time to a first finding | Weeks (needs reference reproductions first) | Days (offline analyses A1, A2, A4 gate everything) |
| Main risk | Compute and infrastructure; ties between methods; crowded area | Low forgetting headroom; low statistical power; prior-art overlap; instrumentation gaps |
| Needs results to be believable? | Yes: a benchmark with no completed runs is not a benchmark | A proposal can stand on a plan plus a pilot |
| Fit to a 10-week course | Heavy engineering; hard to finish | Matches the course structure (reproduce, improve, analyse) |
| Fit to a workshop paper | Reproducibility or benchmark workshops | Continual-learning or modularity workshops |

---

## 3. Project A: the benchmark portfolio project

### What it is
A budget-matched continual-learning benchmark on LIBERO-Long (primary) and LIBERO-Goal. It compares sequential fine-tuning, raw experience replay (at byte budgets), latent replay, DMPEL's coefficient replay, and CLARE (exemplar-free), with success reported against memory in bytes and against GPU-hours. Governing plan: `PI_REVIEW.md`.

### Where it stands (as of 3 Oct 2026)
| Item | Status |
|---|---|
| Phase 0: system of record (harness, metrics with tests, W&B logging, accounting) | Complete, gate passed |
| Phase 1: every method through one pipeline | 5 of 6 micro-runs complete; DMPEL-codebase ER running on `xulab` as of this writing |
| Phases 2–5 (reference reproductions, data-matched CLARE, budget knobs, core benchmark) | Not started, not approved |
| Machines | AIMS unreachable since 3 Oct after two incidents (a GPU driver wedge on 2 Oct, then a crash during the ER run); `xulab` (2 × RTX 5090) is set up under `/mnt/data/users/bbouri` |

### What the work has already produced (useful as preliminary results)
- **DMPEL reproduced on LIBERO-Goal, 3 seeds, full schedule [measured]:** FWT 0.717 / 0.680 / 0.716 (mean 0.704), NBT −0.015 / 0.006 / 0.009 (mean 0.000), AUC 0.839 / 0.782 / 0.798 (mean 0.806), final success 0.845 / 0.825 / 0.810 (mean 0.827). The paper reports 0.68 ± 0.03, 0.00 ± 0.01, 0.78 ± 0.02 [paper]. Within spread.
- **The protocol inflates success [measured]:** DMPEL picks the best epoch on the same 20 initial states it reports. The best epoch beats the last by 9.0, 6.0 and 10.0 points on average over the three seeds (up to 25 points on single tasks).
- **DMPEL's logged forward transfer is wrong [measured, code]:** a counter increments only on improvement, so the logged value reaches 2.52. FWT must be recomputed from the learning curves.
- **DMPEL's memory depends on a setting [measured, code, paper]:** the buffer is a uniform random sample of frames at ratio ρ; the released default is ρ = 1.0 (every frame stored: 404 MB for 10 Goal tasks, plus 28.6 MB of experts). The paper's Fig. 5b says ρ = 5% suffices. The paper's 0.06 GB at 30 tasks is consistent with ρ = 5% (404 MB × 3 × 5% ≈ 0.06 GB) [estimate, not confirmed by the authors].
- **Costs [measured, projected]:** DMPEL on Goal about 9.1 GPU-hours per run; projected about 15 on LIBERO-Long. CLARE about 18 GPU-hours (original protocol) or 11.5 (held-out), dominated by evaluation. Details in `COST_MODEL.md`.
- **Other reproduction findings:** CLARE's LIBERO environment is undeclared and needs specific package pins; CLARE runs "completed" with 0% success because of broken rendering and a success counter that does not count under current Gymnasium; ER's buffer size setting is ignored in DMPEL's code; ER and three other baselines clip mixed-precision-scaled gradients. Three DMPEL issues were posted upstream (#6–#8).

### Strengths
- Strong evidence of execution and rigour; the findings above are real and already measured.
- The harness (metrics, accounting, protocol, logging) is reusable by any project on this codebase.
- Clear, defensible comparison design: two tracks, held-out evaluation, bytes on one axis.

### Weaknesses and risks
- **Compute and reliability:** about 400+ GPU-hours with two crashes so far. A third incident would cost more time.
- **Novelty is borrowed:** the budget-aware idea comes from vision. It needs to be framed as a transfer to manipulation.
- **Possible ties:** if methods do not separate under a fair protocol, the result is "no difference", which is valid but a harder paper.
- **Crowded space:** CLARE, MLR+IFA, LiMoDE and others all claim to be best under their own protocols.
- **Cross-track confounds:** CLARE and DMPEL use different data, simulator versions and policies; separating them needs a 200k-step re-pretraining.
- **Engineering-heavy:** a lot of the effort goes into infrastructure rather than into understanding the method.

### What it would teach you
How the methods run end to end, where evaluation protocols mislead, and what things cost. It would teach you less about **why** a method works, which is what a method proposal needs.

---

## 4. Project B: the course roadmap

### What it is
Reproduce DMPEL, add phase-aware improvements, and analyse why they do or do not work.

| Part | Content |
|---|---|
| Reproduce | R0–R4: DMPEL on Goal and Long, with pruning, 3 seeds |
| Improve | I1 phase-floored coefficient buffer; I2 phase-novelty gate with router-only branch; I3 joint coefficient replay; I4 all combined |
| Analyse | A1–A9: predictive power of phase novelty, buffer starvation, router drift by phase length, whether DMPEL already routes by phase, size–performance trade-off, cost of joint replay, weight-synthesis form, robustness to phase labels, task order |

Approximately 42 runs, 10-week schedule, offline analyses A1, A2, A4 in week 4 as go/no-go gates.

### Strengths
- **Offline analyses first with explicit go/no-go decisions** avoids building a method that cannot work.
- **Pre-registered intuitions**, paired seeds, a stated minimum detectable effect, and a commitment to report negative results.
- **Clean course-to-paper mapping** with analysis as the headline.
- **Released checkpoint**, so no pretraining cost.
- The reproduction targets match the papers (Goal 0.68 / 0.00 / 0.78; Long 55 / 7 / 58) [paper]. R1 is already done on Goal.

### Weaknesses (in order of seriousness)
1. **Forgetting headroom is smaller than the detectable effect.** DMPEL's NBT is 0.00 on Goal and about 7 points on Long [paper]. The roadmap says differences under about 0.06–0.07 are undetectable (my calculation: significance threshold 0.068 at 3 seeds and SD 0.03, and about 0.09 for 80% power [estimate]). So I1 and I3, which protect the router, cannot show a measurable gain in forgetting, and the plan runs I2/I3/I4 mostly on Goal, where there is nothing to fix.
2. **The buffer premise needs the right regime.** The plan's E[b_p] = ρ·n_p is correct [code]. But the released default is ρ = 1.0 (every frame), and DMPEL's own Fig. 5b shows 5% is enough [paper]. For a typical Goal task (about 6,400 frames), a phase covering 15% of the trajectory still gets about 48 entries at ρ = 5%, about 10 at 1%, and about 2 at 0.2% [estimate]. Starvation only appears at ρ ≲ 1%, so ρ must be an explicit experimental axis.
3. **Missing and overlapping prior art** (section 9). I1 overlaps Memory Anchors; I2 overlaps CLARE and LiMoDE; the whole method is a composition of PHASER, CLARE and DMPEL.
4. **"Run DMPEL exactly as released" is not possible.** It needs a data-loader fix, CLIP loaded locally, and newer PyTorch on 5090-class GPUs. DMPEL's logged FWT is also inflated.
5. **The evaluation protocol inflates numbers** (section 3), and arms that change training dynamics (I3) get biased differently.
6. **A1 has low power.** 10 tasks per suite (seeds repeat the same tasks): a Spearman correlation needs |ρ| ≳ 0.65 to be significant [estimate]. DMPEL's 30-task sequence would lower that to about 0.36, if the code supports it [unverified].
7. **The τ sweep tunes on the evaluation sequence.** Choose τ on separate tasks or report the full curve.
8. **Instrumentation gap.** The released code reshuffles stored rows at ρ = 1.0 without saving the permutation [code], so **phase labels cannot be recovered from the existing AIMS runs**. A rerun with saved indices is required.

### What it would teach you
The mechanism of DMPEL's router, expert allocation and coefficient memory, and how phase structure relates to them: the understanding a method proposal needs.

---

## 5. Side by side on what matters for the next 10 days

| Criterion | A. Benchmark | B. Roadmap (revised, section 6) | Better for the proposal |
|---|---|---|---|
| Feasible in 10 days | No (needs weeks of compute) | Yes (plan plus one pilot) | B |
| Strength of preliminary evidence | Already high (section 3) | Can reuse A's evidence and add a pilot | B (inherits A) |
| Dependence on unreliable compute | High | Low to moderate (one 9–10 GPU-hour run) | B |
| Novelty | Moderate, borrowed | Low to moderate; analysis can carry it | Tie; both need honest positioning |
| Course fit (Track 1: reproduce, improve, analyse) | Weak | Strong | B |
| Risk of a null result | High (ties) | Moderate; null is publishable if well powered | B |
| Understanding gained about the method | Low | High | B |
| Portfolio and engineering value | High | Moderate | A |
| Reusability of existing work | Is the existing work | Reuses it | Tie |

---

## 6. Recommended path: a revised roadmap that inherits the benchmark's rigour

**Working title:** *Does phase structure help continual expert routing? A budget-matched analysis of DMPEL.*

### What changes relative to the roadmap
| Roadmap | Revised |
|---|---|
| Run DMPEL exactly as released | Run as released **plus documented deviations**; recompute FWT from learning curves |
| Evaluate with DMPEL's protocol | **Held-out protocol** (select on 10 initial states, evaluate on 40) for all improvement arms; original protocol only for the reproduction |
| Memory implied, ρ fixed | **ρ and total memory in bytes as explicit axes**, including ρ ≲ 1% |
| I1/I3 judged on success at default settings | Judged on **router drift (A3)** and on **small-buffer regimes**, where forgetting actually appears |
| I2/I3/I4 mostly on Goal | Goal for size–performance (I2); small-buffer and long-stream conditions for I1/I3 |
| A1 on 10 tasks | A1 as a **pilot**; extend to a 30-task sequence if feasible |
| τ chosen on the test sequence | τ chosen on a separate subset, or the full curve reported |
| Related work: CLARE, IsCiL, LOTUS, PHASER | Add **Memory Anchors, LiMoDE, MLR+IFA, SCE** and state the differences |
| Cut order: A9, A7, Long for I4 | Same, plus drop R3/R4 if compute is short |

### Hypotheses to pre-register (all falsifiable)
- **H1 (phase novelty predicts expert use):** pre-training phase novelty of task k correlates with how much the router uses expert k. Refuted if |ρ_s| is small with a tight interval.
- **H2 (starvation):** at ρ ≲ 1%, short phases are under-represented in the coefficient buffer, and router drift is higher on them. Refuted if drift is uniform across phase lengths.
- **H3 (floor helps in the starved regime only):** an equal-quota floor reduces router drift at ρ ≲ 1% and not at ρ ≥ 5%.
- **H4 (gate saves parameters):** a phase-novelty gate matches DMPEL + pruning on success with fewer experts. Refuted if pruning matches it at every size.

### Experiment plan with gates

| Stage | Work | Cost | Decision it informs |
|---|---|---|---|
| 0. Pilot (days 1–6) | One **instrumented** DMPEL Goal run on `xulab`: save buffer indices, per-task expert statistics, router checkpoints. Use PHASER's Goal phase table. Offline A1 (phase novelty vs expert use), A2 (entries per phase), A4 (does the router route by phase?) | about 9–10 GPU-hours, plus a few CPU-hours | If A1 is near zero, I2 is dropped and A1 becomes the headline; if A2 shows no starvation at the ρ tested, expect I1 to be null |
| 1. Reproduction notes (days 3–8, no new compute) | Write up Goal reproduction, protocol bias, FWT bug, ρ analysis | none | Preliminary results section |
| 2. Course weeks 2–3 | R2 (DMPEL on Long, 3 seeds) with the held-out protocol and instrumentation | about 45 GPU-hours [estimate] | Whether Long has usable headroom |
| 3. Course weeks 4–6 | I1 at ρ ∈ {5%, 1%, 0.2%}, A3 router drift | about 30–60 GPU-hours | H2, H3 |
| 4. Course weeks 7–8 | I2 and τ sweep (A5) on Goal | about 50 GPU-hours | H4 |
| 5. Stretch | Full benchmark (Project A) phases 2–5; I4; A6–A9 | remainder | Paper depth |

Rigour carried over from the benchmark: paired seeds, bootstrap intervals, bytes on every memory axis, results JSON released.

---

## 7. A 10-day plan to the proposal

| Days | Work | Output |
|---|---|---|
| 1 | Decide the question and scope; confirm the proposal's format, page limit and rubric (section 10) | One-page scope |
| 1–2 | Read the four missing prior-art papers (abstract and method level); fix positioning | Related-work draft |
| 1–4 | Launch and finish the instrumented Goal pilot on `xulab` (needs the guard-protected setup already in place) | Run with buffer indices and expert statistics |
| 3–6 | Offline pilot: A1, A2, A4 on that run; phase labels from PHASER Table 6 | Three figures or tables, flagged as pilot-level |
| 4–8 | Write: motivation, related work, hypotheses, method, experiment plan, preliminary results, risks | Full draft |
| 9 | Cut to the page limit; check every number against `DECISIONS.md` | Near-final |
| 10 | Polish and submit | Proposal |

If `xulab` is unavailable, replace the pilot by the reproduction evidence and present A1/A2/A4 as the first planned step.

---

## 8. What would change the recommendation
- **If the proposal must contain results of a completed study,** do a small benchmark instead (Goal only, original vs held-out protocol, DMPEL vs ER at two byte budgets), which fits in days.
- **If `xulab` is not usable within 2–3 days,** the pilot is at risk; lean on the reproduction and present the analysis as the first planned step.
- **If your advisor or course wants a measurement study or a systems contribution,** the benchmark is the better fit.
- **If the pilot (A1/A2) is clearly null,** the proposal's headline becomes "does DMPEL need phase information?" with the null as the finding, which needs adequate power to be credible.
- **If a team forms,** split: reproduction and evaluation (the harness), phase annotation and offline analyses, implementation of I1–I3 (as the roadmap suggests).

---

## 9. Evidence appendix

### Verified in this project
| Claim | Evidence |
|---|---|
| DMPEL Goal reproduction numbers, selection gap, FWT bug | Our runs on AIMS; results in W&B project `cl-benchmark` (`rahulbouri16`) and `CL-benchmark/results/` on AIMS |
| Buffer is a uniform random frame sample at ratio ρ; default ρ = 1.0; ratio is applied (`int(ratio × total)` with `torch.randperm`); permutation not saved | DMPEL `libero/lifelong/algos/dmpel.py` and `configs/lifelong/dmpel.yaml` at commit b1abe28 |
| Paper's Fig. 5b: ρ = 5% retains the benefit; Table 7: 0.06 GB at 30 tasks | DMPEL arXiv 2506.05985, HTML version |
| Roadmap's R1/R2 targets | DMPEL Table 4 (Goal); Long numbers as reported in CLARE's Table III |
| Roadmap's arXiv IDs for TAIL, LOTUS, LoRAHub, Chaudhry, Agarwal, Díaz-Rodríguez | Recognised as correct; not each re-opened |

### Prior art to position against (read at abstract level, some via summaries; read fully before citing)
CLARE (arXiv 2601.09512); MLR+IFA (2603.10929); LiMoDE (2606.26183); SCE (2606.15685); Stellar VLA (2511.18085); OrthoSkillVLA (2608.19589); Memory Anchors (2608.26545); CPC (2601.22475); pretrained VLAs resist forgetting (2603.03818); PHASER (2606.03598); Zhou et al. (2205.13218); Prabhu et al. (2303.11165); LIBERO-PRO (2510.03827).

### Venue timing
ICLR 2027 workshops are on 29–30 April 2027 in San Francisco; the suggested workshop paper deadline is 1 Feb 2027, with notification by 26 Feb. Which workshops exist is announced about 29 Nov 2026 (source: iclr.cc/Conferences/2027/CallForWorkshops). ICRA, RSS and CoRL workshop timing is **[unverified]**.

### Not verified
The roadmap's references into PHASER's appendices F, H and J and its table numbers beyond Table 6 and 7; the contents of the existing `expert_stats` files on AIMS (A1's response variable); whether the released code supports DMPEL's 30-task sequence; and CLARE's own FWT convention.

---

## 10. Questions that would sharpen the proposal
1. What are the proposal's page limit, rubric and required sections?
2. Must the proposal contain results, or is a plan with preliminary evidence enough?
3. Is this individual or a team (and how many people)?
4. Is the target a course project only at first, or a specific workshop?
5. Is `xulab` available to you for the next two weeks, and is AIMS expected back?
