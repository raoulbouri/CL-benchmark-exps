# Evaluation protocol (v0, 1 Oct 2026)

Governed by `PI_REVIEW.md`. Changes require an entry in `DECISIONS.md` and a version bump here.

## Protocols

| Name | Selection evaluations (during training) | End-of-task evaluation (success matrix) | Used for |
|---|---|---|---|
| `original` | DMPEL: every 2 epochs on the current task, initial states 0–19, keep best. CLARE: none, final checkpoint. | DMPEL: initial states 0–19 (20 episodes). CLARE: 100 episodes cycling all 50 states. | Reproducing each paper (Phase 2) |
| `heldout` | Initial states 40–49 (10 episodes) per selection point; keep best | Initial states 0–39 (40 episodes), never used for selection | All benchmark comparisons (Phases 4–5) |

LIBERO provides 50 fixed initial states per task (`init_files/<suite>/<task>.pruned_init`). In the DMPEL codebase, `heldout` is set with `eval.protocol=heldout`; the selection and test state lists are `eval.select_state_ids` and `eval.test_state_ids` (`libero/configs/eval/default.yaml`).

## Metrics

From the success matrix `c[t][k]` (success on task k after training through task t) and each task's selection curve:

- FWT: mean over tasks of the selection curve held at its peak (LIBERO definition). Never DMPEL's logged `S_fwd`.
- NBT: mean over k < K of mean over t > k of `c[k][k] − c[t][k]`. Normalised NBT divides each term by `c[k][k]`.
- AUC: mean over k of `(FWT_k + Σ_{t>k} c[t][k]) / (1 + K − 1 − k)`.
- Final success: mean of the last row. Mean just-learned success: mean of the diagonal.
- Also recorded: best-minus-last selection success per task (the selection-bias measure, C2).

Primary endpoint (PI_REVIEW.md): AUC on LIBERO-Long under `heldout`, against total memory in bytes.

## Intervals

- Per run: parametric bootstrap, each cell resampled as Binomial(episodes, p) (per-episode outcomes are not stored by DMPEL).
- Across seeds: bootstrap over seeds, tasks fixed. Task-resampled intervals are a secondary view.
