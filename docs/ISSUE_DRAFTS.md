# GitHub issue drafts (for the repository owner to post; not posted automatically)

## HarryLui98/DMPEL

**Title:** Logged forward-transfer (S_fwd) can exceed 1: `cumulated_counter` only increments on improvement

In `libero/lifelong/algos/dmpel.py` (`learn_one_task`), `cumulated_counter += 1.0` sits inside the `if prev_success_rate < success_rate:` branch, so `successes.sum() / cumulated_counter` divides by the number of *improvements* rather than the number of evaluations (LIBERO's `algos/base.py` increments it at every evaluation). On LIBERO-Goal (seed 100) the logged S_fwd averages above 1 for some tasks, while recomputing FWT from the saved `task{k}_auc.log` curves gives values consistent with the paper (FWT 0.70 across 3 seeds vs 0.68 reported). Could you confirm the paper's FWT was computed from the curves rather than from `result.pt`'s `S_fwd`?

**Title:** Storage setting behind Table 7 (0.06 GB for K = 30)?

With the released defaults (`moe_attn_recall_sample_ratio: 1.0`, float32), the coefficient-replay files (`task*_moe_attn_recall_query.pth` + `*_attn.pth`) total about 404 MB after 10 LIBERO-Goal tasks (about 40 MB per task), versus 0.06 GB reported for a 30-task sequence. Which sample ratio and dtype produced Table 7?

**Title:** `lifelong.n_memories` is ignored by ER

`algos/er.py` builds the buffer with `TruncatedSequenceDataset(dataset, len(dataset)//5)` (the first 20% of each task's sequences), so `n_memories: 1000` in `configs/lifelong/er.yaml` has no effect. Was the reported ER baseline run with this 20% truncation?

## learnsyslab/clare

**Title:** Steps per task: paper 10k vs scripts 20k; checkpoint name `_new`; `gym_libero` dependency

1. Table I gives 10,000 training steps per simulation task, but `bash/clare/clare_libero_10.sh` uses `STEPS=20000`. Which produced Table III?
2. The README and scripts reference `dit_flow_mt_libero_90_pretrain_new`, but only `continuallearning/dit_flow_mt_libero_90_pretrain` is on Hugging Face. Are they the same checkpoint?
3. The LIBERO env factory imports `gym_libero`, which is not listed in the repo. Is `ZhangYi1999/gym-libero` the intended package, and at which commit/MuJoCo version were the results produced?
4. One Git LFS object (`lerobot_lsy/tests/artifacts/cameras/image_128x128.png`) returns 404, which makes a plain `git clone` fail at checkout.

**Addendum (CLARE):** with gymnasium ≥ 1.0 (including the 1.2.1 in `pixi.lock`), `eval_peft.py` counts no successes, because success is read only from `info["final_info"]`. Reading `info["is_success"]` (set by `gym_libero`) restores it. Which gymnasium version produced the paper's numbers?

**Addendum (DMPEL):** `algos/er.py` `observe` clips gradients without `self.scaler.unscale_(self.optimizer)` (which `algos/base.py` does), so the clip threshold applies to fp16-scaled gradients. Was the reported ER baseline trained with this behaviour?
