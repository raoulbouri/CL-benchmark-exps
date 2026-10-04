# DMPEL GitHub issues: drafts for approval (NOT POSTED)

Target repository: https://github.com/HarryLui98/DMPEL (commit `b1abe28`). Each issue is verified against the
original commit with `git show HEAD:<file>`; line numbers refer to that commit.
Author line pending confirmation (spelling of the name; whether to show the email publicly).
Not included: the storage question (Table 7 vs released defaults), which is a question rather than a bug.

---

## Issue 1

**Title:** Forward transfer (`S_fwd` in `result.pt`) can exceed 1: `cumulated_counter` only counts improvements

Hi, and thanks for releasing DMPEL. While reproducing the LIBERO-Goal results (commit `b1abe28`, default `exp_scripts/lifelong_scripts/dmpel.sh` settings) I noticed that the forward-transfer value returned by `learn_one_task` and stored as `S_fwd` in `result.pt` can be larger than 1.

In `libero/lifelong/algos/dmpel.py` (L360–365) and `libero/lifelong/algos/base.py` (L297–302), `cumulated_counter += 1.0` sits inside `if prev_success_rate < success_rate:`. The returned value, `successes.sum() / cumulated_counter` (dmpel.py L499, base.py L347), therefore divides by the number of *improvements* rather than the number of evaluations. In the original LIBERO `algos/base.py`, the counter is incremented at every evaluation.

Observed on LIBERO-Goal with seeds 100 / 200 / 300: per-task `S_fwd` values reach 2.52 / 2.12 / 2.50. Recomputing FWT from the saved `task{k}_auc.log` curves with the LIBERO definition gives 0.717 / 0.680 / 0.716, which is consistent with the 0.68 ± 0.03 in the paper.

A possible fix is to move `cumulated_counter += 1.0` out of the `if` block, as in LIBERO. Could you confirm that the FWT and AUC numbers in the paper were computed from the learning curves rather than from `S_fwd`?

---

## Issue 2

**Title:** ER ignores `lifelong.n_memories`; the buffer is the first 20% of each task's sequences

`libero/configs/lifelong/er.yaml` sets `n_memories: 1000`, but `libero/lifelong/algos/er.py` (L56) builds each task's buffer as `TruncatedSequenceDataset(dataset, len(dataset)//5)`, with the `n_memories` argument commented out. `TruncatedSequenceDataset` (`libero/lifelong/datasets.py`, L133) returns indices `0 … buffer_size − 1`, so the buffer holds the first 20% of each task's sequences (the earliest demonstrations) rather than 1,000 sampled ones.

Was the reported ER baseline run with this 20% truncation? If so, documenting it would help others reproduce the baseline. If not, sampling `n_memories` random indices would match the config.

---

## Issue 3

**Title:** ER, EWC, PackNet and A-GEM clip AMP-scaled gradients (no `scaler.unscale_`), unlike `base.py` and `dmpel.py`

With `torch.amp.GradScaler`, gradients should be unscaled with `scaler.unscale_(optimizer)` before `clip_grad_norm_`. `libero/lifelong/algos/base.py` and `libero/lifelong/algos/dmpel.py` do this. `er.py`, `ewc.py`, `packnet.py` and `agem.py` instead call `clip_grad_norm_(..., grad_clip)` (default 100) directly on the scaled gradients (e.g. `er.py` L86–89).

With the default initial scale of 2^16, the threshold then applies to gradients about 65,000 times larger than the true ones, so clipping is likely active at almost every step for these baselines. I have not measured the effect on final success (AdamW normalises much of the gradient magnitude), but it does mean DMPEL and these baselines are trained with different clipping behaviour. Was this intended? Adding `self.scaler.unscale_(self.optimizer)` before clipping in those four files would align them with `base.py`.

---

## Issue 4

**Title:** Installation with the pinned `requirements.txt` fails: `torchtext` pin, and `SequenceDataset(demos=...)` with robomimic 0.2.0

Following the README with `requirements.txt`:

1. `torchtext==2.4.1+cu118` (L18) is not a published version, so `pip install -r requirements.txt` fails. torchtext is not imported on the DMPEL training path, so the line can probably be dropped.
2. `libero/lifelong/datasets.py` (L60) passes `demos=kwargs['demos']`, coming from `libero/lifelong/main.py` (L128), to robomimic's `SequenceDataset`. However, `SequenceDataset.__init__` in the pinned `robomimic==0.2.0` (and in 0.3.0) has no `demos` argument. Loading then fails with `TypeError: __init__() got an unexpected keyword argument 'demos'`, which is caught in `main.py` and surfaces as `UnboundLocalError: local variable 'task_i_dataset' referenced before assignment`.
3. Minor, external: with `transformers==4.21.1`, `AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")` currently fails, because the Hugging Face Hub now returns relative redirects that this version does not follow. Loading CLIP from a local copy works.

Was a modified robomimic used? We worked around item 2 by subclassing `SequenceDataset` to accept `demos` and forward it to `load_demo_info` (which already supports it in 0.2.0). Everything else ran as released.

---

Signature (to be appended to each issue, pending confirmation):
`— Rahul Bouri (rahulbouri16@gmail.com)`
