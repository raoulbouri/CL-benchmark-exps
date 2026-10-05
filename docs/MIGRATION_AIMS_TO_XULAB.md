# Migration from AIMS to xulab (4 Oct 2026)

Goal: everything lives on `xulab` under `/mnt/data/users/bbouri`. AIMS has been unreachable since 3 Oct. This lists what was copied, what was regenerated, and what is still only on AIMS.

## 1. Where everything is

| Item | On the Mac | On xulab | Only on AIMS | How to get it |
|---|---|---|---|---|
| Harness code (`cl_bench/`, `scripts/`, `tests/`), run specs, queues | yes (working copy) | yes (synced) | the git history | Fresh git repo started on xulab 4 Oct |
| Docs (`PI_REVIEW`, `DECISIONS`, `PROTOCOL`, `COST_MODEL`, `REPO_MAP`; `PLAN`, `HANDOFF` and the issue drafts were deleted 4 Oct) | yes | yes (synced 4 Oct) | newer appends (none known beyond the DDP entry, now restored) | when AIMS returns, diff `docs/` against it |
| Third-party patches (`configs/patches/*.patch`) | no | no | yes | Regenerate: apply `scripts/patch_*.py`, then `git diff` in the clone (the scripts are the source of truth) |
| Third-party clones | no | DMPEL @ b1abe28, patched | LIBERO, openpi, CLARE, gym-libero | Public; re-clone at the commits in `configs/third_party.lock` |
| LIBERO demonstrations (100 GB) | no | downloading (log: `/mnt/data/users/bbouri/logs/download.log`) | yes (94 GB) | Public (HF `yifengzhu-hf/LIBERO-datasets`) |
| Checkpoints | no | DMPEL, CLIP | CLARE checkpoint and datasets | Public (HF `leiyuheng/DMPEL`, `continuallearning/*`); CLARE needs the `v2.1` dataset revision |
| Conda environments | no | `dmpel` (py3.10, torch 2.7.1+cu128) | `libero`, `dmpel` (py3.8), `clare` | `scripts/xulab_setup.sh`; a `clare` env needs torch >= 2.7 for the 5090 and is not built yet |
| **Result JSONs (10 files)** | only the ER-on-xulab result | that one | the other 9 (3 Goal replication runs, 1 smoke run, 5 micro-runs) | W&B artifacts `result-<run name>` for online runs (needs W&B login), or rsync from AIMS |
| **Run outputs** (Goal run dirs ~567 MB each: `config.json`, `result.pt`, learning curves, expert statistics, coefficient buffers) | no | no | yes | rsync from AIMS when it returns (priority: the 3 Goal runs). Otherwise regenerate (about 9 GPU-h each) |
| Logs | no | partial | yes | Low value |
| W&B API key (`.env`) | no | no | yes | Create `/mnt/data/users/bbouri/CL-benchmark/.env` yourself (`WANDB_API_KEY=...`, `chmod 600`) |

## 2. Is anything unrecoverable?
Nothing essential. Code and docs are fully recoverable (the Mac working copy matches what was pushed to AIMS, except the AIMS-side git history). Public data, checkpoints and clones can be re-fetched. What is expensive to lose is **the Goal run directories** (about 27 GPU-hours to regenerate); rsync them from AIMS first when it comes back. The numbers from those runs are also recorded in this repository's docs and in W&B.

## 3. When AIMS returns (in this order)
1. `ssh aims` and confirm it is up.
2. Copy the Goal run directories and `results/` to xulab: `rsync -a aims:/usr1/home/bbouri/continual-learning/outputs/dmpel/libero_goal/ /mnt/data/users/bbouri/cl/outputs/dmpel/libero_goal/` and `rsync -a aims:/usr1/home/bbouri/CL-benchmark/results/ /mnt/data/users/bbouri/CL-benchmark/results/`.
3. Bring over the git history: `git remote add aims aims:/usr1/home/bbouri/CL-benchmark && git fetch aims`, then merge with `--allow-unrelated-histories` (the xulab history starts from the Mac copy).
4. Check the sampling ratio of the Goal runs: `python3 -c "import json;print(json.load(open('<run_dir>/config.json'))['lifelong']['moe_attn_recall_sample_ratio'])"`. Expected 1.0.
5. Treat AIMS as a read-only archive afterwards.

## 4. Why the Goal reproduction does not validate the 5% memory claim
- **What ran:** the replication used the default config, with no override of `moe_attn_recall_sample_ratio`; the released default is 1.0 (every frame). Evidence: the config in the public repo; our launch command had no override; and the stored coefficient file for Goal task 9 was 54.5 MB, which matches 8,808 rows x 1,545 floats (every training frame), where 5% would be about 440 rows and 2.7 MB. The saved `config.json` on AIMS will confirm.
- **What that means:** the run shows that the *released defaults* reproduce the paper's Goal numbers (FWT 0.704, NBT 0.000, AUC 0.806 vs 0.68 / 0.00 / 0.78). It does not test whether coefficient replay works at 5%, which is what the paper's Fig. 5b and the 0.06 GB storage figure refer to. We do not know which ρ the paper's main table used.
- **What to do:** run DMPEL on Goal at ρ = 5% (one seed first) as the first experiment. That is the reproduction of the paper's memory claim.
