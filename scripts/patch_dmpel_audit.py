"""Idempotent patch: record what DMPEL's coefficient buffer and router actually did (zero effect on learning).

The released code (a) throws away the random permutation that picks the buffer frames, (b) iterates the shuffled
train loader when it builds the buffer, so even a saved permutation could not be mapped back to (demo, step), and
(c) never records the router output during evaluation rollouts. This patch adds three audit outputs:

  1. algos/dmpel.py   buffer pass iterates the training loader with a sequential sampler (single GPU only), so buffer row i is dataset index i.
                      Writes task{k}_buffer_audit.pth: rho, total frames, kept indices, per-frame (demo id, step),
                      and the router's top-k expert index and weight for EVERY frame (not only the kept ones).
  2. models/modules/adapter.py   MoERouterCoeff.forward appends (top-k idx, top-k coeff, ctx) per call when the
                      router is in eval mode and env CLB_AUDIT=1.
  3. metric.py        evaluate_one_task_success sets the (loop, step) context before each get_action call and
                      saves routing/<task>_<n>_<sel|matrix>.npz after each evaluation. Nothing is recorded unless CLB_AUDIT=1.

The sequential order applies only to the post-training buffer pass (no gradients, augmentation off); the set of
frames is identical, only their order differs. The training sampler is not consumed the same way, so a patched
run is not bit-identical to an unpatched one with the same seed (seeds are not reproducible across GPUs anyway).
DDP runs keep the original path (no audit buffer file).
"""
import os
import pathlib

CL_ROOT = os.environ.get("CL_ROOT", "/usr1/home/bbouri/continual-learning")
D = pathlib.Path(CL_ROOT + "/third_party/DMPEL/libero/lifelong")
MARK = "CLB audit"


def patch(path, edits):
    p = D / path
    s = p.read_text()
    if MARK in s:
        print(f"{path}: already patched")
        return
    for old, new in edits:
        assert s.count(old) == 1, f"{path}: anchor not found or not unique:\n{old}"
        s = s.replace(old, new)
    p.write_text(s)
    compile(s, str(p), "exec")
    print(f"{path}: patched, syntax ok")


V2_BLOCK = """            # CLB audit v2: unshuffled order (single GPU) so that buffer row i is dataset index i. The training loader's
            # persistent workers are reused: workers forked now would inherit robomimic's obs-key table after evaluation reset it.
            audit_loader = train_dataloader
            if not self.cfg.use_ddp:
                from torch.utils.data import SequentialSampler
                train_dataloader.batch_sampler.sampler = SequentialSampler(dataset)
"""

# 1. buffer pass: sequential order + audit file
patch("algos/dmpel.py", [
    (
        """            topk_attn_norm_list = []
            for (idx, data) in enumerate(train_dataloader):
                moe_query_in, topk_idx, topk_attn_norm = self.save_attn_observe(data)
""",
        """            topk_attn_norm_list = []
            # CLB audit v2: unshuffled order (single GPU) so that buffer row i is dataset index i. The training loader's
            # persistent workers are reused: workers forked now would inherit robomimic's obs-key table after evaluation reset it.
            audit_loader = train_dataloader
            if not self.cfg.use_ddp:
                from torch.utils.data import SequentialSampler
                train_dataloader.batch_sampler.sampler = SequentialSampler(dataset)
            for (idx, data) in enumerate(audit_loader):
                moe_query_in, topk_idx, topk_attn_norm = self.save_attn_observe(data)
""",
    ),
    (
        """            rand_indices = torch.randperm(total_sample_num, device=moe_query_in_img_list.device)[:saved_sample_num]
""",
        """            rand_indices = torch.randperm(total_sample_num, device=moe_query_in_img_list.device)[:saved_sample_num]
            if not self.cfg.use_ddp:  # CLB audit: what the buffer holds and what the router did on every frame
                seqds = getattr(dataset, "sequence_dataset", None)
                assert total_sample_num == len(dataset), "audit loader order mismatch"
                demo_id, step = [], []
                for i in range(total_sample_num):
                    d_id = seqds._index_to_demo_id[i] if seqds is not None else ""
                    demo_id.append(d_id)
                    step.append(i - seqds._demo_id_to_start_indices[d_id] if seqds is not None else -1)
                torch.save({
                    "rho": float(self.cfg.lifelong.moe_attn_recall_sample_ratio),
                    "total_sample_num": total_sample_num,
                    "kept_indices": rand_indices.cpu(),
                    "demo_id": demo_id,
                    "step": torch.tensor(step),
                    "topk_idx": topk_idx_list.cpu(),
                    "topk_attn_norm": topk_attn_norm_list.cpu(),
                    },
                    os.path.join(self.experiment_dir, f"task{task_id}_buffer_audit.pth")
                )
""",
    ),
])

# v1 -> v2 upgrade. v1 built a fresh DataLoader for the buffer pass; its workers forked after evaluation and crashed with
# "OBS_KEYS_TO_MODALITIES is None" (robomimic's global obs-key table is reset by the evaluation code). Seen 2026-10-04, end of task 0.
_p = D / "algos/dmpel.py"
_s = _p.read_text()
_v1 = """            # CLB audit: unshuffled loader (single GPU) so that buffer row i is dataset index i
            audit_loader = train_dataloader
            if not self.cfg.use_ddp:
                audit_loader = DataLoader(dataset, batch_size=self.cfg.train.batch_size,
                                          num_workers=self.cfg.train.num_workers, shuffle=False)
"""
if _v1 in _s:
    _s = _s.replace(_v1, V2_BLOCK)
    _p.write_text(_s)
    compile(_s, str(_p), "exec")
    print("algos/dmpel.py: upgraded v1 -> v2, syntax ok")

# metric.py v1 -> v2: v1 dropped router calls whose batch size differed, leaving loop/step misaligned with topk_* (found in the smoke run)
_pm = D / "metric.py"
_sm = _pm.read_text()
_V1M = """            np.savez_compressed(
                os.path.join(rdir, f"eval{n_prev:04d}_task{task_id}_{kind}.npz"),
                loop=np.array([c[0][0] for c in log], dtype=np.int16),
                step=np.array([c[0][1] for c in log], dtype=np.int16),
                topk_idx=np.concatenate([c[1][None] for c in log if c[1].shape == log[0][1].shape]),
                topk_coeff=np.concatenate([c[2][None] for c in log if c[2].shape == log[0][2].shape]),
            )
"""
if _V1M in _sm:
    _sm = _sm.replace(_V1M, """            # one record per router call; rows of call j are consecutive in topk_* (call_rows[j] rows, one per env)
            np.savez_compressed(
                os.path.join(rdir, f"eval{n_prev:04d}_task{task_id}_{kind}.npz"),
                call_loop=np.array([c[0][0] for c in log], dtype=np.int16),
                call_step=np.array([c[0][1] for c in log], dtype=np.int16),
                call_rows=np.array([c[1].shape[0] for c in log], dtype=np.int16),
                topk_idx=np.concatenate([c[1] for c in log], axis=0),
                topk_coeff=np.concatenate([c[2] for c in log], axis=0),
            )
""")
    _pm.write_text(_sm)
    compile(_sm, str(_pm), "exec")
    print("metric.py: upgraded routing flush v1 -> v2, syntax ok")

# 2. router forward: per-call log (eval mode only, opt-in by env var)
patch("models/modules/adapter.py", [
    (
        """import numpy as np

def generate_orthogonal_tensor""",
        """import numpy as np
import os as _clb_os

CLB_ROUTE_LOG = []  # CLB audit: filled by MoERouterCoeff.forward in eval mode when CLB_AUDIT=1


def generate_orthogonal_tensor""",
    ),
    (
        """            topk_coeff, topk_idx = torch.topk(coeff, topk, dim=-1, sorted=False)
        
        return topk_coeff, topk_idx
""",
        """            topk_coeff, topk_idx = torch.topk(coeff, topk, dim=-1, sorted=False)
        
        if (not self.training) and _clb_os.environ.get("CLB_AUDIT") == "1":
            CLB_ROUTE_LOG.append((getattr(self, "clb_ctx", (-1, -1)),
                                  topk_idx.detach().cpu().numpy().astype("int16"),
                                  topk_coeff.detach().float().cpu().numpy().astype("float16")))
        return topk_coeff, topk_idx
""",
    ),
])

# 3. evaluation: context before each action, flush after each evaluation
patch("metric.py", [
    (
        """                actions = algo.policy.get_action(data)
""",
        """                if hasattr(algo.policy, "moe_router"):
                    algo.policy.moe_router.clb_ctx = (i, steps)  # CLB audit
                actions = algo.policy.get_action(data)
""",
    ),
    (
        """    avg_t = np.mean(np.array(inference_time))
""",
        """    if os.environ.get("CLB_AUDIT") == "1" and hasattr(algo.policy, "moe_router"):  # CLB audit
        from libero.lifelong.models.modules import adapter as _clb_adapter
        log = _clb_adapter.CLB_ROUTE_LOG
        if log:
            rdir = os.path.join(cfg.experiment_dir, "routing")
            os.makedirs(rdir, exist_ok=True)
            n_prev = len(os.listdir(rdir))
            kind = "sel" if task_str == "" else "matrix"
            # one record per router call; rows of call j are consecutive in topk_* (call_rows[j] rows, one per env)
            np.savez_compressed(
                os.path.join(rdir, f"eval{n_prev:04d}_task{task_id}_{kind}.npz"),
                call_loop=np.array([c[0][0] for c in log], dtype=np.int16),
                call_step=np.array([c[0][1] for c in log], dtype=np.int16),
                call_rows=np.array([c[1].shape[0] for c in log], dtype=np.int16),
                topk_idx=np.concatenate([c[1] for c in log], axis=0),
                topk_coeff=np.concatenate([c[2] for c in log], axis=0),
            )
        del log[:]
    avg_t = np.mean(np.array(inference_time))
""",
    ),
])
