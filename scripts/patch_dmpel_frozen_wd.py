"""Idempotent, opt-in patch: keep frozen experts exactly constant (fixes AdamW weight decay shrinking them).

Problem (Experiment 3): DMPEL "freezes" an expert only by detaching its rows of a stacked pool tensor in the forward pass.
The tensor stays in the optimizer (AdamW, weight_decay 0.1), its frozen rows get a zero gradient (not None), and AdamW's
decoupled decay still multiplies them by (1 - lr * wd) at every step. Over 10 Long tasks the old experts shrink to ~0.8 of
their norm, which cost task index 1 its performance (restoring the weights recovered it).

Fix: experts are rows of one Parameter shared with the new expert, so they cannot be removed from the optimizer one by one.
Instead, at the first step of each task the frozen rows are snapshotted and, after every optimizer step, written back. The new
expert and the router are untouched (they train and decay exactly as before). Active only when CLB_FIX_FROZEN_WD=1, so default
runs are unchanged.

  models/modules/adapter.py : clb_snapshot_frozen(policy), clb_restore_frozen(snapshot)
  algos/dmpel.py            : observe() snapshots on the first step of a task and restores after scaler.step
"""
import os
import pathlib

CL_ROOT = os.environ.get("CL_ROOT", "/usr1/home/bbouri/continual-learning")
D = pathlib.Path(CL_ROOT + "/third_party/DMPEL/libero/lifelong")

pa = D / "models/modules/adapter.py"
sa = pa.read_text()
if "clb_snapshot_frozen" not in sa:
    sa = sa.rstrip("\n") + '''


# CLB frozen-wd fix (scripts/patch_dmpel_frozen_wd.py): keep frozen expert rows exactly constant
CLB_POOL_ATTRS = ("A_pool", "B_pool", "bias_pool", "A_q_pool", "A_v_pool", "B_q_pool", "B_v_pool")


def clb_snapshot_frozen(policy):
    """[(pool parameter, frozen-row mask, copy of the frozen rows)] for every expert pool that has frozen rows."""
    snap = []
    for m in policy.modules():
        mask = getattr(m, "_frozen_mask", None)
        if mask is None or not bool(mask.any()):
            continue
        for a in CLB_POOL_ATTRS:
            p = getattr(m, a, None)
            if isinstance(p, nn.Parameter) and p.shape[0] == mask.shape[0]:
                snap.append((p, mask.clone(), p.data[mask].clone()))
    return snap


def clb_restore_frozen(snap):
    for p, mask, vals in snap:
        p.data[mask] = vals
'''
    pa.write_text(sa)
    compile(sa, str(pa), "exec")
    print("models/modules/adapter.py: frozen-row helpers added")

pd = D / "algos/dmpel.py"
sd = pd.read_text()
if "CLB frozen-wd fix" not in sd:
    a1 = "        data = self.map_tensor_to_device(data)\n        self.optimizer.zero_grad()\n"
    n1 = a1 + (
        "        if os.environ.get(\"CLB_FIX_FROZEN_WD\") == \"1\":  # CLB frozen-wd fix\n"
        "            from libero.lifelong.models.modules.adapter import clb_snapshot_frozen\n"
        "            if getattr(self, \"_clb_snap_task\", None) != self.current_task:  # first step of this task: frozen rows are final\n"
        "                self._clb_snap = clb_snapshot_frozen(self.policy)\n"
        "                self._clb_snap_task = self.current_task\n")
    a2 = "        self.scaler.step(self.optimizer)\n        self.scaler.update()\n        return bc_loss.item()\n"
    n2 = ("        self.scaler.step(self.optimizer)\n        self.scaler.update()\n"
          "        if os.environ.get(\"CLB_FIX_FROZEN_WD\") == \"1\":  # CLB frozen-wd fix: undo AdamW decay on frozen experts\n"
          "            from libero.lifelong.models.modules.adapter import clb_restore_frozen\n"
          "            clb_restore_frozen(self._clb_snap)\n"
          "        return bc_loss.item()\n")
    for o in (a1, a2):
        assert sd.count(o) == 1, f"algos/dmpel.py anchor not unique:\n{o}"
    sd = sd.replace(a1, n1).replace(a2, n2)
    pd.write_text(sd)
    compile(sd, str(pd), "exec")
    print("algos/dmpel.py: observe() patched (opt-in)")
else:
    print("algos/dmpel.py: already patched")


# Live per-expert norms (always available when CLB_AUDIT=1): printed at the start and at the end of each task's training, before the
# best-checkpoint reload. Needed because the saved task checkpoint is the best-success epoch, which can be the untrained epoch 0.
sa = pa.read_text()
if "def clb_expert_norms" not in sa:
    sa = sa.rstrip("\n") + '''


def clb_expert_norms(policy):
    """live L2 norm of every expert's A-type, B-type and bias rows: {'A': [...], 'B': [...], 'bias': [...]}"""
    kinds = {"A": ("A_pool", "A_q_pool", "A_v_pool"), "B": ("B_pool", "B_q_pool", "B_v_pool"), "bias": ("bias_pool",)}
    out = {}
    for kind, names in kinds.items():
        tot = None
        for n, p in policy.named_parameters():
            if n.split(".")[-1] in names:
                s = (p.detach().float() ** 2).flatten(1).sum(1)
                tot = s if tot is None else tot + s
        out[kind] = [] if tot is None else [round(float(x) ** 0.5, 6) for x in tot]
    return out
'''
    pa.write_text(sa)
    compile(sa, str(pa), "exec")
    print("models/modules/adapter.py: clb_expert_norms added")
sd = pd.read_text()
if "clb_expert_norms" not in sd:
    b1 = "        # start training\n        for epoch in range(0, self.cfg.train.n_epochs + 1):\n"
    b2 = "        # load the best performance agent on the current task\n"
    for o in (b1, b2):
        assert sd.count(o) == 1, f"algos/dmpel.py norm-log anchor not unique:\n{o}"
    log = lambda when: (
        "        if os.environ.get(\"CLB_AUDIT\") == \"1\":  # CLB audit: live expert norms\n"
        "            from libero.lifelong.models.modules.adapter import clb_expert_norms\n"
        f"            print(f\"[clb] task {{task_id}} {when}: expert norms {{clb_expert_norms(self.policy)}}\", flush=True)\n")
    sd = sd.replace(b1, log("training start") + b1).replace(b2, log("training end (before best-checkpoint reload)") + b2)
    pd.write_text(sd)
    compile(sd, str(pd), "exec")
    print("algos/dmpel.py: live expert norm logging added")
