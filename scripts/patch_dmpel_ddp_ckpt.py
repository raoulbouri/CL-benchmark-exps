"""Idempotent patch: DDP checkpoint race in DMPEL's algos/base.py `learn_one_task`.

After training, every rank loads the best checkpoint and then every rank rewrites the same file with
torch.save, without a barrier. With a large checkpoint (ER fully fine-tunes CLIP: 697 MB) one rank reads a
file the other is rewriting -> "PytorchStreamReader failed reading file" (observed 2026-10-03, attempt 6).

Fix: all ranks load, barrier, only rank 0 saves, barrier. Identical weights are written, so learning is unchanged.
Non-DDP runs follow exactly the original path.
"""
import os
import pathlib

CL_ROOT = os.environ.get("CL_ROOT", "/usr1/home/bbouri/continual-learning")

p = pathlib.Path(CL_ROOT + "/third_party/DMPEL/libero/lifelong/algos/base.py")
s = p.read_text()
if "CLB ddp ckpt race" in s:
    print("already patched")
else:
    old = """            # print(f'[info] {msg}')

        torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
"""
    assert s.count(old) == 1, "anchor not found or not unique"
    new = """            # print(f'[info] {msg}')

        # CLB ddp ckpt race: all ranks must finish reading before anyone rewrites the file; only rank 0 writes
        if self.cfg.use_ddp:
            torch.distributed.barrier()
        if (not self.cfg.use_ddp) or int(os.environ["RANK"]) == 0:
            torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
        if self.cfg.use_ddp:
            torch.distributed.barrier()
"""
    p.write_text(s.replace(old, new))
    print("patched", p.name)
compile(p.read_text(), str(p), "exec")
print("syntax ok")
