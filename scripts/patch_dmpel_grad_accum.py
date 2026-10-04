"""Idempotent patch: gradient accumulation for the DMPEL-codebase ER (decided 2026-10-01).

ER fully fine-tunes CLIP and concatenates a replay batch to every batch, so batch 16 (16 + 16 replay)
does not fit a 24 GB RTX 4090. With train.batch_size=8 and train.grad_accum_steps=4, each optimizer
step sees 32 current + 32 replay samples, matching the authors' 2-GPU DDP x 16 setup.
Default train.grad_accum_steps=1 reproduces the original behaviour exactly.

Not changed (fidelity to the released baseline, logged in docs/DECISIONS.md): ER clips gradients
without scaler.unscale_(), i.e. on fp16-scaled gradients, unlike algos/base.py.
"""
import os
import pathlib

CL_ROOT = os.environ.get("CL_ROOT", "/usr1/home/bbouri/continual-learning")

D = pathlib.Path(CL_ROOT + "/third_party/DMPEL/libero")

cfg = D / "configs/train/default.yaml"
s = cfg.read_text()
if "grad_accum_steps" not in s:
    s = s.replace("grad_clip: 100.", "grad_clip: 100.\ngrad_accum_steps: 1 # CLB: micro-batches per optimizer step (ER only)", 1)
    assert "grad_accum_steps" in s, "train/default.yaml anchor missing"
    cfg.write_text(s)
    print("patched train/default.yaml")
else:
    print("already patched: train/default.yaml")

er = D / "lifelong/algos/er.py"
s = er.read_text()
if "CLB grad accumulation" in s:
    print("already patched: er.py")
else:
    old_start = "    def start_task(self, task):\n        super().start_task(task)\n"
    assert s.count(old_start) == 1
    s = s.replace(old_start, old_start + "        self._clb_micro = 0  # CLB grad accumulation: reset per task\n")
    old = """        data = self.map_tensor_to_device(data)

        self.optimizer.zero_grad()
        with amp.autocast('cuda', dtype=torch.float16):
            loss = self.policy.compute_loss(data)
        self.scaler.scale(self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()"""
    assert s.count(old) == 1, "ER observe anchor not found"
    new = """        data = self.map_tensor_to_device(data)

        # CLB grad accumulation: step every `accum` micro-batches (accum=1 is the original code path)
        accum = int(self.cfg.train.get("grad_accum_steps", 1) or 1)
        micro = getattr(self, "_clb_micro", 0)
        if micro % accum == 0:
            self.optimizer.zero_grad()
        with amp.autocast('cuda', dtype=torch.float16):
            loss = self.policy.compute_loss(data)
        self.scaler.scale(self.loss_scale * loss / accum).backward()
        self._clb_micro = micro + 1
        if self._clb_micro % accum == 0:
            if self.cfg.train.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.train.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return loss.item()"""
    s = s.replace(old, new)
    er.write_text(s)
    print("patched er.py")
compile(er.read_text(), str(er), "exec")
print("syntax ok")
