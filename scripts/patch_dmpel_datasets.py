"""Compat patch: DMPEL passes demos= to robomimic SequenceDataset, which no released robomimic accepts."""
import os

CL_ROOT = os.environ.get("CL_ROOT", "/usr1/home/bbouri/continual-learning")
p = CL_ROOT + "/third_party/DMPEL/libero/lifelong/datasets.py"
s = open(p).read()
shim = '''

class _SequenceDatasetWithDemos(SequenceDataset):
    """robomimic 0.2.0 SequenceDataset + a `demos` kwarg (int indices or 'demo_i' keys).

    DMPEL calls SequenceDataset(..., demos=range(n_demos_per_task)), which appears to rely on a locally
    modified robomimic; released robomimic (0.2.0/0.3.0) only supports `demos` inside load_demo_info.
    """

    def __init__(self, *args, demos=None, **kwargs):
        self._demos_override = None if demos is None else [d if isinstance(d, str) else f"demo_{d}" for d in demos]
        super().__init__(*args, **kwargs)

    def load_demo_info(self, filter_by_attribute=None, demos=None):
        return super().load_demo_info(filter_by_attribute=filter_by_attribute,
                                      demos=demos if demos is not None else self._demos_override)
'''
anchor = "from robomimic.utils.dataset import SequenceDataset\n"
if "_SequenceDatasetWithDemos" in s:
    print("already patched"); raise SystemExit(0)
assert anchor in s
s = s.replace(anchor, anchor + shim, 1)
call = "    dataset = SequenceDataset(\n        hdf5_path=dataset_path,"
assert call in s
s = s.replace(call, "    dataset = _SequenceDatasetWithDemos(\n        hdf5_path=dataset_path,", 1)
open(p, "w").write(s)
print("patched")
