"""Phase 0 gate: the metrics library must reproduce the DMPEL LIBERO-Goal numbers computed
before the harness existed (scripts/compute_metrics.py in continual-learning)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cl_bench import CL_ROOT, dmpel_export, metrics  # noqa: E402

EXPECTED = {100: (0.717, -0.015, 0.839), 200: (0.680, 0.006, 0.782)}


@pytest.mark.parametrize("seed", [100, 200])
def test_reproduces_goal_replication(seed):
    spec = {"track": "D", "suite": "libero_goal", "method": "dmpel", "budget": "default",
            "protocol": "original", "seed": seed}
    run_dir = str(CL_ROOT / f"outputs/dmpel/libero_goal/seed_{seed}/seed_{seed}")
    rec = dmpel_export.build_record(spec, run_dir, str(CL_ROOT / f"logs/dmpel/libero_goal_seed{seed}.log"), "complete")
    m = rec["metrics"]
    fwt, nbt, auc = EXPECTED[seed]
    assert rec["tasks_done"] == 10
    assert m["fwt"] == pytest.approx(fwt, abs=5e-4)
    assert m["nbt"] == pytest.approx(nbt, abs=5e-4)
    assert m["auc"] == pytest.approx(auc, abs=5e-4)


def test_fwt_held_at_peak():
    # LIBERO definition: after the best epoch the curve is held at its peak
    assert metrics.fwt_from_curve([0.1, 0.65, 0.6, 0.65, 0.95, 0.7]) == pytest.approx(3.9 / 6)


def test_no_forgetting_gives_zero_nbt():
    c = np.zeros((3, 3))
    for t in range(3):
        c[t, : t + 1] = 0.8
    m = metrics.compute(c, [[0.8]] * 3)
    assert m["nbt"] == pytest.approx(0.0) and m["final_success"] == pytest.approx(0.8)


def test_total_forgetting():
    c = np.array([[1.0, 0, 0], [0.0, 1.0, 0], [0.0, 0.0, 1.0]])
    m = metrics.compute(c, [[1.0]] * 3)
    assert m["nbt"] == pytest.approx(1.0) and m["nbt_normalized"] == pytest.approx(1.0)
