"""CL-Benchmark harness: run specs, results schema, metrics, accounting and W&B logging.

Local JSON in results/ is the source of truth; W&B mirrors it (see PI_REVIEW.md, section 4).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("CLB_ROOT", "/usr1/home/bbouri/CL-benchmark"))
CL_ROOT = Path(os.environ.get("CL_ROOT", "/usr1/home/bbouri/continual-learning"))
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"
SCHEMA_VERSION = 1
WANDB_ENTITY = "rahulbouri16"
WANDB_PROJECT = "cl-benchmark"


def load_env(path=ROOT / ".env"):
    """Load KEY=VALUE lines from .env into os.environ without printing anything."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
