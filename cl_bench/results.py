"""Results schema (one JSON per run in results/, git-tracked) and atomic writes."""
import datetime
import json
import os
import socket
import subprocess

from . import RESULTS, ROOT, SCHEMA_VERSION

SUITE_SHORT = {"libero_10": "long", "libero_goal": "goal", "libero_spatial": "spatial",
               "libero_object": "object", "libero_90": "l90"}


def run_name(spec):
    suite = SUITE_SHORT.get(spec["suite"], spec["suite"])
    if spec.get("n_tasks"):
        suite += str(spec["n_tasks"])
    budget = spec["budget"] + (f"-{spec['tag']}" if spec.get("tag") else "")  # optional tag keeps variants apart
    return f"{spec['track']}-{suite}-{spec['method']}-{budget}-{spec['protocol']}-s{spec['seed']}"


def group_name(spec):
    return run_name(spec).rsplit("-s", 1)[0]


def path_for(name):
    return RESULTS / f"{name}.json"


def load(name):
    p = path_for(name)
    return json.loads(p.read_text()) if p.exists() else None


def write(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec["schema_version"] = SCHEMA_VERSION
    rec["updated_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    p = path_for(rec["name"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    os.replace(tmp, p)  # atomic: a crash never leaves a half-written result
    return p


def _git(cwd, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def provenance():
    lock = ROOT / "configs" / "third_party.lock"
    gpu = None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()[0]
    except Exception:
        pass
    return {"harness_commit": _git(ROOT, "rev-parse", "--short", "HEAD"),
            "harness_dirty": bool(_git(ROOT, "status", "--porcelain")),
            "third_party_lock": lock.read_text() if lock.exists() else None,
            "host": socket.gethostname(), "gpu": gpu}
