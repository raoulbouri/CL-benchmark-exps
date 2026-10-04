"""W&B helper: deterministic run ids, online with offline fallback, small payloads only.

Free plan storage is 5 GB, so only metrics, tables and result.json go to W&B (PI_REVIEW.md section 4).
"""
import hashlib
import os

from . import WANDB_ENTITY, WANDB_PROJECT, load_env


def run_id(name, attempt=0):
    return hashlib.sha1(f"{name}#{attempt}".encode()).hexdigest()[:12]


def init(name, config, group, tags, attempt=0, job_type="train"):
    load_env()
    import wandb
    kw = dict(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, id=run_id(name, attempt),
              resume="allow", group=group, tags=list(tags), job_type=job_type, config=config,
              dir=os.environ.get("WANDB_DIR", "/usr1/home/bbouri/CL-benchmark/wandb"))
    os.makedirs(kw["dir"], exist_ok=True)
    try:
        return wandb.init(**kw), "online"
    except Exception as e:  # network or service failure: keep logging locally, sync later
        print(f"[clb] W&B online init failed ({type(e).__name__}); falling back to offline mode", flush=True)
        os.environ["WANDB_MODE"] = "offline"
        # pass the mode explicitly: wandb has already read its settings, so the env var alone is not picked up
        return wandb.init(**dict(kw, mode="offline")), "offline"
