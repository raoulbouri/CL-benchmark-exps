"""Idempotent patch to CLARE's evaluation for the CL-benchmark held-out protocol (Phase 1).

CLARE's default: gym_libero assigns initial states from a class-wide counter on first reset and
then steps by num_envs (mod 50); with 50 envs and 100 episodes every state is used exactly twice.

When the env var CLB_EVAL_STATE_IDS="0,1,...,39" is set, each vectorised env i is pinned to
state ids[i] before the first rollout. Use eval.batch_size == eval.n_episodes == len(ids) so one
batch covers each listed state exactly once. Unset = CLARE's original behaviour.

Also: success detection under gymnasium>=1.0 (no "final_info"; read info["is_success"]).

python scripts/patch_clare_protocol.py
"""
import os
import pathlib

CL_ROOT = os.environ.get("CL_ROOT", "/usr1/home/bbouri/continual-learning")
p = pathlib.Path(CL_ROOT + "/third_party/clare/lerobot_lsy/src/lerobot/scripts/eval_peft.py")
s = p.read_text()
MARK = "CLB_EVAL_STATE_IDS"
if MARK in s:
    print("already patched")
else:
    anchor = "    # Determine how many batched rollouts we need to get n_episodes."
    fn = s.index("def eval_policy_with_env_init(")
    pos = s.index(anchor, fn)  # first occurrence inside eval_policy_with_env_init
    nxt = s.find("\ndef ", fn + 1)
    assert nxt == -1 or pos < nxt, "anchor not inside eval_policy_with_env_init"
    inject = (
        "    # CLB protocol: pin each env to an explicit initial-state id (see CL-benchmark docs/PROTOCOL.md)\n"
        "    _clb_ids = os.environ.get(\"CLB_EVAL_STATE_IDS\")\n"
        "    if _clb_ids:\n"
        "        _clb_ids = [int(x) for x in _clb_ids.split(\",\") if x.strip()]\n"
        "        assert n_episodes == len(_clb_ids) == env.num_envs, (\n"
        "            f\"CLB heldout needs n_episodes == batch_size == len(state ids); got {n_episodes}, {env.num_envs}, {len(_clb_ids)}\")\n"
        "        for _i in range(env.num_envs):\n"
        "            _e = env.envs[_i].env.env\n"
        "            _e._set_id = True\n"
        "            _e._init_state_id = _clb_ids[_i]\n"
        "        logging.info(f\"[CLB] heldout eval on init states {_clb_ids[0]}..{_clb_ids[-1]} ({len(_clb_ids)})\")\n"
    )
    s = s[:pos] + inject + s[pos:]
    if "\nimport os\n" not in s:
        s = s.replace("import json\n", "import json\nimport os\n", 1)
    if "\nimport logging\n" not in s:
        s = s.replace("import json\n", "import json\nimport logging\n", 1)
    p.write_text(s)
    print("patched", p)
# 2. success detection under gymnasium>=1.0 (no "final_info"; is_success is a per-env array in info)
s = p.read_text()
MARK2 = "CLB gymnasium>=1.0 success"
if MARK2 in s:
    print("already patched (success)")
else:
    old = """            successes = [False] * env.num_envs"""
    assert s.count(old) == 1, "success anchor not unique"
    new = """            successes = [False] * env.num_envs
            if "is_success" in info:  # CLB gymnasium>=1.0 success: final_info no longer exists
                successes = [bool(x) for x in np.asarray(info["is_success"]).reshape(-1)[: env.num_envs]]"""
    s = s.replace(old, new)
    p.write_text(s)
    print("patched success detection")
compile(p.read_text(), str(p), "exec")
print("syntax ok")
