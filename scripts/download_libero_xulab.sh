#!/usr/bin/env bash
# Download the full LIBERO demonstration set (about 100 GB) to /mnt/data/users/bbouri/cl/data/libero on xulab.
# Priority order so the suites needed first finish first: goal, 10 (Long), spatial, object, 90 (largest, only needed for pretraining).
# Resumable (re-run to continue), low priority, 4 parallel file streams to be gentle on the shared network.
#   nohup nice -n 10 bash scripts/download_libero_xulab.sh > /mnt/data/users/bbouri/logs/download.log 2>&1 &
set -euo pipefail
source /mnt/data/users/bbouri/cl/env_dmpel.sh
python - <<'EOF'
import os, time, shutil
from huggingface_hub import snapshot_download
CL = os.environ["CL_ROOT"]; dest = f"{CL}/data/libero"
free_tb = shutil.disk_usage(dest).free / 1e12
print(f"destination {dest} | free {free_tb:.1f} TB", flush=True)
assert free_tb > 1.0, "less than 1 TB free: refusing to download"
for suite in ["libero_goal", "libero_10", "libero_spatial", "libero_object", "libero_90"]:
    t = time.time()
    snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset", local_dir=dest,
                      allow_patterns=[f"{suite}/*"], max_workers=4)
    print(f"{suite}: downloaded in {(time.time() - t) / 60:.1f} min", flush=True)

# verification: expected file counts and 50 demos per file
import glob, h5py
expected = {"libero_goal": 10, "libero_10": 10, "libero_spatial": 10, "libero_object": 10, "libero_90": 90}
ok = True
for suite, n in expected.items():
    files = sorted(glob.glob(f"{dest}/{suite}/*.hdf5"))
    bad = []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                if len(h["data"]) != 50:
                    bad.append((os.path.basename(f), len(h["data"])))
        except Exception as e:
            bad.append((os.path.basename(f), type(e).__name__))
    status = "OK" if (len(files) == n and not bad) else "PROBLEM"
    ok &= status == "OK"
    print(f"{status}: {suite}: {len(files)}/{n} files, {sum(os.path.getsize(f) for f in files) / 1e9:.1f} GB, bad={bad}", flush=True)
print("DOWNLOAD VERIFIED" if ok else "DOWNLOAD HAS PROBLEMS", flush=True)
EOF
echo "DOWNLOAD SCRIPT DONE"
