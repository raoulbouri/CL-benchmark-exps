"""Safety guard for runs on a shared workstation: stop OUR job (never anyone else's) if the machine looks unhealthy.

python scripts/guard.py <gpu_index> [--max-hours 12]

Checks every 60 s: GPU temperature, GPU memory, system-disk free space, host RAM available, driver responsiveness
(nvidia-smi not answering = the failure mode seen on AIMS), and stuck (D-state) processes owned by this user.
On a violation it sends SIGTERM to scripts/launch.py (the launcher then kills the whole training process group,
marks the run 'interrupted' and writes its results) and writes the reason to logs/guard.log.
nvidia-smi is run with a hard deadline and is never waited on after the deadline, so a hung driver cannot hang the guard.
"""
import getpass
import os
import shutil
import subprocess
import sys
import time

GPU = int(sys.argv[1])
MAX_H = float(sys.argv[sys.argv.index("--max-hours") + 1]) if "--max-hours" in sys.argv else 12.0
LOG = os.path.join(os.environ.get("XB", "/mnt/data/users/bbouri"), "logs", "guard.log")
LIM = {"temp_c": 85, "gpu_mem_mib": 29000, "root_free_gb": 4, "ram_avail_gb": 30}
ME = getpass.getuser()


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%F %T')} [guard gpu{GPU}] {msg}\n")


def run(cmd, deadline=20):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    t0 = time.time()
    while p.poll() is None:
        if time.time() - t0 > deadline:
            try:
                p.kill()
            except OSError:
                pass
            return None  # abandon: do not wait on a process stuck in the driver
        time.sleep(0.2)
    return p.stdout.read()


def job_pids():
    out = run(["pgrep", "-u", ME, "-f", "scripts/launch.py"], 10)
    return [int(x) for x in (out or "").split()]


def stop(reason):
    log(f"STOP: {reason}")
    for pid in job_pids():
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    sys.exit(2)


start, seen_job, idle, no_answer, dstate = time.time(), False, 0, 0, 0
log(f"started; limits {LIM}")
while True:
    if time.time() - start > MAX_H * 3600:
        stop(f"max runtime {MAX_H} h reached")
    pids = job_pids()
    seen_job |= bool(pids)
    idle = 0 if pids else idle + 1
    if seen_job and idle >= 3:
        log("job finished; guard exiting")
        sys.exit(0)
    out = run(["nvidia-smi", "-i", str(GPU), "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"])
    if out is None:
        no_answer += 1
        if no_answer >= 2:
            stop("nvidia-smi did not answer twice in a row (driver unresponsive)")
    else:
        no_answer = 0
        t, m = [float(x) for x in out.strip().split(",")]
        if t > LIM["temp_c"]:
            stop(f"GPU temperature {t:.0f} C")
        if m > LIM["gpu_mem_mib"]:
            stop(f"GPU memory {m:.0f} MiB")
    if shutil.disk_usage("/").free / 1e9 < LIM["root_free_gb"]:
        stop("system disk almost full")
    avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0] / 1e6
    if avail < LIM["ram_avail_gb"]:
        stop(f"host RAM available {avail:.0f} GB")
    st = run(["ps", "-u", ME, "-o", "stat="], 10) or ""
    n_d = sum(1 for s in st.split() if s.startswith("D"))
    dstate = dstate + 1 if n_d >= 2 else 0
    if dstate >= 3:
        stop(f"{n_d} of our processes stuck in uninterruptible sleep for 3 checks")
    time.sleep(60)
