#!/usr/bin/env bash
# Run a queue on ONE xulab GPU with the safety guard alongside.  Usage:
#   CLB_MAX_HOURS=24 setsid nohup bash scripts/run_xulab.sh <gpu> <queue_file> > /mnt/data/users/bbouri/logs/run.log 2>&1 < /dev/null &
# One guard per GPU: a guard left over from an earlier launch is stopped first (a stale guard with a short --max-hours would
# otherwise stop this job), and this launch's guard is stopped when the queue ends. CLB_MAX_HOURS defaults to 12.
set -uo pipefail
GPU=$1; QUEUE=$2
source /mnt/data/users/bbouri/env_x.sh
source $CL_ROOT/env_dmpel.sh
cd $CLB_ROOT
mkdir -p $XB/logs
pkill -u "$(id -un)" -f "[g]uard.py $GPU " 2>/dev/null; sleep 1
python scripts/guard.py "$GPU" --max-hours ${CLB_MAX_HOURS:-12} > $XB/logs/guard.out 2>&1 &
GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT
bash scripts/queue.sh "$GPU" "$QUEUE"
