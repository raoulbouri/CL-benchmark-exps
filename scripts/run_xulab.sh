#!/usr/bin/env bash
# Run a queue on ONE xulab GPU with the safety guard alongside.  Usage:
#   nohup bash scripts/run_xulab.sh <gpu> <queue_file> > /mnt/data/users/bbouri/logs/run.log 2>&1 &
set -uo pipefail
GPU=$1; QUEUE=$2
source /mnt/data/users/bbouri/env_x.sh
source $CL_ROOT/env_dmpel.sh
cd $CLB_ROOT
mkdir -p $XB/logs
nohup python scripts/guard.py "$GPU" --max-hours ${CLB_MAX_HOURS:-12} > $XB/logs/guard.out 2>&1 &
bash scripts/queue.sh "$GPU" "$QUEUE"
