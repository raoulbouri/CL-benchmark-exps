#!/usr/bin/env bash
# Run a list of run specs sequentially on one GPU. Completed runs are skipped (see docs/DECISIONS.md).
#   nohup bash scripts/queue.sh <gpu> <queue_file> > logs/queue_gpu<gpu>.log 2>&1 &
# queue_file: one spec path per line; lines starting with # are ignored.
set -uo pipefail
GPU=$1; QUEUE=$2
ROOT=${CLB_ROOT:-/usr1/home/bbouri/CL-benchmark}
source ${CLB_HARNESS_ENV:-/usr1/home/bbouri/continual-learning/env.sh}   # harness env
export WANDB_DIR="$ROOT/wandb"; mkdir -p "$WANDB_DIR"
cd "$ROOT"
echo "[queue] gpu=$GPU queue=$QUEUE start $(date -Is)"
grep -vE '^\s*(#|$)' "$QUEUE" | while read -r SPEC; do
  echo "[queue] $(date -Is) launching $SPEC"
  ${CLB_NICE:-} python scripts/launch.py "$SPEC" --gpu "$GPU"; RC=$?
  echo "[queue] $(date -Is) $SPEC exited rc=$RC"
done
echo "[queue] gpu=$GPU done $(date -Is)"
