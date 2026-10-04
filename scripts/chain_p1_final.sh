#!/usr/bin/env bash
# Phase 1 final runs, strictly one job at a time (host RAM): wait for the running CLARE queue, then
# DMPEL ER on GPU 0, then CLARE original on GPU 1.   nohup bash scripts/chain_p1_final.sh > logs/chain_p1_final.log 2>&1 &
cd /usr1/home/bbouri/CL-benchmark
while pgrep -f "queue.sh 1 queues/p1_clare_seq.txt" > /dev/null; do sleep 60; done
bash scripts/queue.sh 0 queues/p1_final_gpu0.txt
bash scripts/queue.sh 1 queues/p1_final_gpu1.txt
echo "[chain] done $(date -Is)"
