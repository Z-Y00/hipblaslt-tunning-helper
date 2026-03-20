#!/usr/bin/env bash
# Run lm_head mbs=4 tuning with origami disabled (full search space).
# Safe to close laptop — runs under nohup + disown.
#
# Usage:
#   ./run_lm_head_origami0.sh          # start
#   tail -f tunning_results/lm_head_origami0.log   # monitor
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="tunning_results/lm_head_origami0.log"
mkdir -p tunning_results

echo "Starting lm_head origami-0 tuning at $(date)"
echo "Log: $LOG"

nohup python3 run_shapes.py \
    --run \
    --parallel 8 \
    --origami-top-n 0 \
    --filter "Llama-3.1-70B" \
    --filter-layer lm_head \
    --filter-mbs 4 \
    --force \
    > "$LOG" 2>&1 &

PID=$!
disown $PID

echo "PID: $PID"
echo "Disowned — safe to close terminal."
echo "Follow with: tail -f $LOG"
