#!/usr/bin/env bash
# Run BF16 dense GEMM tuning for all shapes using all 8 GPUs
# with Origami analytical pruning (top 30 tiles per shape).
#
# Runs under nohup so it survives terminal disconnects.
# Output is saved to a timestamped log in tunning_results/.
#
# Usage:
#   ./run_all.sh                                # all models, fwd+bwd
#   ./run_all.sh --filter Llama-3.1-8B          # one model
#   ./run_all.sh --fwd-only                     # forward pass only
#   ./run_all.sh --filter Llama-3.1-8B --fwd-only --max-shapes 4
#
# Follow the log:
#   tail -f tunning_results/run_all_*.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="tunning_results/run_all_$(date +%Y%m%d_%H%M%S).log"
mkdir -p tunning_results

echo "Starting tuning run at $(date)"
echo "Log: $LOG"

nohup python3 run_shapes.py \
    --run \
    --parallel 8 \
    --origami-top-n 30 \
    "$@" \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "Follow with: tail -f $LOG"
