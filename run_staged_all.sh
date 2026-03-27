#!/bin/bash
# Launch staged search on all shapes from config.py
# Usage: bash run_staged_all.sh [--device SPEC]
#
# Device spec examples:
#   --device 1         single GPU
#   --device 1,3,5     specific GPUs
#   --device 2-7       range of GPUs
#   --device 1,3-5     mixed
#
# Runs detached (nohup) so the terminal can be closed safely.
# Log: staged_results/run_staged_all_<timestamp>.log

set -euo pipefail
cd "$(dirname "$0")"

DEVICE="1"  # default GPU 1 (avoid GPU 0)
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="staged_results/run_staged_all_${TIMESTAMP}.log"
mkdir -p staged_results

cat <<EOF
=== Staged Search: All Shapes ===
  Device: $DEVICE
  Log:    $LOG
  Start:  $(date)

Launching in background (nohup)...
EOF

nohup python3 -u run_staged_all.py --device "$DEVICE" $EXTRA_ARGS > "$LOG" 2>&1 &
PID=$!
disown "$PID"

echo "  PID:    $PID"
echo ""
echo "Monitor: tail -f $LOG"
echo "Stop:    kill $PID"
