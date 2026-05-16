#!/bin/bash
set -euo pipefail
clear
timestamp=$(date +'%Y%m%d_%H%M%S')
LOG_DIR="../log"
mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/run_${timestamp}.log"
echo "Start Working!!! Output log to $log_file"
python run.py "$1" "$2" 2>&1 | tee "$log_file"