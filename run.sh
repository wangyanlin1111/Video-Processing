#!/bin/bash
timestamp=$(date +'%Y%m%d_%H%M%S')
log_file="run_${timestamp}.log"
echo "Start Working!!! Output log to $log_file"
python run.py "$1" 2>&1 | tee "$log_file"
