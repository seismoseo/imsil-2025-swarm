#!/bin/bash
# Scan every data day of the Imsil period in its own FRESH process (2026-10-08, D20 runs).
# On PAGO the FMF loader's forked worker processes die (BrokenProcessPool) when they are forked from a process that has already
# scanned a day: in run_full_period.sh batch (30-day chunks) only the first scanned day of each chunk succeeded (verified:
# shelly.py 2025 245 246 -> 245 ok, 246 fails; 246 alone ok; with forkserver workers both succeed). Not the cause: CUDA
# initialisation alone, memory limits, BLAS threads, consecutive loads without a scan (all tested 2026-10-08). One day per
# process avoids the failure; the scan entry point and therefore the results are unchanged. shelly.py skips days already scanned.
# usage: tools/fmf_scan_days.sh fmf|fmf_aligned [run]        (dry run without 'run')
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:?usage: tools/fmf_scan_days.sh fmf|fmf_aligned [run]}; run=${2:-}
days=(); for d in $(seq 244 365); do days+=("2025 $d"); done; for d in $(seq 1 31); do days+=("2026 $d"); done
echo "[fmf_scan_days] $dir: ${#days[@]} days (2025 244-365, 2026 1-31), one process per day"
[[ "$run" == "run" ]] || { echo "  (dry run - add 'run')"; exit 0; }
cd "$dir"; n=0
for yd in "${days[@]}"; do
  set -- $yd
  ./run_imsil_5_20H.sh scan "$1" "$2" "$2" 2>&1 | grep -E "DOY|ERROR|Error" || true
  n=$((n + 1))
done
echo "[fmf_scan_days] $dir: done ($n days)"
