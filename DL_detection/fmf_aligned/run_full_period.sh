#!/bin/bash
# FMF template matching on the Imsil 25 m reference templates (IMSIL PORT of the Haenam full-period driver: same commands and
# settings; template csv, GPU lock path, default years and the batch year order changed; GPU budget 12e9 fits the 40 GB A100).
#   ./run_full_period.sh templates            # Beauce picks/windows + Shelly P/S templates (SNR gate, S on Z at KS/KG)
#   ./run_full_period.sh scan YEAR D0 D1      # one GPU scan job (resumable per day)
#   ./run_full_period.sh batch [YEARS]        # years in parallel (default 2020 ... 2026), 4 at a time, renice 15
#   ./run_full_period.sh progress             # per-year completion, rate, ETA, process health
#   ./run_full_period.sh check YEAR DOY       # scan one day (if not yet) + self-detection / timing report
#   ./run_full_period.sh catalog              # merge + floor/gate + matching -> fmf/catalogs/fmf_haenam_catalog_full_period_seg.csv
cd "$(dirname "$0")"
export FMF_RUN=${FMF_RUN:-full_period} FMF_BAND=${FMF_BAND:-2,15} FMF_TEMPLATE_CSV=imsil_templates_ref25.csv MAD_SEGMENTED=1
export FMF_SNR_MIN=${FMF_SNR_MIN:-1.2} FMF_MAD_SUBSAMPLE=${FMF_MAD_SUBSAMPLE:-8} FMF_STEP=${FMF_STEP:-1}
export FMF_GPU_LOCK=${FMF_GPU_LOCK:-/home/msseo/works/72.2025_Imsil_swarm/DL_detection/fmf/logs/gpu.lock}
export FMF_WORKERS=${FMF_WORKERS:-16} FMF_GPU_BUDGET=${FMF_GPU_BUDGET:-12e9} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/msseo/miniforge3/envs/fmf/bin/python; LOG=logs; [ "$FMF_RUN" != "full_period" ] && LOG=logs/$FMF_RUN; mkdir -p $LOG; export FMF_LOG_DIR=$LOG
case "$1" in
  templates) nice -n 10 $PY templates.py 2>&1 | tee $LOG/templates.log && nice -n 10 $PY shelly.py build 2>&1 | tee $LOG/shelly_build.log ;;
  scan) nice -n 15 $PY shelly.py "$2" "$3" "$4" ;;
  batch) YEARS=${2:-"2025 2026"}; echo "batch start $(date) (STEP=${FMF_STEP:-1}, parallel ${FMF_PARALLEL:-1}, years: $YEARS)" >> $LOG/batch_status.txt   # quiescence years first; each year runs in FMF_CHUNK-day chunks with a FRESH process (the FMF GPU calls slow down ~2.5x over 6 h inside one process). MEASURED 2026-09-23: 1 process 90 s/day; 2 processes 45 (out of phase) -> 100 (lockstep); per-batch lock 95; day-level lock 120; 3 processes 138-141. One process is the reliable optimum.
         n=0; for y in $YEARS; do ( t0=$(date +%s); for d0 in $(seq 1 ${FMF_CHUNK:-30} 366); do d1=$(( d0 + ${FMF_CHUNK:-30} - 1 )); [ $d1 -gt 366 ] && d1=366; nice -n 15 $PY shelly.py $y $d0 $d1 >> $LOG/scan_$y.log 2>&1; done; echo "$y done in $(( $(date +%s) - t0 )) s exit $?" >> $LOG/batch_status.txt ) & n=$((n+1)); if [ $n -ge ${FMF_PARALLEL:-1} ]; then wait -n; n=$((n-1)); fi; done; wait; echo "batch end $(date)" >> $LOG/batch_status.txt ;;
  progress) $PY progress.py ;;
  check) $PY check_day.py "$2" "$3" ;;
  catalog) $PY build_catalog.py 2>&1 | tee $LOG/catalog.log ;;
  *) echo "usage: $0 templates | scan YEAR D0 D1 | batch [YEARS] | catalog"; exit 1 ;;
esac
