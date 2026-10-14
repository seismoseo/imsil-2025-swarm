#!/bin/bash
# LONG-PERIOD VARIANT (decisions D25, D26; 2026-10-11): identical to run_imsil_5_20H.sh (aligned code, 5-20 Hz, same 28 templates) except
#   * station list stations_imsil_5_20H_long.csv: S on Z at the 9 stations whose response epoch changes inside 2018-2026 (NAWB, KWJ2 as in D12,
#     plus MALA, SUCA, TAIA, JLSA, HAMB, JAGA, DUSB); P on Z and S on N/E elsewhere, as before;
#   * FMF_RUN=imsil_5_20H_aligned_mad10_long and FMF_MAD_K=10 by default (D23; the 0.20 floor is dropped at the catalog step, FMF_CC_FLOOR=0).
# The scan period 2018-01-01 .. 2026-10-03 is driven by tools/fmf_long_rolling.py (pull from MARA, scan one day per process, delete).
cd "$(dirname "$0")"
export FMF_RUN=${FMF_RUN:-imsil_5_20H_aligned_mad10_long} FMF_MAD_K=${FMF_MAD_K:-10} FMF_BAND=5,20 FMF_STATION_LIST=$(pwd)/stations_imsil_5_20H_long.csv FMF_SEQUENCES=imsil
exec ./run_full_period.sh "$@"
