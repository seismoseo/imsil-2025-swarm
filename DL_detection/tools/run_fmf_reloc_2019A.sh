#!/bin/bash
# Joint relocation of the 2019-04-15 long-scan events with all 235 swarm events (export -> HypoDD/KCR -> 25 m rule), 2026-10-14.
# usage: tools/run_fmf_reloc_2019A.sh [TAG]   TAG names the run root runs/hypodd_fmf_TAG and the outputs (default 2019A).
#   2019A      : first run; the 2019 day held only the 22 FMF stations' files (A events linked at 8-9 stations)
#   2019A_full : after fetching all 53 KS stations with data on 2019-04-15 (tools/link_keep_long.py), as for the swarm events
set -eo pipefail
TAG=${1:-2019A}
cd /home/msseo/works/72.2025_Imsil_swarm/DL_detection
export PATH=/home/msseo/works/72.2025_Imsil_swarm/_external/bin:$PATH; set -a; . ../_external/imsil.env; set +a
echo "[$TAG] start $(date)"
/home/msseo/miniforge3/envs/dl-base/bin/python reloc/export_fmf_detections.py --run-root runs/hypodd_fmf_$TAG --extra-csv catalogs/fmf_long_2019A.csv --apply
XCORR_TRACE_CACHE=30000 /home/msseo/miniforge3/envs/pq-gpu/bin/python reloc/run_fmf_hypodd.py --run-root runs/hypodd_fmf_$TAG --n-boot 1000 --out-csv catalogs/imsil_fmf_relocated_$TAG.csv
/home/msseo/miniforge3/envs/dl-base/bin/python catalogs/build_reference_25m.py --run-root runs/hypodd_fmf_$TAG --relocated catalogs/imsil_fmf_relocated_$TAG.csv --outdir catalogs/reference_25m_fmf_$TAG --apply
echo "[$TAG] done $(date)"
