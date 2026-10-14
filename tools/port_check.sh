#!/usr/bin/env bash
# Port check of 72.2025_Imsil_swarm/DL_detection against the Haenam source (read-only copy of moho
# 27.Haenam_sequence/DL_detection). Read-only; writes DL_detection/logs/port_diff_vs_haenam.txt (shown in notebook 00).
#   1. MUST-BE-IDENTICAL files: sha256 equality (any difference = FAIL)
#   2. WHITELIST files (region values / paths only): full unified diff, renamed files paired with their Haenam file
#   3. every other ported .py/.sh/.csv/.crh: reported (new Imsil files are listed; an unlisted modified copy = FAIL)
#   ./tools/port_check.sh
set -uo pipefail
H=/home/msseo/works/_moho_reference/27.Haenam_sequence/DL_detection
D=/home/msseo/works/72.2025_Imsil_swarm/DL_detection
OUT=$D/logs/port_diff_vs_haenam.txt; mkdir -p $D/logs
IDENTICAL="pipeline/core.py pipeline/pick_augmentation.py pipeline/sumio.py pipeline/association.py pipeline/availability.py
pipeline/detection.py pipeline/make_phs.py pipeline/run_hypoinverse.py pipeline/run_pipeline.py pipeline/plotstyle.py
velocity_model/kim1983.csv HypoInv/kim1983/kim1983_p.crh HypoInv/kim1983/kim1983_s.crh HypoInv/kim2011/kim2011_p.crh HypoInv/kim2011/kim2011_s.crh
reloc/event_sac_export.py local_magnitudes/buan_sheen2018_snapshot.py
fmf/templates.py fmf/shelly.py fmf/scan.py fmf/segmad.py fmf/detect.py fmf/check_day.py fmf/progress.py
analysis/vet_candidate.py tools/xcorr_progress.py"
# imsil_file:haenam_file  (same name unless renamed)
WHITELIST="pipeline/config.py:pipeline/config.py run_imsil.sh:run_haenam.sh tools/mirror_mara.py:tools/mirror_mara.py
tools/link_continuous.py:tools/link_continuous.py tools/run_status.py:tools/run_status.py
station_table/build_stations.py:station_table/build_stations.py station_table/build_station_epochs.py:station_table/build_station_epochs.py
analysis/imsil_catalog.py:analysis/haenam_catalog.py analysis/add_reloc_origin_times.py:analysis/add_reloc_origin_times.py
analysis/decimal_seconds_final_locations.py:analysis/decimal_seconds_final_locations.py
reloc/run_dl_hypodd.py:reloc/run_dl_hypodd.py reloc/imsil_dl_cluster_config.py:reloc/haenam_dl_cluster_config.py
local_magnitudes/haenam_magnitudes.py:local_magnitudes/haenam_magnitudes.py local_magnitudes/jn_magnitudes.py:local_magnitudes/jn_magnitudes.py
local_magnitudes/full_period/full_magnitudes.py:local_magnitudes/full_period/full_magnitudes.py
local_magnitudes/full_period/validate_full_magnitudes.py:local_magnitudes/full_period/validate_full_magnitudes.py
local_magnitudes/full_period/station_review.py:local_magnitudes/full_period/station_review.py
fmf/config.py:fmf/config.py fmf/build_catalog.py:fmf/build_catalog.py fmf/run_full_period.sh:fmf/run_full_period.sh
fmf/run_imsil_5_20H.sh:fmf/run_5_20H.sh catalogs/build_template_catalog_ref25.py:catalogs/build_template_catalog_ref594.py"
nfail=0
{
echo "Port check $(date '+%Y-%m-%d %H:%M')   Imsil: $D   Haenam: $H"
echo; echo "== 1. must-be-identical (sha256)"
for f in $IDENTICAL; do
  a=$(sha256sum $D/$f 2>/dev/null | cut -c1-16); b=$(sha256sum $H/$f 2>/dev/null | cut -c1-16)
  if [[ -n "$a" && "$a" == "$b" ]]; then echo "  ok    $a  $f"; else echo "  FAIL  imsil=$a haenam=$b  $f"; nfail=$((nfail+1)); fi
done
echo; echo "== 2. whitelist diffs (region values / paths only)"
listed=""
for pair in $WHITELIST; do
  i=${pair%%:*}; h=${pair#*:}; listed="$listed $i"
  if cmp -s $D/$i $H/$h; then echo "  (unchanged) $i"; continue; fi
  echo; echo "---------------------------------------------------------------- $i   (vs Haenam $h)"
  diff -u --label "haenam/$h" --label "imsil/$i" $H/$h $D/$i
done
echo; echo "== 3. other files in DL_detection (code/tables)"
cd $D
for f in $(find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.crh' -o -name 'kim1983.csv' \) -not -path './logs/*' -not -path './runs/*' -not -path './models/*' -not -path '*/__pycache__/*' -not -path './notebooks/*' | sed 's#^\./##' | sort); do
  [[ " $IDENTICAL " == *" $f "* || " $listed " == *" $f "* ]] && continue
  if [[ -f $H/$f ]]; then
    if cmp -s $f $H/$f; then echo "  identical copy (not in either list)  $f"; else echo "  FAIL modified copy not whitelisted  $f"; nfail=$((nfail+1)); fi
  else echo "  new Imsil file  $f"; fi
done
echo; echo "RESULT: $([[ $nfail -eq 0 ]] && echo PASS || echo "FAIL ($nfail)")"
} > $OUT 2>&1
grep -E "^== |^  FAIL|^RESULT|^  new Imsil|^  \(unchanged\)|^-{20}" $OUT
echo "full report: $OUT"
exit $nfail
