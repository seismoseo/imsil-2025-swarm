#!/usr/bin/env bash
# Reproducible driver for the IMSIL DL-detection project (72.2025_Imsil_swarm/DL_detection, on PAGO).
# IMSIL PORT of run_haenam.sh (moho 27.Haenam_sequence/DL_detection): same stages and commands; only the parameter
# block (envs, registry, external code, PATH) and paths changed. Imsil period: 2025 DOY 244-365 + 2026 DOY 1-31.
# Imsil order: stations -> mirror list/plan/pull -> link -> epochs -> (smoke: detect 2025 306-307, locate 2025, catalog)
#   -> years 2025-2026 (freeze) -> catalog -> dl-reloc -> magnitudes -> fmf-templates/fmf-scan/fmf-catalog.
# stage-augment (fold-back into a canonical KCR frame) does not apply to Imsil.
#
# The chain is the validated Buan one (20.2024_Buan_EQ_DL_detection + 21.2024_Buan_precursor):
#   PhaseNet+ (EQNet) picks -> PyOcto RELAXED-STRICT association (3P+3S, tol 1.0 s)
#   -> HYPOINVERSE kim1983 -> QC tiers + KMA/template cross-match
#   -> FMF GPU template matching (Shelly two-pass, 195 relocated templates)
#   -> fold-back of new Haenam events into the canonical relocation (PocketQuake --augment)
#
# DRY RUN BY DEFAULT: every stage prints the exact command it would run. Add `run` to execute:
#   ./run_imsil.sh stations run                # station_table/stations.csv from the Imsil 100-km KS list (D1/D2 switches)
#   ./run_imsil.sh mirror list 2025-2026 run   # remote inventory of the 100-km KS stations on MARA (cached)
#   ./run_imsil.sh mirror plan 2025-2026 run   # channel policy + PERIOD filter -> worklists + manifest (GB per year)
#   ./run_imsil.sh mirror pull 2025 run        # rsync one year's period (detached is fine; idempotent, resumable)
#   ./run_imsil.sh mirror status 2025          # manifest table; `mirror status 2025 --assert-complete` = the year gate
#   ./run_imsil.sh link run                    # continuous/ symlink tree over continuous_mara/
#   ./run_imsil.sh epochs run                  # station_table/station_epochs.csv + cross-checks (KWJ2/KWJU)
#   ./run_imsil.sh detect 2025 306-307 run     # SMOKE: PhaseNet+ picks for the swarm days (move outputs aside afterwards)
#   ./run_imsil.sh locate 2025 run             # association -> augment -> PHS -> HYPOINVERSE (whole year)
#   ./run_imsil.sh catalog run                 # catalogs/imsil_dl_catalog.csv (tiers, KMA + known-event cross-match)
#   ./run_imsil.sh years 2025-2026 run         # FROZEN run per year: gate -> link -> parameter hash -> detect 1-366 -> locate -> status
#   ./run_imsil.sh status 2025-2026 run        # logs/full_run_status.csv (per-year linked/picked/located/QC counts)
#   ./run_imsil.sh dl-reloc --years 2025-2026 --min-depth 10 --open-wdcc run   # HypoDD relocation of the DL catalog (GPU)
#   ./run_imsil.sh fmf-templates run           # FMF templates from the 25 m reference catalog
#   ./run_imsil.sh fmf-scan 2025 244-365 run   # GPU scan, resumable per day (env fmf)
#   ./run_imsil.sh fmf-catalog run             # merge + floor/gate + matching -> fmf/catalogs/*.csv
#   WORKERS=4 ./run_imsil.sh detect ...        # DataLoader workers for PhaseNet+ (EQNet shards stations across them)
#
# Parameters (edit here, or override in the environment):
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
PY_BASE=/home/msseo/miniforge3/envs/dl-base/bin/python    # IMSIL PORT: dl-base env = rebuild of moho's base (obspy, seisbench, pyocto, torch cu130)
reg(){ python3 -B -c "import sys; sys.path.insert(0, '/home/msseo/works/72.2025_Imsil_swarm/registry'); import imsil as R; print($1)"; }   # registry (IMSIL PORT)
PY_FMF=/home/msseo/miniforge3/envs/fmf/bin/python         # fmf env: fast_matched_filter (GPU build)
MODEL=${MODEL:-phasenet_plus}
VELMODEL=${VELMODEL:-kim1983}
DEVICE=${DEVICE:-cuda}
STRICT=${STRICT:---strict}                                # "" for the loose 4-pick association
NICE=${NICE:-10}                                          # shared 64-core box: yield to other users
WORKERS=${WORKERS:-0}                                     # PhaseNet+ DataLoader workers (0 = in-process, as the 2026 run)
PARAMS_JSON=logs/full_run_params.json                     # parameter freeze for the full-period run (hash gate in `years`)
PARAM_FILES="pipeline/config.py pipeline/core.py pipeline/pick_augmentation.py station_table/stations.csv /home/msseo/works/72.2025_Imsil_swarm/_external/EQNet/docs/model_phasenet_plus/model_99.pth"
EXT=/home/msseo/works/72.2025_Imsil_swarm/_external                 # IMSIL PORT: EQNet, KCR, FMF source, hypoDD (tools/phase0_copy_from_moho.sh)
export PATH=$EXT/bin:$PATH                                # hyp1.40/ph2dt/ncsn2pha wrappers (libgfortran.so.3 shim) + hypoDD v2.1beta first
export HELVETICA_DIR=/home/msseo/Download/Helvetica      # pipeline/plotstyle.py reads this (PAGO font folder used by the other notebooks)
export PYTHONUNBUFFERED=1
mkdir -p logs

stage=${1:-help}
run=0
for a in "$@"; do [[ "$a" == "run" ]] && run=1; done
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
do_cmd() {                                                # print; execute only with `run`
  echo "    $*"
  if [[ $run -eq 1 ]]; then nice -n "$NICE" "$@"; else echo "    (dry run — append 'run' to execute)"; fi
}

case "$stage" in
  link)
    say "continuous symlink tree <- MARA mirror + NECIS cron archive + JNU tars + Earthworm (stations.csv only)"
    extra=(); for x in "${@:2}"; do [[ "$x" == "run" ]] || extra+=("$x"); done      # e.g. --years 2026 --refresh
    do_cmd $PY_BASE tools/link_continuous.py --apply "${extra[@]}"
    ;;
  stations)
    say "rebuild station_table/stations.csv (KS/KG 100-km set + JU within 100 km)"
    do_cmd $PY_BASE station_table/build_stations.py
    ;;
  detect)
    year=${2:?usage: detect YEAR D0-D1 [run]}; days=${3:?usage: detect YEAR D0-D1 [run]}
    say "PhaseNet+ detection $year DOY $days -> models/$MODEL/detection_location/$year/picks/ (resumable)"
    do_cmd $PY_BASE pipeline/detection.py --model "$MODEL" --year "$year" --days "$days" --device "$DEVICE" --workers "$WORKERS"
    ;;
  redetect)
    year=${2:?usage: redetect YEAR D0-D1 [run]}; days=${3:?usage: redetect YEAR D0-D1 [run]}
    say "RE-pick $year DOY $days ignoring existing day files (use after 'link' adds stations to already-picked days)"
    do_cmd $PY_BASE pipeline/detection.py --model "$MODEL" --year "$year" --days "$days" --device "$DEVICE" --no-skip-existing --workers "$WORKERS"
    ;;
  locate)
    year=${2:?usage: locate YEAR [run]}
    say "association ($STRICT) -> augment -> PHS -> HYPOINVERSE $VELMODEL for $year"
    do_cmd $PY_BASE pipeline/run_pipeline.py --model "$MODEL" --years "$year" --velmodel "$VELMODEL" \
        --stage-from association $STRICT
    ;;
  catalog)
    say "DL catalog: QC tiers, distances, KMA + known-event cross-match -> catalogs/imsil_dl_catalog.csv"
    do_cmd $PY_BASE analysis/imsil_catalog.py
    ;;
  vet)
    say "record section for a candidate: ./run_imsil.sh vet 'YYYY-MM-DD HH:MM:SS' out.png run"
    do_cmd $PY_BASE analysis/vet_candidate.py "${2:?origin time}" "${3:-}"
    ;;
  fmf-templates)
    say "FMF templates (Beauce npz) + Shelly P/S templates from catalogs/imsil_templates_ref25.csv (fmf/run_imsil_5_20H.sh)"
    ( cd "$(reg "R.path(None, 'tm_dir')")" && do_cmd ./run_imsil_5_20H.sh templates )
    ;;
  fmf-scan)
    year=${2:?usage: fmf-scan YEAR D0-D1 [run]}; days=${3:?usage: fmf-scan YEAR D0-D1 [run]}
    d0=${days%-*}; d1=${days#*-}
    say "FMF Shelly GPU scan $year DOY $d0-$d1 (fmf/run_imsil_5_20H.sh, segmented 12xMAD) -> fmf/catalogs/"
    ( cd "$(reg "R.path(None, 'tm_dir')")" && do_cmd ./run_imsil_5_20H.sh scan "$year" "$d0" "$d1" )
    ;;
  fmf-catalog)
    say "merge daily FMF detections, apply floor/gate, cross-match -> fmf/catalogs/fmf_imsil_catalog*.csv"
    ( cd "$(reg "R.path(None, 'tm_dir')")" && do_cmd ./run_imsil_5_20H.sh catalog )
    ;;
  dl-reloc)
    say "standalone HypoDD relocation of the DL catalog (Buan recipe, canonical Haenam settings, Imsil box) -> runs/hypodd_dl/"
    say "  GPU xcorr over all pairs; expect hours. Extra args pass through (e.g. --max-events 30 --n-boot 100)"
    shift; args=(); for x in "$@"; do [[ "$x" == "run" ]] || args+=("$x"); done
    set -a; . $EXT/imsil.env; set +a                        # IMSIL PORT: replaces PocketQuake/.env (no credentials needed)
    do_cmd /home/msseo/miniforge3/envs/pq-gpu/bin/python reloc/run_dl_hypodd.py "${args[@]}"
    ;;
  stage-augment)
    say "not used for Imsil: there is no canonical KCR relocation to fold DL events into (use dl-reloc)"
    ;;
  mirror)
    sub=${2:?usage: mirror list|plan|pull|verify|status YEARS [--assert-complete] [run]}; yrs=${3:-2025-2026}
    say "MARA mirror: $sub $yrs -> /home/msseo/works/72.2025_Imsil_swarm/continuous_mara (tools/mirror_mara.py; period filter PERIOD)"
    extra=(); for x in "${@:4}"; do [[ "$x" == "run" ]] || extra+=("$x"); done
    if [[ "$sub" == "status" || "$sub" == "verify" ]]; then
      $PY_BASE tools/mirror_mara.py "$sub" --years "$yrs" "${extra[@]}"
    elif [[ $run -eq 1 ]]; then do_cmd $PY_BASE tools/mirror_mara.py "$sub" --years "$yrs" --apply "${extra[@]}"
    else $PY_BASE tools/mirror_mara.py "$sub" --years "$yrs" "${extra[@]}"; fi
    ;;
  epochs)
    say "per-year station epochs from the linked tree -> station_table/station_epochs.csv (+ cross-checks)"
    do_cmd $PY_BASE station_table/build_station_epochs.py
    ;;
  status)
    yrs=${2:-2025-2026}
    say "per-year run status $yrs -> logs/full_run_status.csv"
    if [[ $run -eq 1 ]]; then do_cmd $PY_BASE tools/run_status.py --years "$yrs" --append; else $PY_BASE tools/run_status.py --years "$yrs"; fi
    ;;
  years)
    yrs=${2:?usage: years Y0-Y1 [--from detect|locate] [--no-gate] [run]}
    from=detect; gate=1
    for ((i=3; i<=$#; i++)); do
      case "${!i}" in --from) j=$((i+1)); from=${!j};; --no-gate) gate=0;; esac
    done
    y0=${yrs%-*}; y1=${yrs#*-}
    say "full-period run: years $y0..$y1 from stage '$from' (gate=$gate, WORKERS=$WORKERS); same picker/association/QC for every year"
    # parameter freeze: refuse to run a year if any pipeline file or the station table changed since the first year
    hash_now=$(sha256sum $PARAM_FILES | sha256sum | cut -c1-16)
    if [[ -f $PARAMS_JSON ]]; then
      hash_ref=$($PY_BASE -c "import json;print(json.load(open('$PARAMS_JSON'))['hash'])")
      if [[ "$hash_now" != "$hash_ref" ]]; then say "ABORT: parameter hash $hash_now != frozen $hash_ref ($PARAMS_JSON). Re-freeze deliberately or restore the files."; exit 2; fi
    elif [[ $run -eq 1 ]]; then
      $PY_BASE - "$hash_now" <<'PYEOF'
import json, sys, os, hashlib, datetime
sys.path.insert(0, "pipeline"); import config
d = dict(hash=sys.argv[1], frozen=datetime.datetime.now().isoformat(timespec="seconds"),
         PNPLUS_MIN_PROB=config.PNPLUS_MIN_PROB, REGION_STRICT={k: v for k, v in config.REGION_STRICT.items() if not hasattr(v, "__call__")} if isinstance(config.REGION_STRICT, dict) else str(config.REGION_STRICT),
         HAENAM_QC=config.HAENAM_QC, HAENAM_QC_A=config.HAENAM_QC_A, HAENAM_QC_B=config.HAENAM_QC_B, HYP_DIS=config.HYP_DIS, HYP_ZTR=config.HYP_ZTR,
         CHANNEL_POLICY=config.CHANNEL_POLICY, CHANNEL_EXCLUDE={".".join(k): v for k, v in getattr(config, "CHANNEL_EXCLUDE", {}).items()}, SAMPLING_RATE=config.SAMPLING_RATE, ANTIALIAS=config.ANTIALIAS,
         weights_sha256=hashlib.sha256(open(config.EQNET_WEIGHTS, "rb").read()).hexdigest())
json.dump(d, open("logs/full_run_params.json", "w"), indent=1, default=str); print("  frozen parameters ->", "logs/full_run_params.json")
PYEOF
    fi
    summary=()
    for ((y=y0; y<=y1; y++)); do
      say "===== year $y ====="
      if [[ $gate -eq 1 ]] && ! $PY_BASE tools/mirror_mara.py status --years "$y" --year "$y" --assert-complete; then
        say "year $y: mirror not complete -> skipped (use --no-gate to override)"; summary+=("$y: skipped (mirror incomplete)"); continue
      fi
      set +e
      ok=1
      if [[ $run -eq 1 ]]; then
        "$0" link run > "logs/full_${y}_link.log" 2>&1 || ok=0
        if [[ "$from" == "detect" ]]; then "$0" detect "$y" 1-366 run > "logs/full_${y}_detect.log" 2>&1 || ok=0; fi
        "$0" locate "$y" run > "logs/full_${y}_locate.log" 2>&1 || ok=0
        $PY_BASE tools/run_status.py --years "$y" --append > "logs/full_${y}_status.log" 2>&1 || ok=0
      else
        echo "    would run: link -> detect $y 1-366 (WORKERS=$WORKERS) -> locate $y -> status $y   (logs/full_${y}_*.log)"
      fi
      set -e
      summary+=("$y: $([[ $ok -eq 1 ]] && echo done || echo FAILED — see logs/full_${y}_*.log)")
    done
    say "SUMMARY"; printf '    %s
' "${summary[@]}"
    ;;
  *)
    sed -n 2,32p "$0"
    ;;
esac
