#!/usr/bin/env bash
# Phase 0 (S0) of the Imsil DL-catalog port: bring the external code, model weights, binaries and input tables that the
# Haenam pipeline needs onto PAGO, with a provenance record. Plan: ~/.claude/plans/what-i-want-to-shimmying-goose.md
#
# DRY RUN BY DEFAULT — prints every action. Append `run` to execute:   ./tools/phase0_copy_from_moho.sh run
# Re-running is safe (rsync is incremental; existing files are overwritten only if they differ).
#
# What lands where (I = 72.2025_Imsil_swarm, X = I/_external):
#   X/EQNet/                    moho 14.EQNet/EQNet (@ad492ef + local diff), incl. docs/model_phasenet_plus/model_99.pth
#   X/korea-cluster-relocation/ moho 15.PocketQuake/external/korea-cluster-relocation (@663c26b + local diff), CODE ONLY
#                               (no *_cluster/ run trees, no pipeline/runs, pipeline/notebooks, pipeline/velocity_models)
#   X/fast_matched_filter/      moho works/FMF/fast_matched_filter (@25e8263 + Makefile diff), source only (rebuilt on PAGO)
#   X/HypoDD/                   moho Downloads/HypoDD (@fb40407 + MAXDATA diff), source + examples (fallback rebuild, smoke test)
#   X/bin/hypoDD                moho ~/bin/hypoDD (v2.1beta, MAXDATA 15M; needs only libgfortran.so.5, present on PAGO)
#   X/bin/{hyp1.40,ph2dt,ncsn2pha}  wrappers around PAGO ~/bin/<tool> adding X/lib (libgfortran.so.3, missing on Rocky 10)
#   X/lib/libgfortran.so.3*     from PAGO ~/Download/anaconda3/pkgs/libgfortran-3.0.0-1/lib
#   X/imsil.env                 HELVETICA_DIR only (replaces PocketQuake/.env, which holds NECIS/STP credentials — NOT copied)
#   I/data/catalogs/kma_entire_catalog_260309.csv          moho 07.SeismoStats/catalog (KST)
#   I/data/stations/taup/kim1983_buan.{npz,tvel}           moho 27.Haenam_sequence/data/stations/taup
#   I/data/stations/JU_station.xml                         moho 27.Haenam_sequence/JNU_data (lets jn_magnitudes import; 0 JU stations used)
#   I/data/stations/responses/master/KS_KG_metadata_1.0.2.xml   PAGO copy (35.240612_Buan_EQ), identical to Haenam's master
#   I/data/stations/responses/fetched/extracted/RESP.KS.*       I/kma_waveforms/RESP/2025007611.r (identical in all 24 event dirs)
#   X/PROVENANCE.tsv            item, source, commit, sha256/md5, bytes, date;  X/provenance/*.diff = moho `git diff` of each tree
set -euo pipefail
run=0; [[ "${1:-}" == "run" ]] && run=1
I=/home/msseo/works/72.2025_Imsil_swarm
X=$I/_external
M=<moho-host>                                   # moho (PAGO cannot resolve the name)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
RS="rsync -a --info=stats1 -e"
PROV=$X/PROVENANCE.tsv

say(){ echo "[$(date '+%H:%M:%S')] $*"; }
do_cmd(){ echo "    $*"; if [[ $run -eq 1 ]]; then eval "$@"; fi; }
prov(){ [[ $run -eq 1 ]] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$(date '+%Y-%m-%d %H:%M')" >> "$PROV"; true; }

say "preflight: moho reachable?"
$SSH $M true && echo "    ok" || { echo "    moho not reachable with BatchMode SSH"; exit 1; }
do_cmd "mkdir -p $X/bin $X/lib $X/provenance $I/data/catalogs $I/data/stations/taup $I/data/stations/responses/master $I/data/stations/responses/fetched/extracted"
if [[ $run -eq 1 && ! -f $PROV ]]; then printf 'item\tsource\tcommit\tchecksum\tbytes\tdate\n' > $PROV; fi

# ---------------------------------------------------------------- external code trees (moho)
copy_tree(){   # name  remote_dir  local_dir  excludes...
  local name=$1 src=$2 dst=$3; shift 3
  local ex=""; for e in "$@"; do ex="$ex --exclude='$e'"; done
  say "$name: $M:$src -> $dst"
  do_cmd "$RS \"$SSH\" $ex $M:$src/ $dst/"
  if [[ $run -eq 1 ]]; then
    local commit; commit=$($SSH $M "git -C $src rev-parse HEAD 2>/dev/null || echo none")
    $SSH $M "git -C $src status --short 2>/dev/null; echo; git -C $src diff -- . ':(exclude)*.ipynb' 2>/dev/null" > $X/provenance/$name.diff || true   # code diff only (notebooks are not copied)
    prov "$name" "moho:$src" "$commit" "tree (diff: provenance/$name.diff)" "$(du -sb $dst | cut -f1)"
  fi
}
copy_tree EQNet /home/msseo/works/14.EQNet/EQNet $X/EQNet \
  .git __pycache__ .ipynb_checkpoints 'docs/__MACOSX' 'docs/test_data.zip' 'docs/mseed_list.txt'
copy_tree korea-cluster-relocation /home/msseo/works/15.PocketQuake/external/korea-cluster-relocation $X/korea-cluster-relocation \
  '/*_cluster' '/pipeline/runs' '/pipeline/notebooks' '/pipeline/velocity_models' .git __pycache__ .ipynb_checkpoints
copy_tree fast_matched_filter /home/msseo/works/FMF/fast_matched_filter $X/fast_matched_filter \
  .git '*.pdf' '*.egg-info' '*.so' '*.o' .pytest_cache __pycache__
copy_tree HypoDD /home/msseo/Downloads/HypoDD $X/HypoDD \
  .git '*.o' 'src/hypoDD/hypoDD'

# ---------------------------------------------------------------- weights check
W=$X/EQNet/docs/model_phasenet_plus/model_99.pth
WANT=f1643330e34bffaa4f7e3a5f7dd9205fb4019d76e5b9b6cb598232360d22f384   # = Haenam logs/full_run_params.json weights_sha256
say "PhaseNet+ weights sha256 must equal $WANT"
if [[ $run -eq 1 ]]; then
  got=$(sha256sum $W | cut -d' ' -f1); [[ "$got" == "$WANT" ]] && echo "    ok" || { echo "    MISMATCH: $got"; exit 3; }
  prov "model_99.pth" "moho:/home/msseo/works/14.EQNet/EQNet/docs/model_phasenet_plus/model_99.pth" "-" "sha256:$got" "$(stat -c %s $W)"
fi

# ---------------------------------------------------------------- binaries
say "hypoDD v2.1beta binary (moho ~/bin/hypoDD, md5 e36d745c61a6f4508fd6c0dbab89e7c7)"
do_cmd "$RS \"$SSH\" $M:/home/msseo/bin/hypoDD $X/bin/hypoDD"
if [[ $run -eq 1 ]]; then
  got=$(md5sum $X/bin/hypoDD | cut -d' ' -f1); [[ "$got" == e36d745c61a6f4508fd6c0dbab89e7c7 ]] && echo "    md5 ok" || { echo "    md5 MISMATCH $got"; exit 3; }
  prov "hypoDD" "moho:/home/msseo/bin/hypoDD" "fb40407+diff" "md5:$got" "$(stat -c %s $X/bin/hypoDD)"
fi
say "libgfortran.so.3 shim + wrappers for PAGO ~/bin/{hyp1.40,ph2dt,ncsn2pha} (byte-identical to moho's)"
do_cmd "cp -a /home/msseo/Download/anaconda3/pkgs/libgfortran-3.0.0-1/lib/libgfortran.so.3* $X/lib/"
for t in hyp1.40 ph2dt ncsn2pha; do
  do_cmd "printf '#!/bin/sh\n# Imsil port: run PAGO ~/bin/$t with the libgfortran.so.3 shim (Rocky 10 ships only .so.5)\nLD_LIBRARY_PATH=$X/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} exec /home/msseo/bin/$t \"\$@\"\n' > $X/bin/$t && chmod +x $X/bin/$t"
  [[ $run -eq 1 ]] && prov "$t (wrapper)" "PAGO:/home/msseo/bin/$t" "-" "md5:$(md5sum /home/msseo/bin/$t | cut -d' ' -f1)" "$(stat -c %s /home/msseo/bin/$t)"
done
do_cmd "printf '# Imsil port: replaces 15.PocketQuake/.env for the relocation stage (no credentials are needed there)\nHELVETICA_DIR=/home/msseo/Download/Helvetica\n' > $X/imsil.env"

# ---------------------------------------------------------------- input tables
fetch(){   # item remote local
  say "$1: $M:$2 -> $3"; do_cmd "$RS \"$SSH\" $M:$2 $3"
  [[ $run -eq 1 ]] && prov "$1" "moho:$2" "-" "sha256:$(sha256sum $3 | cut -d' ' -f1)" "$(stat -c %s $3)"; true
}
fetch kma_entire_catalog /home/msseo/works/07.SeismoStats/catalog/kma_entire_catalog_260309.csv $I/data/catalogs/kma_entire_catalog_260309.csv
fetch taup_npz /home/msseo/works/27.Haenam_sequence/data/stations/taup/kim1983_buan.npz $I/data/stations/taup/kim1983_buan.npz
fetch taup_tvel /home/msseo/works/27.Haenam_sequence/data/stations/taup/kim1983_buan.tvel $I/data/stations/taup/kim1983_buan.tvel
fetch JU_station.xml /home/msseo/works/27.Haenam_sequence/JNU_data/JU_station.xml $I/data/stations/JU_station.xml
MASTER=/home/msseo/works/35.240612_Buan_EQ/5.P_wave_moment_tensor_inversion/Response/KS_KG_metadata_1.0.2.xml/KS_KG_metadata_1.0.2.xml
say "master StationXML (PAGO copy, identical to Haenam's master)"
do_cmd "cp -a $MASTER $I/data/stations/responses/master/KS_KG_metadata_1.0.2.xml"
[[ $run -eq 1 ]] && prov "KS_KG_metadata_1.0.2.xml" "PAGO:$MASTER" "-" "sha256:$(sha256sum $MASTER | cut -d' ' -f1)" "$(stat -c %s $MASTER)"
say "KMA RESP files for KS (current epochs; identical in all 24 Imsil event dirs)"
do_cmd "cp -a $I/kma_waveforms/RESP/2025007611.r/RESP.KS.* $I/data/stations/responses/fetched/extracted/"
[[ $run -eq 1 ]] && prov "RESP.KS.* (1512 files)" "PAGO:$I/kma_waveforms/RESP/2025007611.r" "-" "md5(cat):$(cat $I/kma_waveforms/RESP/2025007611.r/RESP.KS.* | md5sum | cut -d' ' -f1)" "-"

say "rebuild catalogs/imsil_kma24.csv with the KMA cross-check (needs only the copied KMA catalog)"
if [[ $run -eq 1 ]]; then
  /home/msseo/Download/anaconda3/envs/obspy/bin/python $I/DL_detection/catalogs/build_imsil_kma24.py
else
  echo "    python $I/DL_detection/catalogs/build_imsil_kma24.py"
fi
[[ $run -eq 1 ]] && { say "done; provenance:"; column -t -s$'\t' $PROV | cut -c1-200; } || echo "(dry run — append 'run' to execute)"
