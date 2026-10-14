#!/usr/bin/env bash
# Rebuild moho's three pipeline environments on PAGO (S0 of the Imsil DL-catalog port), in ~/miniforge3/envs.
# The miniforge base env is NOT touched. Versions come from moho's exported specs
# (~/works/_moho_reference/27.Haenam_sequence/docs/environment/*.yml, exported 2026-10-04); check them afterwards with
#   ~/miniforge3/bin/python envs/verify_envs.py
#
# DRY RUN BY DEFAULT — prints every command. Append the env name(s) and `run`:
#   ./envs/build_envs.sh dl-base run        # picker / association / location / catalog / ML / notebooks (moho: base)
#   ./envs/build_envs.sh pq-gpu run         # korea-cluster-relocation (HypoDD, GPU cross-correlation)
#   ./envs/build_envs.sh fmf run            # fast_matched_filter 1.5.0 built for sm_80 (needs _external/fast_matched_filter)
#   ./envs/build_envs.sh all run
#
# dl-base   moho's base env is the whole miniforge base (conda itself + ~280 pip packages), so it is NOT cloned; dl-base
#           pins the packages the pipeline imports to moho's exact versions: python 3.12.12, obspy 1.4.2, numpy 2.3.5,
#           pandas 2.3.3, scipy 1.17.0, h5py 3.15.1, numba 0.63.1, pyrocko 2025.12.04 (PyOcto create_model), fsspec
#           2026.1.0 (EQNet), torch 2.12.0 (+cu13 wheels, as moho), torchvision 0.27.0, seisbench 0.11.0, pyocto 0.1.10,
#           plus pygmt 0.18.0 / gmt 6.6.0 / matplotlib 3.10.8 / nbformat / nbclient / ipykernel for the notebooks.
# pq-gpu    exact replica: conda part of conda_pq-gpu.yml, then its pip part with --no-deps (moho's env pairs torch
#           2.11.0+cu128 with torchvision 0.21.0, which a dependency-resolving install would refuse).
# fmf       exact replica of conda_fmf.yml (incl. cuda-nvcc 12.8.93) + pip part with --no-deps, minus fastmatchedfilter,
#           which is built from _external/fast_matched_filter (Makefile targets sm_80/sm_90/sm_120) and installed -e.
set -euo pipefail
cd "$(dirname "$0")"
# Explicit -p prefixes: mamba 2.x would otherwise create '-n' envs under ~/.local/share/mamba.
# conda-forge (+ pyrocko) only, as moho: --override-channels ignores the 'defaults' (repo.anaconda.com) entry of ~/.condarc
ENVSPEC=/home/msseo/works/_moho_reference/27.Haenam_sequence/docs/environment
MF=/home/msseo/miniforge3
CONDA="$MF/bin/mamba"
FMF_SRC=/home/msseo/works/72.2025_Imsil_swarm/_external/fast_matched_filter
GEN=$(pwd)/generated; mkdir -p "$GEN"
run=0; targets=()
for a in "$@"; do [[ "$a" == "run" ]] && run=1 || targets+=("$a"); done
[[ ${#targets[@]} -eq 0 ]] && { sed -n 2,26p "$0"; exit 0; }
[[ " ${targets[*]} " == *" all "* ]] && targets=(dl-base pq-gpu fmf)
do_cmd(){ echo "    $*"; if [[ $run -eq 1 ]]; then eval "$@" || { echo "FAILED (exit $?): $*"; exit 1; }; fi; }
say(){ echo "[$(date '+%H:%M:%S')] $*"; }
exists(){ [[ -d $MF/envs/$1 ]]; }

split_spec(){   # yml -> generated/<env>.conda.yml (no pip block, no prefix) + generated/<env>.pip.txt
  local env=$1 yml=$2 drop=${3:-}
  awk -v env="$env" 'BEGIN{p=0} /^prefix:/{next} /^name:/{print "name: " env; next}
       /^  - pip:/{p=1; next} p==1 && /^      - /{next} {print}' "$yml" > "$GEN/$env.conda.yml"
  awk 'BEGIN{p=0} /^  - pip:/{p=1; next} p==1 && /^      - /{sub(/^      - /,""); print}' "$yml" \
      | { if [[ -n "$drop" ]]; then grep -viE "$drop"; else cat; fi; } > "$GEN/$env.pip.txt"
  echo "    split $yml -> $GEN/$env.conda.yml ($(grep -c '^  - ' $GEN/$env.conda.yml) conda pkgs), $GEN/$env.pip.txt ($(wc -l < $GEN/$env.pip.txt) pip pkgs)"
}

for t in "${targets[@]}"; do
  case "$t" in
    dl-base)
      say "dl-base (curated, moho base versions)"
      exists dl-base && { echo "    $MF/envs/dl-base exists — skip (remove it first to rebuild)"; continue; }
      cat > "$GEN/dl-base.constraints.txt" <<'EOF'
numpy==2.3.5
pandas==2.3.3
scipy==1.17.0
h5py==3.15.1
obspy==1.4.2
numba==0.63.1
llvmlite==0.46.0
matplotlib==3.10.8
EOF
      # moho split: these came from conda in moho base; numpy/pandas/h5py/numba/llvmlite/pyproj/fsspec came from pip (its export's
      # pip block). Pinning them in conda is unsolvable (conda obspy 1.4.2 wants sqlalchemy<2, conda pandas 2.3.3 wants >=2).
      do_cmd "$CONDA create -y --override-channels -p $MF/envs/dl-base -c conda-forge -c pyrocko python=3.12.12 obspy=1.4.2 scipy=1.17.0 matplotlib=3.10.8 pyrocko=2025.12.04 pygmt=0.18.0 gmt=6.6.0 tqdm=4.67.3 nbformat=5.10.4 nbclient=0.10.4 ipykernel=7.1.0 pip=25.3"
      do_cmd "$MF/envs/dl-base/bin/python -m pip install -c $GEN/dl-base.constraints.txt numpy==2.3.5 pandas==2.3.3 h5py==3.15.1 numba==0.63.1 llvmlite==0.46.0 pyproj==3.7.2 fsspec==2026.1.0 torch==2.12.0 torchvision==0.27.0 seisbench==0.11.0 pyocto==0.1.10 pyarrow==25.0.0"   # pyarrow: moho base pip_freeze (added 2026-10-08 for parquet reads in notebook 06)
      do_cmd "$MF/envs/dl-base/bin/python -m ipykernel install --user --name dl-base --display-name 'dl-base (Imsil DL)'"   # kernel of the Imsil notebooks
      ;;
    pq-gpu)
      say "pq-gpu (exact replica of moho conda_pq-gpu.yml)"
      split_spec pq-gpu "$ENVSPEC/conda_pq-gpu.yml" "^scikit-fmm=="
      if exists pq-gpu; then echo "    $MF/envs/pq-gpu exists — conda step skipped (pip steps still run)"
      else do_cmd "$CONDA env create -y -p $MF/envs/pq-gpu -f $GEN/pq-gpu.conda.yml --override-channels -c conda-forge"; fi
      do_cmd "$MF/envs/pq-gpu/bin/python -m pip install --no-deps -r $GEN/pq-gpu.pip.txt --extra-index-url https://download.pytorch.org/whl/cu128"
      # scikit-fmm 2025.6.23 has no cp310 binary wheel and PAGO has no C++ compiler: build the wheel in a throwaway env with
      # conda-forge's compiler, install it --no-deps (pq-gpu itself stays identical to moho), then delete the throwaway env.
      W=$GEN/skfmm_build
      do_cmd "rm -rf $W && $CONDA create -y --override-channels -p $W/benv -c conda-forge python=3.10.20 gxx_linux-64 pip"
      do_cmd "CC=$W/benv/bin/x86_64-conda-linux-gnu-gcc CXX=$W/benv/bin/x86_64-conda-linux-gnu-g++ $W/benv/bin/python -m pip wheel --no-deps scikit-fmm==2025.6.23 -w $W/wheelhouse"
      do_cmd "$MF/envs/pq-gpu/bin/python -m pip install --no-deps $W/wheelhouse/scikit_fmm-*.whl && rm -rf $W"
      ;;
    fmf)
      say "fmf (exact replica of moho conda_fmf.yml + fast_matched_filter @25e8263 built for sm_80)"
      [[ -f $FMF_SRC/Makefile ]] || { echo "    $FMF_SRC missing — run tools/phase0_copy_from_moho.sh run first"; [[ $run -eq 1 ]] && exit 1; }
      if ! exists fmf; then
        split_spec fmf "$ENVSPEC/conda_fmf.yml" "^fastmatchedfilter=="
        do_cmd "$CONDA env create -y -p $MF/envs/fmf -f $GEN/fmf.conda.yml --override-channels -c conda-forge"
        do_cmd "$MF/envs/fmf/bin/python -m pip install --no-deps -r $GEN/fmf.pip.txt"
      else echo "    $MF/envs/fmf exists — env step skipped (FMF build step still runs)"; fi
      # build with the env's nvcc 12.8; if it refuses PAGO's system gcc 14, retry with a conda gcc 13 host compiler
      do_cmd "cd $FMF_SRC && rm -f fast_matched_filter/lib/*.so && PATH=$MF/envs/fmf/bin:\$PATH make python_gpu python_cpu NVCC=$MF/envs/fmf/bin/nvcc || { $CONDA install -y --override-channels -p $MF/envs/fmf -c conda-forge gxx_linux-64=13 && PATH=$MF/envs/fmf/bin:\$PATH make python_gpu python_cpu NVCC=\"$MF/envs/fmf/bin/nvcc -ccbin $MF/envs/fmf/bin/x86_64-conda-linux-gnu-g++\"; }"
      do_cmd "cd $FMF_SRC && $MF/envs/fmf/bin/python -m pip install --no-deps -e ."
      ;;
    *) echo "unknown env '$t' (dl-base | pq-gpu | fmf | all)"; exit 1 ;;
  esac
done
[[ $run -eq 1 ]] && echo "next: $MF/bin/python $(pwd)/verify_envs.py" || echo "(dry run — append 'run' to execute)"
