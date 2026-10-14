#!/usr/bin/env bash
# S0 smoke tests of the Imsil DL-catalog infrastructure on PAGO (after tools/phase0_copy_from_moho.sh run and
# envs/build_envs.sh all run). Read-only for the project: every test works in a scratch directory under /tmp.
# Prints PASS/FAIL per test and writes logs/smoke_envs_<date>.txt (shown in notebook 00).
#   ./tools/smoke_envs.sh            (no dry-run mode needed: nothing outside the scratch directory is written,
#                                     except one temporary directory on moho for test S6, removed afterwards)
set -uo pipefail
I=/home/msseo/works/72.2025_Imsil_swarm
X=$I/_external
D=$I/DL_detection
MF=/home/msseo/miniforge3
PYB=$MF/envs/dl-base/bin/python; PYQ=$MF/envs/pq-gpu/bin/python; PYF=$MF/envs/fmf/bin/python
export PATH=$X/bin:$PATH HELVETICA_DIR=/home/msseo/Download/Helvetica
S=$(mktemp -d /tmp/imsil_smoke.XXXX)
mkdir -p $D/logs; LOG=$D/logs/smoke_envs_$(date +%Y%m%d_%H%M).txt
npass=0; nfail=0
t(){ local name=$1; shift; echo "=== $name" | tee -a $LOG; if ( "$@" ) >> $LOG 2>&1; then echo "    PASS" | tee -a $LOG; npass=$((npass+1)); else echo "    FAIL (see $LOG)" | tee -a $LOG; nfail=$((nfail+1)); fi; }

s1(){ for py in $PYB $PYQ; do $py -c "
import torch; assert torch.cuda.is_available(), 'no CUDA'; n=torch.cuda.get_device_name(0); a=torch.randn(4096,4096,device='cuda'); b=(a@a).sum().item()
print('$py', torch.__version__, 'cuda', torch.version.cuda, n, 'matmul ok')" || return 1; done; }

s2(){ cd $D && $PYB -c "
import sys, hashlib; sys.path.insert(0,'pipeline'); import config, core
h=hashlib.sha256(open(config.EQNET_WEIGHTS,'rb').read()).hexdigest(); print('weights', h)
assert h=='f1643330e34bffaa4f7e3a5f7dd9205fb4019d76e5b9b6cb598232360d22f384'
net=core._load_pnplus('cuda'); print('PhaseNet+ loaded:', type(net).__name__, sum(p.numel() for p in net.parameters()), 'parameters')"; }

s3(){ cd $D && $PYB -c "
import sys, tempfile, os, pandas as pd, pyocto; sys.path.insert(0,'pipeline'); import config
print('pyocto', pyocto.__version__)
with tempfile.TemporaryDirectory() as td:
    mp=os.path.join(td,'vm'); pyocto.VelocityModel1D.create_model(pd.read_csv(config.VELOCITY_CSV),1.0,100,100,mp)
    vm=pyocto.VelocityModel1D(mp,tolerance=1.0); a=pyocto.OctoAssociator.from_area(velocity_model=vm,**config.REGION_STRICT)
    print('associator ok, REGION_STRICT', config.REGION_STRICT)"; }

s4(){ which hyp1.40 | grep -q "$X/bin" || { echo "hyp1.40 wrapper not first on PATH"; return 1; }
      cp -r $I/1.HypoInv $S/hypoinv && cd $S/hypoinv && rm -f kim2011/Imsil.sum kim2011/Imsil.prt kim2011/Imsil.arc && sh Imsil.sh > run.log 2>&1
      cmp kim2011/Imsil.sum $I/1.HypoInv/kim2011/Imsil.sum && tail -1 kim2011/Imsil.sum; }

s5(){ cp -r $X/HypoDD/examples/example1 $S/ex1 && cd $S/ex1 && rm -f hypoDD.reloc hypoDD.loc hypoDD.res hypoDD.sta hypoDD.src && $X/bin/hypoDD hypoDD.inp > run.log 2>&1
      ssh -o BatchMode=yes <moho-host> 'T=$(mktemp -d /tmp/imsil_hypodd_ex1.XXXX); cp -r /home/msseo/Downloads/HypoDD/examples/example1/. $T/ && cd $T && rm -f hypoDD.reloc && /home/msseo/bin/hypoDD hypoDD.inp >/dev/null 2>&1; cat hypoDD.reloc; rm -rf $T' > moho.reloc 2>/dev/null
      # Same binary, but glibc libm differs (moho Rocky 8, PAGO Rocky 10), so results agree to rounding, not byte for byte
      # (tested 2026-10-07: max 0.7 m; even moho's own libgfortran/libquadmath on PAGO did not change that). Tolerance 1 m, 0.01 s.
      [[ -s moho.reloc ]] && $PYB -c "
import numpy as np; a=np.loadtxt('hypoDD.reloc'); b=np.loadtxt('moho.reloc'); assert a.shape==b.shape and (a[:,0]==b[:,0]).all()
dxyz=np.abs(a[:,4:7]-b[:,4:7]).max(0); dt=np.abs(a[:,15]-b[:,15]).max()
print(f'{len(a)} events; max |dx,dy,dz| (m) {dxyz}; max |d origin| {dt:.3f} s; byte-identical: {open(\"hypoDD.reloc\").read()==open(\"moho.reloc\").read()}')
assert dxyz.max()<=1.0 and dt<=0.01"; }

s6(){ cd $D && $PYQ -c "
import sys; sys.path.insert(0,'reloc'); import imsil_dl_cluster_config as icc
from pipeline.core import pipeline, hypodd, xcorr, evmap
from pipeline.clusters._base import STD_PH2DT
cfg=icc.make_config(output_root='$S/kcr_cfg', open_wdcc=True); print('KCR config ok:', cfg.name, cfg.region_bounds, cfg.xcorr_backend)
print('STD_PH2DT', STD_PH2DT)"; }

s7(){ $PYF -c "
import numpy as np, fast_matched_filter.fast_matched_filter as f
print('GPU_LOADED', f.GPU_LOADED, 'CPU_LOADED', f.CPU_LOADED); assert f.GPU_LOADED and f.CPU_LOADED
rng=np.random.default_rng(0); ns,nc,nt=3,3,150; data=rng.standard_normal((ns,nc,20000)).astype(np.float32)
tpl=data[:,:,5000:5000+nt].copy()[None]; mo=np.zeros((1,ns,nc),np.int32); w=np.ones((1,ns,nc),np.float32)/(ns*nc)
cg=f.matched_filter(tpl,mo,w,data,1,arch='gpu',normalize='short'); cc=f.matched_filter(tpl,mo,w,data,1,arch='cpu',normalize='short')
print('peak', int(np.argmax(cg[0])), float(cg[0].max()), 'max|gpu-cpu|', float(np.nanmax(np.abs(cg-cc))))
assert int(np.argmax(cg[0]))==5000 and cg[0].max()>0.999 and np.nanmax(np.abs(cg-cc))<1e-4"; }

s8(){ cd $D && $PYB -c "
import sys; sys.path.insert(0,'pipeline'); import plotstyle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, pygmt
print('font', matplotlib.rcParams['font.family']); assert 'Helvetica' in matplotlib.rcParams['font.family']
fig=pygmt.Figure(); fig.basemap(region=[126.5,128.1,34.85,36.25], projection='M8c', frame=['af']); fig.savefig('$S/basemap.png'); print('pygmt', pygmt.__version__, 'ok')"; }

t "S1 GPU visible + matmul (dl-base, pq-gpu)" s1
t "S2 PhaseNet+ weights sha256 + model load (dl-base)" s2
t "S3 PyOcto velocity model + REGION_STRICT associator (dl-base)" s3
t "S4 hyp1.40 wrapper reproduces 1.HypoInv/kim2011/Imsil.sum byte-for-byte" s4
t "S5 hypoDD v2.1beta example1 on PAGO = moho within 1 m / 0.01 s" s5
t "S6 korea-cluster-relocation imports + Imsil cluster config (pq-gpu)" s6
t "S7 fast_matched_filter GPU/CPU loaded, synthetic self-detection, GPU = CPU (fmf)" s7
t "S8 Helvetica + PyGMT render (dl-base)" s8
echo "RESULT: $npass passed, $nfail failed   (log $LOG; scratch $S)" | tee -a $LOG
exit $nfail
