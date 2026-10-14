"""Progress of the full-period FMF scan: per-year completion, rate, ETA, process health, detections so far.
Usage: ./run_full_period.sh progress   (or: python progress.py; no env needed)"""
import glob, os, subprocess, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); import sys; sys.path.insert(0, HERE); import config as _C; D = _C.DETECT_DIR
YEARS = {2020: 366, 2021: 365, 2022: 365, 2023: 365, 2024: 366, 2025: 365, 2026: 261}; TOTAL = sum(YEARS.values())
fs = sorted(glob.glob(f"{D}/det_*.parquet"), key=os.path.getmtime); mt = np.array([os.path.getmtime(f) for f in fs]); now = time.time()
print(f"FMF full-period scan  ({time.strftime('%Y-%m-%d %H:%M:%S KST')})")
print("year  done/days   %    bar")
done = {y: 0 for y in YEARS}
for f in fs: done[int(os.path.basename(f)[4:8])] += 1
for y, n in YEARS.items():
    k = done[y]; bar = "#" * int(40 * k / n) + "." * (40 - int(40 * k / n)); print(f"{y}  {k:4d}/{n:<4d} {100*k/n:5.1f}%  {bar}")
print(f"total {len(fs)}/{TOTAL} days ({100*len(fs)/TOTAL:.1f}%)")
for w in (1800, 3600):
    k = int((mt > now - w).sum()); print(f"last {w//60:2d} min: {k:3d} days -> {w/max(k,1):5.0f} s/day aggregate" + (f" -> ETA {(TOTAL-len(fs))*w/k/3600:5.1f} h" if k else ""))
if len(mt): print(f"newest day file {now-mt.max():.0f} s ago")
ps = subprocess.run(["bash", "-c", "ps -eo pid,etime,pcpu,rss,args | grep 'shelly.py 20' | grep -v grep | awk '$6==\"shelly.py\"{printf \"  pid %s up %s cpu %s%% rss %.1f GB year %s\\n\",$1,$2,$3,$4/1e6,$7}'"], capture_output=True, text=True).stdout
print("scan processes:\n" + (ps or "  none running"))
st = os.path.join(os.environ.get("FMF_LOG_DIR", f"{HERE}/logs" if _C.FMF_RUN in ("", "full_period") else f"{HERE}/logs/{_C.FMF_RUN}"), "batch_status.txt")   # as run_full_period.sh sets LOG
if os.path.exists(st): print("status file (last 3 lines):"); print("".join(open(st).readlines()[-3:]).rstrip())
raw = sum(len(pd.read_parquet(f)) for f in fs[-50:]) if fs else 0; print(f"raw 12xMAD detections in the last {min(50,len(fs))} scanned days: {raw}")
cat = _C.CATALOG_CSV
if os.path.exists(cat):
    c = pd.read_csv(cat)   # final catalogue (decision 20, 2026-10-06): reference event or new; as-run catalogues: self / in DL / new
    cls = (f"reference {int(c.is_reference.sum())}, new {int(c.is_new.sum())}; self-detections {int(c.self_detection.sum())}" if "is_reference" in c
           else f"self {int(c.self_detection.sum())}, in DL {int((~c.self_detection & c.in_DL).sum())}, new {int(c.is_new.sum())}")
    print(f"last catalog build: {len(c)} declustered events ({cls}) from {os.path.getmtime(cat) and time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(cat)))}; refresh with: ./run_full_period.sh catalog")
