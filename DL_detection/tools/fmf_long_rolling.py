#!/usr/bin/env python
"""Long-period FMF scan of the Imsil source region, 2018-01-01 .. 2026-10-03 (decisions D25, D26; 2026-10-11).

The same 28 templates and detector as the adopted catalog (aligned code, 5-20 Hz, 10 x MAD per configuration segment, >= 4 stations), with
the long-period station list (fmf_aligned/stations_imsil_5_20H_long.csv: S on Z at the 9 stations whose response epoch changes inside the
period). The data do not fit on disk at once (~1.6 TB for 22 stations; 1.2 TB free), so the period is processed in chunks:
  pull   one chunk of day files from MARA (only the 22 FMF stations, only the template's sensor: HH/HG/EL as in the template npz),
         rsync -aL into the mirror tree ../continuous_mara/<YYYY>/<NET>/<STA>/<CHA>.D/ (same layout as tools/mirror_mara.py);
  link   the files into continuous/<NET>/<STA>/<CHA>.D/ (the layout tools/link_continuous.py builds), so the scan code reads them unchanged;
  scan   one day per fresh process (the PAGO loader rule of tools/fmf_scan_days.sh): fmf_aligned/run_imsil_5_20H_long.sh scan Y D D;
  delete the chunk's links and files once every day of it has a daily result. Days of the adopted period (2025 DOY 244-365, 2026 DOY 1-31)
         are never pulled or deleted: they are scanned first, from the existing mirror, as the control.
The next chunk is pulled while the current one is scanned. State (resumable): ../continuous_mara/_manifest_long/state.json.
DRY RUN BY DEFAULT (prints the chunk plan and remote file counts); --apply executes.
    python tools/fmf_long_rolling.py plan                 # chunk table + remote inventory (read-only)
    python tools/fmf_long_rolling.py run --apply          # the whole period (detach with setsid nohup)
    python tools/fmf_long_rolling.py status               # state per chunk
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FA = os.path.join(ROOT, "fmf_aligned")
MARA = "<user>@<mara-host>"
SDS = "<mara-sds-root>"
LOCAL = os.path.abspath(os.path.join(ROOT, "..", "continuous_mara"))
CONT = os.path.join(ROOT, "continuous")
STATE_DIR = os.path.join(LOCAL, "_manifest_long")
KEEP = os.path.join(LOCAL, "_keep_long")        # day files of days with raw hits, kept (hard links) for waveform vetting
LOG = os.path.join(ROOT, "logs", "fmf_long_rolling.log")
FMF_RUN = "imsil_5_20H_aligned_mad10_long"
DETECT_DIR = os.path.join(FA, "catalogs", f"daily_shelly_{FMF_RUN}_seg")
SHELLY_NPZ = os.path.join(FA, "templates_cache", f"imsil_shelly_{FMF_RUN}.npz")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ControlMaster=auto",
            "-o", "ControlPath=/tmp/mara-cm-%r@%h:%p", "-o", "ControlPersist=900"]
BWLIMIT = 12500            # kB/s per rsync job: 8 jobs x 12.5 MB/s stay near the ~100 MB/s link (tools/mirror_mara.py: 2 x 40 MB/s)
JOBS = 8                   # concurrent rsync jobs, each on its OWN ssh connection (2026-10-12: two jobs multiplexed through one
                           # ControlMaster connection fell to 0.2 MB/s for hours while fresh connections gave 2.5-5 MB/s each)
RSYNC_SSH = "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
STALL_WINDOW = 600         # s: an rsync job whose transfer rate stays below MIN_RATE over this window is restarted ...
MIN_RATE = 0.5e6           # bytes/s
MAX_TRIES = 30             # ... up to this many times (rsync's quick check skips the files already complete)
MIN_FREE_GB = 300          # never start a pull with less free space than this
PROTECTED = {2025: (244, 365), 2026: (1, 31)}          # the adopted period: never pulled, never deleted
CHUNKS = ([("control-2025", 2025, 244, 365, False), ("control-2026", 2026, 1, 31, False)]
          + [(str(y), y, 1, 366 if y in (2020, 2024) else 365, True) for y in range(2018, 2025)]
          + [("2025a", 2025, 1, 243, True), ("2026b", 2026, 32, 276, True)])


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def protected(y, d):
    return y in PROTECTED and PROTECTED[y][0] <= d <= PROTECTED[y][1]


def stations_and_prefixes():
    import numpy as np
    z = np.load(SHELLY_NPZ, allow_pickle=True)
    return [(str(s).split(".")[0], str(s).split(".")[1], str(c)) for s, c in zip(z["stations"], z["channels"])]


def load_state():
    p = os.path.join(STATE_DIR, "state.json")
    return json.load(open(p)) if os.path.exists(p) else {}


_STATE_LOCK = threading.Lock()


def save_state(state):
    with _STATE_LOCK:
        os.makedirs(STATE_DIR, exist_ok=True)
        p = os.path.join(STATE_DIR, "state.json"); json.dump(state, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)


def remote_files(y, d0, d1):
    """relpaths <YYYY>/<NET>/<STA>/<CHA>.D/<file> on MARA for the 22 stations' template sensors, DOY d0..d1."""
    sp = stations_and_prefixes()
    dirs = " ".join(f"{y}/{n}/{s}/{p}{c}.D" for n, s, p in sp for c in "ENZ")
    cmd = f"cd {SDS} && for d in {dirs}; do [ -d $d ] && ls $d | sed \"s|^|$d/|\"; done"
    r = subprocess.run(["ssh", *SSH_OPTS, MARA, cmd], capture_output=True, text=True, timeout=1800)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"remote listing failed: {r.stderr.strip()[:300]}")
    out = []
    for rel in r.stdout.split():
        try:
            doy = int(rel.rsplit(".", 1)[1])
        except ValueError:
            continue
        if d0 <= doy <= d1 and not protected(y, doy):
            out.append(rel)
    return sorted(out)


def free_gb():
    return shutil.disk_usage(LOCAL).free / 1e9


def pull(label, y, d0, d1, state, apply):
    st = state.setdefault(label, {})
    if st.get("pulled"):
        return
    rels = remote_files(y, d0, d1)
    st["n_files"] = len(rels); save_state(state)
    log(f"[pull {label}] {len(rels)} files on MARA ({y} DOY {d0}-{d1})")
    if not apply or not rels:
        return
    while free_gb() < MIN_FREE_GB:
        log(f"[pull {label}] waiting: {free_gb():.0f} GB free < {MIN_FREE_GB}"); time.sleep(600)
    os.makedirs(STATE_DIR, exist_ok=True)
    lists = [[] for _ in range(JOBS)]
    for i, rel in enumerate(rels):          # files interleaved across jobs (2026-10-12: NAWB 2019 sits on a slow MARA NFS volume, ~0.5 MB/s for one
        lists[i % JOBS].append(rel)         # sequential stream, ~12 MB/s for six parallel reads; per-station jobs left one job hours behind)
    t0 = time.time()

    def dir_bytes(dirs):
        n = 0
        for d in dirs:
            try:
                n += sum(e.stat(follow_symlinks=False).st_size for e in os.scandir(d) if e.is_file(follow_symlinks=False))
            except FileNotFoundError:
                pass
        return n

    def job(k):
        lst = os.path.join(STATE_DIR, f"files_{label}_{k}.txt"); open(lst, "w").write("\n".join(lists[k]) + "\n")
        dirs = sorted({os.path.join(LOCAL, os.path.dirname(r)) for r in lists[k]}); code = None
        for attempt in range(1, MAX_TRIES + 1):
            cmd = ["rsync", "-aL", "--partial", f"--bwlimit={BWLIMIT}", "-e", RSYNC_SSH, f"--files-from={lst}", f"{MARA}:{SDS}/", f"{LOCAL}/"]
            pr = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); hist = [(time.time(), dir_bytes(dirs))]
            while pr.poll() is None:
                time.sleep(60); hist.append((time.time(), dir_bytes(dirs)))
                old = [h for h in hist if hist[-1][0] - h[0] >= STALL_WINDOW]
                if old and (hist[-1][1] - old[-1][1]) / (hist[-1][0] - old[-1][0]) < MIN_RATE:
                    rate = (hist[-1][1] - old[-1][1]) / (hist[-1][0] - old[-1][0])
                    log(f"[pull {label}] job {k}: {rate / 1e6:.2f} MB/s over {STALL_WINDOW} s - restarting rsync (attempt {attempt})")
                    pr.terminate()
                    try:
                        pr.wait(60)
                    except subprocess.TimeoutExpired:
                        pr.kill(); pr.wait()
                    break
            code = pr.returncode
            if code == 0:
                return 0
            time.sleep(30)
        return code
    with ThreadPoolExecutor(JOBS) as ex:
        codes = list(ex.map(job, range(JOBS)))
    missing = [r for r in rels if not os.path.isfile(os.path.join(LOCAL, r)) or os.path.getsize(os.path.join(LOCAL, r)) == 0]
    nbytes = sum(os.path.getsize(os.path.join(LOCAL, r)) for r in rels if r not in missing)
    st.update(pulled=not missing and all(c == 0 for c in codes), bytes=nbytes, rsync_codes=codes, missing=len(missing), pull_s=round(time.time() - t0))
    json.dump(rels, open(os.path.join(STATE_DIR, f"files_{label}.json"), "w"))
    save_state(state)
    log(f"[pull {label}] {nbytes / 1e9:.1f} GB in {time.time() - t0:.0f} s; rsync exit {codes}; missing {len(missing)}")


def link(label, state):
    rels = json.load(open(os.path.join(STATE_DIR, f"files_{label}.json")))
    n = 0
    for rel in rels:
        y, net, sta, cha, fn = rel.split("/")
        dst = os.path.join(CONT, net, sta, cha, fn)
        if not os.path.isfile(os.path.join(LOCAL, rel)):
            continue
        if not os.path.lexists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True); os.symlink(os.path.join(LOCAL, rel), dst); n += 1
    log(f"[link {label}] {n} new links")


def scan(label, y, d0, d1, state, apply):
    st = state.setdefault(label, {}); done, failed, nodata = 0, [], []
    for d in range(d0, d1 + 1):
        out = os.path.join(DETECT_DIR, f"det_{y}_{d:03d}.parquet")
        if os.path.exists(out):
            done += 1; continue
        if not apply:
            continue
        has = glob.glob(os.path.join(CONT, "*", "*", "*.D", f"*.D.{y}.{d:03d}"))
        if not has:
            nodata.append(d); continue
        r = subprocess.run(["./run_imsil_5_20H_long.sh", "scan", str(y), str(d), str(d)], cwd=FA, capture_output=True, text=True)
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if "DOY" in ln or "rror" in ln]
        for ln in lines[-3:]:
            log(f"[scan {label}] {ln.strip()}")
        if os.path.exists(out):
            done += 1
        else:
            failed.append(d); log(f"[scan {label}] {y} DOY {d:03d}: no result (exit {r.returncode})")
    st.update(days_done=done, days_failed=failed, days_nodata=nodata, scanned=apply and not failed)
    save_state(state)
    log(f"[scan {label}] {done} days with results; failed {failed}; no data {len(nodata)} days")


def delete(label, y, state):
    st = state[label]
    rels = json.load(open(os.path.join(STATE_DIR, f"files_{label}.json")))
    import pandas as pd
    keep = set()                            # days with any raw hit keep their files (hard links under KEEP) for waveform vetting
    for f in glob.glob(os.path.join(DETECT_DIR, f"det_{y}_*.parquet")):
        try:
            if len(pd.read_parquet(f)):
                keep.add(int(os.path.basename(f)[9:12]))
        except Exception:
            pass
    n = 0; nk = 0
    for rel in rels:
        yy, net, sta, cha, fn = rel.split("/")
        doy = int(fn.rsplit(".", 1)[1])
        if protected(int(yy), doy):
            raise SystemExit(f"refusing to delete a protected day: {rel}")
        if doy in keep and os.path.isfile(os.path.join(LOCAL, rel)) and not os.path.exists(os.path.join(KEEP, rel)):
            os.makedirs(os.path.dirname(os.path.join(KEEP, rel)), exist_ok=True); os.link(os.path.join(LOCAL, rel), os.path.join(KEEP, rel)); nk += 1
        dst = os.path.join(CONT, net, sta, cha, fn)
        if os.path.islink(dst) and os.readlink(dst) == os.path.join(LOCAL, rel):
            os.remove(dst)
        f = os.path.join(LOCAL, rel)
        if os.path.isfile(f):
            os.remove(f); n += 1
    st["deleted"] = True; st["kept_days"] = sorted(keep); save_state(state)
    log(f"[delete {label}] {n} files and their links removed; {nk} files of {len(keep)} days with hits kept under {KEEP}; {free_gb():.0f} GB free")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "status"]); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    state = load_state()
    if a.cmd == "status":
        for label, *_ in CHUNKS:
            print(label, {k: v for k, v in state.get(label, {}).items() if k not in ('days_failed', 'days_nodata') or v})
        total = sum(d1 - d0 + 1 for _, _, d0, d1, _ in CHUNKS); done = len(glob.glob(os.path.join(DETECT_DIR, 'det_*.parquet')))
        last = open(LOG).read().strip().splitlines()[-1] if os.path.exists(LOG) else ''
        print(f'days scanned {done} / {total} ({done / total:.0%}); {free_gb():.0f} GB free; remaining ~{(total - done) * 62 / 3600:.0f} h of scanning at 62 s/day')
        print('last log line:', last)
        return
    if a.cmd == "plan":
        log(f"[plan] {len(CHUNKS)} chunks; FMF_RUN={FMF_RUN}; local {LOCAL}; {free_gb():.0f} GB free")
        for label, y, d0, d1, p in CHUNKS:
            n = len(remote_files(y, d0, d1)) if p else 0
            print(f"  {label:13s} {y} DOY {d0:3d}-{d1:3d}  {'pull ' + str(n) + ' files' if p else 'local (adopted period, control)'}")
        return
    if not a.apply:
        print("(dry run: add --apply)"); return
    log(f"[run] start; {free_gb():.0f} GB free")
    ex = ThreadPoolExecutor(1); fut = {}
    todo = [c for c in CHUNKS if not state.get(c[0], {}).get("deleted") and not (not c[4] and state.get(c[0], {}).get("scanned"))]
    for k, (label, y, d0, d1, p) in enumerate(todo):
        if p and label not in fut:
            fut[label] = ex.submit(pull, label, y, d0, d1, state, True)
        nxt = next((c for c in todo[k + 1:] if c[4]), None)
        if p:
            fut[label].result()
            if not state[label].get("pulled"):
                log(f"[run] {label}: pull incomplete ({state[label].get('missing')} missing) - scanning what arrived, data kept")
            link(label, state)
        if nxt is not None and nxt[0] not in fut:
            fut[nxt[0]] = ex.submit(pull, nxt[0], nxt[1], nxt[2], nxt[3], state, True)
        scan(label, y, d0, d1, state, True)
        if p and state[label].get("scanned") and state[label].get("pulled"):
            delete(label, y, state)
    ex.shutdown(wait=True)
    log("[run] done")


if __name__ == "__main__":
    main()
