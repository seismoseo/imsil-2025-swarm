#!/usr/bin/env python
"""Mirror the Imsil 100-km KS continuous data from the MARA SDS archive (IMSIL PORT of the Haenam tool).

IMSIL PORT: LOCAL, EPICENTER and YEARS_DEFAULT changed, and a PERIOD filter (day-of-year window per year) was added
in load_remote() so that `plan` writes worklists/manifest rows only for the Imsil detection period
(2025 DOY 244-365 + 2026 DOY 1-31) instead of whole years; `list` still inventories whole years (cheap).
KG is kept in NETWORKS (MARA has no KG for 2025-2026, so it contributes nothing).


MARA (<user>@<mara-host>) holds <mara-sds-root>/<YYYY>/<NET>/<STA>/<CHA>.D/
<NET>.<STA>..<CHA>.D.<YYYY>.<DOY>  (the day files are symlinks into /nfs/anystor/...; rsync dereferences them with -L).
The local mirror keeps the SAME layout under LOCAL, so relative paths map 1:1 and tools/link_continuous.py links it
with one regex.  Only ONE sensor per station-day is fetched: the pipeline's own channel policy
(pipeline/config.py CHANNEL_POLICY, HH > EL > HG) is applied to the remote listing, so BG (20 Hz), BH, BA, HA, SL, DE
channels are never transferred and what lands on disk is exactly what the picker would use.

Subcommands (DRY RUN by default; add --apply to execute):
  list    [--years 2020-2026] [--relist]      remote inventory per (year, net) -> _manifest/remote_<Y>_<NET>.tsv (cached)
  plan    [--years ...]                        channel policy -> _manifest/worklist_<Y>.csv + _manifest/manifest.csv, GB per year
  pull    [--years ...] [--stations A,B] [--jobs 2] [--bwlimit KBPS]   rsync one job per (year, net, sta); resumable
  verify  [--years ...]                        local files vs worklist (name + size) -> manifest status
  status  [--years ...] [--year Y --assert-complete]                    print the manifest; exit 1 if a year is incomplete

Typical:  python tools/mirror_mara.py list --years 2020-2026 --apply
          python tools/mirror_mara.py plan --years 2020-2026
          python tools/mirror_mara.py pull --years 2020 --apply          (launch detached; idempotent, resumable)
          python tools/mirror_mara.py status --year 2020 --assert-complete
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- parameters (edit here)
MARA = "<user>@<mara-host>"
SDS = "<mara-sds-root>"
LOCAL = "/home/msseo/works/72.2025_Imsil_swarm/continuous_mara"            # IMSIL PORT: path
MANIFEST_DIR = os.path.join(LOCAL, "_manifest")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIONS_CSV = os.path.join(ROOT, "station_table", "stations.csv")
NETWORKS = ("KS", "KG")
YEARS_DEFAULT = "2025-2026"                                               # IMSIL PORT
# IMSIL PORT: detection period (user decision 2026-10-07: Sep 2025 - Jan 2026). Years absent here are taken whole.
PERIOD = {2025: (244, 365), 2026: (1, 31)}
JOBS_DEFAULT = 2                        # concurrent rsync jobs (one per station-year)
BWLIMIT_DEFAULT = 40000                 # kB/s per job (2 jobs ~ 80 MB/s, below the measured ~100 MB/s link)
EPICENTER = (35.5506, 127.2951)                                           # IMSIL PORT: = config.IMSIL_EPICENTER
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ControlMaster=auto",
            "-o", "ControlPath=/tmp/mara-cm-%r@%h:%p", "-o", "ControlPersist=900"]
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import config  # noqa: E402  (CHANNEL_POLICY, apply_channel_policy)


def parse_years(s):
    out = set()
    for tok in str(s).split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); out.update(range(int(a), int(b) + 1))
        elif tok:
            out.add(int(tok))
    return sorted(out)


def wanted_stations():
    st = pd.read_csv(STATIONS_CSV)
    st = st[st.Network.isin(NETWORKS)].copy()
    cl = np.cos(np.radians(EPICENTER[0]))
    st["dist_km"] = np.hypot((st.Longitude - EPICENTER[1]) * 111.19 * cl, (st.Latitude - EPICENTER[0]) * 111.19)
    return st.sort_values("dist_km").reset_index(drop=True)


def ssh(cmd, timeout=1800):
    r = subprocess.run(["ssh", *SSH_OPTS, MARA, cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and not r.stdout:
        raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr.strip()[:400]}")
    return r.stdout


# ------------------------------------------------------------------ list
def remote_list_path(y, net):
    return os.path.join(MANIFEST_DIR, f"remote_{y}_{net}.tsv")


def do_list(years, relist, apply):
    st = wanted_stations(); os.makedirs(MANIFEST_DIR, exist_ok=True)
    for y in years:
        for net in NETWORKS:
            out = remote_list_path(y, net)
            if os.path.exists(out) and not relist:
                print(f"  [list] {y} {net}: cached ({sum(1 for _ in open(out))} files)"); continue
            stas = " ".join(st[st.Network == net].Code)
            cmd = (f"cd {SDS}/{y}/{net} 2>/dev/null || exit 0; for s in {stas}; do [ -d $s ] || continue; "
                   f"find -L $s -mindepth 2 -maxdepth 2 -type f -printf '%p\\t%s\\n'; done; "
                   f"echo DANGLING $(for s in {stas}; do [ -d $s ] && find $s -mindepth 2 -maxdepth 2 -xtype l; done | wc -l)")
            if not apply:
                print(f"  [list] {y} {net}: would run remote find over {len(st[st.Network == net])} station dirs -> {out}"); continue
            t0 = time.time(); txt = ssh(cmd); lines = txt.strip().split("\n") if txt.strip() else []
            dangling = [l for l in lines if l.startswith("DANGLING")]; rows = [l for l in lines if l and not l.startswith("DANGLING")]
            with open(out, "w") as f:
                f.write("\n".join(rows) + ("\n" if rows else ""))
            print(f"  [list] {y} {net}: {len(rows)} files, {dangling[0] if dangling else 'DANGLING ?'}, {time.time() - t0:.0f} s -> {out}")


# ------------------------------------------------------------------ plan
def load_remote(years):
    rows = []
    for y in years:
        for net in NETWORKS:
            p = remote_list_path(y, net)
            if not os.path.exists(p):
                continue
            for line in open(p):
                line = line.rstrip("\n")
                if not line:
                    continue
                rel, size = line.split("\t"); sta, chad, fname = rel.split("/")
                parts = fname.split(".")
                if len(parts) != 7:
                    continue
                if y in PERIOD and not (PERIOD[y][0] <= int(parts[6]) <= PERIOD[y][1]):   # IMSIL PORT: period filter
                    continue
                if parts[3][:2] in getattr(config, "CHANNEL_EXCLUDE", {}).get((net, sta), []):   # IMSIL PORT: D17 exclusion
                    continue
                rows.append((y, net, sta, parts[3], parts[3][:2], int(parts[6]), int(size), f"{y}/{net}/{sta}/{chad}/{fname}", fname))
    return pd.DataFrame(rows, columns=["year", "net", "sta", "cha", "prefix", "doy", "bytes", "relpath", "fname"])


def worklist_path(y):
    return os.path.join(MANIFEST_DIR, f"worklist_{y}.csv")


def do_plan(years, apply):
    rem = load_remote(years)
    if rem.empty:
        print("  [plan] no remote listings yet — run `list --apply` first"); return
    keep = []
    for (y, net, sta, doy), g in rem.groupby(["year", "net", "sta", "doy"], sort=False):
        chosen = set(config.apply_channel_policy(net, list(g.fname)))
        keep.append(g[g.fname.isin(chosen)])
    wl = pd.concat(keep).sort_values(["year", "net", "sta", "doy", "cha"]).reset_index(drop=True)
    bad = wl[~wl.prefix.isin(sum((config.CHANNEL_POLICY[n] for n in NETWORKS), []))]
    assert bad.empty, f"channel policy leak: {bad.prefix.unique()}"
    dropped = rem[~rem.relpath.isin(wl.relpath)]
    for y in years:
        w = wl[wl.year == y]
        if len(w) == 0:
            continue
        if apply:
            w.to_csv(worklist_path(y), index=False)
        print(f"  [plan] {y}: {len(w)} files, {w.bytes.sum() / 1e9:.1f} GB, stations {w.sta.nunique()} "
              f"(KS {w[w.net == 'KS'].sta.nunique()}, KG {w[w.net == 'KG'].sta.nunique()}), last DOY {w.doy.max()}; "
              f"skipped by channel policy {len(dropped[dropped.year == y])} files")
    man = (wl.groupby(["year", "net", "sta", "prefix"], as_index=False)
             .agg(n_days=("doy", "nunique"), doy_first=("doy", "min"), doy_last=("doy", "max"), n_files=("relpath", "size"), gb=("bytes", lambda b: round(b.sum() / 1e9, 2))))
    man["n_local"] = 0; man["gb_local"] = 0.0; man["status"] = "pending"; man["updated"] = ""
    old = load_manifest()
    if old is not None:                                # keep statuses of rows already verified
        man = man.merge(old[["year", "net", "sta", "prefix", "n_local", "gb_local", "status", "updated"]], on=["year", "net", "sta", "prefix"], how="left", suffixes=("", "_old"))
        for c in ("n_local", "gb_local", "status", "updated"):
            man[c] = man[c + "_old"].where(man[c + "_old"].notna(), man[c]); man = man.drop(columns=[c + "_old"])
    if apply:
        man.to_csv(manifest_path(), index=False); print(f"  [plan] manifest -> {manifest_path()} ({len(man)} rows)")
    else:
        print("  (dry run — add --apply to write the worklists and manifest)")
    print(man.groupby(["year", "net"]).agg(stations=("sta", "nunique"), files=("n_files", "sum"), gb=("gb", "sum")).round(1).to_string())


def manifest_path():
    return os.path.join(MANIFEST_DIR, "manifest.csv")


def load_manifest():
    return pd.read_csv(manifest_path()) if os.path.exists(manifest_path()) else None


# ------------------------------------------------------------------ pull
def job_list_path(y, net, sta):
    d = os.path.join(MANIFEST_DIR, "lists"); os.makedirs(d, exist_ok=True); return os.path.join(d, f"{y}.{net}.{sta}.txt")


def run_job(y, net, sta, relpaths, bwlimit, apply):
    lst = job_list_path(y, net, sta)
    with open(lst, "w") as f:
        f.write("\n".join(relpaths) + "\n")
    logd = os.path.join(MANIFEST_DIR, "logs"); os.makedirs(logd, exist_ok=True); log = os.path.join(logd, f"{y}.{net}.{sta}.log")
    cmd = ["rsync", "-aL", "--partial", f"--bwlimit={bwlimit}", "--info=stats2", "-e", "ssh " + " ".join(SSH_OPTS),
           f"--files-from={lst}", f"{MARA}:{SDS}/", f"{LOCAL}/"]
    if not apply:
        return (y, net, sta, "dry", 0.0, " ".join(cmd))
    t0 = time.time(); rc = None
    for attempt in (1, 2):
        with open(log, "a") as lf:
            lf.write(f"\n[{time.strftime('%m-%d %H:%M')}] attempt {attempt}: {' '.join(cmd)}\n"); lf.flush()
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc == 0 or rc not in (23, 24):
            break
    return (y, net, sta, "ok" if rc == 0 else f"rc{rc}", time.time() - t0, log)


def do_pull(years, stations, jobs, bwlimit, apply):
    st = wanted_stations(); order = list(st.Code)
    tasks = []
    for y in years:
        if not os.path.exists(worklist_path(y)):
            print(f"  [pull] {y}: no worklist — run `plan --apply` first"); continue
        wl = pd.read_csv(worklist_path(y))
        for (net, sta), g in wl.groupby(["net", "sta"]):
            if stations and sta not in stations:
                continue
            tasks.append((y, net, sta, list(g.relpath), order.index(sta) if sta in order else 999))
    tasks.sort(key=lambda t: (t[0], t[4]))
    print(f"  [pull] {len(tasks)} station-year jobs, {jobs} concurrent, bwlimit {bwlimit} kB/s each{'' if apply else ' (DRY RUN)'}")
    os.makedirs(LOCAL, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(run_job, y, net, sta, rel, bwlimit, apply) for (y, net, sta, rel, _) in tasks]
        for fu in futs:
            y, net, sta, res, dt, info = fu.result(); done += 1
            if apply:
                verify_station(y, net, sta)
                print(f"  [pull] {done}/{len(tasks)} {y} {net} {sta}: {res} in {dt / 60:.1f} min", flush=True)
            elif done <= 3:
                print(f"    {info}")
    if apply:
        save_status_summary(years)


# ------------------------------------------------------------------ verify / status
def verify_station(y, net, sta):
    man = load_manifest()
    if man is None or not os.path.exists(worklist_path(y)):
        return
    wl = pd.read_csv(worklist_path(y)); w = wl[(wl.net == net) & (wl.sta == sta)]
    for prefix, g in w.groupby("prefix"):
        n_ok = 0; gb = 0.0
        for rp, b in zip(g.relpath, g.bytes):
            p = os.path.join(LOCAL, rp)
            if os.path.exists(p) and os.path.getsize(p) == b:
                n_ok += 1; gb += b / 1e9
        m = (man.year == y) & (man.net == net) & (man.sta == sta) & (man.prefix == prefix)
        status = "verified" if n_ok == len(g) else ("partial" if n_ok else "pending")
        man.loc[m, ["n_local", "gb_local", "status", "updated"]] = [n_ok, round(gb, 2), status, time.strftime("%Y-%m-%d %H:%M")]
    man.to_csv(manifest_path(), index=False)


def do_verify(years):
    man = load_manifest()
    if man is None:
        print("  [verify] no manifest"); return
    for y in years:
        for r in man[man.year == y][["net", "sta"]].drop_duplicates().itertuples():
            verify_station(y, r.net, r.sta)
    save_status_summary(years)


def save_status_summary(years):
    man = load_manifest()
    g = man[man.year.isin(years)].groupby(["year", "net"]).agg(stations=("sta", "nunique"), files=("n_files", "sum"), files_local=("n_local", "sum"),
                                                              gb=("gb", "sum"), gb_local=("gb_local", "sum"), verified=("status", lambda s: int((s == "verified").sum())), rows=("status", "size"))
    print(g.round(1).to_string())


def do_status(years, year_assert):
    man = load_manifest()
    if man is None:
        print("  [status] no manifest"); sys.exit(1)
    save_status_summary(years)
    if year_assert is not None:
        m = man[man.year == year_assert]
        bad = m[m.status != "verified"]
        if len(m) == 0 or len(bad):
            print(f"  [status] year {year_assert} NOT complete: {len(bad)} of {len(m)} station-prefix rows not verified"); sys.exit(1)
        print(f"  [status] year {year_assert} complete: {len(m)} station-prefix rows verified, {m.gb_local.sum():.1f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "plan", "pull", "verify", "status"])
    ap.add_argument("--years", default=YEARS_DEFAULT); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--relist", action="store_true"); ap.add_argument("--stations", default=None)
    ap.add_argument("--jobs", type=int, default=JOBS_DEFAULT); ap.add_argument("--bwlimit", type=int, default=BWLIMIT_DEFAULT)
    ap.add_argument("--year", type=int, default=None); ap.add_argument("--assert-complete", action="store_true")
    a = ap.parse_args(); years = parse_years(a.years)
    if a.cmd == "list":
        do_list(years, a.relist, a.apply)
    elif a.cmd == "plan":
        do_plan(years, a.apply)
    elif a.cmd == "pull":
        do_pull(years, set(a.stations.split(",")) if a.stations else None, a.jobs, a.bwlimit, a.apply)
    elif a.cmd == "verify":
        do_verify(years)
    else:
        do_status(years, a.year if a.assert_complete else None)


if __name__ == "__main__":
    main()
