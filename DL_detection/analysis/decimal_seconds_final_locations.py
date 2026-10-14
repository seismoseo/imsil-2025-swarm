"""Rewrite the origin_time column of existing final_locations.csv files with the relocated origin time to 0.001 s,
taken from the hypoDD.reloc of the same folder (cuspid -> event_id via event_manifest.csv in the folder or the run root).
Every other column is left byte-identical; a .whole_seconds.bak copy is written first. Checks that the new time agrees
with the old whole-second value within 1 s before writing.
Usage: python analysis/decimal_seconds_final_locations.py DIR [DIR ...]
"""
import os
import shutil
import sys

import pandas as pd

KCR = "/home/msseo/works/72.2025_Imsil_swarm/_external/korea-cluster-relocation"   # IMSIL PORT: PAGO copy
sys.path.insert(0, KCR)
from pipeline.core import sumio  # noqa: E402


def manifest_for(d):
    for cand in (os.path.join(d, "event_manifest.csv"), os.path.join(d, "..", "..", "event_manifest.csv")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"no event_manifest.csv for {d}")


def process(d):
    fl_path = os.path.join(d, "final_locations.csv")
    rl = sumio.read_reloc(os.path.join(d, "hypoDD.reloc"))
    mf = pd.read_csv(manifest_for(d), dtype={"event_id": str})
    cus = {200000 + int(r.event_idx): r.event_id for r in mf.itertuples()}
    t = pd.to_datetime(dict(year=rl.yr, month=rl.mo, day=rl.dy, hour=rl.hr, minute=rl.mi)) + pd.to_timedelta(rl.sc.astype(float), unit="s")
    new = dict(zip(rl.id.astype(int).map(cus), t.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]))
    fl = pd.read_csv(fl_path, dtype=str, keep_default_na=False)
    old_t = pd.to_datetime(fl.origin_time); new_t = pd.to_datetime(fl.event_id.map(new))
    assert new_t.notna().all(), f"{d}: {int(new_t.isna().sum())} events not in hypoDD.reloc"
    dmax = (new_t - old_t).abs().max()
    assert dmax <= pd.Timedelta(seconds=1.0), f"{d}: new and old origin times differ by up to {dmax}"
    bak = fl_path + ".whole_seconds.bak"
    if not os.path.exists(bak):
        shutil.copy2(fl_path, bak)
    fl["origin_time"] = fl.event_id.map(new)
    fl.to_csv(fl_path, index=False)
    print(f"{fl_path}: {len(fl)} rows, origin_time -> 0.001 s (max |change| {dmax.total_seconds():.3f} s); backup {os.path.basename(bak)}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        process(d)
