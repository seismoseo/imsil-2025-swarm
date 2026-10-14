"""Add the RELOCATED origin time with decimal seconds to the project catalogs.

IMSIL PORT: reloc_times() and add() are unchanged; KCR points at the PAGO copy, and __main__ is rewritten for Imsil:
it reads the Imsil run's dt.cc hypoDD.reloc (default runs/hypodd_dl/2.HypoDD/02.dt.cc/hypoDD.reloc, or argv[1]) and
the run's event_manifest.csv, and fills origin_time_reloc in catalogs/imsil_dl_relocated.csv and t_reloc in
catalogs/reference_25m/DL_reference_25m.csv (if present). The Haenam canonical-frame / template / JNU-snapshot
targets do not exist for Imsil.

Haenam docstring (provenance):

korea-cluster-relocation's `viz.location_table` writes `origin_time` with strftime("%Y-%m-%d %H:%M:%S"), i.e. truncated
to whole seconds, although HypoDD solves for an origin-time correction per event and `hypoDD.reloc` carries the
relocated origin time to 0.001 s (columns yr mo dy hr mi sc). This script reads hypoDD.reloc, maps cuspid ->
event_id through event_manifest.csv (cuspid = 200000 + event_idx) and adds `origin_time_reloc` (UTC, 0.001 s) to:
    catalogs/haenam_dl_relocated.csv                     (DL, KS+KG+JU)
    catalogs/haenam_dl_relocated_canonical_frame.csv     (DL, canonical frame; column t_reloc)
    catalogs/haenam_templates.csv                        (KMA events, KS+KG+JNU: reloc_195ev_JNU_snapshot)
Idempotent (columns are replaced). The frozen snapshots themselves are not modified.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KCR = "/home/msseo/works/72.2025_Imsil_swarm/_external/korea-cluster-relocation"      # IMSIL PORT: PAGO copy
DL_DIR = os.path.join(ROOT, "runs", "hypodd_dl")


def reloc_times(reloc_path, manifest_path):
    sys.path.insert(0, KCR)
    from pipeline.core import sumio
    rl = sumio.read_reloc(reloc_path)
    mf = pd.read_csv(manifest_path, dtype={"event_id": str})
    cus = {200000 + int(r.event_idx): r.event_id for r in mf.itertuples()}
    t = pd.to_datetime(dict(year=rl.yr, month=rl.mo, day=rl.dy, hour=rl.hr, minute=rl.mi)) + pd.to_timedelta(rl.sc.astype(float), unit="s")
    return pd.DataFrame(dict(event_id=rl.id.astype(int).map(cus), origin_time_reloc=t.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]))


def add(csv_path, times, key="event_id", col="origin_time_reloc", comment=None):
    hdr = ""
    with open(csv_path) as f:
        first = f.readline()
    if first.startswith("#"):
        hdr = first
    df = pd.read_csv(csv_path, comment="#", dtype={key: str, "template_match": str})
    df = df.drop(columns=[col], errors="ignore").merge(times.rename(columns={"event_id": key, "origin_time_reloc": col}), on=key, how="left")
    with open(csv_path, "w") as f:
        f.write(hdr); df.to_csv(f, index=False)
    print(f"{os.path.relpath(csv_path, ROOT)}: {col} filled for {int(df[col].notna().sum())}/{len(df)} rows; e.g. {df[key].iloc[0]} -> {df[col].iloc[0]}")


if __name__ == "__main__":                                       # IMSIL PORT: Imsil targets
    reloc = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DL_DIR, "2.HypoDD", "02.dt.cc", "hypoDD.reloc")
    dl_t = reloc_times(reloc, os.path.join(DL_DIR, "event_manifest.csv"))
    add(os.path.join(ROOT, "catalogs", "imsil_dl_relocated.csv"), dl_t)
    ref = os.path.join(ROOT, "catalogs", "reference_25m")
    if os.path.exists(os.path.join(ref, "DL_reference_25m.csv")):
        add(os.path.join(ref, "DL_reference_25m.csv"), dl_t, key="eid", col="t_reloc")
