#!/usr/bin/env python
"""Per-year status of the full-period Imsil DL run -> logs/full_run_status.csv (and a printed table).
IMSIL PORT of the Haenam tool: manifest path, DL catalog name and default years changed (HAENAM_* names are config aliases).

Columns: stations/station-days linked per network (continuous/), days with data, days picked, picks, PyOcto events,
located (.sum rows), HAENAM_QC / tier A / tier B counts, in-box (HAENAM_BOX) counts, DL-catalog KMA matches (when
catalogs/haenam_dl_catalog.csv exists), last DOY on MARA (mirror manifest).
Usage: python tools/run_status.py [--years 2020-2026] [--append]     (--append updates the csv, otherwise print only)
"""
import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import config  # noqa: E402
import availability  # noqa: E402
import sumio  # noqa: E402

MODEL = os.environ.get("MODEL", "phasenet_plus")
VELMODEL = os.environ.get("VELMODEL", config.DEFAULT_VELMODEL)
OUT = os.path.join(ROOT, "logs", "full_run_status.csv")
MANIFEST = "/home/msseo/works/72.2025_Imsil_swarm/continuous_mara/_manifest/manifest.csv"   # IMSIL PORT
DL_CATALOG = os.path.join(ROOT, "catalogs", "imsil_dl_catalog.csv")                         # IMSIL PORT


def parse_years(s):
    out = set()
    for tok in str(s).split(","):
        if "-" in tok:
            a, b = tok.split("-"); out.update(range(int(a), int(b) + 1))
        elif tok.strip():
            out.add(int(tok))
    return sorted(out)


def in_box(df):
    b = config.HAENAM_BOX
    return df.lat.between(b[0], b[1]) & df.lon.between(b[2], b[3])


def year_row(y, av):
    r = dict(year=y)
    a = av[av.year == y]
    for net in ("KS", "KG", "JU"):
        g = a[a.network == net]
        r[f"{net}_stations"] = g.station.nunique(); r[f"{net}_station_days"] = g.drop_duplicates(["station", "doy"]).shape[0]
    r["days_with_data"] = a.doy.nunique()
    pk = sorted(glob.glob(os.path.join(config.picks_dir(MODEL, y), f"picks_{y}.*.csv")))
    r["days_picked"] = len(pk); r["picks"] = sum(max(sum(1 for _ in open(p)) - 1, 0) for p in pk)
    pe = config.pyocto_events(MODEL, y); r["pyocto_events"] = sum(1 for _ in open(pe)) - 1 if os.path.exists(pe) else 0
    sp = os.path.join(config.velmodel_dir(MODEL, VELMODEL), f"{config.REGION_NAME}{y}.sum")
    if os.path.exists(sp):
        s = sumio.read_sum(sp); r["located"] = len(s)
        for name, qc in (("qc", config.HAENAM_QC), ("tierA", config.HAENAM_QC_A), ("tierB", config.HAENAM_QC_B)):
            q = sumio.apply_qc(s, qc); r[name] = len(q); r[f"{name}_inbox"] = int(in_box(q).sum())
        r["median_rms"] = round(float(s.rms.median()), 2) if len(s) else None
    else:
        r["located"] = 0
    if os.path.exists(DL_CATALOG):
        c = pd.read_csv(DL_CATALOG, dtype=str); t = pd.to_datetime(c.iloc[:, 1], errors="coerce") if "origin_time" not in c else pd.to_datetime(c.origin_time, errors="coerce")
        cy = c[t.dt.year == y]
        if "KMA_match" in cy:
            r["kma_matches"] = int(cy.KMA_match.replace("-", pd.NA).notna().sum())
    if os.path.exists(MANIFEST):
        m = pd.read_csv(MANIFEST); m = m[m.year == y]
        r["mara_last_doy"] = int(m.doy_last.max()) if len(m) else None; r["mara_verified_frac"] = round(float((m.status == "verified").mean()), 2) if len(m) else None
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--years", default="2025-2026"); ap.add_argument("--append", action="store_true")
    a = ap.parse_args(); years = parse_years(a.years)
    av = availability.scan_continuous(cache_csv=None, force=True)
    rows = [year_row(y, av) for y in years]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); print(df.to_string(index=False))
    if a.append:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
        if len(old):
            old = old[~old.year.isin(years)]
        pd.concat([old, df]).sort_values("year").to_csv(OUT, index=False); print("->", OUT)


if __name__ == "__main__":
    main()
