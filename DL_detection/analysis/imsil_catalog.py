"""Build the Imsil DL catalog from the HYPOINVERSE .sum files (all years present), with QC TIERS,
distances, and cross-matches. IMSIL PORT of 27.Haenam_sequence/DL_detection/analysis/haenam_catalog.py:
  * KMA source = the entire KMA catalog only (copy in 72.2025_Imsil_swarm/data/catalogs; KST -> UTC by -9 h, as in
    Haenam); the Haenam 2026 KMA list is not used (decision D8). It contains all 24 Imsil events.
  * "template" catalog = the 24 KMA-listed Imsil events (catalogs/imsil_kma24.csv, UTC, KMA ML) via the Imsil registry
  * columns dist_haenam_km / in_haenam_box renamed dist_imsil_km / in_imsil_box; output catalogs/imsil_dl_catalog.csv
Haenam docstring (provenance):
distances, and cross-matches. Port of 21.2024_Buan_precursor/analysis/precursor_catalog.py.

For every located event:
  tier            A (gap<180, RMS<0.7, ERH/ERZ<5 km, N>=8) / B (gap<300, RMS<1.0, ERH/ERZ<10, N>=6) / '-'
  dist_haenam_km  epicentral distance to the Haenam cluster (config.HAENAM_EPICENTER)
  in_haenam_box   inside config.HAENAM_BOX (candidates for the relocation fold-back)
  nearest_sta_km  distance to the nearest station of the table (sensitivity proxy)
  KMA_match       'M<mag>' if a KMA event lies within KMA_TOL_S (entire KMA catalog + the 2026 Haenam KMA
                  list used by the relocation; both KST -> UTC), else '-'
  template_match  event_id of the relocated (template) catalog within TPL_TOL_S, else '-'
  unreported      no KMA match
  promote         tier A/B AND in_haenam_box AND no template match  -> stage_for_augment candidates
Writes catalogs/haenam_dl_catalog.csv (one row per located event, all years, sorted by time).

Run:  conda run -n base python analysis/haenam_catalog.py            (after ./run_haenam.sh locate)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import config  # noqa: E402
from sumio import read_sum  # noqa: E402
sys.path.insert(0, "/home/msseo/works/72.2025_Imsil_swarm/registry"); import imsil as haenam   # registry paths (IMSIL PORT)

MODEL = "phasenet_plus"
VELMODEL = "kim1983"
KMA_CSV = "/home/msseo/works/72.2025_Imsil_swarm/data/catalogs/kma_entire_catalog_260309.csv"   # IMSIL PORT: copy of moho 07.SeismoStats
KMA_2026_CSV = ""                                                                             # IMSIL PORT (D8): not used
TEMPLATES_CSV = haenam.path(None, "tm_templates_195")
OUT = os.path.join(ROOT, "catalogs", "imsil_dl_catalog.csv")                                  # IMSIL PORT
KMA_TOL_S = 12.0
TPL_TOL_S = 6.0


def _dist_km(lat, lon, lat0, lon0):
    return 111.2 * np.hypot(lat - lat0, (lon - lon0) * np.cos(np.radians(lat0)))


def qc_tier(r):
    def ok(q):
        return (r.erh < q["erh"] and r.erz < q["erz"] and r.gap < q["gap"]
                and r.num > q["num"] and r.rms < q["rms"])
    if ok(config.HAENAM_QC_A):
        return "A"
    if ok(config.HAENAM_QC_B):
        return "B"
    return "-"


def kma_events():
    k = pd.read_csv(KMA_CSV, parse_dates=["time"], encoding="utf-8-sig")
    k["time"] = k["time"] - pd.Timedelta(hours=9)
    k["magnitude"] = pd.to_numeric(k["magnitude"], errors="coerce")
    k = k[["time", "latitude", "longitude", "magnitude"]]
    if KMA_2026_CSV and os.path.exists(KMA_2026_CSV):
        k2 = pd.read_csv(KMA_2026_CSV)
        k2 = pd.DataFrame(dict(
            time=pd.to_datetime(k2[["Year", "Month", "Day", "Hour", "Minute", "Second"]]) - pd.Timedelta(hours=9),
            latitude=k2.Latitude, longitude=k2.Longitude, magnitude=k2.Magnitude))
        k = pd.concat([k, k2], ignore_index=True).drop_duplicates("time")
    return k.sort_values("time").reset_index(drop=True)


def nearest(t, ref, tol_s):
    d = (ref.time - t).abs()
    if len(d) and d.min() <= pd.Timedelta(seconds=tol_s):
        return d.idxmin()
    return None


def main():
    sums = sorted(glob.glob(f"{config.velmodel_dir(MODEL, VELMODEL)}/{config.REGION_NAME}*.sum"))
    if not sums:
        sys.exit(f"no .sum files under {config.velmodel_dir(MODEL, VELMODEL)} — run ./run_imsil.sh locate first")
    df = pd.concat([read_sum(s) for s in sums], ignore_index=True)
    df = df[df.time.notna() & df.lat.notna()].reset_index(drop=True)

    df["tier"] = df.apply(qc_tier, axis=1)
    la0, lo0 = config.IMSIL_EPICENTER
    df["dist_imsil_km"] = _dist_km(df.lat, df.lon, la0, lo0).round(2)
    b = config.IMSIL_BOX
    df["in_imsil_box"] = (df.lat.between(b[0], b[1]) & df.lon.between(b[2], b[3]))
    st = pd.read_csv(config.STATIONS_CSV)
    df["nearest_sta_km"] = [round(float(_dist_km(st.Latitude, st.Longitude, la, lo).min()), 1)
                            for la, lo in zip(df.lat, df.lon)]

    kma = kma_events()
    km = [nearest(t, kma, KMA_TOL_S) for t in df.time]
    df["KMA_match"] = [f"M{kma.loc[i, 'magnitude']:g}" if i is not None else "-" for i in km]
    df["KMA_dist_km"] = [round(float(_dist_km(kma.loc[i, "latitude"], kma.loc[i, "longitude"], la, lo)), 1)
                         if i is not None else np.nan for i, la, lo in zip(km, df.lat, df.lon)]
    tp = pd.read_csv(TEMPLATES_CSV, dtype={"event_id": str})
    tp["time"] = pd.to_datetime(tp.origin_time_UTC)
    tm = [nearest(t, tp, TPL_TOL_S) for t in df.time]
    df["template_match"] = [tp.loc[i, "event_id"] if i is not None else "-" for i in tm]
    df["unreported"] = df.KMA_match == "-"
    df["promote"] = df.tier.isin(["A", "B"]) & df.in_imsil_box & (df.template_match == "-")
    df["event_id"] = df.time.dt.strftime("%Y%m%d%H%M%S")

    out = df[["event_id", "time", "lat", "lon", "depth", "num", "gap", "rms", "erh", "erz", "qual", "tier",
              "dist_imsil_km", "in_imsil_box", "nearest_sta_km", "KMA_match", "KMA_dist_km",
              "template_match", "unreported", "promote"]].sort_values("time").reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"located events: {len(out)}  ({', '.join(os.path.basename(s) for s in sums)})")
    print(f"  tier A: {(out.tier=='A').sum()}   tier B: {(out.tier=='B').sum()}   fail: {(out.tier=='-').sum()}")
    box = out[out.in_imsil_box]
    print(f"in Imsil box: {len(box)}  (A/B: {box.tier.isin(['A','B']).sum()}, KMA24-matched: "
          f"{(box.template_match!='-').sum()}, KMA-matched: {(~box.unreported).sum()})")
    ntpl = len(tp[tp.time.between(out.time.min(), out.time.max())]) if len(out) else 0
    print(f"KMA-listed Imsil events inside the located time span: {ntpl}; recovered: "
          f"{out.template_match.nunique() - (1 if (out.template_match=='-').any() else 0)}")
    print(f"A/B in box and not one of the 24 KMA-listed events: {out.promote.sum()}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
