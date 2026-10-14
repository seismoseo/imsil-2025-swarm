#!/usr/bin/env python
"""Build catalogs/imsil_kma24.csv: the 24 KMA-listed Imsil events (UTC, KMA ML) — the Imsil "known-event" table.

Role: what the Haenam registry key `tm_templates_195` (the 195-event relocated catalog) provided to
analysis/haenam_catalog.py and reloc/run_dl_hypodd.py — a list of known events with event_id, origin_time_UTC and ML,
matched to DL events within 6 s. For Imsil it is the KMA list (no relocated canonical catalog exists).

Source : 72.2025_Imsil_swarm/event_catalog/event_catalog_utc.csv (already UTC; whole-second KMA origins).
Check  : if data/catalogs/kma_entire_catalog_260309.csv (KST) is present, every row must equal the KMA entry
         shifted by -9 h (time, lat, lon, depth, ML); the script stops on any mismatch.
Columns: event_id (YYYYMMDDHHMMSS, UTC), origin_time_UTC, latitude, longitude, depth_km, ML, sequence
Run    : python catalogs/build_imsil_kma24.py            (writes the csv; deterministic, no other side effects)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))                  # 72.2025_Imsil_swarm
SRC = os.path.join(PROJECT, "event_catalog", "event_catalog_utc.csv")
KMA = os.path.join(PROJECT, "data", "catalogs", "kma_entire_catalog_260309.csv")
OUT = os.path.join(HERE, "imsil_kma24.csv")


def main():
    c = pd.read_csv(SRC, encoding="utf-8-sig")
    t = pd.to_datetime(c[["Year", "Month", "Day", "Hour", "Minute", "Second"]].rename(columns=str.lower))
    out = pd.DataFrame(dict(event_id=t.dt.strftime("%Y%m%d%H%M%S"), origin_time_UTC=t.dt.strftime("%Y-%m-%dT%H:%M:%S"),
                            latitude=c.Latitude, longitude=c.Longitude, depth_km=c.Depth.astype(float),
                            ML=c.Magnitude.astype(float), sequence="2025"))
    assert len(out) == 24 and out.event_id.is_unique, f"expected 24 unique events, got {len(out)}"
    if os.path.exists(KMA):
        k = pd.read_csv(KMA, encoding="utf-8-sig", parse_dates=["time"])
        k["t_utc"] = k.time - pd.Timedelta(hours=9)
        m = k.set_index("t_utc").reindex(t)
        assert m.latitude.notna().all(), f"{m.latitude.isna().sum()} events not found in {KMA}"
        assert (abs(m.latitude.values - out.latitude.values) < 1e-6).all() and (abs(m.longitude.values - out.longitude.values) < 1e-6).all()
        assert (abs(pd.to_numeric(m.magnitude).values - out.ML.values) < 1e-6).all()
        print(f"check: all 24 rows equal {os.path.basename(KMA)} (KST - 9 h)")
    else:
        print(f"check skipped: {KMA} not present yet (copied by tools/phase0_copy_from_moho.sh)")
    out.to_csv(OUT, index=False)
    print(f"-> {OUT}: {len(out)} events, {out.origin_time_UTC.iloc[0]} .. {out.origin_time_UTC.iloc[-1]} UTC, "
          f"ML {out.ML.min():.1f}-{out.ML.max():.1f}")


if __name__ == "__main__":
    main()
