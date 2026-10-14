#!/usr/bin/env python3
"""Build station_table/stations.csv for the Imsil DL-detection project (IMSIL PORT of the Haenam builder).

IMSIL PORT changes (each a decision in docs/decisions.csv; defaults = the Haenam convention):
  * source 1 = kskg_used_stations_100km.csv = copy of 72.2025_Imsil_swarm/station_table/used_stations_100km.csv
    (the 55 KS stations within 100 km of Imsil used for the KMA event waveforms; same schema as Haenam's file)
  * D1 ELEVATION_MODE: "zero" (default; Haenam's KS/KG rows are all 0.0 m, as is the existing Imsil STA file) or
    "table" (real elevations 42-698 m from the Imsil list)
  * D2 DROP_STATIONS: {"KWJU"} (default; same coordinates as KWJ2, dropped in Haenam on 2026-09-17)
  * D7 JU_XML = None: no JU rows (no JU data exist near Imsil before 2026-09-01; the period ends 2026-01-31)
  * EPICENTER = the Imsil source

Haenam docstring (provenance):

Columns (identical to the Buan pipeline's stations.csv, which core.build_station_table subsets
by (Network, Code) and _station_pyocto feeds to PyOcto):
    Network, Code, Latitude, Longitude, Elevation, Sensor, Start, End

Sources
  1. kskg_used_stations_100km.csv — the KS/KG stations within 100 km of the Haenam source that the
     canonical Haenam_joint relocation used (copied from korea-cluster-relocation
     runs/Haenam_joint/station_table/used_stations_100km.csv). KG stations have NO continuous data
     on this machine (event-segment SAC only); they are kept so the table matches the relocation,
     but they simply never appear in the picks.
  2. JU_station.xml — the 30-station JNU network (200 Hz HH). Only stations within RADIUS_KM are
     kept: JN09 (4 km), JN01 (9 km), JN08 (17 km) have the 2025-12 -> 2026-08 tar archive AND the
     Earthworm feed; JN02/JN03/JN04/JN05 (65-96 km) exist only in the Earthworm feed (2026-09-01 on).

A second file stations_with_distance.csv adds dist_km / data source notes for humans.

Run:  python station_table/build_stations.py
"""
import math
import os

import pandas as pd
from obspy import read_inventory

HERE = os.path.dirname(os.path.abspath(__file__))
KSKG = os.path.join(HERE, "kskg_used_stations_100km.csv")
JU_XML = None                        # IMSIL PORT (D7): Haenam read /home/msseo/works/27.Haenam_sequence/JNU_data/JU_station.xml
OUT = os.path.join(HERE, "stations.csv")
OUT_DIST = os.path.join(HERE, "stations_with_distance.csv")

EPICENTER = (35.5506, 127.2951)      # IMSIL PORT: Imsil source (= config.IMSIL_EPICENTER)
ELEVATION_MODE = "zero"              # IMSIL PORT (D1): "zero" (Haenam convention) | "table"
DROP_STATIONS = {"KWJU"}             # IMSIL PORT (D2): co-located with KWJ2 (Haenam dropped KWJU)
RADIUS_KM = 100.0
JU_TAR_STATIONS = {"JN01", "JN08", "JN09"}   # have the historical tar archive (2025-12-04 ->)


def dist_km(lat, lon):
    return math.hypot((lat - EPICENTER[0]) * 111.19,
                      (lon - EPICENTER[1]) * 111.19 * math.cos(math.radians(EPICENTER[0])))


def main():
    ks = pd.read_csv(KSKG)
    ks = ks[~ks.Code.isin(DROP_STATIONS)].reset_index(drop=True)                 # IMSIL PORT (D2)
    if ELEVATION_MODE == "zero":                                                 # IMSIL PORT (D1)
        ks["Elevation"] = 0.0
    ks["Start"] = ""
    ks["End"] = ""
    ks["note"] = ks.Network.map({"KS": "MARA SDS mirror (100 Hz)", "KG": "no continuous data here"})   # IMSIL PORT: data source

    inv = read_inventory(JU_XML) if JU_XML else []                              # IMSIL PORT (D7)
    rows = []
    for net in inv:
        for sta in net:
            d = dist_km(sta.latitude, sta.longitude)
            if d > RADIUS_KM:
                continue
            rows.append(dict(Network=net.code, Code=sta.code, Latitude=round(sta.latitude, 5),
                             Longitude=round(sta.longitude, 5), Elevation=float(sta.elevation or 0.0),
                             Sensor="HH", Start=str(sta.start_date)[:10] if sta.start_date else "",
                             End="",
                             note=("JNU tar 2025-12 -> 2026-08 + Earthworm 2026-09-01 ->"
                                   if sta.code in JU_TAR_STATIONS else "Earthworm 2026-09-01 -> only")))
    ju = pd.DataFrame(rows, columns=["Network", "Code", "Latitude", "Longitude", "Elevation", "Sensor", "Start", "End", "note"])

    df = pd.concat([ks, ju], ignore_index=True)
    df["dist_km"] = [round(dist_km(a, b), 1) for a, b in zip(df.Latitude, df.Longitude)]
    df = df.sort_values("dist_km").reset_index(drop=True)
    cols = ["Network", "Code", "Latitude", "Longitude", "Elevation", "Sensor", "Start", "End"]
    df[cols].to_csv(OUT, index=False)
    df[cols + ["dist_km", "note"]].to_csv(OUT_DIST, index=False)
    print(f"wrote {OUT}: {len(df)} stations (ELEVATION_MODE={ELEVATION_MODE}, dropped {sorted(DROP_STATIONS)}, JU_XML={JU_XML})")
    print(df.groupby("Network").agg(n=("Code", "size"), nearest_km=("dist_km", "min")).to_string())
    print("nearest 10:")
    print(df[["Network", "Code", "dist_km", "Sensor", "note"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
