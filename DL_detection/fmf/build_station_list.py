#!/usr/bin/env python
"""Build fmf/stations_imsil_5_20H.csv, the FMF station list of the Imsil 5-20 Hz run (new Imsil file).

Format = Haenam's stations_full_period_H.csv: network, station, horizontals, s_exclude, note.
  horizontals = 1 -> the S pass uses N/E;  0 -> the S pass uses Z (Haenam "consistent component" rule: horizontals of
                stations whose orientation/sensor changed inside the scan period are not used)
  s_exclude   = 1 -> no S window at all (P only); not used by the Haenam final run, kept 0
Parameters (decisions, logged in docs/decisions.csv; evidence in notebook 06):
  D11 RADIUS_KM : Haenam used the 22 stations within ~58 km that had data over the scan period. Imsil default 60 km
                  -> 22 KS stations (KWJU dropped with D2); 58 km would drop SCHA (59.4 km).
  D12 S_ON_Z    : stations with a response/sensor epoch change inside 2025-09-01 .. 2026-01-31 (KMA RESP epoch starts:
                  NAWB 2025-09-16, KWJ2 2025-10-28; HMPA 2025-09-09 lies outside 60 km). Default: S on Z there, until the
                  P-polarization orientation check (nb06) shows the horizontals were not re-oriented.
DRY RUN BY DEFAULT (prints the table); --apply writes the csv.
    python fmf/build_station_list.py [--apply]
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIONS_CSV = os.path.join(ROOT, "station_table", "stations.csv")
OUT = os.path.join(HERE, "stations_imsil_5_20H.csv")
EPICENTER = (35.5506, 127.2951)            # = config.IMSIL_EPICENTER
RADIUS_KM = 60.0                           # D11
S_ON_Z = {"NAWB": "KMA RESP epoch starts 2025-09-16 (inside the scan period)",
          "KWJ2": "KMA RESP epoch starts 2025-10-28 (inside the scan period)"}   # D12
NETWORKS = ("KS",)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    st = pd.read_csv(STATIONS_CSV)
    st = st[st.Network.isin(NETWORKS)].copy()
    cl = np.cos(np.radians(EPICENTER[0]))
    st["dist_km"] = np.hypot((st.Longitude - EPICENTER[1]) * 111.19 * cl, (st.Latitude - EPICENTER[0]) * 111.19)
    st = st[st.dist_km <= RADIUS_KM].sort_values("dist_km").reset_index(drop=True)
    out = pd.DataFrame(dict(network=st.Network, station=st.Code,
                            horizontals=[0 if c in S_ON_Z else 1 for c in st.Code], s_exclude=0,
                            note=[(f"S on Z (D12): {S_ON_Z[c]}; " if c in S_ON_Z else "S on N/E; ") + f"{d:.1f} km; {s}"
                                  for c, d, s in zip(st.Code, st.dist_km, st.Sensor)]))
    print(out.to_string(index=False))
    print(f"{len(out)} stations within {RADIUS_KM:g} km; S on Z at {int((out.horizontals == 0).sum())}")
    if a.apply:
        out.to_csv(OUT, index=False); print("->", OUT)
    else:
        print("(dry run — add --apply to write)")


if __name__ == "__main__":
    main()
