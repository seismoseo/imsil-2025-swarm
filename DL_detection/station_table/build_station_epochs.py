#!/usr/bin/env python
"""Per-year station epochs of the Imsil network from what is actually linked under continuous/.
IMSIL PORT of the Haenam tool: EPICENTER changed; the KG/JU/KWJ2-KWJU cross-checks are kept as they are (KG and JU
are expected to be absent for the Imsil period).

Writes station_table/station_epochs.csv with one row per (network, code, year): prefix(es) used, first/last day,
n_days, source archive(s), distance to the epicenter. Prints the cross-checks that the full-period run relies on:
  * KG only where data exist (MARA has KG for 2020-2024 only),
  * JU only from 2025-12-03 (JN01/08/09) and 2026-09-01 (the other JU stations),
  * KS stations that start late (NPGA/NSJA/NGJA/NHPA/NYSA 2022, JAEB/DASB/JEDA/DAYA 2023, GUNA/GWJA/BESA 2024, ...),
  * the co-located pair KWJ2/KWJU (identical coordinates in stations.csv): if both record on the same days one of them
    must be dropped from stations.csv, otherwise HYPOINVERSE counts one site twice,
  * candidate constant-anchor stations for later magnitude work (KS within ANCHOR_KM with data in every year).
Usage: python station_table/build_station_epochs.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import availability  # noqa: E402

STATIONS_CSV = os.path.join(HERE, "stations.csv")
OUT = os.path.join(HERE, "station_epochs.csv")
EPICENTER = (35.5506, 127.2951)              # IMSIL PORT: = config.IMSIL_EPICENTER
ANCHOR_KM = 60.0
SOURCE_TAGS = (("continuous_mara", "mara"), ("data/necis", "necis"), ("JNU_data", "jnu_tar"), ("EW_JNU", "earthworm"))


def source_of(path):
    real = os.path.realpath(path)
    for tag, name in SOURCE_TAGS:
        if tag in real:
            return name
    return "other"


def main():
    st = pd.read_csv(STATIONS_CSV)
    cl = np.cos(np.radians(EPICENTER[0]))
    st["dist_km"] = np.hypot((st.Longitude - EPICENTER[1]) * 111.19 * cl, (st.Latitude - EPICENTER[0]) * 111.19).round(1)
    av = availability.scan_continuous(cache_csv=None, force=True)
    if av.empty:
        print("nothing linked under continuous/"); return
    av["prefix"] = av.channel.str[:2]
    rows = []
    for (net, sta, y), g in av.groupby(["network", "station", "year"]):
        sample = os.path.join(availability.CONTINUOUS, g.dirnet.iloc[0], sta, f"{g.channel.iloc[0]}.D",
                              f"{net}.{sta}..{g.channel.iloc[0]}.D.{y}.{int(g.doy.iloc[0]):03d}")
        rows.append(dict(network=net, code=sta, year=y, prefixes="/".join(sorted(g.prefix.unique())), first_day=g.date.min().date(),
                         last_day=g.date.max().date(), n_days=g.date.nunique(), n_files=len(g), source=source_of(sample)))
    ep = pd.DataFrame(rows).merge(st[["Network", "Code", "dist_km"]].rename(columns={"Network": "network", "Code": "code"}), on=["network", "code"], how="left")
    ep = ep.sort_values(["network", "dist_km", "code", "year"]).reset_index(drop=True)
    ep.to_csv(OUT, index=False)
    print(f"station_epochs.csv: {len(ep)} station-years, {ep.code.nunique()} stations, years {ep.year.min()}-{ep.year.max()} -> {OUT}")
    print(ep.groupby(["network", "year"]).agg(stations=("code", "nunique"), station_days=("n_days", "sum")).to_string())
    # ---- cross-checks
    kg = ep[ep.network == "KG"]; print(f"KG years present: {sorted(kg.year.unique())} (expected <= 2024)" if len(kg) else "KG: no data linked")
    ju = ep[ep.network == "JU"]
    if len(ju):
        print(f"JU earliest day: {ju.first_day.min()} (expected 2025-12-03); stations per year: {ju.groupby('year').code.nunique().to_dict()}")
    for code in ("KWJ2", "KWJU"):
        pass
    both = av[av.station.isin(["KWJ2", "KWJU"])].drop_duplicates(["station", "date"])
    if both.station.nunique() == 2:
        shared = len(set(both[both.station == "KWJ2"].date) & set(both[both.station == "KWJU"].date))
        print(f"KWJ2/KWJU (same coordinates): {shared} shared days -> {'DROP ONE from stations.csv' if shared else 'no overlap, fine'}")
    ks = ep[ep.network == "KS"]; first = ks.groupby("code").first_day.min()
    late = first[first > pd.Timestamp("2020-01-31").date()]
    print("KS stations whose linked data start after Jan 2020: " + ", ".join(f"{c} {d}" for c, d in late.sort_values().items()) if len(late) else "KS: all stations present from Jan 2020")
    yrs = sorted(ep.year.unique())
    anchors = [c for c, g in ks.groupby("code") if set(g.year) >= set(yrs) and float(st.loc[st.Code == c, "dist_km"].iloc[0]) <= ANCHOR_KM and (g.n_days >= 300).sum() >= len(yrs) - 1]
    print(f"constant-anchor candidates (KS within {ANCHOR_KM:.0f} km, data in every linked year, >= 300 days/yr): {anchors}")


if __name__ == "__main__":
    main()
