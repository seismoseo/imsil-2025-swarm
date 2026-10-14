#!/usr/bin/env python3
"""Build / refresh the continuous-data symlink tree in the Buan-pipeline layout.

WHY: the validated Buan pipeline (20.2024_Buan_EQ_DL_detection) and the FMF template-matching
code both discover day files as
    continuous/<NET>/<STA>/<CHA>.D/<NET>.<STA>..<CHA>.D.<YYYY>.<DOY>      (7 dot-fields, asserted)
whereas the Haenam data live in three different layouts:
    NECIS cron archive : .../data/necis/continuous/<YYYY>/<STA>/KS.<STA>.<CHA>.<YYYY>.<DOY>.HH.MM.SS
    JNU tar extracts   : .../JNU_data/mseed/JU/<STA>/<STA>.JU.00.<CHA>.<YYYY>.<DOY>
    Earthworm archive  : ~/EW/EW_JNU/data/mseed/JU/<STA>/<STA>.JU.00.<CHA>.<YYYY>.<DOY>
Rather than patching the validated code, we expose every source through ONE symlink tree
(DL_detection/continuous/). Nothing is copied; nothing outside DL_detection/ is touched.
Re-run whenever the archives grow (the NECIS cron adds a day every night; Earthworm every day).

Only stations listed in station_table/stations.csv are linked (the 100-km set), so the tree
stays small. Where the same (station, channel, day) exists in several sources the FIRST source
in SOURCES wins (JNU tar extracts are complete days; Earthworm days may be partial while live).

Usage:
    python tools/link_continuous.py               # DRY RUN: counts what would be linked
    python tools/link_continuous.py --apply       # create the links
    python tools/link_continuous.py --apply --years 2026        # restrict to years
    python tools/link_continuous.py --apply --refresh           # re-point existing links
Sources in precedence order: mara (SDS mirror pulled by tools/mirror_mara.py), necis (nightly cron archive),
jnu_tar, earthworm. To add another archive, append a SOURCE entry with its filename regex and re-run with --apply.
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIONS_CSV = os.path.join(ROOT, "station_table", "stations.csv")
TARGET = os.path.join(ROOT, "continuous")

# --------------------------------------------------------------------------- sources
# Each source: root glob for day files + a regex with named groups sta, cha, year, doy (+ optional net).
# `net` fixed when the filename does not carry it reliably.
SOURCES = [
    # IMSIL PORT: only the MARA SDS mirror exists on PAGO (KS 2025 DOY 244-365, 2026 DOY 1-31). The Haenam necis /
    # jnu_tar / earthworm sources (moho paths; JU near Imsil starts 2026-09-01) are kept below as comments for provenance.
    dict(name="mara",                      # MARA SDS mirror (tools/mirror_mara.py); first source wins
         glob="/home/msseo/works/72.2025_Imsil_swarm/continuous_mara/*/*/*/*.D/*",
         regex=r"^(?P<net>[A-Z]{2})\.(?P<sta>[A-Z0-9]+)\.\.(?P<cha>[A-Z0-9]{3})\.D\.(?P<year>\d{4})\.(?P<doy>\d{3})$",
         net=None),
    # dict(name="necis",
    #      glob="/home/msseo/works/Claude/data/necis/continuous/*/*/*",
    #      regex=r"(?P<net>[A-Z]{2})\.(?P<sta>[A-Z0-9]+)\.(?P<cha>[A-Z0-9]{3})\.(?P<year>\d{4})\.(?P<doy>\d{3})\.\d\d\.\d\d\.\d\d$",
    #      net=None),
    # dict(name="jnu_tar",
    #      glob="/home/msseo/works/27.Haenam_sequence/JNU_data/mseed/JU/*/*",
    #      regex=r"(?P<sta>[A-Z0-9]+)\.(?P<net>JU)\.00\.(?P<cha>[A-Z0-9]{3})\.(?P<year>\d{4})\.(?P<doy>\d{3})$",
    #      net="JU"),
    # dict(name="earthworm",
    #      glob=os.path.expanduser("~/EW/EW_JNU/data/mseed/JU/*/*"),
    #      regex=r"(?P<sta>[A-Z0-9]+)\.(?P<net>JU)\.00\.(?P<cha>[A-Z0-9]{3})\.(?P<year>\d{4})\.(?P<doy>\d{3})$",
    #      net="JU"),
]


sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import config  # noqa: E402   IMSIL PORT: CHANNEL_EXCLUDE (decision D17)


def wanted_stations():
    st = pd.read_csv(STATIONS_CSV)
    return {(r.Network, r.Code) for r in st.itertuples()}


def scan(years=None):
    """Yield (net, sta, cha, year, doy, path, source) for every day file in every source."""
    for src in SOURCES:
        rx = re.compile(src["regex"])
        for p in glob.iglob(src["glob"]):
            m = rx.search(os.path.basename(p))
            if not m:
                continue
            net = src["net"] or m.group("net")
            year = int(m.group("year"))
            if years and year not in years:
                continue
            yield net, m.group("sta"), m.group("cha"), year, int(m.group("doy")), p, src["name"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="create the symlinks (default: dry run)")
    ap.add_argument("--refresh", action="store_true", help="re-point links that already exist")
    ap.add_argument("--years", type=str, default=None, help="comma list, e.g. 2026 or 2020,2026")
    a = ap.parse_args()
    years = {int(y) for y in a.years.split(",")} if a.years else None
    want = wanted_stations()

    planned, seen = {}, {}
    n_skip_station = 0
    for net, sta, cha, year, doy, path, srcname in scan(years):
        if (net, sta) not in want:
            n_skip_station += 1
            continue
        if cha[:2] in config.CHANNEL_EXCLUDE.get((net, sta), []):     # IMSIL PORT: D17, band kept out of the tree
            continue
        key = (net, sta, cha, year, doy)
        if key in planned:                       # first source wins
            continue
        planned[key] = (path, srcname)
        seen.setdefault(srcname, 0)
        seen[srcname] += 1

    n_new = n_exist = n_repoint = 0
    for (net, sta, cha, year, doy), (path, srcname) in sorted(planned.items()):
        d = os.path.join(TARGET, net, sta, f"{cha}.D")
        link = os.path.join(d, f"{net}.{sta}..{cha}.D.{year}.{doy:03d}")
        if os.path.lexists(link):
            if os.path.realpath(link) == os.path.realpath(path):
                n_exist += 1
                continue
            if not a.refresh:
                n_exist += 1
                continue
            n_repoint += 1
            if a.apply:
                os.unlink(link)
        else:
            n_new += 1
        if a.apply:
            os.makedirs(d, exist_ok=True)
            os.symlink(path, link)

    mode = "APPLIED" if a.apply else "DRY RUN"
    print(f"[{mode}] target {TARGET}")
    print(f"  day files found per source (wanted stations only): {seen}")
    print(f"  skipped (station not in {os.path.basename(STATIONS_CSV)}): {n_skip_station}")
    print(f"  links: new {n_new}, already present {n_exist}, re-pointed {n_repoint}")
    if planned:
        df = pd.DataFrame([(k[0], k[1], k[3], k[4], v[1]) for k, v in planned.items()], columns=["net", "sta", "year", "doy", "source"])
        g = df.groupby(["net", "year"]).agg(stations=("sta", "nunique"), days=("doy", "nunique"),
                                            doy_min=("doy", "min"), doy_max=("doy", "max"),
                                            sources=("source", lambda c: "/".join(f"{k}:{v}" for k, v in c.value_counts().items())))
        print(g.to_string())
    if not a.apply:
        print("  (dry run — add --apply to create the links)")


if __name__ == "__main__":
    main()
