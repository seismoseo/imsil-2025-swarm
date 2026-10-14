"""Continuous-archive inventory for the Buan 2024 project.

Scans continuous/<NET>/<STA>/<CHA>.D/<NET>.<STA>..<CHA>.D.<YEAR>.<DOY> day files
into a tidy DataFrame (one row per file) used by notebooks/01 for the
station-configuration and data-availability analysis.

The filename is the source of truth for network/station/doy here; note the BA
quirk (filenames say DP?, trace headers say BH?) does not matter for inventory.
"""
import os
import datetime as dt

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONTINUOUS = os.path.join(ROOT, "continuous")
STATIONS_CSV = os.path.join(ROOT, "station_table", "stations.csv")
CACHE_CSV = os.path.join(ROOT, "notebooks", "cache", "availability.csv")


def scan_continuous(continuous_dir=CONTINUOUS, cache_csv=CACHE_CSV, force=False):
    """One row per day file: dirnet, network, station, channel, year, doy, date, size_mb.

    `dirnet` is the directory name under continuous/ (BG for the JN array);
    `network` is the network code embedded in the filename (matches trace headers).
    """
    if cache_csv and not force and os.path.exists(cache_csv):
        df = pd.read_csv(cache_csv, parse_dates=["date"])
        return df

    rows = []
    for dirnet in sorted(os.listdir(continuous_dir)):
        netdir = os.path.join(continuous_dir, dirnet)
        if not os.path.isdir(netdir):
            continue
        for sta in sorted(os.listdir(netdir)):
            stadir = os.path.join(netdir, sta)
            if not os.path.isdir(stadir):
                continue
            for cha in sorted(os.listdir(stadir)):
                chadir = os.path.join(stadir, cha)
                if not os.path.isdir(chadir):
                    continue
                with os.scandir(chadir) as it:
                    for e in it:
                        parts = e.name.split(".")
                        if len(parts) != 7 or not e.is_file():
                            continue
                        net, code, _, chan, _, year, doy = parts
                        rows.append((dirnet, net, code, chan, int(year),
                                     int(doy), e.stat().st_size / 1e6))
    df = pd.DataFrame(rows, columns=["dirnet", "network", "station", "channel",
                                     "year", "doy", "size_mb"])
    df["date"] = pd.to_datetime(df["year"].astype(str), format="%Y") \
        + pd.to_timedelta(df["doy"] - 1, unit="D")
    df = df.sort_values(["network", "station", "channel", "date"]).reset_index(drop=True)
    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df


def station_days(avail):
    """Collapse channels: one row per (network, station, date) with the per-day
    completeness estimate = total size / median total size of that station."""
    d = (avail.groupby(["network", "station", "date"], as_index=False)
         .agg(size_mb=("size_mb", "sum"), n_chan=("channel", "nunique")))
    med = d.groupby(["network", "station"])["size_mb"].transform("median")
    d["completeness"] = (d["size_mb"] / med).clip(upper=1.0)
    return d


def load_stations(stations_csv=STATIONS_CSV):
    return pd.read_csv(stations_csv)


def coverage_table(avail):
    """Per-station summary: first/last day, n days, channels, dirnet."""
    g = avail.groupby(["network", "station"])
    out = g.agg(dirnet=("dirnet", "first"),
                first=("date", "min"), last=("date", "max"),
                n_days=("date", "nunique"),
                channels=("channel", lambda c: "/".join(sorted(set(c)))),
                total_gb=("size_mb", lambda s: round(s.sum() / 1e3, 2)))
    out["span_days"] = (out["last"] - out["first"]).dt.days + 1
    out["uptime_frac"] = (out["n_days"] / out["span_days"]).round(3)
    return out.reset_index()


def operating_counts(avail, by="network"):
    """Stations operating per day, per network (for the count-vs-time plot)."""
    d = avail.drop_duplicates(["network", "station", "date"])
    return (d.groupby(["date", by])["station"].nunique()
            .unstack(by).fillna(0).astype(int))
