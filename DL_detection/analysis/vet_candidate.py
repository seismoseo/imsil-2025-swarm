"""Waveform vetting for a DL candidate: distance-ordered record section of the continuous data at
the stations PyOcto associated to the event, P/S picks marked. A real local earthquake shows a
coherent move-out; a false association shows scattered picks. Port of the Buan precursor tool.

  conda run -n base python analysis/vet_candidate.py "2026-08-14 08:43:38" out.png
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from obspy import read, UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import config  # noqa: E402
import plotstyle  # noqa: F401,E402

MODEL = "phasenet_plus"
STA = pd.read_csv(config.STATIONS_CSV)


def _zfile(net, sta, year, doy):
    for cha in ("HHZ", "HGZ", "ELZ", "DPZ", "BHZ"):
        f = glob.glob(f"{config.CONTINUOUS}/{net}/{sta}/{cha}.D/*.D.{year}.{doy:03d}")
        if f:
            return f[0]
    return None


def vet(origin_str, outfile=None, win=(-5, 40), bp=(2, 15), extra_stations=()):
    """extra_stations: e.g. ("JU.JN09", "KS.HAWA") — drawn even without picks (empty pick set)."""
    import matplotlib.pyplot as plt
    t0 = pd.Timestamp(origin_str)
    year = t0.year
    ev = pd.read_csv(config.pyocto_events(MODEL, year), parse_dates=["time"])
    ev["time"] = ev.time.dt.tz_localize(None)
    i = (ev.time - t0).abs().idxmin()
    e = ev.loc[i]
    if abs((e.time - t0).total_seconds()) > 10:
        print(f"warn: nearest pyocto event is {abs((e.time-t0).total_seconds()):.1f}s away")
    asg = pd.read_csv(config.pyocto_assign(MODEL, year))
    pk = asg[asg.event_idx == e.idx].copy()
    pk["net"] = pk.station.str.split(".").str[0]
    pk["sta"] = pk.station.str.split(".").str[1]
    origin = UTCDateTime(e.time.to_pydatetime())
    doy = int(origin.julday)
    cosl = np.cos(np.radians(e.latitude))
    rows = []
    for sta, g in pk.groupby("sta"):
        net = g.net.iloc[0]
        m = STA[(STA.Network == net) & (STA.Code == sta)]
        if m.empty:
            continue
        d = 111.2 * np.hypot(m.Latitude.iloc[0] - e.latitude, (m.Longitude.iloc[0] - e.longitude) * cosl)
        ptt = {ph: UTCDateTime(float(t)) - origin for ph, t in zip(g.phase, g.time)}
        rows.append((d, net, sta, ptt))
    have = {(n, st) for _, n, st, _ in rows}
    for ns in extra_stations:                       # nearest stations without picks: show them anyway
        net, sta_ = ns.split(".")
        if (net, sta_) in have:
            continue
        m = STA[(STA.Network == net) & (STA.Code == sta_)]
        if m.empty:
            continue
        d = 111.2 * np.hypot(m.Latitude.iloc[0] - e.latitude, (m.Longitude.iloc[0] - e.longitude) * cosl)
        rows.append((d, net, sta_, {}))
    rows.sort()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.42 * len(rows) + 1)))
    dmax = max((r[0] for r in rows), default=10)
    sc = 0.45 * dmax / max(len(rows), 1)
    for d, net, sta, ptt in rows:
        f = _zfile(net, sta, year, doy)
        if not f:
            continue
        try:
            tr = read(f, starttime=origin + win[0], endtime=origin + win[1])[0]
        except Exception:
            continue
        tr.detrend("demean"); tr.detrend("linear"); tr.taper(0.05)
        tr.filter("bandpass", freqmin=bp[0], freqmax=bp[1], corners=4, zerophase=True)
        t = tr.times() + (tr.stats.starttime - origin)
        w = tr.data
        if np.max(np.abs(w)) > 0:
            w = w / np.max(np.abs(w)) * sc
        ax.plot(t, d + w, color="0.15", lw=0.5, zorder=1)
        if "P" in ptt:
            ax.plot([ptt["P"]] * 2, [d - sc, d + sc], color="#d62728", lw=1.4, zorder=3)
        if "S" in ptt:
            ax.plot([ptt["S"]] * 2, [d - sc, d + sc], color="#1f77b4", lw=1.4, zorder=3)
        ax.text(win[0], d, f"{net}.{sta} " + ("(no pick) " if not ptt else ""), ha="right", va="center", fontsize=6.5, color="0.4" if not ptt else "k")
    ax.axvline(0, color="green", ls="--", lw=0.8, label="origin")
    ax.plot([], [], color="#d62728", lw=1.4, label="P pick")
    ax.plot([], [], color="#1f77b4", lw=1.4, label="S pick")
    ax.set_xlabel("Time after origin (s)"); ax.set_ylabel("Epicentral distance (km)")
    ax.set_xlim(win[0] - 0.12 * (win[1] - win[0]), win[1])
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(f"{e.time:%Y-%m-%d %H:%M:%S}  ·  {e.latitude:.3f}N {e.longitude:.3f}E  "
                 f"·  {e.depth:.1f} km  ·  {int(e.picks)} picks, {len(rows)} stations", fontsize=9)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=160, bbox_inches="tight")
        print(f"wrote {outfile}  ({len(rows)} stations, depth {e.depth:.1f} km)")
    return fig


if __name__ == "__main__":
    vet(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
