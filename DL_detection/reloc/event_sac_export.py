"""Per-event SAC export from the continuous archive for the HypoDD stage (vendored).

Compact vendoring of the Ulsan exporter (02.Ulsan_Fault_detection/src/uflib/event_sac_export.py) with
the two Buan patches (20.2024_Buan_EQ_DL_detection/hypodd/buan_event_sac_export.py) folded in:
  * continuous layout  continuous/<NET>/<CODE>/<CHA>.D/<NET>.<CODE>..<CHA>.D.<YYYY>.<DDD>  + channel policy
  * native sampling rates are KEPT (JU 200 Hz, KS 100 Hz); only sub-50 Hz traces are rejected. The
    framework's xcorr lanczos-interpolates everything to 1000 Hz, so mixed rates are fine, and the SAC
    store keeps the JNU band for later spectral work.
  * near-integer sampling rates are rounded before merge (NECIS 99.99999 Hz segments).
Output layout (plug-compatible with korea-cluster-relocation's xcorr / write_phs / write_sta):
    <out_root>/<event_id>/<event_id>.<NET>.<STA>.<CHA>.sac    reference time = origin (o = 0),
        evla/evlo/evdp, stla/stlo/stel, dist/az/baz, P in `a`, S in `t0` (origin-relative)
    <out_root>/<event_id>/<event_id>_picks.csv               the PyOcto-assigned picks used
Picks come from the PyOcto assignment table of the DL pipeline (one P + one S per station as
associated), matched to the catalog row by origin time within PYOCTO_TIME_TOL_S.
"""
from __future__ import annotations

import functools
import glob
import os
from dataclasses import dataclass

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth

PRE_S_DEFAULT = 30.0           # window: origin - 30 s ... origin + 90 s (Ulsan/Buan convention)
POST_S_DEFAULT = 90.0
PYOCTO_TIME_TOL_S = 5.0
MIN_HZ = 50.0


def event_id_from_time(t) -> str:
    if isinstance(t, UTCDateTime):
        t = t.datetime
    elif isinstance(t, pd.Timestamp):
        t = t.to_pydatetime()
    return t.strftime("%Y%m%d%H%M%S")


def load_stations_for_year(station_table_root, year) -> pd.DataFrame:
    """stations_<YYYY>.csv written by the DL pipeline's association stage (stations that had picks)."""
    target = os.path.join(station_table_root, f"stations_{year}.csv")
    if os.path.exists(target):
        return pd.read_csv(target)
    cands = sorted(glob.glob(os.path.join(station_table_root, "stations_*.csv")))
    years = [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in cands]
    earlier = [(y, p) for y, p in zip(years, cands) if y <= year]
    if not earlier:
        raise FileNotFoundError(f"no stations_<YYYY>.csv with year <= {year} under {station_table_root}")
    return pd.read_csv(earlier[-1][1])


# ------------------------------------------------------------------ PyOcto picks
@functools.lru_cache(maxsize=8)
def _load_pyocto_year_cached(pyocto_root, year, velmodel):
    evt_path = os.path.join(pyocto_root, f"pyocto_{velmodel}_{year}.csv")
    asg_path = os.path.join(pyocto_root, f"pyocto_assignment_{velmodel}_{year}.csv")
    if not (os.path.exists(evt_path) and os.path.exists(asg_path)):
        return None, None
    evt = pd.read_csv(evt_path)
    evt["time"] = pd.to_datetime(evt["time"], utc=True)
    asg = pd.read_csv(asg_path)
    parts = asg["station"].str.rstrip(".").str.split(".", expand=True)
    asg["Network"], asg["Code"] = parts[0], parts[1]
    asg["peak_time"] = pd.to_datetime(asg["time"], unit="s", utc=True)
    return evt, asg


def match_pyocto_event_idx(evt, origin_utc, tol_s=PYOCTO_TIME_TOL_S):
    if evt is None or len(evt) == 0:
        return None
    t = pd.Timestamp(origin_utc.datetime, tz="UTC")
    dt = (evt["time"] - t).abs()
    best = dt.idxmin()
    return int(evt.loc[best, "idx"]) if dt.loc[best].total_seconds() <= tol_s else None


def picks_for_event(asg, ev_idx):
    """{(Network, Code): {"P": Timestamp, "S": Timestamp}} — PyOcto gives one P + one S per station."""
    sub = asg[asg["event_idx"] == ev_idx]
    out = {}
    for r in sub.sort_values("peak_time").itertuples():
        out.setdefault((r.Network, r.Code), {}).setdefault(r.phase, r.peak_time)
    return out, sub


# ------------------------------------------------------------------ continuous archive
def _apply_channel_policy(dirnet, files):
    """One sensor per station (HH > EL > HG for KS/KG, HH for JU) — the DL pipeline's policy."""
    try:
        import config as pipe_config      # DL_detection/pipeline/config.py (on sys.path via the driver)
        return pipe_config.apply_channel_policy(dirnet, files)
    except Exception:                      # noqa: BLE001
        return sorted(files)


def _continuous_glob(continuous_root, network, code, year, julday):
    pat = os.path.join(continuous_root, network, code, "*.D", f"{network}.{code}..*.D.{year}.{julday:03d}")
    files = sorted(glob.glob(pat))
    return _apply_channel_policy(network, files) if files else []


def read_continuous_for_event(continuous_root, network, code, origin_utc,
                              pre_s=PRE_S_DEFAULT, post_s=POST_S_DEFAULT):
    o = origin_utc
    t_start, t_end = o - pre_s, o + post_s
    needed = [(t_start.year, t_start.julday)]
    if (t_end.year, t_end.julday) != needed[0]:
        needed.append((t_end.year, t_end.julday))
    files = []
    for y, jd in needed:
        files.extend(_continuous_glob(continuous_root, network, code, y, jd))
    if not files:
        return obspy.Stream()
    st = obspy.Stream()
    for f in files:
        try:
            st.extend(obspy.read(f, starttime=t_start, endtime=t_end))
        except Exception:                      # noqa: BLE001
            continue
    if not len(st):
        return st
    for tr in st:                              # NECIS 99.99999 Hz vs 100 Hz segments
        r = tr.stats.sampling_rate
        if 0 < abs(r - round(r)) < 1e-3:
            tr.stats.sampling_rate = float(round(r))
    try:
        st.merge(method=1, fill_value=0)
    except Exception:                          # noqa: BLE001
        pass
    for tr in list(st):
        if tr.stats.sampling_rate < MIN_HZ:
            st.remove(tr)
    if not len(st):
        return st
    out = st.slice(starttime=t_start, endtime=t_end)
    for tr in out:
        if np.ma.isMaskedArray(tr.data):
            tr.data = tr.data.filled(0).astype(np.float32)
    return out


# ------------------------------------------------------------------ SAC writer
@dataclass
class EventMeta:
    event_id: str
    origin_utc: UTCDateTime
    lat: float
    lon: float
    depth_km: float


@dataclass
class StationMeta:
    network: str
    code: str
    lat: float
    lon: float
    elev_m: float | None


def _populate_sac_header(tr, ev, sta, p_rel_s=None, s_rel_s=None):
    s = tr.stats.sac if "sac" in tr.stats else tr.stats.setdefault("sac", obspy.core.AttribDict())
    o = ev.origin_utc
    s.nzyear, s.nzjday = o.year, o.julday
    s.nzhour, s.nzmin, s.nzsec = o.hour, o.minute, o.second
    s.nzmsec = int(round(o.microsecond / 1000))
    s.o = 0.0
    s.evla, s.evlo, s.evdp = float(ev.lat), float(ev.lon), float(ev.depth_km)
    s.stla, s.stlo = float(sta.lat), float(sta.lon)
    if sta.elev_m is not None and np.isfinite(sta.elev_m):
        s.stel = float(sta.elev_m)
    dist_m, az, baz = gps2dist_azimuth(ev.lat, ev.lon, sta.lat, sta.lon)
    s.dist, s.az, s.baz = dist_m / 1000.0, float(az), float(baz)
    s.kevnm = ev.event_id[:16]
    s.kstnm, s.knetwk, s.kcmpnm = sta.code[:8], sta.network[:8], tr.stats.channel[:8]
    if p_rel_s is not None and np.isfinite(p_rel_s):
        s.a = float(p_rel_s); s.ka = "P"
    if s_rel_s is not None and np.isfinite(s_rel_s):
        s.t0 = float(s_rel_s); s.kt0 = "S"


def write_event_sac(stream, out_dir, *, event_meta, station_meta, p_rel_s=None, s_rel_s=None):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for tr in stream:
        _populate_sac_header(tr, event_meta, station_meta, p_rel_s=p_rel_s, s_rel_s=s_rel_s)
        path = os.path.join(out_dir, f"{event_meta.event_id}.{station_meta.network}.{station_meta.code}.{tr.stats.channel}.sac")
        tr.write(path, format="SAC")
        written.append(path)
    return written


# ------------------------------------------------------------------ per-event / bulk
def export_event(row, continuous_root, stations, out_root, *, pyocto_root, pyocto_velmodel="kim1983",
                 pre_s=PRE_S_DEFAULT, post_s=POST_S_DEFAULT, pyocto_tol_s=PYOCTO_TIME_TOL_S) -> dict:
    eid, o = row["event_id"], row["origin_utc"]
    ev = EventMeta(eid, o, float(row["lat"]), float(row["lon"]), float(row["depth"]))
    out_dir = os.path.join(out_root, eid)
    evt, asg = _load_pyocto_year_cached(pyocto_root, o.year, pyocto_velmodel)
    if evt is None:
        return dict(event_id=eid, n_sac=0, n_p=0, n_s=0, status=f"no_pyocto_tables_{o.year}")
    ev_idx = match_pyocto_event_idx(evt, o, tol_s=pyocto_tol_s)
    if ev_idx is None:
        return dict(event_id=eid, n_sac=0, n_p=0, n_s=0, status="no_pyocto_match")
    picks, matched = picks_for_event(asg, ev_idx)
    n_sac = n_p = n_s = n_err = 0
    for sr in stations.itertuples():
        sm = StationMeta(sr.Network, sr.Code, float(sr.Latitude), float(sr.Longitude),
                         float(getattr(sr, "Elevation", np.nan)))
        try:
            st = read_continuous_for_event(continuous_root, sm.network, sm.code, o, pre_s=pre_s, post_s=post_s)
            if not len(st):
                continue
            pk = picks.get((sm.network, sm.code), {})
            p_rel = (UTCDateTime(pk["P"].to_pydatetime()) - o) if "P" in pk else None
            s_rel = (UTCDateTime(pk["S"].to_pydatetime()) - o) if "S" in pk else None
            n_p += p_rel is not None; n_s += s_rel is not None
            n_sac += len(write_event_sac(st, out_dir, event_meta=ev, station_meta=sm, p_rel_s=p_rel, s_rel_s=s_rel))
        except Exception:                      # noqa: BLE001
            n_err += 1
    if n_sac and len(matched):
        matched.to_csv(os.path.join(out_dir, f"{eid}_picks.csv"), index=False)
    return dict(event_id=eid, n_sac=n_sac, n_p=n_p, n_s=n_s, n_station_errors=n_err,
                status="ok" if n_sac else "no_streams")


def export_catalog(cat, continuous_root, station_table_root, out_root, *, pyocto_root,
                   pyocto_velmodel="kim1983", skip_existing=True, progress=True) -> pd.DataFrame:
    os.makedirs(out_root, exist_ok=True)
    rows = cat.to_dict("records")
    stations_for_year, out_rows = {}, []
    for i, r in enumerate(rows):
        eid = r["event_id"]; out_dir = os.path.join(out_root, eid)
        if skip_existing and os.path.isdir(out_dir) and any(f.endswith(".sac") for f in os.listdir(out_dir)):
            out_rows.append(dict(event_id=eid, n_sac=-1, n_p=-1, n_s=-1, status="skip_existing")); continue
        yr = r["origin_utc"].year
        if yr not in stations_for_year:
            stations_for_year[yr] = load_stations_for_year(station_table_root, yr)
        try:
            out_rows.append(export_event(r, continuous_root, stations_for_year[yr], out_root,
                                         pyocto_root=pyocto_root, pyocto_velmodel=pyocto_velmodel))
        except Exception as exc:               # noqa: BLE001
            out_rows.append(dict(event_id=eid, n_sac=0, n_p=0, n_s=0, status=f"error: {type(exc).__name__}: {exc}"))
        if progress and (i + 1) % 25 == 0:
            print(f"  exported {i + 1}/{len(rows)} events", flush=True)
    summary = pd.DataFrame(out_rows)
    if len(summary):
        summary.to_csv(os.path.join(out_root, "export_summary.csv"), index=False)
    return summary


# ------------------------------------------------------------------ catalog helpers
def catalog_from_sum(sum_path, *, qc, region_bounds=None):
    """HYPOINVERSE .sum -> export catalog (event_id, origin_utc, time, lat, lon, depth, num, gap, rms, erh, erz).
    QC gate (sumio semantics), optional (latmin, latmax, lonmin, lonmax) box, same-second doublets dropped
    (a 15-char suffixed id would break the framework's fixed-width .phs header)."""
    import sumio
    cat = sumio.read_sum(sum_path)
    n0 = len(cat)
    cat = sumio.apply_qc(cat, qc=qc)
    if region_bounds is not None:
        la0, la1, lo0, lo1 = region_bounds
        cat = cat[cat.lat.between(la0, la1) & cat.lon.between(lo0, lo1)]
    cat = cat.dropna(subset=["time", "lat", "lon", "depth"]).sort_values("time").reset_index(drop=True)
    cat["event_id"] = cat["time"].apply(event_id_from_time)
    n_dup = int(cat.duplicated("event_id").sum())
    if n_dup:
        print(f"[catalog_from_sum] dropping {n_dup} same-second doublet(s)")
        cat = cat.drop_duplicates("event_id", keep="first").reset_index(drop=True)
    cat["origin_utc"] = cat["time"].apply(lambda t: UTCDateTime(t.to_pydatetime()))
    print(f"[catalog_from_sum] {len(cat)} events after QC/box (from {n0} .sum rows)")
    return cat


def write_framework_catalog(cat, out_csv, magnitudes=None, kst_offset_hours=9):
    """KMA-format event_catalog.csv the framework expects (KST whole-second times; load_catalog does
    origin = KST - 9 h so event_id round-trips). `magnitudes`: {event_id: ML} for known events, else 0.0."""
    kst = cat["time"] + pd.Timedelta(hours=kst_offset_hours)
    mags = [float((magnitudes or {}).get(e, 0.0)) for e in cat["event_id"]]
    df = pd.DataFrame({"Year": kst.dt.year, "Month": kst.dt.month, "Day": kst.dt.day, "Hour": kst.dt.hour,
                       "Minute": kst.dt.minute, "Second": kst.dt.second, "Latitude": cat["lat"],
                       "Longitude": cat["lon"], "Depth": cat["depth"], "Magnitude": mags})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[catalog] {len(df)} events -> {out_csv}")
    return out_csv
