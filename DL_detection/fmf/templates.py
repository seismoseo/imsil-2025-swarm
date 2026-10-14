"""Build FMF templates for the Haenam relocated events (catalogs/haenam_templates.csv).

DESIGN (Buan-validated): the template WAVEFORMS are cut from the SAME continuous archive that the
scan reads, with the SAME processing (detrend -> resample 100 Hz -> band-pass 2-15 Hz on the full
day, then slice). Only the PICKS (P = SAC `a`, S = SAC `t0`, relative to the SAC reference time =
origin) come from the relocation run's per-event SAC bundles (C.WF_EXPORTS). This guarantees the
template and the scan use the *same instrument and processing* at every station, so a template
self-detects at ~1.0.

Output (templates_cache/haenam_templates{_nearNN}.npz), FMF layout:
  waveforms (n_tpl, n_sta, n_comp, L) float32   -- Z last
  moveouts  (n_tpl, n_sta, n_comp) int32        -- samples, per-template min = 0
  weights   (n_tpl, n_sta, n_comp) float32      -- 0 where no pick; sum = 1 per template
  ref_s     (n_tpl,) float64                    -- detected_origin = day_start + i/FS - ref_s
  stations  (n_sta,) 'NET.STA';  channels (n_sta,) band prefix used ('HH'/'HG'/...)
  components ['N','E','Z'];  cuspid, origin_utc, event_id, sequence (n_tpl,)
Phase per component: Z<-P (SAC a), N/E<-S (SAC t0). Stations kept if a pick appears in
>= MIN_TEMPLATE_OCCUR templates AND the station exists in the continuous archive.
Templates whose origin day has NO continuous data (e.g. the 2020 set until the 2020 archive
arrives) are DROPPED with a message — never silently zero-filled.

Run:  conda run -n fmf python fmf/templates.py            (env FMF_SEQUENCES=2026 to restrict)
"""
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from obspy import read, UTCDateTime

import config as C

CHAN_PREF = ["HH", "HG", "EL", "BH", "DP"]
SAC_UNDEF = -12345.0


def template_events():
    ev = pd.read_csv(C.TEMPLATE_CSV, dtype={"event_id": str})
    ev = ev[(ev.confidence == C.TEMPLATE_CONFIDENCE) & (ev.sequence.astype(str).isin(C.TEMPLATE_SEQUENCES))]
    return ev.sort_values("cuspid").reset_index(drop=True)


def _sac_for(event_id, net, sta, comp):
    hits = []
    for root in C.WF_EXPORTS:
        hits += glob.glob(f"{root}/{event_id}/*.{net}.{sta}.??{comp}.sac")
    if not hits:
        return None
    hits.sort(key=lambda p: next((i for i, c in enumerate(CHAN_PREF)
                                  if os.path.basename(p).split(".")[3].startswith(c)), 99))
    return hits[0]


def _pick_abs(sacpath, phase):
    """Absolute UTC arrival time of `phase` from a SAC bundle file, or None.
    arrival = (SAC reference time) + (a or t0); the reference time == origin in these bundles."""
    tr = read(sacpath, headonly=True)[0]
    h = tr.stats.sac
    v = h.get("a") if phase == "P" else h.get("t0")
    if v is None or v == SAC_UNDEF:
        return None
    ref = tr.stats.starttime - h.b
    return ref + float(v)


def _cont_prefix(net, sta):
    """Highest-preference band code present for this station in the continuous archive."""
    for pre in CHAN_PREF:
        if glob.glob(f"{C.CONTINUOUS}/{net}/{sta}/{pre}Z.D") or \
           glob.glob(f"{C.CONTINUOUS}/{net}/{sta}/{pre}[ZNE].D"):
            return pre
    return None


def _process_day(net, sta, prefix, comp, year, julday):
    """Read+detrend+resample(100)+bandpass(2-15) the FULL day for one channel — exactly as
    scan.py does — so a window sliced here matches the scanned data bit-for-bit."""
    f = []
    for pre in [prefix] + [q for q in CHAN_PREF if q != prefix]:      # preferred band, else per-day fallback (MUN EL->HH)
        f = glob.glob(f"{C.CONTINUOUS}/{net}/{sta}/{pre}{comp}.D/*.D.{year}.{julday:03d}")
        if f:
            break
    if not f:
        return None
    try:
        from scan import _normalize_rates
        st = _normalize_rates(read(f[0])); st.merge(method=1, fill_value=0); tr = st[0]
        tr.detrend("linear")
        if abs(tr.stats.sampling_rate - C.FS) > 1e-6:
            tr.resample(C.FS)
        tr.filter("bandpass", freqmin=C.BANDPASS[0], freqmax=C.BANDPASS[1],
                  corners=4, zerophase=True)
        return tr
    except Exception:
        return None


def _build_group(arg):
    """Worker: one (station, channel, comp, day) -> process the day once, slice every event
    window that falls on it. Returns [(ti, si, ci, float32[L]), ...]."""
    net, sta, prefix, comp, si, ci, year, julday, tasks = arg
    tr = _process_day(net, sta, prefix, comp, year, julday)
    if tr is None:
        return []
    out = []
    for ti, ws_epoch in tasks:
        start = UTCDateTime(ws_epoch)
        seg = tr.slice(start, start + (C.WIN_SAMPLES - 1) / C.FS)
        w = seg.data.astype(np.float32)
        if w.size < C.WIN_SAMPLES:
            w = np.pad(w, (0, C.WIN_SAMPLES - w.size))
        else:
            w = w[:C.WIN_SAMPLES]
        if np.max(np.abs(w)) < 1e-12:
            continue
        w -= w.mean()
        out.append((ti, si, ci, w))
    return out


def _cnet(n):
    return getattr(C, "NET_REMAP", {}).get(n, n)


def _in_excluded_epoch(net, sta, t):
    for a, b in getattr(C, "EXCLUDE_EPOCHS", {}).get((net, sta), []):
        if UTCDateTime(a) <= t <= UTCDateTime(b) + 86400:
            return True
    return False


def station_list():
    """Explicit station set from C.STATION_LIST (network,station,horizontals) or None -> occurrence rule."""
    f = getattr(C, "STATION_LIST", "")
    if not f or not os.path.exists(f):
        return None
    t = pd.read_csv(f, comment="#")
    if "s_exclude" not in t.columns:
        t["s_exclude"] = 0                                   # 1 -> no S window at this station (P only): horizontals re-oriented inside the period
    return [(r.network, r.station, int(r.horizontals), int(r.s_exclude)) for r in t.itertuples()]


def build(verbose=True):
    from concurrent.futures import ProcessPoolExecutor
    ev = template_events()
    stset = pd.read_csv(C.STATIONS_CSV)
    valid = {(r.Network, r.Code) for r in stset.itertuples() if _cnet(r.Network) in C.NETWORKS}
    slist = station_list()
    if slist is not None:
        valid = {(n, s) for n, s, *_ in slist}

    # ---- pass 1: station selection + cache picks (abs arrival) per (event, station, comp) ----
    occur = defaultdict(int)
    picks = {}
    origins = {}
    for ti, r in enumerate(ev.itertuples()):
        origins[ti] = UTCDateTime(str(r.origin_time_UTC))
        seen = set()
        sacs = []
        for root in C.WF_EXPORTS:
            sacs += glob.glob(f"{root}/{r.event_id}/*.sac")
        for f in sacs:
            p = os.path.basename(f).split(".")
            onet, sta = p[1], p[2]
            if (onet, sta) not in valid:
                continue
            cn = _cnet(onet)
            if _in_excluded_epoch(cn, sta, origins[ti]):
                continue
            got = False
            for comp in C.COMPONENTS:
                if (ti, cn, sta, comp) in picks:
                    got = True
                    continue
                sf = _sac_for(r.event_id, onet, sta, comp)
                if sf is None:
                    continue
                ab = _pick_abs(sf, C.PHASE_FOR_COMP[comp])
                if ab is not None:
                    picks[(ti, cn, sta, comp)] = ab
                    got = True
            if got and (cn, sta) not in seen:
                seen.add((cn, sta)); occur[(cn, sta)] += 1
    if slist is not None:
        chosen = [(n, s) for n, s, *_ in slist if occur.get((n, s), 0) > 0]
        horiz = {(n, s): h for n, s, h, _ in slist}; s_excl = {(n, s): x for n, s, _, x in slist}
        print(f"[templates] STATION_LIST {os.path.basename(C.STATION_LIST)}: {len(chosen)}/{len(slist)} stations carry picks; "
              f"S on horizontals at {sum(horiz[k] for k in chosen)} of them, S excluded at {sum(s_excl[k] for k in chosen)}, on Z elsewhere")
    else:
        chosen = sorted(k for k, n in occur.items() if n >= C.MIN_TEMPLATE_OCCUR)
        horiz = {k: 1 for k in chosen}; s_excl = {k: 0 for k in chosen}
    if C.NEAR_SOURCE_RADIUS_KM:
        from obspy.geodetics import gps2dist_azimuth
        clat, clon = C.NEAR_SOURCE_CENTER
        coord = {(_cnet(r.Network), r.Code): (r.Latitude, r.Longitude) for r in stset.itertuples()}
        def _within(k):
            ll = coord.get(k)
            return ll is not None and gps2dist_azimuth(clat, clon, ll[0], ll[1])[0] / 1000.0 <= C.NEAR_SOURCE_RADIUS_KM
        n0 = len(chosen); chosen = [k for k in chosen if _within(k)]
        print(f"[templates] NEAR-SOURCE: kept {len(chosen)}/{n0} stations within "
              f"{C.NEAR_SOURCE_RADIUS_KM:.0f} km of {C.NEAR_SOURCE_CENTER}")
    stations, prefixes = [], []
    for net, sta in chosen:
        pre = _cont_prefix(net, sta)
        if pre is not None:
            stations.append((net, sta)); prefixes.append(pre)
    if verbose:
        from collections import Counter
        print(f"[templates] {len(ev)} candidate events (sequences {C.TEMPLATE_SEQUENCES}); "
              f"{len(stations)} stations x {len(C.COMPONENTS)} comps "
              f"(picks in >= {C.MIN_TEMPLATE_OCCUR} templates, present in continuous)")
        print("  by net:", dict(Counter(n for n, _ in stations)))
        print("  channel mix:", dict(Counter(prefixes)))

    sidx = {(n, s): i for i, (n, s) in enumerate(stations)}
    nT, nS, nC, L = len(ev), len(stations), len(C.COMPONENTS), C.WIN_SAMPLES

    groups = defaultdict(list)
    mv_raw = np.full((nT, nS, nC), np.nan)
    for (ti, net, sta, comp), ab in picks.items():
        if (net, sta) not in sidx:
            continue
        si = sidx[(net, sta)]; ci = C.COMPONENTS.index(comp); pre = prefixes[si]
        ws = ab - C.PRE_FOR_COMP[comp]
        mv_raw[ti, si, ci] = ws - origins[ti]
        groups[(net, sta, pre, comp, si, ci, ws.year, ws.julday)].append((ti, ws.timestamp))
    if verbose:
        print(f"[templates] {len(groups)} (station,comp,day) groups to read")

    waveforms = np.zeros((nT, nS, nC, L), np.float32)
    weights = np.zeros((nT, nS, nC), np.float32)
    args = [(net, sta, pre, comp, si, ci, yr, jd, tasks)
            for (net, sta, pre, comp, si, ci, yr, jd), tasks in groups.items()]
    done = 0
    with ProcessPoolExecutor(max_workers=C.N_WORKERS) as ex:
        for res in ex.map(_build_group, args, chunksize=8):
            for ti, si, ci, w in res:
                waveforms[ti, si, ci] = w
                weights[ti, si, ci] = 1.0
            done += 1
            if verbose and done % 200 == 0:
                print(f"  processed {done}/{len(args)} groups")

    # ---- drop templates with no live channel (origin day absent from the archive) ----
    live = (weights > 0).reshape(nT, -1).sum(axis=1)
    keep = np.where(live > 0)[0]
    dropped = ev[live == 0]
    if len(dropped):
        print(f"[templates] DROPPED {len(dropped)} templates with no continuous data on their origin day:")
        for r in dropped.itertuples():
            print(f"    {r.event_id}  {r.origin_time_UTC}  ({r.sequence})")
    ev = ev.iloc[keep].reset_index(drop=True)
    waveforms, weights, mv_raw = waveforms[keep], weights[keep], mv_raw[keep]
    nT = len(ev)

    moveouts = np.zeros((nT, nS, nC), np.int32)
    ref_s = np.zeros(nT)
    for ti in range(nT):
        m = np.where(weights[ti] > 0, mv_raw[ti], np.nan)
        present = np.isfinite(m)
        ref = np.nanmin(m); ref_s[ti] = ref
        moveouts[ti] = np.where(present, np.round((m - ref) * C.FS), 0).astype(np.int32)
        weights[ti] /= weights[ti].sum()

    fill = (weights > 0).sum() / max(1, nT * nS * nC)
    live = (weights > 0).reshape(nT, -1).sum(axis=1)
    if verbose:
        print(f"[templates] {nT} templates kept; channel fill {fill:.1%}; "
              f"median live channels/template {int(np.median(live)) if nT else 0}")
    os.makedirs(os.path.dirname(C.TEMPLATE_NPZ), exist_ok=True)
    np.savez_compressed(
        C.TEMPLATE_NPZ, waveforms=waveforms, moveouts=moveouts, weights=weights,
        ref_s=ref_s, stations=np.array([f"{n}.{s}" for n, s in stations]),
        channels=np.array(prefixes), components=np.array(C.COMPONENTS),
        cuspid=ev.cuspid.values.astype(np.int64),
        horizontals=np.array([horiz[(n, s)] for n, s in stations], dtype=np.int8),
        s_exclude=np.array([s_excl.get((n, s), 0) for n, s in stations], dtype=np.int8),
        origin_utc=ev.origin_time_UTC.values.astype(str),
        event_id=ev.event_id.values.astype(str),
        sequence=ev.sequence.values.astype(str))
    print(f"[templates] wrote {C.TEMPLATE_NPZ}  ({nT} tpl, {nS} sta, {nC} comp, L={L}, from continuous)")
    return C.TEMPLATE_NPZ


if __name__ == "__main__":
    build()
