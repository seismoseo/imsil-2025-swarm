"""Configuration-segmented MAD detection (header-derived station-config segments). Buan-validated.

The whole-day MAD assumes the network-CC trace is stationary across the day; it is not when the
recording configuration changes mid-day (a station's data onsets/offsets or gaps): the CC baseline
steps, the single daily MAD is biased low for the busier span, and a burst of transition-artifact
detections fires at the changeover. This module computes the threshold PER config-stationary
segment. Boundaries come from miniSEED HEADER coverage (the loaded array is mean-subtracted, so gap
samples are a nonzero constant and the configuration CANNOT be inferred from it).
Relevant here: KS day files have occasional gaps; the JNU stations switch on 2025-12 (tars) and the
Earthworm feed has known outages (2026-09-01/02 seams).
"""
import glob
import os

import numpy as np
import pandas as pd
from obspy import read

import config as C


def day_coverage(stations, prefixes, year, doy):
    """Per (station_index, comp) real-data coverage [start_s, end_s] (s from day start) from headers."""
    day0 = C.day_start(year, doy)
    cov = {}
    for si, ns in enumerate(stations):
        net, sta = ns.split("."); pre = prefixes[si]
        for comp in C.COMPONENTS:
            f = []
            for p_ in [pre] + [q for q in ("HH", "HG", "EL", "BH", "DP") if q != pre]:   # preferred band, else per-day fallback
                f = glob.glob(f"{C.CONTINUOUS}/{net}/{sta}/{p_}{comp}.D/*.D.{year}.{doy:03d}")
                if f:
                    break
            if not f:
                continue
            try:
                st = read(f[0], headonly=True)
                cov[(si, comp)] = (min(t.stats.starttime for t in st) - day0,
                                   max(t.stats.endtime for t in st) - day0)
            except Exception:
                pass
    return cov


def template_change_points(Pw_tid, Sw_tid, cov, n_corr, merge_s=1800.0, s_comp=None):
    """Change-point sample indices for ONE template's weighted channels (onset/offset times),
    merged within `merge_s`. Pw_tid: (nS,) P weights; Sw_tid: (nS,2) S weights. s_comp[si] == 'Z' means
    the S pass runs on the vertical at that station (slot 0 <- Z), otherwise slots are N, E."""
    nN, nE = C.COMPONENTS[0], C.COMPONENTS[1]
    cps = set()
    for si in range(Pw_tid.shape[0]):
        chans = []
        if Pw_tid[si] > 0: chans.append("Z")
        s0 = "Z" if (s_comp is not None and s_comp[si] == "Z") else nN
        if Sw_tid[si, 0] > 0: chans.append(s0)
        if Sw_tid[si, 1] > 0: chans.append(nE)
        for comp in chans:
            if (si, comp) in cov:
                s0, e0 = cov[(si, comp)]
                if s0 > 60: cps.add(float(s0))
                if e0 < 86400 - 60: cps.add(float(e0))
    merged = []
    for c in sorted(cps):
        if not merged or c - merged[-1] > merge_s:
            merged.append(c)
    return [int(round(c * C.FS / C.STEP)) for c in merged if 0 < int(round(c * C.FS / C.STEP)) < n_corr]   # CC-index units


def segmented_threshold(cc, cps, mad_k=None, min_len_s=600.0, floor=None):
    """Per-segment (median + mad_k*MAD) threshold array; segments split at `cps` (sample indices),
    short segments merged forward to >= min_len_s. Optional absolute `floor` as a backstop."""
    mad_k = C.MAD_K if mad_k is None else mad_k
    edges = sorted(set([0] + list(cps) + [int(cc.size)]))
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] >= int(min_len_s * C.FS / C.STEP) or e == edges[-1]:
            keep.append(e)
    thr = np.empty(cc.size, np.float64)
    for a, b in zip(keep[:-1], keep[1:]):
        k = max(1, int(getattr(C, "MAD_SUBSAMPLE", 1)))
        seg = cc[a:b:k]
        c = np.median(seg); m = np.median(np.abs(seg - c))
        t = c + mad_k * m if m > 0 else np.inf
        if floor is not None:
            t = max(t, floor)
        thr[a:b] = t
    return thr


def detect_day_segmented(cc_sum, sh, day_start, batch_ids, cov,
                         merge_s=1800.0, min_len_s=600.0, floor=None, s_comp=None):
    """Drop-in alternative to detect.detect_day for the Shelly two-pass, per-config-segment threshold."""
    rows = []
    dur = int(C.TEMPLATE_DURATION_S * C.FS / C.STEP)
    cuspid, ref_s = sh["cuspid"], sh["ref_s"]
    Pw, Sw, Pmv, Smv = sh["P_weights"], sh["S_weights"], sh["P_moveouts"], sh["S_moveouts"]
    n_corr = cc_sum.shape[1]
    for b, tid in enumerate(batch_ids):
        cc = cc_sum[b]
        cps = template_change_points(Pw[tid, :, 0], Sw[tid], cov, n_corr, merge_s, s_comp)
        thr = segmented_threshold(cc, cps, C.MAD_K, min_len_s, floor)
        cand = np.flatnonzero(cc > thr)
        if cand.size == 0:
            continue
        if os.environ.get("FMF_DEBUG") and cand.size > 50000:
            print(f"      tpl {tid}: {cand.size} candidates above threshold (thr median {np.median(thr):.4f}, {len(cps)} change points)", flush=True)
        d_mv = (Smv[tid, :, 0] - Pmv[tid, :, 0]); d_mv = d_mv[d_mv != 0]
        med_sp = int(np.median(d_mv) / C.STEP) if d_mv.size else dur
        search = min(3 * dur, max(med_sp, dur))
        order = cand[np.argsort(cc[cand])[::-1]]
        taken, occupied = [], np.zeros(cc.size, bool)
        for idx in order:
            lo, hi = max(0, idx - search // 2), min(cc.size, idx + search // 2)
            if not occupied[lo:hi].any():
                taken.append(idx); occupied[lo:hi] = True
        for idx in sorted(taken):
            origin = (day_start + (idx * C.STEP) / C.FS - float(ref_s[tid])).datetime
            rows.append((int(cuspid[tid]), tid, origin, float(cc[idx]), int(idx), len(cps) + 1))
    return pd.DataFrame(rows, columns=["template_cuspid", "template_idx", "origin_utc", "cc",
                                       "corr_idx", "n_config_segments"])
