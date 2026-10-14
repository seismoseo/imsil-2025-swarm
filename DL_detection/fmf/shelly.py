"""Shelly (2024, Sci. Adv.)-style P/S templates and the GPU two-pass scan — Haenam, multi-year.

Shelly extracts P and S packets SEPARATELY with different window lengths and truncates the P window
so it never overlaps S. FMF's GPU kernel needs ONE fixed length, and per-channel truncation is
CPU-only. We get Shelly's scheme on the GPU because the network CC is a *linear sum over channels*
(Beauce eq. 1): a **P pass** on the verticals (1.5 s window) + an **S pass** on the horizontals
(2.0 s window), and the two network-CC traces are ADDED. The Z channel is dropped at stations where
S-P < SP_DROP (its window would reach S).

Picks are reused from the Beauce template npz (templates.py); waveforms are cut from the same
continuous archive (full-day processing) so template and scan share instrument + processing.

npz: P_waveforms (nT,nS,1,LP) / S_waveforms (nT,nS,2,LS); P_moveouts,S_moveouts (rel a COMMON
global reference); P_weights,S_weights (one global weight per live channel, summing to 1 over P+S);
ref_s; stations, channels, cuspid, origin_utc, event_id, sequence; sp_drop count.

Run:  conda run -n fmf python fmf/shelly.py build
      conda run -n fmf python fmf/shelly.py YEAR [D0 D1]        # GPU scan, resumable per day
      MAD_SEGMENTED=1 ... (config-segmented threshold; output dir gets a _seg suffix)
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from obspy import UTCDateTime

import config as C
import templates as T

# Phase windows (Buan-tuned, kept): P (vertical) [P-0.5, P+1.0] = 1.5 s; S (horizontals)
# [S-0.5, S+1.5] = 2.0 s. A P window ending at P+1.0 only reaches S when S-P <= 1.0 s; at Haenam
# (20 km deep) the closest station JN09 has S-P ~2.5 s, so no vertical is expected to be dropped.
P_WIN, S_WIN, PRE, SP_DROP = 1.5, 2.0, 0.5, 0.5
LP, LS = int(round(P_WIN * C.FS)), int(round(S_WIN * C.FS))
SHELLY_NPZ = C.SHELLY_NPZ
DETECT_DIR = C.DETECT_DIR
# 2026-10-04: transparent huge pages off for this process and its loader pool (inherited): on this fragmented shared box every page
# fault of the multi-GB daily arrays stalled in kernel compaction (~300 stalls/s, ~70 % of the scan's CPU time in the kernel). Memory
# mapping only, results bit-identical (verified on 2022 DOY 056: 654 s -> 158 s per day). FMF_KEEP_THP=1 leaves THP on.
if not os.environ.get("FMF_KEEP_THP"):
    try:
        import ctypes as _ct; _ct.CDLL(None).prctl(41, 1, 0, 0, 0)        # PR_SET_THP_DISABLE
    except Exception:
        pass
# 2026-10-04 speed-up (user-approved): the per-batch CPU steps on the ~2 GB correlation arrays were single-threaded and memory-bound
# (np.nan_to_num: three full passes with temporaries; the S-into-P sum). Both are elementwise, so splitting them over threads gives
# bit-identical arrays (verified on 2022 DOY 056 against the official output before deployment). FMF_CPU_THREADS=1 restores the old path.
from concurrent.futures import ThreadPoolExecutor as _TPE
_NT = int(os.environ.get("FMF_CPU_THREADS", "8"))
_POOL = _TPE(max_workers=_NT) if _NT > 1 else None

def _nan_to_num_(a):
    """np.nan_to_num(a, copy=False) on contiguous chunks in parallel (numpy releases the GIL in these ufuncs)."""
    if _POOL is None or not a.flags.c_contiguous or a.size < (1 << 22):
        np.nan_to_num(a, copy=False); return
    f = a.reshape(-1); b = np.linspace(0, f.size, 4 * _NT + 1).astype(np.int64)
    list(_POOL.map(lambda i: np.nan_to_num(f[b[i]:b[i + 1]], copy=False), range(len(b) - 1)))

def _iadd_(dst, src):
    """dst += src (equal-shape 2-D views), one row per task."""
    if _POOL is None:
        dst += src; return
    list(_POOL.map(lambda r: np.add(dst[r], src[r], out=dst[r]), range(dst.shape[0])))
os.makedirs(DETECT_DIR, exist_ok=True)


def _build_group(arg):
    net, sta, prefix, comp, ci, si, year, julday, L, tasks = arg
    tr = T._process_day(net, sta, prefix, comp, year, julday)
    if tr is None:
        return []
    out = []
    for ti, phase, ws_epoch, p_epoch in tasks:
        seg = tr.slice(UTCDateTime(ws_epoch), UTCDateTime(ws_epoch) + (L - 1) / C.FS)
        w = seg.data.astype(np.float32)
        if w.size < L:
            w = np.pad(w, (0, L - w.size))
        else:
            w = w[:L]
        if np.max(np.abs(w)) < 1e-12:
            continue
        # per-channel SNR (window RMS / pre-P noise RMS on the SAME channel/day) is STORED, not gated here:
        # the gate C.SNR_MIN is applied at scan time from these values, so it can be changed without a rebuild
        pref = UTCDateTime(p_epoch) if p_epoch is not None else UTCDateTime(ws_epoch) - 3.0
        nz = tr.slice(pref - C.NOISE_WIN[0], pref - C.NOISE_WIN[1]).data.astype(np.float32)
        snr = float(np.sqrt(np.mean(w ** 2)) / np.sqrt(np.mean(nz ** 2))) if (nz.size >= 100 and np.max(np.abs(nz)) > 1e-12) else np.nan
        out.append((ti, phase, si, ci, w - w.mean(), snr))
    return out


def build(verbose=True):
    from concurrent.futures import ProcessPoolExecutor
    b = np.load(C.TEMPLATE_NPZ, allow_pickle=True)
    stations = list(b["stations"]); channels = list(b["channels"]); cuspid = b["cuspid"]
    comps = [str(c) for c in b["components"]]; origins = [str(o) for o in b["origin_utc"]]
    ci_N, ci_E, ci_Z = comps.index("N"), comps.index("E"), comps.index("Z")
    mvB, refB, wB = b["moveouts"], b["ref_s"], b["weights"]
    nT, nS = len(cuspid), len(stations)
    horiz = b["horizontals"].astype(bool) if "horizontals" in b.files else np.ones(nS, bool)
    s_excl = b["s_exclude"].astype(bool) if "s_exclude" in b.files else np.zeros(nS, bool)   # P only at these stations
    s_comp = np.array(["H" if h else "Z" for h in horiz])      # where the S pass runs: N/E or Z

    # recover absolute picks (rel origin) from the Beauce moveouts (undo that pass's pre-pick)
    groups = defaultdict(list)
    Pws = np.full((nT, nS), np.nan); Sws = np.full((nT, nS, 2), np.nan)
    ndrop = 0
    net_sta = [s.split(".") for s in stations]
    for ti in range(nT):
        o = UTCDateTime(origins[ti])
        for si in range(nS):
            net, sta = net_sta[si]; pre = str(channels[si])
            P = refB[ti] + mvB[ti, si, ci_Z] / C.FS + C.P_PRE if wB[ti, si, ci_Z] > 0 else None
            s_ci = ci_N if wB[ti, si, ci_N] > 0 else (ci_E if wB[ti, si, ci_E] > 0 else None)
            S = refB[ti] + mvB[ti, si, s_ci] / C.FS + C.S_PRE if s_ci is not None else None
            sp = (S - P) if (P is not None and S is not None) else None
            p_abs = (o + P).timestamp if P is not None else None
            if P is not None and (sp is None or sp >= SP_DROP):
                ws = P - PRE; Pws[ti, si] = ws
                ab = o + ws; groups[(si, ci_Z, pre, ab.year, ab.julday, LP)].append((ti, "P", ab.timestamp, p_abs))
            elif P is not None:
                ndrop += 1
            if s_excl[si]:
                pass                                          # S excluded at this station (re-oriented horizontals): P only
            elif horiz[si]:
                for j, ci in ((0, ci_N), (1, ci_E)):
                    if wB[ti, si, ci] > 0 and S is not None:
                        ws = S - PRE; Sws[ti, si, j] = ws
                        ab = o + ws; groups[(si, ci, pre, ab.year, ab.julday, LS)].append((ti, "S", ab.timestamp, p_abs))
            elif S is not None:                                   # S pass on the vertical: slot 0 <- Z, slot 1 unused
                ws = S - PRE; Sws[ti, si, 0] = ws
                ab = o + ws; groups[(si, ci_Z, pre, ab.year, ab.julday, LS)].append((ti, "S", ab.timestamp, p_abs))
    if verbose:
        print(f"[shelly] {nT} templates x {nS} stations | P win {P_WIN}s, S win {S_WIN}s, pre {PRE}s | "
              f"S on Z at {int((~horiz).sum())} stations, on N/E at {int(horiz.sum())} | SNR gate >= {getattr(C, 'SNR_MIN', 0)}")
        print(f"[shelly] Z dropped at {ndrop} (template,station) where S-P < {SP_DROP}s")
        print(f"[shelly] {len(groups)} (station,comp,day) groups to read")

    Pwf = np.zeros((nT, nS, 1, LP), np.float32); Pmv = np.zeros((nT, nS, 1), np.int32); Pw = np.zeros((nT, nS, 1), np.float32)
    Swf = np.zeros((nT, nS, 2, LS), np.float32); Smv = np.zeros((nT, nS, 2), np.int32); Sw = np.zeros((nT, nS, 2), np.float32)
    Pmask = np.zeros((nT, nS), bool); Smask = np.zeros((nT, nS, 2), bool)
    Psnr = np.full((nT, nS, 1), np.nan, np.float32); Ssnr = np.full((nT, nS, 2), np.nan, np.float32)
    args = [(net_sta[si][0], net_sta[si][1], pre, comps[ci], ci, si, yr, jd, L, tk)
            for (si, ci, pre, yr, jd, L), tk in groups.items()]
    done = 0
    with ProcessPoolExecutor(max_workers=C.N_WORKERS) as ex:
        for res in ex.map(_build_group, args, chunksize=8):
            for ti, phase, si, ci, w, snr in res:
                if phase == "P":
                    Pwf[ti, si, 0] = w; Pmask[ti, si] = True; Psnr[ti, si, 0] = snr
                else:
                    j = 1 if ci == ci_E else 0                    # N or Z -> slot 0, E -> slot 1
                    Swf[ti, si, j] = w; Smask[ti, si, j] = True; Ssnr[ti, si, j] = snr
            done += 1
            if verbose and done % 400 == 0:
                print(f"  processed {done}/{len(args)} groups")

    # build keeps every template with at least one channel (Buan behaviour); SNR / live-channel gates run at scan time
    liveP = Pmask.sum(1); liveAll = liveP + Smask.reshape(nT, -1).sum(1)
    keep = np.where(liveAll >= 1)[0]
    if len(keep) < nT:
        print(f"[shelly] DROPPED {nT - len(keep)} templates with no live channel:")
        for ti in np.where(~np.isin(np.arange(nT), keep))[0]:
            print(f"    {b['event_id'][ti]}  {origins[ti]}  P {liveP[ti]} / all {liveAll[ti]}")
    sel = keep
    Pwf, Pmv, Pw, Swf, Smv, Sw = Pwf[sel], Pmv[sel], Pw[sel], Swf[sel], Smv[sel], Sw[sel]
    Pmask, Smask, Psnr, Ssnr, Pws, Sws = Pmask[sel], Smask[sel], Psnr[sel], Ssnr[sel], Pws[sel], Sws[sel]
    cuspid = cuspid[sel]; origins = [origins[i] for i in sel]; ev_id = b["event_id"][sel]; seq = b["sequence"][sel]
    nT = len(sel)
    ref_s = np.zeros(nT)
    for ti in range(nT):
        starts = []
        for si in range(nS):
            if Pmask[ti, si]: starts.append(Pws[ti, si])
            for j in range(2):
                if Smask[ti, si, j]: starts.append(Sws[ti, si, j])
        if not starts: continue
        gref = min(starts); ref_s[ti] = gref
        w = 1.0 / len(starts)
        for si in range(nS):
            if Pmask[ti, si]:
                Pmv[ti, si, 0] = int(round((Pws[ti, si] - gref) * C.FS)); Pw[ti, si, 0] = w
            for j in range(2):
                if Smask[ti, si, j]:
                    Smv[ti, si, j] = int(round((Sws[ti, si, j] - gref) * C.FS)); Sw[ti, si, j] = w
    if verbose:
        liveP = Pmask.sum(1); liveS = Smask.reshape(nT, -1).sum(1)
        print(f"[shelly] median live: {int(np.median(liveP))} P(Z) + {int(np.median(liveS))} S(horiz) channels/template")
    np.savez_compressed(SHELLY_NPZ, P_waveforms=Pwf, S_waveforms=Swf, P_moveouts=Pmv, S_moveouts=Smv,
                        P_weights=Pw, S_weights=Sw, ref_s=ref_s, stations=np.array(stations),
                        channels=np.array(channels), components=np.array(["N", "E", "Z"]), s_comp=s_comp,
                        P_snr=Psnr, S_snr=Ssnr, cuspid=cuspid, origin_utc=np.array(origins),
                        event_id=ev_id, sequence=seq, sp_drop=ndrop)
    print(f"[shelly] wrote {SHELLY_NPZ}  (P {LP} samp, S {LS} samp)")
    return SHELLY_NPZ


# ---------------------------------------------------------------- scan (two-pass GPU)
def _load():
    z = np.load(SHELLY_NPZ, allow_pickle=True); return {k: z[k] for k in z.files}


def scan_day(sh, year, doy, log=print):
    import fast_matched_filter as fmf
    from detect import detect_day
    stations = [s for s in sh["stations"]]; prefixes = [str(p) for p in sh["channels"]]
    comps = [str(c) for c in sh["components"]]; ci_N, ci_E, ci_Z = comps.index("N"), comps.index("E"), comps.index("Z")
    data, present, day0 = scan.load_day(stations, prefixes, year, doy)
    if not present.any():
        return None
    nT, nS = sh["P_weights"].shape[0], len(stations)
    N = data.shape[-1]

    Pw = sh["P_weights"].copy(); Sw = sh["S_weights"].copy()
    Pw[:, ~present, :] = 0.0; Sw[:, ~present, :] = 0.0
    # per-channel SNR gate (C.SNR_MIN, 0 = off) from the SNR stored at build time; then the live-channel gates
    if getattr(C, "SNR_MIN", 0) > 0 and "P_snr" in sh:
        # gate where affordable: a template keeps the SNR mask only if >= MIN_LIVE_CH channels survive it,
        # otherwise it is scanned ungated (no template is ever dropped by the SNR gate alone)
        Pm = Pw.copy(); Sm = Sw.copy()
        Pm[np.isfinite(sh["P_snr"]) & (sh["P_snr"] < C.SNR_MIN)] = 0.0
        Sm[np.isfinite(sh["S_snr"]) & (sh["S_snr"] < C.SNR_MIN)] = 0.0
        okP = (Pm[:, :, 0] > 0).sum(1) >= getattr(C, "MIN_LIVE_P", 0)
        ok = okP & (((Pm[:, :, 0] > 0).sum(1) + (Sm > 0).reshape(nT, -1).sum(1)) >= getattr(C, "MIN_LIVE_CH", 4))
        Pw[ok] = Pm[ok]; Sw[ok] = Sm[ok]; del Pm, Sm
    liveP = (Pw[:, :, 0] > 0).sum(1); liveAll = liveP + (Sw > 0).reshape(nT, -1).sum(1)
    Pw[liveP < getattr(C, "MIN_LIVE_P", 1)] = 0.0; Sw[liveP < getattr(C, "MIN_LIVE_P", 1)] = 0.0
    Pw[liveAll < getattr(C, "MIN_LIVE_CH", 1)] = 0.0; Sw[liveAll < getattr(C, "MIN_LIVE_CH", 1)] = 0.0
    # sensor-epoch mask: a template and a scan day on different sides of an excluded epoch never share the station
    if getattr(C, "EXCLUDE_EPOCHS", None):
        torig = np.array([UTCDateTime(str(o)).timestamp for o in sh["origin_utc"]])
        for (net, sta), spans in C.EXCLUDE_EPOCHS.items():
            key = f"{net}.{sta}"
            if key not in stations:
                continue
            si = stations.index(key)
            for a, bnd in spans:
                t0, t1 = UTCDateTime(a).timestamp, UTCDateTime(bnd).timestamp + 86400
                day_in = t0 <= day0.timestamp < t1
                tpl_in = (torig >= t0) & (torig < t1)
                off = np.where(tpl_in != day_in)[0]
                Pw[off, si, :] = 0.0; Sw[off, si, :] = 0.0
    tot = Pw.reshape(nT, -1).sum(1) + Sw.reshape(nT, -1).sum(1)
    ids = np.where(tot > 0)[0]
    if ids.size == 0:
        return None
    for arr in (Pw, Sw):
        arr[ids] /= tot[ids][:, None, None]
    # stations present AND live in this template (for the >=N-station gate downstream)
    tpl_has = (Pw[:, :, 0] > 0) | (Sw > 0).any(axis=2)             # after the SNR / live / epoch masks
    n_present_tpl = tpl_has.sum(axis=1)

    s_comp = [str(x) for x in sh["s_comp"]] if "s_comp" in sh else ["H"] * nS
    # S pass split by station type (exact: the network CC is a weighted sum over channels): S-on-Z stations get a
    # 1-component array (no wasted zero channel), horizontal stations a 2-component array
    # only PRESENT stations enter each GPU call (exact: absent stations already have zero weight). FMF takes a ~9x slower
    # path on all-zero data ("correlation computations were skipped"): an all-zero JN block cost ~100 s/day before Dec 2025.
    iz = np.array([i for i in range(nS) if s_comp[i] == "Z" and present[i]], int); ih = np.array([i for i in range(nS) if s_comp[i] != "Z" and present[i]], int)
    ip = np.array([i for i in range(nS) if present[i]], int)
    data_Z = np.ascontiguousarray(data[ip][:, [ci_Z], :])
    data_SZ = np.ascontiguousarray(data[iz][:, [ci_Z], :]) if iz.size else None
    data_SH = np.ascontiguousarray(data[ih][:, [ci_N, ci_E], :]) if ih.size else None
    del data
    nP = int((LP + sh["P_moveouts"].max()))
    nQ = int((LS + sh["S_moveouts"].max()))
    n_corr = N - max(nP, nQ) + 1
    batch, _ = scan._batches(len(ids), nS, 2, N, nQ)
    sp_mv = np.zeros((nT, nS, 2), np.int32)
    sp_mv[:, :, 0] = sh["S_moveouts"][:, :, 0] - sh["P_moveouts"][:, :, 0]
    tdict = {"cuspid": sh["cuspid"], "ref_s": sh["ref_s"], "moveouts": sp_mv}

    cov = None
    if C.MAD_SEGMENTED:
        import segmad
        cov = segmad.day_coverage(stations, prefixes, year, doy)

    dets = []
    import time as _time
    # optional day-level GPU lock: one process holds the GPU for its whole batch loop while the others load/detect
    # (per-batch alternation kept both processes in their GPU phases together and left the GPU idle otherwise)
    lockf = os.environ.get("FMF_GPU_LOCK", ""); lk = open(lockf, "w") if lockf else None
    if lk is not None:
        import fcntl; fcntl.flock(lk, fcntl.LOCK_EX)
    for b0 in range(0, ids.size, batch):
        bid = ids[b0:b0 + batch]; _t0 = _time.time()
        try:
            ccP = fmf.matched_filter(sh["P_waveforms"][bid][:, ip, :, :], sh["P_moveouts"][bid][:, ip, :], Pw[bid][:, ip, :],
                                     data_Z, C.STEP, arch="gpu", normalize="short")
            _nan_to_num_(ccP)
            Lc = min(ccP.shape[1], n_corr)
            if data_SZ is not None:
                ccS = fmf.matched_filter(sh["S_waveforms"][bid][:, iz, :1, :], sh["S_moveouts"][bid][:, iz, :1], Sw[bid][:, iz, :1],
                                         data_SZ, C.STEP, arch="gpu", normalize="short")
                _nan_to_num_(ccS); Lc = min(Lc, ccS.shape[1]); _iadd_(ccP[:, :Lc], ccS[:, :Lc]); del ccS
            if data_SH is not None:
                ccS = fmf.matched_filter(sh["S_waveforms"][bid][:, ih, :, :], sh["S_moveouts"][bid][:, ih, :], Sw[bid][:, ih, :],
                                         data_SH, C.STEP, arch="gpu", normalize="short")
                _nan_to_num_(ccS); Lc = min(Lc, ccS.shape[1]); _iadd_(ccP[:, :Lc], ccS[:, :Lc]); del ccS
        finally:
            pass
        cc = ccP[:, :Lc]; _t1 = _time.time()
        if C.MAD_SEGMENTED:
            import segmad
            d = segmad.detect_day_segmented(cc, sh, day0, bid, cov,
                                            merge_s=C.MAD_SEG_MERGE_S, min_len_s=C.MAD_SEG_MIN_S, s_comp=s_comp)
        else:
            d = detect_day(cc, tdict, day0, bid)
        del cc, ccP
        if os.environ.get("FMF_DEBUG"):
            log(f"    batch {b0 // batch + 1}: {len(bid)} tpl | gpu+sum {_t1 - _t0:.0f} s | detect {_time.time() - _t1:.0f} s | {len(d)} det", flush=True)
        if len(d):
            d["n_present"] = n_present_tpl[d.template_idx.values]
            dets.append(d)
    if lk is not None:
        import fcntl; fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
    cols = ["template_cuspid", "template_idx", "origin_utc", "cc", "corr_idx", "n_present"]
    out = pd.concat(dets, ignore_index=True) if dets else pd.DataFrame(columns=cols)
    log(f"  {year} DOY {doy:03d}: {int(present.sum())} sta, {ids.size} live tpl, batch={batch}, {len(out)} detections")
    return out


def main(year, doy_start=1, doy_end=366):
    import fast_matched_filter as fmf
    assert fmf.GPU_LOADED, "GPU not loaded"
    sh = _load()
    print(f"[shelly-scan] {len(sh['cuspid'])} templates; P {LP} / S {LS} samples; two-pass GPU; year {year}")
    for doy in range(doy_start, doy_end + 1):
        outf = f"{DETECT_DIR}/det_{year}_{doy:03d}.parquet"
        if os.path.exists(outf):
            continue
        try:
            out = scan_day(sh, year, doy)
            if out is None:
                continue
            out.to_parquet(outf, index=False)
        except Exception as e:
            print(f"  {year} DOY {doy:03d}: ERROR {type(e).__name__}: {e} -- skipping", flush=True)
            continue


import scan   # noqa: E402  (after config; reuses load_day/_batches)

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "build":
        build()
    else:
        nums = [int(x) for x in a]
        if not nums:
            raise SystemExit("usage: shelly.py build | shelly.py YEAR [D0 D1]")
        main(*nums)
