"""Per-day GPU matched-filter scan (Beauce single-length templates) — Haenam, multi-year.

Memory-safe by construction (Buan design): one UTC day at a time; per-day weights (missing
stations zeroed + renormalised); a byte guard batches templates under GPU_BUDGET_BYTES; the CC is
discarded per day. Resumable: one parquet per (year, day); existing days are skipped.
The production detector is the Shelly two-pass in shelly.py (which reuses load_day/_batches here).

Run:  conda run -n fmf python fmf/scan.py YEAR [D0 D1]
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from obspy import read

import config as C
import fast_matched_filter as fmf
from detect import detect_day

CHAN_PREF = ["HH", "HG", "EL", "BH", "DP"]


def _normalize_rates(st, tol=1e-3):
    """Round near-integer sampling rates (e.g. 99.99999 Hz -> 100 Hz). Some NECIS KS day files
    (JDO2 on ~1/3 of days, MJDB, SIJA, ...) mix 99.99999237 Hz and 100.0 Hz segments; obspy then
    refuses to merge and the whole station-day would be lost. Sub-ppm drift is far below sample level."""
    for tr in st:
        r = tr.stats.sampling_rate
        if abs(r - round(r)) < tol and abs(r - round(r)) > 0:
            tr.stats.sampling_rate = float(round(r))
    return st


def _chan_file(net, sta, comp, year, doy, prefix=None):
    pres = ([prefix] + [q for q in CHAN_PREF if q != prefix]) if prefix else CHAN_PREF   # preferred band, else per-day fallback
    for pre in pres:
        f = glob.glob(f"{C.CONTINUOUS}/{net}/{sta}/{pre}{comp}.D/*.D.{year}.{doy:03d}")
        if f:
            return f[0]
    return None


def _load_channel(task):
    """Worker: read+resample(100)+bandpass+trim one (station,component) day file.
    Returns (si, ci, float32[N]) or (si, ci, None)."""
    si, ci, net, sta, comp, prefix, year, doy = task
    f = _chan_file(net, sta, comp, year, doy, prefix)
    if not f:
        return si, ci, None
    try:
        st = _normalize_rates(read(f))
        st.merge(method=1, fill_value=0)
        tr = st[0]
    except Exception:
        return si, ci, None
    N = C.SAMPLES_PER_DAY
    day0 = C.day_start(year, doy)
    tr.detrend("linear")
    if abs(tr.stats.sampling_rate - C.FS) > 1e-6:
        tr.resample(C.FS)
    tr.filter("bandpass", freqmin=C.BANDPASS[0], freqmax=C.BANDPASS[1], corners=4, zerophase=True)
    tr.trim(day0, day0 + (N - 1) / C.FS, pad=True, fill_value=0.0)
    w = tr.data.astype(np.float32)[:N]
    if w.size < N:
        w = np.pad(w, (0, N - w.size))
    w -= w.mean()
    return si, ci, w


def _load_templates():
    z = np.load(C.TEMPLATE_NPZ, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_day(stations, prefixes, year, doy):
    """data (n_sta, n_comp, N) float32 + present (n_sta,) bool + day_start."""
    day0 = C.day_start(year, doy)
    N = C.SAMPLES_PER_DAY
    nS, nC = len(stations), len(C.COMPONENTS)
    data = np.zeros((nS, nC, N), np.float32)
    present = np.zeros(nS, bool)
    tasks = []
    for si, ns in enumerate(stations):
        net, sta = ns.split(".")
        for ci, comp in enumerate(C.COMPONENTS):
            tasks.append((si, ci, net, sta, comp, prefixes[si], year, doy))
    with ProcessPoolExecutor(max_workers=C.N_WORKERS) as ex:
        for si, ci, w in ex.map(_load_channel, tasks, chunksize=4):
            if w is not None:
                data[si, ci] = w
                present[si] = True
    return data, present, day0


def _batches(nT, nS, nC, N, max_tpl_samples):
    """Template batch size so estimated device usage stays under GPU_BUDGET_BYTES."""
    n_corr = int((N - max_tpl_samples) / C.STEP + 1)
    data_b = nS * nC * N * 4
    tpl_b = nS * nC * C.WIN_SAMPLES * 4
    budget = C.GPU_BUDGET_BYTES / C.DEVICE_OVERHEAD - data_b - tpl_b
    batch = max(1, int(budget // (n_corr * 4)))
    return min(batch, nT), n_corr


def scan_day(tpl, year, doy, log=print):
    stations = [s for s in tpl["stations"]]
    prefixes = [str(p) for p in tpl["channels"]] if "channels" in tpl else [None] * len(stations)
    data, present, day0 = load_day(stations, prefixes, year, doy)
    if not present.any():
        return None
    nS, nC, N = len(stations), len(C.COMPONENTS), C.SAMPLES_PER_DAY
    W = tpl["weights"].copy()
    W[:, ~present, :] = 0.0
    live = W.reshape(W.shape[0], -1).sum(axis=1)
    live_ids = np.where(live > 0)[0]
    if live_ids.size == 0:
        return None
    W[live_ids] /= live[live_ids][:, None, None]
    n_present_tpl = ((tpl["weights"] > 0).any(axis=2) & present[None, :]).sum(axis=1)
    max_tpl_samples = int((tpl["moveouts"] + C.WIN_SAMPLES).max())
    batch, n_corr = _batches(len(live_ids), nS, nC, N, max_tpl_samples)
    dets = []
    for b0 in range(0, live_ids.size, batch):
        ids = live_ids[b0:b0 + batch]
        cc = fmf.matched_filter(tpl["waveforms"][ids], tpl["moveouts"][ids], W[ids], data,
                                C.STEP, arch="gpu", normalize="short")
        cc = np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)
        d = detect_day(cc, tpl, day0, ids)
        if len(d):
            d["n_present"] = n_present_tpl[d.template_idx.values]
            dets.append(d)
        del cc
    out = pd.concat(dets, ignore_index=True) if dets else pd.DataFrame(
        columns=["template_cuspid", "template_idx", "origin_utc", "cc", "corr_idx", "n_present"])
    log(f"  {year} DOY {doy:03d}: {int(present.sum())} sta, {live_ids.size} live tpl, "
        f"batch={batch}, n_corr={n_corr}, {len(out)} detections")
    return out


def main(year, doy_start=1, doy_end=366):
    assert fmf.GPU_LOADED, "GPU library not loaded — refusing to run (no silent CPU fallback)"
    tpl = _load_templates()
    outdir = f"{C.FMF_ROOT}/catalogs/daily_beauce{C._SUF}"
    os.makedirs(outdir, exist_ok=True)
    print(f"[scan] {len(tpl['cuspid'])} templates x {len(tpl['stations'])} stations; year {year}")
    for doy in range(doy_start, doy_end + 1):
        outf = f"{outdir}/det_{year}_{doy:03d}.parquet"
        if os.path.exists(outf):
            continue
        out = scan_day(tpl, year, doy)
        if out is None:
            continue
        out.to_parquet(outf, index=False)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    if not a:
        raise SystemExit("usage: scan.py YEAR [D0 D1]")
    main(*a)
