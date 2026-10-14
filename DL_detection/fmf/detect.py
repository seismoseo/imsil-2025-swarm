"""Detection extraction from FMF network correlation traces — mirrors the FMF tutorial
`extract_new_detections`: per template, subtract the median, threshold at MAD_K*MAD, then
decluster with an adaptive window sized from the S-P moveout (min 1x template duration, max 3x),
keeping the highest CC. NOTE: the absolute CC floor and the >=N-present-station gate are applied
downstream in build_catalog.py (Buan policy), not here — the daily parquets are the raw 12xMAD set.
"""
import numpy as np
import pandas as pd

import config as C


def detect_day(cc_sum, tpl, day_start, batch_ids):
    """cc_sum: (n_batch, n_corr) for templates `batch_ids`. Returns detections df.
    detected_origin = day_start + idx*step/FS - ref_s[template]."""
    rows = []
    dur = int(C.TEMPLATE_DURATION_S * C.FS / C.STEP)
    cuspid, ref_s, mv = tpl["cuspid"], tpl["ref_s"], tpl["moveouts"]
    for b, tid in enumerate(batch_ids):
        cc = cc_sum[b]
        k = max(1, int(getattr(C, "MAD_SUBSAMPLE", 1)))
        cc -= np.median(cc[::k])                      # in place: cc_sum is discarded after this call
        mad = np.median(np.abs(cc[::k]))
        if mad <= 0:
            continue
        thr = C.MAD_K * mad
        cand = np.flatnonzero(cc > thr)
        if cand.size == 0:
            continue
        d_mv = mv[tid, :, 0] - mv[tid, :, -1]
        d_mv = d_mv[d_mv != 0]
        med_sp = int(np.median(d_mv) / C.STEP) if d_mv.size else dur
        search = min(3 * dur, max(med_sp, dur))
        order = cand[np.argsort(cc[cand])[::-1]]
        taken = []
        occupied = np.zeros(cc.size, bool)
        for idx in order:
            lo, hi = max(0, idx - search // 2), min(cc.size, idx + search // 2)
            if not occupied[lo:hi].any():
                taken.append(idx)
                occupied[lo:hi] = True
        for idx in sorted(taken):
            origin = (day_start + (idx * C.STEP) / C.FS - float(ref_s[tid])).datetime
            rows.append((int(cuspid[tid]), tid, origin, float(cc[idx]), int(idx)))
    return pd.DataFrame(rows, columns=["template_cuspid", "template_idx",
                                       "origin_utc", "cc", "corr_idx"])
