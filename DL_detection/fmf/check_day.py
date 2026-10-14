"""Self-detection and timing check for one scanned day (Buan acceptance test).

Every template whose origin falls on the day must appear in the daily parquet within SELF_S of its origin with a
network CC near 1 (same archive, same processing). Also reports the raw 12xMAD detection count, the CC distribution
and the >= MIN_PRESENT_STATIONS / CC >= ABS_CC_FLOOR survivors. Usage: python check_day.py YEAR DOY
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from obspy import UTCDateTime

import config as C

SELF_S = 3.0


def main(year, doy):
    import shelly
    sh = shelly._load()
    outf = f"{C.DETECT_DIR}/det_{year}_{doy:03d}.parquet"
    if not os.path.exists(outf):
        t0 = time.time(); out = shelly.scan_day(sh, year, doy); dt = time.time() - t0
        if out is None:
            print("no data for that day"); return
        out.to_parquet(outf, index=False); print(f"scan_day took {dt:.0f} s -> {outf}")
    d = pd.read_parquet(outf); d["origin"] = pd.to_datetime(d.origin_utc.apply(lambda u: getattr(u, "datetime", u)))
    day0 = C.day_start(year, doy)
    torig = pd.to_datetime([str(o) for o in sh["origin_utc"]]); onday = np.where((torig >= day0.datetime) & (torig < (day0 + 86400).datetime))[0]
    print(f"{year} DOY {doy:03d}: {len(d)} raw detections (12xMAD, segmented={C.MAD_SEGMENTED}); {len(onday)} templates originate on this day")
    gate = d[(d.cc >= C.ABS_CC_FLOOR) & (d.n_present >= C.MIN_PRESENT_STATIONS)]
    print(f"  after floor cc>={C.ABS_CC_FLOOR} and >={C.MIN_PRESENT_STATIONS} stations: {len(gate)} | cc quantiles (raw) "
          + ", ".join(f"{q:.0%}: {d.cc.quantile(q):.2f}" for q in (.5, .9, .99, 1.0)) if len(d) else "  no detections")
    rows = []
    for ti in onday:
        me = d[d.template_idx == ti]; dt_s = (me.origin - torig[ti]).dt.total_seconds().abs()
        hit = me[dt_s <= SELF_S]
        rows.append(dict(event_id=str(sh["event_id"][ti]), origin=str(torig[ti])[:19], self_cc=float(hit.cc.max()) if len(hit) else np.nan,
                         n_present=int(hit.n_present.max()) if len(hit) else 0, live_P=int((sh["P_weights"][ti, :, 0] > 0).sum()), live_S=int((sh["S_weights"][ti] > 0).sum())))
    r = pd.DataFrame(rows)
    if len(r):
        ok = r.self_cc.notna(); print(f"  self-detection: {int(ok.sum())}/{len(r)} templates (median self CC {r.self_cc.median():.3f}; min {r.self_cc.min():.3f})")
        print(r.to_string(index=False))
    # detections not explained by a template on this day (candidates), declustered by 4 s
    other = gate.sort_values("origin"); grp = (other.origin.diff().dt.total_seconds().fillna(99) > 4).cumsum(); rep = other.loc[other.groupby(grp).cc.idxmax()]
    print(f"  declustered gated detections: {len(rep)}; of which within {SELF_S}s of a template origin on the day: "
          f"{int(sum(((rep.origin - torig[ti]).dt.total_seconds().abs() <= SELF_S).any() for ti in onday)) if len(onday) else 0}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
