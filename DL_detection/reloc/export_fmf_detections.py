#!/usr/bin/env python
"""Run root for the joint relocation of the FMF detections with the 28 reference events (decision D22, 2026-10-08).

Real-pick events: the 28 events of catalogs/reference_25m/DL_reference_25m.csv. Their SAC bundles (PhaseNet+ picks in `a`/`t0`)
are COPIED from the DL relocation run root (runs/hypodd_dl/waveforms_100km): the framework re-references SAC headers in place,
so the original run is never touched.
FMF detections: every new detection of catalogs/fmf_final/imsil_fmf_catalog.csv. Each inherits its representative template's
picks shifted to the detection origin:
    pick(detection, station, phase) = pick(template event, station, phase) + (detection origin - template origin)
with the template's absolute picks read from its SAC headers (reference time + a / t0, as fmf/templates.py reads them) and
the template origin from catalogs/imsil_templates_ref25.csv (the origin the FMF origin estimate refers to). The SAC bundle is
cut from the continuous archive exactly as reloc/event_sac_export.py cuts DL events (origin - 30 s .. + 90 s, same station
table, same channel policy), with the template's relocated position as the initial location.
Writes <run_root>/waveforms_100km/<event_id>/ and <run_root>/fmf_relocation_input.csv (one row per event: kind, origin,
initial location, template, FMF CC, number of inherited P/S picks).
DRY RUN BY DEFAULT (prints the plan); --apply writes.
    python reloc/export_fmf_detections.py [--run-root runs/hypodd_fmf] [--apply]
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import config as pipe_config                                   # noqa: E402  (DL pipeline config)
import event_sac_export as ese                                 # noqa: E402

FMF_CSV = os.path.join(ROOT, "catalogs", "fmf_final", "imsil_fmf_catalog.csv")
TPL_CSV = os.path.join(ROOT, "catalogs", "imsil_templates_ref25.csv")
REF_CSV = os.path.join(ROOT, "catalogs", "reference_25m", "DL_reference_25m.csv")
SRC_RUN = os.path.join(ROOT, "runs", "hypodd_dl")


def template_picks(event_id):
    """{(net, sta): {'P': UTCDateTime, 'S': UTCDateTime}} from the template event's SAC headers (reference time + a / t0)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(SRC_RUN, "waveforms_100km", event_id, "*.sac"))):
        h = obspy.read(f, headonly=True)[0].stats; s = h.sac
        ref = UTCDateTime(year=s.nzyear, julday=s.nzjday, hour=s.nzhour, minute=s.nzmin, second=s.nzsec, microsecond=int(s.nzmsec) * 1000)
        d = out.setdefault((h.network, h.station), {})
        if "a" in s and s.a > -1e4: d.setdefault("P", ref + float(s.a))
        if "t0" in s and s.t0 > -1e4: d.setdefault("S", ref + float(s.t0))
    return {k: v for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=os.path.join(ROOT, "runs", "hypodd_fmf"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--extra-csv", default=None, help="further detections to add (columns origin, template_cuspid, cc, n_present), e.g. the 2019-04-15 events of the long scan")
    a = ap.parse_args()
    fmf = pd.read_csv(FMF_CSV, parse_dates=["origin"]); new = fmf[fmf.is_new].sort_values("origin").reset_index(drop=True)
    if a.extra_csv:                                   # 2026-10-14: long-scan detections relocated with the swarm (consistent with D22)
        ex = pd.read_csv(a.extra_csv, parse_dates=["origin"]); print(f"[plan] + {len(ex)} extra detections from {a.extra_csv}")
        new = pd.concat([new, ex[["origin", "template_cuspid", "cc", "n_present"]]], ignore_index=True).sort_values("origin").reset_index(drop=True)
    tpl = pd.read_csv(TPL_CSV, dtype={"event_id": str}, parse_dates=["origin_time_UTC"]).set_index("cuspid")
    ref = pd.read_csv(REF_CSV, dtype={"eid": str, "event_id": str})
    ref_ids = sorted(ref.eid.astype(str))
    tp_cache = {}
    rows = []
    for r in new.itertuples():
        o = UTCDateTime(pd.Timestamp(r.origin).to_pydatetime()); eid = ese.event_id_from_time(o)
        t = tpl.loc[int(r.template_cuspid)]; te = str(t.event_id)
        if te not in tp_cache: tp_cache[te] = template_picks(te)
        delta = o - UTCDateTime(pd.Timestamp(t.origin_time_UTC).to_pydatetime())
        picks = {k: {ph: (tt + delta) - o for ph, tt in v.items()} for k, v in tp_cache[te].items()}
        rows.append(dict(kind="detection", event_id=eid, origin=str(o)[:23], lat=float(t.latitude), lon=float(t.longitude), depth=float(t.depth_km),
                         template_cuspid=int(r.template_cuspid), template_event_id=te, fmf_cc=float(r.cc), n_present=int(r.n_present),
                         n_P=sum("P" in v for v in picks.values()), n_S=sum("S" in v for v in picks.values()), picks_rel=picks, o_utc=o))
    det = pd.DataFrame(rows)
    clash = set(det.event_id) & set(ref_ids); dup = det.event_id[det.event_id.duplicated()].tolist()
    if clash or dup:
        raise SystemExit(f"event-id collisions: with references {sorted(clash)}; among detections {dup}")
    print(f"[plan] run root {a.run_root}")
    print(f"[plan] {len(ref_ids)} reference events (SAC copied from {SRC_RUN}/waveforms_100km)")
    print(f"[plan] {len(det)} FMF detections from {len(det.template_event_id.unique())} templates; inherited picks per detection: "
          f"P median {int(det.n_P.median())} (min {det.n_P.min()}), S median {int(det.n_S.median())} (min {det.n_S.min()})")
    if not a.apply:
        print("(dry run — add --apply to copy the reference bundles, export the detections and write fmf_relocation_input.csv)"); return
    wf = os.path.join(a.run_root, "waveforms_100km"); os.makedirs(wf, exist_ok=True)
    for eid in ref_ids:
        src, dst = os.path.join(SRC_RUN, "waveforms_100km", eid), os.path.join(wf, eid)
        if not os.path.isdir(dst): shutil.copytree(src, dst)
    stations = ese.load_stations_for_year(pipe_config.station_table_dir("phasenet_plus"), 2025)
    n_done = 0
    for r in det.itertuples():
        out_dir = os.path.join(wf, r.event_id)
        if os.path.isdir(out_dir) and any(f.endswith(".sac") for f in os.listdir(out_dir)): continue
        ev = ese.EventMeta(r.event_id, r.o_utc, r.lat, r.lon, r.depth)
        for sr in stations.itertuples():
            sm = ese.StationMeta(sr.Network, sr.Code, float(sr.Latitude), float(sr.Longitude), float(getattr(sr, "Elevation", np.nan)))
            st = ese.read_continuous_for_event(pipe_config.CONTINUOUS, sm.network, sm.code, r.o_utc)
            if not len(st): continue
            pk = r.picks_rel.get((sm.network, sm.code), {})
            ese.write_event_sac(st, out_dir, event_meta=ev, station_meta=sm, p_rel_s=pk.get("P"), s_rel_s=pk.get("S"))
        n_done += 1
        if n_done % 25 == 0: print(f"  exported {n_done} detections", flush=True)
    refrows = pd.DataFrame(dict(kind="reference", event_id=ref_ids))
    table = pd.concat([refrows, det.drop(columns=["picks_rel", "o_utc"])], ignore_index=True)
    table.to_csv(os.path.join(a.run_root, "fmf_relocation_input.csv"), index=False)
    print(f"[export] {len(ref_ids)} reference bundles copied, {n_done} detection bundles written -> {wf}; table -> {a.run_root}/fmf_relocation_input.csv")


if __name__ == "__main__":
    main()
