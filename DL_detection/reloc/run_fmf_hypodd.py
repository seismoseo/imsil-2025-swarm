#!/usr/bin/env python
"""Joint HypoDD relocation of the FMF detections with the 28 reference events (decision D22, 2026-10-08).

Same framework, settings and stages as reloc/run_dl_hypodd.py (korea-cluster-relocation from stage "hypoinverse": hyp1.40 on the
SAC-header picks -> ph2dt -> dt.ct -> rereference -> GPU xcorr (5-20 Hz, CC >= 0.7) -> dt.cc with the 7-set schedule, open
WDCC, adaptive damping -> 1000-replica bootstrap -> final_locations.csv). Two differences, both decided by the user:
  * events: the 28 reference events (real PhaseNet+ picks) + every new FMF detection (picks inherited from its template,
    reloc/export_fmf_detections.py writes the run root first);
  * catalog differential times only among the 28 references: ph2dt runs as usual (event.dat / event.sel for every event), then
    again on a copy of the phase file that holds only the references, and that dt.ct replaces the full one (kept as
    dt.ct.all_events). The detections enter the inversion through dt.cc alone, so their inherited picks cannot pin them to
    their templates. Driver-level wrap of hypodd.run_ph2dt; the framework files are untouched.
Output: <run_root>/final_locations.csv and catalogs/imsil_fmf_relocated.csv (+ kind, template, FMF CC, origin_time_reloc).
Run (as ./run_imsil.sh dl-reloc does):
    export PATH=/home/msseo/works/72.2025_Imsil_swarm/_external/bin:$PATH; set -a; . ../_external/imsil.env; set +a
    /home/msseo/miniforge3/envs/pq-gpu/bin/python reloc/run_fmf_hypodd.py [--run-root runs/hypodd_fmf] [--n-boot 1000]
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from obspy import UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import config as pipe_config                                   # noqa: E402  (DL pipeline config)
import event_sac_export as ese                                 # noqa: E402
import imsil_dl_cluster_config as hcc                          # noqa: E402  (also puts KCR on sys.path)
from add_reloc_origin_times import reloc_times                 # noqa: E402

OUT_CSV = os.path.join(ROOT, "catalogs", "imsil_fmf_relocated.csv")
KMA_CSV = os.path.join(ROOT, "catalogs", "imsil_kma24.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=os.path.join(ROOT, "runs", "hypodd_fmf"))
    ap.add_argument("--model", default="phasenet_plus")
    ap.add_argument("--velmodel", default="kim1983")
    ap.add_argument("--stage-from", default="hypoinverse", choices=["hypoinverse", "ph2dt", "dtct", "rereference", "xcorr", "dtcc"])
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--blocked-pairs", type=int, default=45)
    ap.add_argument("--xcorr-chunk", type=int, default=4000)
    ap.add_argument("--xcorr-resume", action="store_true")
    ap.add_argument("--bootstrap-only", action="store_true")
    ap.add_argument("--out-csv", default=OUT_CSV, help="final table (default catalogs/imsil_fmf_relocated.csv)")
    a = ap.parse_args()
    inp = pd.read_csv(os.path.join(a.run_root, "fmf_relocation_input.csv"), dtype={"event_id": str, "template_event_id": str})
    inp["template_event_id"] = inp.template_event_id.str.replace(r"\.0$", "", regex=True)   # written as float because reference rows are empty
    inp["template_cuspid"] = inp.template_cuspid.astype("Int64")
    ref_ids = set(inp.event_id[inp.kind == "reference"])

    # 1. catalog: the 28 references from the DL .sum (same QC/box as the DL relocation) + the detections at their template position
    sum_path = os.path.join(pipe_config.velmodel_dir(a.model, a.velmodel), f"{pipe_config.REGION_NAME}2025.sum")
    dl = ese.catalog_from_sum(sum_path, qc=pipe_config.HAENAM_QC, region_bounds=pipe_config.HAENAM_BOX)
    refs = dl[dl.event_id.isin(ref_ids)].copy()
    if len(refs) != len(ref_ids):
        raise SystemExit(f"{len(ref_ids) - len(refs)} reference events not found in {sum_path}")
    det = inp[inp.kind == "detection"].copy()
    det["time"] = pd.to_datetime(det.origin); det["origin_utc"] = det.time.apply(lambda t: UTCDateTime(t.to_pydatetime()))
    for c in ("num", "gap", "rms", "erh", "erz"): det[c] = np.nan
    cat = pd.concat([refs[["event_id", "origin_utc", "time", "lat", "lon", "depth", "num", "gap", "rms", "erh", "erz"]],
                     det[["event_id", "origin_utc", "time", "lat", "lon", "depth", "num", "gap", "rms", "erh", "erz"]]]).sort_values("time").reset_index(drop=True)
    kma = pd.read_csv(KMA_CSV, dtype={"event_id": str}, parse_dates=["origin_time_UTC"])
    mags = {}
    for r in refs.itertuples():
        dt = (kma.origin_time_UTC - r.time).abs(); i = dt.idxmin()
        if dt[i].total_seconds() <= 6.0: mags[r.event_id] = float(kma.loc[i, "ML"])
    print(f"[catalog] {len(refs)} references + {len(det)} detections = {len(cat)} events -> {len(cat) * (len(cat) - 1) // 2:,} xcorr pairs")
    cfg = hcc.make_config(output_root=a.run_root, num_cores=a.cores, onset_dl_ids=[], mainshock_dl_id=None, open_wdcc=True)
    ese.write_framework_catalog(cat, cfg.event_catalog_csv, magnitudes=mags)

    # 2. manifest (all events; SAC bundles were written by reloc/export_fmf_detections.py)
    from pipeline import config as kcr_config
    from pipeline.core import evmap
    wf_dir = kcr_config.waveforms_dir(cfg)
    missing = [e for e in cat.event_id if not os.path.isdir(os.path.join(wf_dir, e))]
    if missing:
        raise SystemExit(f"{len(missing)} events have no SAC bundle under {wf_dir} - run reloc/export_fmf_detections.py --apply first")
    mp = evmap.pin_manifest(cfg, new_event_ids=cat.event_id.astype(str))
    man = pd.read_csv(mp, dtype={"event_id": str})
    ref_cuspids = set(200000 + man.event_idx[man.event_id.isin(ref_ids)])
    print(f"[manifest] {mp}: {len(man)} events; {len(ref_cuspids)} reference cuspids")

    # 3. framework stages, with catalog differential times among the references only
    from pipeline.core import pipeline as kcr_pipeline, hypodd, xcorr as kcr_xcorr
    from pipeline import viz
    _run_ph2dt = hypodd.run_ph2dt

    def run_ph2dt_reference_dtct(cfg_):
        d = _run_ph2dt(cfg_)                                   # event.dat / event.sel / dt.ct for every event
        sub = os.path.join(d, "reference_only"); os.makedirs(sub, exist_ok=True)
        pha = f"{cfg_.region}_mod.pha"; keep = False; n_ev = 0
        with open(os.path.join(d, pha)) as fi, open(os.path.join(sub, pha), "w") as fo:
            for line in fi:
                if line.startswith("#"):
                    keep = int(line.split()[-1]) in ref_cuspids; n_ev += keep
                if keep: fo.write(line)
        for f in ("station.dat", "ph2dt.inp"): shutil.copy(os.path.join(d, f), sub)
        subprocess.run(["ph2dt", "ph2dt.inp"], cwd=sub, check=True, stdout=subprocess.DEVNULL)
        shutil.copy(os.path.join(d, "dt.ct"), os.path.join(d, "dt.ct.all_events"))
        shutil.copy(os.path.join(sub, "dt.ct"), os.path.join(d, "dt.ct"))
        n_pairs = sum(1 for ln in open(os.path.join(d, "dt.ct")) if ln.startswith("#"))
        print(f"[ph2dt] dt.ct restricted to the {n_ev} reference events ({n_pairs} event pairs); full file kept as dt.ct.all_events")
        return d
    hypodd.run_ph2dt = run_ph2dt_reference_dtct
    if a.xcorr_chunk:
        os.environ["XCORR_PAIR_CHUNK"] = str(a.xcorr_chunk)
    if a.blocked_pairs:
        # Cache-friendly pair order, copied from reloc/run_dl_hypodd.py (results are per pair; the order has no effect on dt.cc)
        B = a.blocked_pairs

        def blocked_combinations(events, r=2):
            ev = list(events)
            blocks = [ev[k:k + B] for k in range(0, len(ev), B)]
            for ia, ba in enumerate(blocks):
                for ib in range(ia, len(blocks)):
                    bb = blocks[ib]
                    for i, e1 in enumerate(ba):
                        for j, e2 in enumerate(bb):
                            if ib > ia or j > i:
                                yield (e1, e2)
        kcr_xcorr.combinations = blocked_combinations
        print(f"[xcorr-order] blocked pair order, blocks of {B} events; chunk={os.environ.get('XCORR_PAIR_CHUNK', 'default')}")
    if not a.bootstrap_only:
        res = kcr_pipeline.run_cluster(cfg, stage_from=a.stage_from, through="dtcc", velmodels=("kim1983", "kim2011"), arc_velmodel="kim1983",
                                       cores=a.cores, xcorr_resume=a.xcorr_resume)
        print("[run_fmf_hypodd] framework:", {k: v for k, v in res.items() if k != "waveforms"})

    # 4. bootstrap + final table
    if a.n_boot:
        hypodd.bootstrap_relocation(cfg, "dtcc", n=a.n_boot, cores=None)
    tab = viz.location_table(cfg, branch="dtcc", save=True)
    tab["event_id"] = tab.event_id.astype(str)
    meta = inp.set_index("event_id")[["kind", "template_cuspid", "template_event_id", "fmf_cc", "n_present", "n_P", "n_S"]]
    tab = tab.merge(meta, left_on="event_id", right_index=True, how="left")
    tab["KMA_ML"] = tab.event_id.map(mags)
    times = reloc_times(os.path.join(kcr_config.dtcc_dir(cfg), "hypoDD.reloc"), mp)
    tab = tab.merge(times, on="event_id", how="left")
    tab.to_csv(a.out_csv, index=False)
    for k in ("reference", "detection"):
        g = tab[tab.kind == k]
        print(f"[final] {k}: {len(g)} relocated; under-constrained {int(g.under_constrained.sum())}; cc links median {g.cc_links.median():.0f}; ct links median {g.ct_links.median():.0f}")
    print(f"[final] {len(tab)} events -> {a.out_csv}")


if __name__ == "__main__":
    main()
