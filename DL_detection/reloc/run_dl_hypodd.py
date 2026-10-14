"""Standalone HypoDD relocation of the Imsil DL catalog (Buan recipe on korea-cluster-relocation).

IMSIL PORT of the Haenam driver: logic unchanged. Changed: the cluster config module (imsil_dl_cluster_config: Imsil
box, PAGO KCR copy, empty onset list = D6), the registry (Imsil: the "template" table is the 24 KMA-listed events,
so template_match / ML_template carry the KMA id and KMA ML), output names imsil_dl_relocated*.csv, default --year 2025,
and messages. Run through ./run_imsil.sh dl-reloc ... (sets PATH for hyp1.40/ph2dt/hypoDD and the pq-gpu env).

Haenam docstring (provenance):

Chain: HYPOINVERSE .sum -> QC + Haenam box -> KMA-format event catalog -> per-event SAC from the continuous
archive with the PyOcto picks (KS 100 Hz, JU 200 Hz native) -> framework from stage "hypoinverse":
hyp1.40 re-run on the subset -> ph2dt -> dt.ct -> rereference -> GPU xcorr -> dt.cc (adaptive damping)
-> 1000-replica bootstrap -> final_locations.csv (PocketQuake schema) -> catalogs/haenam_dl_relocated.csv
(+ template_match / tier columns). Settings mirror the canonical Haenam_joint run (see
haenam_dl_cluster_config.py).

Run (GPU env):  set -a; . /home/msseo/works/15.PocketQuake/.env; set +a
                /home/msseo/miniforge3/envs/pq-gpu/bin/python reloc/run_dl_hypodd.py [--max-events N]
                    [--through dtcc] [--cores 24] [--n-boot 1000] [--run-root DIR] [--export-only]
Idempotent: SAC export skips existing event dirs; the framework stages re-run from the chosen stage.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import config as pipe_config                                   # noqa: E402  (DL pipeline config)
import event_sac_export as ese                                 # noqa: E402
import imsil_dl_cluster_config as hcc                          # noqa: E402  (also puts KCR on sys.path)  IMSIL PORT
sys.path.insert(0, "/home/msseo/works/72.2025_Imsil_swarm/registry"); import imsil as haenam   # registry paths (IMSIL PORT)

TEMPLATES_CSV = haenam.path(None, "tm_templates_195")
OUT_CSV = os.path.join(ROOT, "catalogs", "imsil_dl_relocated.csv")                           # IMSIL PORT
MATCH_TOL_S = 6.0


def template_matches(cat):
    """{dl_event_id: (template_event_id, ML)} for DL events within MATCH_TOL_S of a relocated event."""
    tp = pd.read_csv(TEMPLATES_CSV, dtype={"event_id": str})
    tp["t"] = pd.to_datetime(tp.origin_time_UTC)
    out = {}
    for r in cat.itertuples():
        d = (tp.t - r.time).abs()
        i = d.idxmin()
        if d[i].total_seconds() <= MATCH_TOL_S:
            out[r.event_id] = (tp.loc[i, "event_id"], float(tp.loc[i, "ML"]))
    return out


def main():
    ap = argparse.ArgumentParser(description="Imsil DL catalog -> HypoDD relocation (korea-cluster-relocation)")
    ap.add_argument("--model", default="phasenet_plus")
    ap.add_argument("--year", type=int, default=2025)                                         # IMSIL PORT
    ap.add_argument("--years", default=None, help="multi-year catalog, e.g. 2020-2026 or 2020,2026 (overrides --year; joint relocation)")
    ap.add_argument("--velmodel", default="kim1983", help="which year .sum sources the catalog")
    ap.add_argument("--run-root", default=hcc.RUN_ROOT)
    ap.add_argument("--through", default="dtcc", choices=["hypoinverse", "ph2dt", "dtct", "rereference", "xcorr", "dtcc"])
    ap.add_argument("--stage-from", default="hypoinverse", choices=["hypoinverse", "ph2dt", "dtct", "rereference", "xcorr", "dtcc"])
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--max-events", type=int, default=None, help="cap the catalog (testing)")
    ap.add_argument("--after", default=None, help="keep events at/after this UTC time (testing), e.g. 2026-08-14")
    ap.add_argument("--min-depth", type=float, default=None, metavar="KM",
                    help="keep only events at/below this depth. The in-box QC catalog contains shallow quarry blasts "
                         "(98%% of the < 5 km in-box events are 08-18 KST, 11%% weekend); they cannot correlate with the "
                         "deep templates and only add pairs. 10 keeps every source-depth event of 2020/2026 and the "
                         "quiescence candidates.")
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicas (0 = skip)")
    ap.add_argument("--no-onset", action="store_true", help="disable the M>=2 onset-window override")
    ap.add_argument("--blocked-pairs", type=int, default=45, metavar="B",
                    help="order xcorr pairs block-wise (blocks of B events) so the in-process LRU trace cache "
                         "(XCORR_TRACE_CACHE=20000 traces ~ 109 events) is not thrashed by the lexicographic "
                         "all-pairs order; B=45 keeps a block pair (90 events x ~183 traces) resident. 0 = framework order. "
                         "DEFAULT 45: the framework order was measured at 1,400 pairs/h (disk-bound) on 481 events")
    ap.add_argument("--xcorr-chunk", type=int, default=4000,
                    help="XCORR_PAIR_CHUNK: pairs per GPU chunk; per-pair dt.cc files are written after each chunk, "
                         "so a small chunk (e.g. 4000) gives visible progress and pair-level resume. 0 = framework default")
    ap.add_argument("--xcorr-resume", action="store_true", help="skip pairs whose per-pair dt.cc files already exist")
    ap.add_argument("--open-wdcc", action="store_true", help="dt.cc schedule without the cc inter-event distance cutoff (see cluster config)")
    ap.add_argument("--bootstrap-only", action="store_true", help="skip the framework stages: bootstrap the existing dt.cc solution + write the catalogs")
    a = ap.parse_args()

    if a.years:
        years = sorted({y for tok in a.years.split(",") for y in (range(int(tok.split("-")[0]), int(tok.split("-")[1]) + 1) if "-" in tok else [int(tok)])})
    else:
        years = [a.year]
    # 1. catalog: QC gate (gap<180, N>=8, RMS<1, ERH/ERZ<5) + Haenam box + doublet-free, one .sum per year
    parts = []
    for y in years:
        sum_path = os.path.join(pipe_config.velmodel_dir(a.model, a.velmodel), f"{pipe_config.REGION_NAME}{y}.sum")
        if not os.path.exists(sum_path):
            if a.years:
                print(f"[catalog] {y}: missing {sum_path} — skipped (run ./run_imsil.sh locate {y} run)"); continue
            raise SystemExit(f"missing {sum_path} — run ./run_imsil.sh locate {y} run first")
        c = ese.catalog_from_sum(sum_path, qc=pipe_config.HAENAM_QC, region_bounds=pipe_config.HAENAM_BOX)
        print(f"[catalog] {y}: {len(c)} events after QC/box"); parts.append(c)
    if not parts:
        raise SystemExit("no .sum file found for the requested years")
    cat = pd.concat(parts).sort_values("time").reset_index(drop=True)
    if a.min_depth is not None:
        n0 = len(cat); cat = cat[cat.depth >= a.min_depth].reset_index(drop=True)
        print(f"[catalog] depth gate >= {a.min_depth:g} km: {n0} -> {len(cat)} events ({n0 - len(cat)} shallow dropped)")
    if a.after:
        cat = cat[cat.time >= pd.Timestamp(a.after)].reset_index(drop=True)
    if a.max_events:
        cat = cat.head(a.max_events).copy()
    if len(cat) < 2:
        raise SystemExit(f"only {len(cat)} event(s) after QC — HypoDD needs >= 2")
    tm = template_matches(cat)
    n_pairs = len(cat) * (len(cat) - 1) // 2
    print(f"[scale] {len(cat)} events -> {n_pairs:,} xcorr pairs (all pairs, GPU batched); "
          f"{len(tm)} match a relocated (template) event")

    # Onset-window treatment: the canonical M >= 2 list mapped onto DL ids.
    # The canonical ids are a MIX of id spaces: most are template (relocated-catalog) ids that
    # must be translated through `inv`, but an event detected only by DL (no template match,
    # e.g. the 2026-09-09 ML 2.3) carries its DL id already.  Resolving through `inv` alone
    # silently DROPPED such events from the onset list -- the list length changed but the log
    # still printed the old count, with no warning (fixed 2026-09-29).
    inv = {t: d for d, (t, _) in tm.items()}
    _dl_ids = set(cat.event_id.astype(str))
    onset_ids, _unresolved = [], []
    if not a.no_onset:
        for t in (hcc.CANONICAL_ONSET_EVENTS_2020 + hcc.CANONICAL_ONSET_EVENTS_2026):
            if t in inv:                       # template id -> DL id
                onset_ids.append(inv[t])
            elif t in _dl_ids:                 # already a DL id (DL-only detection)
                onset_ids.append(t)
            else:
                _unresolved.append(t)
    _n_want = len(hcc.CANONICAL_ONSET_EVENTS_2020) + len(hcc.CANONICAL_ONSET_EVENTS_2026)
    ms_id = inv.get(hcc.CANONICAL_MAINSHOCK) or (hcc.CANONICAL_MAINSHOCK
                                                 if hcc.CANONICAL_MAINSHOCK in _dl_ids else None)
    if onset_ids:
        print(f"[onset] +/-0.05 s / 1-40 Hz window for {len(onset_ids)} of {_n_want} listed "
              f"events (imsil_dl_cluster_config.ONSET_EVENTS); mainshock = {ms_id}")
    elif not a.no_onset and _n_want == 0:
        print("[onset] no onset-window events listed (imsil_dl_cluster_config.ONSET_EVENTS = [], decision D6)")
    if _unresolved:                            # never fail silently again
        print(f"[onset] WARNING: {len(_unresolved)} canonical event(s) not in this catalog, "
              f"NO onset window applied: {', '.join(_unresolved)}")
    cfg = hcc.make_config(output_root=a.run_root, num_cores=a.cores, onset_dl_ids=onset_ids, mainshock_dl_id=ms_id, open_wdcc=a.open_wdcc)
    ese.write_framework_catalog(cat, cfg.event_catalog_csv, magnitudes={d: ml for d, (_, ml) in tm.items()})

    # 2. per-event SAC export (idempotent)
    from pipeline import config as kcr_config
    from pipeline.core import evmap
    wf_dir = kcr_config.waveforms_dir(cfg)
    summary = ese.export_catalog(cat, continuous_root=pipe_config.CONTINUOUS,
                                 station_table_root=pipe_config.station_table_dir(a.model), out_root=wf_dir,
                                 pyocto_root=pipe_config.pyocto_dir(a.model), pyocto_velmodel=pipe_config.PYOCTO_VELMODEL)
    ok = int((summary.status == "ok").sum() + (summary.status == "skip_existing").sum())
    print(f"[export] {ok}/{len(summary)} events have SACs under {wf_dir}")
    if ok < 2:
        raise SystemExit("fewer than 2 events exported")
    # Freeze cuspid <-> event-dir mapping AND append catalog events the manifest does not know yet.  The framework
    # processes ONLY manifest events, so without new_event_ids a run staged from an older run root silently keeps the
    # old event list (2026-09-30: the 9 events of the extended detection were exported but never relocated).
    _n0 = len(pd.read_csv(evmap.manifest_path(cfg))) if os.path.exists(evmap.manifest_path(cfg)) else 0
    mp = evmap.pin_manifest(cfg, new_event_ids=cat.event_id.astype(str))
    _man = pd.read_csv(mp, dtype={"event_id": str})
    _miss = set(cat.event_id.astype(str)) - set(_man.event_id)
    if _miss:
        raise SystemExit(f"[manifest] {len(_miss)} catalog events missing from {mp}: {sorted(_miss)[:5]}")
    print(f"[manifest] {mp}: {len(_man)} events ({len(_man) - _n0} appended; {len(_man) - len(cat)} not in this catalog)")
    if a.export_only:
        return

    # 3. framework stages on the exported subset
    from pipeline.core import pipeline as kcr_pipeline, hypodd, xcorr as kcr_xcorr
    from pipeline import viz
    if a.xcorr_chunk:
        os.environ["XCORR_PAIR_CHUNK"] = str(a.xcorr_chunk)
    if a.blocked_pairs:
        # Cache-friendly pair order (driver-local monkeypatch; the framework file is untouched):
        # run_xcorr builds pairs with itertools.combinations(events, 2) in lexicographic order, whose
        # working set per row is every remaining event -> the 20k-trace LRU thrashes and each row
        # re-reads most of the on-disk interp cache (measured 23.7% of 115k pairs in 8.6 h, disk-bound).
        # Block order: for block a, iterate blocks b >= a and emit all (i in a, j in b, i<j): block a's
        # traces stay resident while b's stream once -> ~10-20x fewer trace loads. Results are per pair,
        # so the order has no effect on dt.cc content.
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
    if a.bootstrap_only:
        print("[run_dl_hypodd] --bootstrap-only: using the existing dt.cc solution in", kcr_config.dtcc_dir(cfg))
    else:
        res = kcr_pipeline.run_cluster(cfg, stage_from=a.stage_from, through=a.through,
                                       velmodels=("kim1983", "kim2011"), arc_velmodel="kim1983", cores=a.cores,
                                       xcorr_resume=a.xcorr_resume)
        print("[run_dl_hypodd] framework:", {k: v for k, v in res.items() if k != "waveforms"})
        if a.through != "dtcc":
            return

    # 4. bootstrap errors + final catalog (PocketQuake schema)
    if a.n_boot:
        hypodd.bootstrap_relocation(cfg, "dtcc", n=a.n_boot, cores=None)
    tab = viz.location_table(cfg, branch="dtcc", save=True)
    print(f"[final] final_locations.csv: {len(tab)} events, {int(tab.under_constrained.sum()) if len(tab) else 0} under-constrained")

    # 5. project catalog: + template match, ML, tier columns from the DL catalog
    tab["event_id"] = tab.event_id.astype(str)
    tab["template_match"] = tab.event_id.map(lambda e: tm.get(e, ("-", np.nan))[0])
    tab["ML_template"] = tab.event_id.map(lambda e: tm.get(e, ("-", np.nan))[1])
    q = cat.set_index("event_id")[["num", "gap", "rms", "erh", "erz"]].add_prefix("hyp_")
    tab = tab.merge(q, left_on="event_id", right_index=True, how="left")
    tab["hyp_num"] = tab.hyp_num.astype("Int64")
    out_csv = OUT_CSV if os.path.realpath(a.run_root) == os.path.realpath(hcc.RUN_ROOT) \
        else os.path.join(a.run_root, "imsil_dl_relocated.csv")       # test runs stay in their run root
    tab.to_csv(out_csv, index=False)
    if len(years) > 1 and os.path.basename(os.path.normpath(a.run_root)) == "hypodd_dl_full":   # the joint 2020-2026 product
        full_csv = os.path.join(os.path.dirname(OUT_CSV), "imsil_dl_relocated_full.csv"); tab.to_csv(full_csv, index=False); print("[final] joint catalog ->", full_csv)
    hc = tab[(np.hypot(tab.ex95_m, tab.ey95_m) < 100) & (tab.ez95_m < 100) & ~tab.under_constrained]
    print(f"[final] {out_csv}: {len(tab)} relocated; high-confidence (<100 m 95% errors) {len(hc)}; "
          f"template-matched {int((tab.template_match != '-').sum())}")


if __name__ == "__main__":
    main()
