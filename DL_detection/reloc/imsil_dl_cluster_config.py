"""ClusterConfig for the standalone HypoDD relocation of the Imsil DL catalog (korea-cluster-relocation).

IMSIL PORT of reloc/haenam_dl_cluster_config.py. Every relocation setting is unchanged (STD_HYP, STD_PH2DT, dt.ct
5-row and dt.cc 7-row schedules, OBSCC 4, LSQR, adaptive damping, xcorr 1000 Hz / 5-20 Hz / +/-0.5 s / CC >= 0.7, GPU
batched backend, cuspid offset 200000, distance pick weights, optional open WDCC). Changed: KCR_ROOT (PAGO copy),
EPICENTER / REGION_BOUNDS (= config.IMSIL_EPICENTER / IMSIL_BOX), name/region labels, and the onset-window list:
Haenam applied +/-0.05 s / 1-40 Hz windows to its explicit list of M >= 2 events; Imsil has no M >= 2 event, so the
lists are EMPTY by default (decision D6). To test the ML 1.9 alone, put its KMA id "20251102203835" (an id of
catalogs/imsil_kma24.csv, mapped to the DL id by run_dl_hypodd) in ONSET_EVENTS and use a scratch --run-root.

Haenam docstring (provenance):

Mirrors the CANONICAL Haenam run (pipeline/clusters/Haenam_joint.py via clusters/_base.mixed_cluster) so
the DL-relocated catalog is directly comparable with the 195-event PocketQuake result:
  * kim1983 (+ kim2011 for the .sum) with the framework's STD_HYP / STD_PH2DT control blocks
  * dt.ct then dt.cc with the 5-row / 7-row weighting schedules, OBSCC 4, adaptive damping
  * xcorr: 1000 Hz lanczos, 5-20 Hz, +/-0.5 s window, cc >= 0.7, GPU batched backend, interp cache
  * onset-window override (+/-0.05 s, 1-40 Hz) for every M >= 2 event, mainshock = the M2.7B
    (20260830174920) exactly as in the canonical config — mapped here onto the DL event ids
  * LSQR (isolv 2): the canonical run's hypoDD.inp is ISOLV 2 as well (its SVD attempt overflowed);
    errors come from the 1000-replica bootstrap in both cases.
The SAC tree is produced by reloc/event_sac_export.py from the continuous archive (PyOcto picks), and
the framework is entered at stage_from="hypoinverse" (its write_phs/write_sta re-run hyp1.40 on the
exported subset so the cuspid <-> SAC-dir mapping is correct by construction — Buan recipe).
"""
import dataclasses
import os
import sys

KCR_ROOT = "/home/msseo/works/72.2025_Imsil_swarm/_external/korea-cluster-relocation"   # IMSIL PORT: PAGO copy (moho @663c26b + local diff)
if KCR_ROOT not in sys.path:
    sys.path.insert(0, KCR_ROOT)

from pipeline.config import ClusterConfig, VelModel                       # noqa: E402
from pipeline.clusters._base import STD_HYP, STD_PH2DT, KIM1983, KIM2011, _dtct_inp, _dtcc_variants  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # DL_detection
RUN_ROOT = os.path.join(ROOT, "runs", "hypodd_dl")
EPICENTER = (35.5506, 127.2951)                                              # IMSIL PORT: = config.IMSIL_EPICENTER
REGION_BOUNDS = (35.44, 35.66, 127.17, 127.42)                              # IMSIL PORT: = config.IMSIL_BOX
OBSCC = 4                                                                   # canonical Haenam value (framework default)
ISOLV = 2                                                                   # LSQR, as the canonical hypoDD.inp

# The canonical onset-window list (clusters/Haenam_joint.py): all 2026 M >= 2 events. Keyed by the
# relocation's event ids; the driver maps them to the DL event ids (origin times can differ by ~1 s).
# IMSIL PORT (D6): the Imsil onset list. Empty = no onset-window override (no M >= 2 event in the Imsil catalog).
# The Haenam lists below are kept for provenance only and are NOT used (run_dl_hypodd reads ONSET_EVENTS).
ONSET_EVENTS = [e for e in os.environ.get("IMSIL_ONSET_EVENTS", "").split(",") if e]   # default [] (D6); test: IMSIL_ONSET_EVENTS=20251102203835
ONSET_MAINSHOCK = None                     # only selects the event dropped from the auxiliary dt.cc _no_main variant
HAENAM_ONSET_EVENTS_2026 = [
    "20260814084338", "20260814233134", "20260816002344", "20260816103411", "20260820235000",
    "20260821192129", "20260822214301", "20260825004122", "20260826070459", "20260827224426",
    "20260829013013", "20260830174920",
    # 2026-09-09 ML 2.3: postdates the original list (frozen before it occurred).  Added
    # 2026-09-29 so it gets the same +/-0.05 s / 1-40 Hz cross-correlation onset window as the
    # other M >= 2 events.  CANONICAL_MAINSHOCK is deliberately NOT changed: that only selects
    # which event is DROPPED from the auxiliary dt.cc..._no_main leave-one-out variant.
    "20260909000405"]
HAENAM_ONSET_EVENTS_2020 = [                                                # the five 2020 M >= 2 events of the canonical list
    "20200428035201", "20200429221313", "20200501175702", "20200503130714", "20200508165739"]
HAENAM_MAINSHOCK = "20260830174920"                                          # M2.7B (Haenam; not used)
# names read by run_dl_hypodd.py
CANONICAL_ONSET_EVENTS_2020, CANONICAL_ONSET_EVENTS_2026, CANONICAL_MAINSHOCK = [], list(ONSET_EVENTS), ONSET_MAINSHOCK


def _onset_extras(dl_ids, mainshock_dl_id):
    if not dl_ids:
        return {}
    return dict(mainshock_event_id=mainshock_dl_id or dl_ids[0],
                xcorr_pair_overrides={frozenset(dl_ids): dict(pre=0.05, post=0.05, bandpass=(1, 40))})


def _open_wdcc(inp):
    """Same 7-row schedule with the cc inter-event distance cutoff (WDCC) removed in the four cc-dominant sets.
    Rationale (2026-09-10): with the canonical 4/2/1/0.5 km cutoffs, 49 of 475 events — including two of the
    Aug 14 foreshocks with ~2,700 cc observations >= 0.7 — were pushed off the core by the catalog-dominated
    sets 1-3 and then lost every cc pair in the tightening cc sets (nccp = nccs = 0, catalog-only positions).
    Scratch test on the full 481-event dt.cc set (2026-09-10): the open schedule moves the well-linked (>= 20 cc
    links) events by 1 m median / 3 m 95th pct relative to the canonical schedule and gives all 49 stranded events
    cc links (39 with >= 20). Rejected alternatives (tested, NOT used): removing the cc residual cut (WRCC -9) in the
    first one or two cc-dominant sets perturbs the well-linked events by 17 m median / ~250 m 95th pct and still
    leaves the two badly-started foreshocks (200005, 200009) S-only and ~0.6 km shallow; their cluster membership
    rests on the JN01 S-P times (within 20 ms of the M2.1) and waveform similarity instead. Mechanism of the strand:
    ISTART=1 starts every event at the centroid, sets 1-3 move an event with a bad far-station DL pick (e.g. MIAA S
    260 ms early for 200009) 1-3 km off the core, and the cc residual cut (6 sigma) then removes its cc data before
    they can pull it back. Follow-up (same day): analysis/sp_consistency.py (JN01 S-P versus the relocated position) flags
    44 events relocated 1-8 km from the core as pick-driven mislocations; cc-first weighting in sets 1-3 (equal cc/catalog
    weights, or cc-dominant throughout) halves them and fixes both foreshocks but doubles the depth residual against the
    canonical 2026 relocation (3 -> 6 m median, 90th pct 12 -> 21 m), so the catalog-first SOTA schedule is kept."""
    rows = tuple(tuple(list(r[:4]) + [-9] + list(r[5:])) if r[1] >= 0.5 else r for r in inp.iter_sets)
    return dataclasses.replace(inp, iter_sets=rows)


def make_config(output_root=RUN_ROOT, num_cores=24, onset_dl_ids=(), mainshock_dl_id=None, obscc=OBSCC, isolv=ISOLV,
                open_wdcc=False):
    variants = {k: dataclasses.replace(v, obscc=obscc) for k, v in _dtcc_variants(isolv=isolv).items()}
    if open_wdcc:
        variants = {k: _open_wdcc(v) for k, v in variants.items()}
    return ClusterConfig(
        name="imsil_dl",                           # IMSIL PORT: label
        region="ImsilDL",
        src_root=output_root,
        event_catalog_csv=os.path.join(output_root, "event_catalog", "event_catalog.csv"),
        station_master_csvs=(),                     # stations come from the SAC headers (write_sta)
        epicenter=EPICENTER,
        output_root=output_root,
        region_bounds=REGION_BOUNDS,
        velocity_models=(VelModel("kim1983", p_rows=KIM1983["p"], s_rows=KIM1983["s"]),
                         VelModel("kim2011", p_rows=KIM2011["p"], s_rows=KIM2011["s"])),
        hyp_control=STD_HYP,
        ph2dt=STD_PH2DT,
        hypodd_dtct=_dtct_inp(isolv=isolv),
        hypodd_dtcc_variants=variants,
        phs_weight_scheme="distance",               # exported SACs carry no per-pick probabilities
        xcorr_backend="cctorch_gpu_batched",
        xcorr_interp_cache=True,
        cuspid_offset=200000,
        num_cores=num_cores,
        **_onset_extras(list(onset_dl_ids), mainshock_dl_id),
    )
