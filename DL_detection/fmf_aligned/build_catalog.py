"""Merge the daily FMF detections into ONE catalog (final workflow, user decisions 2026-10-06).

IMSIL PORT: rules unchanged; REFERENCE_CSV = the Imsil 25 m reference (catalogs/reference_25m/DL_reference_25m.csv);
EXCLUDED_TEMPLATES emptied (the Haenam cuspids would collide with Imsil cuspids 400000+; Imsil coda templates are
decision D13, screened in notebook 06).


Four steps, no other rule:
  1. detector policy (Buan): 12xMAD (already applied per day) AND cc >= ABS_CC_FLOOR AND n_present >= MIN_PRESENT_STATIONS
  2. excluded templates: every hit of EXCLUDED_TEMPLATES is dropped before step 3, as if those events had never been templates
  3. merge: hits closer than GROUP_S in time are one detection; the highest-CC template represents it
  4. classify: a detection is a reference event when an origin of REFERENCE_CSV lies within GROUP_S of it; every other detection is new
Replaces the earlier copy rule (COPY_WITHIN_S), the 20 s near-large-event rule (ALIAS_S) and the DL/KMA time matching (MATCH_S).
Evidence for every choice: fmf/logs/full_period_5_20H/catalog_rules_review/ (README.md there).

Writes C.CATALOG_CSV, *_raw.csv (all daily hits, ungated, for the record) and *_excluded_template_detections.csv (detections that
exist only through an excluded template's hits). FMF_CATALOG_OUT=<csv> writes elsewhere (dry runs); FMF_REFERENCE_CSV overrides the
reference catalog.

Run:  FMF_RUN=full_period_5_20H conda run -n fmf python fmf/build_catalog.py
"""
import glob
import os

import numpy as np
import pandas as pd

import config as C

GROUP_S = 4.0                  # one detection = hits within 4 s (literature declusters at 3-5 s); also the reference-match window
SELF_S = 3.0                   # self_detection flag only (the template's own event): information, not a class
REFERENCE_CSV = os.environ.get("FMF_REFERENCE_CSV", os.path.join(C.PROJECT, "catalogs", "reference_25m", "DL_reference_25m.csv"))   # IMSIL PORT
# Templates whose own event came 5-31 s after a larger event: their correlation windows hold that event's waves (coda / S energy in
# the P windows), so they re-detect every comparable large event at the same delay. Found by their repeat detections at exactly that
# delay (tested over all 594 templates, no time limit: only these six). User decision 2026-10-06: exclude these six only.
EXCLUDED_TEMPLATES = {}   # IMSIL PORT (D13). Haenam values (provenance, NOT used):
HAENAM_EXCLUDED_TEMPLATES = {
    400554: "5.28 s after a larger event (107 picks, ML 1.57); background before its P windows 49x normal; 5 repeat detections",
    400317: "13.15 s after a larger event (67 picks, ML 1.29); background 3.7x normal; 3 repeat detections",
    400187: "21.09 s after a larger event (85 picks, ML 1.29); background 3.2x normal; 2 repeat detections",
    400444: "23.34 s after a larger event (109 picks, ML 2.65); background 23x normal; 7 repeat detections",
    400357: "29.55 s after a larger event (107 picks, ML 2.47); background 7.8x normal; 6 repeat detections",
    400273: "31.33 s after a larger event (106 picks, ML 2.47); background 25x normal; 8 repeat detections",
}


def merge(df):
    """Hits closer than GROUP_S (chained) form one detection, represented by its highest-CC hit."""
    df = df.sort_values("origin").reset_index(drop=True)
    t = df.origin.values.astype("datetime64[ns]").astype("int64") / 1e9
    df["group"] = np.r_[0, np.cumsum(np.diff(t) > GROUP_S)]
    rep = df.sort_values("cc", ascending=False).groupby("group").first().reset_index()
    rep["n_templates"] = df.groupby("group").size().values
    return rep.sort_values("origin").reset_index(drop=True)


def nearest(times, ref):
    """For each time: index into ref of the nearest reference time and the absolute offset in seconds."""
    ref = np.asarray(ref, dtype="datetime64[ns]"); t = np.asarray(times, dtype="datetime64[ns]")
    i = np.clip(np.searchsorted(ref, t), 1, len(ref) - 1)
    d0 = np.abs((t - ref[i - 1]) / np.timedelta64(1, "s")); d1 = np.abs((ref[i] - t) / np.timedelta64(1, "s"))
    j = np.where(d0 <= d1, i - 1, i)
    return j, np.minimum(d0, d1)


def main():
    out_csv = os.environ.get("FMF_CATALOG_OUT", C.CATALOG_CSV)
    files = sorted(glob.glob(f"{C.DETECT_DIR}/det_*.parquet"))
    if not files:
        raise SystemExit(f"no daily detections in {C.DETECT_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["origin"] = pd.to_datetime(df.origin_utc.apply(lambda u: u.datetime if hasattr(u, "datetime") else u),
                                  utc=True).dt.tz_localize(None)
    df = df.sort_values("origin").reset_index(drop=True)
    n_raw = len(df)
    df.to_csv(out_csv.replace(".csv", "_raw.csv"), index=False)

    # 1. detector policy
    df = df[(df.cc >= C.ABS_CC_FLOOR) & (df.n_present >= C.MIN_PRESENT_STATIONS)].reset_index(drop=True)
    n_gated = len(df)

    # 2. excluded templates: drop their hits before the merge (never "every event they fired on")
    bad = df.template_cuspid.astype(int).isin(EXCLUDED_TEMPLATES)
    with_bad = merge(df)
    df = df[~bad].reset_index(drop=True)

    # 3. merge
    rep = merge(df)
    # record the detections that existed only through excluded templates' hits
    _, d = nearest(with_bad.origin.values, rep.origin.values)
    lost = with_bad[d > GROUP_S]
    lost[["origin", "template_cuspid", "cc", "n_present", "n_templates"]].to_csv(
        out_csv.replace(".csv", "_excluded_template_detections.csv"), index=False)

    # 4. classify against the reference catalog
    ref = pd.read_csv(REFERENCE_CSV, parse_dates=["origin_time"]).sort_values("origin_time").reset_index(drop=True)
    j, d = nearest(rep.origin.values, ref.origin_time.values)
    rep["is_reference"] = d <= GROUP_S
    rep["reference_event_id"] = np.where(rep.is_reference, ref.event_id.astype(str).values[j], "")
    rep["reference_offset_s"] = np.where(rep.is_reference, d, np.nan)
    rep["is_new"] = ~rep.is_reference
    tpl = np.load(C.SHELLY_NPZ if os.path.exists(C.SHELLY_NPZ) else C.TEMPLATE_NPZ, allow_pickle=True)
    tmap = dict(zip(tpl["cuspid"].astype(int), pd.to_datetime(pd.Series(tpl["origin_utc"].astype(str))).values))
    rep["self_detection"] = [abs((np.datetime64(o) - tmap[int(c)]) / np.timedelta64(1, "s")) < SELF_S
                             for c, o in zip(rep.template_cuspid, rep.origin)]
    rep["year"] = rep.origin.dt.year
    rep["doy"] = rep.origin.dt.dayofyear

    # checks: every reference event inside the scanned span is one detection, and no detection holds two reference events
    span = (ref.origin_time >= rep.origin.min() - pd.Timedelta(seconds=GROUP_S)) & (ref.origin_time <= rep.origin.max() + pd.Timedelta(seconds=GROUP_S))
    k, dk = nearest(ref.origin_time[span].values, rep.origin.values)
    unmatched = ref[span][dk > GROUP_S]
    dup = rep.reference_event_id[rep.is_reference].duplicated().sum()
    if dup:
        raise SystemExit(f"{dup} reference events matched by more than one detection")

    cols = ["origin", "year", "doy", "template_cuspid", "template_idx", "cc", "n_present", "n_templates", "self_detection",
            "is_reference", "reference_event_id", "reference_offset_s", "is_new"] + \
           (["n_config_segments"] if "n_config_segments" in rep.columns else [])
    rep[cols].to_csv(out_csv, index=False)
    print(f"daily files: {len(files)}  raw hits: {n_raw}  after floor/gate: {n_gated}")
    print(f"excluded templates: {len(EXCLUDED_TEMPLATES)}  hits dropped: {int(bad.sum())}  detections lost: {len(lost)}")
    print(f"detections: {len(rep)}  reference events: {int(rep.is_reference.sum())} of {int(span.sum())} in the scanned span "
          f"(unmatched: {len(unmatched)}; largest offset {np.nanmax(rep.reference_offset_s):.3f} s)  new: {int(rep.is_new.sum())}")
    print(f"reference catalog: {REFERENCE_CSV}")
    print(f"-> {out_csv}")


if __name__ == "__main__":
    main()
