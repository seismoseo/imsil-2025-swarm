"""Template catalog from the Imsil 25 m reference catalog (IMSIL PORT of build_template_catalog_ref594.py).

IMSIL PORT: source = catalogs/reference_25m/DL_reference_25m.csv (catalogs/build_reference_25m.py); ML = absolute ML ml_min3
from local_magnitudes/full_period/full_period_ML.csv when present (informational only: FMF does not use ML); origin =
origin_time_reloc (0.001 s) when present, else origin_time; sequence = "imsil" for every template; picks are read from
runs/hypodd_dl/waveforms_100km/<event_id>/. cuspid = 400000 + chronological index, as Haenam.

Haenam docstring (provenance):

Source: catalogs/reference_full_25m_fitted/DL_focused_25m.csv (all pass the 25 m bootstrap gate) with the nb34
combined magnitudes (local_magnitudes/full_period/relative/focused_ML_combined.csv). cuspid = 400000 + row index
(chronological). Picks for every event live in runs/hypodd_dl_full/waveforms_100km/<event_id>/ (SAC a = P, t0 = S).
Output: catalogs/haenam_templates_ref594.csv in the fmf/templates.py format (sequence 2020 / quiescence / 2026,
confidence high_conf).
Run:  python catalogs/build_template_catalog_ref594.py
"""
import os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import sys; sys.path.insert(0, "/home/msseo/works/72.2025_Imsil_swarm/registry"); import imsil as haenam   # registry paths (IMSIL PORT)
REF = os.path.join(haenam.path(None, "reference_25m_dir"), "DL_reference_25m.csv")                       # IMSIL PORT
ML = os.path.join(ROOT, "local_magnitudes", "full_period", "full_period_ML.csv")                         # IMSIL PORT (absolute ML)
OUT = haenam.path(None, "tm_templates_594")
CUSPID_OFFSET = 400000
SEQ = {"2020 period": "2020", "Quiescence": "quiescence", "2026 sequence": "2026"}   # Haenam (not used)


def main():
    d = pd.read_csv(REF, dtype={"event_id": str})                                                        # IMSIL PORT
    if os.path.exists(ML):
        m = pd.read_csv(ML, dtype={"event_id": str, "eid": str}); m["event_id"] = m.get("eid", m.event_id)
        d = d.merge(m[["event_id", "ml_min3"]].rename(columns={"ml_min3": "ml_combined"}), on="event_id", how="left", validate="one_to_one")
        d["ml_source"] = np.where(d.ml_combined.notna(), "ml_min3", "")
    else:
        d["ml_combined"] = np.nan; d["ml_source"] = ""
    tcol = "origin_time_reloc" if "origin_time_reloc" in d else ("t_reloc" if "t_reloc" in d else "origin_time")
    d["origin_time_UTC"] = pd.to_datetime(d[tcol]); d = d.sort_values("origin_time_UTC").reset_index(drop=True)
    d["cuspid"] = CUSPID_OFFSET + np.arange(len(d))
    d["horiz_err95_m"] = np.hypot(d.ex95_m, d.ey95_m); d["vert_err95_m"] = d.ez95_m
    d["sequence"] = "imsil"; d["confidence"] = "high_conf"                                               # IMSIL PORT
    out = d[["cuspid", "event_id", "origin_time_UTC", "latitude", "longitude", "depth_km", "ml_combined", "ml_source",
             "horiz_err95_m", "vert_err95_m", "cc_links", "ct_links", "under_constrained", "sequence", "confidence"]]
    out = out.rename(columns={"ml_combined": "ML", "cc_links": "n_cc_links", "ct_links": "n_ct_links"})
    for e in out.event_id:
        assert os.path.isdir(os.path.join(ROOT, "runs", "hypodd_dl", "waveforms_100km", e)), e           # IMSIL PORT
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} templates"); print(out.groupby("sequence").size().to_string())
    print("ML available:", int(out.ML.notna().sum()), "| median horiz95 / vert95 (m):", round(out.horiz_err95_m.median(), 1), "/", round(out.vert_err95_m.median(), 1))


if __name__ == "__main__":
    main()
