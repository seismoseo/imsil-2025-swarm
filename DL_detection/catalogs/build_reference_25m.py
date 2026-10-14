#!/usr/bin/env python
"""Build the Imsil 25 m reference catalog from the DL relocation (S4 of the Imsil port).

Selection = the Haenam rule, copied from 27.Haenam_sequence/DL_detection/notebooks/full_period_publication.py load():
an event is RETAINED when its bootstrap 95 % half-widths (Q97.5 - Q2.5)/2 in east, north and depth are all finite and
<= 25 m (computed before rounding). No phase/station, magnitude, plane-distance or depth gate applies to membership.
As in Haenam, the half-widths of bootstrap_errors.csv are re-computed from bootstrap_samples.npz and must agree.

Inputs  (run root runs/hypodd_dl unless --run-root):
    catalogs/imsil_dl_relocated.csv            (reloc/run_dl_hypodd.py; + origin_time_reloc from analysis/add_reloc_origin_times.py)
    <run>/event_manifest.csv                   (cuspid = 200000 + event_idx)
    <run>/2.HypoDD/02.dt.cc/bootstrap_errors.csv, bootstrap_samples.npz
Outputs (catalogs/reference_25m/):
    DL_all_with_uncertainty.csv   every relocated event with ex95/ey95/ez95 (m), max95_m, pass25, reason
    DL_reference_25m.csv          the retained events (column eid = event_id, as Haenam's reference files)
    threshold_sensitivity.csv     retained counts for bounds 5 ... 1000 m
    selection_manifest.json       rule, counts, input sha256;   README.md
DRY RUN BY DEFAULT (prints counts only); --apply writes the outputs.
    python catalogs/build_reference_25m.py [--run-root runs/hypodd_dl] [--apply]
"""
import argparse
import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                     # DL_detection
BOUND_M = 25.0                                                   # Haenam reference rule
OUTDIR = os.path.join(HERE, "reference_25m")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=os.path.join(ROOT, "runs", "hypodd_dl"))
    ap.add_argument("--relocated", default=os.path.join(HERE, "imsil_dl_relocated.csv"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--outdir", default=OUTDIR, help="output folder (default catalogs/reference_25m); e.g. catalogs/reference_25m_fmf for the FMF relocation")
    a = ap.parse_args()
    branch = os.path.join(a.run_root, "2.HypoDD", "02.dt.cc")
    paths = dict(relocated=a.relocated, manifest=os.path.join(a.run_root, "event_manifest.csv"),
                 errors=os.path.join(branch, "bootstrap_errors.csv"), samples=os.path.join(branch, "bootstrap_samples.npz"))
    for k, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f"missing {k}: {p}")

    # --- same merge and rule as Haenam full_period_publication.load()
    d = pd.read_csv(paths["relocated"], dtype={"event_id": str, "template_match": str})
    m = pd.read_csv(paths["manifest"], dtype={"event_id": str}); m["id"] = 200000 + m.event_idx
    e = pd.read_csv(paths["errors"], comment="#").merge(m[["id", "event_id"]], on="id", validate="one_to_one")
    d = d.merge(e[["event_id", "id", "ex95", "ey95", "ez95", "n_boot"]], on="event_id", validate="one_to_one", suffixes=("", "_raw"))
    assert d.event_id.is_unique
    errs = d[["ex95", "ey95", "ez95"]].to_numpy()
    d["pass25"] = np.isfinite(errs).all(axis=1) & (errs <= BOUND_M).all(axis=1)
    d["max95_m"] = np.max(errs, axis=1)
    d["reason"] = np.where(d.pass25, "retained",
                           ["; ".join(k for k, v in zip(["east", "north", "vertical"], row) if not np.isfinite(v) or v > BOUND_M) for row in errs])
    # --- check: half-widths re-computed from the raw replicas (Haenam assert)
    sample = np.load(paths["samples"])["data"]; ids = sample[:, 0].astype(int); order = np.argsort(ids, kind="stable"); sample = sample[order]
    uid, start, count = np.unique(sample[:, 0].astype(int), return_index=True, return_counts=True)
    samples = {i: sample[j:j + n, 1:] for i, j, n in zip(uid, start, count)}
    recomputed = np.array([np.diff(np.percentile(samples[i], [2.5, 97.5], axis=0), axis=0)[0] / 2 for i in d.id])
    np.testing.assert_allclose(recomputed, errs, rtol=1e-10, atol=1e-8)
    np.testing.assert_array_equal([len(samples[i]) for i in d.id], d.n_boot_raw)
    d["eid"] = d.event_id
    q = d[d.pass25].copy()
    thr = pd.DataFrame([dict(bound_m=b, total=len(d), retained=int((d.max95_m <= b).sum())) for b in (5, 10, 15, 25, 50, 100, 250, 500, 1000)])
    kma = int((q.template_match.notna() & (q.template_match != "-")).sum()) if "template_match" in q else 0
    print(f"relocated {len(d)}; retained (all three 95 % half-widths finite and <= {BOUND_M:g} m) {len(q)}; "
          f"of them KMA-listed {kma}; excluded {len(d) - len(q)}")
    print(thr.to_string(index=False))
    if not a.apply:
        print("(dry run — add --apply to write catalogs/reference_25m/)"); return
    OUT = a.outdir; os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "DL_all_with_uncertainty.csv"), index=False)
    q.to_csv(os.path.join(OUT, "DL_reference_25m.csv"), index=False)
    thr.to_csv(os.path.join(OUT, "threshold_sensitivity.csv"), index=False)
    man = dict(reference="Imsil 2025 DL catalog, 25 m bootstrap reference (Haenam rule)",
               built=datetime.datetime.now().isoformat(timespec="seconds"),
               selection=f"All E/N/depth (Q97.5-Q2.5)/2 finite and <= {BOUND_M:g} m; calculated before rounding. "
                         "No phase/station, ML, plane-distance or depth gate (the relocation itself used --min-depth as run).",
               counts=dict(relocated=len(d), retained=len(q), retained_KMA_listed=kma),
               inputs=[dict(name=k, path=p, sha256=sha(p)) for k, p in paths.items()])
    json.dump(man, open(os.path.join(OUT, "selection_manifest.json"), "w"), indent=1)
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("# Imsil 25 m reference catalog\n\n"
                f"`DL_reference_25m.csv` holds the {len(q)} of {len(d)} relocated DL events whose bootstrap 95 % half-widths in "
                "east, north and depth are all finite and at most 25 m (the Haenam reference rule). No other location "
                "selection applies. `DL_all_with_uncertainty.csv` lists every relocated event with its half-widths and the "
                "reason for exclusion. Inputs and their hashes are in `selection_manifest.json`. Built by "
                "`catalogs/build_reference_25m.py`.\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
