"""HYPOINVERSE .sum parsing + QC filter (vendored).

The Buan project imported these from the Ulsan project's uf_cluster module through a path that
has since moved (02.Ulsan_Fault_detection/KS_KG/HypoInv -> src/uflib). To keep this project
self-contained the two functions the pipeline needs are vendored here VERBATIM from
/home/msseo/works/02.Ulsan_Fault_detection/src/uflib/uf_cluster.py (read_sum, apply_qc, QC).
"""
import pandas as pd

# Ulsan default; the Haenam project passes config.HAENAM_QC / _QC_A / _QC_B explicitly.
QC = dict(erh=5.0, erz=5.0, gap=270.0, num=5, rms=1.0)

_SUM_COLMAP = [("LAT", "lat"), ("LON", "lon"), ("DEPTH", "depth"), ("NUM", "num"),
               ("GAP", "gap"), ("RMS", "rms"), ("ERH", "erh"), ("ERZ", "erz"), ("QASR", "qual")]


def _col(df, name):
    """A single column by (stripped) name, taking the first if duplicated — space-padded HYPOINVERSE
    headers can strip to the same name."""
    s = df[name]
    return s.iloc[:, 0] if getattr(s, "ndim", 1) > 1 else s


def read_sum(path):
    """Read one HYPOINVERSE `.sum` (comma-separated, space-padded headers) into a tidy frame:
    time, lat, lon, depth, num, gap, rms, erh, erz, qual.

    Robust to the overflow rows PhaseNet+ can emit (`********` in a numeric field): every numeric
    column is coerced (`errors="coerce"`, bad -> NaN), so one junk row survives as NaN (then dropped
    by `apply_qc`) instead of crashing the whole year. `qual` keeps the QASR string (1st char
    is the A/B/C/D quality)."""
    df = pd.read_csv(path, sep=",")
    df.columns = df.columns.str.strip()
    datecol = [c for c in df.columns if c.startswith("DATE")][0]
    sec = pd.to_numeric(_col(df, "SEC"), errors="coerce")            # coerce overflow / str -> NaN
    out = pd.DataFrame()
    out["time"] = (pd.to_datetime(df[datecol], format="%Y/%m/%d %H:%M", errors="coerce")
                   + pd.to_timedelta(sec, unit="s"))
    for src, dst in _SUM_COLMAP:
        s = _col(df, src)
        out[dst] = s.astype(str).str.strip() if dst == "qual" else pd.to_numeric(s, errors="coerce")
    return out


def apply_qc(df, qc=QC):
    """Filter a catalog by the quality criterion: erh<v, erz<v, gap<v, num>v, rms<v
    (strict `<` on each; num>7 == >= 8 picks). NaN rows (overflow / unlocated errors) fail every
    comparison and drop out, which is what we want. Returns a re-indexed copy."""
    m = ((df["erh"] < qc["erh"]) & (df["erz"] < qc["erz"])
         & (df["gap"] < qc["gap"]) & (df["num"] > qc["num"])
         & (df["rms"] < qc["rms"]))
    return df[m].reset_index(drop=True)
