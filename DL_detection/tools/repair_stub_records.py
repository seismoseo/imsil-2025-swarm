#!/usr/bin/env python
"""Remove junk stub records from MARA miniSEED day files and relink the cleaned copies (decision D19, 2026-10-08).

Some KS day files carry 1-2 junk 4096-byte records (header start 00:00:00 or another impossible time, 3-4 samples, a Steim2
frame the decoder rejects) where the real data stop for up to a minute (2025-12-09 and 2025-12-22, 05:44-05:45 UTC). ObsPy
rejects the whole file on such a record, so the station-day is lost. This script:
  1. scans every KS file of the given days record by record (4096-byte records) and lists the unreadable records;
  2. writes a cleaned copy of each affected file to ../continuous_repaired/<year>/<net>/<sta>/<cha>.D/<file>, made of the
     ORIGINAL bytes of every readable record in the original order (the stub records are left out; nothing is decoded,
     re-encoded or invented);
  3. verifies each copy: ObsPy reads it, every kept record is byte-identical to the original, the stubs are the only records
     dropped, and the sample count equals the sum over the kept records;
  4. points the continuous/ symlink of that file at the cleaned copy (the MARA mirror is never modified).
Re-running tools/link_continuous.py restores the original links; run this script again afterwards.
DRY RUN BY DEFAULT (lists the stub records and the planned copies); --apply writes copies, relinks and the manifest.
    python tools/repair_stub_records.py --days 2025.343 2025.356 [--apply]
"""
import argparse
import glob
import io
import json
import os
import struct
import warnings

import obspy
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                    # DL_detection
PROJECT = os.path.dirname(ROOT)
MIRROR = os.path.join(PROJECT, "continuous_mara")
REPAIRED = os.path.join(PROJECT, "continuous_repaired")
LINKS = os.path.join(ROOT, "continuous")
RL = 4096
warnings.simplefilter("ignore")


def header(rec):
    yr, jd, hh, mm, ss, _, fr = struct.unpack(">HHBBBBH", rec[20:30]); ns, = struct.unpack(">H", rec[30:32])
    return f"{yr}.{jd:03d} {hh:02d}:{mm:02d}:{ss:02d}.{fr:04d}", ns


def scan(path):
    b = open(path, "rb").read()
    assert len(b) % RL == 0, f"{path}: size {len(b)} not a multiple of {RL}"
    good, bad = [], []
    for i in range(len(b) // RL):
        rec = b[i * RL:(i + 1) * RL]
        try:
            tr = obspy.read(io.BytesIO(rec), format="MSEED")[0]; good.append((i, tr.stats.npts))
        except Exception as exc:                                       # noqa: BLE001
            t, ns = header(rec); bad.append(dict(record=i, header_time=t, header_nsamp=ns, error=str(exc).splitlines()[-1][:80]))
    return b, good, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", required=True, help="YEAR.DOY, e.g. 2025.343")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    rows = []
    for day in a.days:
        y, doy = day.split(".")
        for src in sorted(glob.glob(f"{MIRROR}/{y}/KS/*/*.D/KS.*.D.{y}.{doy}")):
            try:
                obspy.read(src, format="MSEED"); continue                 # whole file reads: nothing to do
            except Exception:                                          # noqa: BLE001
                pass
            b, good, bad = scan(src)
            rel = os.path.relpath(src, MIRROR); dst = os.path.join(REPAIRED, rel)
            link = os.path.join(LINKS, *rel.split(os.sep)[1:])           # continuous/<net>/<sta>/<cha>.D/<file>
            rows.append(dict(day=day, file=rel, n_records=len(b) // RL, n_stub=len(bad), stubs=json.dumps(bad), dst=dst, link=link,
                             kept_npts=sum(n for _, n in good)))
            if not a.apply:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                for i, _ in good:
                    f.write(b[i * RL:(i + 1) * RL])
            # verification: readable, kept records byte-identical, only the stubs dropped, sample count preserved
            c = open(dst, "rb").read(); kept = [i for i, _ in good]
            assert len(c) == RL * len(kept)
            assert all(c[k * RL:(k + 1) * RL] == b[i * RL:(i + 1) * RL] for k, i in enumerate(kept))
            assert sorted(kept + [x["record"] for x in bad]) == list(range(len(b) // RL))
            st = obspy.read(dst, format="MSEED"); assert sum(tr.stats.npts for tr in st) == rows[-1]["kept_npts"]
            assert os.path.islink(link) and os.path.realpath(link) == os.path.realpath(src), f"unexpected link state: {link}"
            os.remove(link); os.symlink(dst, link)
            rows[-1]["verified"] = True
    t = pd.DataFrame(rows)
    if t.empty:
        print("no unreadable files on", a.days); return
    print(t.groupby("day").agg(files=("file", "size"), stub_records=("n_stub", "sum")).to_string())
    print(f"{len(t)} files, {int(t.n_stub.sum())} stub records; all stubs claim <= {max(json.loads(s)[0]['header_nsamp'] for s in t.stubs)} samples")
    if not a.apply:
        print("(dry run — add --apply to write cleaned copies to ../continuous_repaired/ and relink continuous/)"); return
    man = os.path.join(REPAIRED, "repair_manifest.csv"); os.makedirs(REPAIRED, exist_ok=True)
    t.to_csv(man, mode="a", header=not os.path.exists(man), index=False)
    print(f"cleaned, verified and relinked {int(t.verified.sum())} files -> {REPAIRED}; manifest {man}")


if __name__ == "__main__":
    main()
