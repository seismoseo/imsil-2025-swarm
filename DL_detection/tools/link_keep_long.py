#!/usr/bin/env python
"""Make the kept day files of the long FMF scan readable by the FMF code again (notebook 13, 2026-10-14).

tools/fmf_long_rolling.py kept the day files of every day with a raw hit as hard links under ../continuous_mara/_keep_long/<YYYY>/<NET>/<STA>/<CHA>.D/
(same layout as the mirror). This script
  --fetch Y.DOY[,Y.DOY...]   pulls the template-sensor files of further days from MARA into _keep_long (the 2018 hit days were deleted
                             before the keep rule existed; DOY 065 of 2018 is one day of the March 2018 noise burst, shown for reference);
  (always)                   links every _keep_long file into continuous/<NET>/<STA>/<CHA>.D/ (the layout tools/link_continuous.py builds),
                             skipping links that already exist.
DRY RUN BY DEFAULT; --apply executes.   python tools/link_keep_long.py [--fetch 2018.041,2018.065,2018.173] [--apply]
"""
import argparse
import glob
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEEP = os.path.abspath(os.path.join(ROOT, "..", "continuous_mara", "_keep_long"))
CONT = os.path.join(ROOT, "continuous")
MARA = "<user>@<mara-host>"
SDS = "<mara-sds-root>"
SHELLY_NPZ = os.path.join(ROOT, "fmf_aligned", "templates_cache", "imsil_shelly_imsil_5_20H_aligned_mad10_long.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", default=""); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        z = np.load(SHELLY_NPZ, allow_pickle=True)
        sp = [(str(s).split(".")[0], str(s).split(".")[1], str(c)) for s, c in zip(z["stations"], z["channels"])]
        rels = [f"{y}/{n}/{s}/{p}{c}.D/{n}.{s}..{p}{c}.D.{y}.{int(d):03d}" for y, d in (x.split(".") for x in a.fetch.split(",")) for n, s, p in sp for c in "ENZ"]
        print(f"[fetch] {len(rels)} candidate files ({a.fetch}); missing ones (stations not yet installed) are skipped")
        if a.apply:
            lst = os.path.join(KEEP, "_fetch_list.txt"); os.makedirs(KEEP, exist_ok=True); open(lst, "w").write("\n".join(rels) + "\n")
            r = subprocess.run(["rsync", "-aL", "--ignore-missing-args", "-e", "ssh -o BatchMode=yes", f"--files-from={lst}", f"{MARA}:{SDS}/", f"{KEEP}/"])
            print(f"[fetch] rsync exit {r.returncode}")
    files = [f for f in glob.glob(os.path.join(KEEP, "[12][0-9][0-9][0-9]", "*", "*", "*.D", "*")) if os.path.isfile(f)]
    todo = []
    for f in files:
        y, net, sta, cha, fn = os.path.relpath(f, KEEP).split(os.sep)
        dst = os.path.join(CONT, net, sta, cha, fn)
        if not os.path.lexists(dst):
            todo.append((f, dst))
    print(f"[link] {len(files)} kept files; {len(todo)} links to create")
    if a.apply:
        for f, dst in todo:
            os.makedirs(os.path.dirname(dst), exist_ok=True); os.symlink(f, dst)
        print(f"[link] {len(todo)} links created")
    else:
        print("(dry run: add --apply)")


if __name__ == "__main__":
    main()
