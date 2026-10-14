#!/usr/bin/env python3
"""Print the live cross-correlation progress of runs/hypodd_dl (pair index from a py-spy stack dump).
Usage: xcorr_progress.py [PID]  -> 'done total pct' or 'nopair'."""
import os, subprocess, sys, re
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pid = sys.argv[1] if len(sys.argv) > 1 else subprocess.run(
    ["pgrep", "-f", "^/home/msseo/miniforge3/envs/pq-gpu/bin/python reloc/run_dl_hypodd.py"],
    capture_output=True, text=True).stdout.split()[0]
dump = subprocess.run(["timeout", "60", "/home/msseo/miniforge3/bin/py-spy", "dump", "--pid", pid, "--locals"],
                      capture_output=True, text=True).stdout
m = re.search(r"^\s+pair: \(\"(\d{14})\", \"(\d{14})\"\)", dump, re.M)
if not m:
    print("nopair"); sys.exit(0)
ids = sorted(pd.read_csv(os.path.join(ROOT, "runs/hypodd_dl/event_manifest.csv"), dtype={"event_id": str}).event_id)
n = len(ids); i, j = ids.index(m.group(1)), ids.index(m.group(2))
done = sum(n - 1 - k for k in range(i)) + (j - i - 1)
tot = n * (n - 1) // 2
print(done, tot, round(100 * done / tot, 1))
