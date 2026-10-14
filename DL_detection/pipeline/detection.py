"""CLI: picker detection for one year.

Examples:
  python detection.py --model phasenet_plus --year 2024 --days 164
  python detection.py --model phasenet_plus --year 2024 --days 164-366
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def parse_days(s):
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-")
        return range(int(a), int(b) + 1)
    return [int(s)]


def main():
    ap = argparse.ArgumentParser(description="Run picker detection for one year.")
    ap.add_argument("--model", default="phasenet_plus",
                    help="picker model (phasenet_plus or a SeisBench PhaseNet weight)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--days", default=None, help="day-of-year range 'A-B' or single 'A' (default: whole year)")
    ap.add_argument("--stations", default=None,
                    help="comma-separated 'NET/STA' station dirs (default: auto-discover)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"], help="default: cuda if available")
    ap.add_argument("--workers", type=int, default=None, help="preprocess worker processes")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="recompute days whose picks CSV already exists")
    ap.add_argument("--min-prob", type=float, default=None,
                    help="pick probability threshold (phasenet_plus; default config.PNPLUS_MIN_PROB)")
    ap.add_argument("--highpass", type=float, default=None,
                    help="highpass freq Hz for phasenet_plus (0=raw; default config.PNPLUS_HIGHPASS)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    stations = a.stations.split(",") if a.stations else None
    core.run_detection_year(a.model, a.year, days=parse_days(a.days), stations=stations,
                            skip_existing=a.skip_existing, device=a.device,
                            workers=a.workers, force=a.force,
                            min_prob=a.min_prob, highpass=a.highpass)


if __name__ == "__main__":
    main()
