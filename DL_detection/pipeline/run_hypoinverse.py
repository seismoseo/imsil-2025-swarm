"""CLI: run HYPOINVERSE (hyp1.40) for one year + velocity model.

Example:
  python run_hypoinverse.py --model phasenet_plus --year 2024 --velmodel kim1983
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import config


def main():
    ap = argparse.ArgumentParser(description="Locate events with HYPOINVERSE.")
    ap.add_argument("--model", default="phasenet_plus")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--velmodel", default=config.DEFAULT_VELMODEL, help="crustal model dir (kim1983/kim2011)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    core.run_hypoinverse_year(a.model, a.year, velmodel=a.velmodel, force=a.force)


if __name__ == "__main__":
    main()
