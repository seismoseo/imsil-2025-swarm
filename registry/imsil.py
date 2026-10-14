"""Path registry of the Imsil DL catalog project (72.2025_Imsil_swarm, on PAGO).

Drop-in for the Haenam registry loader as used by the ported DL_detection scripts, which call only
``haenam.path(None, key)`` (they import this module as ``import imsil as haenam``). Each Haenam key used by
those scripts is mapped to its Imsil counterpart in layout.csv; keys that have no Imsil counterpart (the Haenam
canonical 195-event frame and plane) raise KeyError with the reason, instead of silently pointing somewhere.

    import sys; sys.path.insert(0, "/home/msseo/works/72.2025_Imsil_swarm/registry"); import imsil as R
    R.path(None, "tm_templates_195")   # the 24 KMA-listed Imsil events (UTC, KMA ML)
    R.path(None, "tm_dir")             # DL_detection/fmf
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                                   # 72.2025_Imsil_swarm
LAYOUT_CSV = os.path.join(HERE, "layout.csv")


def _layout():
    with open(LAYOUT_CSV, newline="") as f:
        return {r["product"]: r for r in csv.DictReader(f)}


def path(k, product, variant=None):
    """Absolute path of a global product. Only k=None is supported (Imsil has no per-event products here)."""
    if k is not None or variant is not None:
        raise KeyError(f"imsil registry: per-event products are not defined (k={k!r}, variant={variant!r})")
    lay = _layout()
    if product not in lay:
        raise KeyError(f"imsil registry: unknown product {product!r}; known: {sorted(lay)}")
    row = lay[product]
    if not row["path"]:
        raise KeyError(f"imsil registry: {product!r} has no Imsil counterpart — {row['note']}")
    return os.path.join(PROJECT, row["path"])


def products():
    """{product: (path, note)} for inspection."""
    return {p: (r["path"], r["note"]) for p, r in _layout().items()}
