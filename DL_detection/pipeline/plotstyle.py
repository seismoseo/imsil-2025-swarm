"""Shared plot conventions: Helvetica with per-glyph fallback, sentence-case labels.

Importing this module configures matplotlib once. PyGMT already defaults to
Helvetica, so only matplotlib needs the font registration.
"""
import os

import matplotlib as mpl

HELVETICA_DIR = os.environ.get("HELVETICA_DIR", "/home/msseo/Downloads/Helvetica")


def _use_helvetica():
    """Helvetica primary + DejaVu Sans per-glyph fallback; never break plotting."""
    import matplotlib.font_manager as fm
    try:
        if os.path.isdir(HELVETICA_DIR):
            for fpath in fm.findSystemFonts(HELVETICA_DIR):
                fm.fontManager.addfont(fpath)
        if "Helvetica" in {f.name for f in fm.fontManager.ttflist}:
            mpl.rcParams["font.family"] = ["Helvetica", "DejaVu Sans"]
    except Exception:
        pass
    mpl.rcParams["axes.unicode_minus"] = False


_use_helvetica()

# one consistent color per network across all notebooks
NETWORK_COLORS = {
    "BA": "#d62728",   # temporary geophones, phase-swapped sites
    "JN": "#ff7f0e",   # BG dense geophone array (net code JN)
    "NS": "#1f77b4",   # temporary broadband
    "KG": "#2ca02c",   # permanent KIGAM
    "KS": "#9467bd",   # permanent KMA
}
