"""Configuration for FMF GPU template matching on the Imsil swarm (IMSIL PORT of the Haenam config).

IMSIL PORT: every detection setting is unchanged (band from FMF_BAND, windows, SNR gate, 12xMAD per station-configuration
segment, CC floor 0.20, >= 4 stations, STEP, FS). Changed: WF_EXPORTS (the Imsil relocation SAC bundles), DL_CATALOG,
KMA files, default TEMPLATE_CSV / TEMPLATE_SEQUENCES ("imsil"), NEAR_SOURCE_CENTER, EXCLUDE_EPOCHS (Haenam HANB entry
removed; Imsil epochs are decision D12), and the haenam_* output names (-> imsil_*).

Haenam docstring (provenance):

Ported from /home/msseo/works/FMF/buan_fmf (the Buan prototype). Same design, Haenam values:
  * templates = the 195-event JNU-augmented Haenam_joint relocation (catalogs/haenam_templates.csv);
    PICKS (P = SAC `a`, S = SAC `t0`) come from that run's per-event SAC bundles, WAVEFORMS are cut
    from the SAME continuous archive the scan reads (identical processing) -> self-detection ~1.0;
  * scan = the continuous symlink tree DL_detection/continuous (KS 100 Hz + JU 200 Hz -> 100 Hz);
  * multi-year: the scan takes an explicit YEAR (the Haenam story spans 2020 -> 2026).
Nothing outside DL_detection/fmf is written.
"""
import os

# ---- paths ----
FMF_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(FMF_ROOT)                                 # .../DL_detection

# ---- one source of truth for a run's result-affecting settings (2026-10-01) -----------------------------------------------
# logs/<FMF_RUN>/run_settings.json holds the settings the run was scanned with. Any tool started with FMF_RUN set to that run
# inherits them (so figures and checks use exactly the scan's settings), and a conflicting value set in the environment is an
# error, not a silent override. (Figures for nb45/nb46 had been drawn with the default SNR gate 2.0 instead of the run's 1.2.)
def _load_run_settings():
    import json as _json
    run = os.environ.get("FMF_RUN", "")
    if not run:
        return
    p = os.path.join(FMF_ROOT, "logs", run, "run_settings.json")
    if not os.path.exists(p):
        return
    want = _json.load(open(p))["settings"]
    bad = {k: (os.environ[k], v) for k, v in want.items() if k in os.environ and os.environ[k] != v}
    if bad:
        raise SystemExit(f"[config] environment conflicts with {p}: " + ", ".join(f"{k}={a!r} (run: {b!r})" for k, (a, b) in bad.items()))
    for k, v in want.items():
        os.environ.setdefault(k, v)
_load_run_settings()
CONTINUOUS = os.path.join(PROJECT, "continuous")                    # symlink tree (tools/link_continuous.py)
STATIONS_CSV = os.path.join(PROJECT, "station_table", "stations.csv")
# Template catalog: haenam_templates.csv = the 195 canonical events; haenam_templates_dl.csv (built with
# catalogs/build_template_catalog.py --with-dl) adds the high-confidence DL-relocated events (cuspid 300000+).
TEMPLATE_CSV = os.path.join(PROJECT, "catalogs", os.environ.get("FMF_TEMPLATE_CSV", "imsil_templates_ref25.csv"))   # IMSIL PORT
# Per-event SAC bundles carrying the picks (a = P, t0 = S, reference time = origin, b = -30 s).
# Searched in order; KS/KG (100 Hz) live in waveforms_100km, the JNU stations (200 Hz) in waveforms_aug_JNU.
WF_EXPORTS = [os.path.join(PROJECT, "runs", "hypodd_dl", "waveforms_100km")]   # IMSIL PORT: the Imsil DL relocation bundles (PyOcto picks)
DL_CATALOG = os.path.join(PROJECT, "catalogs", "imsil_dl_catalog.csv")    # analysis/imsil_catalog.py output (IMSIL PORT)
KMA_CSV = "/home/msseo/works/72.2025_Imsil_swarm/data/catalogs/kma_entire_catalog_260309.csv"   # KST times (IMSIL PORT)
KMA_2026_CSV = ""                                                          # IMSIL PORT (D8): not used

# ---- template subset ----
# TEMPLATE_SEQUENCES: which `sequence` values of haenam_templates.csv to use ("2020", "quiescence", "2026").
# A template is only buildable if its origin DAY exists in the continuous archive: today that is the 2026
# set (+ the 2022 quiescence event once 2022 data arrive; the 2020 set once 2020 data arrive). Events whose
# day files are missing are dropped with a message, never silently zero-filled.
TEMPLATE_SEQUENCES = tuple(os.environ.get("FMF_SEQUENCES", "imsil").split(","))      # IMSIL PORT: one label for all templates
TEMPLATE_CONFIDENCE = "high_conf"

# ---- optional NEAR-SOURCE round (env-driven; default off) ----
# NEAR_SOURCE_KM=30 keeps only stations within that radius of the source (JNU + HAWA/MIAA/MOPB/HANB/JDO2/
# NPGA): a near-constant geometry where the simple 12xMAD threshold is clean (Buan lesson).
NEAR_SOURCE_RADIUS_KM = (lambda v: float(v) if v and float(v) > 0 else None)(os.environ.get("NEAR_SOURCE_KM", ""))
NEAR_SOURCE_CENTER = (35.5506, 127.2951)                     # IMSIL PORT: Imsil source (lat, lon)
FMF_RUN = os.environ.get("FMF_RUN", "")                       # run name -> output suffix (e.g. full_period)
_SUF = (f"_near{int(NEAR_SOURCE_RADIUS_KM)}" if NEAR_SOURCE_RADIUS_KM else "") + (f"_{FMF_RUN}" if FMF_RUN else "")

# ---- full-period station policy (2026-09-22) ----
# STATION_LIST: explicit station set (network,station,horizontals). horizontals=1 -> Shelly S pass on N/E (JU:
# single verified orientation period); 0 -> S pass on Z (KS/KG borehole horizontals were rotated/reversed between
# epochs, verticals are all normal). Empty/missing file -> the original occurrence rule (MIN_TEMPLATE_OCCUR).
STATION_LIST = os.environ.get("FMF_STATION_LIST", os.path.join(FMF_ROOT, "stations_imsil_5_20H.csv"))   # IMSIL PORT (fmf/build_station_list.py)
SNR_MIN = float(os.environ.get("FMF_SNR_MIN", "2.0"))          # per-channel template gate: window RMS / pre-P noise RMS; 0 = off
NOISE_WIN = (6.0, 1.0)                                          # noise = [P-6 s, P-1 s] on the same channel (S-3 s if no P)
MIN_LIVE_P = int(os.environ.get("FMF_MIN_LIVE_P", "0"))       # at scan time (after the SNR mask): a template needs >= this many P (Z) channels (0: S-only templates allowed) ...
MIN_LIVE_CH = int(os.environ.get("FMF_MIN_LIVE_CH", "4"))      # ... and >= this many channels in total to be scanned that day
# Sensor epochs that must never be mixed: templates inside the epoch drop the station; on scan days inside the epoch,
# templates from outside it get zero weight there (and vice versa).
EXCLUDE_EPOCHS = {}   # IMSIL PORT (D12): Haenam had {("KS", "HANB"): [("2020-05-02", "2022-12-13")]}; Imsil epochs decided from nb06

TEMPLATE_NPZ = f"{FMF_ROOT}/templates_cache/imsil_templates{_SUF}.npz"   # IMSIL PORT: names
SHELLY_NPZ = f"{FMF_ROOT}/templates_cache/imsil_shelly{_SUF}.npz"

# ---- station / channel set ----
NETWORKS = ["KS", "KG", "JU"]          # KG has no continuous data here -> dropped by _cont_prefix
NET_REMAP = {}                         # SAC-pick/station-table net -> continuous net (none needed)
COMPONENTS = ["N", "E", "Z"]           # Z LAST (FMF idiom: vertical=last)
FS = 100.0                             # common sampling rate (Hz); JU 200 Hz is resampled
# Detection band (FMF prescribes none). 2 Hz kills microseisms; 15 Hz keeps the small-event energy well
# inside the 50 Hz Nyquist. Haenam's 20-km-deep sources at 4-100 km are in the same regime as Buan.
BANDPASS = tuple(float(x) for x in os.environ.get("FMF_BAND", "2,15").split(","))   # Hz; env FMF_BAND="5,20" for the 5-20 Hz run (the dt.cc band; band_test.py 2026-09-27)

# ---- template windows (Beauce single-length pass; picks are reused by shelly.py) ----
# Haenam S-P: ~2.5 s at JN09 (4 km epicentral, 20 km deep) to ~9 s at 100 km, so a 4 s window
# (1 s pre-pick) keeps the S window free of P on essentially every channel (Buan rationale).
WIN_LEN = 4.0
WIN_SAMPLES = int(round(WIN_LEN * FS))                      # 400 samples
P_PRE = 1.0                                                  # s before P (vertical window)
S_PRE = 1.0                                                  # s before S (horizontal windows)
PHASE_FOR_COMP = {"N": "S", "E": "S", "Z": "P"}
PRE_FOR_COMP = {"N": S_PRE, "E": S_PRE, "Z": P_PRE}
MIN_TEMPLATE_OCCUR = 8        # keep a station if it has picks in >= this many templates

# ---- scan ----
STEP = int(os.environ.get("FMF_STEP", "1"))                  # CC evaluated every STEP samples (1 = every sample, Buan policy)
SAMPLES_PER_DAY = int(round(86400 * FS))                    # 8_640_000

# ---- parallel data loading (the per-day bottleneck) ----
N_WORKERS = int(os.environ.get("FMF_WORKERS", "24"))        # shared 64-core box: stay polite (per process)

# ---- memory safety (never OOM) ----
# Device usage ~= DEVICE_OVERHEAD*(data + batch*n_corr*4) + templates (calibrated on Buan: 45.8 GB
# measured on 96 sta x 116 tpl). RTX PRO 6000 (98 GB) -> 45 GB ceiling leaves room for other users.
GPU_BUDGET_BYTES = int(float(os.environ.get("FMF_GPU_BUDGET", "45e9")))   # per process; lower when running years in parallel
DEVICE_OVERHEAD = 3.3

# ---- detection ----
MAD_K = 12.0                                                # threshold = MAD_K * MAD(CC)
MAD_SUBSAMPLE = int(os.environ.get("FMF_MAD_SUBSAMPLE", "8"))   # median/MAD estimated on every k-th CC sample (8.6M -> 1.1M): same threshold, 8x faster
# Absolute network-CC floor + station gate (applied in build_catalog.py, exactly as the Buan notebooks
# did): network-CC MAD ~ 1/sqrt(n_stations), so on a large network 12xMAD collapses to noise level;
# the floor makes the threshold density-robust. Catalog = 12*MAD AND cc>=ABS_CC_FLOOR AND
# n_present>=MIN_PRESENT_STATIONS, declustered.
ABS_CC_FLOOR = 0.20
MIN_PRESENT_STATIONS = 4
# Config-segmented MAD (env MAD_SEGMENTED=1): threshold PER station-configuration segment (segmad.py);
# removes the station-onset/offset transition artifact. Boundaries from miniSEED HEADER coverage.
MAD_SEGMENTED = os.environ.get("MAD_SEGMENTED", "") not in ("", "0", "false", "False")
MAD_SEG_MERGE_S = 1800.0
MAD_SEG_MIN_S = 600.0
TEMPLATE_DURATION_S = WIN_LEN                               # declustering scales with this
DETECT_DIR = f"{FMF_ROOT}/catalogs/daily_shelly{_SUF}{'_seg' if MAD_SEGMENTED else ''}"
CATALOG_CSV = f"{FMF_ROOT}/catalogs/fmf_imsil_catalog{_SUF}{'_seg' if MAD_SEGMENTED else ''}.csv"   # IMSIL PORT: name

os.makedirs(DETECT_DIR, exist_ok=True)


def day_start(year, doy):
    from obspy import UTCDateTime
    return UTCDateTime(f"{year}-01-01") + (doy - 1) * 86400
