"""
Shared configuration for the IMSIL DL-detection pipeline (72.2025_Imsil_swarm/DL_detection, on PAGO).

Port of the Haenam DL-detection pipeline (moho 27.Haenam_sequence/DL_detection, read-only copy on PAGO at
~/works/_moho_reference/27.Haenam_sequence/DL_detection) to the 2025-11-02/03 UTC Imsil swarm. core.py and
pick_augmentation.py are byte-identical to the Haenam copy; only the region values below changed, each by an
explicit rule (see docs/decisions.csv and the "IMSIL PORT" comments). Network for Sep 2025 - Jan 2026: KS only
(no KG data exist for 2025; the JU stations near Imsil start 2026-09-01).

Haenam docstring (kept for provenance):
This is the validated Buan pipeline (20.2024_Buan_EQ_DL_detection, itself a port of the Ulsan
KS_KG pipeline) applied to the 2020/2026 Haenam sequences. core.py is byte-identical to the Buan
copy except that the HYPOINVERSE DIS/ZTR lines are read from this file (HYP_DIS / HYP_ZTR).
Everything region-specific lives HERE:
  - continuous layout  continuous/<NET>/<STA>/<CHA>.D/<NET>.<STA>..<CHA>.D.<YYYY>.<DOY>
    (a symlink tree over the MARA mirror / NECIS / JNU-tar / Earthworm archives; tools/link_continuous.py)
  - networks: KS permanent 100 Hz (HG/HH/EL), JU = JNU 200 Hz HH (anti-aliased to 100 Hz),
    KG present 2020-2024 (MARA mirror), absent 2025-2026
  - PhaseNet+ (EQNet) picker, PyOcto association, HYPOINVERSE kim1983 — same as Buan / same as
    the canonical Haenam_joint relocation (korea-cluster-relocation), so DL locations are
    directly comparable with the 195-event relocated catalog.

The picker model ("phasenet_plus", "stead", ...) and velocity model ("kim1983",
"kim2011") are independent dimensions, exactly as in the Ulsan tree.
"""
import os
import sys
import calendar

# allow `import config` whether run as a script or imported from a notebook
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------- paths
ROOT = "/home/msseo/works/72.2025_Imsil_swarm/DL_detection"          # IMSIL PORT: path
MODELS = os.path.join(ROOT, "models")
CONTINUOUS = os.path.join(ROOT, "continuous")
STATIONS_CSV = os.path.join(ROOT, "station_table", "stations.csv")
VELOCITY_CSV = os.path.join(ROOT, "velocity_model", "kim1983.csv")
SHARED_HYPOINV = os.path.join(ROOT, "HypoInv")          # shared .crh crustal models

REGION_NAME = "IMSIL"                                    # file label: IMSIL<year>   (IMSIL PORT: label)

# ------------------------------------------------- per-network channel policy
# Keyed by the directory name under continuous/ (= the network code in this project).
# Value = file-channel 2-char prefixes in PRIORITY order; the first prefix with
# files present for a station-day wins, the rest are dropped (one sensor per
# station — never mix instruments). Channels not listed are excluded entirely.
# KS: HH (broadband) > EL (short-period velocity) > HG (accelerometer) — the Buan/Ulsan
#     order; most KS stations within 50 km of Haenam are HG-only (HAWA, MIAA, NPGA, ...).
# JU: JNU 200 Hz HH only (anti-aliased + resampled to 100 Hz in core.py before picking).
# KS.KWJU dropped 2026-09-17: same coordinates as KS.KWJ2 and 474 shared days (would count one site twice in HYPOINVERSE); KWJ2 kept.
# KG: continuous data 2020-2024 from the MARA SDS mirror (tools/mirror_mara.py); none anywhere for 2025-2026.
# IMSIL PORT: policy unchanged. KWJU/KWJ2 (same coordinates) are handled in station_table/build_stations.py
# (decision D2, default = drop KWJU as Haenam did). In used_stations_100km.csv (event-waveform sensor list, Nov 2025)
# 35 of the 55 Imsil stations are HG-only (picked as acceleration, as in Haenam), 16 HH, 4 EL.
CHANNEL_POLICY = {
    "KS": ["HH", "EL", "HG"],
    "KG": ["HH", "EL", "HG"],
    "JU": ["HH"],
}
# IMSIL PORT (decision D17, user 2026-10-07): sensor bands kept OUT of the data tree for single stations, applied by
# tools/mirror_mara.py (never fetched) and tools/link_continuous.py (never linked), so every stage (picker, SAC export, FMF)
# sees only the remaining band. CHANNEL_POLICY and apply_channel_policy are unchanged.
#   KS.HMPA HH: on MARA only on 2025 DOY 253 and 357, as 8-192 kB fragments (HG files those days 6.8-8.6 MB; HG every day);
#   the policy (HH > EL > HG) would otherwise replace a full HG day by an HH fragment on those two days.
CHANNEL_EXCLUDE = {("KS", "HMPA"): ["HH"]}

# --------------------------------------------- detection (PhaseNet) defaults
SAMPLING_RATE = 100.0
PAD_TIME = 10.0
BANDPASS = dict(freqmin=1.0, freqmax=40.0, corners=4, zerophase=False)
MERGE = dict(method=1, fill_value=0)
P_THRESHOLD = 0.2
S_THRESHOLD = 0.2
# Anti-alias lowpass applied BEFORE downsampling 250/200 -> 100 Hz (obspy
# interpolate / EQNet's linear interpolation do NOT anti-alias on their own).
ANTIALIAS = dict(freq=45.0, corners=8, zerophase=True)
# Shared 64-core server — keep the footprint polite (see Ulsan performance notes).
MAX_CORES = 24
MAX_SEGMENTS = 2000          # above this: log "fragmented, slow merge" but still process
HARD_MAX_SEGMENTS = 300000   # above this: skip (abnormal — avoid an unbounded stall)

# ---------------------------------------------- association (PyOcto) defaults
# Haenam box: source 34.66N 126.40E (20 km deep); the 100-km station set spans
# ~34.2-35.4N / 125.9-127.0E. Box = +/-0.7 deg lat, +/-0.8 deg lon (~+/-75 km) so that
# events anywhere under the network are associated (blasts, other clusters), not only Haenam.
# NOTE PyOcto's travel-time table (core.py create_model) extends to 100 km / 100 km depth.
# IMSIL PORT (rule): the SAME half-widths (+/-0.70 deg lat, +/-0.80 deg lon, rounded to 0.05 deg as Haenam)
# centred on the Imsil source 35.5506N 127.2951E -> lat 34.85-36.25, lon 126.50-128.10. Haenam values were
# lat (33.95, 35.35), lon (125.60, 127.20). All other association parameters are unchanged.
REGION = dict(
    lat=(34.85, 36.25),
    lon=(126.50, 128.10),
    zlim=(0, 30),
    time_before=300,
    n_picks=4,
    n_p_picks=2,
    n_s_picks=2,
    n_p_and_s_picks=1,
)
# --strict set. Haenam is NOT a dense-array case: the near field is 3 JNU stations (4-17 km)
# + HAWA (7 km) + 4 KS at 16-22 km, so the Buan v3 dense-array gate (8 P + 8 S) would silently
# drop real small events. We use the station-adaptive RELAXED-STRICT set validated in the Buan
# PRECURSOR project (21.2024_Buan_precursor, REGION_STRICT_RELAXED): equal minimums 3 P + 3 S,
# both phases at >= 2 stations, tight pick_match_tolerance = 1.0 s to reject chimeras. Quality
# control then happens at HYPOINVERSE in tiers (HAENAM_QC_A / _B below), not as a hard gate.
REGION_STRICT = dict(
    lat=(34.85, 36.25),                                  # IMSIL PORT: same half-widths, Imsil centre
    lon=(126.50, 128.10),
    zlim=(0, 30),
    time_before=300,
    n_picks=6,
    n_p_picks=3,
    n_s_picks=3,
    n_p_and_s_picks=2,
    pick_match_tolerance=1.0,
    min_node_size=2.0,
    min_node_size_location=0.5,
    refinement_iterations=8,
    min_interevent_time=2.0,
    n_threads=16,
)
# The Buan v3 DENSE gate (8 P + 8 S), kept for reference / for a future dense deployment.
REGION_STRICT_DENSE = dict(REGION_STRICT, n_picks=16, n_p_picks=8, n_s_picks=8)

# HYPOINVERSE QC. Filter semantics (sumio.apply_qc): erh<v, erz<v, gap<v, num>v, rms<v
# -> num=7 yields N>=8. Reported in TIERS (analysis/haenam_catalog.py), never as a hard gate:
#   A (high quality)         : gap<180, RMS<0.7 s, ERH<5, ERZ<5 km, N>=8   (Buan precursor A)
#   B (sparse-OK, review)    : gap<300, RMS<1.0 s, ERH<10, ERZ<10 km, N>=6
# HAENAM_QC (= Buan v3 BUAN_QC) is the gate for events promoted to the relocation stage.
IMSIL_QC = dict(erh=5.0, erz=5.0, gap=180.0, num=7, rms=1.0)          # IMSIL PORT: values unchanged
IMSIL_QC_A = dict(erh=5.0, erz=5.0, gap=180.0, num=7, rms=0.7)
IMSIL_QC_B = dict(erh=10.0, erz=10.0, gap=300.0, num=5, rms=1.0)
# Aliases so the Haenam-derived wrappers (run_dl_hypodd, run_status, the freeze record in run_imsil.sh) keep
# working unchanged and the frozen-parameter JSON keeps Haenam's key names (value-by-value comparison).
HAENAM_QC, HAENAM_QC_A, HAENAM_QC_B = IMSIL_QC, IMSIL_QC_A, IMSIL_QC_B
BUAN_QC = HAENAM_QC          # alias so Buan-derived code keeps working

# Haenam source region (the 2020+2026 relocated cluster) used for catalog cross-matching and
# for selecting the events that are folded back into the canonical relocation.
# Haenam values: EPICENTER (34.660, 126.405), BOX (34.55, 34.77, 126.28, 126.53) = +/-0.11 deg lat, +/-0.125 deg lon.
# IMSIL PORT (rule): epicentre = the relocated Imsil source (the ML 1.9 hand-pick HYPOINVERSE solution,
# 1.HypoInv/kim2011/Imsil.sum: 35.5506N 127.2951E 14.40 km); box = the SAME half-widths around it.
IMSIL_EPICENTER = (35.5506, 127.2951)
IMSIL_BOX = (35.44, 35.66, 127.17, 127.42)      # (latmin, latmax, lonmin, lonmax), ~+/-12 km
HAENAM_EPICENTER, HAENAM_BOX = IMSIL_EPICENTER, IMSIL_BOX                  # aliases (see above)
# label embedded in pyocto_<label>_<year>.csv (the PyOcto layered velocity model)
PYOCTO_VELMODEL = "kim1983"

# -------------------------------------------------- HYPOINVERSE crustal model
DEFAULT_VELMODEL = "kim1983"
PHASE_CHANNELS = {"P": "HHZ", "S": "HHN"}
# HYPOINVERSE control lines read by core.HYP_TEMPLATE (the only core.py change vs Buan).
# DIS 4 50 1 3 = full weight to 50 km (the korea-cluster-relocation STD_HYP used for the canonical
# Haenam_joint absolute locations); Buan used 25 km because its arrays sat within 10 km. Haenam's
# source is 20 km deep with most stations at 20-100 km, so the 50-km taper is appropriate.
# ZTR 10 = trial depth 10 km (as in the canonical run; the 20-km Haenam depth converges from it).
HYP_DIS = "4 50 1 3"
HYP_ZTR = "10 F"

# ------------------------------------------------ picker backends (by model)
SEISBENCH_MODELS = {"stead", "original", "instance", "ethz", "scedc", "geofon", "neic"}
EQNET_MODELS = {"phasenet_plus"}

# EQNet (AI4EPS) — external clone required for PhaseNet+ (not vendored here)
EQNET_DIR = "/home/msseo/works/72.2025_Imsil_swarm/_external/EQNet"   # IMSIL PORT: path (copy of moho 14.EQNet/EQNet @ad492ef)
EQNET_WEIGHTS = os.path.join(EQNET_DIR, "docs", "model_phasenet_plus", "model_99.pth")

# PhaseNet+ params. PhaseNet-style pickers expect RAW (demeaned) data + their own
# internal normalization — do NOT bandpass. A gentle highpass is optional.
PNPLUS_MIN_PROB = 0.2   # v2 (full 152-station run); v1 used 0.3 — see models/phasenet_plus_v1/
PNPLUS_HIGHPASS = 0.0
PNPLUS_NT = 1024 * 36


def days_in_year(year):
    return 366 if calendar.isleap(year) else 365


def channel_prefix(fname):
    """2-char channel prefix from a day-file name NET.STA..CHA.D.YYYY.DOY."""
    parts = os.path.basename(fname).split(".")
    return parts[3][:2] if len(parts) == 7 else ""


def apply_channel_policy(dirnet, files):
    """Keep only the highest-priority sensor's files for one station-day."""
    prefs = CHANNEL_POLICY.get(dirnet)
    if not prefs:
        return sorted(files)
    by_prefix = {}
    for f in files:
        by_prefix.setdefault(channel_prefix(f), []).append(f)
    for p in prefs:
        if by_prefix.get(p):
            return sorted(by_prefix[p])
    return []


# ------------------------------------------------------------- path resolvers
def model_root(model):              return os.path.join(MODELS, model)
def detection_year_dir(model, y):   return os.path.join(MODELS, model, "detection_location", str(y))
def picks_dir(model, y):            return os.path.join(detection_year_dir(model, y), "picks")
def pyocto_dir(model):              return os.path.join(MODELS, model, "pyocto")
def pyocto_events(model, y):        return os.path.join(pyocto_dir(model), f"pyocto_{PYOCTO_VELMODEL}_{y}.csv")
def pyocto_assign(model, y):        return os.path.join(pyocto_dir(model), f"pyocto_assignment_{PYOCTO_VELMODEL}_{y}.csv")
def station_table_dir(model):       return os.path.join(MODELS, model, "station_table")
def stations_year_csv(model, y):    return os.path.join(station_table_dir(model), f"stations_{y}.csv")
def hyp_dir(model):                 return os.path.join(MODELS, model, "HypoInv")
def phs_dir(model):                 return os.path.join(hyp_dir(model), "PHS")
def phs_file(model, y):             return os.path.join(phs_dir(model), f"{REGION_NAME}{y}.phs")
def velmodel_dir(model, vm):        return os.path.join(hyp_dir(model), vm)
def phasenet_plus_raw_dir(model, y): return os.path.join(MODELS, model, "phasenet_plus_raw", str(y))


# --------------------------------------------------------------- safety guard
class SteadWriteError(RuntimeError):
    pass


def assert_writable(model, force=False):
    """Kept for parity with the Ulsan pipeline (no read-only reference run here)."""
    return True
