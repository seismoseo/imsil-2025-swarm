"""Strict Sheen, Kang & Rhie (2018) local-magnitude scale for South Korea.

Faithful re-implementation of the procedure in
  Sheen, D.-H., Kang, T.-S., & Rhie, J. (2018). "A Local Magnitude Scale for
  South Korea." BSSA 108(5A), 2748-2755. doi:10.1785/0120180112

Procedure (paper "Data and Preliminary Analysis", p. 2749), applied per event:

  1. Remove mean + linear trend.
  2. Band-pass **0.5-10 Hz, six-pole Butterworth** (suppress microseismic noise).
  3. Correct the instrument response.
  4. Convolve with the **revised Wood-Anderson** response (Uhrhammer & Collins
     1990: static magnification 2080, free period 0.8 s, damping 0.7).
  5. Measure the **zero-to-peak** amplitude (mm) of each WA seismogram within the
     **4.0-2.0 km/s group-velocity window** (Sg/Lg), per component.
  6. Keep only readings whose **zero-to-peak SNR (snr_pp = peak / peak-noise) >= 2**.
  7. **Horizontal** amplitude = geometric mean of the two horizontal peaks
     (sqrt(A_N * A_E)); **vertical** amplitude = the Z peak.
  8. Station magnitude, with **R = epicentral distance (km)** (Sheen calibrated on
     epicentral distance — confirmed against the paper):
       horizontal: ML = log10(A_h) + 0.5869 log10(R/100) + 0.001680 (R-100) + 3 + S
       vertical:   ML = log10(A_z) + 0.5107 log10(R/100) + 0.001699 (R-100) + 3 + S
     Station corrections S are set to 0 (we apply the published scale; we do not
     re-invert station terms on this small dense-array catalog).
  9. Event ML = median of the station magnitudes (robust to per-station outliers
     in an automated catalog). The horizontal scale is the KMA-comparable one
     (KMA derives ML from horizontal peaks).

SOTA windowing / SNR (ported from the Ulsan-Fault `ml_pipeline.py`):
  * **Signal (ML) window** : Sheen [dist/4, dist/2] s after origin, timed with the
    **HYPOCENTRAL** distance R = sqrt(epicentral^2 + depth^2) — NOT epicentral. For a
    near-source dense array a station a few km from an 8-km-deep source has epicentral
    << hypocentral, and an epicentral window lands the S box on the P coda. (The
    magnitude *formula* still uses epicentral R, per Sheen; only the window timing is
    hypocentral.)
  * **Noise window** : short LOCAL pre-P window [P - 6 s, P - 1 s] (5 s ending 1 s
    before P). Not the whole pre-event segment — that would sweep up a prior event's
    coda in a dense sequence and wrongly reject aftershocks.
  * **P reference** : prefer the PhaseNet+ pick (SAC.a); when P was MISSED but a clear
    S exists, fall back to the **theoretical first-arriving P** through the SAME
    kim1983 1-D model used for Buan location (`kim1983_taup_model`). So a station with
    a missed P still contributes its S amplitude. The chosen reference is recorded as
    `p_source` in {"pick", "taup"}.

Only KS + KG stations carry instrument responses; BA / JN(BG) / NS temporary
geophone & nodal arrays have none and are skipped automatically (no response in
the inventory -> the trace is dropped before amplitude measurement).
"""
from __future__ import annotations

import glob
import os
import warnings

import numpy as np
import obspy
import pandas as pd

# Revised Wood-Anderson (Uhrhammer & Collins 1990) for DISPLACEMENT input:
# two zeros at the origin, damped-oscillator poles (f0 = 1.25 Hz, h = 0.7),
# static magnification 2080 carried in `sensitivity`.
WOOD_ANDERSON_PAZ = {
    "sensitivity": 2080.0,
    "zeros": [0j, 0j],
    "poles": [-6.283 + 4.7124j, -6.283 - 4.7124j],
    "gain": 1.0,
}

# Sheen 2018 parametric attenuation coefficients (Eqs. 7-8; their Table notation).
COEF = {
    "H": dict(a=0.5869, b=0.001680),   # horizontal (geometric mean of N,E)
    "V": dict(a=0.5107, b=0.001699),   # vertical
}
BANDPASS = dict(freqmin=0.5, freqmax=10.0, corners=6, zerophase=False)  # 6-pole BW
VEL_FAST, VEL_SLOW = 4.0, 2.0          # km/s group-velocity window (Sg/Lg)
SNR_MIN = 2.0                          # zero-to-peak SNR (snr_pp) must be >= 2
NOISE_PRE_PAD = 1.0                    # gap (s) between noise window end and the P pick
NOISE_WIN_S = 5.0                      # pre-P noise window length (s) -> [P-6, P-1]
MIN_STA = 3                            # report ML only with >= 3 PICKED station measurements; else NaN


def ml_horizontal(A_mm, R_km, S=0.0):
    c = COEF["H"]
    return np.log10(A_mm) + c["a"] * np.log10(R_km / 100.0) + c["b"] * (R_km - 100.0) + 3.0 + S


def ml_vertical(A_mm, R_km, S=0.0):
    c = COEF["V"]
    return np.log10(A_mm) + c["a"] * np.log10(R_km / 100.0) + c["b"] * (R_km - 100.0) + 3.0 + S


def load_inventory(master_dir, fetched_dir=None):
    """Master StationXML(s) + NECIS-fetched SEED RESP files -> one Inventory."""
    inv = obspy.read_inventory(os.path.join(master_dir, "*.xml"))
    if fetched_dir and os.path.isdir(fetched_dir):
        ext = os.path.join(fetched_dir, "extracted")
        for f in sorted(glob.glob(os.path.join(ext, "RESP.*"))) + \
                 sorted(glob.glob(os.path.join(fetched_dir, "*.xml"))):
            try:
                inv += obspy.read_inventory(f)
            except Exception as exc:                       # noqa: BLE001
                warnings.warn(f"load {f}: {exc}", RuntimeWarning)
    return inv


# --- theoretical P through the Buan location velocity model (kim1983) ----------
_KIM1983_TAUP = None   # lazily-built TauPyModel cache


def kim1983_taup_model():
    """The kim1983 (Buan location) velocity model as a TauP model, for *theoretical* P.

    PhaseNet+ does not always pick P even when a clear S exists (missed P at near or
    noisy stations), so for those traces the pre-P noise window is anchored on the
    **theoretical first-arriving P** computed through the SAME kim1983 1-D model used
    for HYPOINVERSE / PyOcto location of this catalog — never on a guess.

    kim1983 is 3 flat layers (velocity_model/kim1983.csv):
        0-15 km : Vp 5.98  Vs 3.40
        15-32 km: Vp 6.38  Vs 3.79
        >32 km  : Vp 7.95  Vs 4.58   (Moho at 32 km)
    The crust is splice onto ak135 below the Moho so TauP can build a full-earth model
    (the mantle splice is irrelevant at the local distances ML uses). Built once into
    ``taup_model/kim1983_buan.npz`` next to this module and cached."""
    global _KIM1983_TAUP
    if _KIM1983_TAUP is not None:
        return _KIM1983_TAUP
    from obspy.taup import TauPyModel
    from obspy.taup.taup_create import build_taup_model
    here = os.path.dirname(os.path.abspath(__file__))
    mdir = os.path.join(here, "taup_model")
    npz = os.path.join(mdir, "kim1983_buan.npz")
    if not os.path.exists(npz):
        os.makedirs(mdir, exist_ok=True)
        tvel = os.path.join(mdir, "kim1983_buan.tvel")
        # kim1983 crust (depth Vp Vs density); constant per layer w/ first-order
        # discontinuities. Densities ~ Nafe-Drake (do not affect travel times).
        # 7.95 Moho half-space carried to 35 km, then ak135 mantle below.
        crust = [(0.00, 5.98, 3.40, 2.70), (15.00, 5.98, 3.40, 2.70),
                 (15.00, 6.38, 3.79, 2.80), (32.00, 6.38, 3.79, 2.80),
                 (32.00, 7.95, 4.58, 3.30), (35.00, 7.95, 4.58, 3.30)]
        ak = os.path.join(os.path.dirname(obspy.taup.__file__), "data", "ak135.tvel")
        with open(tvel, "w") as f:
            f.write("kim1983_buan - P\nkim1983_buan - S\n")
            for z, vp, vs, rho in crust:
                f.write(f"{z:8.3f} {vp:7.4f} {vs:7.4f} {rho:7.4f}\n")
            for ln in open(ak):
                p = ln.split()
                if len(p) == 4:
                    try:
                        z = float(p[0])
                    except ValueError:
                        continue
                    if z > 35.0:
                        f.write(ln if ln.endswith("\n") else ln + "\n")
        build_taup_model(tvel, output_folder=mdir)
    _KIM1983_TAUP = TauPyModel(model=npz)
    return _KIM1983_TAUP


def theoretical_p_rel(epi_km, dep_km, b, taup_model):
    """First-arriving theoretical P from the first sample: ``tP_origin - b``.

    Uses epicentral distance (km) + focal depth (km) through `taup_model`. Returns
    NaN if inputs are missing or TauP returns no arrival."""
    if not (np.isfinite(epi_km) and np.isfinite(dep_km)):
        return np.nan
    try:
        arr = taup_model.get_travel_times(source_depth_in_km=max(dep_km, 0.0),
                                          distance_in_degree=epi_km / 111.195,
                                          phase_list=("p", "P", "Pn", "Pg"))
    except Exception:                                      # noqa: BLE001
        return np.nan
    if not arr:
        return np.nan
    return float(min(a.time for a in arr)) - b


def _wa_trace(tr, inventory):
    """demean/detrend -> 0.5-10 Hz 6-pole BW -> remove response (DISP) ->
    convolve revised WA. Returns the WA trace in **mm**, or None if no response."""
    key = f"{tr.stats.network}.{tr.stats.station}..{tr.stats.channel}"
    try:
        inventory.get_response(key, tr.stats.starttime)
    except Exception:                                      # noqa: BLE001
        return None
    w = tr.copy().detrend("demean").detrend("linear")
    w.taper(0.05)
    w.filter("bandpass", **BANDPASS)
    try:
        # bandpass already band-limits the input, so a narrow pre_filt is not
        # needed; water_level stabilises the deconvolution at the response notch.
        w.remove_response(inventory=inventory, output="DISP", water_level=60.0, plot=False)
    except Exception as exc:                               # noqa: BLE001
        warnings.warn(f"remove_response {tr.id}: {exc}", RuntimeWarning)
        return None
    w.simulate(paz_simulate=WOOD_ANDERSON_PAZ)
    w.data = np.asarray(w.data, dtype=float) * 1000.0       # m -> mm
    return w


def _peak_in_window(wa, p_rel, R_time_km, b):
    """Zero-to-peak |amp| (mm) in the 4-2 km/s window, plus pre-P RMS- and peak-noise (mm).

    The SAC headers `o`/`a`/`t0` are relative to the **reference time = origin**;
    the trace itself begins at ``b`` seconds relative to that reference (the
    export pads ~30 s before origin), so a feature at origin-relative time ``t``
    sits at sample ``(t - b) * sr`` from trace start. ``R_time_km`` is the
    **hypocentral** distance: the Sg/Lg energy travels the hypocentral path, so
    the arrival-time window must use it (Sheen 2018 used epicentral because at
    10-600 km epi==hypo; for this near-array, stations a few km from an 8 km-deep
    source differ enough that an epicentral window lands on the P coda). The
    magnitude formula still uses the *epicentral* R, per Sheen.

    NOISE window = [P - NOISE_PRE_PAD - NOISE_WIN_S, P - NOISE_PRE_PAD] (default
    [P-6, P-1] s): a 5 s window ending 1 s before P. Returns
    (peak_mm, noise_rms_mm, noise_peak_mm) or (nan, nan, nan)."""
    sr = float(wa.stats.sampling_rate)
    data = wa.data
    n = len(data)
    off = -b                                   # seconds from trace start to origin
    t0 = off + R_time_km / VEL_FAST            # fast edge (s from trace start)
    t1 = off + R_time_km / VEL_SLOW            # slow edge
    i0, i1 = int(t0 * sr), int(np.ceil(t1 * sr))
    i0 = max(i0, 0); i1 = min(max(i1, i0 + 1), n)
    if i1 - i0 < 2:
        return np.nan, np.nan, np.nan
    peak = float(np.max(np.abs(data[i0:i1])))
    # pre-P noise: 5 s window ending 1 s before the P pick (origin-relative p_rel).
    noise_rms = noise_peak = np.nan
    if p_rel is not None:
        p_start = off + p_rel                          # P sample (s from trace start)
        j1 = int((p_start - NOISE_PRE_PAD) * sr)
        j0 = int((p_start - NOISE_PRE_PAD - NOISE_WIN_S) * sr)
        j0 = max(j0, 0); j1 = min(j1, n)
        if j1 - j0 >= 2:
            seg = data[j0:j1]
            noise_rms = float(np.sqrt(np.mean(seg ** 2)))
            noise_peak = float(np.max(np.abs(seg)))
    return peak, noise_rms, noise_peak


def event_station_amps(event_dir, inventory, depth_km=0.0, require_pick=True,
                       taup_model=None, use_taup=True):
    """Per-station amplitudes for one event directory. Returns a DataFrame with
    one row per station: dist_km (epicentral, from SAC), A_h (geom-mean horiz mm),
    A_z (vertical mm), the station magnitudes, and `p_source` (pick/taup).

    ``depth_km`` (HypoDD relocated focal depth) is used (a) to time the Sg/Lg
    group-velocity window via the hypocentral distance and (b) for the theoretical
    P. The Sheen magnitude formula uses the epicentral distance from the SAC header.

    P reference: the PhaseNet+ pick (SAC `a`) where present; otherwise — when
    ``use_taup`` (default True) — the **theoretical kim1983 P** (the Buan location
    model), so a station with a missed P but a clear S still contributes. With
    ``require_pick=True`` and ``use_taup=False`` a station with no pick is skipped.

    SNR gate: a component peak is kept only if its **zero-to-peak SNR**
    ``snr_pp = peak / peak-noise >= SNR_MIN`` (2)."""
    if use_taup and taup_model is None:
        taup_model = kim1983_taup_model()
    sacs = sorted(glob.glob(os.path.join(event_dir, "*.sac")))
    # group files by (net, sta)
    by_sta: dict = {}
    for f in sacs:
        tr = obspy.read(f)[0]
        st = tr.stats
        by_sta.setdefault((st.network, st.station), []).append(tr)
    rows = []
    for (net, sta), trs in by_sta.items():
        comp_peak = {}                          # 'Z'/'N'/'E' -> peak_mm (snr_pp-passed)
        # epicentral distance (SAC dist) and the station P pick (SAC a), if any.
        R = next((float(t.stats.sac["dist"]) for t in trs
                  if "dist" in (getattr(t.stats, "sac", {}) or {})), None)
        p_rel = next((float(t.stats.sac["a"]) for t in trs
                      if "a" in (getattr(t.stats, "sac", {}) or {})
                      and float(t.stats.sac["a"]) > -1e4), None)
        if R is None:
            continue
        # P reference for the noise window: PhaseNet+ pick, else theoretical kim1983 P.
        b0 = next((float(getattr(t.stats, "sac", {}).get("b", 0.0)) for t in trs), 0.0)
        p_source = None
        if p_rel is not None:
            p_source = "pick"
        elif use_taup and taup_model is not None:
            tp = theoretical_p_rel(R, depth_km, b0, taup_model)
            if np.isfinite(tp):
                p_rel = tp; p_source = "taup"
        # no P reference at all -> can't form a noise window -> skip (unless caller
        # explicitly disabled both pick-requirement and taup, in which case there is
        # nothing to anchor the SNR test and the station is dropped anyway).
        if p_rel is None:
            if require_pick:
                continue
        for tr in trs:
            sac = getattr(tr.stats, "sac", {}) or {}
            if "dist" not in sac:
                continue
            b = float(sac.get("b", 0.0))         # first-sample time rel. to origin
            wa = _wa_trace(tr, inventory)
            if wa is None:
                continue
            R_time = float(np.sqrt(R ** 2 + depth_km ** 2))   # hypocentral, for window timing
            peak, noise_rms, noise_peak = _peak_in_window(wa, p_rel, R_time, b)
            if not np.isfinite(peak) or peak <= 0:
                continue
            # zero-to-peak SNR (snr_pp); fall back to RMS-SNR only if peak-noise missing.
            snr_pp = peak / noise_peak if noise_peak and noise_peak > 0 else np.nan
            if not np.isfinite(snr_pp) or snr_pp < SNR_MIN:
                continue
            comp_peak[tr.stats.channel[-1]] = peak
        A_z = comp_peak.get("Z", np.nan)
        if "N" in comp_peak and "E" in comp_peak:
            A_h = float(np.sqrt(comp_peak["N"] * comp_peak["E"]))
        else:
            A_h = np.nan
        if not (np.isfinite(A_z) or np.isfinite(A_h)):
            continue
        rows.append(dict(network=net, station=sta, dist_km=R, p_source=p_source,
                         A_h_mm=A_h, A_z_mm=A_z,
                         ml_h=ml_horizontal(A_h, R) if np.isfinite(A_h) else np.nan,
                         ml_v=ml_vertical(A_z, R) if np.isfinite(A_z) else np.nan))
    return pd.DataFrame(rows)


def event_ml(per_station_df):
    """Aggregate per-station magnitudes to an event ML (strict Sheen): the median of
    the station magnitudes. Only stations that passed the snr_pp test enter
    ``per_station_df`` -- see ``event_station_amps``.

    **Minimum-station rule** (``MIN_STA`` = 3): the event ML is reported only when it
    rests on >= 3 **genuinely-picked** station measurements (``n_pick``); otherwise it is
    dropped (NaN). Each station contributes one magnitude -- its horizontal (KMA-comparable)
    ML where both N and E pass ``snr_pp``, else its vertical ML -- and the event ML is the
    median over all such station magnitudes (picked AND theoretical-P/``taup`` rescued).

    The gate counts only PICKED stations because a ``taup``-rescued far station (no
    PhaseNet+ pick) is not evidence the calibrated network actually recorded the event:
    instrument responses exist only for the regional KS/KG network, so a micro-event seen
    only by the response-less dense arrays (BA/BG/NS) has its KS/KG "measurements" coming
    purely from far-station noise that clears ``snr_pp`` by chance. Such noise events have
    <= 1 picked calibrated station; real (KMA-confirmed) events have >= 2-3. Gating on
    picked stations therefore drops the noise events while the ``taup`` rescues still refine
    the estimate of genuinely-recorded events. ``ml_h``/``ml_v`` (pure horizontal / vertical
    medians) are returned as diagnostic columns."""
    if not len(per_station_df):
        return dict(ml=np.nan, ml_h=np.nan, ml_v=np.nan, ml_std=np.nan,
                    n_h=0, n_v=0, n_sta=0, n_pick=0)
    df = per_station_df
    # one magnitude per station: horizontal preferred, vertical fallback
    ml_sta = df["ml_h"].where(df["ml_h"].notna(), df["ml_v"])
    valid = ml_sta.notna()
    src = df["p_source"] if "p_source" in df.columns else pd.Series(["pick"] * len(df), index=df.index)
    n_pick = int(((src == "pick") & valid).sum())             # picked stations with a measurement
    if n_pick >= MIN_STA:
        ml = float(ml_sta[valid].median())
        ml_std = float(ml_sta[valid].std()) if valid.sum() > 1 else np.nan
    else:
        ml = np.nan; ml_std = np.nan                          # below the ML-measurability floor
    h = df["ml_h"].dropna(); v = df["ml_v"].dropna()
    return dict(
        ml=ml, ml_h=float(h.median()) if len(h) else np.nan,
        ml_v=float(v.median()) if len(v) else np.nan,
        ml_std=ml_std,
        n_h=int(len(h)), n_v=int(len(v)),
        n_sta=int(df.loc[valid, "station"].nunique()), n_pick=n_pick,
    )
