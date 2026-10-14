"""
Core pipeline stage functions for the Buan 2024 seismicity catalog.

Ported from the Ulsan KS_KG pipeline (02.Ulsan_Fault_detection) with three
substantive adaptations:

  1. Network-level continuous layout: stations are addressed as "NET/STA"
     relative directories (continuous/<NET>/<STA>/<CHA>.D/...), discovered by
     discover_stations() and globbed by day_files().
  2. Per-network channel policy (config.CHANNEL_POLICY): drops the 20 Hz KG
     BG?/SL?/BH? channels and enforces one sensor per station-day.
  3. Anti-alias resampling: the 250 Hz (BA/JN) and 200 Hz (NS) data are
     lowpassed (config.ANTIALIAS) before any interpolation down to 100 Hz —
     neither obspy interpolate nor EQNet's linear resampling anti-aliases.
     For PhaseNet+ this is done in _AntialiasSeismicTraceDataset.read_mseed.

Stages:  detection -> association -> augment -> PHS -> HYPOINVERSE
"""
import os
import sys
import glob
import concurrent.futures
import multiprocessing

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ============================================================ detection (1)
def discover_stations(base_dir, year):
    """Station dirs with data for `year`, as 'NET/STA' relative paths."""
    out = []
    for net in sorted(os.listdir(base_dir)):
        netdir = os.path.join(base_dir, net)
        if not os.path.isdir(netdir):
            continue
        for sta in sorted(os.listdir(netdir)):
            p = os.path.join(netdir, sta)
            if os.path.isdir(p) and glob.glob(f"{p}/*/*.{year}.*"):
                out.append(f"{net}/{sta}")
    return out


def day_files(code, jday, base_dir):
    """Component day files for one 'NET/STA' station-day after the channel policy."""
    files = glob.glob(f"{base_dir}/{code}/*/*.{jday}")
    dirnet = code.split("/")[0]
    return config.apply_channel_policy(dirnet, files)


def preprocess_station(code, jday, base_dir):
    """Load + preprocess one station-day. Runs in a worker process.

    Ulsan recipe with an anti-alias lowpass inserted before the 100 Hz
    interpolation (the Buan arrays record at 250/200 Hz natively).
    """
    from obspy import read, Stream
    import config as cfg
    try:
        sel = day_files(code, jday, base_dir)
        if not sel:
            return None
        stream = Stream()
        for f in sel:
            stream += read(f)
        if not stream:
            return None
        nseg = len(stream)
        if nseg > cfg.HARD_MAX_SEGMENTS:
            print(f"  ! skip [{code}] {jday}: {nseg} segments > HARD_MAX_SEGMENTS "
                  f"({cfg.HARD_MAX_SEGMENTS}) — abnormal, skipping to avoid an unbounded stall")
            return None
        if nseg > cfg.MAX_SEGMENTS:
            print(f"  . [{code}] {jday}: {nseg} fragmented records — lossless merge (slow ~{nseg//400}s)")
        for tr in stream:                      # Haenam insertion: see read_mseed (99.99999 Hz files)
            if 0 < abs(tr.stats.sampling_rate - round(tr.stats.sampling_rate)) < 1e-3:
                tr.stats.sampling_rate = float(round(tr.stats.sampling_rate))
        if any(tr.stats.sampling_rate > cfg.SAMPLING_RATE for tr in stream):
            stream.detrend("demean")
            stream.taper(max_percentage=None, max_length=1.0)
            stream.filter("lowpass", **cfg.ANTIALIAS)
        stream.interpolate(sampling_rate=cfg.SAMPLING_RATE)
        stream.merge(**cfg.MERGE)
        t_start = max(tr.stats.starttime for tr in stream)
        t_end = min(tr.stats.endtime for tr in stream)
        stream.trim(starttime=t_start - cfg.PAD_TIME, endtime=t_end + cfg.PAD_TIME,
                    pad=True, fill_value=0)
        stream.detrend("demean")
        stream.taper(max_percentage=None, max_length=cfg.PAD_TIME)
        stream_filt = stream.filter("bandpass", **cfg.BANDPASS).detrend("demean")
        stream_filt.trim(starttime=t_start, endtime=t_end)
        return stream_filt
    except Exception as e:
        print(f"  ! preprocess failed [{code}] {jday}: {e}")
        return None


def canonical_station(trace_id):
    """Normalize a pick trace_id to canonical 'NET.STA' (trace headers are the
    source of truth — e.g. the dense 'BG' array records as JN.BGxx)."""
    parts = str(trace_id).split(".")
    if len(parts) >= 2 and parts[1]:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def detect_day(pn_model, stations, jday, base_dir, executor=None, workers=None):
    """Preprocess all stations in parallel (CPU), then one GPU classify() for the day."""
    from obspy import Stream
    daily = Stream()

    def _gather(ex):
        local = Stream()
        futures = [ex.submit(preprocess_station, c, jday, base_dir) for c in stations]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                local += res
        return local

    if executor is not None:
        daily = _gather(executor)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            daily = _gather(ex)
    if not daily:
        return None
    outputs = pn_model.classify(daily, P_threshold=config.P_THRESHOLD, S_threshold=config.S_THRESHOLD)
    rows = [{"station": canonical_station(p.trace_id), "phase": p.phase,
             "peak_time": str(p.peak_time), "probability": p.peak_value}
            for p in outputs.picks]
    return pd.DataFrame(rows, columns=["station", "phase", "peak_time", "probability"])


def run_detection_year(model, year, days=None, stations=None, skip_existing=True,
                       device=None, workers=None, force=False, min_prob=None, highpass=None):
    """Run detection for a year; write daily picks CSVs.

    Routes to the EQNet PhaseNet+ backend for models in config.EQNET_MODELS,
    otherwise uses the SeisBench PhaseNet backend (`from_pretrained(model)`).
    """
    if model in config.EQNET_MODELS:
        return _run_detection_year_eqnet(
            model, year, days=days, stations=stations, skip_existing=skip_existing,
            device=device, workers=(workers or 0), force=force, min_prob=min_prob, highpass=highpass)

    import torch
    import seisbench.models as sbm
    navail = max(1, min(len(os.sched_getaffinity(0)), config.MAX_CORES))
    torch.set_num_threads(navail)

    config.assert_writable(model, force)
    base_dir = config.CONTINUOUS
    out_dir = config.picks_dir(model, year)
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  ! WARNING: CUDA not available — running PhaseNet inference on CPU (much slower). "
              "Check the GPU/torch install if a GPU is expected.")
    print(f"[detection] model={model} year={year} device={device}")

    if stations is None:
        stations = discover_stations(base_dir, year)
    print(f"[detection] {len(stations)} stations with {year} data")

    if days is None:
        days = range(1, config.days_in_year(year) + 1)

    n_workers = max(1, min(len(stations), (workers or navail)))
    ctx = multiprocessing.get_context("forkserver")
    n_written = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        print(f"[detection] preprocessing pool: {n_workers} forkserver workers (reused across days)")
        pn_model = sbm.PhaseNet.from_pretrained(model)
        pn_model.to(torch.device(device))
        for day in days:
            jday = f"{year}.{day:03d}"
            out = os.path.join(out_dir, f"picks_{jday}.csv")
            if skip_existing and os.path.exists(out):
                continue
            df = detect_day(pn_model, stations, jday, base_dir, executor=ex)
            if df is None or df.empty:
                continue
            df.to_csv(out, index=False)
            n_written += 1
            print(f"  {jday}: {len(df)} picks -> {os.path.relpath(out, config.MODELS)}")
    print(f"[detection] done: {n_written} daily file(s) written into {os.path.relpath(out_dir, config.MODELS)}")
    return out_dir


# ------------------------------------------- detection (1b): EQNet PhaseNet+
def _pnplus_postprocess(meta, output, polarity_scale=1, event_scale=16):
    """Trim model outputs back to the un-padded patch size (mirrors EQNet predict.py)."""
    nt, nx = int(meta["nt"]), int(meta["nx"])
    meta["data"] = meta["data"][:, :, :nt, :nx]
    if "phase" in output:
        output["phase"] = output["phase"][:, :, :nt, :nx]
    if output.get("polarity") is not None:
        output["polarity"] = output["polarity"][:, :, : (nt - 1) // polarity_scale + 1, :nx]
    if output.get("event_center") is not None:
        output["event_center"] = output["event_center"][:, :, : (nt - 1) // event_scale + 1, :nx]
    if output.get("event_time") is not None:
        output["event_time"] = output["event_time"][:, :, : (nt - 1) // event_scale + 1, :nx]
    return meta, output


def _load_pnplus(device):
    """Build PhaseNet+ and load the bundled EQNet weights onto `device`."""
    import torch
    if config.EQNET_DIR not in sys.path:
        sys.path.insert(0, config.EQNET_DIR)
    import eqnet  # noqa: F401
    net = eqnet.models.__dict__["phasenet_plus"].build_model(
        backbone="unet", in_channels=1, out_channels=3)
    ckpt = torch.load(config.EQNET_WEIGHTS, map_location="cpu", weights_only=False)  # trusted local file
    net.load_state_dict(ckpt["model"], strict=True)
    net.to(torch.device(device)).eval()
    return net


def _make_antialias_dataset_cls():
    """Subclass EQNet's SeismicTraceIterableDataset so miniSEED reading lowpasses
    before downsampling (EQNet's read_mseed linear-interpolates 250/200 -> 100 Hz
    with NO anti-alias filter — fine for its 100 Hz training data, aliased here).

    read_mseed is copied from EQNet eqnet/data/seismic_trace.py with the marked
    insertion; the clone itself is never edited.
    """
    import logging
    from collections import defaultdict

    import fsspec
    import obspy
    import torch
    from eqnet.data import SeismicTraceIterableDataset

    class AntialiasSeismicTraceDataset(SeismicTraceIterableDataset):
        def read_mseed(self, fname, response_path=None, response_xml=None,
                       highpass_filter=False, sampling_rate=100):
            try:
                stream = obspy.Stream()
                for tmp in fname.split(","):
                    with fsspec.open(tmp, "rb") as fs:
                        meta = obspy.read(fs, format="MSEED")
                        if response_path is not None:
                            inv = obspy.read_inventory(
                                os.path.join(response_path, meta[0].id[:-1]) + ".xml")
                            meta = meta.remove_sensitivity(inv)
                        stream += meta
                # --- Haenam insertion: NECIS KS day files mix 99.99999237 Hz and 100.0 Hz segments
                # (JDO2 on ~1/3 of days, MJDB, SIJA, ...); obspy refuses to merge them and the whole
                # station-day would be lost. Round near-integer rates first (sub-ppm, << 1 sample/day).
                for _tr in stream:
                    _r = _tr.stats.sampling_rate
                    if 0 < abs(_r - round(_r)) < 1e-3:
                        _tr.stats.sampling_rate = float(round(_r))
                # ----------------------------------------------------------------------------------
                stream = stream.merge(fill_value="latest")
                if (response_path is None) and (response_xml is not None):
                    response = obspy.read_inventory(response_xml)
                    stream = stream.remove_sensitivity(response)
            except Exception as e:
                print(f"Error reading {fname}:\n{e}")
                return None

            tmp_stream = obspy.Stream()
            for trace in stream:
                if len(trace.data) < 10:
                    continue
                if trace.stats.sampling_rate != sampling_rate:
                    try:
                        # --- Buan insertion: anti-alias before downsampling ---
                        if trace.stats.sampling_rate > sampling_rate:
                            trace = trace.detrend("demean")
                            trace = trace.taper(max_percentage=None, max_length=1.0)
                            trace = trace.filter("lowpass", **config.ANTIALIAS)
                        # ------------------------------------------------------
                        trace = trace.interpolate(sampling_rate, method="linear")
                    except Exception as e:
                        print(f"Error resampling {trace.id}:\n{e}")
                trace = trace.detrend("demean")
                if highpass_filter > 0.0:
                    trace = trace.filter("highpass", freq=highpass_filter)
                tmp_stream.append(trace)

            if len(tmp_stream) == 0:
                return None
            stream = tmp_stream

            begin_time = min([st.stats.starttime for st in stream])
            end_time = max([st.stats.endtime for st in stream])
            stream = stream.trim(begin_time, end_time, pad=True, fill_value=0)

            comp = ["3", "2", "1", "E", "N", "Z"]
            comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

            station_ids = defaultdict(list)
            for tr in stream:
                station_ids[tr.id[:-1]].append(tr.id[-1])
                if tr.id[-1] not in comp:
                    print(f"Unknown component {tr.id[-1]}")

            station_keys = sorted(list(station_ids.keys()))
            nx = len(station_ids)
            nt = len(stream[0].data)
            data = np.zeros([3, nt, nx], dtype=np.float32)
            for i, sta in enumerate(station_keys):
                for c in station_ids[sta]:
                    j = comp2idx[c]
                    if len(stream.select(id=sta + c)) == 0:
                        print(f"Empty trace: {sta+c} {begin_time}")
                        continue
                    trace = stream.select(id=sta + c)[0]
                    if sta[-1] == "N":
                        trace = trace.integrate().filter("highpass", freq=1.0)
                    tmp = trace.data.astype("float32")
                    data[j, : len(tmp), i] = tmp[:nt]

            return {
                "waveform": torch.from_numpy(data),
                "station_id": station_keys,
                "begin_time": begin_time.datetime.isoformat(timespec="milliseconds"),
                "dt_s": 1 / sampling_rate,
            }

    return AntialiasSeismicTraceDataset


def _pnplus_infer(net, meta, min_prob):
    """One PhaseNet+ forward pass for a patch (see Ulsan core for details)."""
    import torch
    from eqnet.utils import detect_peaks, extract_picks, extract_events
    output = net(meta)
    meta, output = _pnplus_postprocess(meta, output)
    dt = meta["dt_s"]
    phase = torch.softmax(output["phase"], dim=1)
    polarity = (torch.softmax(output["polarity"], dim=1)
                if output.get("polarity") is not None else None)
    ts, ti = detect_peaks(phase, vmin=min_prob, kernel=128, dt=dt.min().item())
    picks = extract_picks(
        ti, ts, file_name=meta["file_name"], station_id=meta["station_id"],
        begin_time=meta.get("begin_time"), begin_time_index=meta.get("begin_time_index"),
        dt=dt, vmin=min_prob, phases=["P", "S"], polarity_score=polarity, waveform=meta["data"])
    event_prob = event_time = None
    events = []
    if output.get("event_center") is not None:
        event_prob = torch.sigmoid(output["event_center"])
        event_time = output["event_time"]
        es, ei = detect_peaks(event_prob, vmin=min_prob, kernel=16, dt=dt.min().item() * 16.0)
        events = extract_events(
            ei, es, file_name=meta["file_name"], station_id=meta["station_id"],
            begin_time=meta.get("begin_time"), begin_time_index=meta.get("begin_time_index"),
            dt=dt, vmin=min_prob, event_time=event_time, waveform=meta["data"])
    return dict(phase=phase, polarity=polarity, event_prob=event_prob, event_time=event_time,
                picks=picks, events=events, meta=meta)


def _run_detection_year_eqnet(model, year, days=None, stations=None, skip_existing=True,
                              device=None, workers=0, force=False, min_prob=None, highpass=None):
    """EQNet PhaseNet+ detection backend (in-process; no edits to the EQNet clone).

    Writes canonical per-day picks (station="NET.STA", phase, peak_time, probability)
    plus the raw PhaseNet+ picks (polarity/amplitude) and single-station event
    detections under phasenet_plus_raw/.
    """
    import tempfile
    import torch
    import torch.utils.data
    torch.set_num_threads(max(1, min(len(os.sched_getaffinity(0)), config.MAX_CORES)))

    config.assert_writable(model, force)
    if config.EQNET_DIR not in sys.path:
        sys.path.insert(0, config.EQNET_DIR)
    import eqnet  # noqa: F401

    DatasetCls = _make_antialias_dataset_cls()

    min_prob = config.PNPLUS_MIN_PROB if min_prob is None else min_prob
    highpass = config.PNPLUS_HIGHPASS if highpass is None else highpass
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("  ! WARNING: CUDA not available — PhaseNet+ inference on CPU (much slower).")
    base_dir = config.CONTINUOUS
    out_dir = config.picks_dir(model, year)
    raw_dir = config.phasenet_plus_raw_dir(model, year)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"[detection/phasenet_plus] year={year} device={device} min_prob={min_prob} highpass={highpass}")
    net = _load_pnplus(device)

    if stations is None:
        stations = discover_stations(base_dir, year)
    print(f"[detection/phasenet_plus] {len(stations)} stations with {year} data")
    if days is None:
        days = range(1, config.days_in_year(year) + 1)

    n_written = 0
    for day in days:
        jday = f"{year}.{day:03d}"
        out = os.path.join(out_dir, f"picks_{jday}.csv")
        if skip_existing and os.path.exists(out):
            continue
        # one data_list line per station = comma-joined component files for this day
        lines = []
        for code in stations:
            comps = day_files(code, jday, base_dir)
            if comps:
                lines.append(",".join(comps))
        if not lines:
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            tf.write("\n".join(lines))
            list_path = tf.name
        try:
            dataset = DatasetCls(
                data_path="", data_list=list_path, format="mseed", dataset="seismic_trace",
                training=False, sampling_rate=config.SAMPLING_RATE, highpass_filter=highpass,
                cut_patch=False, nt=config.PNPLUS_NT,
            )
            dataset.cut_patch = True
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=workers, collate_fn=None, drop_last=False)
            picks_all, events_all = [], []
            with torch.inference_mode():
                for meta in loader:
                    r = _pnplus_infer(net, meta, min_prob)
                    for p in r["picks"]:
                        picks_all.extend(p)
                    for e in r["events"]:
                        events_all.extend(e)
        finally:
            os.unlink(list_path)

        if events_all:
            pd.DataFrame(events_all).to_csv(os.path.join(raw_dir, f"events_{jday}.csv"), index=False)
        if picks_all:
            raw = pd.DataFrame(picks_all)
            raw.to_csv(os.path.join(raw_dir, f"picks_{jday}.csv"), index=False)   # keep polarity/amplitude
            pd.DataFrame({
                "station": raw["station_id"].map(canonical_station),
                "phase": raw["phase_type"],
                "peak_time": raw["phase_time"],
                "probability": raw["phase_score"],
            }).to_csv(out, index=False)
            n_written += 1
            print(f"  {jday}: {len(raw)} picks ({len(events_all)} events) -> {os.path.relpath(out, config.MODELS)}")
    print(f"[detection/phasenet_plus] done: {n_written} daily file(s) written into "
          f"{os.path.relpath(out_dir, config.MODELS)}")
    return out_dir


def annotate_phasenet_plus(year, day, station, t0=0.0, t1=600.0, device=None, min_prob=None):
    """Run PhaseNet+ on one station-day ('NET/STA') and return per-sample probability
    traces for inspection — the PhaseNet+ analogue of SeisBench `model.annotate`."""
    import tempfile
    import torch
    import torch.utils.data
    if config.EQNET_DIR not in sys.path:
        sys.path.insert(0, config.EQNET_DIR)
    import eqnet  # noqa: F401

    DatasetCls = _make_antialias_dataset_cls()

    min_prob = config.PNPLUS_MIN_PROB if min_prob is None else min_prob
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("  ! WARNING: CUDA not available — PhaseNet+ annotate on CPU.")
    sr = config.SAMPLING_RATE
    jday = f"{year}.{day:03d}"
    comps = day_files(station, jday, config.CONTINUOUS)
    if not comps:
        raise FileNotFoundError(f"no waveforms for {station} {jday} under {config.CONTINUOUS}")
    i0, i1 = int(t0 * sr), int(t1 * sr)
    n = i1 - i0
    EV_SCALE = 16
    e0, e1 = i0 // EV_SCALE, i1 // EV_SCALE

    P = np.zeros(n, np.float32); S = np.zeros(n, np.float32); N = np.zeros(n, np.float32)
    POL = np.full(n, np.nan, np.float32)
    EV = np.zeros(max(0, e1 - e0), np.float32)
    picks, events = [], []

    net = _load_pnplus(device)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(",".join(comps))
        list_path = tf.name
    try:
        dataset = DatasetCls(
            data_path="", data_list=list_path, format="mseed", dataset="seismic_trace",
            training=False, sampling_rate=sr, highpass_filter=config.PNPLUS_HIGHPASS,
            cut_patch=False, nt=config.PNPLUS_NT)
        dataset.cut_patch = True
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
        with torch.inference_mode():
            for meta in loader:
                bidx = int(meta["begin_time_index"][0]) if meta.get("begin_time_index") is not None else 0
                npatch = int(meta["nt"])
                if bidx + npatch <= i0 or bidx >= i1:       # patch outside the window
                    continue
                r = _pnplus_infer(net, meta, min_prob)
                picks += [p for sub in r["picks"] for p in sub]
                events += [e for sub in r["events"] for e in sub]
                ph = r["phase"][0].cpu().numpy()             # [3, L, nx=1]
                L = ph.shape[1]
                lo, hi = max(i0, bidx), min(i1, bidx + L)
                if hi <= lo:
                    continue
                ws, we, ps, pe = lo - i0, hi - i0, lo - bidx, hi - bidx
                N[ws:we], P[ws:we], S[ws:we] = ph[0, ps:pe, 0], ph[1, ps:pe, 0], ph[2, ps:pe, 0]
                if r["polarity"] is not None:
                    pol = r["polarity"][0].cpu().numpy()
                    POL[ws:we] = pol[1, ps:pe, 0] - pol[2, ps:pe, 0]
                if r["event_prob"] is not None:
                    ep = r["event_prob"][0].cpu().numpy()
                    eb = bidx // EV_SCALE
                    elo, ehi = max(e0, eb), min(e1, eb + ep.shape[1])
                    if ehi > elo:
                        EV[elo - e0:ehi - e0] = ep[0, elo - eb:ehi - eb, 0]
    finally:
        os.unlink(list_path)

    def _in_win(d):
        idx = d.get("phase_index", d.get("event_index"))
        return idx is not None and i0 <= int(idx) < i1
    picks = [p for p in picks if _in_win(p)]
    events = [e for e in events if e.get("event_index") is None or i0 <= int(e["event_index"]) < i1]

    return dict(
        t=np.arange(n) / sr, prob_P=P, prob_S=S, prob_noise=N, polarity=POL,
        t_event=(np.arange(len(EV)) * EV_SCALE) / sr, event_prob=EV,
        picks=picks, events=events,
        meta=dict(sampling_rate=sr, t0=t0, t1=t1, station=station, jday=jday, comps=comps))


# ============================================================ association (2)
def load_picks(model, year):
    files = sorted(glob.glob(os.path.join(config.picks_dir(model, year), f"picks_{year}.*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No picks for model={model} year={year} in {config.picks_dir(model, year)} "
            f"(run detection first).")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def build_station_table(picks_df):
    """Station table for used stations, keyed by (Network, Code) from the unified
    stations.csv (the Ulsan code-only station_update.dat lookup would collide on
    multi-network codes)."""
    parts = picks_df["station"].str.split(".", expand=True)
    used = set(zip(parts[0], parts[1]))
    allsta = pd.read_csv(config.STATIONS_CSV)
    mask = allsta.apply(lambda r: (r["Network"], r["Code"]) in used, axis=1)
    st = allsta[mask][["Network", "Code", "Latitude", "Longitude", "Elevation"]] \
        .drop_duplicates(subset=["Network", "Code"]).reset_index(drop=True)
    missing = used - set(zip(st["Network"], st["Code"]))
    if missing:
        print(f"  ! {len(missing)} picked station(s) missing from stations.csv; "
              f"their picks will be dropped: {sorted(missing)}")
    return st


def _station_pyocto(st):
    s = st.copy()
    s["id"] = s["Network"] + "." + s["Code"] + "."
    s = s.rename(columns={"Longitude": "longitude", "Latitude": "latitude", "Elevation": "elevation"})
    return s[["id", "longitude", "latitude", "elevation"]].copy()


def run_association_year(model, year, force=False, strict=False):
    """Associate picks into events with PyOcto; write events + assignments + station table."""
    import datetime as dt
    import tempfile
    import pyocto

    config.assert_writable(model, force)
    picks_df = load_picks(model, year).copy()
    parts = picks_df["station"].str.split(".", expand=True)
    picks_df["net"], picks_df["code"] = parts[0], parts[1]

    st = build_station_table(picks_df)
    station_pyocto = _station_pyocto(st)

    # side-output: stations_<year>.csv into the model tree
    os.makedirs(config.station_table_dir(model), exist_ok=True)
    fs = station_pyocto.copy()
    fs[["Network", "Code"]] = fs["id"].str.split(".", expand=True)[[0, 1]]
    fs[["Network", "Code", "latitude", "longitude", "elevation"]] \
        .rename(columns=str.capitalize) \
        .to_csv(config.stations_year_csv(model, year), index=False)

    pk = picks_df.rename(columns={"peak_time": "time"}).copy()
    pk["time"] = pd.to_datetime(pk["time"]).dt.tz_localize(None)
    pk["station"] = pk["net"] + "." + pk["code"] + "."
    picks_pyocto = pk[["station", "phase", "time"]].copy()

    layers = pd.read_csv(config.VELOCITY_CSV)
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "vel_model")
        pyocto.VelocityModel1D.create_model(layers, 1.0, 100, 100, mp)
        velocity_model = pyocto.VelocityModel1D(mp, tolerance=1.0)
        region = config.REGION_STRICT if strict else config.REGION
        if strict:
            print(f"[association] STRICT mode — pick_match_tolerance="
                  f"{region.get('pick_match_tolerance', 1.5)} s, "
                  f"n_picks={region['n_picks']}, n_p_and_s_picks={region['n_p_and_s_picks']}")
        associator = pyocto.OctoAssociator.from_area(velocity_model=velocity_model, **region)
        associator.transform_stations(station_pyocto)
        picks_pyocto["time"] = picks_pyocto["time"].apply(lambda x: x.timestamp())
        events, assignments = associator.associate(picks_pyocto, station_pyocto)
        associator.transform_events(events)

    if len(events):
        events["time"] = events["time"].apply(dt.datetime.fromtimestamp, tz=dt.timezone.utc)
    os.makedirs(config.pyocto_dir(model), exist_ok=True)
    events.to_csv(config.pyocto_events(model, year), index=False)
    assignments.to_csv(config.pyocto_assign(model, year), index=False)
    n_pk = int(events["picks"].sum()) if len(events) else 0
    print(f"[association] {model} {year}: events={len(events)} picks={n_pk} "
          f"-> {os.path.relpath(config.pyocto_events(model, year), config.MODELS)}")
    return events, assignments


# ============================================================ pick augmentation (2.5)
def run_augment_year(model, year, force=False, *,
                     radius_km=100.0, tolerance_s=1.0,
                     min_pick_probability=0.3, tie_threshold_s=0.2):
    """Post-PyOcto pick augmentation — recover orphan picks missed by PyOcto's
    streaming associator (see pick_augmentation.py docstring). Idempotent."""
    config.assert_writable(model, force)
    import pick_augmentation as pa
    events_csv = config.pyocto_events(model, year)
    assign_csv = config.pyocto_assign(model, year)
    if not (os.path.exists(events_csv) and os.path.exists(assign_csv)):
        raise FileNotFoundError(
            f"augment requires PyOcto outputs at {events_csv} and {assign_csv} - "
            "run the association stage first")
    summary = pa.augment_year(
        pyocto_events_csv=events_csv,
        pyocto_assignment_csv=assign_csv,
        stations_csv=config.stations_year_csv(model, year),
        picks_dir=config.picks_dir(model, year),
        year=year,
        radius_km=radius_km,
        tolerance_s=tolerance_s,
        min_pick_probability=min_pick_probability,
        tie_threshold_s=tie_threshold_s,
    )
    print(f"[augment] {model} {year}: +{summary['n_accepted']} picks across "
          f"{summary['n_events']} events (rejected {summary['n_dropped_ambiguous']} as ambiguous)"
          f" -> {os.path.relpath(summary['out_path'], config.MODELS)}")
    return summary


# ============================================================ PHS file (3)
def _deg2min(angle):
    return int(angle), int(100 * 60 * (angle - int(angle)))


def write_phs(model, year, force=False):
    """Write the HYPO71-format .phs file from PyOcto events+assignments.
    Fixed-width layout ported verbatim from the Ulsan pipeline."""
    from obspy import UTCDateTime as utc
    from datetime import datetime, timezone

    config.assert_writable(model, force)
    ev = pd.read_csv(config.pyocto_events(model, year))
    pk = pd.read_csv(config.pyocto_assign(model, year))
    os.makedirs(config.phs_dir(model), exist_ok=True)
    out = config.phs_file(model, year)

    with open(out, "w") as f:
        idn = 0
        for i in range(len(ev)):
            eid = ev["idx"][i]
            ot = utc(str(ev["time"][i]).split("+")[0])
            cat = f"{ot.year}{ot.month:02d}{ot.day:02d}{ot.hour:02d}{ot.minute:02d}{ot.second:02d}"
            ms = str(int(ot.microsecond / 1000)).zfill(2)
            la_d, la_m = _deg2min(ev["latitude"][i])
            lo_d, lo_m = _deg2min(ev["longitude"][i])
            f.write(f"{cat}{ms}{la_d}N{str(la_m).zfill(4)}{lo_d}E{str(lo_m).zfill(4)}\n")

            ept = pk[pk["event_idx"] == eid].reset_index(drop=True)
            for j in range(len(ept)):
                pt = utc(datetime.fromtimestamp(ept["time"][j], tz=timezone.utc))
                net = ept["station"][j].split(".")[0]
                sta = ept["station"][j].split(".")[1]
                phase = ept["phase"][j]
                if phase == "P":
                    f.write(sta.ljust(5)); f.write(net.ljust(4))
                    f.write(config.PHASE_CHANNELS["P"][:3].ljust(4))
                    f.write("IP".ljust(3)); f.write("0")
                    f.write(str(pt.year)); f.write(str(pt.month).zfill(2)); f.write(str(pt.day).zfill(2))
                    f.write(str(pt.hour).zfill(2)); f.write(str(pt.minute).zfill(2).ljust(3))
                    f.write(str(pt.second).zfill(2)); f.write(str(pt.microsecond).zfill(6)[:2])
                    f.write("\n")
                elif phase == "S":
                    f.write(sta.ljust(5)); f.write(net.ljust(4))
                    f.write(config.PHASE_CHANNELS["S"].ljust(4)); f.write("    ")
                    f.write(str(pt.year)); f.write(str(pt.month).zfill(2)); f.write(str(pt.day).zfill(2))
                    f.write(str(pt.hour).zfill(2)); f.write(str(pt.minute).zfill(2).ljust(15))
                    f.write(str(pt.second).zfill(2)); f.write(str(pt.microsecond).zfill(6)[:2])
                    f.write("ES".ljust(3)); f.write("1"); f.write("\n")

            f.write(" " * 66 + "20" + f"{idn}".zfill(4) + "\n")
            idn += 1

    print(f"[phs] {model} {year}: {len(ev)} events -> {os.path.relpath(out, config.MODELS)}")
    return out


# ============================================================ HYPOINVERSE (4)
HYP_TEMPLATE = """
REP T T
CON 50
MIN 4
ZTR __ZTR__
DIS __DIS__
RMS 4 .12 2 4

* OUTPUT FORMAT
ERF T
TOP F
LST 2 0 1
KPR 3
H71 4 1 3

* STATION DATA
STA 'STA/__REGION___hyp.sta'

* CRUSTAL MODEL
CRH 1 '__MODEL__/__MODEL___p.crh'
CRH 2 '__MODEL__/__MODEL___s.crh'
SAL 1 2

* PHASE FILE
PHS 'PHS/__REGION__.phs'
FIL

PRT '__MODEL__/__REGION__.prt'
SUM '__MODEL__/__REGION__.sum'
ARC '__MODEL__/__REGION__.arc'

LOC

STO
"""


def write_hyp_sta(model, year):
    """Generate the HYPOINVERSE station file STA/<REGION>_hyp.sta from the
    stations_<year>.csv side-output of the association stage.

    Fixed-width hyp1.40 format (matches the Ulsan UF<year>_hyp.sta files):
      cols 0-4 station, 6-7 net, 10-12 chan, 15-16 lat deg, 18-24 lat min,
      25 'N', 26-28 lon deg, 30-36 lon min, 37 'E', 38-41 elevation (m).
    """
    src = config.stations_year_csv(model, year)
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} missing - run the association stage first")
    st = pd.read_csv(src)
    sta_dir = os.path.join(config.hyp_dir(model), "STA")
    os.makedirs(sta_dir, exist_ok=True)
    out = os.path.join(sta_dir, f"{config.REGION_NAME}{year}_hyp.sta")
    with open(out, "w") as f:
        for _, r in st.iterrows():
            latd = int(r["Latitude"]); latm = (r["Latitude"] - latd) * 60.0
            lond = int(r["Longitude"]); lonm = (r["Longitude"] - lond) * 60.0
            f.write(f"{r['Code']:<5} {r['Network']:<2}  HHZ  "
                    f"{latd:2d} {latm:7.4f}N{lond:3d} {lonm:7.4f}E{r['Elevation']:4.0f}\n")
    print(f"[sta] {len(st)} stations -> {os.path.relpath(out, config.MODELS)}")
    return out


def ensure_crh(model, velmodel):
    """Copy the shared crustal-model files into models/<model>/HypoInv/<velmodel>/
    (hyp1.40 reads them at a RELATIVE path; a missing .crh silently falls back to
    its built-in default model — see the Ulsan pipeline notes)."""
    import shutil
    vmdir = config.velmodel_dir(model, velmodel)
    os.makedirs(vmdir, exist_ok=True)
    shared = os.path.join(config.SHARED_HYPOINV, velmodel)
    for ph in ("p", "s"):
        name = f"{velmodel}_{ph}.crh"
        dst = os.path.join(vmdir, name)
        src = os.path.join(shared, name)
        if not os.path.exists(dst):
            if not os.path.exists(src):
                raise FileNotFoundError(f"shared crustal model missing: {src}")
            shutil.copy2(src, dst)
            print(f"[crh] copied {os.path.relpath(dst, config.MODELS)} <- {src}")
    return vmdir


def run_hypoinverse_year(model, year, velmodel=None, force=False):
    """Run hyp1.40 for one year/velocity-model; outputs land in models/<model>/HypoInv/<velmodel>/."""
    import shutil
    import subprocess

    config.assert_writable(model, force)
    velmodel = velmodel or config.DEFAULT_VELMODEL
    hd = config.hyp_dir(model)
    region = f"{config.REGION_NAME}{year}"
    write_hyp_sta(model, year)

    phs = config.phs_file(model, year)
    if not os.path.exists(phs):
        raise FileNotFoundError(f"PHS missing: {phs} (run make_phs first)")
    if shutil.which("hyp1.40") is None:
        raise RuntimeError("hyp1.40 not found on PATH")
    os.makedirs(config.velmodel_dir(model, velmodel), exist_ok=True)
    ensure_crh(model, velmodel)

    control = (HYP_TEMPLATE.replace("__REGION__", region).replace("__MODEL__", velmodel)
               .replace("__DIS__", getattr(config, "HYP_DIS", "4 25 1 3"))
               .replace("__ZTR__", getattr(config, "HYP_ZTR", "10 F")))
    print(f"[hypoinverse] {model} {year} velmodel={velmodel} (cwd={os.path.relpath(hd, config.MODELS)})")
    proc = subprocess.run(["hyp1.40"], input=control, text=True, cwd=hd, capture_output=True)

    sumf = os.path.join(config.velmodel_dir(model, velmodel), f"{region}.sum")
    if not os.path.exists(sumf):
        print("---- hyp1.40 stdout (tail) ----\n", proc.stdout[-2000:])
        print("---- hyp1.40 stderr (tail) ----\n", proc.stderr[-2000:])
        raise RuntimeError(f"HYPOINVERSE produced no summary file: {sumf}")
    n = sum(1 for _ in open(sumf))
    try:
        n_ev = max(0, sum(1 for _ in open(config.pyocto_events(model, year))) - 1)
    except OSError:
        n_ev = None
    if n_ev and n < max(5, 0.02 * n_ev):
        print(f"[hypoinverse] !! WARNING: only {n} located of ~{n_ev} associated events for "
              f"{model} {year} — likely a station-file/PHS mismatch (check STA/{region}_hyp.sta).")
    try:
        import pandas as _pd
        _rms = _pd.to_numeric(_pd.read_csv(sumf, sep=",").iloc[:, 11], errors="coerce").median()
        if _pd.notna(_rms) and _rms > 1.0:
            print(f"[hypoinverse] !! WARNING: median RMS {_rms:.2f} s for {model} {year} is far too high "
                  f"— check the crustal model ({velmodel}/{velmodel}_*.crh) and station/phase units.")
    except Exception:                          # noqa: BLE001 — never let the sanity check break the run
        pass
    print(f"[hypoinverse] {n} located events -> {os.path.relpath(sumf, config.MODELS)}")
    return sumf
