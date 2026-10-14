#!/usr/bin/env python
# Builder of notebooks/13_long_fmf_candidates.ipynb (Imsil): waveforms of the candidate detections of the long FMF scan
# (2018-01-01 .. 2026-10-03; decisions D25, D26) outside the 2025 swarm, so that each can be judged by eye.
# Figures are Haenam's, as in notebooks 07 and 08: similarity.plot_detection_set, similarity.plot_full_trace_comparison and
# station_gather.station_gather (fmf_aligned/, unchanged). The per-channel window check is notebook 08 (a), with a same-day noise control.
# The kept day files are linked into continuous/ by tools/link_keep_long.py.
#   python notebooks/build_13_long_fmf_candidates.py
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "13_long_fmf_candidates.ipynb")
cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def co(s):
    s = s.strip("\n")
    compile(s, f"<cell {len(cells)}>", "exec")
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))


md(r"""
# 13 Candidates of the long FMF scan, 2018–2026: waveforms for judging by eye

The long scan (decisions D25 and D26) ran the 28 swarm templates over every day from 2018-01-01 to 2026-10-03. It used the aligned code at 5–20 Hz, 10 × MAD, at least 4 stations and no CC floor, with S on Z at the 9 re-instrumented stations. Its catalog has 6,369 detections:
- 212 inside the 2025 swarm window;
- 6,135 on 6–8 March 2018, shown to be noise triggers by a stacking test: the median stack of the strongest 400 has no arrivals at any of 13 stations (decision log D25);
- **22 other detections**, the subject of this notebook.

**Groups.**
- **A. Multi-template (3).** On 2019-04-15, several templates fire at the same origin.
- **B. Template 400005 alone (12).** September 2023 – March 2025. This template has the fewest stations (7) of the 28.
- **C. Other single-template hits (7).**
- **R. Reference.** The strongest detection of the March 2018 noise burst, to show what a noise trigger looks like in the same figures.

**Figures (the styles of notebooks 07 and 08).**
1. **Timeline** of all detections outside the swarm.
2. **Window evidence per candidate** (notebook 08, check a). For each live template channel: the CC between the template window and the data at the detection, and the window's signal-to-noise ratio against the 5 s before its arrival. The control is the same template at 40 random times on the same day.
3. **Detection sets** (`similarity.plot_detection_set`). Every live channel of the template is a row, ordered by distance: the record in gray, the template's P (red) and S (blue) windows in place, and their correlation. The right-hand panels show the detector's view, the template's network CC over the whole day and ±12 s around the detection.
4. **Full waveforms** for group A (`similarity.plot_full_trace_comparison`): template in black, detection in red.
5. **Repeater gathers** (`station_gather.station_gather`). Every detection of one template at one station, in time order, aligned on the template's pick, with the template on top. The long-period candidates come first, then the template's 2025 swarm detections for comparison.

**What a genuine Imsil event should show.**
- Arrivals at the template's P and S times at several stations at once, above the noise.
- Correlation on more than one or two channels.
- A waveform resembling the swarm detections in the gathers.

A noise trigger instead shows one or two channels carrying the correlation, or no visible arrival at the predicted times.

The data are the kept day files (`../continuous_mara/_keep_long`), linked into `continuous/` by `tools/link_keep_long.py`, and loaded exactly as the scan loaded them.
""")
co(r"""
import os, sys, glob, functools, warnings, contextlib, io
import numpy as np, pandas as pd
warnings.filterwarnings('ignore', message='Discarding nonzero nanoseconds')
import multiprocessing as mp, concurrent.futures as cf
DL = '/home/msseo/works/72.2025_Imsil_swarm/DL_detection'; FA = f'{DL}/fmf_aligned'
os.environ.update(FMF_RUN='imsil_5_20H_aligned_mad10_long', FMF_MAD_K='10', FMF_CC_FLOOR='0', FMF_BAND='5,20', FMF_STATION_LIST=f'{FA}/stations_imsil_5_20H_long.csv',
                  FMF_SEQUENCES='imsil', FMF_TEMPLATE_CSV='imsil_templates_ref25.csv', MAD_SEGMENTED='1', FMF_SNR_MIN='1.2', FMF_MAD_SUBSAMPLE='8', FMF_STEP='1',
                  FMF_WORKERS='16', FMF_GPU_BUDGET='12e9', HELVETICA_DIR='/home/msseo/Download/Helvetica')
sys.path.insert(0, FA)
import config as C, scan, similarity as S
# loader workers started fresh (forkserver): forked workers die on PAGO (see README); the computation is unchanged
scan.ProcessPoolExecutor = functools.partial(cf.ProcessPoolExecutor, mp_context=mp.get_context('forkserver'))
import importlib.util
spec = importlib.util.spec_from_file_location('plotstyle', f'{DL}/pipeline/plotstyle.py'); plotstyle = importlib.util.module_from_spec(spec); spec.loader.exec_module(plotstyle)
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from obspy import UTCDateTime
from IPython.display import display, Image
sh = S.load_shelly(); ev = [str(e) for e in sh['event_id']]; cusp = list(sh['cuspid'].astype(int))
stations = [str(s) for s in sh['stations']]; prefixes = [str(p) for p in sh['channels']]; comps = [str(c) for c in sh['components']]
cat = pd.read_csv(C.CATALOG_CSV, parse_dates=['origin']).sort_values('origin').reset_index(drop=True)
tpl_loc = pd.read_csv(f'{DL}/catalogs/imsil_templates_ref25.csv').set_index('cuspid')
FIG = f'{DL}/notebooks/figs_13'; os.makedirs(FIG, exist_ok=True); band = f'{C.BANDPASS[0]:g}–{C.BANDPASS[1]:g} Hz'
sw = (cat.origin >= '2025-11-02 19:50') & (cat.origin <= '2025-11-03 05:00'); burst = (cat.origin >= '2018-03-06') & (cat.origin < '2018-03-09')
cand = cat[~sw & ~burst].copy()
cand['group'] = np.where(cand.n_templates >= 2, 'A', np.where(cand.template_cuspid == 400005, 'B', 'C'))
ref_burst = cat[burst].sort_values('cc').iloc[[-1]].copy(); ref_burst['group'] = 'R'
cand = pd.concat([cand, ref_burst]).sort_values('origin').reset_index(drop=True)
cand['label'] = [f'{g}{i + 1}' for g, i in zip(cand.group, cand.groupby('group').cumcount())]
print(f'{C.CATALOG_CSV.split("/")[-1]}: {len(cat)} detections; swarm window {int(sw.sum())}, March 2018 burst {int(burst.sum())}, others {int((~sw & ~burst).sum())}')
display(cand[['label', 'origin', 'template_cuspid', 'cc', 'n_present', 'n_templates']].assign(cc=lambda x: x.cc.round(3)))
""")

md(r"""
## 1. Timeline of the detections outside the 2025 swarm

The network CC of each detection against time. Marker size grows with the number of templates that fired. Shading marks the station configuration: 15 of the 22 FMF stations in 2018, 21 from 2019-03-18 (IMSB and five others installed), and 22 from December 2024 (SIDA).
""")
co(r"""
gcol = {'A': '#c1352b', 'B': '#1b3fa8', 'C': '#28867c', 'R': '0.45'}
gname = {'A': 'A: several templates', 'B': 'B: template 400005 alone', 'C': 'C: other single template', 'R': 'R: March 2018 noise burst (6,135; strongest shown)'}
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2019-03-18'), color='0.93', lw=0); ax.axvspan(pd.Timestamp('2024-12-23'), pd.Timestamp('2026-10-04'), color='0.96', lw=0)
ax.axvspan(pd.Timestamp('2025-11-02'), pd.Timestamp('2025-11-04'), color='#f4a6a6', lw=0)
bt = cat[burst]; ax.scatter(bt.origin, bt.cc, s=4, color='0.75', zorder=1)
for g in ('R', 'C', 'B', 'A'):
    q = cand[cand.group == g]; ax.scatter(q.origin, q.cc, s=18 + 22 * q.n_templates, color=gcol[g], edgecolor='black', lw=0.4, zorder=3, label=f'{gname[g]} ({len(q) if g != "R" else int(burst.sum())})')
for r in cand[cand.group.isin(['A'])].itertuples(): ax.annotate(r.label, (r.origin, r.cc), xytext=(5, 4), textcoords='offset points', fontsize=8)
ax.text(pd.Timestamp('2018-02-01'), 0.47, '15 stations', fontsize=8, color='0.35'); ax.text(pd.Timestamp('2021-06-01'), 0.47, '21 stations (IMSB from 2019-03-18)', fontsize=8, color='0.35')
ax.text(pd.Timestamp('2025-01-10'), 0.47, '22', fontsize=8, color='0.35'); ax.text(pd.Timestamp('2025-11-10'), 0.05, '2025 swarm', fontsize=8, color='#c1352b', rotation=90)
ax.set_xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2026-10-04')); ax.set_ylim(0, 0.5); ax.set_ylabel('Network CC'); ax.set_xlabel('Date (UTC)')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True, fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
## 2. Window evidence per candidate, with a same-day noise control

These are the per-channel numbers behind each detection set, the check of notebook 08 (a). The scan's own data arrays are loaded day by day. For each live template channel the table gives:
- the CC between the template window and the data at the detection;
- the window's signal-to-noise ratio (RMS of the window over RMS of the 5 s before its arrival).

The control is the same template at 40 random times on the same day, at least 60 s from any detection. The column `p_n_cc05` is the fraction of control draws with at least as many channels at CC ≥ 0.5; `p_med_cc` is the same for the median channel CC.

**Caveat (found when this notebook was first run).** Every candidate, including the noise reference R1, beats this control. A detection is by construction a peak of the network CC, so it always correlates better than a random time. The `p_` columns therefore do **not** separate real from false detections. The informative numbers are:
- the fraction of template windows with SNR ≥ 2, meaning real energy at the predicted arrivals;
- **which** channels carry the correlation (table below the scatter plot): many stations at once for an Imsil event, or one station again and again for a local source near that station.

The detection sets (section 3) are drawn in the same loop, so each day is read only once.
""")
co(r"""
def windows(ti, origin, day):
    # notebook 08 (a), verbatim apart from the day argument: per live channel, CC with the scanned data and the window SNR against the 5 s before its arrival
    o = UTCDateTime(pd.Timestamp(origin).to_pydatetime()); data, present, day0 = day; out = []
    for si, ph, ci, mo, n, tw, j in S.channels_of(sh, ti):
        if not present[si]: continue
        i0 = int(round((o + float(sh['ref_s'][ti]) - day0) * C.FS)) + mo
        if i0 - 600 < 0 or i0 + n > data.shape[-1]: continue
        d = data[si, ci, i0:i0 + n].astype(np.float64); nz = data[si, ci, i0 + 50 - 600:i0 + 50 - 100].astype(np.float64)
        if not np.any(d): continue
        a, b = tw - tw.mean(), d - d.mean(); den = np.sqrt((a ** 2).sum() * (b ** 2).sum()); cc = float((a * b).sum() / den) if den > 0 else np.nan
        snr = float(np.sqrt((b ** 2).mean()) / np.sqrt(((nz - nz.mean()) ** 2).mean())) if np.any(nz) else np.nan
        out.append(dict(station=stations[si], phase=ph, comp=comps[ci], cc=cc, snr=snr))
    return pd.DataFrame(out)
def summarize(w):
    return dict(n_ch=len(w), med_cc=w.cc.median(), n_cc05=int((w.cc >= 0.5).sum()), frac_snr2=float((w.snr >= 2).mean()), med_snr=w.snr.median())
rng = np.random.default_rng(20261014); rows = []; WIN = {}; PNG = {}
for day_key, g in cand.groupby(cand.origin.dt.strftime('%Y-%j')):
    y, doy = map(int, day_key.split('-')); day = scan.load_day(stations, prefixes, y, doy); d0 = pd.Timestamp(day[2].datetime)
    t_all = cat.origin[cat.origin.dt.strftime('%Y-%j') == day_key].values.astype('datetime64[s]').astype('int64')
    for r in g.itertuples():
        k = cusp.index(int(r.template_cuspid)); w = windows(k, r.origin, day); WIN[r.label] = w; s_ = summarize(w)
        ctl = []
        while len(ctl) < 40:
            t = d0 + pd.Timedelta(seconds=float(rng.uniform(600, 86400 - 600)))
            if np.all(np.abs(t_all - t.value // 10**9) > 60): ctl.append(summarize(windows(k, t, day)))
        ctl = pd.DataFrame(ctl)
        ch, net, mean = S.detection_vs_template(sh, k, r.origin, present=day[1])
        with contextlib.redirect_stdout(io.StringIO()): res = S.template_cc_day(sh, k, y, doy, day=day)   # silences FMF's dead-channel notices
        lat, lon = tpl_loc.loc[int(r.template_cuspid), ['latitude', 'longitude']]; det = pd.Timestamp(r.origin)
        png = f'{FIG}/{r.label}_{det:%Y%m%d_%H%M%S}_{int(r.template_cuspid)}_set.png'
        S.plot_detection_set(sh, k, r.origin, ch, f'{r.label}  ·  {det:%Y-%m-%d %H:%M:%S} UTC  ·  template {int(r.template_cuspid)}  ·  {band}  ·  network CC {net:.2f}  ·  {int(day[1].sum())} stations with data',
                             png, res=res, tpl_lat=lat, tpl_lon=lon); PNG[r.label] = png; plt.close('all')
        rows.append(dict(label=r.label, group=r.group, origin=det, template=int(r.template_cuspid), cc_catalog=r.cc, cc_recomputed=net, n_templates=int(r.n_templates), **s_,
                         ctl_n_cc05_p95=ctl.n_cc05.quantile(0.95), ctl_med_cc_p95=ctl.med_cc.quantile(0.95),
                         p_n_cc05=float((ctl.n_cc05 >= s_['n_cc05']).mean()), p_med_cc=float((ctl.med_cc >= s_['med_cc']).mean()), day_threshold=res['thr'] if res else np.nan))
    del day
EVD = pd.DataFrame(rows).sort_values('origin').reset_index(drop=True)
display(EVD[['label', 'origin', 'template', 'n_templates', 'cc_recomputed', 'day_threshold', 'n_ch', 'n_cc05', 'ctl_n_cc05_p95', 'p_n_cc05', 'med_cc', 'ctl_med_cc_p95', 'p_med_cc', 'frac_snr2', 'med_snr']].round(3))
""")
co(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for g in ('R', 'C', 'B', 'A'):
    q = EVD[EVD.group == g]
    ax[0].scatter(q.n_cc05, q.frac_snr2, s=40, color=gcol[g], edgecolor='black', lw=0.4, label=gname[g].split(' (')[0])
    ax[1].scatter(q.med_cc, q.p_med_cc, s=40, color=gcol[g], edgecolor='black', lw=0.4)
    for r in q.itertuples():
        ax[0].annotate(r.label, (r.n_cc05, r.frac_snr2), xytext=(4, 3), textcoords='offset points', fontsize=7); ax[1].annotate(r.label, (r.med_cc, r.p_med_cc), xytext=(4, 3), textcoords='offset points', fontsize=7)
ax[0].set_xlabel('Template channels with CC ≥ 0.5'); ax[0].set_ylabel('Fraction of template windows with SNR ≥ 2')
ax[1].set_xlabel('Median channel CC at the detection'); ax[1].set_ylabel('Share of same-day controls with an equal or higher median CC'); ax[1].axhline(0.05, color='0.5', ls='--', lw=0.8)
ax[0].legend(loc='lower center', bbox_to_anchor=(1.1, 1.02), ncol=4, frameon=True, fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
### Which channels carry the correlation

For each candidate: its local time (KST = UTC + 9 h), the channels with CC ≥ 0.5 (station.component, phase, CC and SNR), and the channels whose window has SNR ≥ 2.
""")
co(r"""
rows = []
for r in EVD.itertuples():
    w = WIN[r.label]; hi = w[w.cc >= 0.5].sort_values('cc', ascending=False); en = w[w.snr >= 2]
    rows.append(dict(label=r.label, KST=(r.origin + pd.Timedelta(hours=9)).strftime('%Y-%m-%d %H:%M'), n_cc05=len(hi),
                     cc_ge_05=', '.join(f'{q.station.split(".")[1]}.{q.comp} {q.phase} {q.cc:.2f}/{q.snr:.1f}' for q in hi.itertuples()),
                     stations_with_snr2=f'{en.station.nunique()} of {w.station.nunique()}: ' + ', '.join(sorted({q.station.split('.')[1] for q in en.itertuples()}))))
CHN = pd.DataFrame(rows); display(CHN)
b = EVD[EVD.group == 'B'].label; top = pd.Series([c.split(' ')[0] for v in CHN[CHN.label.isin(b)].cc_ge_05 for c in v.split(', ') if c]).value_counts()
print('group B: channels with CC >= 0.5 across its', len(b), 'candidates:', top.to_dict())
print('group B local hours (KST):', sorted((EVD[EVD.group == 'B'].origin + pd.Timedelta(hours=9)).dt.strftime('%H:%M')))
""")

md(r"""
## 3. Detection sets

### Group A: several templates at the same origin (2019-04-15)
""")
co(r"""
for r in EVD[EVD.group == 'A'].itertuples():
    print(f'{r.label}: {r.origin:%Y-%m-%d %H:%M:%S} UTC, template {r.template}, {r.n_templates} templates, network CC {r.cc_recomputed:.2f}; channels with CC >= 0.5: {r.n_cc05} (same-day control 95th percentile {r.ctl_n_cc05_p95:.0f})')
    display(Image(PNG[r.label]))
""")
md(r"""
### Group B: template 400005 alone (September 2023 – March 2025)
""")
co(r"""
for r in EVD[EVD.group == 'B'].itertuples():
    print(f'{r.label}: {r.origin:%Y-%m-%d %H:%M:%S} UTC, network CC {r.cc_recomputed:.2f}; channels with CC >= 0.5: {r.n_cc05} (control 95th pct {r.ctl_n_cc05_p95:.0f}); median channel CC {r.med_cc:.2f} (control 95th pct {r.ctl_med_cc_p95:.2f})')
    display(Image(PNG[r.label]))
""")
md(r"""
### Group C: other single-template hits
""")
co(r"""
for r in EVD[EVD.group == 'C'].itertuples():
    print(f'{r.label}: {r.origin:%Y-%m-%d %H:%M:%S} UTC, template {r.template}, network CC {r.cc_recomputed:.2f}; channels with CC >= 0.5: {r.n_cc05} (control 95th pct {r.ctl_n_cc05_p95:.0f}); median channel CC {r.med_cc:.2f}')
    display(Image(PNG[r.label]))
""")
md(r"""
### Reference R: the strongest detection of the March 2018 noise burst
""")
co(r"""
for r in EVD[EVD.group == 'R'].itertuples():
    print(f'{r.label}: {r.origin:%Y-%m-%d %H:%M:%S} UTC, template {r.template}, network CC {r.cc_recomputed:.2f}; channels with CC >= 0.5: {r.n_cc05}; median channel CC {r.med_cc:.2f}')
    display(Image(PNG[r.label]))
""")

md(r"""
## 4. Full waveforms of group A

The template (black) and the detection (red) over the full P-to-S-coda window at the nearest stations of the template, as in notebook 07.
""")
co(r"""
for r in EVD[EVD.group == 'A'].itertuples():
    k = cusp.index(r.template); lat, lon = tpl_loc.loc[r.template, ['latitude', 'longitude']]; png = f'{FIG}/{r.label}_{r.origin:%Y%m%d_%H%M%S}_{r.template}_full.png'
    S.plot_full_trace_comparison(sh, k, r.origin, f'{r.label}: template {r.template} (black) and the detection of {r.origin:%Y-%m-%d %H:%M:%S} UTC (red), {band}', png, tpl_lat=lat, tpl_lon=lon); plt.close('all')
    display(Image(png))
""")

md(r"""
## 5. Repeater gathers: the candidates beside the swarm detections of the same template

Each gather stacks every detection of one template at one station in time order, aligned on the template's pick (P where the template has one, otherwise S). The template is on top. Red marks the template's own event, blue a new detection, gray a reference event. Rows are in time order, so the long-period candidates come first, followed by the template's 2025 swarm detections (up to 40 rows). The number on each trace is its correlation with the template on that component.

Shown: the templates of group A (400007 and 400001) and template 400005 (group B), each at the two nearest stations where the template has windows.
""")
co(r"""
import station_gather as G          # imported here, not at the top: it switches matplotlib to Agg on import
coords = S._station_coords()
def nearest_stations(c, n=2):
    k = cusp.index(c); lat, lon = tpl_loc.loc[c, ['latitude', 'longitude']]; have = sorted({stations[si] for si, *_ in S.channels_of(sh, k)})
    return sorted(have, key=lambda s: np.hypot((coords[s][0] - lat) * 111.19, (coords[s][1] - lon) * 111.19 * np.cos(np.radians(lat))))[:n]
for c, t0 in ((400007, '2019-01-01'), (400001, '2019-01-01'), (400005, '2023-01-01')):
    for sta in nearest_stations(c):
        png = f'{FIG}/gather_{c}_{sta.split(".")[1]}.png'
        try:
            G.station_gather(c, sta, t0, '2025-11-03 06:00', png, title=f'Template {c} at {sta.split(".")[1]}: long-period candidates (first rows), then its 2025 swarm detections', same_template=True, maxn=40)
            plt.close('all'); display(Image(png))
        except SystemExit as e:
            print(f'{c} {sta}: {e}')
""")

md(r"""
## Comprehensive summary
""")
co(r"""
grp = EVD.groupby('group').agg(n=('label', 'size'), templates=('n_templates', 'median'), net_cc=('cc_recomputed', 'median'), n_cc05=('n_cc05', 'median'),
                               p_med_cc_below_005=('p_med_cc', lambda v: int((v <= 0.05).sum())), frac_snr2=('frac_snr2', 'median')).rename(index=gname)
display(grp.round(3))
summary = pd.DataFrame([
    ('Long-scan catalog (2018-01-01 .. 2026-10-03)', f'{len(cat)} detections: swarm window {int(sw.sum())}, March 2018 burst {int(burst.sum())}, others {int((~sw & ~burst).sum())}'),
    ('Group A (several templates, 2019-04-15)', '; '.join(f'{r.label} {r.origin:%H:%M:%S}: {r.n_templates} templates, CC {r.cc_recomputed:.2f}, {r.n_cc05} channels >= 0.5 (control p95 {r.ctl_n_cc05_p95:.0f}), p_med_cc {r.p_med_cc:.2f}' for r in EVD[EVD.group == 'A'].itertuples())),
    ('Group B (template 400005 alone)', f'{int((EVD.group == "B").sum())}; median channels >= 0.5: {EVD[EVD.group == "B"].n_cc05.median():.0f}; with p_med_cc <= 0.05: {int((EVD[EVD.group == "B"].p_med_cc <= 0.05).sum())}'),
    ('Group C (other single template)', f'{int((EVD.group == "C").sum())}; median channels >= 0.5: {EVD[EVD.group == "C"].n_cc05.median():.0f}; with p_med_cc <= 0.05: {int((EVD[EVD.group == "C"].p_med_cc <= 0.05).sum())}'),
    ('Reference R (strongest noise-burst detection)', '; '.join(f'CC {r.cc_recomputed:.2f}, {r.n_cc05} channels >= 0.5, p_med_cc {r.p_med_cc:.2f}' for r in EVD[EVD.group == 'R'].itertuples())),
], columns=['quantity', 'value'])
display(summary)
EVD.to_csv(f'{DL}/catalogs/fmf_long_candidates_evidence.csv', index=False)
print('Evidence only: which candidates are genuine is your judgment. Per-candidate numbers: catalogs/fmf_long_candidates_evidence.csv; figures: notebooks/figs_13/.')
""")

nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(kernelspec=dict(name="fmf-imsil", display_name="fmf (Imsil FMF)", language="python"),
                        language_info=dict(name="python")))
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
json.dump(nb, open(OUT, "w"), indent=1)
print(f"wrote {OUT}: {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code, all compiled)")
