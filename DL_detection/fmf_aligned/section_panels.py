"""Near-event detection panels (Haenam notebook 46, figures W03/W04), copied from Haenam fmf/catalog_procedure_figs.py
(2026-10-08). The plotting logic of section_panels and legend_above is unchanged; differences: the module no longer loads
Haenam's catalogs at import (the caller passes the shelly npz arrays, the station sensor table and the output file), and all
four spines are kept (closed frames, user rule); the legend says 'P'/'S' for the other event instead of 'picked P'/'picked S',
because at Imsil the other event can be a stronger detection whose arrivals are predicted, not picked.

section_panels(fig_id, rows, marks, ...): Z records at the nsta nearest stations; each trace scaled inside the detection's own
predicted P-S window at that station (clipped at 1.5x), so a small detection next to a larger event is visible. Marks: the
detection's predicted P/S, the other event's P/S, and raw PhaseNet+ picks.
  rows : figure, row, detection, t0, t1, stations ('NET.STA:km;...'), template, cc, n_templates, other_time, other_label
  marks: row, station, kind ('detection' | 'other' | 'raw'), phase ('P' | 'S'), time
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import UTCDateTime
import config as C
import similarity as S

P_C, S_C, DET_C, OTH_C, PREV_C = '#c1352b', '#1b3fa8', '#1b3fa8', '#c1352b', '#e07b39'


def legend_above(fig, handles, ncol, y=1.0):
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=True, framealpha=1.0, edgecolor='0.7', fontsize=8.5)


def section_panels(fig_id, rows, marks, stations, prefixes, sensor, outfile, nsta=3):
    """W03/W04: Z records at the nsta nearest stations (to the detecting template's event) with data; each trace scaled inside the
    detection's own predicted P-S window at that station (clipped at 1.5x), so a small detection next to a larger event is visible."""
    BAND = f'{C.BANDPASS[0]:g}–{C.BANDPASS[1]:g} Hz'
    rows = rows[rows.figure == fig_id].reset_index(drop=True); n = len(rows); ncol = 2 if n > 4 else 1; nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(7.4 * ncol if ncol > 1 else 11.0, 2.25 * nrow + .2), layout='constrained', squeeze=False)   # one column: wide enough for the legend
    for ax in axs.ravel()[n:]: ax.set_visible(False)
    for ax, r in zip(axs.ravel(), rows.itertuples()):
        det = UTCDateTime(r.detection.to_pydatetime()); t0 = UTCDateTime(r.t0.to_pydatetime()); t1 = UTCDateTime(r.t1.to_pydatetime()); npts = int((t1 - t0) * C.FS)
        mk = marks[marks.row == r.row]; k = 0; names = []
        for item in str(r.stations).split(';'):
            name, km = item.split(':'); net, sta = name.split('.')
            pre = prefixes[stations.index(name)] if name in stations else sensor.get(name, 'HH')
            d = S.read_window(net, sta, 'Z', pre, t0, npts)
            if d is None: continue
            x = np.arange(d.size) / C.FS + (t0 - det); m = mk[mk.station == name]
            dp = m[(m.kind == 'detection')]; tp_ = [(UTCDateTime(t.to_pydatetime()) - det) for t in dp.time]
            lo, hi = (min(tp_) - .5, max(tp_) + 1.5) if tp_ else (0., 10.); w = (x >= lo) & (x <= hi)
            sc = np.abs(d[w]).max() if w.any() else np.abs(d).max(); y = np.clip(d / (sc or 1), -1.5, 1.5) * .42
            ax.plot(x, y + k, color='.3', lw=.4)
            for q in m.itertuples():
                tq = UTCDateTime(q.time.to_pydatetime()) - det
                if q.kind == 'raw': ax.plot(tq, k + .5, marker='v', ms=4.5, color='k', mfc='k' if q.phase == 'P' else 'white', mew=.8, zorder=5)
                else: ax.plot([tq, tq], [k - .45, k + .45], color=DET_C if q.kind == 'detection' else OTH_C, lw=1.2, ls='-' if q.phase == 'P' else (0, (3, 1.5)), zorder=4)
            names.append(f'{sta} {float(km):.0f} km'); k += 1
            if k == nsta: break
        dt = (r.detection - r.other_time).total_seconds()
        ax.set_yticks(range(len(names)), names, fontsize=7.5); ax.set_ylim(-.7, max(len(names), 1) - .25); ax.set_xlim(t0 - det, t1 - det)
        ax.set_title(f'{r.detection:%Y-%m-%d %H:%M:%S.%f}'[:-4] + f' · template {r.template} · CC {r.cc:.2f} · {r.n_templates} template{"s" if r.n_templates != 1 else ""} · '
                     f'{dt:+.2f} s from the {r.other_label}', loc='left', fontsize=8)
    for col in range(ncol):
        vis = [axs[i, col] for i in range(nrow) if i * ncol + col < n]
        if vis: vis[-1].set_xlabel('Seconds from the detection origin')
    other = rows.other_label.iloc[0]
    legend_above(fig, [Line2D([], [], color='.3', lw=.8, label=f'Z record ({BAND}), scaled inside the detection\'s predicted P–S window, clipped at 1.5×'),
                       Line2D([], [], color=DET_C, lw=1.2, label='Detection: predicted P'), Line2D([], [], color=DET_C, lw=1.2, ls=(0, (3, 1.5)), label='Detection: predicted S'),
                       Line2D([], [], color=OTH_C, lw=1.2, label=f'The {other}: P'), Line2D([], [], color=OTH_C, lw=1.2, ls=(0, (3, 1.5)), label=f'The {other}: S'),
                       Line2D([], [], marker='v', ls='', color='k', mfc='k', label='Raw PhaseNet+ P pick'), Line2D([], [], marker='v', ls='', color='k', mfc='white', label='Raw PhaseNet+ S pick')], 4)
    fig.savefig(outfile, dpi=120, bbox_inches='tight'); plt.close(fig)
    return outfile
