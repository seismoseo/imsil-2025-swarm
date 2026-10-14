"""Copied unchanged from Haenam fmf/station_gather.py (2026-10-08); reads windows through this folder's similarity.py (aligned).
All detections of a period at ONE station, stacked: the classic repeater display.

One row per detection (time-ordered), each row the record at a single station-component aligned on the template's predicted
arrival, template on top for reference. Right-hand labels give the detection time, its network CC and the correlation of that
row against the template. This shows at a glance that the detections are the same waveform repeating.

usage (run env): python station_gather.py CUSPID NET.STA 'START' 'END' OUT ['TITLE']
   e.g.          python station_gather.py 400177 KS.JDO2 '2023-01-01' '2023-12-31' fig.png
"""
import os,sys
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import config as C, similarity as S
HERE=os.path.dirname(os.path.abspath(__file__))

def station_gather(cuspid,station,t0,t1,outfile,title=None,pre=2.0,span=8.0,maxn=40,same_template=True,comps_want=('Z','N','E'),dedup_s=0.0):
    """All detections of one template at ONE station, as a Z | N | E panel set (the classic repeater display).

    Every component uses the SAME window - from `pre` s before the template's P pick (or its S pick where the template has no
    P on that station) through `span` s - so P, the P-S interval and the S coda are on the same time axis for Z, N and E.
    Both picks are annotated: P as a red dashed line, S as a blue dashed line. The template is on top, the detections beneath
    in time order, each labelled with its network CC and its correlation on that component.
    """
    sh=S.load_shelly();cusp=list(sh['cuspid'].astype(int));ti=cusp.index(int(cuspid))
    stations=[str(x) for x in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']]
    net,sta=station.split('.');name=f'{net}.{sta}'
    if name not in stations:raise SystemExit(f'{name} is not in the template set')
    si=stations.index(name)
    # the template's picks at this station, relative to the detection origin (window start + 0.5 s = the pick)
    picks={}
    for s_i,ph,ci,mo,n,tw,j in S.channels_of(sh,ti):
        if s_i==si:picks.setdefault(ph,mo/C.FS+float(sh['ref_s'][ti])+0.5)
    if not picks:raise SystemExit(f'template {cuspid} has no window at {name}')
    t_ref=picks.get('P',picks.get('S'))                      # align on P where the template has one, else on S
    npts=int(round((span+pre)*C.FS))
    cat=pd.read_csv(C.CATALOG_CSV,parse_dates=['origin'])
    det=cat[(cat.origin>=pd.Timestamp(t0))&(cat.origin<=pd.Timestamp(t1))].sort_values('origin')
    if same_template:det=det[det.template_cuspid==int(cuspid)]
    if dedup_s>0 and len(det)>1:                             # one event matched by two templates a few s apart: keep the higher network CC
        keep=[];d_=det.sort_values('cc',ascending=False)
        for r in d_.itertuples():
            if all(abs((r.origin-k).total_seconds())>dedup_s for k in keep):keep.append(r.origin)
        det=det[det.origin.isin(keep)].sort_values('origin')
    torig=UTCDateTime(str(sh['origin_utc'][ti]))
    have=[c for c in comps_want if S.read_window(net,sta,c,prefixes[si],torig+t_ref-pre,npts) is not None]
    if not have:raise SystemExit('no component readable for the template event')
    tpl={c:S.read_window(net,sta,c,prefixes[si],torig+t_ref-pre,npts) for c in have}
    rows=[]
    for r in det.itertuples():
        o=UTCDateTime(pd.Timestamp(r.origin).to_pydatetime());dd={}
        for c in have:
            d=S.read_window(net,sta,c,prefixes[si],o+t_ref-pre,npts)
            if d is None:continue
            a=tpl[c]-tpl[c].mean();b=d-d.mean();den=np.sqrt((a**2).sum()*(b**2).sum())
            dd[c]=dict(d=d,cc=float((a*b).sum()/den) if den>0 else np.nan)
        # colors follow the final catalog's classes (decision 20): is_new = not a reference event; self = the template's own event
        if dd:rows.append(dict(origin=r.origin,net_cc=r.cc,self=bool(r.self_detection),new=bool(r.is_new),tpl=int(r.template_cuspid),comp=dd))
    if not rows:raise SystemExit('no detections with data in the window')
    rows=rows[:maxn];n=len(rows);t=np.arange(npts)/C.FS-pre
    H=0.42*(n+1)+2.2                                   # header (title, key) placed in inches so few-trace gathers do not collide
    fig,axs=plt.subplots(1,len(have),figsize=(4.6*len(have),H),sharey=True,squeeze=False)
    axs=axs[0]
    for ax,c in zip(axs,have):
        ax.plot(t,0.42*tpl[c]/(np.abs(tpl[c]).max() or 1)+n,color='k',lw=.9)
        for k,r in enumerate(rows[::-1]):
            e=r['comp'].get(c)
            if e is None:continue
            col='#c1352b' if r['self'] else ('#1b3fa8' if r['new'] else '.35')
            ax.plot(t,0.42*e['d']/(np.abs(e['d']).max() or 1)+k,color=col,lw=.7)
            ax.text(.995,k+0.16,f"{e['cc']:.2f}",transform=ax.get_yaxis_transform(),fontsize=6.5,va='center',ha='right',color=col)
        for ph,colr in (('P','#c1352b'),('S','#1b3fa8')):     # BOTH picks on every component, same window
            if ph in picks:ax.axvline(picks[ph]-t_ref,color=colr,lw=.9,ls=(0,(4,2.5)),zorder=0,alpha=.85)
        ax.set_title(f'{c} component',fontsize=9.5);ax.set_xlim(-pre,span);ax.set_xlabel(f'Seconds from the template P pick at {name}')
    axs[0].set_yticks([]);axs[0].set_ylim(-.8,n+.8)
    for k,r in enumerate(rows[::-1]):
        col='#c1352b' if r['self'] else ('#1b3fa8' if r['new'] else '.35')
        axs[0].text(-0.015,k,f"{pd.Timestamp(r['origin']):%Y-%m-%d %H:%M:%S}  CC {r['net_cc']:.2f}"+('' if r['tpl']==int(cuspid) else f"  (by {r['tpl']})"),transform=axs[0].get_yaxis_transform(),
                    fontsize=7,va='center',ha='right',color=col)
    axs[0].text(-0.015,n,'TEMPLATE',transform=axs[0].get_yaxis_transform(),fontsize=7.5,va='center',ha='right',fontweight='bold')
    fig.suptitle(title or f'Every detection of template {cuspid} at {name}',fontsize=11,y=1-0.08/H)
    fig.text(0.5,1-0.42/H,'template (black) · the template\'s own event (red) · new (blue) · reference event (gray) · dashed: template P (red) and S (blue) picks · '
             'the number on each trace is its correlation with the template on that component',ha='center',fontsize=7.5)
    fig.subplots_adjust(top=1-0.95/H,bottom=0.09,left=0.135,right=0.985,wspace=0.06)
    fig.savefig(outfile,dpi=150,bbox_inches='tight',pad_inches=0.12);plt.close(fig)
    print(f'{len(rows)} detections x {len(have)} components at {name} -> {outfile}')
    return rows

if __name__=='__main__':
    station_gather(int(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6] if len(sys.argv)>6 else None)
