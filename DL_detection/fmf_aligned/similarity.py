"""ALIGNED VARIANT (D20, 2026-10-08): copied from Haenam fmf/similarity.py; only read_window adds scan._to_grid so windows match the
aligned scan.
Template-similarity views for the Haenam two-pass (Shelly) FMF run — the Buan `buan_fmf/similarity.py` ported to P/S
windows, so every detection is interpretable rather than a number in a catalog:
  * detection_vs_template : the continuous window at a detection overlaid on the template, per live channel, with the
                            per-channel CC and the weighted network CC recomputed with the scan-day weights
  * plot_overlay          : per-channel template (black) vs detected (red), rows sorted by CC (the Buan figure)
  * template_cc_day       : the network-CC trace of ONE template over one day, summed exactly as the scan does (P pass on Z,
                            S pass on Z or on the horizontals, day weights, SNR/epoch masks) + the whole-day 12xMAD threshold
  * plot_cc_trace         : full-day context + a magnified window around the detection (the Buan figure)
Run with the scan's environment (the exports of run_full_period.sh) so config.py resolves to the full_period run.
"""
import os,glob
import numpy as np
from obspy import read,UTCDateTime
import config as C
import scan
import shelly
from shelly import LP,LS
_KEYS=('P_waveforms','S_waveforms','P_moveouts','S_moveouts','P_weights','S_weights','ref_s','stations','channels','components','s_comp','P_snr','S_snr','cuspid','origin_utc','event_id','sequence')

def load_shelly():
    z=np.load(C.SHELLY_NPZ,allow_pickle=True);return {k:z[k] for k in z.files if k in _KEYS}

def _use_helvetica():
    import matplotlib as mpl,matplotlib.font_manager as fm
    d=os.environ.get('HELVETICA_DIR','/home/msseo/Downloads/Helvetica')
    try:
        if os.path.isdir(d):
            for f in fm.findSystemFonts(d):fm.fontManager.addfont(f)
        if 'Helvetica' in {f.name for f in fm.fontManager.ttflist}:mpl.rcParams['font.family']=['Helvetica','DejaVu Sans']
    except Exception:pass
    mpl.rcParams['axes.unicode_minus']=False

def read_window(net,sta,comp,prefix,t0,npts):
    """Continuous window [t0, t0 + npts/FS) processed as scan._load_channel does (detrend, resample to FS, 2-15 Hz bandpass);
    the template's band code first, then the fallbacks the scan allows. None when no data covers the window."""
    for pre in [prefix]+[q for q in ('HH','HG','EL') if q!=prefix]:
        f=glob.glob(f"{C.CONTINUOUS}/{net}/{sta}/{pre}{comp}.D/*.D.{t0.year}.{t0.julday:03d}")
        if f:break
    else:return None
    try:
        st=read(f[0],starttime=t0-10,endtime=t0+npts/C.FS+10);st.merge(method=1,fill_value=0);tr=st[0];tr.detrend('linear')
        if abs(tr.stats.sampling_rate-C.FS)>1e-6:tr.resample(C.FS)
        tr=scan._to_grid(tr)   # ALIGNED VARIANT (D20): the same grid shift as scan._load_channel
        tr.filter('bandpass',freqmin=C.BANDPASS[0],freqmax=C.BANDPASS[1],corners=4,zerophase=True);seg=tr.slice(t0,t0+(npts-1)/C.FS)
        d=seg.data.astype(np.float64)
        return d[:npts] if d.size>=npts and np.abs(d).max()>0 else None
    except Exception:return None

def day_weights(sh,present,day0,ti):
    """The scan-day channel weights of template ti (present stations, SNR gate where affordable, live-channel gates, sensor-epoch
    mask, normalised to 1) — the same rules as shelly.scan_day, for one template."""
    stations=[str(s) for s in sh['stations']];Pw=sh['P_weights'][ti].astype(np.float64).copy();Sw=sh['S_weights'][ti].astype(np.float64).copy()
    Pw[~present,:]=0.0;Sw[~present,:]=0.0
    if getattr(C,'SNR_MIN',0)>0 and 'P_snr' in sh:
        Pm=Pw.copy();Sm=Sw.copy();ps=sh['P_snr'][ti];ss=sh['S_snr'][ti]
        Pm[np.isfinite(ps)&(ps<C.SNR_MIN)]=0.0;Sm[np.isfinite(ss)&(ss<C.SNR_MIN)]=0.0
        okP=(Pm[:,0]>0).sum()>=getattr(C,'MIN_LIVE_P',0);ok=okP and ((Pm[:,0]>0).sum()+(Sm>0).sum())>=getattr(C,'MIN_LIVE_CH',4)
        if ok:Pw,Sw=Pm,Sm
    liveP=(Pw[:,0]>0).sum();liveAll=liveP+(Sw>0).sum()
    if liveP<getattr(C,'MIN_LIVE_P',1) or liveAll<getattr(C,'MIN_LIVE_CH',1):Pw[:]=0.0;Sw[:]=0.0
    if getattr(C,'EXCLUDE_EPOCHS',None):
        torig=UTCDateTime(str(sh['origin_utc'][ti])).timestamp
        for (net,sta),spans in C.EXCLUDE_EPOCHS.items():
            key=f'{net}.{sta}'
            if key not in stations:continue
            si=stations.index(key)
            for a,bnd in spans:
                t0,t1=UTCDateTime(a).timestamp,UTCDateTime(bnd).timestamp+86400
                if (t0<=day0.timestamp<t1)!=(t0<=torig<t1):Pw[si,:]=0.0;Sw[si,:]=0.0
    tot=Pw.sum()+Sw.sum()
    if tot>0:Pw/=tot;Sw/=tot
    return Pw,Sw

def channels_of(sh,ti):
    """Live channels of template ti: (si, phase, comp index, moveout samples, window length, template waveform, build weight)."""
    comps=[str(c) for c in sh['components']];ci_N,ci_E,ci_Z=[comps.index(c) for c in 'NEZ'];s_comp=[str(x) for x in sh['s_comp']];out=[]
    for si in range(len(sh['stations'])):
        if sh['P_weights'][ti,si,0]>0:out.append((si,'P',ci_Z,int(sh['P_moveouts'][ti,si,0]),LP,sh['P_waveforms'][ti,si,0].astype(np.float64),0))
        if s_comp[si]=='Z':
            if sh['S_weights'][ti,si,0]>0:out.append((si,'S',ci_Z,int(sh['S_moveouts'][ti,si,0]),LS,sh['S_waveforms'][ti,si,0].astype(np.float64),0))
        else:
            for j,ci in ((0,ci_N),(1,ci_E)):
                if sh['S_weights'][ti,si,j]>0:out.append((si,'S',ci,int(sh['S_moveouts'][ti,si,j]),LS,sh['S_waveforms'][ti,si,j].astype(np.float64),j))
    return out

def detection_vs_template(sh,ti,det_origin,present=None):
    """Per live channel of template ti: the continuous window at the detection (start = origin + ref_s + moveout/FS, as the
    scan aligns it) and its normalised CC against the template window. Returns (channels, network_cc, mean_cc): network_cc is
    the weighted sum with the scan-day weights (pass the day's `present` mask from template_cc_day to reproduce the scan's
    weights exactly; otherwise the stations with data at the detection stand in for it), mean_cc the plain per-channel mean."""
    det=UTCDateTime(det_origin);stations=[str(s) for s in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']]
    chans=[];present_read=np.zeros(len(stations),bool)
    for si,ph,ci,mo,n,tw,j in channels_of(sh,ti):
        net,sta=stations[si].split('.');t0=det+float(sh['ref_s'][ti])+mo/C.FS;d=read_window(net,sta,comps[ci],prefixes[si],t0,n)
        if d is None:continue
        d=d-d.mean();t=tw-tw.mean();den=np.sqrt((d**2).sum()*(t**2).sum());cc=float((d*t).sum()/den) if den>0 else 0.0
        # pre-arrival noise on the SAME channel (C.NOISE_WIN = [arrival-6 s, arrival-1 s]) -> per-channel SNR of the DETECTION and
        # of the TEMPLATE window; a ratio measured on a channel where either side is noise is not an amplitude measurement.
        n0,n1=getattr(C,'NOISE_WIN',(6.0,1.0));nw=read_window(net,sta,comps[ci],prefixes[si],t0+0.5-n0,int(round((n0-n1)*C.FS)))
        snr=float(np.sqrt((d**2).mean())/np.sqrt((nw**2).mean())) if (nw is not None and np.sqrt((nw**2).mean())>0) else np.nan
        chans.append(dict(si=si,station=stations[si],comp=comps[ci],phase=ph,j=j,template=t,data=d,cc=cc,t0=t0,snr=snr));present_read[si]=True
    if not chans:return [],np.nan,np.nan
    day0=C.day_start(det.year,det.julday);Pw,Sw=day_weights(sh,np.asarray(present,bool) if present is not None else present_read,day0,ti)
    for o in chans:o['weight']=float(Pw[o['si'],0] if o['phase']=='P' else Sw[o['si'],o['j']])
    net_cc=float(sum(o['weight']*o['cc'] for o in chans));mean_cc=float(np.mean([o['cc'] for o in chans]))
    return chans,net_cc,mean_cc

def plot_overlay(chans,title,outfile=None,max_rows=14):
    """Per-channel template (black) vs detected (red) overlay, rows sorted by CC, per-row CC and weight (the Buan figure)."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;_use_helvetica()
    rows=sorted(chans,key=lambda o:-o['cc'])[:max_rows];n=len(rows)
    fig,ax=plt.subplots(n,1,figsize=(7.2,0.62*n+0.6),sharex=True);ax=np.atleast_1d(ax)
    for a,o in zip(ax,rows):
        t=np.arange(o['template'].size)/C.FS;pre=0.5
        a.plot(t-pre,o['template']/(np.abs(o['template']).max() or 1),'k',lw=.8,label='template');a.plot(t-pre,o['data']/(np.abs(o['data']).max() or 1),'r',lw=.8,alpha=.85,label='detected')
        a.axvline(0,color='.6',lw=.5,ls=':');a.set_yticks([]);a.set_ylabel(f"{o['station']}.{o['comp']} {o['phase']}",fontsize=6.5,rotation=0,ha='right',va='center')
        a.text(.99,.86,f"CC {o['cc']:.2f}  w {o.get('weight',0):.2f}",transform=a.transAxes,ha='right',va='top',fontsize=6.5,color='#b22222')
    ax[0].legend(loc='upper left',fontsize=6,ncol=2,frameon=True,framealpha=1);ax[-1].set_xlabel('Time from the template pick (s)')
    fig.suptitle(title,fontsize=9);fig.tight_layout(rect=(0,0,1,.97))
    if outfile:fig.savefig(outfile,dpi=160,bbox_inches='tight');plt.close(fig)
    return fig

def plot_station_overlay(sh,ti,det_origin,chans,title,outfile=None,pre=1.5,post=2.5):
    """One row per station and component: the continuous trace at the detection from pre s before the predicted P to post s
    after the predicted S (same band as the scan), with the template's P window (red) and S window (blue) overlaid in place,
    each scaled to the data in its own window and labelled with its CC and scan weight. Rows ordered by S-P (near stations at
    the bottom). `chans` from detection_vs_template."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;_use_helvetica()
    stations=[str(x) for x in sh['stations']];prefixes=[str(x) for x in sh['channels']];groups={}
    for o in chans:groups.setdefault((o['station'],o['comp']),{})[o['phase']]=o
    rows=[]
    for (sta,comp),g in groups.items():
        tP=g['P']['t0']+0.5 if 'P' in g else None;tS=g['S']['t0']+0.5 if 'S' in g else None;ref=tP if tP is not None else tS;end=tS if tS is not None else tP
        t0=ref-pre;n=int(round((end+post-t0)*C.FS));net,name=sta.split('.');d=read_window(net,name,comp,prefixes[stations.index(sta)],t0,n)
        if d is None:continue
        rows.append(dict(sta=sta,comp=comp,g=g,ref=ref,t=np.arange(n)/C.FS-pre,d=d/(np.abs(d).max() or 1),sp=(tS-tP) if (tP is not None and tS is not None) else (99.0 if tS is None else 50.0)))
    rows.sort(key=lambda r:-r['sp']);nrow=len(rows)
    fig,ax=plt.subplots(figsize=(9,0.62*nrow+1.3),layout='constrained')
    for k,r in enumerate(rows):
        ax.plot(r['t'],r['d']*0.42+k,color='k',lw=.6)
        for ph,colr in (('P','#c1352b'),('S','#1b3fa8')):
            if ph not in r['g']:continue
            o=r['g'][ph];tt=np.arange(o['template'].size)/C.FS+(o['t0']-r['ref']);m=(r['t']>=tt[0])&(r['t']<=tt[-1]);amp=np.abs(r['d'][m]).max() if m.any() else 1.0
            ax.plot(tt,o['template']/(np.abs(o['template']).max() or 1)*amp*0.42+k,color=colr,lw=.9,alpha=.9);ax.axvline(o['t0']+0.5-r['ref'],color=colr,lw=.5,ls=':',ymin=(k+0.05)/nrow,ymax=(k+0.95)/nrow)
            ax.text(tt[-1]+0.05,k+0.3,f"{ph} {o['cc']:.2f} (w {o['weight']:.2f})",color=colr,fontsize=6.5,va='center')
    ax.set_yticks(range(nrow),[f"{r['sta']}.{r['comp']}" for r in rows],fontsize=7);ax.set_ylim(-.7,nrow-.3);ax.set_xlim(-pre,max(r['t'][-1] for r in rows)+2.2)
    ax.set_xlabel('Seconds from the predicted P (S-only rows: from the predicted S)');ax.set_title(title,fontsize=8.5)
    if outfile:fig.savefig(outfile,dpi=160,bbox_inches='tight');plt.close(fig)
    return fig

def _station_coords():
    import pandas as pd
    t=pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'station_table','stations.csv'))
    return {f'{r.Network}.{r.Code}':(r.Latitude,r.Longitude) for r in t.itertuples()}

def plot_detection_set(sh,ti,det_origin,chans,title,outfile=None,nmax=18,res=None,half_s=12.0,tpl_lat=None,tpl_lon=None,pre=1.5,post=3.0):
    """ONE figure per detection. Each row is a station-component on a single continuous time axis: the record (gray) from
    `pre` s before the predicted P to `post` s after the predicted S, with the template's P window (red) and S window (blue)
    overlaid in place, each labelled with its own correlation. Rows are ordered by distance from the template event and
    labelled with station, component and distance; a station with no P pick in the template shows only the S window (P picks
    come from the template event's own PhaseNet picks, and a small event at distance often has S but no P). Below, the
    detector's own view: the template's network-CC trace over the day and the +-half_s window around the detection."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from obspy.geodetics import gps2dist_azimuth
    PICK_PRE=0.5                                     # the template windows start this long before the pick (shelly.PRE)
    _use_helvetica();det=UTCDateTime(det_origin);coords=_station_coords()
    stations=[str(x) for x in sh['stations']];prefixes=[str(p) for p in sh['channels']]
    def dist(name):
        if tpl_lat is None or name not in coords:return np.nan
        return gps2dist_azimuth(tpl_lat,tpl_lon,*coords[name])[0]/1000.0
    # group the measured channels by (station, component): each becomes one row carrying its P and/or S window
    # ONE time frame for every row: seconds from the detection origin, so the moveout across stations is readable and a row
    # without a P window still sits at its true position.
    by={}
    for o in chans:by.setdefault((o['station'],o['comp']),{})[o['phase']]=o
    tmin=min(min((o['t0']-det) for o in ph.values()) for ph in by.values())-pre
    tmax=max(max((o['t0']-det)+o['template'].size/C.FS for o in ph.values()) for ph in by.values())+post
    n=int(round((tmax-tmin)*C.FS));rows=[]
    for (name,comp),ph in by.items():
        net,sta=name.split('.');d=read_window(net,sta,comp,prefixes[stations.index(name)],det+tmin,n)
        if d is None:continue
        rows.append(dict(name=name,comp=comp,ph=ph,t=np.arange(n)/C.FS+tmin,d=d/(np.abs(d).max() or 1),km=dist(name),cc=max(float(o['cc']) for o in ph.values())))
    if not rows:return None
    # SELECTION (disclosed in the caption): the channels that actually carry the match - highest full-trace correlation
    # first - then drawn in distance order so the moveout still reads. Selecting by distance instead would fill the figure
    # with the nearest station, which here (HAWA, 7 km) is also the noisiest.
    order={'Z':0,'N':1,'E':2}
    rows=sorted(rows,key=lambda r:-(r['cc'] if np.isfinite(r['cc']) else -9))[:nmax]
    rows=sorted(rows,key=lambda r:(np.nan_to_num(r['km'],nan=9e9),r['name'],order.get(r['comp'],9)))
    nr=len(rows);hcc=2.2 if res is not None else 0.0
    fig=plt.figure(figsize=(12,0.5*nr+hcc+0.8))
    gs=fig.add_gridspec(2,2,height_ratios=[0.5*nr+0.5,hcc if res is not None else 0.01],hspace=0.28,wspace=0.16,top=0.955,bottom=0.06)
    ax=fig.add_subplot(gs[0,:])
    for k,r in enumerate(rows):
        ax.plot(r['t'],r['d']*0.42+k,color='.55',lw=.55,zorder=1)
        lab=[]
        for phase,colr in (('P','#c1352b'),('S','#1b3fa8')):
            o=r['ph'].get(phase)
            if o is None:continue
            tt=np.arange(o['template'].size)/C.FS+(o['t0']-det);m=(r['t']>=tt[0])&(r['t']<=tt[-1])
            amp=np.abs(r['d'][m]).max() if m.any() else 1.0
            ax.plot(tt,o['template']/(np.abs(o['template']).max() or 1)*amp*0.42+k,color=colr,lw=1.1,zorder=3)
            # the PICK itself: the window starts 0.5 s before the template's pick, so the arrival is at tt[0] + PRE
            ax.plot([tt[0]+PICK_PRE,tt[0]+PICK_PRE],[k-.46,k+.46],color=colr,lw=.8,ls=(0,(3,2)),zorder=5)
            lab.append((phase,colr,o['cc'],tt[-1]))
        for phase,colr,cc,xe in lab:
            ax.text(xe+0.06,k+(0.17 if phase=='P' else -0.17),f'{phase} {cc:.2f}',color=colr,fontsize=7,va='center',ha='left',fontweight='bold',zorder=4)
    ax.set_yticks(range(nr),[f"{r['name'].split('.')[1]}.{r['comp']}" + (f"  {r['km']:.0f} km" if np.isfinite(r['km']) else '') for r in rows],fontsize=7.5)
    ax.set_ylim(-.8,nr-.2);ax.set_xlabel('Seconds from the detection origin')
    ax.set_xlim(tmin,tmax)
    ax.text(.005,1.012,'record (gray) · template P window (red) · template S window (blue) · dashed lines: the template\'s P and S picks · '
            'CC at the end of each window · a row with no red window had no P pick in the template event',transform=ax.transAxes,fontsize=7.5,va='bottom')
    if res is not None:
        axD=fig.add_subplot(gs[1,0]);axZ=fig.add_subplot(gs[1,1])
        i,j,pk=cc_at(res,sh,ti,det_origin,half_s);cc=res['cc'];t=res['t']
        axD.plot(t/3600,cc,color='.35',lw=.4);axD.axhline(res['thr'],color='#2a9d3f',lw=1,label=f"12×MAD = {res['thr']:.2f}")
        axD.axhline(res['floor'],color='#e08a1e',lw=1,ls='--',label=f"floor {res['floor']:.1f}")
        axD.plot(t[j]/3600,cc[j],'v',color='#c1352b',ms=8,zorder=5)
        axD.set(xlabel='Hour of day (UTC)',ylabel='Network CC',xlim=(0,24),ylim=(min(-.1,cc.min()),1.02),title="The detector's view: whole day")
        axD.legend(fontsize=7,loc='upper left',frameon=True,framealpha=1).set_zorder(10)
        hh=int(half_s*C.FS/C.STEP);a,b=max(0,i-hh),min(cc.size,i+hh+1);tt=(np.arange(a,b)-i)*C.STEP/C.FS
        axZ.plot(tt,cc[a:b],color='#1b3fa8',lw=.7);axZ.axhline(res['thr'],color='#2a9d3f',lw=1);axZ.axhline(res['floor'],color='#e08a1e',lw=1,ls='--')
        axZ.plot((j-i)*C.STEP/C.FS,pk,'v',color='#c1352b',ms=8,zorder=5);axZ.text((j-i)*C.STEP/C.FS,pk,f'  CC {pk:.2f}',fontsize=8,va='bottom',color='#c1352b')
        axZ.set(xlabel='Seconds from the detection',xlim=(-half_s,half_s),ylim=(min(-.1,cc[a:b].min()),1.02),title=f'±{half_s:.0f} s')
    fig.suptitle(title,fontsize=10.5,y=0.995)
    if outfile:fig.savefig(outfile,dpi=150,bbox_inches='tight',pad_inches=0.15);plt.close(fig)
    return fig

def plot_full_trace_comparison(sh,ti,det_origin,title,outfile=None,nmax=8,pre=2.0,post=None,tpl_lat=None,tpl_lon=None,scale='signal',SPAN=8.0):
    """Full-waveform comparison: for each station-component, the TEMPLATE event's record (black) and the DETECTION's record
    (red) over the SAME span — from `pre` s before P through the S coda — overlaid after aligning on the template's own P
    (or S where no P pick exists). Unlike the correlation-window views this shows the entire signal, so P, the P-S interval
    and the coda can be compared by eye; the correlation of the full span is printed per row. Rows: nearest stations first.
    `scale='each'` normalises each trace to its own peak (shape comparison); 'template' keeps their true amplitude ratio."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from obspy.geodetics import gps2dist_azimuth
    _use_helvetica();det=UTCDateTime(det_origin);torig=UTCDateTime(str(sh['origin_utc'][ti]));coords=_station_coords()
    stations=[str(x) for x in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']]
    dist=lambda n:(gps2dist_azimuth(tpl_lat,tpl_lon,*coords[n])[0]/1000.0 if (tpl_lat is not None and n in coords) else np.nan)
    # per station-component: window start = the template's own pick minus `pre`; length covers P through the S coda
    by={}
    for si,ph,ci,mo,n,tw,j in channels_of(sh,ti):
        by.setdefault((stations[si],comps[ci]),{})[ph]=dict(si=si,mo=mo,n=n)
    rows=[]
    for (name,comp),d in by.items():
        ref=min(v['mo'] for v in d.values())/C.FS+float(sh['ref_s'][ti])           # first pick of this channel, rel. origin
        end=max(v['mo']+v['n'] for v in d.values())/C.FS+float(sh['ref_s'][ti])
        span=SPAN if SPAN else (end-ref)+(post if post is not None else 3.0)        # one common span for every row
        npts=int(round((span+pre)*C.FS));net,sta=name.split('.')
        a=read_window(net,sta,comp,prefixes[stations.index(name)],torig+ref-pre,npts)   # the template event itself
        b=read_window(net,sta,comp,prefixes[stations.index(name)],det+ref-pre,npts)     # the detection
        if a is None or b is None:continue
        a=a-a.mean();b=b-b.mean()
        w=np.arange(len(a))/C.FS-pre>=-0.2                                      # correlate the signal span, not the lead-in noise
        den=np.sqrt((a[w]**2).sum()*(b[w]**2).sum());cc=float((a[w]*b[w]).sum()/den) if den>0 else np.nan
        rows.append(dict(name=name,comp=comp,km=dist(name),a=a,b=b,cc=cc,t=np.arange(npts)/C.FS-pre,
                         picks=[(p,(v['mo']/C.FS+float(sh['ref_s'][ti]))-ref+0.5) for p,v in d.items()]))
    if not rows:return None
    # SELECTION (disclosed in the caption): the channels that actually carry the match - highest full-trace correlation
    # first - then drawn in distance order so the moveout still reads. Selecting by distance instead would fill the figure
    # with the nearest station, which here (HAWA, 7 km) is also the noisiest.
    order={'Z':0,'N':1,'E':2}
    rows=sorted(rows,key=lambda r:-(r['cc'] if np.isfinite(r['cc']) else -9))[:nmax]
    rows=sorted(rows,key=lambda r:(np.nan_to_num(r['km'],nan=9e9),r['name'],order.get(r['comp'],9)))
    nr=len(rows);fig,axs=plt.subplots(nr,1,figsize=(11,1.15*nr+0.9),sharex=True)
    axs=np.atleast_1d(axs)
    for ax,r in zip(axs,rows):
        # normalise on the SIGNAL portion (from the first pick onward), not on the whole trace: otherwise a noisy row is
        # scaled by its noise and the comparison looks worse than it is
        sig=r['t']>=min(tp for _,tp in r['picks'])-0.2
        na=(np.abs(r['a'][sig]).max() if sig.any() else np.abs(r['a']).max()) or 1
        nb=(np.abs(r['b'][sig]).max() if sig.any() else np.abs(r['b']).max()) or 1
        if scale in ('each','signal'):A,B=r['a']/na,r['b']/nb
        else:A,B=r['a']/na,r['b']/na
        ax.set_ylim(-1.35,1.35)
        ax.plot(r['t'],A,color='k',lw=.8,label='template event' if ax is axs[0] else None)
        ax.plot(r['t'],B,color='#c1352b',lw=.8,alpha=.85,label='detection' if ax is axs[0] else None)
        for p,tp in r['picks']:ax.axvline(tp,color='#c1352b' if p=='P' else '#1b3fa8',lw=.7,ls=(0,(3,2)),zorder=0)
        ax.set_yticks([]);ax.set_ylabel(f"{r['name'].split('.')[1]}.{r['comp']}\n{r['km']:.0f} km" if np.isfinite(r['km']) else f"{r['name'].split('.')[1]}.{r['comp']}",
                      fontsize=7.5,rotation=0,ha='right',va='center')
        ax.text(.995,.88,f"full-trace CC {r['cc']:.2f}",transform=ax.transAxes,ha='right',va='top',fontsize=7.5,color='#c1352b')
    axs[0].legend(fontsize=7.5,loc='upper left',ncol=2,frameon=True,framealpha=1)
    axs[-1].set_xlabel('Seconds from the template pick (dashed: P red, S blue)')
    fig.suptitle(title,fontsize=10,y=0.997);fig.subplots_adjust(top=0.945,bottom=0.075,hspace=0.12,left=0.10,right=0.985)
    if outfile:fig.savefig(outfile,dpi=150,bbox_inches='tight',pad_inches=0.12);plt.close(fig)
    return fig

def plot_full_trace_grid(sh,ti,det_origin,title,outfile=None,nsta=6,pre=2.0,span=8.0,tpl_lat=None,tpl_lon=None):
    """Full-waveform comparison as a STATION x COMPONENT grid: one row per station, columns Z | N | E, template (black) over
    detection (red) on the same span, so all three components of a station are seen together. Stations are chosen by their best
    full-trace correlation and drawn nearest first; a component the template does not carry is left blank."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from obspy.geodetics import gps2dist_azimuth
    _use_helvetica();det=UTCDateTime(det_origin);torig=UTCDateTime(str(sh['origin_utc'][ti]));coords=_station_coords()
    stations=[str(x) for x in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']]
    dist=lambda n:(gps2dist_azimuth(tpl_lat,tpl_lon,*coords[n])[0]/1000.0 if (tpl_lat is not None and n in coords) else np.nan)
    by={}
    for si,ph,ci,mo,n,tw,j in channels_of(sh,ti):by.setdefault(stations[si],{}).setdefault(comps[ci],{})[ph]=dict(mo=mo,n=n)
    data={}
    for name,cc_ in by.items():
        net,sta=name.split('.');ref=min(v['mo'] for c_ in cc_.values() for v in c_.values())/C.FS+float(sh['ref_s'][ti])
        npts=int(round((span+pre)*C.FS));rows={}
        for comp,ph in cc_.items():
            a=read_window(net,sta,comp,prefixes[stations.index(name)],torig+ref-pre,npts)
            b=read_window(net,sta,comp,prefixes[stations.index(name)],det+ref-pre,npts)
            if a is None or b is None:continue
            a=a-a.mean();b=b-b.mean();w=np.arange(len(a))/C.FS-pre>=-0.2
            den=np.sqrt((a[w]**2).sum()*(b[w]**2).sum());r=float((a[w]*b[w]).sum()/den) if den>0 else np.nan
            rows[comp]=dict(a=a,b=b,cc=r,t=np.arange(npts)/C.FS-pre,
                            picks=[(p,(v['mo']/C.FS+float(sh['ref_s'][ti]))-ref+0.5) for p,v in ph.items()])
        if rows:data[name]=dict(rows=rows,km=dist(name),best=np.nanmax([v['cc'] for v in rows.values()]))
    if not data:return None
    keep=sorted(data,key=lambda n:-(data[n]['best'] if np.isfinite(data[n]['best']) else -9))[:nsta]
    keep=sorted(keep,key=lambda n:np.nan_to_num(data[n]['km'],nan=9e9))
    CO=['Z','N','E'];fig,axs=plt.subplots(len(keep),3,figsize=(13,1.05*len(keep)+1.0),sharex=True,squeeze=False)
    for i,name in enumerate(keep):
        for j,comp in enumerate(CO):
            ax=axs[i][j];r=data[name]['rows'].get(comp)
            if r is None:ax.axis('off');continue
            sig=r['t']>=min(tp for _,tp in r['picks'])-0.2
            na=(np.abs(r['a'][sig]).max() if sig.any() else 1) or 1;nb=(np.abs(r['b'][sig]).max() if sig.any() else 1) or 1
            ax.plot(r['t'],r['a']/na,color='k',lw=.7);ax.plot(r['t'],r['b']/nb,color='#c1352b',lw=.7,alpha=.85)
            for p,tp in r['picks']:ax.axvline(tp,color='#c1352b' if p=='P' else '#1b3fa8',lw=.7,ls=(0,(3,2)),zorder=0)
            ax.set_yticks([]);ax.set_ylim(-1.3,1.3)
            ax.text(.985,.9,f"{r['cc']:.2f}",transform=ax.transAxes,ha='right',va='top',fontsize=7.5,color='#c1352b',fontweight='bold')
            if i==0:ax.set_title(f'{comp} component',fontsize=9)
        # label the row on whichever panel of it exists (a station may lack Z, or lack N/E)
        lab=f"{name.split('.')[1]}\n{data[name]['km']:.0f} km" if np.isfinite(data[name]['km']) else name.split('.')[1]
        first=next((axs[i][j] for j,c_ in enumerate(CO) if c_ in data[name]['rows']),None)
        if first is not None:first.set_ylabel(lab,fontsize=8,rotation=0,ha='right',va='center');first.yaxis.set_label_coords(-0.06,0.5)
    for j,c_ in enumerate(CO):                                  # x label on the lowest panel that exists in each column
        last=next((axs[i][j] for i in range(len(keep)-1,-1,-1) if c_ in data[keep[i]]['rows']),None)
        if last is not None:last.set_xlabel('Seconds from the template pick');last.tick_params(labelbottom=True)
    axs[0][0].plot([],[],color='k',lw=.8,label='template event');axs[0][0].plot([],[],color='#c1352b',lw=.8,label='detection')
    axs[0][0].legend(fontsize=7,loc='lower left',frameon=True,framealpha=1,ncol=2)
    fig.suptitle(title,fontsize=10,y=0.997);fig.subplots_adjust(top=0.93,bottom=0.09,hspace=0.14,wspace=0.06,left=0.075,right=0.99)
    if outfile:fig.savefig(outfile,dpi=150,bbox_inches='tight',pad_inches=0.12);plt.close(fig)
    return fig

def template_cc_day(sh,ti,year,doy,day=None):
    """Network-CC trace of one template over one day, summed as shelly.scan_day does (P pass on Z, S pass on Z at S-on-Z stations,
    on N/E elsewhere, scan-day weights), median-subtracted, with the whole-day MAD_K x MAD threshold and the absolute floor.
    Returns None when the template has no live channel that day. `day` = (data, present, day0) from scan.load_day, loaded once
    per day by the caller; loaded here otherwise."""
    import fast_matched_filter as fmf
    stations=[str(s) for s in sh['stations']];prefixes=[str(p) for p in sh['channels']];comps=[str(c) for c in sh['components']];ci_N,ci_E,ci_Z=[comps.index(c) for c in 'NEZ']
    data,present,day0=day if day is not None else scan.load_day(stations,prefixes,year,doy)
    if not present.any():return None
    Pw,Sw=day_weights(sh,present,day0,ti)
    if Pw.sum()+Sw.sum()<=0:return None
    s_comp=[str(x) for x in sh['s_comp']];nS=len(stations)
    ip=np.array([i for i in range(nS) if present[i]],int);iz=np.array([i for i in range(nS) if s_comp[i]=='Z' and present[i]],int);ih=np.array([i for i in range(nS) if s_comp[i]!='Z' and present[i]],int)
    N=data.shape[-1];nP=int(LP+sh['P_moveouts'].max());nQ=int(LS+sh['S_moveouts'].max());n_corr=N-max(nP,nQ)+1;sl=slice(ti,ti+1)
    cc=fmf.matched_filter(sh['P_waveforms'][sl][:,ip],sh['P_moveouts'][sl][:,ip],Pw[None][:,ip],np.ascontiguousarray(data[ip][:,[ci_Z],:]),C.STEP,arch='gpu',normalize='short')
    cc=np.nan_to_num(cc[0]).astype(np.float64);Lc=min(cc.size,n_corr)
    if iz.size:
        s=fmf.matched_filter(sh['S_waveforms'][sl][:,iz,:1,:],sh['S_moveouts'][sl][:,iz,:1],Sw[None][:,iz,:1],np.ascontiguousarray(data[iz][:,[ci_Z],:]),C.STEP,arch='gpu',normalize='short');s=np.nan_to_num(s[0]);Lc=min(Lc,s.size);cc[:Lc]+=s[:Lc]
    if ih.size:
        s=fmf.matched_filter(sh['S_waveforms'][sl][:,ih,:,:],sh['S_moveouts'][sl][:,ih,:],Sw[None][:,ih,:],np.ascontiguousarray(data[ih][:,[ci_N,ci_E],:]),C.STEP,arch='gpu',normalize='short');s=np.nan_to_num(s[0]);Lc=min(Lc,s.size);cc[:Lc]+=s[:Lc]
    cc=cc[:Lc];k=max(1,int(getattr(C,'MAD_SUBSAMPLE',1)));med=float(np.median(cc[::k]));cc-=med;mad=float(np.median(np.abs(cc[::k])))
    return dict(cc=cc,t=np.arange(Lc)*C.STEP/C.FS,day0=day0,median=med,mad=mad,thr=C.MAD_K*mad,floor=C.ABS_CC_FLOOR,n_present=int(((Pw[:,0]>0)|(Sw>0).any(1)).sum()),n_channels=int((Pw>0).sum()+(Sw>0).sum()))

def cc_at(res,sh,ti,det_origin,half_s=12.0,peak_s=0.1):
    """Sample index of the detection in the day trace and the trace's maximum within ±peak_s of it (the detection's own CC;
    a wider search can land on a neighbouring event). half_s is kept for the callers' plotting window."""
    det=UTCDateTime(det_origin);i=int(round(((det-res['day0'])+float(sh['ref_s'][ti]))*C.FS/C.STEP));h=max(1,int(round(peak_s*C.FS/C.STEP)));a,b=max(0,i-h),min(res['cc'].size,i+h+1)
    seg=res['cc'][a:b];j=a+int(np.argmax(seg)) if seg.size else i
    return i,j,float(res['cc'][j]) if seg.size else np.nan

def plot_cc_trace(res,sh,ti,det_origin,title,outfile=None,half_s=12.0):
    """Full-day network CC (median-subtracted) with the whole-day 12xMAD threshold and the absolute floor, and a ±half_s window
    around the detection (the Buan figure). The catalog used per-segment thresholds; the whole-day line is the context."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;_use_helvetica()
    i,j,ccpk=cc_at(res,sh,ti,det_origin,half_s);t=res['t'];cc=res['cc']
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,3.1),gridspec_kw={'width_ratios':[2.2,1]},layout='constrained')
    a1.plot(t/3600,cc,color='.35',lw=.4);a1.axhline(res['thr'],color='#2a9d3f',lw=1,label=f"12×MAD = {res['thr']:.2f} (whole day)");a1.axhline(res['floor'],color='#e08a1e',lw=1,ls='--',label=f"floor {res['floor']:.1f}")
    a1.plot(t[j]/3600,cc[j],'v',color='#c1352b',ms=8,zorder=5);a1.set(xlabel='Hour of day (UTC)',ylabel='Network CC (median-subtracted)',xlim=(0,24),ylim=(min(-.1,cc.min()),1.02),title='Full day')
    a1.legend(loc='upper left',fontsize=7,frameon=True,framealpha=1).set_zorder(10)
    h=int(half_s*C.FS/C.STEP);a,b=max(0,i-h),min(cc.size,i+h+1);tt=(np.arange(a,b)-i)*C.STEP/C.FS
    a2.plot(tt,cc[a:b],color='#1b3fa8',lw=.7);a2.axhline(res['thr'],color='#2a9d3f',lw=1);a2.axhline(res['floor'],color='#e08a1e',lw=1,ls='--');a2.plot((j-i)*C.STEP/C.FS,ccpk,'v',color='#c1352b',ms=8,zorder=5)
    a2.text((j-i)*C.STEP/C.FS,ccpk,f'  CC {ccpk:.2f}',fontsize=8,va='bottom',ha='left',color='#c1352b');a2.set(xlabel='Time from the detection (s)',xlim=(-half_s,half_s),ylim=(min(-.1,cc[a:b].min()),1.02),title=f'±{half_s:.0f} s')
    fig.suptitle(title,fontsize=9)
    if outfile:fig.savefig(outfile,dpi=160,bbox_inches='tight');plt.close(fig)
    return fig,(i,j,ccpk)
