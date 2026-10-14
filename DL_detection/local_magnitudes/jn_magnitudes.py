"""Add measured JU.JN01/JN08/JN09 amplitudes to the existing Buan-method catalog.

IMSIL PORT: paths only. No JU waveforms exist for the Imsil period, so this module contributes no station (n_JN = 0).
"""
from pathlib import Path
import sys,json,hashlib,warnings
import numpy as np
import pandas as pd
import obspy
ROOT=Path('/home/msseo/works/72.2025_Imsil_swarm/DL_detection')   # IMSIL PORT
sys.path.insert(0,str(ROOT/'local_magnitudes'))
import haenam_magnitudes as H
XML=ROOT.parent/'data/stations/JU_station.xml'   # IMSIL PORT (copied from moho by tools/phase0_copy_from_moho.sh)
JN=['JN01','JN08','JN09']
INV=None;TAUP=None

def wa_trace(tr,inv):
    # Same signal processing as Buan _wa_trace, but honor the FULL SEED ID, including location.
    inv.get_response(tr.id,tr.stats.starttime)
    w=tr.copy().detrend('demean').detrend('linear');w.taper(.05)
    w.filter('bandpass',**H.S.BANDPASS)
    w.remove_response(inventory=inv,output='DISP',water_level=60.0,plot=False)
    w.simulate(paz_simulate=H.S.WOOD_ANDERSON_PAZ);w.data=np.asarray(w.data,dtype=float)*1000
    return w

def init():
    global INV,TAUP
    INV=obspy.read_inventory(str(XML))
    from obspy.taup import TauPyModel
    TAUP=TauPyModel(model=str(H.BUAN/'taup/kim1983_buan.npz'))

def event(row):
    eid=row['eid'];components=[];stations=[];sources=[]
    for sta in JN:
        paths=sorted((H.WF/eid).glob('*.JU.'+sta+'.*.sac'))
        if not paths:continue
        trs=[obspy.read(str(p))[0] for p in paths]
        R=float(trs[0].stats.sac.dist)
        p=next((float(t.stats.sac.a) for t in trs if 'a' in t.stats.sac and t.stats.sac.a>-1e4),None)
        source='pick' if p is not None else 'taup'
        if p is None:p=H.S.theoretical_p_rel(R,row['depth_km'],0,TAUP)
        cc=[]
        for path,tr in zip(paths,trs):
            sources.append(dict(eid=eid,path=str(path),sha256=H.sha(path)))
            r=dict(eid=eid,network='JU',station=sta,trace_id=tr.id,file=path.name,component=tr.stats.channel[-1],
                   dist_km=R,p_source=source,p_origin_s=p,b_s=float(tr.stats.sac.b),sampling_rate=float(tr.stats.sampling_rate))
            small=INV.select(network='JU',station=sta,location=tr.stats.location,channel=tr.stats.channel,time=tr.stats.starttime)
            channels=[ch for net in small for st in net for ch in st]
            if len(channels)!=1:
                r['status']='response_missing_or_ambiguous';components.append(r);continue
            channel=channels[0]
            if not np.isclose(channel.sample_rate,tr.stats.sampling_rate,rtol=1e-5):
                r['status']='sample_rate_mismatch';components.append(r);continue
            resp=channel.response;s=resp.instrument_sensitivity
            native=abs(resp.get_evalresp_response_for_frequencies([s.frequency],output='DEF',hide_sensitivity_mismatch_warning=True)[0])
            r.update(response_input_units=s.input_units,response_sensitivity=s.value,stages_over_sensitivity=native/s.value,
                     response_start=str(channel.start_date),response_end=str(channel.end_date),azimuth=channel.azimuth,dip=channel.dip)
            if not np.isfinite(tr.data).all() or not np.isfinite(p):
                r['status']='invalid_waveform_or_P';components.append(r);continue
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always');wa=wa_trace(tr,small)
            r['warnings']=' | '.join(sorted(set(str(w.message) for w in ww)))
            rt=float(np.hypot(R,row['depth_km']));amp,noise,snr=H.window_stats(wa,p,rt,r['b_s'])
            r.update(status='processed',peak_mm=amp,noise_peak_mm=noise,snr_corrected=snr,
                     signal_start_s=rt/4,signal_end_s=rt/2,noise_start_s=p-6,noise_end_s=p-1,
                     signal_full=(rt/4>=r['b_s'] and rt/2<=r['b_s']+(wa.stats.npts-1)/wa.stats.sampling_rate),
                     noise_full=(p-6>=r['b_s'] and p-1<=r['b_s']+(wa.stats.npts-1)/wa.stats.sampling_rate))
            cc.append(r);components.append(r)
        ss=H.assemble_station('JU',sta,R,source,cc,'corrected')
        if ss:stations.append(dict(eid=eid,**ss))
    return components,stations,sources

def run(out,workers=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);init()
    base=pd.read_csv(ROOT/'local_magnitudes/haenam_local_magnitudes.csv',dtype={'eid':str},parse_dates=['t'])
    old=pd.read_csv(ROOT/'local_magnitudes/haenam_ml_per_station.csv',dtype={'eid':str});old=old[old['mode'].eq('corrected')]
    from multiprocessing import get_context
    cs=[];ss=[];sources=[]
    with get_context('fork').Pool(workers) as pool:
        for i,(c,s,f) in enumerate(pool.imap_unordered(event,base.to_dict('records')),1):
            cs.extend(c);ss.extend(s);sources.extend(f)
            if i%100==0 or i==len(base):print('JN measured',i,'/',len(base),flush=True)
    new=pd.DataFrame(ss);combined=pd.concat([old,new],ignore_index=True)
    assert not combined.duplicated(['eid','network','station']).any()
    rows=[]
    for eid,g in combined.groupby('eid'):
        agg=H.S.event_ml(g);jg=g[g.network.eq('JU')];hvalid=g.ml_h.notna()&g.p_source.eq('pick')
        rows.append(dict(eid=eid,ml_with_JN=agg['ml'],ml_std_with_JN=agg['ml_std'],n_pick_with_JN=agg['n_pick'],n_sta_with_JN=agg['n_sta'],
                         n_h_with_JN=agg['n_h'],n_v_with_JN=agg['n_v'],n_JN=len(jg),n_JN_picked=int(jg.p_source.eq('pick').sum()),
                         ml_horizontal_with_JN=float(g.ml_h.median()) if hvalid.sum()>=3 else np.nan))
    cat=base.merge(pd.DataFrame(rows),on='eid',how='left',validate='one_to_one')
    for col in ['n_pick_with_JN','n_sta_with_JN','n_h_with_JN','n_v_with_JN','n_JN','n_JN_picked']:cat[col]=cat[col].fillna(0).astype(int)
    assert cat.ml_with_JN.notna().equals(cat.n_pick_with_JN.ge(3))
    assert (cat.n_pick_with_JN>=cat.n_pick).all()
    cat['newly_measured']=cat.ml.isna()&cat.ml_with_JN.notna();cat['delta_ML_JN']=cat.ml_with_JN-cat.ml
    comp=pd.DataFrame(cs);manifest=pd.DataFrame(sources)
    for name,d in [('haenam_ML_with_JN',cat),('JN_component_measurements',comp),('JN_station_magnitudes',new),('combined_station_magnitudes',combined),('JN_waveform_hashes',manifest)]:
        d.to_csv(out/(name+'.csv'),index=False)
    files=[Path(__file__),XML,Path(H.__file__),Path(H.S.__file__),ROOT/'local_magnitudes/haenam_local_magnitudes.csv',ROOT/'local_magnitudes/haenam_ml_per_station.csv']
    (out/'provenance.json').write_text(json.dumps(dict(method='Buan processing, corrected P convention; add JU.JN01/JN08/JN09 using full SEED ID',
       changes='Network inclusion and location-code matching only; no SNR/station gate relaxation or magnitude calibration to KMA.',obspy=obspy.__version__,
       inputs=[dict(path=str(p),sha256=H.sha(p)) for p in files]),indent=2))
    print(cat.groupby('cls').agg(n=('eid','size'),old=('ml','count'),with_JN=('ml_with_JN','count'),added=('newly_measured','sum')).to_string(),flush=True)
    return cat,comp,combined

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.out,a.workers)
