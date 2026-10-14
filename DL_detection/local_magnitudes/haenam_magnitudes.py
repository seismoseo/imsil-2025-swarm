"""Buan-compatible Sheen ML measurements with explicit fallback timing audit.

IMSIL PORT: ROOT, registry, BUAN (= 72.2025_Imsil_swarm/data/stations: responses/ and taup/) and WF paths, and the
response-epoch window of run() changed; the measurement code is unchanged. The module name is kept because
full_magnitudes.py / validate_full_magnitudes.py import it. Imsil uses full_period/full_magnitudes.py.

Reads original SAC counts. Outputs only to the caller's output directory.
The copied Buan engine is unmodified; its source hash is recorded.
"""
from pathlib import Path
import hashlib,json,os,warnings,sys,time
import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime
from obspy.core.inventory import Inventory
import buan_sheen2018_snapshot as S

ROOT=Path('/home/msseo/works/72.2025_Imsil_swarm/DL_detection')   # IMSIL PORT
sys.path.insert(0, "/home/msseo/works/72.2025_Imsil_swarm/registry"); import imsil as haenam   # registry paths (IMSIL PORT)
BUAN=Path('/home/msseo/works/72.2025_Imsil_swarm/data/stations')   # IMSIL PORT: responses/ (master XML + KMA RESP) and taup/ (tools/phase0_copy_from_moho.sh)
WF=ROOT/'runs/hypodd_dl/waveforms_100km'
HERE=Path(__file__).resolve().parent
INV=None;TAUP=None;OUT=None
VERSION='haenam-ml-1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def catalog():
    d=pd.read_csv(ROOT/'catalogs/haenam_dl_relocated_canonical_frame.csv',comment='#',dtype={'eid':str,'template_match':str},parse_dates=['t'])
    t=pd.read_csv(Path(haenam.path(None,'tm_templates_195')),dtype={'event_id':str})
    m=d.merge(t[['event_id','ML','origin_time_UTC']],left_on='template_match',right_on='event_id',how='left',validate='many_to_one')
    matched=m.ML.notna();assert m.loc[matched,'template_match'].is_unique
    assert np.allclose(m.loc[matched,'ML'],m.loc[matched,'ML_template'])
    m['kma_dt_s']=(m.t-pd.to_datetime(m.origin_time_UTC)).dt.total_seconds()
    assert m.loc[matched,'kma_dt_s'].abs().max()<=2
    return m.rename(columns={'ML':'ml_kma'}).sort_values('t').reset_index(drop=True)

def initialize(out):
    global INV,TAUP,OUT
    OUT=Path(out);OUT.mkdir(parents=True,exist_ok=True);(OUT/'event_cache').mkdir(exist_ok=True)
    ip=OUT/'responses_buan_subset.xml'
    if ip.exists():
        INV=obspy.read_inventory(str(ip))
    else:
        big=S.load_inventory(str(BUAN/'responses/master'),str(BUAN/'responses/fetched'))
        stations=pd.read_csv(ROOT/'station_table/stations.csv')
        INV=Inventory(networks=[],source='Buan KS/KG metadata, selected for Haenam stations; no epoch extension')
        for r in stations[stations.Network.isin(['KS','KG'])].itertuples():
            INV+=big.select(network=r.Network,station=r.Code,starttime=UTCDateTime('2025-09-01'),endtime=UTCDateTime('2026-02-01'))   # IMSIL PORT: period
        INV.write(str(ip),format='STATIONXML')
    from obspy.taup import TauPyModel
    model=BUAN/'taup/kim1983_buan.npz'
    TAUP=TauPyModel(model=str(model))
    S._KIM1983_TAUP=TAUP
    return INV

def window_stats(w,p,Rtime,b):
    # Exactly the same peak and noise extraction as the Buan implementation.
    amp,rms,noise=S._peak_in_window(w,p,Rtime,b)
    snr=amp/noise if np.isfinite(noise) and noise>0 else np.nan
    return amp,noise,snr

def assemble_station(net,sta,R,src,components,mode):
    peaks={r['component']:r['peak_mm'] for r in components
           if r.get('status')=='processed' and r['snr_'+mode]>=S.SNR_MIN}
    hz=np.sqrt(peaks['N']*peaks['E']) if 'N' in peaks and 'E' in peaks else np.nan
    zz=peaks.get('Z',np.nan)
    if not (np.isfinite(hz) or np.isfinite(zz)):return None
    return dict(network=net,station=sta,dist_km=R,p_source=src,A_h_mm=hz,A_z_mm=zz,
                ml_h=S.ml_horizontal(hz,R) if np.isfinite(hz) else np.nan,
                ml_v=S.ml_vertical(zz,R) if np.isfinite(zz) else np.nan,mode=mode)

def measure_one(row):
    eid=row['eid'];start=time.monotonic();ep=WF/eid
    cache=OUT/'event_cache'/f'{eid}.json'
    files=sorted(ep.glob('*.sac'))
    signature=hashlib.sha256(json.dumps([(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in files]).encode()).hexdigest()
    if cache.exists():
        data=json.loads(cache.read_text())
        if data.get('version')==VERSION and data.get('waveform_signature')==signature and data.get('depth_km')==row['depth_km']:
            return data
    by={};components=[];stations=[]
    for p in files:
        # Match Buan: temporary JU response is deliberately outside this comparison.
        h=obspy.read(str(p),headonly=True)[0]
        if h.stats.network not in ['KS','KG']:
            components.append(dict(eid=eid,trace_id=h.id,status='network_excluded_buan',file=p.name));continue
        tr=obspy.read(str(p))[0]
        by.setdefault((tr.stats.network,tr.stats.station),[]).append((p,tr))
    for (net,sta),trs in by.items():
        R=next((float(tr.stats.sac.dist) for _,tr in trs if 'dist' in tr.stats.sac),np.nan)
        p=next((float(tr.stats.sac.a) for _,tr in trs if 'a' in tr.stats.sac and tr.stats.sac.a>-1e4),None)
        b0=float(trs[0][1].stats.sac.get('b',0))
        src='pick' if p is not None else 'taup'
        if p is None:
            # Origin-relative P; legacy engine subtracts b here and again in _peak_in_window.
            p=S.theoretical_p_rel(R,row['depth_km'],0.0,TAUP)
        pleg=p if src=='pick' else p-b0
        small=INV.select(network=net,station=sta)
        cc=[]
        for fp,tr in trs:
            b=float(tr.stats.sac.get('b',0));rr=dict(eid=eid,trace_id=tr.id,network=net,station=sta,
                    component=tr.stats.channel[-1],file=fp.name,dist_km=R,p_source=src,b_s=b,p_origin_s=p,
                    p_legacy_argument_s=pleg,sampling_rate=float(tr.stats.sampling_rate))
            if not np.isfinite(R) or R<=0 or not np.isfinite(p):
                rr['status']='invalid_distance_or_P';components.append(rr);continue
            try:
                response=small.get_response(tr.id,tr.stats.starttime)
            except Exception:
                rr['status']='no_response_at_trace_epoch';components.append(rr);continue
            if not np.isfinite(tr.data).all():
                rr['status']='nonfinite_waveform';components.append(rr);continue
            with warnings.catch_warnings(record=True) as warn:
                warnings.simplefilter('always');wa=S._wa_trace(tr,small)
            rr['response_warnings']=' | '.join(sorted(set(str(w.message) for w in warn)))
            if wa is None:
                rr['status']='response_failure';components.append(rr);continue
            rt=float(np.hypot(R,row['depth_km']))
            a,n,snr=window_stats(wa,p,rt,b);al,nl,sl=window_stats(wa,pleg,rt,b)
            assert np.isclose(a,al,equal_nan=True)
            rr.update(status='processed',peak_mm=a,noise_peak_mm=n,noise_legacy_mm=nl,
                      snr_corrected=snr,snr_legacy=sl,signal_start_s=rt/4,signal_end_s=rt/2,
                      noise_start_s=p-6,noise_end_s=p-1,noise_legacy_start_s=pleg-6,noise_legacy_end_s=pleg-1,
                      signal_full=(rt/4>=b and rt/2<=b+(wa.stats.npts-1)/wa.stats.sampling_rate),
                      noise_full=(p-6>=b and p-1<=b+(wa.stats.npts-1)/wa.stats.sampling_rate),
                      response_input_units=response.instrument_sensitivity.input_units,
                      response_sensitivity=response.instrument_sensitivity.value)
            # Diagnostic only: a repeated extreme may indicate clipping, but digitizer rails are unknown.
            raw=np.asarray(tr.data);rr['max_identical_extreme_samples']=int(max((raw==raw.max()).sum(),(raw==raw.min()).sum()))
            cc.append(rr);components.append(rr)
        for mode in ['corrected','legacy']:
            ss=assemble_station(net,sta,R,src,cc,mode)
            if ss is not None:stations.append(dict(eid=eid,**ss))
    ag={}
    for mode in ['corrected','legacy']:
        q=pd.DataFrame([r for r in stations if r['mode']==mode]);a=S.event_ml(q)
        if len(q):
            hp=q.ml_h.notna()&q.p_source.eq('pick')
            a['n_h_pick']=int(hp.sum());a['ml_h_strict']=float(q.ml_h.median()) if hp.sum()>=S.MIN_STA else np.nan
        else:a.update(n_h_pick=0,ml_h_strict=np.nan)
        ag[mode]=a
    data=dict(version=VERSION,eid=eid,depth_km=row['depth_km'],waveform_signature=signature,
              waveform_files=len(files),event=ag,stations=stations,components=components,seconds=time.monotonic()-start)
    cache.write_text(json.dumps(data,allow_nan=True))
    return data

def run(out,workers=6,limit=None):
    initialize(out);cat=catalog()
    if limit:cat=cat.head(limit)
    from multiprocessing import get_context
    rows=[];st=[];co=[];t=time.monotonic()
    with get_context('fork').Pool(workers) as pool:
        for i,r in enumerate(pool.imap_unordered(measure_one,cat.to_dict('records')),1):
            a=r['event']['corrected'];b=r['event']['legacy']
            rows.append(dict(eid=r['eid'],**a,ml_buan_exact=b['ml'],n_pick_buan_exact=b['n_pick'],
                             n_sta_buan_exact=b['n_sta'],ml_std_buan_exact=b['ml_std'],waveform_signature=r['waveform_signature'],
                             waveform_files=r['waveform_files'],status='measured' if np.isfinite(a['ml']) else 'insufficient_picked_stations'))
            st.extend(r['stations']);co.extend(r['components'])
            if i%20==0 or i==len(cat):print(f'{i}/{len(cat)} events; {time.monotonic()-t:.0f} s; measured {sum(np.isfinite(x["ml"]) for x in rows)}',flush=True)
    ev=cat.merge(pd.DataFrame(rows),on='eid',validate='one_to_one');sta=pd.DataFrame(st);comp=pd.DataFrame(co)
    for name,df in [('haenam_local_magnitudes',ev),('haenam_ml_per_station',sta),('haenam_ml_per_component',comp)]:df.to_csv(OUT/(name+'.csv'),index=False)
    inputs=[Path(S.__file__),Path(__file__),BUAN/'responses/master/KS_KG_metadata_1.0.2.xml',BUAN/'taup/kim1983_buan.npz',
            ROOT/'catalogs/haenam_dl_relocated_canonical_frame.csv',Path(haenam.path(None,'tm_templates_195')),ROOT/'velocity_model/kim1983.csv']
    inputs+=list((BUAN/'responses/fetched/extracted').glob('RESP.*'))
    prov=dict(method='Buan Sheen 2018, current executable engine',primary='corrected origin-relative theoretical P',comparison='literal Buan fallback behavior',
              workers=workers,n_events=len(ev),obspy=obspy.__version__,numpy=np.__version__,pandas=pd.__version__,
              inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],response_subset_sha256=sha(OUT/'responses_buan_subset.xml'),
              waveform_provenance='Per event SHA256 of sorted (filename,size,mtime_ns), not a content hash; original SACs remain in place.')
    (OUT/'provenance_ml.json').write_text(json.dumps(prov,indent=2))
    print(ev.groupby('cls').ml.agg(['size','count','min','median','max']).to_string(),flush=True)
    return ev,sta,comp

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.out,a.workers,a.limit)
