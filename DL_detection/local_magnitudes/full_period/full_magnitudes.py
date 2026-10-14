"""Full-period Buan-method ML; reads original SAC files in place.

IMSIL PORT: ROOT / WF (the Imsil relocation run runs/hypodd_dl) / catalog (catalogs/imsil_dl_relocated.csv) / response
epoch window (2025-09-01 .. 2026-02-01) changed, plus one switch for decision D4, env IMSIL_RESPONSE_MODE:
  master_first (default) = the literal Haenam loader: master StationXML first, KMA RESP appended; where epochs overlap
                           ObsPy uses the first match (the master), i.e. RESP only fills gaps (e.g. SIDA)
  resp_current           = the KMA RESP epochs replace the master epochs from each RESP start date onward
Use a separate --out folder per mode (the inventory cache responses_full_period.xml lives in --out).


Preserves the previous Haenam KS/KG engine plus full-SEED-ID JU response handling,
SAC epicentral distances, new joint-run depth, corrected theoretical-P timing,
SNR>=2, horizontal preference, vertical fallback, and >=3 picked stations.
"""
from pathlib import Path
import sys,json,hashlib,time,os,traceback
import numpy as np
import pandas as pd
import obspy
ROOT=Path('/home/msseo/works/72.2025_Imsil_swarm/DL_detection')   # IMSIL PORT
sys.path.insert(0,str(ROOT/'local_magnitudes'))
import haenam_magnitudes as H
import jn_magnitudes as J
S=H.S
WF=ROOT/'runs/hypodd_dl/waveforms_100km'   # IMSIL PORT: the Imsil relocation run root
CATALOG=ROOT/'catalogs/imsil_dl_relocated.csv'   # IMSIL PORT
RESPONSE_MODE=os.environ.get('IMSIL_RESPONSE_MODE','master_first')   # IMSIL PORT (D4)
OUT=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def initialize(out):
    global OUT
    OUT=Path(out);OUT.mkdir(parents=True,exist_ok=True);(OUT/'event_cache').mkdir(exist_ok=True);(OUT/'combined_cache').mkdir(exist_ok=True)
    invpath=OUT/'responses_full_period.xml'
    if not invpath.exists():
        print('Loading full-epoch Buan metadata',flush=True)
        if RESPONSE_MODE=='master_first':big=S.load_inventory(str(H.BUAN/'responses/master'),str(H.BUAN/'responses/fetched'))
        elif RESPONSE_MODE=='resp_current':big=load_inventory_resp_current(str(H.BUAN/'responses/master'),str(H.BUAN/'responses/fetched'))
        else:raise SystemExit(f'IMSIL_RESPONSE_MODE={RESPONSE_MODE!r}: use master_first or resp_current')
        print('Response mode (D4):',RESPONSE_MODE,flush=True)
        from obspy.core.inventory import Inventory
        inv=Inventory(networks=[],source='Existing Buan metadata, Haenam stations, original response epochs retained')
        stations=pd.read_csv(ROOT/'station_table/stations.csv')
        for r in stations[stations.Network.isin(['KS','KG'])].itertuples():
            inv+=big.select(network=r.Network,station=r.Code,starttime=obspy.UTCDateTime('2025-09-01'),endtime=obspy.UTCDateTime('2026-02-01'))   # IMSIL PORT: period
        inv.write(str(invpath),format='STATIONXML')
        H.INV=inv
    else:H.INV=obspy.read_inventory(str(invpath))
    from obspy.taup import TauPyModel
    H.TAUP=TauPyModel(model=str(H.BUAN/'taup/kim1983_buan.npz'));S._KIM1983_TAUP=H.TAUP
    H.WF=WF;H.OUT=OUT
    # Cache invalidates on code, inventory or method changes as well as waveform stat signatures/depth.
    H.VERSION='full-ml-v1-'+sha(invpath)[:12]+'-'+sha(__file__)[:12]
    J.INV=obspy.read_inventory(str(J.XML));J.TAUP=H.TAUP;J.JN=sorted({sta.code for net in J.INV for sta in net if sta.code.startswith('JN')})
    print('Response inventory ready; JU stations',J.JN,flush=True)


def load_inventory_resp_current(master_dir,fetched_dir):
    """IMSIL PORT (D4 option resp_current): master StationXML + KMA RESP, the RESP epochs replacing the master epochs of the
    same channel from the RESP start date onward (master epochs are cut one second before it; later master epochs dropped)."""
    import glob,warnings
    from obspy.core.inventory import Inventory
    master=obspy.read_inventory(os.path.join(master_dir,'*.xml'));resp=Inventory(networks=[],source='KMA RESP')
    for f in sorted(glob.glob(os.path.join(fetched_dir,'extracted','RESP.*'))):
        try:resp+=obspy.read_inventory(f)
        except Exception as exc:warnings.warn(f'load {f}: {exc}',RuntimeWarning)
    starts={}
    for net in resp:
        for sta in net:
            for ch in sta:
                k=(net.code,sta.code,ch.location_code,ch.code);starts[k]=min(starts.get(k,ch.start_date),ch.start_date)
    n_cut=n_drop=0
    for net in master:
        for sta in net:
            keep=[]
            for ch in sta.channels:
                t0=starts.get((net.code,sta.code,ch.location_code,ch.code))
                if t0 is None:keep.append(ch);continue
                if ch.start_date>=t0:n_drop+=1;continue
                if ch.end_date is None or ch.end_date>t0:ch.end_date=t0-1;n_cut+=1
                keep.append(ch)
            sta.channels=keep
    print(f'resp_current: {len(starts)} RESP channels; master epochs cut {n_cut}, dropped {n_drop}',flush=True)
    return master+resp


def event(row):
    p=OUT/'combined_cache'/(row['eid']+'.json')
    old=H.measure_one(row)
    version=H.VERSION+'-'+sha(J.XML)[:12]
    if p.exists():
        cached=json.loads(p.read_text())
        if cached.get('version')==version and cached.get('waveform_signature')==old['waveform_signature'] and cached.get('depth_km')==row['depth_km']:return cached
    # JN event uses the same H.WF/H helpers and full SEED ID response epochs.
    co,st,files=J.event(row)
    stations=[r for r in old['stations'] if r['mode']=='corrected']+st
    components=[r for r in old['components'] if r.get('status')!='network_excluded_buan']+co
    tab=pd.DataFrame(stations)
    if len(tab):assert not tab.duplicated(['network','station']).any()
    agg=S.event_ml(tab)
    if len(tab):
        mlsta=tab.ml_h.where(tab.ml_h.notna(),tab.ml_v);med=float(mlsta.median());spread=float(mlsta.std()) if len(mlsta)>1 else np.nan
        hp=tab.ml_h.notna()&tab.p_source.eq('pick')
        agg.update(ml_min1=med if agg['n_pick']>=1 else np.nan,ml_min2=med if agg['n_pick']>=2 else np.nan,ml_min3=agg['ml'],station_scatter=spread,ml_h_strict=float(tab.ml_h.median()) if hp.sum()>=3 else np.nan,n_JN=int(tab.network.eq('JU').sum()),n_JN_picked=int((tab.network.eq('JU')&tab.p_source.eq('pick')).sum()))
    else:agg.update(ml_min1=np.nan,ml_min2=np.nan,ml_min3=np.nan,station_scatter=np.nan,ml_h_strict=np.nan,n_JN=0,n_JN_picked=0)
    result=dict(version=version,eid=row['eid'],depth_km=row['depth_km'],waveform_signature=old['waveform_signature'],waveform_files=old['waveform_files'],event=agg,stations=stations,components=components,JN_sources=files,ml_KS_KG=old['event']['corrected']['ml'])
    p.write_text(json.dumps(result,allow_nan=True));return result


def run(out,workers=4,limit=None):
    initialize(out)
    cat=pd.read_csv(CATALOG,dtype={'event_id':str,'template_match':str});cat['eid']=cat.event_id   # IMSIL PORT
    if limit:cat=cat.head(limit)
    from multiprocessing import get_context
    events=[];stations=[];components=[];jn=[];start=time.monotonic()
    with get_context('fork').Pool(workers) as pool:
        for i,r in enumerate(pool.imap_unordered(event,cat.to_dict('records')),1):
            events.append(dict(eid=r['eid'],**r['event'],ml_KS_KG=r['ml_KS_KG'],waveform_signature=r['waveform_signature'],waveform_files=r['waveform_files']))
            stations.extend(r['stations']);components.extend(r['components']);jn.extend(r['JN_sources'])
            if i%25==0 or i==len(cat):print(f'{i}/{len(cat)} events, {time.monotonic()-start:.0f}s; ML3 {sum(np.isfinite(x["ml_min3"]) for x in events)}, ML1 {sum(np.isfinite(x["ml_min1"]) for x in events)}',flush=True)
    ev=cat.merge(pd.DataFrame(events),on='eid',validate='one_to_one');co=pd.DataFrame(components);st=pd.DataFrame(stations)
    for name,df in [('full_period_ML',ev),('station_magnitudes',st),('component_measurements',co),('JN_waveform_hashes',pd.DataFrame(jn))]:df.to_csv(OUT/(name+'.csv'),index=False)
    for n in [1,2,3]:assert ev['ml_min'+str(n)].notna().equals(ev.n_pick>=n)
    co.groupby(['status','network']).size().to_csv(OUT/'component_status_counts.csv')
    sources=[Path(__file__),Path(H.__file__),Path(J.__file__),Path(S.__file__),J.XML,OUT/'responses_full_period.xml',H.BUAN/'taup/kim1983_buan.npz',CATALOG]   # IMSIL PORT
    prov=dict(response_mode=RESPONSE_MODE,recipe='Prior Buan KS/KG processing plus prior Haenam JU full-SEED-ID extension; full-period original response epochs',distance='SAC epicentral distance preserved to reproduce previous recipe; hypocentral timing uses joint depth',gates='SNR >=2; median H-preferred/V-fallback station ML; primary >=3 picked measured stations; 1/2 station results are sensitivity estimates',n_events=len(ev),n_ML3=int(ev.ml_min3.notna().sum()),n_ML1=int(ev.ml_min1.notna().sum()),waveforms='Read in place; per-event SHA256 of filename/size/mtime signatures; JU files additionally content-hashed',inputs=[dict(path=str(p),sha256=sha(p)) for p in sources])
    (OUT/'provenance.json').write_text(json.dumps(prov,indent=2));print(json.dumps({k:v for k,v in prov.items() if k!='inputs'},indent=2),flush=True)
    return ev

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args();run(a.out,a.workers,a.limit)
