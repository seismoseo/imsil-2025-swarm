"""Quarantine undefined response units without changing the Buan amplitude recipe.

IMSIL PORT: ROOT and catalog path only.

Run after full_magnitudes.py; regenerates from its immutable measurement caches.
No waveform processing, location selection or calibration to KMA occurs here.
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
ROOT=Path('/home/msseo/works/72.2025_Imsil_swarm/DL_detection')   # IMSIL PORT
sys.path.insert(0,str(ROOT/'local_magnitudes'))
import haenam_magnitudes as H

def aggregate(st):
    a=H.S.event_ml(st)
    if len(st):
        values=st.ml_h.where(st.ml_h.notna(),st.ml_v); med=float(values.median())
        a.update(station_scatter=float(values.std()),ml_h_strict=float(st.ml_h.median()) if (st.ml_h.notna()&st.p_source.eq('pick')).sum()>=3 else np.nan,
                 n_JN=int(st.network.eq('JU').sum()),n_JN_picked=int((st.network.eq('JU')&st.p_source.eq('pick')).sum()),
                 ml_KS_KG=H.S.event_ml(st[st.network.ne('JU')])['ml'])
    else:
        med=np.nan;a.update(station_scatter=np.nan,ml_h_strict=np.nan,n_JN=0,n_JN_picked=0,ml_KS_KG=np.nan)
    for n in [1,2,3]:a['ml_min'+str(n)]=med if a['n_pick']>=n else np.nan
    return a

def run(out):
    out=Path(out);raw=out/'raw_recipe';raw.mkdir(exist_ok=True)
    cat=pd.read_csv(ROOT/'catalogs/imsil_dl_relocated.csv',dtype={'event_id':str,'template_match':str});cat['eid']=cat.event_id   # IMSIL PORT
    events=[];stations=[];components=[]
    for eid in cat.eid:
        r=json.loads((out/'combined_cache'/f'{eid}.json').read_text())
        events.append(dict(eid=eid,**r['event'],ml_KS_KG=r['ml_KS_KG'],waveform_signature=r['waveform_signature'],waveform_files=r['waveform_files']))
        stations.extend(r['stations']);components.extend(r['components'])
    co=pd.DataFrame(components);st=pd.DataFrame(stations);ev=cat.merge(pd.DataFrame(events),on='eid',validate='one_to_one')
    for name,df in [('component_measurements',co),('station_magnitudes',st),('full_period_ML',ev)]:df.to_csv(raw/(name+'.csv'),index=False)
    invalid=co.response_warnings.fillna('').str.contains("not known to ObsPy",regex=False)
    co['response_units_validated']=co.status.eq('processed') & ~invalid
    co.loc[invalid,'status']='response_units_unresolved'
    badkeys=set(map(tuple,co.loc[invalid,['eid','network','station']].drop_duplicates().to_numpy()))
    keep=[]
    for r in st.to_dict('records'):
        key=(r['eid'],r['network'],r['station'])
        if key not in badkeys:keep.append(r);continue
        g=co[co.eid.eq(r['eid'])&co.network.eq(r['network'])&co.station.eq(r['station'])]
        replacement=H.assemble_station(r['network'],r['station'],r['dist_km'],r['p_source'],g.to_dict('records'),'corrected')
        if replacement:keep.append(dict(eid=r['eid'],**replacement))
    valid_st=pd.DataFrame(keep);groups={eid:g for eid,g in valid_st.groupby('eid')};rows=[]
    for eid in ev.eid:rows.append(dict(eid=eid,**aggregate(groups.get(eid,valid_st.iloc[:0]))))
    agg=pd.DataFrame(rows);result=ev.drop(columns=[c for c in agg if c!='eid']).merge(agg,on='eid',validate='one_to_one')
    audit=ev[['eid','ml_min3','ml_min1','n_pick']].merge(result[['eid','ml_min3','ml_min1','n_pick']],on='eid',suffixes=('_raw','_validated'))
    audit['affected_response_station']=audit.eid.isin([key[0] for key in badkeys])
    for n in [1,2,3]:assert result['ml_min'+str(n)].notna().equals(result.n_pick.ge(n))
    for name,df in [('component_measurements',co),('station_magnitudes',valid_st),('full_period_ML',result),('response_validation_event_comparison',audit)]:df.to_csv(out/(name+'.csv'),index=False)
    co.groupby(['status','network']).size().to_csv(out/'component_status_counts.csv')
    co[invalid][['eid','trace_id','response_warnings','response_input_units','snr_corrected']].to_csv(out/'response_units_quarantine.csv',index=False)
    provpath=out/'provenance.json';p=json.loads(provpath.read_text())
    if 'postprocessing' not in p:(out/'measurement_provenance.json').write_text(json.dumps(p,indent=2))
    p['postprocessing']='Undefined response-unit components quarantined; same amplitude recipe, SNR and station gates. Raw recipe outputs preserved.'
    p['n_ML3']=int(result.ml_min3.notna().sum());p['n_ML1']=int(result.ml_min1.notna().sum())
    p['response_units_quarantine']=dict(components=int(invalid.sum()),removed_station_measurements=len(st)-len(valid_st),raw_ML3=int(ev.ml_min3.notna().sum()),raw_ML1=int(ev.ml_min1.notna().sum()))
    script=Path(__file__);p['response_validation_script']=dict(path=str(script),sha256=hashlib.sha256(script.read_bytes()).hexdigest());provpath.write_text(json.dumps(p,indent=2))
    print(json.dumps({k:v for k,v in p.items() if k not in ['inputs']},indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
