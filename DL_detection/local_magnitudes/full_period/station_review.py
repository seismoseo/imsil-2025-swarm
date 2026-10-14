"""Reaggregate existing ML after excluding independently identified anomalous stations.

IMSIL PORT: ROOT changed; EXCLUDED is EMPTY by default (decision D16 — Haenam excluded JDO2/WAN2 after its own station
review; Imsil stations need their own review); the Haenam-specific assert on event 20200425131318 is removed.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path('/home/msseo/works/72.2025_Imsil_swarm/DL_detection')   # IMSIL PORT
sys.path.insert(0,str(ROOT/'local_magnitudes/full_period'))
from validate_full_magnitudes import aggregate
EXCLUDED=[]   # IMSIL PORT (D16); Haenam: ['JDO2','WAN2']

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);base=ROOT/'local_magnitudes/full_period'
    old=pd.read_csv(base/'full_period_ML.csv',dtype={'eid':str,'event_id':str,'template_match':str})
    st=pd.read_csv(base/'station_magnitudes.csv',dtype={'eid':str});kept=st[~st.station.isin(EXCLUDED)].copy()
    groups={eid:g for eid,g in kept.groupby('eid')};rows=[dict(eid=eid,**aggregate(groups.get(eid,kept.iloc[:0]))) for eid in old.eid]
    agg=pd.DataFrame(rows);new=old.drop(columns=[col for col in agg if col!='eid']).merge(agg,on='eid',validate='one_to_one')
    for n in [1,2,3]:assert new['ml_min'+str(n)].notna().equals(new.n_pick.ge(n))
    # IMSIL PORT: Haenam-specific assert on event 20200425131318 removed
    diff=old[['eid','ml_min1','ml_min2','ml_min3','n_pick']].merge(new[['eid','ml_min1','ml_min2','ml_min3','n_pick']],on='eid',suffixes=('_legacy','_reviewed'))
    for name,df in [('full_period_ML',new),('station_magnitudes',kept),('removed_station_measurements',st[st.station.isin(EXCLUDED)]),('magnitude_changes',diff)]:df.to_csv(out/(name+'.csv'),index=False)
    sources=[Path(__file__),base/'validate_full_magnitudes.py',base/'full_period_ML.csv',base/'station_magnitudes.csv']
    prov=dict(decision='Exclude JDO2 and WAN2 from absolute ML aggregation throughout the full period. No empirical gain correction is imposed.',reason='JDO2 has large era-dependent station residuals; WAN2 has persistently high residuals. The first JDO2-only onset estimate is contaminated by low-frequency background oscillations. Comprehensive correction trial failed to improve event-held-out KMA consistency.',recipe='Original Buan amplitudes, responses, distance formula, peak SNR>=2, H preference/V fallback and median aggregation unchanged; >=3 picked stations primary, >=1/2 sensitivity',counts=dict(events=len(new),ML3=int(new.ml_min3.notna().sum()),ML1=int(new.ml_min1.notna().sum())),limitations='Remaining 1/2-station values are sensitivity estimates, not vetted event magnitudes. No claim that station exclusion improves absolute accuracy for every event or establishes completeness.',inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
    (out/'provenance.json').write_text(json.dumps(prov,indent=2));print(json.dumps(prov['counts']))
    return new
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
