from __future__ import annotations
import json,math
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).parent

def run():
    a=pd.read_csv(ROOT/'results/FULL_FRESH.csv');a=a[a.step==2400]
    world=a.groupby(['world','method'],as_index=False).agg({'test_all_BCE':'mean','test_all_accuracy':'mean','test_worst_pair_BCE':'mean','inference_bytes':'mean','parameters':'mean','wall_s':'mean'})
    world.to_csv(ROOT/'results/FULL_WORLD_MEANS.csv',index=False)
    med=world.groupby('method').median(numeric_only=True).drop(columns='world')
    pw=[]
    for w in sorted(world.world.unique()):
        z=world[world.world==w].set_index('method')
        r={"world":int(w),"mirror_vs_gain_BCE_ratio":float(z.loc['mirror','test_all_BCE']/z.loc['gain','test_all_BCE']),"mirror_vs_independent_BCE_ratio":float(z.loc['mirror','test_all_BCE']/z.loc['independent','test_all_BCE']),"mirror_vs_independent_byte_ratio":float(z.loc['mirror','inference_bytes']/z.loc['independent','inference_bytes'])};pw.append(r)
    wins=sum(x['mirror_vs_independent_BCE_ratio']<=1.10 and x['mirror_vs_independent_byte_ratio']<=.5 and x['mirror_vs_gain_BCE_ratio']<=.90 for x in pw)
    rec={'world_means':pw,'median_of_world_means':{n:row.to_dict() for n,row in med.iterrows()},'gate':{'passed_worlds':wins,'needed':2,'pass':wins>=2}}
    (ROOT/'results/FULL_ANALYSIS.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec,indent=2))
if __name__=='__main__':run()
