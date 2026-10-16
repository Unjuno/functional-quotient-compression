"""Evaluate world-paired heldout/native gates, never use these to retune fresh."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).parent

def summarize():
    report={}
    for stage,fn,step,seed_col in [('teacher_distill','FRESH_CORE.csv',1800,'init'),('direct_labels','DIRECT_FRESH.csv',2400,'seed')]:
        r=pd.read_csv(ROOT/'results'/fn)
        final=r[r['update' if stage=='teacher_distill' else 'step']==step]
        per=final.groupby(['world','method'],as_index=False).agg({k:'mean' for k in ['test_seen_BCE','test_unseen_BCE','test_seen_accuracy','test_unseen_accuracy','inference_bytes','parameters','wall_s','val_seen_BCE']})
        summaries={}
        for method,rows in per.groupby('method'):
            summaries[method]={k:float(rows[k].median()) for k in ['test_seen_BCE','test_unseen_BCE','test_unseen_accuracy','inference_bytes','parameters','wall_s']}
        w=sorted(per.world.unique()); comparisons=[]
        for world in w:
            subs=per[per.world==world].set_index('method')
            mir=subs.loc['mirror'];gn=subs.loc['gain'];tie=subs.loc['tied'];pair=subs.loc['pair_mirror']
            comparisons.append({'world':int(world),'mirror_gain_unseen_BCE_ratio':float(mir.test_unseen_BCE/gn.test_unseen_BCE),
                'mirror_pair_unseen_BCE_ratio':float(mir.test_unseen_BCE/pair.test_unseen_BCE),
                'mirror_tied_unseen_BCE_ratio':float(mir.test_unseen_BCE/tie.test_unseen_BCE),
                'mirror_gain_seen_BCE_ratio':float(mir.test_seen_BCE/gn.test_seen_BCE)})
        df=pd.DataFrame(comparisons)
        if stage=='teacher_distill':
            wins=((df.mirror_gain_unseen_BCE_ratio<=0.95)&(df.mirror_gain_seen_BCE_ratio<=1.01)).sum()
            gate={'n_world_pass':int(wins),'n_world_required':2,'pass':bool(wins>=2),
                'important_countertest_mirror_better_than_tied':int((df.mirror_tied_unseen_BCE_ratio<1).sum())}
        else:
            wins=((df.mirror_gain_unseen_BCE_ratio<1)&(df.mirror_gain_seen_BCE_ratio<=1.05)&(df.mirror_tied_unseen_BCE_ratio<1)).sum()
            gate={'n_world_pass':int(wins),'n_world_required':2,'pass':bool(wins>=2)}
        report[stage]={'final_models':len(final),'per_method_median_of_world_seed_means':summaries,'paired_per_world':comparisons,'preregistered_gate':gate}
        per.to_csv(ROOT/'results'/f'{stage.upper()}_WORLD_MEANS.csv',index=False)
    (ROOT/'results/ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':summarize()
