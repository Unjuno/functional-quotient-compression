"""Aggregate frozen MA-1162 L1 screen without dropping negative rows."""
import argparse
import json
from pathlib import Path
import pandas as pd

BASE = ('native', 'tied', 'gain4', 'rank1_4')
CAUSAL = ('copy','retrieve','ordered')


def summarize(fixed_path, match_path, output):
    fixed = pd.read_csv(fixed_path)
    matched = pd.read_csv(match_path)
    assert len(fixed)==192, f'full fresh grid expected 192 rows; got {len(fixed)}'
    assert len(matched)==4*4*3*2, f'four controls 3 worlds 2 seeds 4 tasks: {len(matched)}'
    assert not fixed.duplicated(['model','task','world','seed']).any()
    assert not matched.duplicated(['model','task','world','seed']).any()
    worlds=sorted(fixed.world.unique())
    assert worlds==[618201,618202,618203]
    assert set(matched.world.unique())==set(worlds)
    candidate=fixed[fixed.model=='mirror4']
    per_task=(fixed.groupby(['task','model'])[['nll','accuracy','train_seconds','infer_ms_batch8']]
              .mean().reset_index())
    world_rows=[]
    for world in worlds:
        c=candidate[(candidate.world==world)&candidate.task.isin(CAUSAL)]
        cmean=c.nll.mean()
        eligible=matched[(matched.world==world)&matched.task.isin(CAUSAL)&matched.model.isin(BASE)]
        control_nll=eligible.groupby('model').nll.mean().sort_values()
        win=control_nll.index[0]
        baseline=matched[(matched.world==world)&(matched.model==win)]
        ratios=[]
        for task in CAUSAL:
            ctask=c[c.task==task].nll.mean()
            btask=baseline[baseline.task==task].nll.mean()
            ratios.append({'task':task,'ratio':float(ctask/btask),'mirror_nll':float(ctask),'control_nll':float(btask)})
        world_rows.append({'world':int(world),'candidate_nll_mean':float(cmean),
                           'best_control':win,'control_nll_mean':float(control_nll.iloc[0]),
                           'ratio':float(cmean/control_nll.iloc[0]),'task_ratios':ratios,
                           'meets_world_gate':bool(cmean <= .95*control_nll.iloc[0] and all(a['ratio']<=1.10 for a in ratios))})
    match_steps=(matched[['model','updates','forward_mac_proxy_per_example']].drop_duplicates()
                 .sort_values('model').to_dict('records'))
    negative=fixed[fixed.task=='hidden'].groupby('model')[['nll','accuracy']].agg(['mean','std']).round(6)
    audit={'date':'2026-10-18','evidence':'frozen L1 query classification screen, NOT corpus LM',
           'rows_fixed':int(len(fixed)), 'rows_compute_matched':int(len(matched)),
           'worlds':world_rows,'world_gates_passed':sum(x['meets_world_gate'] for x in world_rows),
           'primary_pass':sum(x['meets_world_gate'] for x in world_rows)>=2,
           'matched_steps':match_steps,
           'per_task_models':per_task.to_dict('records'),
           'negative_control':negative.to_string(),
           'gpu':False,'matched_compute_caveat':'3x dense MAC proxy excludes actual rotations and nonlinear kernels; not equal measured time'}
    output=Path(output);output.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('World paired ratio Mirror / best control')
    for w in world_rows: print(w['world'],w['best_control'],round(w['ratio'],4), w['meets_world_gate'])
    print('world gates passed',audit['world_gates_passed'])
    print('per task mean')
    print(per_task.pivot(index='model',columns='task',values='accuracy').round(3).to_string())
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fixed',required=True)
    p.add_argument('--matched',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    summarize(a.fixed,a.matched,a.out)
