"""Paired long-budget L1b gate and storage/runtime/negative-control audit."""
import argparse
import json
from pathlib import Path
import pandas as pd

CONTROLS=('native','tied','gain4','rank1_4')
TASKS=('copy','retrieve','ordered')
WORLDS=(618211,618212,618213)


def audit(fixed_path, matched_path, negative_path, out):
    f=pd.read_csv(fixed_path)
    m=pd.read_csv(matched_path)
    h=pd.read_csv(negative_path)
    assert len(f)==108 and len(m)==72 and len(h)==12, (len(f),len(m),len(h))
    for d in (f,m,h):
        assert sorted(d.world.unique().tolist())==list(WORLDS)
        assert sorted(d.seed.unique().tolist())==[0,1]
        assert not d.duplicated(['model','task','world','seed']).any()
    gate_worlds=[]
    for w in WORLDS:
        c=f[(f.world==w)&(f.model=='mirror4')]
        control=m[(m.world==w)&m.model.isin(CONTROLS)]
        means=control.groupby('model').nll.mean().sort_values()
        best=means.index[0]
        b=control[control.model==best]
        bytask=[]
        for t in TASKS:
            cn=c[c.task==t].nll.mean()
            bn=b[b.task==t].nll.mean()
            bytask.append({'task':t,'mirror_nll':float(cn),'control_nll':float(bn),'ratio':float(cn/bn)})
        ratio=float(c.nll.mean()/means.iloc[0])
        gate=bool(ratio <= .95 and max(x['ratio'] for x in bytask)<=1.10)
        gate_worlds.append({'world':w,'best_control':best,'mirror_nll_mean':float(c.nll.mean()),
                            'control_nll_mean':float(means.iloc[0]),'ratio':ratio,'by_task':bytask,
                            'world_gate':gate})
    model_task=(f.groupby(['model','task']).agg(nll=('nll','mean'),accuracy=('accuracy','mean'),
                train_s=('train_seconds','median'),infer_ms_batch8=('infer_ms_batch8','median'),
                bytes=('serialized_bytes','first'),params=('physical_params','first'),
                mac=('forward_mac_proxy_per_example','first')).reset_index())
    h_summary=(h.groupby('model').agg(nll=('nll','mean'),accuracy=('accuracy','mean')).reset_index())
    matched_steps=(m[['model','updates','forward_mac_proxy_per_example']].drop_duplicates()
                   .sort_values('model').to_dict('records'))
    result={'stage':'MA-1162 L1b long-budget fresh query NLL only',
            'fixed_rows':len(f),'matched_rows':len(m),'negative_rows':len(h),
            'world_gates':gate_worlds,'passed_worlds':sum(x['world_gate'] for x in gate_worlds),
            'primary_gate_pass':sum(x['world_gate'] for x in gate_worlds)>=2,
            'per_model_task':model_task.to_dict('records'),
            'negative_control':h_summary.to_dict('records'),
            'matched_steps':matched_steps,
            'fixed_updates':800,'eval_examples_per_run':384,
            'limitations':'synthetic single query, MAC proxy excludes softmax/rotations, 3 worlds insufficient confidence for generalization'}
    Path(out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('Gate worlds:')
    for w in gate_worlds:print(w['world'],w['best_control'],'ratio',round(w['ratio'],5),'PASS',w['world_gate'])
    print('Model x task accuracy:')
    print(model_task.pivot(index='model',columns='task',values='accuracy').round(3).to_string())
    print('Model x task NLL:')
    print(model_task.pivot(index='model',columns='task',values='nll').round(5).to_string())
    print('Hidden:',h_summary.to_dict('records'))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('fixed','matched','negative','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    audit(a.fixed,a.matched,a.negative,a.out)
