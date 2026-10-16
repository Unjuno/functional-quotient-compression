"""World-cluster descriptive statistics and preregistered grokking criteria."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t

STEPS=(400,800,1200,2000,4000,6000,8000)

def onset(history,key,threshold):
    seq=[r['step'] for r in history if r[key]>=threshold]
    return int(seq[0]) if seq else None

def grokking_events(history):
    train=onset(history,'train_accuracy',.99)
    tests={f'test_{int(v*100)}_step':onset(history,'test_accuracy',v) for v in (.5,.7,.9)}
    at=next((r for r in history if r['step']==train),None)
    test_at_train=(at or {}).get('test_accuracy')
    delayed=train is not None and test_at_train<=.30 and tests['test_50_step'] is not None and tests['test_50_step']-train>=1000
    strong=train is not None and test_at_train<=.30 and tests['test_90_step'] is not None and tests['test_90_step']-train>=1000
    # A large accuracy jump occurring within a rolling <=2000 updates after reaching train mastery
    fast_jump=False
    if train is not None:
        records=[r for r in history if r['step']>=train]
        for i,row in enumerate(records):
            if any(new['step']-row['step']<=2000 and new['test_accuracy']-row['test_accuracy']>=.30 for new in records[i+1:]):
                fast_jump=True;break
    return {'train99_step':train,**tests,'test_at_train99':test_at_train,'delayed_generalization':delayed,'strong_grokking':strong,'rapid_test_jump_0p3_per_2000':fast_jump}

def aggregate(root):
    rows=[];hists={}
    for f in sorted(root.glob('result_*.json')):
        r=json.loads(f.read_text());hist=json.loads((root/f"history_{r['world']}_{r['seed']}_{r['label']}.json").read_text())
        events=grokking_events(hist)
        hrs=np.array([v['step'] for v in hist],dtype=float);scores=np.array([v['test_accuracy'] for v in hist],dtype=float)
        r.update(events)
        r['test_accuracy_area']=float(np.trapezoid(scores,hrs)/(hrs[-1]-hrs[0]))
        hists[(r['world'],r['seed'],r['label'])]=hist
        rows.append(r)
    labels=sorted({x['label'] for x in rows});worlds=sorted({x['world'] for x in rows});seeds=sorted({x['seed'] for x in rows})
    stats=[];curves=[]
    for label in labels:
        rs=[r for r in rows if r['label']==label]
        mm=np.array([[[r['final_test_accuracy'] for r in rs if r['world']==w and r['seed']==s][0] for s in seeds] for w in worlds])
        tt=np.array([[[r['final_train_accuracy'] for r in rs if r['world']==w and r['seed']==s][0] for s in seeds] for w in worlds])
        ss=[r['test_50_step'] for r in rs if r['test_50_step'] is not None]
        stats.append({'method':label,'n_worlds':len(worlds),'n_seeds':len(seeds),
                      'median_world_mean_test_accuracy':float(np.median(mm.mean(1))),
                      'range_world_mean_test_accuracy':[float(mm.mean(1).min()),float(mm.mean(1).max())],
                      'median_world_mean_train_accuracy':float(np.median(tt.mean(1))),
                      'seed_test_acc_std_median_world':float(np.median(mm.std(1,ddof=1))),
                      'mean_8000_test_ce':float(np.mean([r['final_test_ce'] for r in rs])),
                      'median_inference_bytes':int(np.median([r['serialized_inference_bytes'] for r in rs])),
                      'parameters':rs[0]['parameters'],
                      'mean_wall_seconds_per_fit':float(np.mean([r['elapsed_seconds'] for r in rs])),
                      'delayed_generalization_count':sum(r['delayed_generalization'] for r in rs),
                      'strong_grokking_count':sum(r['strong_grokking'] for r in rs),
                      'test_50_reached_count':len(ss),'test_50_step_median_observed':float(np.median(ss)) if ss else None,
                      'area_under_test_accuracy':float(np.mean([r['test_accuracy_area'] for r in rs]))})
        for step in STEPS:
            a=np.array([[next(z for z in hists[(w,s,label)] if z['step']==step)['test_accuracy'] for s in seeds] for w in worlds])
            b=np.array([[next(z for z in hists[(w,s,label)] if z['step']==step)['train_accuracy'] for s in seeds] for w in worlds])
            curves.append({'method':label,'step':step,'median_world_mean_test_accuracy':float(np.median(a.mean(1))),
                           'median_world_mean_train_accuracy':float(np.median(b.mean(1)))})
    pairs=[]
    for label in labels:
        if label=='mirror':continue
        arr=np.array([np.mean([r['final_test_accuracy'] for r in rows if r['label']=='mirror' and r['world']==w])-
                      np.mean([r['final_test_accuracy'] for r in rows if r['label']==label and r['world']==w]) for w in worlds])
        ci=(float(arr.mean()-t.ppf(.975,len(arr)-1)*arr.std(ddof=1)/np.sqrt(len(arr))),float(arr.mean()+t.ppf(.975,len(arr)-1)*arr.std(ddof=1)/np.sqrt(len(arr))))
        pairs.append({'against':label,'mirror_minus_other_test_accuracy_world_deltas':arr.tolist(),
                      'paired_mean_difference':float(arr.mean()),'descriptive_95p_t_interval':ci,
                      'mirror_better_worlds':int((arr>0).sum())})
    out={'independent_unit':'random split of same p=31 rule universe','worlds':worlds,'seeds':seeds,'n_runs':len(rows),
         'summary':stats,'paired_effects':pairs,'strict_labeling':'test >=90% after train mastery for strong grokking; slow improvement only counts delayed generalization',
         'causal_limits':'fixed training format; wd control only for Mirror; fixed compute not equal running time, no clean mechanism identification'}
    (root/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
    with (root/'LEARNING_CURVES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=curves[0].keys());writer.writeheader();writer.writerows(curves)
    with (root/'GROKKING_PER_RUN.csv').open('w',newline='') as f:
        keys=['world','seed','label','train99_step','test_50_step','test_70_step','test_90_step','test_at_train99','delayed_generalization','strong_grokking','rapid_test_jump_0p3_per_2000','final_test_accuracy','test_accuracy_area','serialized_inference_bytes']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:v for k,v in r.items() if k in keys} for r in rows)
    print('ANALYSIS:')
    for a in stats:print(a['method'], 'acc=',round(a['median_world_mean_test_accuracy'],4),'strong=',a['strong_grokking_count'],'delayed=',a['delayed_generalization_count'])
    return out
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--path',required=True)
    aggregate(Path(parser.parse_args().path))
